"""Hybrid matcher: BM25 + embedding retrieval, fused with RRF, then LLM-reranked.

For a small single-distributor catalog (low hundreds of SKUs), both indices are
built in memory at startup — rebuilding takes well under a second.
"""

from __future__ import annotations

import logging
import re

import faiss
import numpy as np
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi

from app.config import (
    BM25_TOP_K,
    EMBEDDING_MODEL,
    EMBEDDING_TOP_K,
    HIGH_CONFIDENCE_THRESHOLD,
    MEDIUM_CONFIDENCE_THRESHOLD,
    RERANK_CANDIDATES,
    RRF_K,
)
from app.db.models import CatalogItem
from app.db.session import get_session
from app.llm import LLMClient
from app.normalize.units import build_query_text

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9/.\-]+", text.lower())


def confidence_bucket(confidence: float) -> str:
    """HIGH: auto-fill. MEDIUM: fill + flag. LOW: abstain / escalate."""
    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
        return "HIGH"
    if confidence >= MEDIUM_CONFIDENCE_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def _infer_units(diameter: str) -> str | None:
    """Infer 'metric'/'imperial' from a canonical diameter string when not extracted directly."""
    if re.match(r"^[mM]\d", diameter):
        return "metric"
    if diameter.startswith("#") or '"' in diameter:
        return "imperial"
    return None


class HybridMatcher:
    """BM25 + embedding indices over the catalog, fused via RRF, LLM-reranked."""

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient()
        self._catalog: list[dict] = []
        self._bm25: BM25Okapi | None = None
        self._faiss_index: faiss.Index | None = None
        self._embedder: TextEmbedding | None = None
        self._build_indices()

    def _load_catalog(self) -> list[dict]:
        with get_session() as session:
            items = session.query(CatalogItem).all()
            return [
                {
                    "sku": c.sku,
                    "type": c.type,
                    "description": c.description,
                    "diameter": c.diameter,
                    "thread_pitch_or_tpi": c.thread_pitch_or_tpi,
                    "length": c.length,
                    "grade_or_class": c.grade_or_class,
                    "material": c.material,
                    "finish_coating": c.finish_coating,
                    "head_type": c.head_type,
                    "drive_type": c.drive_type,
                    "standard": c.standard,
                    "thread_direction": c.thread_direction,
                    "units": c.units,
                    "price_usd": c.price_usd,
                }
                for c in items
            ]

    def _build_indices(self) -> None:
        self._catalog = self._load_catalog()
        if not self._catalog:
            raise RuntimeError("Catalog is empty — run `python -m app.ingest.catalog` first.")

        descriptions = [item["description"] for item in self._catalog]

        tokenized = [_tokenize(d) for d in descriptions]
        self._bm25 = BM25Okapi(tokenized)

        self._embedder = TextEmbedding(model_name=EMBEDDING_MODEL)
        embeddings = np.array(list(self._embedder.embed(descriptions)), dtype="float32")
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        self._faiss_index = index

    def _bm25_candidates(self, query: str, allowed: set[int]) -> list[int]:
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(allowed, key=lambda i: scores[i], reverse=True)
        return [i for i in ranked if scores[i] > 0][:BM25_TOP_K]

    def _embedding_candidates(self, query: str, allowed: set[int]) -> list[int]:
        query_emb = np.array(list(self._embedder.embed([query])), dtype="float32")
        faiss.normalize_L2(query_emb)
        _, indices = self._faiss_index.search(query_emb, len(self._catalog))
        return [int(i) for i in indices[0] if i >= 0 and int(i) in allowed][:EMBEDDING_TOP_K]

    def _hard_filter(self, attributes: dict) -> list[int] | None:
        """Restrict to catalog SKUs matching diameter, thread spec, length, and unit
        system exactly. Returns None if the line item itself lacks enough structural
        information (diameter and/or thread pitch/TPI) to identify a fastener at all.
        """
        diameter = attributes.get("diameter")
        thread = attributes.get("thread_pitch_or_tpi")
        if not diameter or not thread:
            return None

        units = attributes.get("units") or _infer_units(diameter)
        length = attributes.get("length")
        thread_direction = attributes.get("thread_direction") or "RH"

        matches = []
        for i, item in enumerate(self._catalog):
            if item["diameter"] != diameter:
                continue
            if item["thread_pitch_or_tpi"] != thread:
                continue
            if units and item["units"] != units:
                continue
            if item["thread_direction"] != thread_direction:
                continue
            if length and item["length"] != length:
                continue
            matches.append(i)
        return matches

    @staticmethod
    def _rrf_fuse(*ranked_lists: list[int]) -> list[int]:
        """Reciprocal rank fusion: score(d) = sum over lists of 1 / (RRF_K + rank)."""
        scores: dict[int, float] = {}
        for ranked in ranked_lists:
            for rank, idx in enumerate(ranked, start=1):
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (RRF_K + rank)
        return sorted(scores, key=lambda i: scores[i], reverse=True)

    def match(self, line_item: dict) -> dict:
        """Match a normalized line item to catalog SKUs.

        Returns {"sku", "confidence", "price_usd", "alternatives", "reasoning"}
        per the match schema (sku=None / confidence=0.0 means abstain).

        Matching is a hard filter followed by a soft rank: candidates must match
        diameter, thread pitch/TPI, length (if given), thread direction, and unit
        system (imperial/metric) exactly before BM25/embedding/LLM ever see them.
        This guarantees a 1/4-20 request never gets ranked against 5/16-18 SKUs
        just because their descriptions are textually similar.
        """
        attributes = line_item.get("attributes") or {}
        allowed_indices = self._hard_filter(attributes)

        if allowed_indices is None:
            return {
                "sku": None,
                "confidence": 0.0,
                "price_usd": None,
                "alternatives": [],
                "reasoning": "Could not determine diameter and/or thread pitch/TPI from "
                "the request, so no SKU can be confidently identified.",
            }

        if not allowed_indices:
            spec = f"diameter={attributes.get('diameter')!r}, thread={attributes.get('thread_pitch_or_tpi')!r}"
            if attributes.get("length"):
                spec += f", length={attributes['length']!r}"
            return {
                "sku": None,
                "confidence": 0.0,
                "price_usd": None,
                "alternatives": [],
                "reasoning": f"No catalog SKU matches the requested spec ({spec}).",
            }

        query = build_query_text(line_item)
        allowed = set(allowed_indices)

        bm25_ranked = self._bm25_candidates(query, allowed)
        emb_ranked = self._embedding_candidates(query, allowed)
        fused = self._rrf_fuse(bm25_ranked, emb_ranked)

        top_indices = fused[:RERANK_CANDIDATES]
        candidates = [self._catalog[i] for i in top_indices]

        ranked = self.llm.rerank_candidates(line_item, candidates)

        if not ranked:
            return {
                "sku": None,
                "confidence": 0.0,
                "price_usd": None,
                "alternatives": [],
                "reasoning": "LLM reranker found no plausible match among retrieved candidates.",
            }

        catalog_by_sku = {c["sku"]: c for c in candidates}
        top = ranked[0]
        alternatives = [
            {
                "sku": r["sku"],
                "confidence": r["confidence"],
                "description": catalog_by_sku.get(r["sku"], {}).get("description", ""),
                "reasoning": r.get("reasoning", ""),
            }
            for r in ranked[1:]
            if r["sku"] in catalog_by_sku
        ]

        return {
            "sku": top["sku"],
            "confidence": top["confidence"],
            "price_usd": catalog_by_sku.get(top["sku"], {}).get("price_usd"),
            "alternatives": alternatives,
            "reasoning": top.get("reasoning", ""),
        }
