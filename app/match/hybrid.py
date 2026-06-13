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
                    "size": c.size,
                    "thread_standard": c.thread_standard,
                    "material_grade": c.material_grade,
                    "pressure_rating_psi": c.pressure_rating_psi,
                    "price_usd": c.price_usd,
                    "common_end_markets": c.common_end_markets,
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

    def _bm25_candidates(self, query: str) -> list[int]:
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = np.argsort(scores)[::-1]
        return [int(i) for i in ranked[:BM25_TOP_K] if scores[i] > 0]

    def _embedding_candidates(self, query: str) -> list[int]:
        query_emb = np.array(list(self._embedder.embed([query])), dtype="float32")
        faiss.normalize_L2(query_emb)
        _, indices = self._faiss_index.search(query_emb, EMBEDDING_TOP_K)
        return [int(i) for i in indices[0] if i >= 0]

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
        """
        query = build_query_text(line_item)

        bm25_ranked = self._bm25_candidates(query)
        emb_ranked = self._embedding_candidates(query)
        fused = self._rrf_fuse(bm25_ranked, emb_ranked)

        if not fused:
            return {
                "sku": None,
                "confidence": 0.0,
                "price_usd": None,
                "alternatives": [],
                "reasoning": "No candidate SKUs found by BM25 or embedding search.",
            }

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
