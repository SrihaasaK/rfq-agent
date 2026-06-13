"""Single interface for all LLM calls in the pipeline (extraction + reranking).

Isolating calls here means swapping providers or adding caching only requires
changes in this module.
"""

from __future__ import annotations

import json
import logging

from dotenv import load_dotenv
from groq import Groq

from app.config import EXTRACTION_MODEL, RERANK_MODEL

load_dotenv()
logger = logging.getLogger(__name__)


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return raw


EXTRACTION_PROMPT_TEMPLATE = """You are an RFQ intake clerk for an industrial brass/bronze plumbing fittings \
distributor. Read the RFQ text below and extract every distinct line item \
(product request) into structured JSON.

<rfq_text>
{rfq_text}
</rfq_text>

For each line item, extract:
- "raw_text": the exact snippet of text this line item came from
- "quantity": numeric quantity requested (number, or null if not stated)
- "unit": unit of measure as written (e.g. "each", "pcs", "box"), or null
- "description": a short plain-text description of the product requested
- "attributes": a dict of any specs mentioned. Common keys for this catalog: \
"type" (elbow, elbow_45, tee, coupling, nipple, union, reducer, plug, cap, cross, bushing), \
"size" (e.g. "1/2\\"", "3/4\\" x 1/2\\""), "thread_standard" (NPT, BSP, compression, sweat, flare), \
"material_grade" (C36000, C46400, C84400). Only include keys that are actually \
mentioned or strongly implied by the text. Leave out keys you can't determine.
- "target_date": any requested delivery/lead-time date mentioned, or null
- "notes": any other relevant free-text notes for this line, or null

Return ONLY valid JSON, no markdown fences, in this exact shape:
{{"line_items": [{{"raw_text": "...", "quantity": ..., "unit": ..., "description": "...", \
"attributes": {{...}}, "target_date": ..., "notes": ...}}]}}

If the text contains no identifiable product requests, return {{"line_items": []}}."""


RERANK_PROMPT_TEMPLATE = """You are an expert brass/bronze plumbing fittings distributor with 30 years of \
experience. A customer RFQ line item needs to be matched to one SKU in our catalog.

<line_item>
{line_item}
</line_item>

<candidates>
{candidates}
</candidates>

Instructions:
1. Rank the candidates by how well they match the line item (best first). Only \
include candidates that are plausible matches — you do not need to rank all of them.
2. Assign each ranked candidate a confidence score (0.0-1.0) reflecting genuine \
certainty that THIS SKU is what the customer wants:
   - >= 0.85: every attribute mentioned in the line item (type, size, and, if \
stated, thread standard / material grade) matches this candidate exactly, with \
no other candidate equally plausible.
   - 0.60-0.84: type and size match, but a secondary attribute (thread standard \
or material grade) was not specified by the customer and more than one candidate \
could satisfy the request.
   - < 0.60: the product type or size itself is unclear, no candidate is a good \
match, or the request doesn't map cleanly to this catalog.
3. For each ranked candidate, give a one-sentence "reasoning" explaining the \
confidence score, calling out which attributes matched and which were assumed.

Return ONLY valid JSON, no markdown fences, in this exact shape:
{{"ranked": [{{"sku": "...", "confidence": 0.0, "reasoning": "..."}}]}}

If NO candidate is a plausible match, return {{"ranked": []}}."""


class LLMClient:
    """Wraps the Groq client. Every LLM call in the pipeline goes through here."""

    def __init__(self, client: Groq | None = None):
        self.client = client or Groq()

    def extract_line_items(self, rfq_text: str) -> list[dict]:
        """Extract structured line items from raw RFQ text."""
        prompt = EXTRACTION_PROMPT_TEMPLATE.format(rfq_text=rfq_text)

        response = self.client.chat.completions.create(
            model=EXTRACTION_MODEL,
            messages=[
                {"role": "system", "content": "You are a structured data extraction assistant. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2048,
            temperature=0.0,
        )

        raw = _strip_fences(response.choices[0].message.content)
        data = json.loads(raw)
        return data.get("line_items", [])

    def rerank_candidates(self, line_item: dict, candidates: list[dict]) -> list[dict]:
        """Rank candidate catalog SKUs for a line item, with confidence + reasoning.

        Returns a list of {"sku", "confidence", "reasoning"} dicts, best first.
        """
        prompt = RERANK_PROMPT_TEMPLATE.format(
            line_item=json.dumps(line_item, indent=2),
            candidates=json.dumps(candidates, indent=2),
        )

        response = self.client.chat.completions.create(
            model=RERANK_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert industrial distributor. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,
            temperature=0.1,
        )

        raw = _strip_fences(response.choices[0].message.content)
        data = json.loads(raw)
        return data.get("ranked", [])
