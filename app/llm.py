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


EXTRACTION_PROMPT_TEMPLATE = """You are an RFQ intake clerk for an industrial fastener distributor (cap \
screws, hex bolts, machine screws, nuts, washers, anchors). Read the RFQ text \
below and extract every distinct line item (product request) into structured JSON.

<rfq_text>
{rfq_text}
</rfq_text>

For each line item, extract:
- "raw_text": the exact snippet of text this line item came from
- "quantity": numeric quantity requested (number, or null if not stated)
- "unit": unit of measure as written (e.g. "each", "pcs", "box"), or null
- "description": a short plain-text description of the product requested
- "attributes": a dict of fastener specs. Keys:
  - "type": one of "cap_screw", "hex_bolt", "machine_screw", "nut", "washer", "anchor"
  - "diameter": the fastener diameter, e.g. "1/4\\"", "#10" (numbered machine \
screw size), or "M8" (metric)
  - "thread_pitch_or_tpi": threads-per-inch for imperial (e.g. "20"), or thread \
pitch in mm for metric (e.g. "1.25")
  - "length": fastener length, e.g. "1\\"" or "30mm"
  - "grade_or_class": e.g. "Grade 5", "Grade 8", "Class 8.8", "Class 10.9", \
"Class 12.9", "A2-70", "A4-70", "18-8"
  - "material": "Steel", "Alloy Steel", "Stainless Steel", or "Brass"
  - "finish_coating": "Zinc", "Zinc Yellow", "Black Oxide", "Hot-Dip Galvanized", \
or "Plain"
  - "head_type": "Socket Cap", "Hex", "Flat", "Button", "Pan", "Round", "Oval", \
or "Truss"
  - "drive_type": "Hex/Allen", "Phillips", "Slotted", "Torx", or "Square"
  - "standard": e.g. "DIN 912", "ISO 4762", "ASME B18.3", "ANSI B18.6.3", "IFI 100/107"
  - "thread_direction": "RH" (default, only include if left-hand) or "LH"
  - "units": "imperial" or "metric" — infer from the diameter notation \
(fractional/numbered = imperial, "M" prefix = metric)

  Only include keys that are actually mentioned or strongly implied by the \
text. Leave out keys you can't determine — do NOT guess a diameter, thread \
pitch, or length that isn't stated or clearly implied.

  Expand common fastener shorthand using this glossary:
  - SHCS = socket head cap screw -> head_type="Socket Cap", drive_type="Hex/Allen", type="cap_screw"
  - HHCS = hex head cap screw -> head_type="Hex", drive_type="Hex (Wrench)", type="cap_screw"
  - FHCS / FHSCS = flat head socket cap screw -> head_type="Flat", drive_type="Hex/Allen", type="cap_screw"
  - BHCS = button head socket cap screw -> head_type="Button", drive_type="Hex/Allen", type="cap_screw"
  - HCS = hex cap screw -> head_type="Hex", drive_type="Hex (Wrench)", type="cap_screw"
  - PHMS = pan head machine screw, Phillips -> head_type="Pan", drive_type="Phillips", type="machine_screw"
  - FHMS = flat head machine screw -> head_type="Flat", type="machine_screw"
  - SS / A2 / A4 / 18-8 = stainless steel grades -> material="Stainless Steel", \
grade_or_class="A2-70"/"A4-70"/"18-8" as stated
  - Z = zinc plated -> finish_coating="Zinc"
  - HDG = hot-dip galvanized -> finish_coating="Hot-Dip Galvanized"
  - GR5 / GR8 = SAE Grade 5 / Grade 8 -> grade_or_class="Grade 5"/"Grade 8"
  - A callout like "1/4-20 x 1 SHCS Z" means diameter="1/4\\"", \
thread_pitch_or_tpi="20", length="1\\"", head_type="Socket Cap", \
drive_type="Hex/Allen", finish_coating="Zinc", units="imperial"
  - A callout like "M8-1.25 x 30 A2 HHCS" means diameter="M8", \
thread_pitch_or_tpi="1.25", length="30mm", grade_or_class="A2-70", \
material="Stainless Steel", head_type="Hex", drive_type="Hex (Wrench)", units="metric"
- "target_date": any requested delivery/lead-time date mentioned, or null
- "notes": any other relevant free-text notes for this line, or null

Return ONLY valid JSON, no markdown fences, in this exact shape:
{{"line_items": [{{"raw_text": "...", "quantity": ..., "unit": ..., "description": "...", \
"attributes": {{...}}, "target_date": ..., "notes": ...}}]}}

If the text contains no identifiable product requests, return {{"line_items": []}}."""


RERANK_PROMPT_TEMPLATE = """You are an expert industrial fastener distributor with 30 years of experience. \
A customer RFQ line item needs to be matched to one SKU in our catalog.

All candidates below have already been hard-filtered to match the line item's \
diameter, thread pitch/TPI, length, and unit system (imperial vs metric) exactly \
— those attributes are NOT in question. Your job is to rank candidates on the \
remaining attributes: grade/class, material, finish/coating, head type, drive \
type, and standard.

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
   - >= 0.85: every attribute the customer specified (grade/class, material, \
finish, head type, drive type, standard) matches this candidate exactly, with \
no other candidate equally plausible.
   - 0.60-0.84: the structural specs (diameter/thread/length) match, but a \
secondary attribute (grade, finish, head/drive type, or standard) was not \
specified by the customer and more than one candidate could satisfy the request.
   - < 0.60: even among structurally-matching candidates, none is a good fit \
for the secondary attributes the customer specified, or the request is too \
vague to disambiguate.
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
