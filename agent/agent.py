"""Core RFQ processing agent — two-LLM-call pipeline using Groq."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from groq import Groq

from agent.models import (
    ClarifyingQuestion,
    ExtractedIntent,
    HumanEscalation,
    Hypothesis,
    QuoteDraft,
    RFQResult,
    Route,
)

load_dotenv()
logger = logging.getLogger(__name__)

CATALOG_PATH = Path(__file__).parent.parent / "catalog" / "catalog.json"
CUSTOMERS_PATH = Path(__file__).parent.parent / "customers" / "customer_history.json"

# Groq models: fast 8B for extraction, 70B for hypothesis reasoning
EXTRACTION_MODEL = "llama-3.1-8b-instant"
HYPOTHESIS_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

CONFIDENCE_QUOTE_THRESHOLD = 0.80
CONFIDENCE_HUMAN_THRESHOLD = 0.65


def _load_catalog() -> list[dict[str, Any]]:
    with open(CATALOG_PATH) as f:
        return json.load(f)


def _load_customers() -> list[dict[str, Any]]:
    with open(CUSTOMERS_PATH) as f:
        return json.load(f)


def _find_customer(name: str | None, customers: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not name:
        return None
    name_lower = name.lower()
    for c in customers:
        if name_lower in c["name"].lower() or name_lower in c.get("company", "").lower():
            return c
    return None


def _extract_intent(client: Groq, email_text: str) -> ExtractedIntent:
    """Call 1: Extract structured intent from raw email using Llama 8B (fast, free)."""

    prompt = f"""Analyze this RFQ (Request for Quote) email for brass fittings. Extract structured information.

<email>
{email_text}
</email>

Return a JSON object with exactly these fields:
- "product_type": the type of fitting requested (elbow/tee/coupling/nipple/union/reducer) or null if unclear
- "specs_mentioned": dict of specs explicitly stated. Keys can include: "size", "thread_standard" (NPT/BSP/compression/sweat), "material_grade" (C36000/C46400/C84400), "pressure_rating_psi"
- "specs_missing": list of critical specs NOT mentioned. Critical specs are: size, thread_standard, material_grade
- "quantity": integer quantity requested, or null
- "urgency": "standard", "rush", or "ASAP" based on tone
- "customer_signals": list of strings — any end-market signals, project names, industry references
- "referenced_past_orders": boolean — does the email reference previous orders or "same as last time"?

Return ONLY valid JSON, no markdown fences."""

    response = client.chat.completions.create(
        model=EXTRACTION_MODEL,
        messages=[
            {"role": "system", "content": "You are a structured data extraction assistant. Return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1024,
        temperature=0.0,
    )

    raw = response.choices[0].message.content.strip()
    # Handle potential markdown fences
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    data = json.loads(raw)
    return ExtractedIntent(**data)


def _generate_hypotheses(
    client: Groq,
    intent: ExtractedIntent,
    customer: dict[str, Any] | None,
    catalog: list[dict[str, Any]],
) -> tuple[list[Hypothesis], str]:
    """Call 2: Generate ranked SKU hypotheses using Llama 70B (quality reasoning)."""

    catalog_json = json.dumps(catalog, indent=2)
    intent_json = intent.model_dump_json(indent=2)
    customer_json = json.dumps(customer, indent=2) if customer else "No customer history available."

    prompt = f"""You are an expert brass fittings distributor with 30 years of experience. Given a parsed RFQ intent, customer history, and product catalog, generate ranked SKU hypotheses.

<parsed_intent>
{intent_json}
</parsed_intent>

<customer_history>
{customer_json}
</customer_history>

<catalog>
{catalog_json}
</catalog>

Instructions:
1. Generate 2-3 ranked hypotheses for which SKU(s) the customer most likely needs.
2. Use customer order history and end-market signals to disambiguate when specs are missing.
3. Assign confidence scores (0.0-1.0) that reflect genuine certainty. Be calibrated:
   - >0.85 only when specs + history converge on one SKU unambiguously
   - 0.5-0.85 when you have a strong guess but a critical spec is missing
   - <0.5 when the request is too vague or doesn't match your catalog
4. If the request is fundamentally unclear (no product type, no size, references missing attachments, requires custom/non-standard products, asks for a full catalog, or requires products outside the catalog), you MUST set all confidences below 0.5. Do not guess when the customer hasn't specified enough to narrow to a category.

Return a JSON object with:
- "hypotheses": array of objects with "rank", "sku", "description", "confidence", "reasoning"
- "route_reasoning": one sentence explaining the routing decision

If the email is too vague to generate meaningful hypotheses, return an empty hypotheses array with route_reasoning explaining why.

Return ONLY valid JSON, no markdown fences."""

    response = client.chat.completions.create(
        model=HYPOTHESIS_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert industrial distributor. Return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=2048,
        temperature=0.1,
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    data = json.loads(raw)
    hypotheses = [Hypothesis(**h) for h in data.get("hypotheses", [])]
    route_reasoning = data.get("route_reasoning", "")
    return hypotheses, route_reasoning


def _route_result(
    email_text: str,
    customer_name: str | None,
    intent: ExtractedIntent,
    hypotheses: list[Hypothesis],
    route_reasoning: str,
    catalog: list[dict[str, Any]],
) -> RFQResult:
    """Route based on top hypothesis confidence."""

    if not hypotheses:
        return RFQResult(
            email_text=email_text,
            customer_name=customer_name,
            extracted_intent=intent,
            hypotheses=[],
            route=Route.HUMAN,
            route_reasoning=route_reasoning or "No viable hypotheses could be generated.",
            escalation=HumanEscalation(
                reasoning="The request is too vague or outside catalog scope to generate hypotheses.",
                issues=intent.specs_missing or ["Insufficient information to identify any product"],
            ),
        )

    top = hypotheses[0]

    if top.confidence >= CONFIDENCE_QUOTE_THRESHOLD:
        # Find the catalog entry for pricing
        cat_entry = next((c for c in catalog if c["sku"] == top.sku), None)
        unit_price = cat_entry["price_usd"] if cat_entry else 0.0
        quantity = intent.quantity or 1

        return RFQResult(
            email_text=email_text,
            customer_name=customer_name,
            extracted_intent=intent,
            hypotheses=hypotheses,
            route=Route.QUOTE,
            route_reasoning=route_reasoning,
            quote=QuoteDraft(
                sku=top.sku,
                description=top.description,
                quantity=quantity,
                unit_price_usd=unit_price,
                total_price_usd=round(unit_price * quantity, 2),
            ),
        )

    if top.confidence >= CONFIDENCE_HUMAN_THRESHOLD:
        # Generate a targeted clarifying question
        top_two = hypotheses[:2]
        disambig_specs = intent.specs_missing[:2] if intent.specs_missing else ["thread_standard"]

        if len(top_two) >= 2:
            question = (
                f"Quick check before I send the quote — should these be "
                f"{top_two[0].description} or {top_two[1].description}? "
                f"The key difference is {', '.join(disambig_specs)}. "
                f"They're priced similarly but aren't interchangeable."
            )
        else:
            question = (
                f"To get you the right part, I need to confirm: "
                f"what {', '.join(disambig_specs)} do you need? "
                f"My best guess is {top_two[0].description} based on your history."
            )

        return RFQResult(
            email_text=email_text,
            customer_name=customer_name,
            extracted_intent=intent,
            hypotheses=hypotheses,
            route=Route.CLARIFY,
            route_reasoning=route_reasoning,
            clarification=ClarifyingQuestion(
                question=question,
                top_hypotheses=top_two,
                disambiguating_specs=disambig_specs,
            ),
        )

    # Low confidence — escalate
    return RFQResult(
        email_text=email_text,
        customer_name=customer_name,
        extracted_intent=intent,
        hypotheses=hypotheses,
        route=Route.HUMAN,
        route_reasoning=route_reasoning,
        escalation=HumanEscalation(
            reasoning=route_reasoning,
            issues=intent.specs_missing or ["Low confidence across all hypotheses"],
        ),
    )


def process_rfq(email_text: str, customer_name: str | None = None) -> RFQResult:
    """Main entry point: process an RFQ email through the two-call pipeline.

    Call 1 (Llama 8B via Groq): Extract structured intent from raw email.
    Call 2 (Llama 70B via Groq): Generate ranked SKU hypotheses with confidence scores.
    Then route based on top confidence: quote / clarify / escalate.
    """
    client = Groq()
    catalog = _load_catalog()
    customers = _load_customers()
    customer = _find_customer(customer_name, customers)

    logger.info("Processing RFQ for customer=%s", customer_name)

    # Call 1: Extract intent
    intent = _extract_intent(client, email_text)
    logger.info("Extracted intent: product_type=%s, specs_missing=%s", intent.product_type, intent.specs_missing)

    # Call 2: Generate hypotheses
    hypotheses, route_reasoning = _generate_hypotheses(client, intent, customer, catalog)
    logger.info(
        "Generated %d hypotheses, top confidence=%.2f",
        len(hypotheses),
        hypotheses[0].confidence if hypotheses else 0.0,
    )

    # Route and build result
    return _route_result(email_text, customer_name, intent, hypotheses, route_reasoning, catalog)
