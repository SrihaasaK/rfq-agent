"""Pydantic models for RFQ agent I/O."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Route(str, Enum):
    QUOTE = "quote"
    CLARIFY = "clarify"
    HUMAN = "human"


class ExtractedIntent(BaseModel):
    """Structured extraction from an RFQ email."""

    product_type: str | None = Field(None, description="e.g. elbow, tee, coupling, nipple, union, reducer")
    specs_mentioned: dict[str, Any] = Field(default_factory=dict, description="Specs explicitly stated: size, thread_standard, material_grade, pressure_rating_psi")
    specs_missing: list[str] = Field(default_factory=list, description="Critical specs not mentioned")
    quantity: int | None = None
    urgency: str | None = Field(None, description="e.g. standard, rush, ASAP")
    customer_signals: list[str] = Field(default_factory=list, description="End-market or project signals extracted from the email")
    referenced_past_orders: bool = Field(False, description="Whether the email references previous orders")


class Hypothesis(BaseModel):
    """A single SKU hypothesis with confidence and reasoning."""

    rank: int
    sku: str
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class QuoteDraft(BaseModel):
    """Output when confidence > 0.85."""

    sku: str
    description: str
    quantity: int
    unit_price_usd: float
    total_price_usd: float


class ClarifyingQuestion(BaseModel):
    """Output when 0.5 <= confidence <= 0.85."""

    question: str
    top_hypotheses: list[Hypothesis]
    disambiguating_specs: list[str] = Field(description="Which specs the question aims to resolve")


class HumanEscalation(BaseModel):
    """Output when confidence < 0.5."""

    reasoning: str
    issues: list[str]


class RFQResult(BaseModel):
    """Complete result from processing an RFQ."""

    email_text: str
    customer_name: str | None
    extracted_intent: ExtractedIntent
    hypotheses: list[Hypothesis]
    route: Route
    route_reasoning: str
    quote: QuoteDraft | None = None
    clarification: ClarifyingQuestion | None = None
    escalation: HumanEscalation | None = None
