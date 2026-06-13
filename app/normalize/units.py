"""Normalize size/thread-standard/material-grade attributes to catalog-canonical form.

This is a deterministic pass run after LLM extraction, so identical specs always
map to the same canonical string regardless of how the customer phrased them —
important both for matcher consistency and for the audit trail.
"""

from __future__ import annotations

import re

# Canonical fraction strings, ordered for substitution matching.
_WORD_SIZES = {
    "quarter": "1/4",
    "three eighths": "3/8",
    "three-eighths": "3/8",
    "half": "1/2",
    "three quarter": "3/4",
    "three-quarter": "3/4",
    "three quarters": "3/4",
    "one and a quarter": "1-1/4",
    "one and a half": "1-1/2",
    "two inch": "2",
}

# Decimal inches -> canonical fraction (covers the sizes present in this catalog).
_DECIMAL_SIZES = {
    "0.25": "1/4",
    "0.375": "3/8",
    "0.5": "1/2",
    "0.75": "3/4",
    "1.0": "1",
    "1.25": "1-1/4",
    "1.5": "1-1/2",
    "2.0": "2",
    "2": "2",
    "1": "1",
}

_THREAD_ALIASES = {
    "npt": "NPT",
    "n.p.t.": "NPT",
    "national pipe thread": "NPT",
    "bsp": "BSP",
    "bspt": "BSP",
    "british standard pipe": "BSP",
    "compression": "compression",
    "comp": "compression",
    "sweat": "sweat",
    "solder": "sweat",
    "soldered": "sweat",
    "flare": "flare",
    "flared": "flare",
}

_MATERIAL_ALIASES = {
    "c36000": "C36000",
    "c-36000": "C36000",
    "360 brass": "C36000",
    "free-machining brass": "C36000",
    "free machining brass": "C36000",
    "c46400": "C46400",
    "c-46400": "C46400",
    "naval brass": "C46400",
    "c84400": "C84400",
    "c-84400": "C84400",
    "lead-free bronze": "C84400",
    "lead free bronze": "C84400",
}


def _normalize_single_size(token: str) -> str:
    token = token.strip().lower().rstrip('"').strip()
    token = token.replace("inch", "").replace("in", "").strip()

    for word, frac in _WORD_SIZES.items():
        if word in token:
            return f'{frac}"'

    # "1 1/4" -> "1-1/4"
    mixed = re.match(r"^(\d+)\s+(\d+/\d+)$", token)
    if mixed:
        token = f"{mixed.group(1)}-{mixed.group(2)}"

    if token in _DECIMAL_SIZES:
        return f'{_DECIMAL_SIZES[token]}"'

    # Already a fraction like "1/2", "1-1/4", or whole number like "2"
    if re.match(r"^\d+(-\d+/\d+)?(/\d+)?$", token):
        return f'{token}"'

    return token


def normalize_size(value: str | None) -> str | None:
    """Normalize a size or size-pair (e.g. reducer/bushing "3/4 x 1/2") to canonical form."""
    if not value:
        return value

    # Size pair, e.g. "3/4 x 1/2", "3/4\" X 1/2\""
    pair = re.split(r"\s*[xX]\s*", value.strip())
    if len(pair) == 2:
        return f"{_normalize_single_size(pair[0])} x {_normalize_single_size(pair[1])}"

    return _normalize_single_size(value)


def normalize_thread_standard(value: str | None) -> str | None:
    if not value:
        return value
    key = value.strip().lower()
    return _THREAD_ALIASES.get(key, value)


def normalize_material_grade(value: str | None) -> str | None:
    if not value:
        return value
    key = value.strip().lower()
    return _MATERIAL_ALIASES.get(key, value.upper() if re.match(r"^c\d{5}$", key) else value)


def normalize_unit(value: str | None) -> str | None:
    if not value:
        return value
    key = value.strip().lower()
    if key in ("ea", "each", "pc", "pcs", "piece", "pieces", "unit", "units"):
        return "each"
    return value


def normalize_line_item(line_item: dict) -> dict:
    """Return a copy of the extracted line item with standardized attributes/unit."""
    normalized = dict(line_item)
    attributes = dict(normalized.get("attributes") or {})

    if "size" in attributes:
        attributes["size"] = normalize_size(attributes["size"])
    if "thread_standard" in attributes:
        attributes["thread_standard"] = normalize_thread_standard(attributes["thread_standard"])
    if "material_grade" in attributes:
        attributes["material_grade"] = normalize_material_grade(attributes["material_grade"])

    normalized["attributes"] = attributes
    normalized["unit"] = normalize_unit(normalized.get("unit"))
    return normalized


def build_query_text(line_item: dict) -> str:
    """Build a single text string from a (normalized) line item for the matcher to search on.

    Deliberately excludes raw_text: conversational filler ("need", "please
    advise") dilutes BM25/embedding relevance versus the catalog's terse
    spec-sheet style descriptions.
    """
    parts = [line_item.get("description") or ""]
    attributes = line_item.get("attributes") or {}
    for key in ("type", "size", "thread_standard", "material_grade"):
        if attributes.get(key):
            parts.append(str(attributes[key]))

    query = " ".join(p for p in parts if p)
    return query or (line_item.get("raw_text") or "")
