"""Normalize fastener attributes to catalog-canonical form.

This is a deterministic pass run after LLM extraction, so identical specs always
map to the same canonical string regardless of how the customer phrased them —
important both for matcher consistency (hard-filtering needs exact-string
comparisons) and for the audit trail.

Canonical forms:
- diameter: imperial fraction with trailing '"' (e.g. '1/4"'), imperial numbered
  screw size like '#10', or metric like 'M8'.
- thread_pitch_or_tpi: TPI as a plain integer string (e.g. '20') for imperial,
  or metric pitch in mm as a decimal string (e.g. '1.25').
- length: imperial fraction with trailing '"' (e.g. '1-1/2"'), or metric length
  in mm as a plain number string (e.g. '30').
- grade_or_class: 'Grade 5', 'Grade 8', 'Class 8.8', 'Class 10.9', 'Class 12.9',
  'A2-70', 'A4-70', '18-8'.
- material: 'Steel', 'Alloy Steel', 'Stainless Steel', 'Brass'.
- finish_coating: 'Zinc', 'Zinc Yellow', 'Black Oxide', 'Hot-Dip Galvanized', 'Plain'.
- head_type / drive_type: Title Case (e.g. 'Socket Cap' / 'Hex/Allen').
- standard: uppercase body + number as written (e.g. 'DIN 912', 'ASME B18.3').
- thread_direction: 'RH' or 'LH'.
- units: 'imperial' or 'metric'.
"""

from __future__ import annotations

import re

# Canonical fraction strings, ordered for substitution matching.
_WORD_SIZES = {
    "quarter": "1/4",
    "three eighths": "3/8",
    "three-eighths": "3/8",
    "five sixteenths": "5/16",
    "five-sixteenths": "5/16",
    "half": "1/2",
    "seven sixteenths": "7/16",
    "seven-sixteenths": "7/16",
    "three quarter": "3/4",
    "three-quarter": "3/4",
    "three quarters": "3/4",
    "five eighths": "5/8",
    "five-eighths": "5/8",
    "one and a quarter": "1-1/4",
    "one and a half": "1-1/2",
}

# Decimal inches -> canonical fraction (covers common fastener sizes).
_DECIMAL_SIZES = {
    "0.25": "1/4",
    "0.3125": "5/16",
    "0.375": "3/8",
    "0.4375": "7/16",
    "0.5": "1/2",
    "0.5625": "9/16",
    "0.625": "5/8",
    "0.75": "3/4",
    "0.875": "7/8",
    "1.0": "1",
    "1": "1",
    "1.25": "1-1/4",
    "1.5": "1-1/2",
    "2.0": "2",
    "2": "2",
}

_GRADE_ALIASES = {
    "grade 5": "Grade 5", "grade5": "Grade 5", "gr5": "Grade 5", "gr 5": "Grade 5",
    "grade 8": "Grade 8", "grade8": "Grade 8", "gr8": "Grade 8", "gr 8": "Grade 8",
    "class 8.8": "Class 8.8", "8.8": "Class 8.8",
    "class 10.9": "Class 10.9", "10.9": "Class 10.9",
    "class 12.9": "Class 12.9", "12.9": "Class 12.9",
    "a2": "A2-70", "a2-70": "A2-70", "a2 70": "A2-70",
    "a4": "A4-70", "a4-70": "A4-70", "a4 70": "A4-70",
    "18-8": "18-8", "18/8": "18-8",
}

# Grades that imply a material when material isn't stated separately.
_GRADE_TO_MATERIAL = {
    "Grade 5": "Steel",
    "Grade 8": "Alloy Steel",
    "Class 8.8": "Steel",
    "Class 10.9": "Alloy Steel",
    "Class 12.9": "Alloy Steel",
    "A2-70": "Stainless Steel",
    "A4-70": "Stainless Steel",
    "18-8": "Stainless Steel",
}

_MATERIAL_ALIASES = {
    "steel": "Steel",
    "carbon steel": "Steel",
    "alloy steel": "Alloy Steel",
    "alloy": "Alloy Steel",
    "stainless": "Stainless Steel",
    "stainless steel": "Stainless Steel",
    "ss": "Stainless Steel",
    "brass": "Brass",
}

_FINISH_ALIASES = {
    "z": "Zinc", "zinc": "Zinc", "zinc plated": "Zinc", "zp": "Zinc",
    "zinc yellow": "Zinc Yellow", "yellow zinc": "Zinc Yellow", "zinc-yellow": "Zinc Yellow",
    "hdg": "Hot-Dip Galvanized", "hot dip galvanized": "Hot-Dip Galvanized",
    "hot-dip galvanized": "Hot-Dip Galvanized", "galvanized": "Hot-Dip Galvanized",
    "black oxide": "Black Oxide", "black-oxide": "Black Oxide", "bo": "Black Oxide",
    "plain": "Plain", "none": "Plain", "uncoated": "Plain", "natural": "Plain",
}

_HEAD_ALIASES = {
    "socket cap": "Socket Cap", "socket head cap": "Socket Cap", "socket": "Socket Cap",
    "hex": "Hex", "hex head": "Hex",
    "flat": "Flat", "flat head": "Flat",
    "button": "Button", "button head": "Button",
    "pan": "Pan", "pan head": "Pan",
    "round": "Round", "round head": "Round",
    "oval": "Oval", "oval head": "Oval",
    "truss": "Truss", "truss head": "Truss",
}

_DRIVE_ALIASES = {
    "hex": "Hex/Allen", "allen": "Hex/Allen", "hex/allen": "Hex/Allen", "socket": "Hex/Allen",
    "hex wrench": "Hex (Wrench)", "wrench": "Hex (Wrench)",
    "phillips": "Phillips", "ph": "Phillips",
    "slotted": "Slotted", "slot": "Slotted",
    "torx": "Torx", "star": "Torx",
    "square": "Square", "robertson": "Square",
}

_THREAD_DIRECTION_ALIASES = {
    "lh": "LH", "left hand": "LH", "left-hand": "LH", "left": "LH",
    "rh": "RH", "right hand": "RH", "right-hand": "RH", "right": "RH",
}


def _normalize_imperial_diameter(token: str) -> str | None:
    """Try to parse `token` as an imperial diameter. Returns canonical form or None."""
    token = token.strip().lower().rstrip('"').strip()
    token = token.replace("inch", "").replace("in", "").strip()

    # Numbered machine-screw sizes: "#10", "no. 10", "no 10"
    numbered = re.match(r"^(?:#|no\.?\s*)(\d{1,2})$", token)
    if numbered:
        return f"#{numbered.group(1)}"

    for word, frac in _WORD_SIZES.items():
        if word in token:
            return f'{frac}"'

    # "1 1/4" -> "1-1/4"
    mixed = re.match(r"^(\d+)\s+(\d+/\d+)$", token)
    if mixed:
        token = f"{mixed.group(1)}-{mixed.group(2)}"

    if token in _DECIMAL_SIZES:
        return f'{_DECIMAL_SIZES[token]}"'

    # Already a fraction like "1/4", "1-1/4", or whole number like "1"
    if re.match(r"^\d+(-\d+/\d+)?(/\d+)?$", token):
        return f'{token}"'

    return None


def _normalize_metric_diameter(token: str) -> str | None:
    """Try to parse `token` as a metric diameter (e.g. 'M8', '8mm'). Returns 'M8' or None."""
    token = token.strip().lower()

    m = re.match(r"^m\s*(\d+(?:\.\d+)?)$", token)
    if m:
        val = m.group(1)
        return f"M{int(float(val)) if float(val) == int(float(val)) else val}"

    m = re.match(r"^(\d+(?:\.\d+)?)\s*mm$", token)
    if m:
        val = m.group(1)
        return f"M{int(float(val)) if float(val) == int(float(val)) else val}"

    return None


def normalize_diameter(value: str | None) -> str | None:
    """Normalize a fastener diameter to '1/4"', '#10', or 'M8' form."""
    if not value:
        return value

    metric = _normalize_metric_diameter(value)
    if metric:
        return metric

    imperial = _normalize_imperial_diameter(value)
    if imperial:
        return imperial

    return value


def normalize_thread_pitch_or_tpi(value: str | int | float | None) -> str | None:
    """Normalize TPI (e.g. '20') or metric pitch in mm (e.g. '1.25') to a plain string."""
    if value is None or value == "":
        return None
    text = str(value).strip().lower()
    text = re.sub(r"\s*(tpi|threads?\s*per\s*inch|mm\s*pitch|pitch)\s*$", "", text).strip()
    m = re.match(r"^(\d+(?:\.\d+)?)$", text)
    if m:
        val = m.group(1)
        if "." in val:
            return val
        return str(int(val))
    return text or None


def normalize_length(value: str | None) -> str | None:
    """Normalize a fastener length to imperial fraction ('1-1/2"') or metric mm ('30')."""
    if not value:
        return value

    text = value.strip().lower()

    m = re.match(r"^(\d+(?:\.\d+)?)\s*mm$", text)
    if m:
        val = m.group(1)
        return str(int(float(val))) if float(val) == int(float(val)) else val

    imperial = _normalize_imperial_diameter(text)
    if imperial:
        return imperial

    return value


def normalize_grade_or_class(value: str | None) -> str | None:
    if not value:
        return value
    key = value.strip().lower()
    return _GRADE_ALIASES.get(key, value)


def normalize_material(value: str | None, grade_or_class: str | None = None) -> str | None:
    if value:
        key = value.strip().lower()
        return _MATERIAL_ALIASES.get(key, value)
    if grade_or_class:
        return _GRADE_TO_MATERIAL.get(grade_or_class)
    return value


def normalize_finish_coating(value: str | None) -> str | None:
    if not value:
        return value
    key = value.strip().lower()
    return _FINISH_ALIASES.get(key, value)


def normalize_head_type(value: str | None) -> str | None:
    if not value:
        return value
    key = value.strip().lower()
    return _HEAD_ALIASES.get(key, value.title())


def normalize_drive_type(value: str | None) -> str | None:
    if not value:
        return value
    key = value.strip().lower()
    return _DRIVE_ALIASES.get(key, value.title())


def normalize_standard(value: str | None) -> str | None:
    if not value:
        return value
    return value.strip().upper()


def normalize_thread_direction(value: str | None) -> str | None:
    if not value:
        return value
    key = value.strip().lower()
    return _THREAD_DIRECTION_ALIASES.get(key, value.upper())


def normalize_units(value: str | None) -> str | None:
    if not value:
        return value
    key = value.strip().lower()
    if key in ("metric", "imperial"):
        return key
    return value


def normalize_unit(value: str | None) -> str | None:
    """Normalize the order-line quantity unit (e.g. 'pcs' -> 'each')."""
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

    if "diameter" in attributes:
        attributes["diameter"] = normalize_diameter(attributes["diameter"])
    if "thread_pitch_or_tpi" in attributes:
        attributes["thread_pitch_or_tpi"] = normalize_thread_pitch_or_tpi(attributes["thread_pitch_or_tpi"])
    if "length" in attributes:
        attributes["length"] = normalize_length(attributes["length"])
    if "grade_or_class" in attributes:
        attributes["grade_or_class"] = normalize_grade_or_class(attributes["grade_or_class"])
    if "material" in attributes or "grade_or_class" in attributes:
        attributes["material"] = normalize_material(attributes.get("material"), attributes.get("grade_or_class"))
    if "finish_coating" in attributes:
        attributes["finish_coating"] = normalize_finish_coating(attributes["finish_coating"])
    if "head_type" in attributes:
        attributes["head_type"] = normalize_head_type(attributes["head_type"])
    if "drive_type" in attributes:
        attributes["drive_type"] = normalize_drive_type(attributes["drive_type"])
    if "standard" in attributes:
        attributes["standard"] = normalize_standard(attributes["standard"])
    if "thread_direction" in attributes:
        attributes["thread_direction"] = normalize_thread_direction(attributes["thread_direction"])
    if "units" in attributes:
        attributes["units"] = normalize_units(attributes["units"])

    # Drop keys normalized to None (i.e. never had a usable value).
    attributes = {k: v for k, v in attributes.items() if v is not None}

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
    for key in (
        "type", "diameter", "thread_pitch_or_tpi", "length", "grade_or_class",
        "material", "finish_coating", "head_type", "drive_type", "standard",
    ):
        if attributes.get(key):
            parts.append(str(attributes[key]))

    query = " ".join(p for p in parts if p)
    return query or (line_item.get("raw_text") or "")
