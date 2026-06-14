"""Load RFQ text from plain-text, PDF, or Excel BOM files.

PDF and Excel inputs are flattened to plain text so they can go through the
same LLM extraction prompt as an emailed RFQ — the extraction prompt is robust
to tabular/CSV-like text as well as prose.
"""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd


def extract_text_from_pdf(path: Path) -> str:
    """Concatenate the text of every page in a PDF."""
    with fitz.open(path) as doc:
        return "\n".join(page.get_text() for page in doc)


def extract_text_from_excel(path: Path) -> str:
    """Flatten every sheet of an Excel workbook into CSV-like text."""
    sheets = pd.read_excel(path, sheet_name=None, dtype=str)

    parts = []
    for sheet_name, df in sheets.items():
        df = df.fillna("")
        parts.append(f"Sheet: {sheet_name}")
        parts.append(", ".join(str(c) for c in df.columns))
        for _, row in df.iterrows():
            parts.append(", ".join(str(v) for v in row.values))

    return "\n".join(parts)


def load_rfq_text(path: Path) -> str:
    """Load RFQ text from a .txt, .pdf, or .xlsx/.xls file based on its extension."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    if suffix in (".xlsx", ".xls"):
        return extract_text_from_excel(path)
    return path.read_text()
