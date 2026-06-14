"""Generate the PDF and Excel sample RFQs under samples/.

The plain-text samples (fastener_email_01, 02, 05) are checked in directly;
this script produces the binary samples (03 = PDF, 04 = Excel BOM) so they
can be regenerated if their content needs to change.

Run: python3 scripts/generate_fastener_samples.py
"""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd

SAMPLES_DIR = Path(__file__).parent.parent / "samples"

PDF_TEXT = """RFQ #4471
Vendor: ABC Fasteners Inc.
Date: 2026-06-10

Please quote the following line items:

1. Qty 2000 - M6 x 16mm DIN 912 socket head cap screw, A2-70 stainless steel
2. Qty 50 - 3/4-10 x 8" Grade 8 hex bolt, left-hand thread (LH), for rotating
   shaft assembly

Please provide unit price and lead time for each line item.

Regards,
Procurement Team
ABC Fasteners Inc.
"""


def write_pdf() -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), PDF_TEXT, fontsize=11, fontname="helv")
    out_path = SAMPLES_DIR / "fastener_email_03_rfq.pdf"
    doc.save(out_path)
    doc.close()
    print(f"Wrote {out_path}")


def write_excel() -> None:
    rows = [
        {"Item": 1, "Description": '1/4-20 x 3/4" Socket Head Cap Screw, A2 stainless', "Qty": 300, "Notes": ""},
        {"Item": 2, "Description": "M10-1.5 x 50mm Hex Bolt, Class 10.9", "Qty": 100, "Notes": "zinc preferred"},
        {"Item": 3, "Description": '3/8-16 Flat Washer, stainless', "Qty": 500, "Notes": ""},
        {"Item": 4, "Description": '#8-32 x 1/2" Pan Head Phillips Machine Screw, stainless steel', "Qty": 1000, "Notes": ""},
    ]
    df = pd.DataFrame(rows)
    out_path = SAMPLES_DIR / "fastener_email_04_bom.xlsx"
    df.to_excel(out_path, index=False, sheet_name="BOM")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    write_pdf()
    write_excel()
