"""Load a catalog CSV into the catalog_items table (replaces existing rows)."""

from __future__ import annotations

import csv
from pathlib import Path

from sqlalchemy import delete

from app.config import CATALOG_CSV_PATH
from app.db.models import CatalogItem
from app.db.session import get_session


def load_catalog_csv(path: Path = CATALOG_CSV_PATH) -> int:
    """Replace the catalog_items table with the contents of the given CSV.

    Expected columns: sku, type, description, diameter, thread_pitch_or_tpi,
    length, grade_or_class, material, finish_coating, head_type, drive_type,
    standard, thread_direction, units, price_usd.

    Returns the number of rows loaded.
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    with get_session() as session:
        session.execute(delete(CatalogItem))
        for row in rows:
            session.add(
                CatalogItem(
                    sku=row["sku"],
                    type=row["type"],
                    description=row["description"],
                    diameter=row["diameter"],
                    thread_pitch_or_tpi=row["thread_pitch_or_tpi"],
                    length=row["length"],
                    grade_or_class=row["grade_or_class"],
                    material=row["material"],
                    finish_coating=row["finish_coating"],
                    head_type=row["head_type"],
                    drive_type=row["drive_type"],
                    standard=row["standard"],
                    thread_direction=row["thread_direction"],
                    units=row["units"],
                    price_usd=float(row["price_usd"]),
                )
            )

    return len(rows)


if __name__ == "__main__":
    count = load_catalog_csv()
    print(f"Loaded {count} catalog items from {CATALOG_CSV_PATH}")
