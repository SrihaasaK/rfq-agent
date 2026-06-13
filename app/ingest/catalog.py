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

    Expected columns: sku, type, description, size, thread_standard,
    material_grade, pressure_rating_psi, price_usd, common_end_markets
    (common_end_markets is a ';'-separated list).

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
                    size=row["size"],
                    thread_standard=row["thread_standard"],
                    material_grade=row["material_grade"],
                    pressure_rating_psi=int(row["pressure_rating_psi"]),
                    price_usd=float(row["price_usd"]),
                    common_end_markets=row["common_end_markets"].split(";") if row["common_end_markets"] else [],
                )
            )

    return len(rows)


if __name__ == "__main__":
    count = load_catalog_csv()
    print(f"Loaded {count} catalog items from {CATALOG_CSV_PATH}")
