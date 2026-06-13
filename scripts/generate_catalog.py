"""Generate a synthetic ~170-SKU brass/bronze plumbing-fittings catalog CSV.

Scales up the original 28-SKU catalog (catalog/catalog.json) with more sizes,
thread standards, materials, and fitting types so the hybrid matcher (Phase 2)
has a realistic-sized catalog to index and search.

Run: python3 scripts/generate_catalog.py
Writes: catalog/catalog.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

OUT_PATH = Path(__file__).parent.parent / "catalog" / "catalog.csv"

ALL_SIZES = ['1/4"', '3/8"', '1/2"', '3/4"', '1"', '1-1/4"', '1-1/2"', '2"']
SMALL_SIZES = ['1/4"', '3/8"', '1/2"', '3/4"']
SMALL3 = ['1/4"', '3/8"', '1/2"']
MED_SIZES = ['3/4"', '1"', '1-1/4"', '1-1/2"']
SWEAT_SIZES = ['1/4"', '3/8"', '1/2"', '3/4"', '1"']
NAVAL_SIZES_4 = ['1/2"', '3/4"', '1"', '1-1/4"']
BSP_SIZES_3 = ['1/2"', '3/4"', '1"']

SIZE_MULT = {
    '1/4"': 0.55,
    '3/8"': 0.70,
    '1/2"': 1.00,
    '3/4"': 1.45,
    '1"': 2.10,
    '1-1/4"': 2.80,
    '1-1/2"': 3.50,
    '2"': 4.60,
}

SIZE_VALUE = {
    '1/4"': 0.25, '3/8"': 0.375, '1/2"': 0.5, '3/4"': 0.75,
    '1"': 1.0, '1-1/4"': 1.25, '1-1/2"': 1.5, '2"': 2.0,
}

MATERIALS = {
    "C36000": {"name": "Brass", "mult": 1.00},
    "C46400": {"name": "Naval Brass", "mult": 1.55},
    "C84400": {"name": "Lead-Free Bronze", "mult": 1.75},
}

THREADS = {
    "NPT": {"mult": 1.00},
    "BSP": {"mult": 1.06},
    "compression": {"mult": 1.18},
    "sweat": {"mult": 0.96},
    "flare": {"mult": 1.10},
}

PRESSURE = {
    ("C36000", "NPT"): 150,
    ("C36000", "BSP"): 150,
    ("C36000", "compression"): 200,
    ("C36000", "sweat"): 200,
    ("C36000", "flare"): 200,
    ("C46400", "NPT"): 300,
    ("C46400", "BSP"): 300,
    ("C84400", "NPT"): 250,
    ("C84400", "BSP"): 250,
}

END_MARKETS = {
    ("C36000", "NPT"): ["commercial plumbing", "HVAC", "industrial"],
    ("C36000", "BSP"): ["marine", "import equipment", "industrial"],
    ("C36000", "compression"): ["residential plumbing", "water heater", "refrigeration"],
    ("C36000", "sweat"): ["residential plumbing", "hydronic heating"],
    ("C36000", "flare"): ["refrigeration", "HVAC", "gas lines"],
    ("C46400", "NPT"): ["marine", "naval", "offshore"],
    ("C46400", "BSP"): ["marine", "naval", "offshore"],
    ("C84400", "NPT"): ["potable water", "food processing", "pharmaceutical"],
    ("C84400", "BSP"): ["potable water", "food processing", "pharmaceutical"],
}

# Each entry: (type_key, label, base_price, sku_start, combos)
# combos: list of (sizes, thread, material) for single-size types,
#         or (size_pairs, thread, material) for two-size types (reducer/bushing)
TYPE_CONFIGS = [
    ("elbow", "90° Elbow", 4.00, 1000, [
        (ALL_SIZES, "NPT", "C36000"),
        (BSP_SIZES_3, "BSP", "C36000"),
        (SMALL_SIZES, "compression", "C36000"),
        (SWEAT_SIZES, "sweat", "C36000"),
        (NAVAL_SIZES_4, "NPT", "C46400"),
        (MED_SIZES, "NPT", "C84400"),
        (SMALL3, "flare", "C36000"),
    ]),
    ("elbow_45", "45° Elbow", 4.50, 1100, [
        (['1/2"', '3/4"', '1"', '1-1/4"'], "NPT", "C36000"),
        (['1/2"', '3/4"'], "BSP", "C36000"),
        (['1/2"', '3/4"'], "NPT", "C46400"),
    ]),
    ("tee", "Tee", 6.00, 2000, [
        (ALL_SIZES, "NPT", "C36000"),
        (SMALL_SIZES, "compression", "C36000"),
        (MED_SIZES, "NPT", "C84400"),
        (BSP_SIZES_3, "BSP", "C36000"),
        (['1/4"', '3/8"', '1/2"'], "sweat", "C36000"),
    ]),
    ("coupling", "Coupling", 3.20, 3000, [
        (ALL_SIZES, "NPT", "C36000"),
        (BSP_SIZES_3, "BSP", "C36000"),
        (MED_SIZES, "NPT", "C84400"),
        (SMALL_SIZES, "compression", "C36000"),
        (SWEAT_SIZES, "sweat", "C36000"),
        (SMALL3, "flare", "C36000"),
    ]),
    ("nipple", "Nipple", 2.50, 4000, [
        (ALL_SIZES, "NPT", "C36000"),
        (NAVAL_SIZES_4, "NPT", "C46400"),
        (MED_SIZES, "NPT", "C84400"),
        (SMALL3, "flare", "C36000"),
    ]),
    ("union", "Union", 10.00, 5000, [
        (ALL_SIZES, "NPT", "C36000"),
        (MED_SIZES, "NPT", "C84400"),
        (SMALL_SIZES, "compression", "C36000"),
    ]),
    ("plug", "Plug", 1.80, 7000, [
        (ALL_SIZES, "NPT", "C36000"),
        (['1/2"', '3/4"', '1"'], "NPT", "C46400"),
    ]),
    ("cap", "Cap", 1.60, 8000, [
        (ALL_SIZES, "NPT", "C36000"),
        (['3/4"', '1"'], "NPT", "C84400"),
    ]),
    ("cross", "Cross", 8.00, 9000, [
        (['1/2"', '3/4"', '1"', '1-1/4"'], "NPT", "C36000"),
    ]),
]

REDUCER_PAIRS = [
    ('3/4"', '1/2"'), ('1"', '3/4"'), ('1"', '1/2"'), ('1-1/4"', '1"'),
    ('1-1/4"', '3/4"'), ('1-1/2"', '1"'), ('1-1/2"', '1-1/4"'),
    ('2"', '1-1/2"'), ('2"', '1"'), ('1/2"', '3/8"'), ('3/8"', '1/4"'),
]

BUSHING_PAIRS = [
    ('1"', '3/4"'), ('3/4"', '1/2"'), ('1/2"', '3/8"'), ('3/4"', '3/8"'), ('1"', '1/2"'),
]


def _pressure(material: str, thread: str, size_value: float) -> int:
    base = PRESSURE.get((material, thread), 150)
    if size_value >= 1.0:
        base -= 25
    return base


def _row(sku: str, type_key: str, label: str, size_label: str, size_value: float,
         thread: str, material: str, base_price: float, two_size_factor: float = 1.0) -> dict:
    mat = MATERIALS[material]
    price = round(base_price * SIZE_MULT_LOOKUP(size_value) * mat["mult"] * THREADS[thread]["mult"] * two_size_factor, 2)
    pressure = _pressure(material, thread, size_value)
    markets = END_MARKETS.get((material, thread), ["industrial"])
    description = f'{size_label} {thread} {mat["name"]} ({material}) {label}, {pressure} PSI'
    return {
        "sku": sku,
        "type": type_key,
        "description": description,
        "size": size_label,
        "thread_standard": thread,
        "material_grade": material,
        "pressure_rating_psi": pressure,
        "price_usd": price,
        "common_end_markets": ";".join(markets),
    }


def SIZE_MULT_LOOKUP(size_value: float) -> float:
    for size_label, val in SIZE_VALUE.items():
        if val == size_value:
            return SIZE_MULT[size_label]
    raise KeyError(size_value)


def generate() -> list[dict]:
    rows: list[dict] = []

    for type_key, label, base_price, sku_start, combos in TYPE_CONFIGS:
        counter = 0
        for sizes, thread, material in combos:
            for size_label in sizes:
                sku = f"BF-{sku_start + counter}"
                rows.append(_row(
                    sku, type_key, label, size_label, SIZE_VALUE[size_label],
                    thread, material, base_price,
                ))
                counter += 1

    # Reducers (two-size SKUs): price keyed off the larger size, +10% machining premium
    counter = 0
    for sizes, thread, material in [
        (REDUCER_PAIRS, "NPT", "C36000"),
        (REDUCER_PAIRS[:4], "NPT", "C84400"),
    ]:
        for big, small in sizes:
            sku = f"BF-{6000 + counter}"
            size_label = f"{big} x {small}"
            mat = MATERIALS[material]
            price = round(4.00 * SIZE_MULT[big] * mat["mult"] * THREADS[thread]["mult"] * 1.10, 2)
            pressure = _pressure(material, thread, SIZE_VALUE[big])
            markets = END_MARKETS.get((material, thread), ["industrial"])
            description = f'{size_label} {thread} {mat["name"]} ({material}) Reducer, {pressure} PSI'
            rows.append({
                "sku": sku, "type": "reducer", "description": description,
                "size": size_label, "thread_standard": thread, "material_grade": material,
                "pressure_rating_psi": pressure, "price_usd": price,
                "common_end_markets": ";".join(markets),
            })
            counter += 1

    # Bushings (two-size SKUs)
    counter = 0
    for sizes, thread, material in [
        (BUSHING_PAIRS, "NPT", "C36000"),
        (BUSHING_PAIRS[:2], "NPT", "C46400"),
    ]:
        for big, small in sizes:
            sku = f"BF-{9500 + counter}"
            size_label = f"{big} x {small}"
            mat = MATERIALS[material]
            price = round(3.00 * SIZE_MULT[big] * mat["mult"] * THREADS[thread]["mult"] * 1.10, 2)
            pressure = _pressure(material, thread, SIZE_VALUE[big])
            markets = END_MARKETS.get((material, thread), ["industrial"])
            description = f'{size_label} {thread} {mat["name"]} ({material}) Bushing, {pressure} PSI'
            rows.append({
                "sku": sku, "type": "bushing", "description": description,
                "size": size_label, "thread_standard": thread, "material_grade": material,
                "pressure_rating_psi": pressure, "price_usd": price,
                "common_end_markets": ";".join(markets),
            })
            counter += 1

    return rows


def main() -> None:
    rows = generate()
    fieldnames = ["sku", "type", "description", "size", "thread_standard", "material_grade",
                   "pressure_rating_psi", "price_usd", "common_end_markets"]
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} SKUs to {OUT_PATH}")


if __name__ == "__main__":
    main()
