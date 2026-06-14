"""Generate a synthetic ~3000-SKU industrial fastener catalog CSV.

Covers cap screws, hex bolts, machine screws, nuts, washers, and anchors across
imperial and metric size systems, with grade/material/finish/head/drive variation,
so the hybrid matcher's diameter/thread/length hard filter has realistic depth
to retrieve and rank within.

Run: python3 scripts/generate_fastener_catalog.py
Writes: catalog/catalog.csv
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

OUT_PATH = Path(__file__).parent.parent / "catalog" / "catalog.csv"
FIELDNAMES = [
    "sku", "type", "description", "diameter", "thread_pitch_or_tpi", "length",
    "grade_or_class", "material", "finish_coating", "head_type", "drive_type",
    "standard", "thread_direction", "units", "price_usd",
]

rng = random.Random(42)

# UNC (coarse) threads-per-inch by imperial diameter.
UNC_TPI = {
    "#2": "56", "#4": "40", "#6": "32", "#8": "32", "#10": "24", "#12": "24",
    '1/4"': "20", '5/16"': "18", '3/8"': "16", '7/16"': "14", '1/2"': "13",
    '5/8"': "11", '3/4"': "10", '7/8"': "9", '1"': "8", '1-1/4"': "7", '1-1/2"': "6",
}

# Coarse metric pitch (mm) by diameter.
METRIC_PITCH = {
    "M2": "0.4", "M2.5": "0.45", "M3": "0.5", "M4": "0.7", "M5": "0.8",
    "M6": "1", "M8": "1.25", "M10": "1.5", "M12": "1.75", "M16": "2", "M20": "2.5", "M24": "3",
}

# Relative size weight used for price scaling (roughly proportional to mass/diameter^2).
DIAM_WEIGHT = {
    "#2": 0.30, "#4": 0.35, "#6": 0.40, "#8": 0.45, "#10": 0.50, "#12": 0.55,
    '1/4"': 0.60, '5/16"': 0.75, '3/8"': 0.90, '7/16"': 1.05, '1/2"': 1.20,
    '5/8"': 1.50, '3/4"': 1.80, '7/8"': 2.10, '1"': 2.40, '1-1/4"': 3.00, '1-1/2"': 3.60,
    "M2": 0.30, "M2.5": 0.35, "M3": 0.40, "M4": 0.50, "M5": 0.60, "M6": 0.75,
    "M8": 1.00, "M10": 1.25, "M12": 1.50, "M16": 2.00, "M20": 2.50, "M24": 3.00,
}

MATERIAL_MULT = {"Steel": 1.0, "Alloy Steel": 1.25, "Stainless Steel": 2.3}
FINISH_MULT = {"Zinc": 1.0, "Zinc Yellow": 1.05, "Black Oxide": 1.05, "Hot-Dip Galvanized": 1.18, "Plain": 1.0}


def price(base: float, diameter: str, length_idx: int, material: str, finish: str) -> float:
    p = base * DIAM_WEIGHT[diameter] * (1 + 0.10 * length_idx) * MATERIAL_MULT[material] * FINISH_MULT.get(finish, 1.0)
    p *= rng.uniform(0.92, 1.08)
    return round(p, 4)


def disp(diameter: str) -> str:
    """Diameter for use in human-readable descriptions (strip trailing inch mark)."""
    return diameter.rstrip('"')


_counters: dict[str, int] = {}


def next_sku(prefix: str) -> str:
    _counters[prefix] = _counters.get(prefix, 0) + 1
    return f"{prefix}-{_counters[prefix]:05d}"


def row(prefix, type_, description, diameter, thread, length, grade, material, finish,
        head, drive, standard, units, price_usd) -> dict:
    return {
        "sku": next_sku(prefix),
        "type": type_,
        "description": description,
        "diameter": diameter,
        "thread_pitch_or_tpi": thread,
        "length": length,
        "grade_or_class": grade,
        "material": material,
        "finish_coating": finish,
        "head_type": head,
        "drive_type": drive,
        "standard": standard,
        "thread_direction": "RH",
        "units": units,
        "price_usd": price_usd,
    }


# ---------------------------------------------------------------------------
# Cap screws (socket head cap screws)
# ---------------------------------------------------------------------------

def gen_cap_screws_imperial() -> list[dict]:
    rows = []
    tiers = [
        (['#4', '#6', '#8', '#10'], ['1/4"', '3/8"', '1/2"', '5/8"', '3/4"', '1"', '1-1/4"', '1-1/2"']),
        (['1/4"', '5/16"', '3/8"'], ['3/8"', '1/2"', '5/8"', '3/4"', '1"', '1-1/4"', '1-1/2"', '2"', '2-1/2"']),
        (['7/16"', '1/2"', '5/8"', '3/4"', '7/8"', '1"'],
         ['1/2"', '5/8"', '3/4"', '1"', '1-1/4"', '1-1/2"', '2"', '2-1/2"', '3"', '4"']),
    ]
    combos = [
        ("Grade 8", "Alloy Steel", "Black Oxide"),
        ("Grade 8", "Alloy Steel", "Zinc"),
        ("A2-70", "Stainless Steel", "Plain"),
        ("A4-70", "Stainless Steel", "Plain"),
    ]
    for diameters, lengths in tiers:
        for d in diameters:
            tpi = UNC_TPI[d]
            for li, length in enumerate(lengths):
                for grade, material, finish in combos:
                    desc = f'{disp(d)}-{tpi} x {length} Socket Head Cap Screw, {grade} {material}, {finish}, ASME B18.3'
                    rows.append(row("CSI", "cap_screw", desc, d, tpi, length, grade, material, finish,
                                     "Socket Cap", "Hex/Allen", "ASME B18.3", "imperial",
                                     price(0.08, d, li, material, finish)))
    return rows


def gen_cap_screws_metric() -> list[dict]:
    rows = []
    tiers = [
        (['M3', 'M4', 'M5'], ["8", "10", "12", "16", "20", "25", "30", "35"]),
        (['M6', 'M8', 'M10'], ["12", "16", "20", "25", "30", "35", "40", "50", "60"]),
        (['M12', 'M16', 'M20', 'M24'], ["20", "25", "30", "40", "50", "60", "70", "80", "100", "120"]),
    ]
    combos = [
        ("Class 12.9", "Alloy Steel", "Black Oxide"),
        ("Class 12.9", "Alloy Steel", "Zinc"),
        ("Class 10.9", "Alloy Steel", "Plain"),
        ("A2-70", "Stainless Steel", "Plain"),
    ]
    for diameters, lengths in tiers:
        for d in diameters:
            pitch = METRIC_PITCH[d]
            for li, length in enumerate(lengths):
                for grade, material, finish in combos:
                    desc = f'{d}x{pitch} x {length}mm Socket Head Cap Screw, {grade} {material}, {finish}, DIN 912'
                    rows.append(row("CSM", "cap_screw", desc, d, pitch, length, grade, material, finish,
                                     "Socket Cap", "Hex/Allen", "DIN 912", "metric",
                                     price(0.06, d, li, material, finish)))
    return rows


# ---------------------------------------------------------------------------
# Hex bolts
# ---------------------------------------------------------------------------

def gen_hex_bolts_imperial() -> list[dict]:
    rows = []
    tiers = [
        (['1/4"', '5/16"', '3/8"'], ['1"', '1-1/4"', '1-1/2"', '2"', '2-1/2"', '3"', '3-1/2"', '4"']),
        (['7/16"', '1/2"', '5/8"'], ['1"', '1-1/2"', '2"', '2-1/2"', '3"', '3-1/2"', '4"', '5"', '6"']),
        (['3/4"', '7/8"', '1"'], ['1-1/2"', '2"', '2-1/2"', '3"', '3-1/2"', '4"', '5"', '6"', '8"', '10"']),
        (['1-1/4"', '1-1/2"'], ['2"', '2-1/2"', '3"', '4"', '5"', '6"', '8"', '10"', '12"']),
    ]
    combos = [
        ("Grade 5", "Steel", "Zinc"),
        ("Grade 5", "Steel", "Hot-Dip Galvanized"),
        ("Grade 8", "Alloy Steel", "Plain"),
        ("A2-70", "Stainless Steel", "Plain"),
        ("A4-70", "Stainless Steel", "Plain"),
    ]
    for diameters, lengths in tiers:
        for d in diameters:
            tpi = UNC_TPI[d]
            for li, length in enumerate(lengths):
                for grade, material, finish in combos:
                    desc = f'{disp(d)}-{tpi} x {length} Hex Bolt, {grade} {material}, {finish}, ASME B18.2.1'
                    rows.append(row("HBI", "hex_bolt", desc, d, tpi, length, grade, material, finish,
                                     "Hex", "Hex (Wrench)", "ASME B18.2.1", "imperial",
                                     price(0.10, d, li, material, finish)))
    return rows


def gen_hex_bolts_metric() -> list[dict]:
    rows = []
    tiers = [
        (['M6', 'M8'], ["16", "20", "25", "30", "35", "40", "45", "50"]),
        (['M10', 'M12'], ["20", "25", "30", "40", "50", "60", "70", "80"]),
        (['M16', 'M20', 'M24'], ["30", "40", "50", "60", "70", "80", "100", "120"]),
    ]
    combos = [
        ("Class 8.8", "Steel", "Zinc"),
        ("Class 8.8", "Steel", "Hot-Dip Galvanized"),
        ("Class 10.9", "Alloy Steel", "Plain"),
        ("Class 12.9", "Alloy Steel", "Black Oxide"),
        ("A2-70", "Stainless Steel", "Plain"),
    ]
    for diameters, lengths in tiers:
        for d in diameters:
            pitch = METRIC_PITCH[d]
            for li, length in enumerate(lengths):
                for grade, material, finish in combos:
                    desc = f'{d}x{pitch} x {length}mm Hex Bolt, {grade} {material}, {finish}, DIN 933'
                    rows.append(row("HBM", "hex_bolt", desc, d, pitch, length, grade, material, finish,
                                     "Hex", "Hex (Wrench)", "DIN 933", "metric",
                                     price(0.08, d, li, material, finish)))
    return rows


# ---------------------------------------------------------------------------
# Machine screws
# ---------------------------------------------------------------------------

def gen_machine_screws_imperial() -> list[dict]:
    rows = []
    diameters = ['#2', '#4', '#6', '#8', '#10', '#12', '1/4"']
    lengths = ['1/4"', '3/8"', '1/2"', '5/8"', '3/4"', '1"', '1-1/4"', '1-1/2"', '2"']
    head_drive = [("Pan", "Phillips"), ("Flat", "Phillips"), ("Round", "Slotted"), ("Truss", "Phillips")]
    materials = [("18-8", "Stainless Steel", "Plain"), ("Commercial", "Steel", "Zinc")]
    for d in diameters:
        tpi = UNC_TPI[d]
        for li, length in enumerate(lengths):
            for head, drive in head_drive:
                for grade, material, finish in materials:
                    desc = f'{disp(d)}-{tpi} x {length} {head} Head {drive} Machine Screw, {material}, {finish}, ASME B18.6.3'
                    rows.append(row("MSI", "machine_screw", desc, d, tpi, length, grade, material, finish,
                                     head, drive, "ASME B18.6.3", "imperial",
                                     price(0.03, d, li, material, finish)))
    return rows


def gen_machine_screws_metric() -> list[dict]:
    rows = []
    diameters = ['M2', 'M2.5', 'M3', 'M4', 'M5', 'M6']
    lengths = ["6", "8", "10", "12", "16", "20", "25", "30", "35"]
    head_drive = [("Pan", "Phillips"), ("Flat", "Phillips"), ("Button", "Hex/Allen"), ("Round", "Slotted")]
    materials = [("A2-70", "Stainless Steel", "Plain"), ("Commercial", "Steel", "Zinc")]
    for d in diameters:
        pitch = METRIC_PITCH[d]
        for li, length in enumerate(lengths):
            for head, drive in head_drive:
                for grade, material, finish in materials:
                    desc = f'{d}x{pitch} x {length}mm {head} Head {drive} Machine Screw, {material}, {finish}, DIN 7985'
                    rows.append(row("MSM", "machine_screw", desc, d, pitch, length, grade, material, finish,
                                     head, drive, "DIN 7985", "metric",
                                     price(0.025, d, li, material, finish)))
    return rows


# ---------------------------------------------------------------------------
# Nuts (length is not a meaningful spec for nuts -> "N/A")
# ---------------------------------------------------------------------------

def gen_nuts_imperial() -> list[dict]:
    rows = []
    diameters = ['#4', '#6', '#8', '#10', '#12', '1/4"', '5/16"', '3/8"', '7/16"',
                  '1/2"', '5/8"', '3/4"', '7/8"', '1"', '1-1/4"']
    combos = [
        ("Grade 5", "Steel", "Zinc"),
        ("Grade 5", "Steel", "Hot-Dip Galvanized"),
        ("Grade 8", "Alloy Steel", "Plain"),
        ("A2-70", "Stainless Steel", "Plain"),
        ("A4-70", "Stainless Steel", "Plain"),
        ("18-8", "Stainless Steel", "Plain"),
    ]
    for d in diameters:
        tpi = UNC_TPI[d]
        for grade, material, finish in combos:
            desc = f'{disp(d)}-{tpi} Hex Nut, {grade} {material}, {finish}, ASME B18.2.2'
            rows.append(row("NUI", "nut", desc, d, tpi, "N/A", grade, material, finish,
                             "Hex", "Hex (Wrench)", "ASME B18.2.2", "imperial",
                             price(0.04, d, 0, material, finish)))
    return rows


def gen_nuts_metric() -> list[dict]:
    rows = []
    diameters = ['M2.5', 'M3', 'M4', 'M5', 'M6', 'M8', 'M10', 'M12', 'M16', 'M20', 'M24']
    combos = [
        ("Class 8.8", "Steel", "Zinc"),
        ("Class 8.8", "Steel", "Hot-Dip Galvanized"),
        ("Class 10.9", "Alloy Steel", "Plain"),
        ("A2-70", "Stainless Steel", "Plain"),
        ("A4-70", "Stainless Steel", "Plain"),
        ("18-8", "Stainless Steel", "Plain"),
    ]
    for d in diameters:
        pitch = METRIC_PITCH[d]
        for grade, material, finish in combos:
            desc = f'{d}x{pitch} Hex Nut, {grade} {material}, {finish}, DIN 934'
            rows.append(row("NUM", "nut", desc, d, pitch, "N/A", grade, material, finish,
                             "Hex", "Hex (Wrench)", "DIN 934", "metric",
                             price(0.03, d, 0, material, finish)))
    return rows


# ---------------------------------------------------------------------------
# Washers (flat and lock; length is not a meaningful spec -> "N/A")
# ---------------------------------------------------------------------------

def gen_washers_flat_imperial() -> list[dict]:
    rows = []
    diameters = ['#4', '#6', '#8', '#10', '#12', '1/4"', '5/16"', '3/8"', '7/16"',
                  '1/2"', '5/8"', '3/4"', '7/8"', '1"', '1-1/4"']
    combos = [
        ("Commercial", "Steel", "Zinc"),
        ("Commercial", "Steel", "Hot-Dip Galvanized"),
        ("18-8", "Stainless Steel", "Plain"),
    ]
    for d in diameters:
        tpi = UNC_TPI[d]
        for grade, material, finish in combos:
            desc = f'{disp(d)} Flat Washer, {material}, {finish}, ASME B18.21.1'
            rows.append(row("WFI", "washer", desc, d, tpi, "N/A", grade, material, finish,
                             "Flat", "None", "ASME B18.21.1", "imperial",
                             price(0.02, d, 0, material, finish)))
    return rows


def gen_washers_flat_metric() -> list[dict]:
    rows = []
    diameters = ['M2.5', 'M3', 'M4', 'M5', 'M6', 'M8', 'M10', 'M12', 'M16', 'M20', 'M24']
    combos = [
        ("Commercial", "Steel", "Zinc"),
        ("Commercial", "Steel", "Hot-Dip Galvanized"),
        ("18-8", "Stainless Steel", "Plain"),
    ]
    for d in diameters:
        pitch = METRIC_PITCH[d]
        for grade, material, finish in combos:
            desc = f'{d} Flat Washer, {material}, {finish}, DIN 125'
            rows.append(row("WFM", "washer", desc, d, pitch, "N/A", grade, material, finish,
                             "Flat", "None", "DIN 125", "metric",
                             price(0.015, d, 0, material, finish)))
    return rows


def gen_washers_lock_imperial() -> list[dict]:
    rows = []
    diameters = ['#4', '#6', '#8', '#10', '#12', '1/4"', '5/16"', '3/8"', '7/16"', '1/2"', '5/8"', '3/4"']
    combos = [
        ("Commercial", "Steel", "Zinc"),
        ("18-8", "Stainless Steel", "Plain"),
    ]
    for d in diameters:
        tpi = UNC_TPI[d]
        for grade, material, finish in combos:
            desc = f'{disp(d)} Split Lock Washer, {material}, {finish}, ASME B18.21.1'
            rows.append(row("WLI", "washer", desc, d, tpi, "N/A", grade, material, finish,
                             "Lock", "None", "ASME B18.21.1", "imperial",
                             price(0.015, d, 0, material, finish)))
    return rows


def gen_washers_lock_metric() -> list[dict]:
    rows = []
    diameters = ['M3', 'M4', 'M5', 'M6', 'M8', 'M10', 'M12', 'M16', 'M20']
    combos = [
        ("Commercial", "Steel", "Zinc"),
        ("18-8", "Stainless Steel", "Plain"),
    ]
    for d in diameters:
        pitch = METRIC_PITCH[d]
        for grade, material, finish in combos:
            desc = f'{d} Split Lock Washer, {material}, {finish}, DIN 127'
            rows.append(row("WLM", "washer", desc, d, pitch, "N/A", grade, material, finish,
                             "Lock", "None", "DIN 127", "metric",
                             price(0.012, d, 0, material, finish)))
    return rows


# ---------------------------------------------------------------------------
# Anchors (wedge anchors)
# ---------------------------------------------------------------------------

def gen_anchors_imperial() -> list[dict]:
    rows = []
    diameters = ['1/4"', '3/8"', '1/2"', '5/8"', '3/4"']
    lengths = ['2-1/4"', '3"', '3-3/4"', '4-1/4"', '5"', '5-1/2"', '7"', '8-1/2"']
    combos = [
        ("Commercial", "Steel", "Zinc"),
        ("18-8", "Stainless Steel", "Plain"),
    ]
    for d in diameters:
        tpi = UNC_TPI[d]
        for li, length in enumerate(lengths):
            for grade, material, finish in combos:
                desc = f'{disp(d)}-{tpi} x {length} Wedge Anchor, {material}, {finish}, ANSI'
                rows.append(row("ANI", "anchor", desc, d, tpi, length, grade, material, finish,
                                 "Stud", "Hex (Wrench)", "ANSI", "imperial",
                                 price(0.25, d, li, material, finish)))
    return rows


def gen_anchors_metric() -> list[dict]:
    rows = []
    diameters = ['M8', 'M10', 'M12']
    lengths = ["65", "75", "90", "100", "130", "160"]
    combos = [
        ("Commercial", "Steel", "Zinc"),
        ("A2-70", "Stainless Steel", "Plain"),
    ]
    for d in diameters:
        pitch = METRIC_PITCH[d]
        for li, length in enumerate(lengths):
            for grade, material, finish in combos:
                desc = f'{d}x{pitch} x {length}mm Wedge Anchor, {material}, {finish}, ANSI'
                rows.append(row("ANM", "anchor", desc, d, pitch, length, grade, material, finish,
                                 "Stud", "Hex (Wrench)", "ANSI", "metric",
                                 price(0.20, d, li, material, finish)))
    return rows


def main() -> None:
    generators = [
        gen_cap_screws_imperial, gen_cap_screws_metric,
        gen_hex_bolts_imperial, gen_hex_bolts_metric,
        gen_machine_screws_imperial, gen_machine_screws_metric,
        gen_nuts_imperial, gen_nuts_metric,
        gen_washers_flat_imperial, gen_washers_flat_metric,
        gen_washers_lock_imperial, gen_washers_lock_metric,
        gen_anchors_imperial, gen_anchors_metric,
    ]

    all_rows = []
    print("Catalog generation breakdown:")
    for gen in generators:
        rows = gen()
        all_rows.extend(rows)
        print(f"  {gen.__name__:28s} {len(rows):5d}")

    print(f"  {'TOTAL':28s} {len(all_rows):5d}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
