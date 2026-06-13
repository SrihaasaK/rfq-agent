"""CLI: RFQ text in, matched lines + confidence out.

Usage:
    python -m app.cli samples/email_01_clear_coupling.txt
    python -m app.cli samples/email_01_clear_coupling.txt --json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from app.db.models import AuditLogEntry, LineItem, RFQ
from app.db.session import get_session
from app.llm import LLMClient
from app.match.hybrid import HybridMatcher, confidence_bucket
from app.normalize.units import normalize_line_item

logger = logging.getLogger(__name__)


def process_rfq_text(rfq_text: str, source: str = "cli", llm: LLMClient | None = None,
                      matcher: HybridMatcher | None = None) -> dict:
    """Run extract -> normalize -> match -> route for a raw RFQ text blob.

    Persists the RFQ, line items, and audit log entries. Returns a summary dict.
    """
    llm = llm or LLMClient()
    matcher = matcher or HybridMatcher(llm=llm)

    with get_session() as session:
        rfq = RFQ(source=source, raw_text=rfq_text)
        session.add(rfq)
        session.flush()  # assign rfq.id

        session.add(AuditLogEntry(event_type="ingest", rfq_id=rfq.id, payload={"source": source}))

        raw_line_items = llm.extract_line_items(rfq_text)
        session.add(AuditLogEntry(
            event_type="extract", rfq_id=rfq.id,
            payload={"line_item_count": len(raw_line_items), "line_items": raw_line_items},
        ))

        results = []
        for i, raw_li in enumerate(raw_line_items, start=1):
            normalized = normalize_line_item(raw_li)

            match_result = matcher.match(normalized)
            bucket = confidence_bucket(match_result["confidence"])

            line_item = LineItem(
                rfq_id=rfq.id,
                line_number=i,
                raw_text=normalized.get("raw_text", ""),
                quantity=normalized.get("quantity"),
                unit=normalized.get("unit"),
                description=normalized.get("description", ""),
                attributes=normalized.get("attributes", {}),
                target_date=normalized.get("target_date"),
                notes=normalized.get("notes"),
                matched_sku=match_result["sku"],
                match_confidence=match_result["confidence"],
                match_alternatives=match_result["alternatives"],
                match_reasoning=match_result["reasoning"],
                confidence_bucket=bucket,
                unit_price=match_result.get("price_usd"),
            )
            session.add(line_item)
            session.flush()

            session.add(AuditLogEntry(
                event_type="match", rfq_id=rfq.id, line_item_id=line_item.id,
                payload={"match": match_result, "confidence_bucket": bucket},
            ))

            results.append({
                "line_number": i,
                "raw_text": normalized.get("raw_text", ""),
                "quantity": normalized.get("quantity"),
                "unit": normalized.get("unit"),
                "description": normalized.get("description", ""),
                "attributes": normalized.get("attributes", {}),
                "match": match_result,
                "confidence_bucket": bucket,
            })

        return {"rfq_id": rfq.id, "line_items": results}


def print_results(report: dict) -> None:
    print(f"\nRFQ #{report['rfq_id']} — {len(report['line_items'])} line item(s)")
    print("=" * 60)
    for li in report["line_items"]:
        bucket = li["confidence_bucket"]
        match = li["match"]
        print(f"\nLine {li['line_number']}: {li['raw_text']!r}")
        print(f"  qty={li['quantity']} unit={li['unit']} desc={li['description']!r}")
        print(f"  attributes: {li['attributes']}")
        if match["sku"]:
            print(f"  [{bucket}] -> {match['sku']} (confidence={match['confidence']:.2f}, price=${match.get('price_usd')})")
            print(f"  reasoning: {match['reasoning']}")
            if match["alternatives"]:
                print("  alternatives:")
                for alt in match["alternatives"]:
                    print(f"    - {alt['sku']} ({alt['confidence']:.2f}): {alt['description']}")
        else:
            print(f"  [{bucket}] ABSTAIN — {match['reasoning']}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Process an RFQ: extract, normalize, match against catalog.")
    parser.add_argument("rfq_file", type=Path, help="Path to a plain-text RFQ file")
    parser.add_argument("--json", action="store_true", help="Print raw JSON instead of formatted output")
    args = parser.parse_args()

    rfq_text = args.rfq_file.read_text()
    report = process_rfq_text(rfq_text, source=str(args.rfq_file))

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print_results(report)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
