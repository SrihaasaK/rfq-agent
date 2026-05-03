"""Eval harness for the RFQ agent — route accuracy, SKU match, question quality."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

from agent.agent import process_rfq
from agent.models import RFQResult, Route

logger = logging.getLogger(__name__)

SAMPLES_DIR = Path(__file__).parent.parent / "samples"
EVALS_DIR = Path(__file__).parent
TEST_CASES_PATH = EVALS_DIR / "test_cases.json"
REPORT_PATH = EVALS_DIR / "latest_report.json"


def _load_test_cases() -> list[dict]:
    with open(TEST_CASES_PATH) as f:
        return json.load(f)


def _get_email_text(tc: dict) -> str:
    if tc.get("email_text"):
        return tc["email_text"]
    if tc.get("email_file"):
        return (SAMPLES_DIR / tc["email_file"]).read_text()
    raise ValueError(f"Test case {tc['id']} has no email_text or email_file")


def _score_question_quality(result: RFQResult, disambiguating_specs: list[str]) -> dict:
    """Score whether the clarifying question addresses the right specs."""
    if not result.clarification or not disambiguating_specs:
        return {"score": 0.0, "addressed_specs": [], "missed_specs": disambiguating_specs}

    question_lower = result.clarification.question.lower()
    addressed = []
    missed = []

    spec_keywords = {
        "thread_standard": ["npt", "bsp", "compression", "sweat", "thread"],
        "material_grade": ["c36000", "c46400", "c84400", "material", "grade", "naval", "lead-free", "food-safe"],
        "size": ["1/2", "3/4", "1\"", "half inch", "three quarter", "size"],
        "pressure_rating_psi": ["psi", "pressure", "rating"],
    }

    for spec in disambiguating_specs:
        keywords = spec_keywords.get(spec, [spec])
        if any(kw in question_lower for kw in keywords):
            addressed.append(spec)
        else:
            missed.append(spec)

    score = len(addressed) / len(disambiguating_specs) if disambiguating_specs else 0.0
    return {"score": score, "addressed_specs": addressed, "missed_specs": missed}


def run_eval(test_cases: list[dict] | None = None, progress_callback=None) -> dict:
    """Run the full eval suite. Returns the report dict."""
    if test_cases is None:
        test_cases = _load_test_cases()

    results = []
    total = len(test_cases)
    route_correct = 0
    sku_correct = 0
    sku_total = 0
    question_scores = []
    human_correct = 0
    human_total = 0

    for i, tc in enumerate(test_cases):
        tc_id = tc["id"]
        logger.info("Running test case %s (%d/%d)", tc_id, i + 1, total)

        if progress_callback:
            progress_callback(i, total, tc_id)

        email_text = _get_email_text(tc)
        start = time.time()

        try:
            result = process_rfq(email_text, tc.get("customer_name"))
            elapsed = round(time.time() - start, 2)
        except Exception as e:
            logger.error("Test case %s failed: %s", tc_id, e)
            results.append({
                "id": tc_id,
                "status": "error",
                "error": str(e),
                "elapsed_seconds": round(time.time() - start, 2),
            })
            continue

        expected_route = tc["expected_route"]
        actual_route = result.route.value
        route_match = actual_route == expected_route
        if route_match:
            route_correct += 1

        tc_result = {
            "id": tc_id,
            "status": "ok",
            "elapsed_seconds": elapsed,
            "expected_route": expected_route,
            "actual_route": actual_route,
            "route_correct": route_match,
            "route_reasoning": result.route_reasoning,
            "hypotheses": [h.model_dump() for h in result.hypotheses],
        }

        # SKU match for quote cases
        if expected_route == "quote" and tc.get("expected_sku"):
            sku_total += 1
            actual_sku = result.quote.sku if result.quote else None
            sku_match = actual_sku == tc["expected_sku"]
            if sku_match:
                sku_correct += 1
            tc_result["expected_sku"] = tc["expected_sku"]
            tc_result["actual_sku"] = actual_sku
            tc_result["sku_correct"] = sku_match

        # Question quality for clarify cases
        if expected_route == "clarify" and tc.get("disambiguating_specs"):
            q_score = _score_question_quality(result, tc["disambiguating_specs"])
            question_scores.append(q_score["score"])
            tc_result["question_quality"] = q_score

        # Human escalation check
        if expected_route == "human":
            human_total += 1
            if actual_route == "human":
                human_correct += 1

        # Failure analysis for misses
        if not route_match:
            tc_result["failure_analysis"] = {
                "expected": expected_route,
                "got": actual_route,
                "top_confidence": result.hypotheses[0].confidence if result.hypotheses else 0.0,
                "hypothesis_count": len(result.hypotheses),
            }

        results.append(tc_result)

    # Build summary
    report = {
        "summary": {
            "total_cases": total,
            "route_accuracy": round(route_correct / total, 3) if total else 0,
            "route_correct": route_correct,
            "sku_match_rate": round(sku_correct / sku_total, 3) if sku_total else 0,
            "sku_correct": sku_correct,
            "sku_total": sku_total,
            "question_quality_avg": round(sum(question_scores) / len(question_scores), 3) if question_scores else 0,
            "question_quality_scores": question_scores,
            "human_escalation_rate": round(human_correct / human_total, 3) if human_total else 0,
            "human_correct": human_correct,
            "human_total": human_total,
        },
        "results": results,
    }

    # Write report
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    return report


def print_report(report: dict) -> None:
    """Pretty-print the eval report to stdout."""
    s = report["summary"]
    print("\n" + "=" * 60)
    print("RFQ AGENT EVAL REPORT")
    print("=" * 60)
    print(f"Total cases:           {s['total_cases']}")
    print(f"Route accuracy:        {s['route_accuracy']:.1%} ({s['route_correct']}/{s['total_cases']})")
    print(f"SKU match rate:        {s['sku_match_rate']:.1%} ({s['sku_correct']}/{s['sku_total']})")
    print(f"Question quality avg:  {s['question_quality_avg']:.1%}")
    print(f"Human escalation rate: {s['human_escalation_rate']:.1%} ({s['human_correct']}/{s['human_total']})")
    print("=" * 60)

    failures = [r for r in report["results"] if r.get("status") == "error" or not r.get("route_correct", True)]
    if failures:
        print(f"\nFAILURES ({len(failures)}):")
        for f in failures:
            if f.get("status") == "error":
                print(f"  {f['id']}: ERROR — {f['error']}")
            else:
                fa = f.get("failure_analysis", {})
                print(f"  {f['id']}: expected={fa.get('expected')}, got={fa.get('got')}, top_conf={fa.get('top_confidence', 'N/A')}")
    else:
        print("\nAll cases passed!")

    print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    report = run_eval()
    print_report(report)
