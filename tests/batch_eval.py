"""Batch evaluation script for the Clinical NLP pipeline.

Reads test cases from a CSV, runs each through the pipeline, writes a detailed
results CSV, and prints a summary metrics table.

Usage (run from project root):
    python tests/batch_eval.py
    python tests/batch_eval.py --input tests/batch_test_cases.csv
    python tests/batch_eval.py --input tests/batch_test_cases.csv --output results/
    python tests/batch_eval.py --concurrency 1   # slower but easier to debug

CSV columns (all optional except id, input):
    id, category, description, input,
    expected_snomed_codes     pipe-separated codes that MUST appear in output
    expected_snomed_absent    pipe-separated codes that must NOT appear (negation)
    expected_city             exact or fuzzy city match
    expected_state            must appear anywhere in state.values list
    expected_phase            fuzzy match
    expected_investigator_name fuzzy match
    expected_site_name        fuzzy match
"""

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from rapidfuzz import fuzz

from src.pipeline import NLPPipeline
from src.assembler import NLPOutput
from src.preprocessor import PreprocessorError
from src.extractor import ExtractionError

FUZZY_THRESHOLD = 88


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _parse_codes(raw: str) -> list[str]:
    """Parse a pipe-separated string of SNOMED codes into a list."""
    if not raw or not raw.strip():
        return []
    return [c.strip() for c in raw.split("|") if c.strip()]


def _fuzzy_match(actual: Optional[str], expected: str) -> bool:
    if not actual:
        return False
    return fuzz.token_sort_ratio(str(actual).lower(), expected.lower()) >= FUZZY_THRESHOLD


def _state_match(state_values: list[str], expected: str) -> bool:
    if not state_values:
        return False
    return any(
        fuzz.token_sort_ratio(s.lower(), expected.lower()) >= FUZZY_THRESHOLD
        for s in state_values
    )


# ── Single test runner ────────────────────────────────────────────────────────

def run_one(pipeline: NLPPipeline, row: dict) -> dict:
    """Run a single test case and return a result dict."""
    tc_id = row.get("id", "?")
    query = row.get("input", "").strip()

    result_row = {
        "id": tc_id,
        "category": row.get("category", ""),
        "description": row.get("description", ""),
        "input": query,
        "status": "",
        "error": "",
        "processing_time_ms": "",
        # SNOMED
        "snomed_codes_found": "",
        "snomed_codes_expected": row.get("expected_snomed_codes", ""),
        "snomed_codes_absent": row.get("expected_snomed_absent", ""),
        "snomed_precision": "",
        "snomed_recall": "",
        "snomed_absent_violated": "",
        # Filters
        "city_found": "",
        "city_expected": row.get("expected_city", ""),
        "city_pass": "",
        "state_found": "",
        "state_expected": row.get("expected_state", ""),
        "state_pass": "",
        "phase_found": "",
        "phase_expected": row.get("expected_phase", ""),
        "phase_pass": "",
        "investigator_found": "",
        "investigator_expected": row.get("expected_investigator_name", ""),
        "investigator_pass": "",
        "site_found": "",
        "site_expected": row.get("expected_site_name", ""),
        "site_pass": "",
        "failures": "",
    }

    try:
        output: NLPOutput = pipeline.run(query)
    except (PreprocessorError, ExtractionError) as exc:
        result_row["status"] = "ERROR"
        result_row["error"] = str(exc)
        return result_row
    except Exception as exc:
        result_row["status"] = "ERROR"
        result_row["error"] = f"Unexpected: {exc}"
        return result_row

    # ── Populate found values ─────────────────────────────────────────────
    found_codes = {t.code for t in output.snomed_terms}
    result_row["snomed_codes_found"] = "|".join(sorted(found_codes))
    result_row["processing_time_ms"] = output.metadata.processing_time_ms
    result_row["city_found"] = output.filters.city.value or ""
    result_row["state_found"] = "|".join(output.filters.state.values)
    result_row["phase_found"] = output.filters.phase.value or ""
    result_row["investigator_found"] = output.filters.investigator_name.value or ""
    result_row["site_found"] = output.filters.site_name.value or ""

    failures = []

    # ── SNOMED present check ──────────────────────────────────────────────
    expected_codes = _parse_codes(row.get("expected_snomed_codes", ""))
    absent_codes = _parse_codes(row.get("expected_snomed_absent", ""))

    if expected_codes:
        matched = [c for c in expected_codes if c in found_codes]
        precision = len(matched) / len(found_codes) if found_codes else 0.0
        recall = len(matched) / len(expected_codes)
        result_row["snomed_precision"] = f"{precision:.2f}"
        result_row["snomed_recall"] = f"{recall:.2f}"
        for code in expected_codes:
            if code not in found_codes:
                failures.append(f"SNOMED {code} missing")
    else:
        result_row["snomed_precision"] = "N/A"
        result_row["snomed_recall"] = "N/A"

    # ── SNOMED absent check ───────────────────────────────────────────────
    violated_absent = [c for c in absent_codes if c in found_codes]
    result_row["snomed_absent_violated"] = "|".join(violated_absent)
    for code in violated_absent:
        failures.append(f"Negated SNOMED {code} present in output")

    # ── Filter checks ─────────────────────────────────────────────────────
    def _check_filter(field: str, actual: Optional[str], expected: str) -> bool:
        if not expected:
            return True
        passed = _fuzzy_match(actual, expected)
        result_row[f"{field}_pass"] = "PASS" if passed else "FAIL"
        if not passed:
            failures.append(f"{field}: expected '{expected}', got '{actual or 'None'}'")
        return passed

    def _check_state(expected: str) -> bool:
        if not expected:
            return True
        passed = _state_match(output.filters.state.values, expected)
        result_row["state_pass"] = "PASS" if passed else "FAIL"
        if not passed:
            failures.append(
                f"state: expected '{expected}', got {output.filters.state.values or 'None'}"
            )
        return passed

    _check_filter("city",         output.filters.city.value,              row.get("expected_city", ""))
    _check_state(                                                          row.get("expected_state", ""))
    _check_filter("phase",        output.filters.phase.value,             row.get("expected_phase", ""))
    _check_filter("investigator", output.filters.investigator_name.value, row.get("expected_investigator_name", ""))
    _check_filter("site",         output.filters.site_name.value,         row.get("expected_site_name", ""))

    result_row["failures"] = "; ".join(failures) if failures else ""
    result_row["status"] = "PASS" if not failures else "FAIL"
    return result_row


# ── Metrics summary ───────────────────────────────────────────────────────────

def print_summary(results: list[dict]) -> None:
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    errored = sum(1 for r in results if r["status"] == "ERROR")

    times = [int(r["processing_time_ms"]) for r in results if str(r["processing_time_ms"]).isdigit()]
    avg_ms = int(sum(times) / len(times)) if times else 0
    min_ms = min(times) if times else 0
    max_ms = max(times) if times else 0

    # Filter-level accuracy
    filter_fields = ["city", "state", "phase", "investigator", "site"]
    filter_stats: dict[str, dict] = {}
    for field in filter_fields:
        key = f"{field}_pass"
        expected_key = f"{field}_expected"
        tested = [r for r in results if r.get(expected_key, "").strip()]
        passes = sum(1 for r in tested if r.get(key) == "PASS")
        filter_stats[field] = {"tested": len(tested), "passed": passes}

    # SNOMED recall (average across cases with expected codes)
    recall_vals = []
    for r in results:
        v = r.get("snomed_recall", "")
        if v not in ("", "N/A"):
            try:
                recall_vals.append(float(v))
            except ValueError:
                pass
    avg_recall = sum(recall_vals) / len(recall_vals) if recall_vals else None

    # Per-category breakdown
    by_cat: dict[str, dict] = defaultdict(lambda: {"total": 0, "pass": 0})
    for r in results:
        cat = r.get("category", "uncategorized") or "uncategorized"
        by_cat[cat]["total"] += 1
        if r["status"] == "PASS":
            by_cat[cat]["pass"] += 1

    W = 60
    print()
    print("=" * W)
    print(" BATCH EVALUATION SUMMARY")
    print("=" * W)
    print(f"  Total test cases : {total}")
    print(f"  Passed           : {passed}  ({passed/total*100:.1f}%)")
    print(f"  Failed           : {failed}  ({failed/total*100:.1f}%)")
    print(f"  Errors           : {errored}  ({errored/total*100:.1f}%)")
    print()
    print(f"  Processing time  : avg {avg_ms}ms  |  min {min_ms}ms  |  max {max_ms}ms")
    if avg_recall is not None:
        print(f"  SNOMED recall    : {avg_recall:.2f} (avg over {len(recall_vals)} cases with expected codes)")
    print()
    print("  Filter accuracy (cases with an expected value):")
    for field, s in filter_stats.items():
        if s["tested"] > 0:
            pct = s["passed"] / s["tested"] * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"    {field:<18} {bar}  {s['passed']}/{s['tested']} ({pct:.0f}%)")
    print()
    print("  Results by category:")
    for cat, s in sorted(by_cat.items()):
        pct = s["pass"] / s["total"] * 100 if s["total"] else 0
        print(f"    {cat:<30} {s['pass']:>3}/{s['total']:<3}  ({pct:.0f}%)")
    print("=" * W)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Batch evaluate the Clinical NLP pipeline.")
    parser.add_argument("--input",  default="tests/batch_test_cases.csv",
                        help="Path to input CSV (default: tests/batch_test_cases.csv)")
    parser.add_argument("--output", default="tests/results",
                        help="Directory to write results CSV (default: tests/results/)")
    parser.add_argument("--limit",  type=int, default=None,
                        help="Only run the first N cases (useful for quick smoke-testing)")
    args = parser.parse_args()

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        print("ERROR: GROQ_API_KEY not set. Add it to .env or environment.")
        return 1

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return 1

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"batch_results_{timestamp}.csv"

    print(f"Loading pipeline (first run may take 30-60s)...")
    pipeline = NLPPipeline(groq_api_key=api_key)
    print("Pipeline ready.\n")

    with open(input_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    if args.limit:
        rows = rows[: args.limit]

    total = len(rows)
    results = []

    OUTPUT_COLS = [
        "id", "category", "description", "input", "status", "error",
        "processing_time_ms",
        "snomed_codes_found", "snomed_codes_expected", "snomed_precision", "snomed_recall",
        "snomed_codes_absent", "snomed_absent_violated",
        "city_found", "city_expected", "city_pass",
        "state_found", "state_expected", "state_pass",
        "phase_found", "phase_expected", "phase_pass",
        "investigator_found", "investigator_expected", "investigator_pass",
        "site_found", "site_expected", "site_pass",
        "failures",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as out_fh:
        writer = csv.DictWriter(out_fh, fieldnames=OUTPUT_COLS, extrasaction="ignore")
        writer.writeheader()

        for i, row in enumerate(rows, 1):
            tc_id = row.get("id", f"#{i}")
            print(f"[{i:>3}/{total}] {tc_id} ...", end=" ", flush=True)

            result = run_one(pipeline, row)
            results.append(result)
            writer.writerow(result)
            out_fh.flush()  # write incrementally so partial results survive interruption

            status = result["status"]
            if status == "PASS":
                print("PASS")
            elif status == "ERROR":
                print(f"ERROR — {result['error'][:60]}")
            else:
                short = result["failures"][:70]
                print(f"FAIL — {short}")

    print_summary(results)
    print(f"\nDetailed results saved to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
