"""Batch evaluation script for the Clinical NLP pipeline.

Reads test cases from a CSV, runs each through the pipeline, writes a detailed
results CSV, and prints a summary metrics table.

Usage (run from project root):
    python tests/batch_eval.py
    python tests/batch_eval.py --input tests/batch_test_cases.csv --output results/
    python tests/batch_eval.py --concurrency 1   # slower but easier to debug
    python tests/batch_eval.py --legacy           # use legacy pipeline.run() single-turn path
    python tests/batch_eval.py --strategy hybrid_cascade  # set SNOMED strategy per run

CSV columns (all optional except id, input):
    id, category, description, input,
    expected_snomed_codes       pipe-separated codes that MUST appear in output
    expected_snomed_absent      pipe-separated codes that must NOT appear (negation)
    expected_city               exact or fuzzy city match
    expected_state              must appear anywhere in state.values list
    expected_phase              fuzzy match
    expected_investigator_name  fuzzy match
    expected_site_name          fuzzy match
    expected_type               "search" or "clarification" (default: "search")
    expected_clarification_field  trigger name expected to fire (e.g. "cancer")
    expected_options_contain    pipe-separated substrings expected in options

Multi-turn syntax: use ">>>" in the input column to separate turns.
    e.g. "cancer>>>Lung Cancer" → turn 1: "cancer", turn 2: "Lung Cancer"
    In --legacy mode only the first segment is used.
"""

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from rapidfuzz import fuzz

from src.pipeline import NLPPipeline
from src.assembler import NLPOutput, ClarificationOutput
from src.conversation import ConversationSession
from src.preprocessor import PreprocessorError
from src.exceptions import ExtractionError

FUZZY_THRESHOLD = 88


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _parse_codes(raw: str) -> list[str]:
    """Parse a pipe-separated string of SNOMED codes into a list."""
    if not raw or not raw.strip():
        return []
    return [c.strip() for c in raw.split("|") if c.strip()]


def _parse_options_contain(raw: str) -> list[str]:
    """Parse a pipe-separated string of expected option substrings."""
    if not raw or not raw.strip():
        return []
    return [s.strip() for s in raw.split("|") if s.strip()]


def _parse_turns(raw_input: str) -> list[str]:
    """Split a multi-turn input string on '>>>' separator."""
    return [seg.strip() for seg in raw_input.split(">>>") if seg.strip()]


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

def run_one(
    pipeline: NLPPipeline,
    row: dict,
    legacy: bool = False,
) -> dict:
    """Run a single test case and return a result dict.

    Supports multi-turn '>>>' syntax and the new expected_type /
    expected_clarification_field / expected_options_contain columns.
    In --legacy mode, always calls pipeline.run() with the first turn only.
    """
    tc_id = row.get("id", "?")
    raw_input = row.get("input", "").strip()

    expected_type = (row.get("expected_type") or "search").strip().lower()
    expected_clarification_field = (row.get("expected_clarification_field") or "").strip()
    expected_options_contain = _parse_options_contain(
        row.get("expected_options_contain") or ""
    )

    result_row = {
        "id": tc_id,
        "category": row.get("category", ""),
        "description": row.get("description", ""),
        "input": raw_input,
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
        # New v2 type/clarification checks
        "output_type": "",
        "expected_type": expected_type,
        "type_pass": "",
        "clarification_field_pass": "",
        "options_contain_pass": "",
        "failures": "",
    }

    turns = _parse_turns(raw_input)
    if not turns:
        result_row["status"] = "ERROR"
        result_row["error"] = "Empty input"
        return result_row

    # ── Legacy mode: single-turn via pipeline.run() ───────────────────────────
    if legacy:
        query = turns[0]
        result_row["input"] = query
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
        return _evaluate_nlp_output(output, result_row, row)

    # ── Multi-turn mode via run_with_session() ────────────────────────────────
    session = ConversationSession.new()
    final_result: Union[NLPOutput, ClarificationOutput, None] = None

    try:
        for turn_text in turns:
            final_result = pipeline.run_with_session(turn_text, session)
    except (PreprocessorError, ExtractionError) as exc:
        result_row["status"] = "ERROR"
        result_row["error"] = str(exc)
        return result_row
    except Exception as exc:
        result_row["status"] = "ERROR"
        result_row["error"] = f"Unexpected: {exc}"
        return result_row

    if final_result is None:
        result_row["status"] = "ERROR"
        result_row["error"] = "No output produced"
        return result_row

    # ── Evaluate output type ──────────────────────────────────────────────────
    failures = []

    if isinstance(final_result, ClarificationOutput):
        result_row["output_type"] = "clarification"
        result_row["processing_time_ms"] = final_result.metadata.processing_time_ms

        if expected_type == "clarification":
            result_row["type_pass"] = "PASS"
        else:
            result_row["type_pass"] = "FAIL"
            failures.append(
                f"expected type 'search', got 'clarification'"
            )

        # Check clarification field (triggered_by in the session's last turn)
        if expected_clarification_field:
            last_turn = session.turns[-1] if session.turns else None
            triggered_by = (
                last_turn.decision.triggered_by
                if last_turn and last_turn.decision
                else None
            ) or ""
            if expected_clarification_field.lower() in triggered_by.lower():
                result_row["clarification_field_pass"] = "PASS"
            else:
                result_row["clarification_field_pass"] = "FAIL"
                failures.append(
                    f"clarification_field: expected '{expected_clarification_field}', "
                    f"got '{triggered_by}'"
                )

        # Check options contain
        if expected_options_contain:
            options_text = " | ".join(final_result.options)
            missing = [
                s for s in expected_options_contain
                if s.lower() not in options_text.lower()
            ]
            if missing:
                result_row["options_contain_pass"] = "FAIL"
                failures.append(
                    f"options missing substrings: {missing}"
                )
            else:
                result_row["options_contain_pass"] = "PASS"

        result_row["failures"] = "; ".join(failures) if failures else ""
        result_row["status"] = "PASS" if not failures else "FAIL"
        return result_row

    else:
        # NLPOutput
        result_row["output_type"] = "search"
        if expected_type == "search":
            result_row["type_pass"] = "PASS"
        else:
            result_row["type_pass"] = "FAIL"
            failures.append("expected type 'clarification', got 'search'")

        return _evaluate_nlp_output(final_result, result_row, row, pre_failures=failures)


def _evaluate_nlp_output(
    output: NLPOutput,
    result_row: dict,
    row: dict,
    pre_failures: Optional[list] = None,
) -> dict:
    """Populate result_row with NLPOutput evaluation against expected CSV columns."""
    failures = list(pre_failures or [])

    found_codes = {t.code for t in output.snomed_terms}
    result_row["snomed_codes_found"] = "|".join(sorted(found_codes))
    result_row["processing_time_ms"] = output.metadata.processing_time_ms
    result_row["city_found"] = output.filters.city.value or ""
    result_row["state_found"] = "|".join(output.filters.state.values)
    result_row["phase_found"] = "|".join(output.filters.phase.values) if output.filters.phase.values else ""
    result_row["investigator_found"] = output.filters.investigator_name.value or ""
    result_row["site_found"] = output.filters.site_name.value or ""

    # ── SNOMED present check ──────────────────────────────────────────────────
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

    # ── SNOMED absent check ───────────────────────────────────────────────────
    violated_absent = [c for c in absent_codes if c in found_codes]
    result_row["snomed_absent_violated"] = "|".join(violated_absent)
    for code in violated_absent:
        failures.append(f"Negated SNOMED {code} present in output")

    # ── Filter checks ─────────────────────────────────────────────────────────
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
    _check_filter("phase",        "|".join(output.filters.phase.values) if output.filters.phase.values else "", row.get("expected_phase", ""))
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

    # Clarification precision
    clarif_expected = [r for r in results if r.get("expected_type", "search").strip().lower() == "clarification"]
    clarif_correct = [
        r for r in clarif_expected
        if r.get("output_type", "") == "clarification"
    ]
    clarif_precision = (
        len(clarif_correct) / len(clarif_expected)
        if clarif_expected
        else None
    )

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
    if clarif_precision is not None:
        pct = clarif_precision * 100
        print(
            f"  Clarification precision : {pct:.1f}%  "
            f"({len(clarif_correct)}/{len(clarif_expected)} expected clarification cases returned ClarificationOutput)"
        )
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
    parser.add_argument("--legacy", action="store_true",
                        help="Use legacy pipeline.run() single-turn path instead of run_with_session()")
    parser.add_argument("--strategy", default=None,
                        help="Set SNOMED_SEARCH_STRATEGY env var before pipeline init "
                             "(e.g. hybrid_cascade, aho_corasick, ngram_lookup)")
    args = parser.parse_args()

    # --strategy sets the env var BEFORE pipeline init so the registry picks it up
    if args.strategy:
        os.environ["SNOMED_SEARCH_STRATEGY"] = args.strategy

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return 1

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    strategy_tag = f"_{args.strategy}" if args.strategy else ""
    legacy_tag = "_legacy" if args.legacy else ""
    output_path = output_dir / f"batch_results{strategy_tag}{legacy_tag}_{timestamp}.csv"

    mode_desc = "legacy pipeline.run()" if args.legacy else "run_with_session()"
    strategy_desc = f" | strategy={args.strategy}" if args.strategy else ""
    print(f"Loading pipeline (first run may take 30-60s)... [{mode_desc}{strategy_desc}]")
    pipeline = NLPPipeline()
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
        # v2 type/clarification columns
        "output_type", "expected_type", "type_pass",
        "clarification_field_pass", "options_contain_pass",
        "failures",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as out_fh:
        writer = csv.DictWriter(out_fh, fieldnames=OUTPUT_COLS, extrasaction="ignore")
        writer.writeheader()

        for i, row in enumerate(rows, 1):
            tc_id = row.get("id", f"#{i}")
            print(f"[{i:>3}/{total}] {tc_id} ...", end=" ", flush=True)

            result = run_one(pipeline, row, legacy=args.legacy)
            results.append(result)
            writer.writerow(result)
            out_fh.flush()

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
