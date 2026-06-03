"""Test runner for Clinical Research NLP pipeline.

Run from the project root directory:
    python tests/run_tests.py

Requires GROQ_API_KEY in .env or environment.
Exit code 0 if >= 15 tests pass, exit code 1 otherwise.
"""
# OBSOLETE-AT-SCALE: legacy single-turn regression suite — supplant with batch_eval.py multi-turn cases at cleanup

import json
import os
import sys
from pathlib import Path

# Must add project root to sys.path before any src imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from rapidfuzz import fuzz

from src.pipeline import NLPPipeline
from src.preprocessor import PreprocessorError
from src.exceptions import ExtractionError

PASS_THRESHOLD = 15
FILTER_FUZZY_THRESHOLD = 88


def run_tests() -> int:
    test_file = Path(__file__).parent / "test_cases.json"
    with open(test_file, "r", encoding="utf-8") as fh:
        test_cases = json.load(fh)

    print(f"Loading NLPPipeline (this may take 30-60s on first run)...")
    pipeline = NLPPipeline()
    print("Pipeline loaded.\n")

    passed = 0
    failed = 0

    for tc in test_cases:
        tc_id = tc["id"]
        description = tc["description"]
        query = tc["input"]
        expected = tc["expected"]

        try:
            result = pipeline.run(query)
        except (PreprocessorError, ExtractionError) as exc:
            print(f"FAIL [{tc_id}] {description}")
            print(f"     Pipeline error: {exc}\n")
            failed += 1
            continue
        except Exception as exc:
            print(f"FAIL [{tc_id}] {description}")
            print(f"     Unexpected error: {exc}\n")
            failed += 1
            continue

        result_codes = {t.code for t in result.snomed_terms}
        # state is now a list — keep scalar fields separate
        scalar_filters = {
            "investigator_name": result.filters.investigator_name.value,
            "site_name": result.filters.site_name.value,
            "city": result.filters.city.value,
            "phase": "|".join(result.filters.phase.values) if result.filters.phase.values else None,
        }
        result_state_values = result.filters.state.values  # list[str]

        failures = []

        # TC008 and TC020: negation test — codes must be absent
        if "snomed_codes_absent" in expected:
            for code in expected["snomed_codes_absent"]:
                if code in result_codes:
                    failures.append(f"Negated SNOMED code {code} found in output (should be excluded)")

        # Check required codes are present
        if "snomed_codes_present" in expected:
            for code in expected["snomed_codes_present"]:
                if code not in result_codes:
                    failures.append(f"Expected SNOMED code {code} not found in output")
        elif "snomed_codes" in expected:
            for code in expected["snomed_codes"]:
                if code not in result_codes:
                    failures.append(f"Expected SNOMED code {code} not found in output")

        # Check filter values using fuzzy matching
        for field, expected_value in expected.get("filters", {}).items():
            if expected_value is None:
                continue

            if field == "state":
                # State is a list — pass if any element fuzzy-matches the expected value
                if not result_state_values:
                    failures.append(f"Filter 'state': expected '{expected_value}', got empty list")
                    continue
                best_score = max(
                    fuzz.token_sort_ratio(s.lower(), str(expected_value).lower())
                    for s in result_state_values
                )
                if best_score < FILTER_FUZZY_THRESHOLD:
                    failures.append(
                        f"Filter 'state': expected '{expected_value}', "
                        f"got {result_state_values} (best score {best_score})"
                    )
            else:
                actual_value = scalar_filters.get(field)
                if actual_value is None:
                    failures.append(f"Filter '{field}': expected '{expected_value}', got None")
                    continue
                score = fuzz.token_sort_ratio(
                    str(actual_value).lower(), str(expected_value).lower()
                )
                if score < FILTER_FUZZY_THRESHOLD:
                    failures.append(
                        f"Filter '{field}': expected '{expected_value}', "
                        f"got '{actual_value}' (score {score})"
                    )

        if failures:
            print(f"FAIL [{tc_id}] {description}")
            for reason in failures:
                print(f"     - {reason}")
            print()
            failed += 1
        else:
            print(f"PASS [{tc_id}] {description}")
            passed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed}/{len(test_cases)} passed")
    print(f"{'='*60}")

    if passed >= PASS_THRESHOLD:
        print(f"SUCCESS: {passed} tests passed (threshold: {PASS_THRESHOLD})")
        return 0
    else:
        print(f"FAILURE: Only {passed} tests passed (threshold: {PASS_THRESHOLD})")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
