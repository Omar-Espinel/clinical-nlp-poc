"""
test_agent.py — Clinical NLP Test Agent
========================================
Runs test_cases_202_fixed.json against NLPPipeline and writes results.md

USAGE (from project root):
    python test_agent.py
    python test_agent.py --input tests/test_cases_202_fixed.json
    python test_agent.py --limit 20
    python test_agent.py --category injection
    python test_agent.py --output tests/results/

RATE LIMIT STRATEGY (Groq free tier: 30 req/min, 500 req/day):
    • Cases that never hit the API (injection, harmful, edge_case) run instantly
    • API cases are throttled to --rpm requests-per-minute (default 25, safe under 30 limit)
    • On RateLimitError: exponential backoff up to 60s, then retry
    • After MAX_RETRIES: case marked RATE_LIMITED (skipped, not failed)
    • Run order: no-API categories first, then API-needing ones
    • Use --limit N for smoke tests to conserve daily quota
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from rapidfuzz import fuzz

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# Support both .env and env_variable.env
for _env_file in [".env", "env_variable.env"]:
    if (ROOT / _env_file).exists():
        load_dotenv(ROOT / _env_file)
        break

logging.basicConfig(level=logging.WARNING)

from src.pipeline import NLPPipeline
from src.assembler import NLPOutput
from src.preprocessor import PreprocessorError
from src.extractor import ExtractionError

# ── Configuration ─────────────────────────────────────────────────────────────
FUZZY_THRESHOLD    = 88
DEFAULT_INPUT      = ROOT / "test_cases_202.json"
DEFAULT_OUTPUT_DIR = ROOT /  "results"
DEFAULT_RPM        = 25      # safe under Groq's 30 req/min free tier limit
MAX_RETRIES        = 3       # retries on rate limit before marking as RATE_LIMITED

# These categories are blocked by the preprocessor — Groq API is never called
NO_API_CATEGORIES = {"injection", "harmful", "edge_case"}


# ── Data classes ──────────────────────────────────────────────────────────────
@dataclass
class TestResult:
    tc_id:        str
    category:     str
    description:  str
    input_text:   str
    passed:       bool
    error:        Optional[str]  = None
    failures:     list           = field(default_factory=list)
    actual_codes: list           = field(default_factory=list)
    actual_city:  Optional[str]  = None
    actual_state: list           = field(default_factory=list)
    actual_phase: Optional[str]  = None
    expected:     dict           = field(default_factory=dict)
    latency_ms:   int            = 0
    was_rejected: bool           = False
    rate_limited: bool           = False


@dataclass
class RunSummary:
    total:       int  = 0
    passed:      int  = 0
    failed:      int  = 0
    skipped:     int  = 0
    errors:      int  = 0
    by_category: dict = field(default_factory=dict)
    total_ms:    int  = 0
    results:     list = field(default_factory=list)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _fuzzy(a: Optional[str], b: Optional[str]) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return fuzz.token_sort_ratio(a.lower().strip(), b.lower().strip()) >= FUZZY_THRESHOLD


def _state_match(expected: Optional[str], actual: list) -> bool:
    if not expected:
        return True
    return any(_fuzzy(expected, s) for s in actual)


def _codes_match(expected: list, actual: list) -> tuple:
    if not expected:
        return True, ""
    actual_deduped = list(dict.fromkeys(actual))   # handle P4 duplicate-code bug
    missing = [c for c in expected if c not in actual_deduped]
    if missing:
        return False, f"Missing codes: {missing}"
    return True, ""


# ── Rate limiter ──────────────────────────────────────────────────────────────
class RateLimiter:
    """Token bucket — keeps Groq API calls under rpm requests per minute."""

    def __init__(self, rpm: int):
        self.interval   = 60.0 / rpm
        self._last_call = 0.0

    def wait(self):
        elapsed = time.time() - self._last_call
        gap     = self.interval - elapsed
        if gap > 0:
            time.sleep(gap)
        self._last_call = time.time()

    def backoff(self, attempt: int):
        wait_s = min(15 * (2 ** attempt), 60)
        print(f"\n  ⏳ Rate limit — waiting {wait_s}s (attempt {attempt+1}/{MAX_RETRIES})...")
        time.sleep(wait_s)


# ── Core evaluator ────────────────────────────────────────────────────────────
def evaluate_case(tc: dict, pipeline: NLPPipeline,
                  limiter: Optional[RateLimiter]) -> TestResult:

    tc_id         = tc["id"]
    category      = tc.get("category", "valid")
    description   = tc.get("description", "")
    input_text    = tc.get("input", "")
    expected      = tc.get("expected", {})
    exp_codes     = expected.get("snomed_codes", [])
    exp_filters   = expected.get("filters", {})
    should_reject = expected.get("should_reject", False)

    result = TestResult(
        tc_id=tc_id, category=category,
        description=description, input_text=input_text,
        passed=False, expected=expected,
    )

    if limiter:
        limiter.wait()

    # ── Run pipeline with rate-limit retry ────────────────────────────────────
    output         = None
    pipeline_error = None
    was_rejected   = False
    t0             = time.time()

    for attempt in range(MAX_RETRIES):
        try:
            output = pipeline.run(input_text)
            break
        except (PreprocessorError, ExtractionError) as exc:
            msg = str(exc)
            if "rate limit" in msg.lower() or "429" in msg:
                if attempt < MAX_RETRIES - 1:
                    if limiter:
                        limiter.backoff(attempt)
                    continue
                result.rate_limited = True
                result.latency_ms   = int((time.time() - t0) * 1000)
                result.error        = f"RATE_LIMITED after {MAX_RETRIES} retries"
                result.passed       = False
                return result
            pipeline_error = f"{type(exc).__name__}: {exc}"
            was_rejected   = True
            break
        except Exception as exc:
            pipeline_error = f"UnexpectedError: {exc}"
            break

    result.latency_ms   = int((time.time() - t0) * 1000)
    result.was_rejected = was_rejected
    if pipeline_error and not was_rejected:
        result.error = pipeline_error

    if output is not None:
        seen, deduped = set(), []
        for t in output.snomed_terms:
            if t.code not in seen:
                seen.add(t.code)
                deduped.append(t.code)
        result.actual_codes = deduped
        result.actual_city  = output.filters.city.value
        result.actual_state = list(output.filters.state.values)
        result.actual_phase = output.filters.phase.value

    # ── Evaluate ──────────────────────────────────────────────────────────────
    failures = []

    if should_reject:
        if not was_rejected:
            failures.append(
                f"Expected REJECTION — got result (codes={result.actual_codes})"
            )
        result.failures = failures
        result.passed   = not failures
        return result

    if was_rejected:
        # Correct behavior: pipeline blocked a non-clinical/unsafe query
        if category in ("missing_condition", "edge_case", "invalid_nonsense"):
            result.passed = True
            return result
        failures.append(f"Unexpected rejection: {pipeline_error}")
        result.failures = failures
        result.passed   = False
        return result

    if result.error:
        failures.append(f"Pipeline crash: {result.error}")
        result.failures = failures
        result.passed   = False
        return result

    ok, detail = _codes_match(exp_codes, result.actual_codes)
    if not ok:
        failures.append(f"SNOMED: {detail} | actual={result.actual_codes}")

    if exp_filters.get("city") and not _fuzzy(exp_filters["city"], result.actual_city):
        failures.append(f"City: expected='{exp_filters['city']}' actual='{result.actual_city}'")

    if exp_filters.get("state") and not _state_match(exp_filters["state"], result.actual_state):
        failures.append(f"State: expected='{exp_filters['state']}' actual={result.actual_state}")

    if exp_filters.get("phase") and not _fuzzy(exp_filters["phase"], result.actual_phase):
        failures.append(f"Phase: expected='{exp_filters['phase']}' actual='{result.actual_phase}'")

    result.failures = failures
    result.passed   = not failures
    return result


# ── Report writer ─────────────────────────────────────────────────────────────
def write_results_md(summary: RunSummary, path: Path) -> None:
    now_str   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_total = summary.total - summary.skipped
    pass_rate = (summary.passed / run_total * 100) if run_total else 0
    avg_ms    = summary.total_ms // run_total if run_total else 0
    failed    = [r for r in summary.results if not r.passed and not r.rate_limited]
    skipped   = [r for r in summary.results if r.rate_limited]

    L = []
    a = L.append

    a("# Clinical NLP — Test Results")
    a(f"\n**Run date:** {now_str}  ")
    a(f"**Test file:** `tests/test_cases_202_fixed.json`  ")
    a(f"**Pipeline:** `NLPPipeline` · `src/pipeline.py`  ")
    a(f"**Groq rate limit:** {DEFAULT_RPM} req/min\n")
    a("---\n")

    # Summary
    a("## Summary\n")
    a("| Metric | Value |")
    a("|---|---|")
    a(f"| Total cases | **{summary.total}** |")
    a(f"| ✅ Passed | **{summary.passed}** |")
    a(f"| ❌ Failed | **{summary.failed}** |")
    a(f"| ⏭️ Skipped (rate limited) | **{summary.skipped}** |")
    a(f"| 💥 Pipeline errors | **{summary.errors}** |")
    a(f"| Pass rate (excl. skipped) | **{pass_rate:.1f}%** |")
    a(f"| Avg latency | **{avg_ms} ms** |")
    a(f"| Total run time | **{summary.total_ms/1000:.1f} s** |")
    a("")

    # By category
    a("## Results by Category\n")
    a("| Category | Total | Passed | Failed | Skipped | Pass % |")
    a("|---|---|---|---|---|---|")
    for cat, s in sorted(summary.by_category.items()):
        run = s["total"] - s.get("skipped", 0)
        pct = (s["passed"] / run * 100) if run else 0
        icon = "✅" if pct == 100 else ("⚠️" if pct >= 70 else "❌")
        a(f"| {icon} `{cat}` | {s['total']} | {s['passed']} | {s['failed']} | {s.get('skipped',0)} | {pct:.0f}% |")
    a("")

    # Accuracy breakdown
    valid_r = [r for r in summary.results
               if not r.rate_limited and r.category not in ("injection","harmful")]
    with_codes  = [r for r in valid_r if r.expected.get("snomed_codes")]
    codes_ok    = [r for r in with_codes if not any("SNOMED" in f for f in r.failures)]
    snomed_pct  = (len(codes_ok)/len(with_codes)*100) if with_codes else 0
    reject_r    = [r for r in summary.results if r.expected.get("should_reject")]
    reject_ok   = [r for r in reject_r if r.was_rejected]
    reject_pct  = (len(reject_ok)/len(reject_r)*100) if reject_r else 0

    fc = {"city":[0,0], "state":[0,0], "phase":[0,0]}
    for r in valid_r:
        ef = r.expected.get("filters", {})
        for k, kw in [("city","City"),("state","State"),("phase","Phase")]:
            if ef.get(k):
                fc[k][0] += 1
                if not any(kw in f for f in r.failures):
                    fc[k][1] += 1

    a("## Accuracy Breakdown\n")
    a("| Check | Cases | Correct | Accuracy |")
    a("|---|---|---|---|")
    a(f"| SNOMED code match | {len(with_codes)} | {len(codes_ok)} | {snomed_pct:.1f}% |")
    for k in ["city","state","phase"]:
        tot, ok = fc[k]
        pct = (ok/tot*100) if tot else 0
        a(f"| Filter: {k} | {tot} | {ok} | {pct:.1f}% |")
    a(f"| Injection/harmful rejection | {len(reject_r)} | {len(reject_ok)} | {reject_pct:.1f}% |")
    a("")

    # Skipped
    if skipped:
        a("---\n")
        a(f"## ⏭️ Skipped — Rate Limited ({len(skipped)})\n")
        a("Re-run after Groq quota resets (midnight UTC). These are NOT failures.\n")
        a("| ID | Category | Input |")
        a("|---|---|---|")
        for r in skipped:
            a(f"| `{r.tc_id}` | `{r.category}` | {r.input_text[:70]} |")
        a("")

    # Failed cases
    a("---\n")
    a(f"## ❌ Failed Cases ({len(failed)})\n")

    if not failed:
        a("🎉 **All executed cases passed!**\n")
    else:
        by_cat = {}
        for r in failed:
            by_cat.setdefault(r.category, []).append(r)
        for cat, items in sorted(by_cat.items()):
            a(f"### `{cat}` — {len(items)} failure(s)\n")
            for r in items:
                if r.error and not r.was_rejected:
                    status = "💥 CRASH"
                elif r.was_rejected and not r.expected.get("should_reject"):
                    status = "🚫 UNEXPECTED REJECT"
                else:
                    status = "❌ FAIL"
                a(f"#### {status} · `{r.tc_id}`")
                a(f"> {r.description}\n")
                a(f"**Input:** `{r.input_text[:120]}`\n")
                ef = r.expected.get("filters", {})
                a("**Expected:**")
                if r.expected.get("should_reject"):
                    a("- Pipeline should reject this input")
                else:
                    ec = r.expected.get("snomed_codes", [])
                    if ec:      a(f"- SNOMED: `{ec}`")
                    if ef.get("city"):  a(f"- city: `{ef['city']}`")
                    if ef.get("state"): a(f"- state: `{ef['state']}`")
                    if ef.get("phase"): a(f"- phase: `{ef['phase']}`")
                    if not ec and not ef: a("- *(no specific expectations)*")
                a("")
                if r.was_rejected:
                    a(f"**Actual:** Rejected → `{r.error}`")
                elif r.error:
                    a(f"**Actual:** Crashed → `{r.error}`")
                else:
                    a(f"**Actual:** codes=`{r.actual_codes}` city=`{r.actual_city}` "
                      f"state=`{r.actual_state}` phase=`{r.actual_phase}`")
                a("")
                if r.failures:
                    a("**Failure reasons:**")
                    for f_msg in r.failures:
                        a(f"- {f_msg}")
                a(f"\n*Latency: {r.latency_ms} ms*\n")
                a("---\n")

    # Passed summary
    passed_list = [r for r in summary.results if r.passed]
    a(f"## ✅ Passed Cases ({len(passed_list)})\n")
    a("| ID | Category | Description | Latency |")
    a("|---|---|---|---|")
    for r in passed_list:
        desc = r.description[:60] + ("..." if len(r.description)>60 else "")
        a(f"| `{r.tc_id}` | `{r.category}` | {desc} | {r.latency_ms} ms |")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"\n📄 Report → {path}")


# ── Runner ────────────────────────────────────────────────────────────────────
def run(input_path: Path, output_dir: Path, rpm: int,
        limit: Optional[int] = None, category_filter: Optional[str] = None) -> RunSummary:

    with open(input_path, encoding="utf-8") as f:
        cases = json.load(f)

    if category_filter:
        cases = [tc for tc in cases if tc.get("category") == category_filter]
        print(f"Filtered to '{category_filter}': {len(cases)} cases")
    if limit:
        cases = cases[:limit]

    # Run no-API categories first (free, instant)
    cases = sorted(cases, key=lambda tc: (0 if tc.get("category") in NO_API_CATEGORIES else 1))

    api_count = sum(1 for tc in cases if tc.get("category") not in NO_API_CATEGORIES)
    est_min   = api_count / rpm

    print(f"\n🔬 Clinical NLP Test Agent")
    print(f"   Cases        : {len(cases)}  (API calls: {api_count}, no-API: {len(cases)-api_count})")
    print(f"   Rate limit   : {rpm} req/min  →  est. {est_min:.1f} min\n")

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        print("❌ GROQ_API_KEY not found. Check .env or env_variable.env")
        sys.exit(1)

    print("⏳ Loading pipeline...")
    t0 = time.time()
    pipeline = NLPPipeline(groq_api_key=api_key)
    print(f"✅ Ready in {time.time()-t0:.1f}s\n")

    limiter = RateLimiter(rpm)
    summary = RunSummary()

    for i, tc in enumerate(cases, 1):
        tc_id    = tc.get("id", f"TC{i:04d}")
        category = tc.get("category", "valid")
        needs_api = category not in NO_API_CATEGORIES

        print(f"[{i:3d}/{len(cases)}] {tc_id} ({category}) {tc.get('input','')[:50]!r:.48}", end="  ")

        result = evaluate_case(tc, pipeline, limiter if needs_api else None)

        summary.total    += 1
        summary.total_ms += result.latency_ms
        summary.results.append(result)

        if category not in summary.by_category:
            summary.by_category[category] = {"total":0,"passed":0,"failed":0,"skipped":0}
        summary.by_category[category]["total"] += 1

        if result.rate_limited:
            summary.skipped += 1
            summary.by_category[category]["skipped"] += 1
            print(f"⏭️  rate limited")
        elif result.passed:
            summary.passed += 1
            summary.by_category[category]["passed"] += 1
            print(f"✅ {result.latency_ms}ms")
        else:
            summary.failed += 1
            summary.by_category[category]["failed"] += 1
            if result.error and not result.was_rejected:
                summary.errors += 1
            short = result.failures[0][:60] if result.failures else (result.error or "?")
            print(f"❌ {result.latency_ms}ms — {short}")

    run_total = summary.total - summary.skipped
    pass_rate = (summary.passed / run_total * 100) if run_total else 0
    print(f"\n{'─'*58}")
    print(f"  Total   : {summary.total}  |  Passed: {summary.passed} ({pass_rate:.1f}%)")
    print(f"  Failed  : {summary.failed}  |  Skipped: {summary.skipped}  |  Errors: {summary.errors}")
    print(f"{'─'*58}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    write_results_md(summary, output_dir / f"results_{ts}.md")
    write_results_md(summary, output_dir / "results.md")
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clinical NLP Test Agent")
    parser.add_argument("--input",    "-i", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output",   "-o", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rpm",            type=int,  default=DEFAULT_RPM,
                        help="Groq requests/min (default 25, free tier max 30)")
    parser.add_argument("--limit",    "-n", type=int,  default=None)
    parser.add_argument("--category", "-c", type=str,  default=None)
    args = parser.parse_args()

    s = run(args.input, args.output, args.rpm, args.limit, args.category)
    run_total = s.total - s.skipped
    sys.exit(0 if run_total and (s.passed / run_total) >= 0.70 else 1)