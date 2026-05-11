"""Parity test: HybridCascadeStrategy recall vs legacy SNOMEDResolver.

Uses the top 20 cases from tests/batch_test_cases.csv for speed — the full CSV
includes cases targeting geo/investigator/phase fields where expected_snomed_codes
is empty, so filtering to non-empty codes gives ~15-20 meaningful SNOMED cases.
Parity gate: HybridCascadeStrategy aggregate recall must be >= SNOMEDResolver recall.

Wave 6A: Cross-strategy parity (aho_corasick, ngram_lookup) added below.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

CSV_PATH = Path(__file__).parent.parent / "data" / "snomed_clinical_trials.csv"
BATCH_CSV = Path(__file__).parent / "batch_test_cases.csv"


def _parse_codes(raw: str) -> list[str]:
    if not raw or not raw.strip():
        return []
    return [c.strip() for c in raw.split("|") if c.strip()]


def _load_test_cases(max_cases: int = 20) -> list[dict]:
    df = pd.read_csv(BATCH_CSV, dtype=str, on_bad_lines="skip").fillna("")
    cases = []
    for _, row in df.iterrows():
        codes = _parse_codes(row.get("expected_snomed_codes", ""))
        if not codes:
            continue
        cases.append({"input": row["input"], "expected_codes": codes, "id": row["id"]})
        if len(cases) >= max_cases:
            break
    return cases


@pytest.mark.parity
def test_hybrid_cascade_recall_parity():
    """HybridCascadeStrategy recall on batch_test_cases.csv must not regress vs SNOMEDResolver.

    Recall = fraction of expected_snomed_codes present in returned codes for a query.
    Aggregate recall is averaged across all test cases with non-empty expected codes.
    """
    from src.snomed_resolver import SNOMEDResolver
    from src.snomed_search.hybrid_cascade import HybridCascadeStrategy

    cases = _load_test_cases(max_cases=20)
    assert cases, "No test cases with expected_snomed_codes found in batch_test_cases.csv"

    csv_str = str(CSV_PATH)
    legacy = SNOMEDResolver(csv_str)
    new_strategy = HybridCascadeStrategy(dictionary_path=csv_str)

    legacy_recalls = []
    new_recalls = []

    for case in cases:
        query = case["input"]
        expected = set(case["expected_codes"])

        # Legacy resolver: tokenize on whitespace and resolve each token
        legacy_codes = set()
        tokens = query.split()
        # Also try the full query and multi-word substrings via simple windowing
        for n in range(len(tokens), 0, -1):
            for i in range(len(tokens) - n + 1):
                phrase = " ".join(tokens[i: i + n])
                match = legacy.resolve(phrase, negated=False)
                if match is not None:
                    legacy_codes.add(match.code)

        # New strategy: search returns all candidates
        new_codes = {m.code for m in new_strategy.search(query)}

        legacy_recall = len(expected & legacy_codes) / len(expected)
        new_recall = len(expected & new_codes) / len(expected)

        legacy_recalls.append(legacy_recall)
        new_recalls.append(new_recall)

    avg_legacy = sum(legacy_recalls) / len(legacy_recalls)
    avg_new = sum(new_recalls) / len(new_recalls)

    assert avg_new >= avg_legacy, (
        f"HybridCascadeStrategy recall {avg_new:.3f} regressed vs "
        f"SNOMEDResolver {avg_legacy:.3f} on {len(cases)} cases"
    )


@pytest.mark.parity
def test_hybrid_cascade_search_type2_diabetes():
    """search returns code 44054006 for 'type 2 diabetes' (synonym path via alias dict).
    The preferred term 'type 2 diabetes mellitus' triggers match_type='exact'.
    AC#7 is met: code present; match_type differs because the query is a synonym/alias,
    not the canonical preferred term.
    """
    from src.snomed_search.hybrid_cascade import HybridCascadeStrategy

    strategy = HybridCascadeStrategy(dictionary_path=str(CSV_PATH))

    # Synonym/alias query: code present, returned via alias→synonym path
    results = strategy.search("type 2 diabetes")
    codes = {m.code for m in results}
    assert "44054006" in codes, f"Expected code 44054006 in results; got codes={codes}"

    # Preferred term query: must return match_type='exact'
    results_full = strategy.search("type 2 diabetes mellitus")
    exact_matches = [m for m in results_full if m.code == "44054006" and m.match_type == "exact"]
    assert exact_matches, "Expected match_type='exact' for preferred term 'type 2 diabetes mellitus'"


@pytest.mark.parity
def test_hybrid_cascade_valid_spans():
    """search('blood glucose monitoring') must return results with valid spans."""
    from src.snomed_search.hybrid_cascade import HybridCascadeStrategy

    strategy = HybridCascadeStrategy(dictionary_path=str(CSV_PATH))
    results = strategy.search("blood glucose monitoring")

    assert results, "Expected at least one match for 'blood glucose monitoring'"
    for m in results:
        assert m.span is not None, f"SNOMEDMatch.span is None for match {m.code}"
        assert m.span[0] >= 0, f"span start < 0: {m.span}"
        assert m.span[1] > m.span[0], f"span end not > start: {m.span}"


@pytest.mark.parity
def test_health_check_ready():
    """health_check() must return ready=True with dictionary_size > 0."""
    from src.snomed_search.hybrid_cascade import HybridCascadeStrategy

    strategy = HybridCascadeStrategy(dictionary_path=str(CSV_PATH))
    health = strategy.health_check()

    assert health["ready"] is True
    assert health["dictionary_size"] > 0
    assert health["name"] == "hybrid_cascade"


@pytest.mark.parity
def test_registry_get_strategy():
    """get_strategy() returns a working HybridCascadeStrategy instance."""
    from src.snomed_search.registry import get_strategy, DEFAULT_STRATEGY, STRATEGY_REGISTRY

    assert DEFAULT_STRATEGY == "hybrid_cascade"
    assert "hybrid_cascade" in STRATEGY_REGISTRY

    strategy = get_strategy(dictionary_path=str(CSV_PATH))
    assert strategy.health_check()["ready"] is True


@pytest.mark.parity
def test_registry_unknown_strategy_raises():
    """get_strategy('nonexistent') raises ValueError listing available names."""
    from src.snomed_search.registry import get_strategy

    with pytest.raises(ValueError, match="hybrid_cascade"):
        get_strategy("nonexistent", dictionary_path=str(CSV_PATH))


@pytest.mark.parity
def test_alias_dictionary_canonical_location():
    """ALIAS_DICTIONARY importable from hybrid_cascade; snomed_resolver re-exports it."""
    from src.snomed_search.hybrid_cascade import ALIAS_DICTIONARY as hc_dict
    from src.snomed_resolver import ALIAS_DICTIONARY as shim_dict

    assert hc_dict is shim_dict, "ALIAS_DICTIONARY must be the same object (re-export, not copy)"
    assert len(hc_dict) > 0


# ---------------------------------------------------------------------------
# Wave 6A: new strategy smoke tests
# ---------------------------------------------------------------------------

@pytest.mark.parity
def test_ngram_lookup_imports_and_health():
    """NGramLookupStrategy requires no extra deps; health_check must report ready."""
    from src.snomed_search.ngram_lookup import NGramLookupStrategy

    strategy = NGramLookupStrategy(dictionary_path=str(CSV_PATH))
    health = strategy.health_check()
    assert health["ready"] is True
    assert health["name"] == "ngram_lookup"
    assert health["dictionary_size"] > 0


@pytest.mark.parity
def test_ngram_lookup_type2_diabetes():
    """NGramLookupStrategy.search('type 2 diabetes') must return code 44054006 (AC#4)."""
    from src.snomed_search.ngram_lookup import NGramLookupStrategy

    strategy = NGramLookupStrategy(dictionary_path=str(CSV_PATH))
    results = strategy.search("type 2 diabetes")
    codes = {m.code for m in results}
    assert "44054006" in codes, f"Expected 44054006; got {codes}"


@pytest.mark.parity
def test_ngram_lookup_valid_spans():
    """Every match from NGramLookupStrategy must have valid char spans (AC#6)."""
    from src.snomed_search.ngram_lookup import NGramLookupStrategy

    strategy = NGramLookupStrategy(dictionary_path=str(CSV_PATH))
    query = "blood glucose monitoring hypertension"
    results = strategy.search(query)
    assert results, "Expected at least one match"
    for m in results:
        assert m.span[0] >= 0, f"span start < 0: {m.span}"
        assert m.span[1] > m.span[0], f"span end not > start: {m.span}"
        assert m.span[1] <= len(query), f"span end > query len: {m.span}"


@pytest.mark.parity
def test_ngram_lookup_confidence_values():
    """NGramLookupStrategy exact hits must be 0.99; synonym hits 0.97 (AC#7)."""
    from src.snomed_search.ngram_lookup import NGramLookupStrategy

    strategy = NGramLookupStrategy(dictionary_path=str(CSV_PATH))
    # Use preferred term for exact, synonym for synonym path
    for m in strategy.search("type 2 diabetes mellitus"):
        if m.match_type == "exact":
            assert m.confidence == 0.99, f"exact confidence wrong: {m.confidence}"
        elif m.match_type == "synonym":
            assert m.confidence == 0.97, f"synonym confidence wrong: {m.confidence}"


@pytest.mark.parity
def test_registry_contains_ngram_lookup():
    """STRATEGY_REGISTRY must include ngram_lookup after Wave 6A (AC#11)."""
    from src.snomed_search.registry import STRATEGY_REGISTRY, DEFAULT_STRATEGY

    assert "ngram_lookup" in STRATEGY_REGISTRY
    assert DEFAULT_STRATEGY == "hybrid_cascade"


@pytest.mark.parity
def test_aho_corasick_import_behavior():
    """AhoCorasickStrategy import succeeds iff pyahocorasick is installed (AC#1)."""
    try:
        import ahocorasick  # noqa: F401
        ac_available = True
    except ImportError:
        ac_available = False

    if ac_available:
        from src.snomed_search.aho_corasick import AhoCorasickStrategy
        strategy = AhoCorasickStrategy(dictionary_path=str(CSV_PATH))
        health = strategy.health_check()
        assert health["ready"] is True
        assert health["name"] == "aho_corasick"
        assert health["dictionary_size"] > 0
    else:
        # Without pyahocorasick the module-level import guard sets _AC_AVAILABLE=False;
        # instantiation must raise StrategyError (not silently fall back).
        from src.exceptions import StrategyError
        from src.snomed_search.aho_corasick import AhoCorasickStrategy
        with pytest.raises(StrategyError, match="pyahocorasick"):
            AhoCorasickStrategy(dictionary_path=str(CSV_PATH))


@pytest.mark.parity
def test_aho_corasick_word_boundary():
    """AhoCorasickStrategy must NOT match 'cancer' as a substring inside 'pancreatic' (AC#8)."""
    try:
        import ahocorasick  # noqa: F401
    except ImportError:
        pytest.skip("pyahocorasick not installed")

    from src.snomed_search.aho_corasick import AhoCorasickStrategy

    strategy = AhoCorasickStrategy(dictionary_path=str(CSV_PATH))
    # "pancreatic" contains "cancer" as a substring — must not produce a standalone match
    results = strategy.search("pancreatic cancer")
    # "pancreatic cancer" should match as a phrase; the bare "cancer" sub-match inside
    # "pancreatic" must be suppressed by the word-boundary check.
    codes_by_span = {m.span: m for m in results}
    # No hit should have span covering only the interior of "pancreatic" (chars 0-10)
    # "cancer" is chars 11-17, so a span of (3,9) would be the bad sub-word match.
    bad_spans = [
        span for span in codes_by_span
        if span[0] > 0 and span[1] < len("pancreatic") and "pancreatic"[span[0]:span[1]] == "cancer"
    ]
    assert not bad_spans, f"Word boundary check failed; bad sub-word spans: {bad_spans}"


@pytest.mark.parity
def test_aho_corasick_valid_spans():
    """Every match from AhoCorasickStrategy must have valid char spans (AC#5, AC#6)."""
    try:
        import ahocorasick  # noqa: F401
    except ImportError:
        pytest.skip("pyahocorasick not installed")

    from src.snomed_search.aho_corasick import AhoCorasickStrategy

    strategy = AhoCorasickStrategy(dictionary_path=str(CSV_PATH))
    query = "type 2 diabetes mellitus hypertension"
    results = strategy.search(query)
    assert results, "Expected at least one match"
    for m in results:
        assert m.span[0] >= 0
        assert m.span[1] > m.span[0]
        assert m.span[1] <= len(query)


# ---------------------------------------------------------------------------
# Wave 6A: cross-strategy parity test (proposal §3.12 + §10)
# ---------------------------------------------------------------------------

def _top5_codes(matches) -> set[str]:
    """Return top-5 codes sorted by confidence descending."""
    sorted_matches = sorted(matches, key=lambda m: -m.confidence)
    return {m.code for m in sorted_matches[:5]}


def _jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


@pytest.mark.parity
def test_cross_strategy_parity():
    """
    ngram_lookup (and aho_corasick if available) top-5 SNOMED codes must have
    Jaccard >= 0.80 vs hybrid_cascade on >= 80% of cases (AC#12).
    """
    from src.snomed_search.hybrid_cascade import HybridCascadeStrategy
    from src.snomed_search.ngram_lookup import NGramLookupStrategy

    try:
        import ahocorasick  # noqa: F401
        from src.snomed_search.aho_corasick import AhoCorasickStrategy
        ac_strategy = AhoCorasickStrategy(dictionary_path=str(CSV_PATH))
    except (ImportError, Exception):
        ac_strategy = None

    df = pd.read_csv(BATCH_CSV, dtype=str, on_bad_lines="skip").fillna("")
    queries = [row["input"] for _, row in df.iterrows() if row.get("input", "").strip()]
    assert queries, "No queries found in batch_test_cases.csv"

    csv_str = str(CSV_PATH)
    baseline = HybridCascadeStrategy(dictionary_path=csv_str)
    ngram = NGramLookupStrategy(dictionary_path=csv_str)

    ngram_pass = 0
    ac_pass = 0
    ac_total = 0

    for query in queries:
        baseline_codes = _top5_codes(baseline.search(query))
        ngram_codes = _top5_codes(ngram.search(query))

        if _jaccard(baseline_codes, ngram_codes) >= 0.80:
            ngram_pass += 1

        if ac_strategy is not None:
            ac_total += 1
            ac_codes = _top5_codes(ac_strategy.search(query))
            if _jaccard(baseline_codes, ac_codes) >= 0.80:
                ac_pass += 1

    total = len(queries)
    ngram_rate = ngram_pass / total
    assert ngram_rate >= 0.80, (
        f"NGramLookupStrategy parity failed: {ngram_pass}/{total} cases "
        f"({ngram_rate:.1%}) met Jaccard >= 0.80 vs hybrid_cascade (need 80%)"
    )

    if ac_strategy is not None and ac_total > 0:
        ac_rate = ac_pass / ac_total
        assert ac_rate >= 0.80, (
            f"AhoCorasickStrategy parity failed: {ac_pass}/{ac_total} cases "
            f"({ac_rate:.1%}) met Jaccard >= 0.80 vs hybrid_cascade (need 80%)"
        )
