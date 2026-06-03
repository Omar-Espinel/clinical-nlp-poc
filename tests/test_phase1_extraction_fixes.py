"""Tests for Phase 1 extraction quality fixes (Fixes 1-6).

Fix 1: ambiguous_terms.json — allow_parent_search consistency.
Fix 2: NameExtractor strips request prefixes from start of query.
Fix 3: _build_derived_entries excludes tokens equal to a full preferred_term.
Fix 4: MetricIntentResolver requires numeric/comparator/label context in ±10-token window.
Fix 5: PhaseExtractor handles conjunctions ("Phase 2 or 3", "Phase 2/3").
Fix 6: PreflightMandatoryCheck Signal D — capitalized non-function-word token.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
AMBIG_JSON_PATH = PROJECT_ROOT / "data" / "ambiguous_terms.json"
SNOMED_CSV_PATH = str(PROJECT_ROOT / "data" / "snomed_clinical_trials.csv")
INST_KW_PATH = str(PROJECT_ROOT / "data" / "institution_keywords.json")
GEO_PATH = str(PROJECT_ROOT / "data" / "geo_canonical.json")


# ---------------------------------------------------------------------------
# Fix 1: ambiguous_terms.json — allow_parent_search consistency
# ---------------------------------------------------------------------------

def test_fix1_allow_parent_search_consistency():
    """Every entry with non-null snomed_parent_code must have allow_parent_search=true."""
    data = json.loads(AMBIG_JSON_PATH.read_text(encoding="utf-8"))
    for trigger, entry in data.items():
        parent_code = entry.get("snomed_parent_code")
        allow = entry.get("allow_parent_search", False)
        if parent_code is not None:
            assert allow is True, (
                f"Entry '{trigger}' has snomed_parent_code={parent_code!r} "
                f"but allow_parent_search={allow!r} (expected true)"
            )
        else:
            assert allow is False, (
                f"Entry '{trigger}' has snomed_parent_code=null "
                f"but allow_parent_search={allow!r} (expected false)"
            )


# ---------------------------------------------------------------------------
# Fix 2: NameExtractor strips request prefixes
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def name_extractor():
    from src.extractors.names import NameExtractor
    return NameExtractor(
        institution_keywords_path=INST_KW_PATH,
        geo_json_path=GEO_PATH,
        snomed_known_terms=frozenset({"gout", "diabetes"}),
    )


def test_fix2_strips_can_you_find_me(name_extractor):
    """'Can you find me studies on gout' — filler phrases must not produce
    spurious name candidates 'Can', 'You', 'Find', 'Me'."""
    result = name_extractor.extract("Can you find me studies on gout")
    all_names = [
        result.investigator_name,
        result.site_name,
    ] + [am.text for am in result.ambiguous_names]
    disallowed = {"can", "you", "find", "me"}
    for name in all_names:
        if name is not None:
            assert name.lower() not in disallowed, (
                f"Request phrase '{name}' must not appear in extracted names"
            )


@pytest.mark.parametrize("query,disallowed_tokens", [
    ("show me phase 3 trials", {"show", "me"}),
    ("give me studies for diabetes", {"give", "me"}),
    ("please find trials at mayo clinic", {"please"}),
    ("looking for trials by dr smith", {"looking"}),
    ("search for diabetes studies", {"search"}),
])
def test_fix2_various_prefixes(name_extractor, query, disallowed_tokens):
    """Request-prefix phrases are not extracted as name candidates."""
    result = name_extractor.extract(query)
    all_names = [
        result.investigator_name,
        result.site_name,
    ] + [am.text for am in result.ambiguous_names]
    for name in all_names:
        if name is not None:
            assert name.lower() not in disallowed_tokens, (
                f"Request phrase '{name}' must not appear in extracted names for: {query!r}"
            )


# ---------------------------------------------------------------------------
# Fix 3: _build_derived_entries excludes tokens equal to a full preferred_term
# ---------------------------------------------------------------------------

def test_fix3_leukemia_not_a_derived_trigger():
    """'leukemia' appears as a full preferred_term in the CSV and must NOT become
    a derived ambiguous trigger — it is already directly searchable."""
    import pandas as pd
    from src.sufficiency_gate import _build_derived_entries

    derived = _build_derived_entries(Path(SNOMED_CSV_PATH))
    # Check that "leukemia" is not in derived triggers
    assert "leukemia" not in derived, (
        "'leukemia' should not be a derived trigger because it is a full preferred_term"
    )


def test_fix3_preferred_terms_not_in_derived():
    """No token that exactly equals a CSV preferred_term should appear as a derived trigger."""
    import pandas as pd
    from src.sufficiency_gate import _build_derived_entries

    df = pd.read_csv(SNOMED_CSV_PATH, dtype=str).fillna("")
    preferred_terms = frozenset(
        t.strip().lower()
        for t in df["preferred_term"].tolist()
        if t.strip()
    )

    derived = _build_derived_entries(Path(SNOMED_CSV_PATH))
    for token in derived:
        assert token not in preferred_terms, (
            f"Derived trigger '{token}' is identical to a CSV preferred_term — must be excluded"
        )


# ---------------------------------------------------------------------------
# Fix 4: MetricIntentResolver requires context window for matches
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def metric_resolver():
    from src.normalizers.metric import MetricIntentResolver
    return MetricIntentResolver(
        str(PROJECT_ROOT / "data" / "metric_filters.json"),
        strict_validation=False,
    )


def test_fix4_no_metric_match_without_numeric_context(metric_resolver):
    """'lung cancer research in New York' has no numeric value or comparator —
    the metric resolver must not produce any matches."""
    matches = metric_resolver.resolve("lung cancer research in New York")
    assert matches == [], (
        f"Expected no metric matches for generic research query, got: {matches}"
    )


def test_fix4_metric_match_kept_with_numeric_context(metric_resolver):
    """A query with an explicit number should still produce metric matches."""
    matches = metric_resolver.resolve("trials with enrollment over 100")
    # At least one metric match should survive (enrollment has numeric context "100")
    assert len(matches) >= 1, (
        "Expected at least one metric match with explicit numeric context"
    )


# ---------------------------------------------------------------------------
# Fix 5: PhaseExtractor handles conjunctions
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def phase_extractor():
    from src.extractors.phase import PhaseExtractor
    return PhaseExtractor()


@pytest.mark.parametrize("query,expected_values", [
    ("Phase 2 or 3 studies", ["Phase 2", "Phase 3"]),
    ("Phase 2/3 diabetes trial", ["Phase 2", "Phase 3"]),
    ("phase II or III oncology", ["Phase 2", "Phase 3"]),
    ("Phase II/III study", ["Phase 2", "Phase 3"]),
    ("Phase 2 and 3 research", ["Phase 2", "Phase 3"]),
    ("phase i/ii dose finding", ["Phase 1", "Phase 2"]),
])
def test_fix5_phase_conjunction(phase_extractor, query, expected_values):
    """Phase conjunctions expand to multi-value PhaseResult."""
    result = phase_extractor.extract(query)
    assert result.values is not None, (
        f"Expected multi-value result for {query!r}, got values=None"
    )
    assert sorted(result.values) == sorted(expected_values), (
        f"Phase values mismatch for {query!r}: got {result.values!r}, "
        f"expected {expected_values!r}"
    )
    assert result.confidence >= 0.95


def test_fix5_single_phase_no_conjunction(phase_extractor):
    """A single-phase query must NOT produce a multi-value result."""
    result = phase_extractor.extract("Phase 3 cancer trial")
    assert result.value == "Phase 3"
    assert result.confidence == 0.95
    # values may be None or a single-element list; must NOT be multi-value
    if result.values is not None:
        assert len(result.values) == 1


def test_fix5_phase_filter_shape():
    """PhaseFilter (the new type) is importable and has the correct shape."""
    from src.filter_extractor import PhaseFilter
    pf = PhaseFilter(values=["Phase 2", "Phase 3"], confidence=0.95)
    assert pf.values == ["Phase 2", "Phase 3"]
    assert pf.confidence == 0.95


def test_fix5_phase_filter_output_shape():
    """PhaseFilterOutput in assembler has correct shape."""
    from src.assembler import PhaseFilterOutput
    out = PhaseFilterOutput(values=["Phase 2", "Phase 3"], confidence=0.95)
    assert out.values == ["Phase 2", "Phase 3"]


# ---------------------------------------------------------------------------
# Fix 6: PreflightMandatoryCheck Signal D (capitalized non-function-word token)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def preflight():
    from src.extractors.preflight import PreflightMandatoryCheck
    from src.extractors.names import PERSON_PREFIXES, PERSON_SUFFIXES, CONTEXT_PERSON, CONTEXT_SITE

    known_terms = frozenset({"diabetes", "hypertension"})
    institution_kw = frozenset({"hospital", "clinic", "university"})
    geo_multiword = frozenset({"new york", "los angeles", "east coast", "west coast", "new england"})
    return PreflightMandatoryCheck(
        known_terms=known_terms,
        person_prefixes=PERSON_PREFIXES,
        person_suffixes=PERSON_SUFFIXES,
        institution_keywords=institution_kw,
        context_signals=CONTEXT_PERSON | CONTEXT_SITE,
        geo_multiword_keys=geo_multiword,
    )


def test_fix6_gout_michigan_passes(preflight):
    """'Gout studies in Michigan' — 'Gout' is a capitalized non-function-word token
    and should fire Signal D (new addition in Fix 6), making the query pass preflight."""
    result = preflight.evaluate("Gout studies in Michigan")
    assert result.passed is True, (
        f"'Gout studies in Michigan' should pass preflight via Signal D, "
        f"got signal_count={result.signal_count}"
    )
    assert result.signal_count >= 1


def test_fix6_phase_ii_dr_holmes_passes(preflight):
    """'Phase ii studies by Dr Holmes' — Dr fires Signal B (person prefix),
    Holmes fires Signal D; both should contribute."""
    result = preflight.evaluate("Phase ii studies by Dr Holmes")
    assert result.passed is True, (
        f"'Phase ii studies by Dr Holmes' should pass preflight, "
        f"got signal_count={result.signal_count}"
    )
    assert result.signal_count >= 2


@pytest.mark.parametrize("query", [
    "Alzheimer Disease Phase 3",
    "Parkinson Disease trials",
    "Johnson oncology",
])
def test_fix6_capitalized_medical_terms_pass(preflight, query):
    """Capitalized medical terms trigger Signal D and allow the query through."""
    result = preflight.evaluate(query)
    assert result.passed is True, (
        f"{query!r} should pass preflight via Signal D, "
        f"got signal_count={result.signal_count}"
    )
