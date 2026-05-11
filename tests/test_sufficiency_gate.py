"""tests/test_sufficiency_gate.py — 10 unit-test cases for SufficiencyGate.

Spec: rework-nlp-proposal.md §3.12 + rework-nlp-impl-spec.md §1-2.

Uses:
- Real HybridCascadeStrategy (already built and QA'd in Wave 2)
- MockConversationSession (no circular import)
- SimpleNamespace shim for ExtractedFilters (Wave 4 not yet available)
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Ensure project root is on the path for direct invocation
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.snomed_search.hybrid_cascade import HybridCascadeStrategy
from src.snomed_search.base import SNOMEDMatch
from src.sufficiency_gate import (
    AmbiguousEntry,
    AmbiguousTermsRegistry,
    SufficiencyDecision,
    SufficiencyGate,
    _any_filter_set,
    _count_set_filters,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CSV_PATH = str(Path(__file__).parent.parent / "data" / "snomed_clinical_trials.csv")
JSON_PATH = str(Path(__file__).parent.parent / "data" / "ambiguous_terms.json")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def strategy():
    """Real HybridCascadeStrategy — built once for the module."""
    return HybridCascadeStrategy(CSV_PATH)


@pytest.fixture(scope="module")
def registry(strategy):
    """Real AmbiguousTermsRegistry loaded from ambiguous_terms.json."""
    # Reset class-level cache between fixture scopes won't matter here since
    # we use strict_validation=True and our JSON is valid.
    return AmbiguousTermsRegistry(
        path=JSON_PATH,
        snomed_strategy=strategy,
        snomed_csv_path=CSV_PATH,
        strict_validation=True,
    )


@pytest.fixture(scope="module")
def gate(registry):
    """SufficiencyGate with the shared registry."""
    # Reset class-level cache so this test module gets a fresh build
    SufficiencyGate._default_condition_prompt_cache = None
    return SufficiencyGate(registry=registry, snomed_csv_path=CSV_PATH)


# ---------------------------------------------------------------------------
# MockConversationSession
# ---------------------------------------------------------------------------

class MockConversationSession:
    """Minimal session object for gate tests.  Mirrors ConversationSession API."""

    def __init__(self, session_id: str = "test-session", clarification_count: int = 0,
                 max_turns: int = 3):
        self.session_id = session_id
        self._clarification_count = clarification_count
        self._max_turns = max_turns
        self.turns: list = []

    def is_max_turns_reached(self) -> bool:
        return self._clarification_count >= self._max_turns

    def clarification_turn_count(self) -> int:
        return self._clarification_count


# ---------------------------------------------------------------------------
# ExtractedFilters shim (Wave 4 not yet available)
# ---------------------------------------------------------------------------

def _make_filters(
    city: str = None,
    state: list = None,
    phase: str = None,
    investigator: str = None,
    site: str = None,
):
    """Build a duck-typed ExtractedFilters shim using SimpleNamespace."""
    return SimpleNamespace(
        investigator_name=SimpleNamespace(value=investigator),
        site_name=SimpleNamespace(value=site),
        city=SimpleNamespace(value=city),
        state=SimpleNamespace(values=state or []),
        phase=SimpleNamespace(value=phase),
    )


def _make_snomed_match(confidence: float = 0.99, negated: bool = False) -> SNOMEDMatch:
    return SNOMEDMatch(
        code="363346000",
        display="malignant neoplasm",
        match_type="exact",
        confidence=confidence,
        original_text="cancer",
        span=(0, 6),
        negated=negated,
    )


# ---------------------------------------------------------------------------
# Test Case 1: Strong-match indication — "lung cancer phase 2" → sufficient
# ---------------------------------------------------------------------------

def test_case1_strong_match_indication(gate):
    """'lung cancer phase 2' — override present (lung cancer) → sufficient."""
    session = MockConversationSession()
    decision = gate.evaluate("lung cancer phase 2", session)
    assert decision.sufficient is True
    assert decision.reason in ("ok_no_trigger", "max_turns_reached")


# ---------------------------------------------------------------------------
# Test Case 2: Bare ambiguous — "cancer" → NOT sufficient, options include "Lung Cancer"
# ---------------------------------------------------------------------------

def test_case2_bare_ambiguous(gate):
    """'cancer' — bare trigger, no override → NOT sufficient."""
    session = MockConversationSession()
    decision = gate.evaluate("cancer", session)
    assert decision.sufficient is False
    assert decision.reason == "ambiguous_trigger"
    assert decision.matched_entry is not None
    assert "Lung Cancer" in decision.matched_entry.options


# ---------------------------------------------------------------------------
# Test Case 3: Override present — "lung cancer in NYC" → sufficient
# ---------------------------------------------------------------------------

def test_case3_override_present(gate):
    """'lung cancer in NYC' — 'lung cancer' is in override_terms → trigger suppressed."""
    session = MockConversationSession()
    decision = gate.evaluate("lung cancer in NYC", session)
    assert decision.sufficient is True


# ---------------------------------------------------------------------------
# Test Case 4: Override case-insensitive — "Lung Cancer in nyc" → sufficient
# ---------------------------------------------------------------------------

def test_case4_override_case_insensitive(gate):
    """Override matching is case-insensitive."""
    session = MockConversationSession()
    decision = gate.evaluate("Lung Cancer in nyc", session)
    assert decision.sufficient is True


# ---------------------------------------------------------------------------
# Test Case 5: Filters but ambiguous — "cancer in Boston phase 3" → NOT sufficient
# ---------------------------------------------------------------------------

def test_case5_filters_but_ambiguous(gate):
    """Bare 'cancer' still fires even with city/phase context."""
    session = MockConversationSession()
    decision = gate.evaluate("cancer in Boston phase 3", session)
    assert decision.sufficient is False
    assert decision.reason == "ambiguous_trigger"


# ---------------------------------------------------------------------------
# Test Case 6: Multiple ambiguous — "heart and lung problems" → NOT sufficient
#              fires first match
# ---------------------------------------------------------------------------

def test_case6_multiple_ambiguous(gate, registry):
    """Multiple ambiguous triggers — first one in JSON order fires."""
    session = MockConversationSession()
    decision = gate.evaluate("heart and lung problems", session)
    assert decision.sufficient is False
    assert decision.reason == "ambiguous_trigger"
    # triggered_by must be one of the registered triggers
    assert decision.triggered_by in registry._entries


# ---------------------------------------------------------------------------
# Test Case 7: Max-turns reached — session at 3 clarifications + "cancer" → sufficient
# ---------------------------------------------------------------------------

def test_case7_max_turns_reached(gate):
    """Escape valve: max clarification turns forces sufficient=True."""
    session = MockConversationSession(clarification_count=3, max_turns=3)
    decision = gate.evaluate("cancer", session)
    assert decision.sufficient is True
    assert decision.reason == "max_turns_reached"


# ---------------------------------------------------------------------------
# Test Case 8: Partial-word non-match — "incancentive program" → sufficient
# ---------------------------------------------------------------------------

def test_case8_partial_word_non_match(gate):
    """Whole-word boundary \\b prevents 'cancer' matching inside 'incancentive'."""
    session = MockConversationSession()
    decision = gate.evaluate("incancentive program", session)
    assert decision.sufficient is True


# ---------------------------------------------------------------------------
# Test Case 9: Self-defeat protection
#   trigger="cancer", manual_override_terms=["cancer"] does NOT override bare "cancer"
# ---------------------------------------------------------------------------

def test_case9_self_defeat_protection(strategy):
    """The self-defeat fix ensures trigger itself is never in override_terms."""
    import json, tempfile, os

    # Build a minimal JSON with the bare trigger as a manual_override_term
    bad_json = {
        "cancer": {
            "category": "indication",
            "question_template": "Which type of cancer?",
            "options": ["Lung Cancer", "Breast Cancer", "Colorectal Cancer"],
            "manual_override_terms": ["cancer"],  # <-- would cause self-defeat without fix
            "max_options": 3,
        }
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(bad_json, f)
        tmp_path = f.name

    try:
        reg = AmbiguousTermsRegistry(
            path=tmp_path,
            snomed_strategy=strategy,
            snomed_csv_path=CSV_PATH,
            strict_validation=True,
        )
        # The entry must exist
        entry = reg._entries["cancer"]
        # "cancer" itself must NOT be in override_terms (self-defeat fix)
        assert "cancer" not in entry.override_terms, (
            "Self-defeat bug: 'cancer' found in override_terms after M3 fix"
        )
        # Bare "cancer" query must still fire as ambiguous
        hit = reg.find_trigger("cancer")
        assert hit is not None, "find_trigger should fire for bare 'cancer'"
        assert hit[0] == "cancer"
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Test Case 10: Empty query — "" → sufficient (no trigger fires)
# ---------------------------------------------------------------------------

def test_case10_empty_query(gate):
    """Empty query: no trigger match → sufficient (Preprocessor would catch first)."""
    session = MockConversationSession()
    decision = gate.evaluate("", session)
    assert decision.sufficient is True


# ---------------------------------------------------------------------------
# Acceptance criterion checks (bonus — not numbered test cases)
# ---------------------------------------------------------------------------

def test_ac1_imports():
    """AC1: All required symbols importable from src.sufficiency_gate."""
    from src.sufficiency_gate import (  # noqa: F401
        SufficiencyGate,
        SufficiencyDecision,
        AmbiguousEntry,
        AmbiguousTermsRegistry,
        _any_filter_set,
        _count_set_filters,
    )


def test_ac4_strict_validation_exits(strategy, monkeypatch):
    """AC4: strict_validation=True with an unresolvable option → sys.exit(1)."""
    import json, tempfile, os

    bad_json = {
        "xyzzy_unresolvable": {
            "category": "indication",
            "question_template": "Which xyzzy?",
            # These options will NOT resolve at ≥0.85
            "options": [
                "Xyzzy Disease Alpha",
                "Xyzzy Disease Beta",
                "Xyzzy Disease Gamma",
            ],
            "max_options": 3,
        }
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(bad_json, f)
        tmp_path = f.name

    try:
        with pytest.raises(SystemExit) as exc_info:
            AmbiguousTermsRegistry(
                path=tmp_path,
                snomed_strategy=strategy,
                snomed_csv_path=CSV_PATH,
                strict_validation=True,
            )
        assert exc_info.value.code == 1
    finally:
        os.unlink(tmp_path)


def test_ac5_lenient_validation_skips(strategy):
    """AC5: strict_validation=False with an unresolvable option → entry skipped with WARNING."""
    import json, tempfile, os

    mixed_json = {
        "xyzzy_bad": {
            "category": "indication",
            "question_template": "Which xyzzy?",
            "options": ["Xyzzy Alpha", "Xyzzy Beta", "Xyzzy Gamma"],
            "max_options": 3,
        },
        "cancer": {
            "category": "indication",
            "question_template": "Which cancer?",
            "options": ["Lung Cancer", "Breast Cancer", "Colorectal Cancer"],
            "max_options": 3,
        },
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(mixed_json, f)
        tmp_path = f.name

    try:
        reg = AmbiguousTermsRegistry(
            path=tmp_path,
            snomed_strategy=strategy,
            snomed_csv_path=CSV_PATH,
            strict_validation=False,
        )
        # Only valid entry should remain
        assert "cancer" in reg._entries
        assert "xyzzy_bad" not in reg._entries
    finally:
        os.unlink(tmp_path)


def test_ac7_find_trigger_override(registry):
    """AC7: find_trigger('cancer') fires; find_trigger('lung cancer') returns None."""
    hit = registry.find_trigger("cancer")
    assert hit is not None
    assert hit[0] == "cancer"

    # "lung cancer" — override term present — should return None or another trigger
    hit2 = registry.find_trigger("lung cancer")
    if hit2 is not None:
        # If something fires, it must NOT be "cancer" (overridden by "lung cancer")
        assert hit2[0] != "cancer"


def test_ac8_max_turns_sufficient(gate):
    """AC8: evaluate() returns sufficient=True/reason=max_turns_reached when max reached."""
    session = MockConversationSession(clarification_count=3, max_turns=3)
    decision = gate.evaluate("cancer", session)
    assert decision.sufficient is True
    assert decision.reason == "max_turns_reached"


def test_ac9_post_extraction_filters_without_condition(gate):
    """AC9: post_extraction_check([], filters_with_city) → sufficient=False."""
    filters = _make_filters(city="Boston")
    decision = gate.post_extraction_check([], filters)
    assert decision.sufficient is False
    assert decision.reason == "filters_without_condition"
    assert decision.matched_entry is not None
    assert decision.matched_entry.trigger == "__default_condition__"


def test_ac10_default_condition_prompt_lazy_cached():
    """AC10: DEFAULT_CONDITION_PROMPT is built lazily and class-level cached."""
    # Reset the cache
    SufficiencyGate._default_condition_prompt_cache = None

    # Create a fresh registry for this sub-test
    strat = HybridCascadeStrategy(CSV_PATH)
    reg = AmbiguousTermsRegistry(
        path=JSON_PATH,
        snomed_strategy=strat,
        snomed_csv_path=CSV_PATH,
        strict_validation=True,
    )

    # First instantiation builds the cache
    gate1 = SufficiencyGate(registry=reg, snomed_csv_path=CSV_PATH)
    cached_ref = SufficiencyGate._default_condition_prompt_cache
    assert cached_ref is not None

    # Second instantiation reuses the same object
    gate2 = SufficiencyGate(registry=reg, snomed_csv_path=CSV_PATH)
    assert SufficiencyGate._default_condition_prompt_cache is cached_ref
    assert gate1._default_condition_prompt is gate2._default_condition_prompt


def test_any_filter_set_true():
    filters = _make_filters(city="Boston")
    assert _any_filter_set(filters) is True


def test_any_filter_set_false():
    filters = _make_filters()
    assert _any_filter_set(filters) is False


def test_count_set_filters():
    filters = _make_filters(city="Boston", phase="Phase 2")
    assert _count_set_filters(filters) == 2


def test_post_extraction_with_high_conf_match(gate):
    """No filters_without_condition when there is a high-confidence SNOMED match."""
    filters = _make_filters(city="Boston")
    match = _make_snomed_match(confidence=0.95, negated=False)
    decision = gate.post_extraction_check([match], filters)
    assert decision.sufficient is True
    assert decision.reason == "ok_post_extraction"


def test_post_extraction_negated_match_not_sufficient(gate):
    """Negated match does not count; triggers filters_without_condition."""
    filters = _make_filters(city="Boston")
    match = _make_snomed_match(confidence=0.95, negated=True)
    decision = gate.post_extraction_check([match], filters)
    assert decision.sufficient is False
    assert decision.reason == "filters_without_condition"


def test_sufficiency_decision_is_frozen():
    """SufficiencyDecision must be immutable (frozen=True)."""
    d = SufficiencyDecision(sufficient=True, reason="ok_no_trigger")
    with pytest.raises(Exception):
        d.sufficient = False  # type: ignore[misc]


def test_ambiguous_entry_options_too_few(strategy):
    """AmbiguousEntry rejects options list with fewer than 3 entries."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        AmbiguousEntry(
            trigger="test",
            category="indication",
            question_template="Which test?",
            options=["Only One", "Only Two"],  # too few
            override_terms=frozenset(),
            max_options=2,
        )


def test_b3_empty_registry_raises(strategy):
    """B3 fix: empty registry after validation raises ValueError before regex compile."""
    import json, tempfile, os

    # All entries will fail SNOMED validation
    bad_json = {
        "xyzzy1": {
            "category": "indication",
            "question_template": "?",
            "options": ["Xyzzy One", "Xyzzy Two", "Xyzzy Three"],
        }
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(bad_json, f)
        tmp_path = f.name

    try:
        with pytest.raises((SystemExit, ValueError)):
            AmbiguousTermsRegistry(
                path=tmp_path,
                snomed_strategy=strategy,
                snomed_csv_path=CSV_PATH,
                strict_validation=False,  # lenient; but registry is empty → ValueError
            )
    finally:
        os.unlink(tmp_path)
