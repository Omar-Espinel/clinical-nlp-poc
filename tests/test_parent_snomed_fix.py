"""tests/test_parent_snomed_fix.py — Tests for the parent-SNOMED resolution path.

Covers:
  - TriggerResult.use_parent_snomed flag propagation
  - ok_parent_snomed_used reason in SufficiencyGate.evaluate()
  - backward-compatible tuple unpacking on TriggerResult
  - SufficiencyDecision validation with the new reason
  - Existing clarification behavior preserved for entries without allow_parent_search

Uses temp JSON files (same pattern as test_case9_self_defeat_protection) when the
production ambiguous_terms.json does not have allow_parent_search set.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

# Ensure project root is on the path for direct invocation
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.snomed_search.hybrid_cascade import HybridCascadeStrategy
from src.snomed_search.base import SNOMEDMatch
from src.sufficiency_gate import (
    AmbiguousEntry,
    AmbiguousTermsRegistry,
    SufficiencyDecision,
    SufficiencyGate,
    TriggerResult,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CSV_PATH  = str(Path(__file__).parent.parent / "data" / "snomed_clinical_trials.csv")
JSON_PATH = str(Path(__file__).parent.parent / "data" / "ambiguous_terms.json")

# Parent SNOMED code for "malignant neoplasm" (broad cancer concept).
CANCER_PARENT_CODE = "363346000"

# ---------------------------------------------------------------------------
# Shared fixtures (module-scoped — HybridCascadeStrategy is expensive to build)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def strategy():
    """Real HybridCascadeStrategy — built once for the module."""
    return HybridCascadeStrategy(CSV_PATH)


@pytest.fixture(scope="module")
def real_registry(strategy):
    """Real AmbiguousTermsRegistry loaded from production ambiguous_terms.json."""
    return AmbiguousTermsRegistry(
        path=JSON_PATH,
        snomed_strategy=strategy,
        snomed_csv_path=CSV_PATH,
        strict_validation=True,
    )


@pytest.fixture(scope="module")
def real_gate(real_registry):
    """SufficiencyGate backed by the real production registry."""
    SufficiencyGate._default_condition_prompt_cache = None
    return SufficiencyGate(registry=real_registry, snomed_csv_path=CSV_PATH)


# ---------------------------------------------------------------------------
# Minimal MockConversationSession (mirrors test_sufficiency_gate.py pattern)
# ---------------------------------------------------------------------------

class MockConversationSession:
    """Minimal session shim for gate tests."""

    def __init__(self, session_id: str = "test-psf", clarification_count: int = 0,
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
# Helpers: build a minimal temporary registry JSON with allow_parent_search
# ---------------------------------------------------------------------------

def _write_temp_registry(strategy, entries: dict, strict: bool = True) -> AmbiguousTermsRegistry:
    """Write entries dict to a temp JSON file and return an AmbiguousTermsRegistry.

    Uses the same create-temp-file pattern as test_case9_self_defeat_protection.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(entries, f)
        tmp_path = f.name
    try:
        reg = AmbiguousTermsRegistry(
            path=tmp_path,
            snomed_strategy=strategy,
            snomed_csv_path=CSV_PATH,
            strict_validation=strict,
        )
    finally:
        os.unlink(tmp_path)
    return reg


# ---------------------------------------------------------------------------
# Test 1 — "cancer trials in Boston" with allow_parent_search=True
#           should return sufficient=True, reason="ok_parent_snomed_used"
# ---------------------------------------------------------------------------

def test_cancer_alone_returns_search_not_clarification(strategy):
    """Query with bare 'cancer' and allow_parent_search=True skips clarification.

    Uses a minimal temp registry with allow_parent_search set, since the
    production ambiguous_terms.json intentionally omits that flag for now.
    """
    reg = _write_temp_registry(strategy, {
        "cancer": {
            "category": "indication",
            "question_template": "Which type of {trigger} are you looking for?",
            "options": ["Lung Cancer", "Breast Cancer", "Colorectal Cancer"],
            "manual_override_terms": [],
            "max_options": 3,
            "allow_parent_search": True,
            "snomed_parent_code": CANCER_PARENT_CODE,
        }
    })
    gate = SufficiencyGate(registry=reg, snomed_csv_path=CSV_PATH)
    session = MockConversationSession()
    decision = gate.evaluate("cancer trials in Boston", session)
    assert decision.sufficient is True
    assert decision.reason == "ok_parent_snomed_used"


# ---------------------------------------------------------------------------
# Test 2 — "lung cancer trials in Boston" must NOT trigger clarification
#           (override_terms fires for "lung cancer" — existing behaviour)
# ---------------------------------------------------------------------------

def test_lung_cancer_still_bypasses_trigger(real_gate):
    """Specific compound term 'lung cancer' suppresses the bare 'cancer' trigger.

    The override mechanism predates parent-SNOMED and must be unaffected.
    """
    session = MockConversationSession()
    decision = real_gate.evaluate("lung cancer trials in Boston", session)
    assert decision.sufficient is True
    assert decision.reason != "ambiguous_trigger", (
        "'lung cancer' override should suppress the bare cancer trigger"
    )


# ---------------------------------------------------------------------------
# Test 3 — parent_match is injected into snomed_terms for "cancer"
#           Tested via gate decision + parent_match construction logic.
#           Full pipeline integration is validated indirectly via test 1 path.
# ---------------------------------------------------------------------------

def test_cancer_parent_snomed_in_decision(strategy):
    """When allow_parent_search=True, matched_entry carries the parent SNOMED code."""
    reg = _write_temp_registry(strategy, {
        "cancer": {
            "category": "indication",
            "question_template": "Which type of {trigger} are you looking for?",
            "options": ["Lung Cancer", "Breast Cancer", "Colorectal Cancer"],
            "manual_override_terms": [],
            "max_options": 3,
            "allow_parent_search": True,
            "snomed_parent_code": CANCER_PARENT_CODE,
        }
    })
    gate = SufficiencyGate(registry=reg, snomed_csv_path=CSV_PATH)
    session = MockConversationSession()
    decision = gate.evaluate("cancer", session)
    assert decision.reason == "ok_parent_snomed_used"
    assert decision.matched_entry is not None
    assert decision.matched_entry.snomed_parent_code == CANCER_PARENT_CODE


# ---------------------------------------------------------------------------
# Test 4 — "diabetes Phase 2 trials" with allow_parent_search=True
# ---------------------------------------------------------------------------

def test_diabetes_alone_returns_search(strategy):
    """Bare 'diabetes' with allow_parent_search=True resolves via parent SNOMED."""
    # SNOMED parent code for "Diabetes Mellitus" (broad concept)
    diabetes_parent_code = "73211009"
    reg = _write_temp_registry(strategy, {
        "diabetes": {
            "category": "indication",
            "question_template": "Which type of {trigger} are you looking for?",
            "options": ["Type 1 Diabetes Mellitus", "Type 2 Diabetes Mellitus", "Diabetes Mellitus"],
            "manual_override_terms": [],
            "max_options": 3,
            "allow_parent_search": True,
            "snomed_parent_code": diabetes_parent_code,
        }
    })
    gate = SufficiencyGate(registry=reg, snomed_csv_path=CSV_PATH)
    session = MockConversationSession()
    decision = gate.evaluate("diabetes Phase 2 trials", session)
    assert decision.sufficient is True
    assert decision.reason == "ok_parent_snomed_used"


# ---------------------------------------------------------------------------
# Test 5 — Real "depression" entry options match production data
# ---------------------------------------------------------------------------

def test_depression_options_are_correct(real_registry):
    """The 'depression' entry must have clinically correct psychiatric options.

    Prior to this fix the entry had neurological conditions (Alzheimer Disease,
    Mild Cognitive Impairment, Multiple Sclerosis) — clearly wrong. The fix
    replaces them with actual depression-spectrum conditions.
    """
    entry = real_registry._entries.get("depression")
    assert entry is not None, "'depression' entry missing from registry"
    assert "Major Depressive Disorder" in entry.options
    assert "Alzheimer Disease" not in entry.options


# ---------------------------------------------------------------------------
# Test 6 — "brain trials" still clarifies (no allow_parent_search on brain entry)
# ---------------------------------------------------------------------------

def test_brain_alone_still_proceeds(real_gate):
    """Bare 'brain' fires a trigger but Phase 2 allows it to proceed (sufficient=True).

    Phase 2 migration: previously entries without allow_parent_search returned
    ambiguous_trigger (sufficient=False). Now all triggered entries return ok_no_trigger
    (sufficient=True) so extraction proceeds without clarification.
    """
    session = MockConversationSession()
    decision = real_gate.evaluate("brain trials", session)
    assert decision.sufficient is True
    assert decision.reason in ("ok_no_trigger", "ok_parent_snomed_used", "max_turns_reached")


# ---------------------------------------------------------------------------
# Test 7 — "blood disease trials" still clarifies
# ---------------------------------------------------------------------------

def test_blood_alone_still_proceeds(real_gate):
    """Bare 'blood' fires a trigger but Phase 2 allows it to proceed (sufficient=True).

    Phase 2 migration: previously returned ambiguous_trigger (sufficient=False).
    Now returns ok_no_trigger or ok_parent_snomed_used (sufficient=True).
    """
    session = MockConversationSession()
    decision = real_gate.evaluate("blood disease trials", session)
    assert decision.sufficient is True
    assert decision.reason in ("ok_no_trigger", "ok_parent_snomed_used", "max_turns_reached")


# ---------------------------------------------------------------------------
# Test 8 — "ok_parent_snomed_used" is a valid SufficiencyDecision reason
# ---------------------------------------------------------------------------

def test_ok_parent_snomed_used_is_valid_reason():
    """SufficiencyDecision accepts 'ok_parent_snomed_used' without ValidationError."""
    # Should not raise
    decision = SufficiencyDecision(
        sufficient=True,
        reason="ok_parent_snomed_used",
        triggered_by=None,
        matched_entry=None,
    )
    assert decision.reason == "ok_parent_snomed_used"
    assert decision.sufficient is True


# ---------------------------------------------------------------------------
# Test 9 — TriggerResult.use_parent_snomed flag propagates correctly
# ---------------------------------------------------------------------------

def test_trigger_result_dataclass_use_parent_flag(strategy):
    """find_trigger() returns use_parent_snomed=True when allow_parent_search is set."""
    # Registry with allow_parent_search=True
    reg_with_parent = _write_temp_registry(strategy, {
        "cancer": {
            "category": "indication",
            "question_template": "Which type of {trigger} are you looking for?",
            "options": ["Lung Cancer", "Breast Cancer", "Colorectal Cancer"],
            "manual_override_terms": [],
            "max_options": 3,
            "allow_parent_search": True,
            "snomed_parent_code": CANCER_PARENT_CODE,
        }
    })
    hit = reg_with_parent.find_trigger("cancer")
    assert hit is not None
    assert isinstance(hit, TriggerResult)
    assert hit.use_parent_snomed is True
    assert hit.entry.snomed_parent_code == CANCER_PARENT_CODE

    # Registry with allow_parent_search=False (default)
    reg_without_parent = _write_temp_registry(strategy, {
        "cancer": {
            "category": "indication",
            "question_template": "Which type of {trigger} are you looking for?",
            "options": ["Lung Cancer", "Breast Cancer", "Colorectal Cancer"],
            "manual_override_terms": [],
            "max_options": 3,
            "allow_parent_search": False,
        }
    })
    hit2 = reg_without_parent.find_trigger("cancer")
    assert hit2 is not None
    assert isinstance(hit2, TriggerResult)
    assert hit2.use_parent_snomed is False


# ---------------------------------------------------------------------------
# Test 10 — "breast cancer" returns None from find_trigger (override fires)
# ---------------------------------------------------------------------------

def test_find_trigger_returns_none_for_specific_term(real_registry):
    """Compound term 'breast cancer' is in override_terms, suppressing 'cancer' trigger.

    This verifies that existing override-suppression behavior is intact after the
    find_trigger() return-type change from tuple to TriggerResult.
    """
    hit = real_registry.find_trigger("breast cancer")
    if hit is not None:
        # If something fires, it must NOT be the bare 'cancer' trigger
        # (breast cancer is an override term for cancer)
        assert hit.trigger != "cancer", (
            "'breast cancer' should suppress the 'cancer' trigger via override_terms"
        )
