"""tests/test_conversation.py — 13 regression cases (C1-C13) for ConversationSession.

Spec: rework-nlp-proposal.md §3.3, §3.12, §14 erratum #6.

Wave 4 types (ExtractedFilters, GeoResult) are not yet implemented.
Tests use SimpleNamespace shims and mock objects where those types are needed.

HIPAA: no user_input, canonical_query, or filter values are written to log statements.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest

# Ensure project root on path for direct invocation
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.conversation import ConversationSession, Turn
from src.snomed_search.base import SNOMEDMatch
from src.sufficiency_gate import AmbiguousEntry, SufficiencyDecision


# ---------------------------------------------------------------------------
# Helpers / factories
# ---------------------------------------------------------------------------

def _make_snomed_match(
    code: str = "363346000",
    display: str = "malignant neoplasm",
    confidence: float = 0.99,
    negated: bool = False,
) -> SNOMEDMatch:
    return SNOMEDMatch(
        code=code,
        display=display,
        match_type="exact",
        confidence=confidence,
        original_text="cancer",
        span=(0, 6),
        negated=negated,
    )


def _make_ambiguous_entry(
    trigger: str = "cancer",
    triggered_by: Optional[str] = None,
) -> AmbiguousEntry:
    """Build a minimal AmbiguousEntry for use in SufficiencyDecision.matched_entry."""
    return AmbiguousEntry(
        trigger=trigger,
        category="indication",
        question_template="Which type of {trigger} research are you looking for?",
        options=["Lung Cancer", "Breast Cancer", "Colorectal Cancer"],
        override_terms=frozenset(["lung cancer", "breast cancer", "colorectal cancer"]),
        max_options=5,
    )


def _make_clarification_decision(
    triggered_by: str = "cancer",
) -> SufficiencyDecision:
    """SufficiencyDecision representing a pre-extraction ambiguous trigger."""
    entry = _make_ambiguous_entry(trigger=triggered_by)
    return SufficiencyDecision(
        sufficient=False,
        reason="ambiguous_trigger",
        triggered_by=triggered_by,
        matched_entry=entry,
    )


def _make_sufficient_decision() -> SufficiencyDecision:
    return SufficiencyDecision(sufficient=True, reason="ok_no_trigger")


def _make_max_turns_decision() -> SufficiencyDecision:
    return SufficiencyDecision(sufficient=True, reason="max_turns_reached")


def _make_filters(
    city: str = None,
    state: list = None,
    phase: str = None,
    investigator: str = None,
    site: str = None,
) -> SimpleNamespace:
    """Duck-typed ExtractedFilters shim using SimpleNamespace."""
    return SimpleNamespace(
        investigator_name=SimpleNamespace(value=investigator),
        site_name=SimpleNamespace(value=site),
        city=SimpleNamespace(value=city),
        state=SimpleNamespace(values=state or []),
        phase=SimpleNamespace(value=phase),
    )


def _make_turn(
    turn_index: int,
    user_input: str,
    canonical_query: str,
    decision: Optional[SufficiencyDecision] = None,
    filters=None,
    snomed_matches=None,
    geo=None,
    timestamp: float = None,
) -> Turn:
    return Turn(
        turn_index=turn_index,
        user_input=user_input,
        canonical_query=canonical_query,
        decision=decision,
        filters=filters,
        snomed_matches=snomed_matches or [],
        geo=geo,
        timestamp=timestamp if timestamp is not None else time.time(),
    )


# ---------------------------------------------------------------------------
# C1 — Single "cancer" input creates a clarification turn
# ---------------------------------------------------------------------------

def test_c1_single_ambiguous_turn_is_clarification():
    """C1: ['cancer'] → turn 0 is a clarification turn (decision.sufficient=False)."""
    session = ConversationSession.new()
    decision = _make_clarification_decision("cancer")
    turn = _make_turn(0, "cancer", "cancer", decision=decision)

    session.update_canonical_query("cancer")
    session.append_turn(turn)

    assert len(session.turns) == 1
    t = session.turns[0]
    assert t.decision is not None
    assert t.decision.sufficient is False
    assert t.decision.reason == "ambiguous_trigger"
    assert session.clarification_turn_count() == 1
    assert session.is_max_turns_reached() is False


# ---------------------------------------------------------------------------
# C2 — Two turns: "cancer" then "Lung Cancer" — substitution
# ---------------------------------------------------------------------------

def test_c2_substitution_cancer_to_lung_cancer():
    """C2: canonical 'cancer' + answer 'Lung Cancer' → canonical becomes 'Lung Cancer'."""
    session = ConversationSession.new()

    # Turn 0: user typed "cancer" → clarification
    decision0 = _make_clarification_decision("cancer")
    turn0 = _make_turn(0, "cancer", "cancer", decision=decision0)
    session.set_canonical_query("cancer")
    session.append_turn(turn0)

    # Now user answers "Lung Cancer"
    new_canonical = session.compute_canonical_query("Lung Cancer")
    assert new_canonical == "Lung Cancer", (
        f"Expected substitution 'Lung Cancer', got '{new_canonical}'"
    )

    session.set_canonical_query(new_canonical)
    assert session.canonical_query == "Lung Cancer"


# ---------------------------------------------------------------------------
# C3 — "cancer in Boston phase 3" then "Lung Cancer" — targeted substitution
# ---------------------------------------------------------------------------

def test_c3_substitution_preserves_context():
    """C3: canonical 'cancer in Boston phase 3' + answer 'Lung Cancer'
    → 'Lung Cancer in Boston phase 3'."""
    session = ConversationSession.new()

    original_query = "cancer in Boston phase 3"
    decision0 = _make_clarification_decision("cancer")
    turn0 = _make_turn(0, original_query, original_query, decision=decision0)
    session.set_canonical_query(original_query)
    session.append_turn(turn0)

    new_canonical = session.compute_canonical_query("Lung Cancer")
    assert new_canonical == "Lung Cancer in Boston phase 3", (
        f"Expected 'Lung Cancer in Boston phase 3', got '{new_canonical}'"
    )

    session.set_canonical_query(new_canonical)
    assert session.canonical_query == "Lung Cancer in Boston phase 3"


# ---------------------------------------------------------------------------
# C4 — 4 junk turns → is_max_turns_reached() at turn 4
# ---------------------------------------------------------------------------

def test_c4_max_turns_reached_at_4_clarifications():
    """C4: 4 clarification turns → is_max_turns_reached() True at turn index 3."""
    session = ConversationSession.new()

    for i in range(4):
        decision = _make_clarification_decision("cancer")
        turn = _make_turn(i, "junk", "junk", decision=decision)
        session.append_turn(turn)

    assert session.clarification_turn_count() == 4
    # max_clarification_turns default = 3, so 4 >= 3 → True
    assert session.is_max_turns_reached() is True


# ---------------------------------------------------------------------------
# C5 — "lung cancer" (override fires) → sufficient on first turn
# ---------------------------------------------------------------------------

def test_c5_lung_cancer_is_sufficient():
    """C5: 'lung cancer' override present → first turn is sufficient (no clarification)."""
    session = ConversationSession.new()

    decision = _make_sufficient_decision()
    turn = _make_turn(0, "lung cancer", "lung cancer", decision=decision)
    session.set_canonical_query("lung cancer")
    session.append_turn(turn)

    assert session.clarification_turn_count() == 0
    assert session.is_max_turns_reached() is False
    assert session.turns[0].decision.sufficient is True


# ---------------------------------------------------------------------------
# C6 — Preprocessor block: session retains prior turn history
# ---------------------------------------------------------------------------

def test_c6_preprocessor_blocked_retains_history():
    """C6: After a preprocessor block, session retains turn 0 history.

    Preprocessor is Wave 5.  We simulate the block by simply NOT calling
    append_turn() for the blocked input, which is what run_with_session() will do.
    """
    session = ConversationSession.new()

    # Turn 0: valid clarification
    decision0 = _make_clarification_decision("cancer")
    turn0 = _make_turn(0, "cancer", "cancer", decision=decision0)
    session.set_canonical_query("cancer")
    session.append_turn(turn0)

    history_before = len(session.turns)
    canonical_before = session.canonical_query

    # Simulate preprocessor block: compute is called (pure), assert_safe raises,
    # set_canonical_query and append_turn are never called.
    _ = session.compute_canonical_query("<malicious payload>")
    # assert_safe() would raise here — we skip it in this unit test
    # session.set_canonical_query(...) NOT called
    # session.append_turn(...) NOT called

    # Session state is unchanged
    assert len(session.turns) == history_before
    assert session.canonical_query == canonical_before
    assert session.clarification_turn_count() == 1


# ---------------------------------------------------------------------------
# C7 — Multiple identical clarification answers; turn 4 forces sufficient
# ---------------------------------------------------------------------------

def test_c7_repeated_answers_eventually_max_turns():
    """C7: ['cancer', 'Lung Cancer', 'Lung Cancer', 'Lung Cancer'] → turn 4 via max-turns."""
    session = ConversationSession.new()

    # Turn 0: clarification
    d0 = _make_clarification_decision("cancer")
    t0 = _make_turn(0, "cancer", "cancer", decision=d0)
    session.set_canonical_query("cancer")
    session.append_turn(t0)

    # Turns 1, 2 also clarifications (gate still fires on merged query)
    for i in range(1, 3):
        canonical = session.compute_canonical_query("Lung Cancer")
        session.set_canonical_query(canonical)
        di = _make_clarification_decision("cancer")
        ti = _make_turn(i, "Lung Cancer", canonical, decision=di)
        session.append_turn(ti)

    # At this point 3 clarification turns → max reached
    assert session.is_max_turns_reached() is True

    # Turn 3: escape valve — max_turns_reached → sufficient
    d3 = _make_max_turns_decision()
    canonical3 = session.compute_canonical_query("Lung Cancer")
    session.set_canonical_query(canonical3)
    t3 = _make_turn(3, "Lung Cancer", canonical3, decision=d3)
    session.append_turn(t3)

    assert len(session.turns) == 4
    assert session.turns[3].decision.sufficient is True
    assert session.turns[3].decision.reason == "max_turns_reached"


# ---------------------------------------------------------------------------
# C8 — Round-trip: from_dict(to_dict()) equals original session
# ---------------------------------------------------------------------------

def test_c8_round_trip_serialization():
    """C8: from_dict(s.to_dict()) produces a session equal to the original."""
    session = ConversationSession.new()

    decision = _make_clarification_decision("cancer")
    turn = _make_turn(0, "cancer", "cancer", decision=decision)
    session.set_canonical_query("cancer")
    session.append_turn(turn)

    serialized = session.to_dict()
    restored = ConversationSession.from_dict(serialized)

    # Core identity equality
    assert restored.session_id == session.session_id
    assert restored.canonical_query == session.canonical_query
    assert restored.max_clarification_turns == session.max_clarification_turns
    assert restored.created_at == session.created_at
    assert len(restored.turns) == len(session.turns)

    t_orig = session.turns[0]
    t_rest = restored.turns[0]
    assert t_rest.turn_index == t_orig.turn_index
    assert t_rest.user_input == t_orig.user_input
    assert t_rest.canonical_query == t_orig.canonical_query
    assert t_rest.decision.sufficient == t_orig.decision.sufficient
    assert t_rest.decision.reason == t_orig.decision.reason
    assert t_rest.decision.triggered_by == t_orig.decision.triggered_by


# ---------------------------------------------------------------------------
# C9 — Parallel sessions: no state leak
# ---------------------------------------------------------------------------

def test_c9_parallel_sessions_no_state_leak():
    """C9: Two sessions submitting 'cancer' in parallel share no state."""
    session_a = ConversationSession.new()
    session_b = ConversationSession.new()

    assert session_a.session_id != session_b.session_id, "session_ids must be unique"

    decision_a = _make_clarification_decision("cancer")
    turn_a = _make_turn(0, "cancer", "cancer", decision=decision_a)
    session_a.set_canonical_query("cancer")
    session_a.append_turn(turn_a)

    # session_b remains clean
    assert len(session_b.turns) == 0
    assert session_b.canonical_query == ""
    assert session_b.clarification_turn_count() == 0

    # session_a unaffected by any changes to session_b
    session_b.set_canonical_query("diabetes")
    assert session_a.canonical_query == "cancer"


# ---------------------------------------------------------------------------
# C10 — Post-extraction safety: SufficiencyDecision shape check
# ---------------------------------------------------------------------------

def test_c10_post_extraction_decision_shape():
    """C10: Post-extraction safety decision has correct shape.

    Full pipeline integration requires Wave 4-5 components.  This test confirms
    that a SufficiencyDecision with reason='filters_without_condition' and
    sufficient=False is structurally correct and can be stored in a Turn.
    """
    entry = _make_ambiguous_entry("__default_condition__")
    decision = SufficiencyDecision(
        sufficient=False,
        reason="filters_without_condition",
        triggered_by=None,
        matched_entry=entry,
    )

    assert decision.sufficient is False
    assert decision.reason == "filters_without_condition"
    assert decision.triggered_by is None
    assert decision.matched_entry is not None
    assert "Lung Cancer" in decision.matched_entry.options

    # Confirm it can be stored in a Turn without error
    turn = _make_turn(0, "What's at Mayo?", "What's at Mayo?", decision=decision)
    assert turn.decision.reason == "filters_without_condition"


# ---------------------------------------------------------------------------
# C11 — "cancers" (plural) does NOT match trigger "cancer" (whole-word \b)
# ---------------------------------------------------------------------------

def test_c11_plural_does_not_substitute():
    """C11: trigger='cancer' does NOT match 'cancers' due to whole-word boundary.

    This is a documented known false negative (§13 open risks).
    When the trigger doesn't appear in canonical_query, compute falls back to append.
    """
    session = ConversationSession.new()

    # Set up session with canonical_query = "cancers" and trigger = "cancer"
    decision0 = _make_clarification_decision("cancer")
    turn0 = _make_turn(0, "cancers", "cancers", decision=decision0)
    session.set_canonical_query("cancers")
    session.append_turn(turn0)

    # compute_canonical_query should fall back to append (not substitute)
    new_canonical = session.compute_canonical_query("Lung Cancer")

    # \bcancer\b does NOT match "cancers" → append fallback
    assert new_canonical == "cancers Lung Cancer", (
        f"Expected append fallback 'cancers Lung Cancer', got '{new_canonical}'"
    )


# ---------------------------------------------------------------------------
# C12 — from_dict with corrupted data, on_error="new_session"
# ---------------------------------------------------------------------------

def test_c12_corrupted_dict_returns_fresh_session(caplog):
    """C12: from_dict(corrupted_dict, on_error='new_session') → fresh session + WARN log."""
    corrupted = {
        "session_id": "old-session",
        "turns": [{"turn_index": "not-an-int", "bad_field": True}],  # invalid schema
        "canonical_query": "cancer",
        "max_clarification_turns": 3,
        "created_at": 1234567890.0,
    }

    with caplog.at_level(logging.WARNING, logger="src.conversation"):
        result = ConversationSession.from_dict(corrupted, on_error="new_session")

    # Fresh session returned
    assert isinstance(result, ConversationSession)
    assert result.turns == []
    assert result.canonical_query == ""
    assert len(result.session_id) == 32  # uuid4().hex

    # WARN logged
    assert any("corrupted state" in record.message for record in caplog.records), (
        "Expected a WARN log message about corrupted state"
    )


def test_c12b_corrupted_dict_raises_by_default():
    """C12b: from_dict(corrupted_dict) with default on_error='raise' raises ValidationError."""
    from pydantic import ValidationError as PydanticValidationError

    corrupted = {
        "session_id": 12345,  # should be str
        "turns": "not-a-list",
        "created_at": "not-a-float",
    }

    with pytest.raises(PydanticValidationError):
        ConversationSession.from_dict(corrupted, on_error="raise")


# ---------------------------------------------------------------------------
# C13 — Injection-pattern substitution: compute is pure, no validation
# ---------------------------------------------------------------------------

def test_c13_injection_pattern_compute_is_pure():
    """C13: compute_canonical_query does NOT validate for injection patterns.

    The pipeline's assert_safe() (Wave 5) is responsible for validation.
    compute_canonical_query merges strings correctly regardless of content —
    the assert_safe guard is in run_with_session(), not here.

    This test confirms:
    1. compute_canonical_query returns the merged canonical correctly.
    2. self.canonical_query is NOT mutated by compute.
    3. Only set_canonical_query (called after assert_safe) mutates state.
    """
    session = ConversationSession.new()

    # Turn 0: legitimate "cancer" clarification
    decision0 = _make_clarification_decision("cancer")
    turn0 = _make_turn(0, "cancer", "cancer", decision=decision0)
    session.set_canonical_query("cancer")
    session.append_turn(turn0)

    canonical_before = session.canonical_query  # "cancer"

    # Simulate an adversarial user_input that would normally be caught by assert_safe
    adversarial_input = "Lung Cancer; DROP TABLE trials;"

    # compute_canonical_query is PURE — returns merged result without mutating
    merged = session.compute_canonical_query(adversarial_input)

    # Merged canonical is computed correctly (substitution — trigger "cancer" present)
    assert "Lung Cancer" in merged
    # Original canonical_query is NOT mutated
    assert session.canonical_query == canonical_before, (
        "compute_canonical_query must not mutate self.canonical_query"
    )

    # Only set_canonical_query writes the value (called after assert_safe passes)
    session.set_canonical_query(merged)
    assert session.canonical_query == merged


# ---------------------------------------------------------------------------
# Acceptance criteria direct checks
# ---------------------------------------------------------------------------

def test_ac1_import():
    """AC1: from src.conversation import ConversationSession, Turn succeeds."""
    from src.conversation import ConversationSession as CS, Turn as T  # noqa: F401
    assert CS is not None
    assert T is not None


def test_ac2_new_session_shape():
    """AC2: new() returns correct defaults."""
    s = ConversationSession.new()
    assert len(s.session_id) == 32           # uuid4().hex is 32 hex chars
    assert s.turns == []
    assert s.canonical_query == ""
    assert s.created_at > 0


def test_ac3_compute_is_pure():
    """AC3: compute_canonical_query does NOT modify self.canonical_query."""
    session = ConversationSession.new()
    decision = _make_clarification_decision("cancer")
    turn = _make_turn(0, "cancer", "cancer", decision=decision)
    session.set_canonical_query("cancer")
    session.append_turn(turn)

    before = session.canonical_query
    result = session.compute_canonical_query("Lung Cancer")

    assert result == "Lung Cancer"
    assert session.canonical_query == before, "compute must not mutate canonical_query"


def test_ac4_set_canonical_query():
    """AC4: set_canonical_query writes self.canonical_query."""
    session = ConversationSession.new()
    session.set_canonical_query("lung cancer trial")
    assert session.canonical_query == "lung cancer trial"


def test_ac5_update_canonical_query_does_both():
    """AC5: update_canonical_query computes and sets."""
    session = ConversationSession.new()
    decision = _make_clarification_decision("cancer")
    turn = _make_turn(0, "cancer", "cancer", decision=decision)
    session.set_canonical_query("cancer")
    session.append_turn(turn)

    returned = session.update_canonical_query("Lung Cancer")
    assert returned == "Lung Cancer"
    assert session.canonical_query == "Lung Cancer"


def test_ac8_clarification_turn_count():
    """AC8: clarification_turn_count counts only turns where decision.sufficient==False."""
    session = ConversationSession.new()

    t0 = _make_turn(0, "cancer", "cancer", decision=_make_clarification_decision())
    t1 = _make_turn(1, "Lung Cancer", "Lung Cancer", decision=_make_sufficient_decision())
    session.append_turn(t0)
    session.append_turn(t1)

    assert session.clarification_turn_count() == 1


def test_ac9_is_max_turns_reached_exact_threshold():
    """AC9: is_max_turns_reached() returns True at exactly max_clarification_turns."""
    session = ConversationSession.new()
    assert session.max_clarification_turns == 3

    for i in range(3):
        t = _make_turn(i, "x", "x", decision=_make_clarification_decision())
        session.append_turn(t)

    assert session.is_max_turns_reached() is True

    # One short: 2 clarification turns → not reached
    session2 = ConversationSession.new()
    for i in range(2):
        t = _make_turn(i, "x", "x", decision=_make_clarification_decision())
        session2.append_turn(t)
    assert session2.is_max_turns_reached() is False


def test_ac13_summary_for_logging_safe_fields_only():
    """AC13: summary_for_logging returns only safe fields; no canonical_query or user_input."""
    session = ConversationSession.new()
    decision = _make_clarification_decision("cancer")
    turn = _make_turn(0, "cancer", "cancer", decision=decision)
    session.set_canonical_query("cancer")
    session.append_turn(turn)

    summary = session.summary_for_logging()

    assert set(summary.keys()) == {
        "session_id",
        "turn_count",
        "clarification_count",
        "max_turns",
        "created_at",
    }
    assert "canonical_query" not in summary
    assert "user_input" not in summary
    assert "turns" not in summary
    assert summary["turn_count"] == 1
    assert summary["clarification_count"] == 1
    assert summary["max_turns"] == 3
