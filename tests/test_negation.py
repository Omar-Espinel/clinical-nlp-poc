"""
Tests for NegationAnnotator (proposal §3.12, 9 cases).

Fixtures use SNOMEDMatch(code="X", display=term, match_type="exact",
confidence=0.99, original_text=term, span=(start, end)).
"""

import pytest

from src.snomed_search.base import SNOMEDMatch
from src.snomed_search.negation import NegationAnnotator


def make_match(display: str, span: tuple[int, int]) -> SNOMEDMatch:
    return SNOMEDMatch(
        code="X",
        display=display,
        match_type="exact",
        confidence=0.99,
        original_text=display,
        span=span,
    )


@pytest.fixture
def annotator() -> NegationAnnotator:
    return NegationAnnotator()


# ---------------------------------------------------------------------------
# 1. Plain — no negation cue present
# ---------------------------------------------------------------------------
def test_plain_not_negated(annotator):
    query = "type 2 diabetes"
    matches = [make_match("diabetes", (7, 15))]
    result = annotator.annotate(query, matches)
    assert result[0].negated is False


# ---------------------------------------------------------------------------
# 2. Pre-cue — "no history of" immediately before the match
# ---------------------------------------------------------------------------
def test_pre_cue_negated(annotator):
    query = "no history of diabetes"
    matches = [make_match("diabetes", (14, 22))]
    result = annotator.annotate(query, matches)
    assert result[0].negated is True


# ---------------------------------------------------------------------------
# 3. Pre-cue distance — "no history of" is > 5 tokens before "diabetes";
#    window truncates so diabetes is NOT negated.
# ---------------------------------------------------------------------------
def test_pre_cue_distance_not_negated(annotator):
    query = "the patient has no history of significant cardiac events including diabetes"
    # "diabetes" is token index 10; the 5-token pre-window covers tokens 5-9
    # ("of", "significant", "cardiac", "events", "including") — no cue found.
    matches = [make_match("diabetes", (67, 75))]
    result = annotator.annotate(query, matches)
    assert result[0].negated is False


# ---------------------------------------------------------------------------
# 4. Sentence boundary — period stops the scan; "Diabetes" is NOT negated
# ---------------------------------------------------------------------------
def test_sentence_boundary_not_negated(annotator):
    query = "no cancer. Diabetes trials"
    matches = [make_match("Diabetes", (11, 19))]
    result = annotator.annotate(query, matches)
    assert result[0].negated is False


# ---------------------------------------------------------------------------
# 5. Comma chain — comma does NOT stop the scan; both terms are negated
# ---------------------------------------------------------------------------
def test_comma_chain_both_negated(annotator):
    query = "no history of cardiac issues, diabetes, stroke"
    matches = [
        make_match("diabetes", (30, 38)),
        make_match("stroke", (40, 46)),
    ]
    result = annotator.annotate(query, matches)
    assert result[0].negated is True, "diabetes should be negated"
    assert result[1].negated is True, "stroke should be negated"


# ---------------------------------------------------------------------------
# 6. Post-cue — "ruled out" follows the match
# ---------------------------------------------------------------------------
def test_post_cue_negated(annotator):
    query = "diabetes ruled out"
    matches = [make_match("diabetes", (0, 8))]
    result = annotator.annotate(query, matches)
    assert result[0].negated is True


# ---------------------------------------------------------------------------
# 7. Multiple matches — "cancer" is negated; "diabetes" is not
# ---------------------------------------------------------------------------
def test_multiple_matches_selective_negation(annotator):
    query = "no cancer but has diabetes"
    matches = [
        make_match("cancer", (3, 9)),
        make_match("diabetes", (18, 26)),
    ]
    result = annotator.annotate(query, matches)
    assert result[0].negated is True,  "cancer should be negated"
    assert result[1].negated is False, "diabetes should not be negated"


# ---------------------------------------------------------------------------
# 8. Pseudo-negation — "no contraindication for" suppresses negation
# ---------------------------------------------------------------------------
def test_pseudo_negation_not_negated(annotator):
    query = "no contraindication for diabetes trials"
    matches = [make_match("diabetes", (24, 32))]
    result = annotator.annotate(query, matches)
    assert result[0].negated is False


# ---------------------------------------------------------------------------
# 9. Window edge — single isolated term; window is empty, not negated
# ---------------------------------------------------------------------------
def test_window_edge_not_negated(annotator):
    query = "diabetes"
    matches = [make_match("diabetes", (0, 8))]
    result = annotator.annotate(query, matches)
    assert result[0].negated is False


# ---------------------------------------------------------------------------
# Acceptance criteria: SNOMEDMatch validation
# ---------------------------------------------------------------------------
def test_snomed_match_span_equal_raises():
    with pytest.raises(ValueError):
        SNOMEDMatch(
            code="X", display="d", match_type="exact",
            confidence=0.9, original_text="d", span=(5, 5),
        )


def test_snomed_match_negative_start_raises():
    with pytest.raises(ValueError):
        SNOMEDMatch(
            code="X", display="d", match_type="exact",
            confidence=0.9, original_text="d", span=(-1, 3),
        )


def test_snomed_match_none_span_raises():
    with pytest.raises((ValueError, TypeError)):
        SNOMEDMatch(
            code="X", display="d", match_type="exact",
            confidence=0.9, original_text="d", span=None,
        )


# ---------------------------------------------------------------------------
# Acceptance criteria: annotate() does not mutate input
# ---------------------------------------------------------------------------
def test_annotate_does_not_mutate_input(annotator):
    query = "no diabetes"
    original = make_match("diabetes", (3, 11))
    original_negated = original.negated
    result = annotator.annotate(query, [original])
    # frozen dataclass — original object must be unchanged
    assert original.negated == original_negated
    assert result[0] is not original
    assert result[0].negated is True
