import pytest
from src.autocomplete.index import AutocompleteSuggestion
from src.autocomplete.ranker import deduplicate, rank_and_trim
from src.autocomplete.segmenter import SegmentResult, ContextType


def _make_suggestion(display, snomed_code=None, raw_score=0.90, match_type="prefix", category="snomed"):
    return AutocompleteSuggestion(
        display=display,
        snomed_code=snomed_code,
        category=category,
        raw_score=raw_score,
        match_type=match_type,
    )


def _make_segment(prefix="diab", context_types=None):
    return SegmentResult(
        context_tokens=[],
        active_prefix=prefix,
        context_types=context_types or [],
        last_token_complete=False,
    )


def test_dedup_same_snomed_code_keeps_highest_score():
    suggestions = [
        _make_suggestion("diabetes mellitus", snomed_code="73211009", raw_score=0.95),
        _make_suggestion("diabetes mellitus", snomed_code="73211009", raw_score=0.80),
    ]
    result = deduplicate(suggestions)
    assert len(result) == 1
    assert result[0].raw_score == 0.95


def test_dedup_same_display_keeps_highest_score():
    suggestions = [
        _make_suggestion("Hypertension", snomed_code="A1", raw_score=0.90),
        _make_suggestion("hypertension", snomed_code="A2", raw_score=0.70),
    ]
    result = deduplicate(suggestions)
    assert len(result) == 1


def test_hierarchy_collapse_removes_parent():
    parent = _make_suggestion("diabetes mellitus", snomed_code="73211009", raw_score=0.85)
    child = _make_suggestion("type 2 diabetes mellitus", snomed_code="44054006", raw_score=0.87)
    segment = _make_segment(prefix="type 2 diab")
    result = rank_and_trim([parent, child], segment, limit=7)
    displays = [s.display for s in result]
    assert "type 2 diabetes mellitus" in displays
    assert "diabetes mellitus" not in displays


def test_rank_and_trim_caps_at_limit():
    suggestions = [_make_suggestion(f"term {i}", snomed_code=str(i)) for i in range(20)]
    result = rank_and_trim(suggestions, _make_segment(), limit=7)
    assert len(result) <= 7


def test_phase_gets_context_bonus():
    phase_sug = _make_suggestion("Phase 3", snomed_code=None, category="phase", raw_score=0.90)
    snomed_sug = _make_suggestion("diabetes mellitus", snomed_code="73211009", category="snomed", raw_score=0.90)
    segment = _make_segment(prefix="Pha", context_types=[ContextType.PHASE])
    result = rank_and_trim([phase_sug, snomed_sug], segment, limit=7)
    assert result[0].category == "phase"


def test_tier_weights_prefix_beats_fuzzy():
    prefix_sug = _make_suggestion("diabetes mellitus", snomed_code="A", raw_score=0.85, match_type="prefix")
    fuzzy_sug = _make_suggestion("diabetes insipidus", snomed_code="B", raw_score=0.85, match_type="fuzzy")
    result = rank_and_trim([fuzzy_sug, prefix_sug], _make_segment(), limit=7)
    assert result[0].match_type == "prefix"


def test_specificity_boost_long_prefix():
    specific = _make_suggestion("type 2 diabetes mellitus", snomed_code="A", raw_score=0.85)
    generic = _make_suggestion("diabetes mellitus", snomed_code="B", raw_score=0.85)
    segment = _make_segment(prefix="type 2 diabetes")
    result = rank_and_trim([generic, specific], segment, limit=7)
    assert result[0].display == "type 2 diabetes mellitus"


def test_context_bonus_snomed_boosts_phase():
    from src.autocomplete.ranker import _context_bonus
    assert _context_bonus("phase", [ContextType.SNOMED]) == 1.15


def test_context_bonus_snomed_boosts_geo():
    from src.autocomplete.ranker import _context_bonus
    assert _context_bonus("geo_city", [ContextType.SNOMED]) == 1.10


def test_context_bonus_snomed_does_not_boost_snomed():
    from src.autocomplete.ranker import _context_bonus
    assert _context_bonus("snomed", [ContextType.SNOMED]) == 1.0


def test_context_bonus_phase_context_boosts_phase():
    from src.autocomplete.ranker import _context_bonus
    assert _context_bonus("phase", [ContextType.PHASE]) == 1.05


def test_context_bonus_no_context_returns_one():
    from src.autocomplete.ranker import _context_bonus
    assert _context_bonus("snomed", []) == 1.0
