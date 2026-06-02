import pytest
from src.autocomplete.segmenter import QuerySegmenter, ContextType

GEO_KEYS = frozenset({
    "boston", "new york city", "california", "bay area",
    "los angeles", "chicago", "massachusetts",
})


@pytest.fixture
def segmenter():
    return QuerySegmenter(GEO_KEYS)


def test_phase_prefix_extracted(segmenter):
    result = segmenter.segment("Phase 3 diab")
    assert "Phase 3" in result.context_tokens
    assert result.active_prefix == "diab"
    assert ContextType.PHASE in result.context_types


def test_bare_prefix_no_context(segmenter):
    result = segmenter.segment("diabetes")
    assert result.context_tokens == []
    assert result.active_prefix == "diabetes"


def test_geo_prefix_extracted(segmenter):
    result = segmenter.segment("boston diab")
    assert any("boston" in t.lower() for t in result.context_tokens)
    assert result.active_prefix == "diab"
    assert ContextType.GEO in result.context_types


def test_short_prefix_returns_empty_active(segmenter):
    result = segmenter.segment("di")
    assert result.active_prefix == ""


def test_phase_only_no_trailing_prefix(segmenter):
    result = segmenter.segment("Phase 3")
    assert result.active_prefix == "" or len(result.active_prefix) < 3


def test_case_insensitive_phase(segmenter):
    result = segmenter.segment("phase 2 card")
    assert result.active_prefix == "card"
    assert ContextType.PHASE in result.context_types


def test_two_char_geo_keys_filtered():
    seg = QuerySegmenter(frozenset({"ca", "ny", "boston"}))
    result = seg.segment("ca diab")
    assert ContextType.GEO not in result.context_types


def test_multiword_geo_extracted(segmenter):
    result = segmenter.segment("bay area diab")
    assert ContextType.GEO in result.context_types
    assert result.active_prefix == "diab"
