import numpy as np
import pytest
from unittest.mock import MagicMock
from src.autocomplete.index import AutocompleteIndex, AutocompleteSuggestion
from src.snomed_search.base import SNOMEDMatch


def _make_mock_strategy(semantic_available=False):
    strategy = MagicMock()
    strategy._exact_index = {
        "diabetes mellitus": {"concept_id": "73211009", "preferred_term": "diabetes mellitus"},
        "myocardial infarction": {"concept_id": "22298006", "preferred_term": "myocardial infarction"},
        "hypertension": {"concept_id": "38341003", "preferred_term": "hypertension"},
        "lung cancer": {"concept_id": "363358000", "preferred_term": "lung cancer"},
    }
    strategy._synonym_index = {
        "heart attack": {"concept_id": "22298006", "preferred_term": "myocardial infarction"},
    }
    strategy._alias_dict = {
        "heart attack": "myocardial infarction",
        "high blood pressure": "hypertension",
    }
    strategy._np_embeddings = None
    strategy._np_terms = []
    strategy._collection = None
    strategy._embedder = None
    strategy.semantic_available = semantic_available
    return strategy


_GEO_DATA = {
    "cities": {
        "boston": {"canonical": "Boston", "state": "Massachusetts", "country": "US"},
        "new york city": {"canonical": "New York City", "state": "New York", "country": "US"},
    },
    "states": {
        "massachusetts": "Massachusetts",
        "california": "California",
        "cal": "California",
    },
    "regions": {
        "bay area": {"type": "region", "primary_city": "San Francisco", "state": "California"},
    },
}


@pytest.fixture
def index():
    return AutocompleteIndex(_make_mock_strategy(), _GEO_DATA)


def test_prefix_search_exact_match(index):
    results = index.prefix_search("diabetes", limit=10)
    assert any("diabetes mellitus" in r.display for r in results)


def test_prefix_search_alias_resolves_to_preferred(index):
    results = index.prefix_search("heart attack", limit=10)
    assert any("myocardial infarction" in r.display for r in results)


def test_prefix_search_geo_city(index):
    results = index.prefix_search("bos", limit=10)
    assert any("Boston" in r.display for r in results)


def test_prefix_search_phase(index):
    results = index.prefix_search("Phase", limit=10)
    displays = [r.display for r in results]
    assert "Phase 1" in displays
    assert "Phase 2" in displays
    assert "Phase 3" in displays


def test_fuzzy_search_typo(index):
    results = index.fuzzy_search("diabtes", limit=10)
    assert any("diabetes" in r.display for r in results)


def test_fuzzy_search_no_match_gibberish(index):
    results = index.fuzzy_search("zzzzzzz", limit=10)
    assert results == []


def test_prefix_search_no_match(index):
    results = index.prefix_search("zzzzzzz", limit=10)
    assert results == []


def test_semantic_search_unavailable_returns_empty(index):
    results = index.semantic_search("heart attack", limit=10)
    assert results == []


def test_semantic_search_via_get_top_neighbors():
    strategy = _make_mock_strategy(semantic_available=True)
    strategy.get_top_neighbors = MagicMock(return_value=[
        SNOMEDMatch(
            code="22298006", display="myocardial infarction",
            match_type="semantic", confidence=0.88,
            original_text="heart attack", span=(0, 12), negated=False,
        )
    ])
    idx = AutocompleteIndex(strategy, _GEO_DATA)
    results = idx.semantic_search("heart attack", limit=10)
    assert len(results) == 1
    assert results[0].display == "myocardial infarction"
    assert results[0].match_type == "semantic"


def test_prefix_search_skips_two_char_state_keys(index):
    results = index.prefix_search("cal", limit=10)
    assert any("California" in r.display for r in results)
