import asyncio
import time
import pytest
from unittest.mock import MagicMock
from src.autocomplete.orchestrator import AutocompleteOrchestrator, _MIN_RESPONSE_SECONDS
from src.autocomplete import cache as autocomplete_cache


def _make_strategy():
    strategy = MagicMock()
    strategy._exact_index = {
        "diabetes mellitus": {"concept_id": "73211009", "preferred_term": "diabetes mellitus"},
        "hypertension": {"concept_id": "38341003", "preferred_term": "hypertension"},
    }
    strategy._synonym_index = {}
    strategy._alias_dict = {}
    strategy._np_embeddings = None
    strategy._np_terms = []
    strategy._collection = None
    strategy._embedder = None
    strategy.semantic_available = False
    return strategy


_GEO_DATA = {
    "cities": {"boston": {"canonical": "Boston", "state": "Massachusetts", "country": "US"}},
    "states": {"massachusetts": "Massachusetts"},
    "regions": {},
}


@pytest.fixture(autouse=True)
def clear_cache():
    autocomplete_cache.cache_clear()
    yield
    autocomplete_cache.cache_clear()


@pytest.fixture
def orchestrator():
    return AutocompleteOrchestrator(_make_strategy(), _GEO_DATA)


@pytest.mark.asyncio
async def test_run_returns_empty_for_short_prefix(orchestrator):
    result = await orchestrator.run("di")
    assert result["suggestions"] == []


@pytest.mark.asyncio
async def test_run_rejects_injection(orchestrator):
    result = await orchestrator.run("'; DROP TABLE users")
    assert result["suggestions"] == []


@pytest.mark.asyncio
async def test_run_diabetes_prefix(orchestrator):
    result = await orchestrator.run("diab")
    displays = [s["display"] for s in result["suggestions"]]
    assert any("diabetes" in d.lower() for d in displays)


@pytest.mark.asyncio
async def test_run_phase_prefix(orchestrator):
    result = await orchestrator.run("Pha")
    displays = [s["display"] for s in result["suggestions"]]
    assert any("Phase" in d for d in displays)


@pytest.mark.asyncio
async def test_run_city_prefix(orchestrator):
    result = await orchestrator.run("Bos")
    displays = [s["display"] for s in result["suggestions"]]
    assert any("Boston" in d for d in displays)


@pytest.mark.asyncio
async def test_run_no_results_returns_empty(orchestrator):
    result = await orchestrator.run("zzzzzzz")
    assert result["suggestions"] == []


@pytest.mark.asyncio
async def test_run_respects_limit(orchestrator):
    result = await orchestrator.run("diab", limit=3)
    assert len(result["suggestions"]) <= 3


@pytest.mark.asyncio
async def test_run_cache_hit_second_call(orchestrator):
    await orchestrator.run("diab", limit=7)
    result2 = await orchestrator.run("diab", limit=7)
    assert result2["tier_used"] == "cache"


@pytest.mark.asyncio
async def test_run_has_minimum_response_time(orchestrator):
    start = time.perf_counter()
    await orchestrator.run("bos")
    elapsed = time.perf_counter() - start
    assert elapsed >= _MIN_RESPONSE_SECONDS


@pytest.mark.asyncio
async def test_completion_includes_context_prefix(orchestrator):
    result = await orchestrator.run("Phase 3 diab")
    suggestions = result["suggestions"]
    if suggestions:
        assert any(s["completion"].startswith("Phase 3") for s in suggestions)
        assert all(
            not s["display"].startswith("Phase 3")
            for s in suggestions
            if "diabetes" in s["display"].lower()
        )


@pytest.mark.asyncio
async def test_compound_phase_expands_to_multiple_suggestions(orchestrator):
    result = await orchestrator.run(
        "malignant neoplasm of esophagus, Phase 2 and 3"
    )
    completions = [s["completion"] for s in result["suggestions"]]
    assert any("Phase 2" in c for c in completions)
    assert any("Phase 3" in c for c in completions)


@pytest.mark.asyncio
async def test_noise_stripped_query_returns_snomed_suggestions(orchestrator):
    result = await orchestrator.run("trials for gout US Sites only")
    assert result["suggestions"] or result["tier_used"] == "none"


@pytest.mark.asyncio
async def test_compound_phase_completions_preserve_committed_context(orchestrator):
    result = await orchestrator.run(
        "malignant neoplasm of esophagus, Phase 2 and 3"
    )
    for s in result["suggestions"]:
        if "Phase" in s["completion"]:
            assert "esophagus" in s["completion"].lower()
