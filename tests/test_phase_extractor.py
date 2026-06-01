"""Tests for PhaseExtractor."""

import pytest

from src.extractors.phase import PhaseExtractor


@pytest.fixture(scope="module")
def extractor() -> PhaseExtractor:
    return PhaseExtractor()


@pytest.mark.parametrize(
    "query,expected_value,expected_conf",
    [
        ("Phase 3 diabetes", "Phase 3", 0.95),
        ("phase III oncology", "Phase 3", 0.95),
        ("p3 oncology", "Phase 3", 0.95),
        ("pivotal trial diabetes", "Phase 3", 0.85),
        ("registrational study", "Phase 3", 0.85),
        ("first in human safety", "Phase 1", 0.85),
        ("first-in-human study", "Phase 1", 0.85),
        ("fih safety study", "Phase 1", 0.85),
        ("dose escalation oncology", "Phase 1", 0.85),
        ("dose expansion study", "Phase 1/2", 0.85),
        ("proof of concept diabetes", "Phase 2", 0.85),
        ("poc trial boston", "Phase 2", 0.85),
        ("phase 1/2 safety oncology", "Phase 1/2", 0.95),
        ("phase 2/3 diabetes", "Phase 2/3", 0.95),
        ("phase 2b breast cancer", "Phase 2b", 0.95),
        ("phase 1b dose finding", "Phase 1b", 0.95),
    ],
)
def test_exact_and_alias(extractor, query, expected_value, expected_conf):
    result = extractor.extract(query)
    assert result.value == expected_value, f"value mismatch for {query!r}"
    assert result.confidence == expected_conf, f"conf mismatch for {query!r}"
    assert result.span is not None
    assert 0 <= result.span[0] < result.span[1] <= len(query)


@pytest.mark.parametrize(
    "query,expected_value",
    [
        ("phas 3 diabetes", "Phase 3"),
        ("phse 2 trials", "Phase 2"),
    ],
)
def test_fuzzy(extractor, query, expected_value):
    result = extractor.extract(query)
    assert result.value == expected_value, f"value mismatch for {query!r}"
    assert result.confidence == 0.70
    assert result.span is not None


@pytest.mark.parametrize(
    "query",
    ["diabetes trials boston", "dr smith cancer research"],
)
def test_no_phase(extractor, query):
    result = extractor.extract(query)
    assert result.value is None
    assert result.confidence == 0.0
    assert result.span is None
