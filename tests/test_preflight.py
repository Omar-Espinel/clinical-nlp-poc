"""Tests for PreflightMandatoryCheck (FIX 2: token-based + SECONDARY skiplist)."""

import pytest

from src.extractors.preflight import PreflightMandatoryCheck
from src.extractors.names import (
    PERSON_PREFIXES,
    PERSON_SUFFIXES,
    CONTEXT_PERSON,
    CONTEXT_SITE,
)


@pytest.fixture(scope="module")
def preflight() -> PreflightMandatoryCheck:
    known_terms = frozenset({
        "diabetes",
        "hypertension",
        "type 2 diabetes mellitus",
        "myocardial infarction",
        "breast cancer",
    })
    institution_kw = frozenset({
        "hospital", "clinic", "university",
        "cancer center", "medical center",
    })
    # Per FIX 2 / test note: include "midwest" so the test for
    # "active trials midwest" treats it as unambiguous geo.
    geo_multiword = frozenset({
        "new york", "los angeles", "east coast", "west coast",
        "new england", "the midwest", "pacific northwest",
        "greater boston", "san francisco",
        "midwest",
    })
    return PreflightMandatoryCheck(
        known_terms=known_terms,
        person_prefixes=PERSON_PREFIXES,
        person_suffixes=PERSON_SUFFIXES,
        institution_keywords=institution_kw,
        context_signals=CONTEXT_PERSON | CONTEXT_SITE,
        geo_multiword_keys=geo_multiword,
    )


@pytest.mark.parametrize("query", [
    "trials washington",
    "phase 3 cleveland",
    "johnson diabetes phase 2",
    "northwestern oncology",
    "dr smith phase 2",
    "diabetes trials",
    "trials at mass general",
    "trials by johnson",
])
def test_passes_preflight(preflight, query):
    """These queries must NOT be rejected (at least one mandatory signal)."""
    r = preflight.evaluate(query)
    assert r.passed is True, f"{query!r} should pass (signals={r.signal_count})"
    assert r.signal_count >= 1


@pytest.mark.parametrize("query", [
    "phase 3 east coast",
    "active trials midwest",
    "fast fpe low deviations",
    "phase 2 new england",
])
def test_rejected_at_preflight(preflight, query):
    """These queries must be rejected — no mandatory term signals at all."""
    r = preflight.evaluate(query)
    assert r.passed is False, (
        f"{query!r} should be rejected (signals={r.signal_count})"
    )
    assert r.signal_count == 0
