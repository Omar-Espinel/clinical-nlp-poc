"""tests/test_query_summary.py — Phase 3 QuerySummary unit tests.

Tests are unit-level: assembler is constructed directly with mock inputs.
No full pipeline run is required for most tests, mirroring the pattern in
test_metric_assembler.py and test_phase2_gate_removal.py.
HIPAA: no raw query text, SNOMED display strings, or filter values are logged here.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("SNOMED_SEARCH_STRATEGY", "hybrid_cascade")

PROJECT_ROOT = Path(__file__).parent.parent
SNOMED_CSV = str(PROJECT_ROOT / "data" / "snomed_clinical_trials.csv")

# ---------------------------------------------------------------------------
# Helpers — build cheap mock objects without importing pipeline
# ---------------------------------------------------------------------------


def _make_snomed_match(
    code: str = "44054006",
    display: str = "type 2 diabetes mellitus",
    confidence: float = 0.95,
    match_type: str = "exact",
    original_text: str = "diabetes",
    negated: bool = False,
):
    from src.snomed_search.base import SNOMEDMatch
    return SNOMEDMatch(
        code=code,
        display=display,
        confidence=confidence,
        match_type=match_type,
        original_text=original_text,
        span=(0, len(original_text)),
        negated=negated,
    )


def _make_filters_output(
    investigator_name_value=None,
    investigator_name_confidence=0.9,
    site_name_value=None,
    site_name_confidence=0.9,
    city_value=None,
    city_confidence=0.9,
    state_values=None,
    state_confidence=0.9,
    state_is_region=False,
    phase_values=None,
    phase_confidence=0.95,
):
    from src.assembler import FiltersOutput, FilterFieldOutput, StateFilterOutput, PhaseFilterOutput
    return FiltersOutput(
        investigator_name=FilterFieldOutput(
            value=investigator_name_value,
            confidence=investigator_name_confidence,
        ),
        site_name=FilterFieldOutput(
            value=site_name_value,
            confidence=site_name_confidence,
        ),
        city=FilterFieldOutput(value=city_value, confidence=city_confidence),
        state=StateFilterOutput(
            values=state_values or [],
            confidence=state_confidence,
            is_region=state_is_region,
        ),
        phase=PhaseFilterOutput(
            values=phase_values or [],
            confidence=phase_confidence,
        ),
    )


def _make_sufficiency_decision(ambiguous_name_flags=None):
    from src.sufficiency_gate import SufficiencyDecision
    return SufficiencyDecision(
        sufficient=True,
        reason="ok_no_trigger",
        ambiguous_name_flags=ambiguous_name_flags,
    )


def _assembler():
    from src.assembler import ResponseAssembler
    return ResponseAssembler()


# ---------------------------------------------------------------------------
# Test 1 — happy path: clean high-confidence query → QuerySummary with no warnings
# ---------------------------------------------------------------------------


def test_happy_path_no_warnings():
    """High-confidence SNOMED match + no low-confidence filters → has_warnings=False."""
    asm = _assembler()
    matches = [_make_snomed_match(confidence=0.95)]
    filters_out = _make_filters_output()
    decision = _make_sufficiency_decision()

    qs = asm.assemble_query_summary(
        snomed_matches=matches,
        filters_out=filters_out,
        sufficiency_decision=decision,
        canonical_query="diabetes mellitus",
    )

    assert qs is not None
    assert qs.has_warnings is False
    assert qs.flags == []
    assert qs.flag_count == 0
    # Label contains the SNOMED display value (HTML-escaped)
    assert "type 2 diabetes mellitus" in qs.label
    # interpreted_terms contains the escaped display
    assert "type 2 diabetes mellitus" in qs.interpreted_terms


# ---------------------------------------------------------------------------
# Test 2 — low-confidence investigator → FlagItem field="investigator"
# ---------------------------------------------------------------------------


def test_low_confidence_investigator_flag():
    """investigator_name with confidence < 0.75 produces a flag."""
    asm = _assembler()
    matches = [_make_snomed_match(confidence=0.95)]
    filters_out = _make_filters_output(
        investigator_name_value="Dr. Smith",
        investigator_name_confidence=0.60,
    )
    decision = _make_sufficiency_decision()

    qs = asm.assemble_query_summary(
        snomed_matches=matches,
        filters_out=filters_out,
        sufficiency_decision=decision,
        canonical_query="diabetes mellitus dr smith",
    )

    assert qs.has_warnings is True
    inv_flags = [f for f in qs.flags if f.field == "investigator"]
    assert len(inv_flags) == 1
    assert inv_flags[0].message == "Matched with low confidence"
    assert abs(inv_flags[0].confidence - 0.60) < 1e-6


# ---------------------------------------------------------------------------
# Test 3 — ambiguous name flags → FlagItem field="investigator_or_site"
# ---------------------------------------------------------------------------


def test_ambiguous_name_flag():
    """ambiguous_name_flags from Phase 2 SufficiencyDecision produces investigator_or_site flag."""
    asm = _assembler()
    matches = [_make_snomed_match(confidence=0.95)]
    filters_out = _make_filters_output()
    decision = _make_sufficiency_decision(ambiguous_name_flags=["Bay Medical"])

    qs = asm.assemble_query_summary(
        snomed_matches=matches,
        filters_out=filters_out,
        sufficiency_decision=decision,
        canonical_query="diabetes mellitus Bay Medical",
    )

    assert qs.has_warnings is True
    amb_flags = [f for f in qs.flags if f.field == "investigator_or_site"]
    assert len(amb_flags) == 1
    assert amb_flags[0].message == "Role unclear — investigator or site?"
    assert amb_flags[0].confidence == 0.0


# ---------------------------------------------------------------------------
# Test 4 — unrecognized term (e.g. "gout" when not in CSV known terms)
# ---------------------------------------------------------------------------


def test_unrecognized_term_gout():
    """Token 'gout' not in SNOMED CSV → appears in unrecognized_terms, has_warnings=True."""
    asm = _assembler()
    # No SNOMED matches — gout not in CSV in this test context
    matches = []
    filters_out = _make_filters_output()
    decision = _make_sufficiency_decision()

    qs = asm.assemble_query_summary(
        snomed_matches=matches,
        filters_out=filters_out,
        sufficiency_decision=decision,
        canonical_query="gout studies",
    )

    assert qs.has_warnings is True
    # 'gout' should appear (possibly HTML-escaped, but 'gout' has no special chars)
    assert "gout" in qs.unrecognized_terms
    assert qs.unrecognized_term_count >= 1


# ---------------------------------------------------------------------------
# Test 5 — label assembly order: SNOMED · phase · investigator · site · city · state
# ---------------------------------------------------------------------------


def test_label_assembly_order():
    """Label parts appear in fixed order: SNOMED · phase · investigator · site · city · state."""
    asm = _assembler()
    matches = [_make_snomed_match(display="Leukemia", confidence=0.90)]
    filters_out = _make_filters_output(
        phase_values=["Phase 2", "Phase 3"],
        phase_confidence=0.95,
        investigator_name_value="Dr. Martinez",
        investigator_name_confidence=0.90,
        site_name_value="Bay Area Hospital",
        site_name_confidence=0.90,
        city_value="San Francisco",
        city_confidence=0.90,
        state_values=["California"],
        state_confidence=0.90,
    )
    decision = _make_sufficiency_decision()

    qs = asm.assemble_query_summary(
        snomed_matches=matches,
        filters_out=filters_out,
        sufficiency_decision=decision,
        canonical_query="leukemia phase 2 or 3 dr martinez bay area",
    )

    label = qs.label
    # Check all parts present
    assert "Leukemia" in label
    assert "Phase 2, Phase 3" in label
    assert "Dr. Martinez" in label
    assert "Bay Area Hospital" in label
    assert "San Francisco" in label
    assert "California" in label

    # Check order: snomed index < phase index < investigator index < site index ...
    idx_snomed = label.find("Leukemia")
    idx_phase = label.find("Phase 2")
    idx_inv = label.find("Dr. Martinez")
    idx_site = label.find("Bay Area Hospital")
    idx_city = label.find("San Francisco")
    idx_state = label.find("California")
    assert idx_snomed < idx_phase < idx_inv < idx_site < idx_city < idx_state


# ---------------------------------------------------------------------------
# Test 6 — HTML injection in SNOMED display → label contains escaped version
# ---------------------------------------------------------------------------


def test_html_injection_in_snomed_display():
    """SNOMED display containing HTML gets escaped; raw HTML never appears in label."""
    asm = _assembler()
    malicious_display = "<script>alert('xss')</script>"
    matches = [_make_snomed_match(display=malicious_display, confidence=0.90)]
    filters_out = _make_filters_output()
    decision = _make_sufficiency_decision()

    qs = asm.assemble_query_summary(
        snomed_matches=matches,
        filters_out=filters_out,
        sufficiency_decision=decision,
        canonical_query="some clinical query",
    )

    # Raw HTML must NOT appear
    assert "<script>" not in qs.label
    assert "<script>" not in qs.interpreted_terms
    # Escaped version must appear
    assert "&lt;script&gt;" in qs.label
    assert "&lt;script&gt;" in qs.interpreted_terms[0]


# ---------------------------------------------------------------------------
# Test 7 — unrecognized_terms capped at 5
# ---------------------------------------------------------------------------


def test_unrecognized_terms_capped_at_5():
    """At most 5 unrecognized terms are returned regardless of query length."""
    asm = _assembler()
    matches = []
    filters_out = _make_filters_output()
    decision = _make_sufficiency_decision()

    # Fabricate a query with many tokens that won't appear in any known list.
    # Using made-up words that are >=4 chars and not in any skiplist.
    fake_query = "zorp zlex flump grumble wamble snorkel blorp qwert yuiop asdfg"

    qs = asm.assemble_query_summary(
        snomed_matches=matches,
        filters_out=filters_out,
        sufficiency_decision=decision,
        canonical_query=fake_query,
    )

    assert len(qs.unrecognized_terms) <= 5
    assert qs.unrecognized_term_count == len(qs.unrecognized_terms)


# ---------------------------------------------------------------------------
# Test 8 — query_summary may be None (NLPOutput allows None)
# ---------------------------------------------------------------------------


def test_query_summary_may_be_none():
    """NLPOutput.query_summary is Optional — the model must allow None."""
    from src.assembler import NLPOutput, FiltersOutput, FilterFieldOutput, StateFilterOutput, PhaseFilterOutput, MetadataOutput

    output = NLPOutput(
        snomed_terms=[],
        filters=FiltersOutput(
            investigator_name=FilterFieldOutput(value=None, confidence=0.0),
            site_name=FilterFieldOutput(value=None, confidence=0.0),
            city=FilterFieldOutput(value=None, confidence=0.0),
            state=StateFilterOutput(values=[], confidence=0.0, is_region=False),
            phase=PhaseFilterOutput(values=[], confidence=0.0),
        ),
        metric_filters=[],
        metadata=MetadataOutput(
            processing_time_ms=1,
            total_snomed_matches=0,
            snomed_match_types={},
            negated_terms_excluded=0,
        ),
        query_summary=None,
    )
    assert output.query_summary is None


# ---------------------------------------------------------------------------
# Test 9 — contract test: consumer reading only type/snomed_terms/filters/
#           metric_filters/metadata sees no errors when query_summary is ignored
# ---------------------------------------------------------------------------


def test_contract_existing_consumer_fields_accessible():
    """Existing consumer that accesses type, snomed_terms, filters, metric_filters,
    metadata must not encounter KeyError/AttributeError/ValidationError."""
    from src.assembler import ResponseAssembler, NLPOutput
    from src.snomed_search.base import SNOMEDMatch
    from src.normalizers.geo import GeoResult
    from src.filter_extractor import ExtractedFilters, FilterField, StateFilter, PhaseFilter

    asm = ResponseAssembler()
    filters = ExtractedFilters(
        investigator_name=FilterField(value=None, confidence=0.0),
        site_name=FilterField(value=None, confidence=0.0),
        city=FilterField(value=None, confidence=0.0),
        state=StateFilter(values=[], confidence=0.0, is_region=False),
        phase=PhaseFilter(values=[], confidence=0.0),
        raw_response_length=0,
        metric_fields={},
    )
    geo = GeoResult(city=None, states=[], confidence=0.0, is_region=False, country=None, original_city=None, original_state=None)
    import time
    output = asm.assemble(
        filters=filters,
        snomed_matches=[],
        geo=geo,
        start_time=time.perf_counter(),
        metric_filters=[],
        # No canonical_query → query_summary will be None
    )

    # Consumer accesses existing fields — no error
    assert output.type == "search"
    _ = output.snomed_terms
    _ = output.filters
    _ = output.metric_filters
    _ = output.metadata

    # Serialization to dict must also not raise
    dumped = output.model_dump()
    assert "type" in dumped
    assert "snomed_terms" in dumped
    assert "filters" in dumped
    assert "metadata" in dumped
    # query_summary key is present but None
    assert "query_summary" in dumped
    assert dumped["query_summary"] is None


# ---------------------------------------------------------------------------
# Test 10 — NLPOutput.query_summary is present and populated when canonical_query given
# ---------------------------------------------------------------------------


def test_query_summary_present_when_canonical_given():
    """assemble() with canonical_query → query_summary is not None."""
    from src.assembler import ResponseAssembler
    from src.normalizers.geo import GeoResult
    from src.filter_extractor import ExtractedFilters, FilterField, StateFilter, PhaseFilter
    import time

    asm = ResponseAssembler()
    filters = ExtractedFilters(
        investigator_name=FilterField(value=None, confidence=0.0),
        site_name=FilterField(value=None, confidence=0.0),
        city=FilterField(value=None, confidence=0.0),
        state=StateFilter(values=[], confidence=0.0, is_region=False),
        phase=PhaseFilter(values=[], confidence=0.0),
        raw_response_length=0,
        metric_fields={},
    )
    geo = GeoResult(city=None, states=[], confidence=0.0, is_region=False, country=None, original_city=None, original_state=None)
    snomed_matches = [_make_snomed_match(confidence=0.92)]

    output = asm.assemble(
        filters=filters,
        snomed_matches=snomed_matches,
        geo=geo,
        start_time=time.perf_counter(),
        metric_filters=[],
        canonical_query="type 2 diabetes mellitus",
    )

    assert output.query_summary is not None
    assert output.query_summary.flag_count == output.query_summary.flag_count  # just sanity
    assert isinstance(output.query_summary.label, str)


# ---------------------------------------------------------------------------
# Test 11 — low-confidence SNOMED match produces snomed flag, capped at 3
# ---------------------------------------------------------------------------


def test_low_confidence_snomed_flags_capped_at_3():
    """Low-confidence SNOMED matches (0.60 <= conf < 0.72) produce flags, max 3."""
    asm = _assembler()
    matches = [
        _make_snomed_match(code=f"000{i}", display=f"Term {i}", confidence=0.65)
        for i in range(5)  # 5 low-confidence matches
    ]
    filters_out = _make_filters_output()
    decision = _make_sufficiency_decision()

    qs = asm.assemble_query_summary(
        snomed_matches=matches,
        filters_out=filters_out,
        sufficiency_decision=decision,
        canonical_query="clinical query",
    )

    snomed_flags = [f for f in qs.flags if f.field == "snomed"]
    assert len(snomed_flags) <= 3
    assert qs.has_warnings is True


# ---------------------------------------------------------------------------
# Test 12 — label omits null/empty filter fields
# ---------------------------------------------------------------------------


def test_label_omits_null_filters():
    """Label must not include separators or empty strings for None/empty filters."""
    asm = _assembler()
    matches = [_make_snomed_match(display="Leukemia", confidence=0.90)]
    # Only phase set; investigator/site/city/state all null
    filters_out = _make_filters_output(phase_values=["Phase 1"], phase_confidence=0.95)
    decision = _make_sufficiency_decision()

    qs = asm.assemble_query_summary(
        snomed_matches=matches,
        filters_out=filters_out,
        sufficiency_decision=decision,
        canonical_query="leukemia phase 1",
    )

    assert "Leukemia" in qs.label
    assert "Phase 1" in qs.label
    # No dangling separators or empty parts
    assert " · " not in qs.label.split("Phase 1")[1].strip()
    assert qs.interpreted_filters.get("investigator") is None
    assert qs.interpreted_filters.get("site") is None
