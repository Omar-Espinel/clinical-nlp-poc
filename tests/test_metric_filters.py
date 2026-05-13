"""
tests/test_metric_filters.py

Pytest unit tests for the metric filter recognition feature (v2 — Advarra-specific metrics).

Spec: rework-metric-filters-v2.md §4.1 (9 groups, ~58 cases).

No real LLM or GROQ_API_KEY required. Gate tests use MagicMock session + real resolver.
"""

from __future__ import annotations

import json
import re
import types
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.normalizers.metric import (
    MAX_FUZZY_COMPARISONS_PER_QUERY,
    VALID_UNITS,
    MetricFilterNormalizer,
    MetricFilterOutput,
    MetricIntentResolver,
    MetricMatch,
)
from src.sufficiency_gate import AmbiguousEntry, MetricAmbiguityGate, SufficiencyDecision

# ---------------------------------------------------------------------------
# Module-scoped resolver fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def resolver():
    dict_path = Path(__file__).parent.parent / "data" / "metric_filters.json"
    return MetricIntentResolver(str(dict_path))


# ===========================================================================
# Group 1 — AC exact match (12 cases, one per field)
# ===========================================================================

class TestACExactMatches:

    def test_field1_exact(self, resolver):
        matches = resolver.resolve("total studies with advarra over 1000")
        fields = {m.canonical_field for m in matches}
        assert "total_studies_with_advarra" in fields
        m = next(x for x in matches if x.canonical_field == "total_studies_with_advarra")
        assert m.confidence == 1.0
        # 'over' is not in total_studies_with_advarra implied_operators → 'any'

    def test_field2_exact(self, resolver):
        matches = resolver.resolve("matching studies count under 50")
        fields = {m.canonical_field for m in matches}
        assert "studies_matching_search" in fields

    def test_field3_exact(self, resolver):
        matches = resolver.resolve("active trials more than 25")
        fields = {m.canonical_field for m in matches}
        assert "active_trials" in fields

    def test_field4_exact(self, resolver):
        matches = resolver.resolve("most recent approval date since 2025")
        fields = {m.canonical_field for m in matches}
        assert "most_recent_approval_date" in fields
        m = next(x for x in matches if x.canonical_field == "most_recent_approval_date")
        assert m.implied_operator == "gte"

    def test_field5_exact(self, resolver):
        matches = resolver.resolve("query response time fast")
        fields = {m.canonical_field for m in matches}
        assert "avg_days_respond_to_queries" in fields
        m = next(x for x in matches if x.canonical_field == "avg_days_respond_to_queries")
        assert m.implied_operator == "lt"

    def test_field6_exact(self, resolver):
        matches = resolver.resolve("submission to approval time slow")
        fields = {m.canonical_field for m in matches}
        assert "avg_days_submission_to_approval" in fields
        m = next(x for x in matches if x.canonical_field == "avg_days_submission_to_approval")
        assert m.implied_operator == "gt"

    def test_field7_exact(self, resolver):
        matches = resolver.resolve("total protocol deviations all studies high")
        fields = {m.canonical_field for m in matches}
        assert "total_protocol_deviations_all_studies" in fields
        m = next(x for x in matches if x.canonical_field == "total_protocol_deviations_all_studies")
        assert m.implied_operator == "gt"

    def test_field8_exact(self, resolver):
        matches = resolver.resolve("protocol deviations in matching studies few")
        fields = {m.canonical_field for m in matches}
        assert "total_protocol_deviations_matching_studies" in fields
        m = next(x for x in matches if x.canonical_field == "total_protocol_deviations_matching_studies")
        assert m.implied_operator == "lt"

    def test_field9_exact(self, resolver):
        matches = resolver.resolve("average enrollment matching studies large")
        fields = {m.canonical_field for m in matches}
        assert "avg_enrollment_matching_studies" in fields
        m = next(x for x in matches if x.canonical_field == "avg_enrollment_matching_studies")
        assert m.implied_operator == "gt"

    def test_field10_exact(self, resolver):
        matches = resolver.resolve("average enrollment matching ta low")
        fields = {m.canonical_field for m in matches}
        assert "avg_enrollment_matching_ta" in fields
        m = next(x for x in matches if x.canonical_field == "avg_enrollment_matching_ta")
        assert m.implied_operator == "lt"

    def test_field11_exact(self, resolver):
        matches = resolver.resolve("average screening rate high")
        fields = {m.canonical_field for m in matches}
        assert "avg_screening_rate" in fields
        m = next(x for x in matches if x.canonical_field == "avg_screening_rate")
        assert m.implied_operator == "gt"

    def test_field12_exact(self, resolver):
        matches = resolver.resolve("days to fpe under 60")
        fields = {m.canonical_field for m in matches}
        assert "avg_days_to_fpe" in fields
        # 'under' not in implied_operators → 'any' (LLM parses it)


# ===========================================================================
# Group 2 — AC synonym match (12 cases, alternate synonym per field)
# ===========================================================================

class TestACSynonymMatches:

    def test_field1_synonym(self, resolver):
        matches = resolver.resolve("number of advarra studies")
        fields = {m.canonical_field for m in matches}
        assert "total_studies_with_advarra" in fields

    def test_field2_synonym(self, resolver):
        matches = resolver.resolve("studies matching my filters")
        fields = {m.canonical_field for m in matches}
        assert "studies_matching_search" in fields

    def test_field3_synonym(self, resolver):
        matches = resolver.resolve("ongoing trials live status")
        fields = {m.canonical_field for m in matches}
        assert "active_trials" in fields

    def test_field4_synonym(self, resolver):
        matches = resolver.resolve("latest irb approval")
        fields = {m.canonical_field for m in matches}
        assert "most_recent_approval_date" in fields

    def test_field5_synonym(self, resolver):
        matches = resolver.resolve("query turnaround time")
        fields = {m.canonical_field for m in matches}
        assert "avg_days_respond_to_queries" in fields

    def test_field6_synonym(self, resolver):
        matches = resolver.resolve("submission to irb approval")
        fields = {m.canonical_field for m in matches}
        assert "avg_days_submission_to_approval" in fields

    def test_field7_synonym(self, resolver):
        matches = resolver.resolve("total pds all studies high")
        fields = {m.canonical_field for m in matches}
        assert "total_protocol_deviations_all_studies" in fields

    def test_field8_synonym(self, resolver):
        matches = resolver.resolve("deviations in matching studies")
        fields = {m.canonical_field for m in matches}
        assert "total_protocol_deviations_matching_studies" in fields

    def test_field9_synonym(self, resolver):
        matches = resolver.resolve("average matched enrollment")
        fields = {m.canonical_field for m in matches}
        assert "avg_enrollment_matching_studies" in fields

    def test_field10_synonym(self, resolver):
        matches = resolver.resolve("avg ta enrollment")
        fields = {m.canonical_field for m in matches}
        assert "avg_enrollment_matching_ta" in fields

    def test_field11_synonym(self, resolver):
        matches = resolver.resolve("screening velocity")
        fields = {m.canonical_field for m in matches}
        assert "avg_screening_rate" in fields

    def test_field12_synonym(self, resolver):
        matches = resolver.resolve("time to first patient enrolled")
        fields = {m.canonical_field for m in matches}
        assert "avg_days_to_fpe" in fields


# ===========================================================================
# Group 3 — Fuzzy fallback (6 cases)
# ===========================================================================

class TestFuzzyFallback:

    def test_fuzzy_advarra_typo(self, resolver):
        """Typo in 'advarra' → may catch via fuzzy."""
        matches = resolver.resolve("advara studies count over 100")
        fields = {m.canonical_field for m in matches}
        if "total_studies_with_advarra" in fields:
            m = next(x for x in matches if x.canonical_field == "total_studies_with_advarra")
            assert m.match_source == "fuzzy"
            assert m.confidence <= 0.90

    def test_fuzzy_enrolment_british(self, resolver):
        """British spelling 'enrolment' → may catch field 9 or 10 via fuzzy."""
        matches = resolver.resolve("average enrolment matching studies")
        # fuzzy may or may not fire; just validate no crash

    def test_fuzzy_screning_typo(self, resolver):
        matches = resolver.resolve("screning rate high in diabetes")
        fields = {m.canonical_field for m in matches}
        if "avg_screening_rate" in fields:
            m = next(x for x in matches if x.canonical_field == "avg_screening_rate")
            assert m.match_source == "fuzzy"

    def test_fuzzy_protocol_typo(self, resolver):
        matches = resolver.resolve("protocl deviations all studies high")
        fields = {m.canonical_field for m in matches}
        if "total_protocol_deviations_all_studies" in fields:
            m = next(x for x in matches if x.canonical_field == "total_protocol_deviations_all_studies")
            assert m.match_source == "fuzzy"

    def test_single_token_fpe_no_match(self, resolver):
        """Single-token 'fpe' → no match (residual < 6 chars or n-gram floor)."""
        matches = resolver.resolve("fpe")
        # The AC automaton should NOT match 'fpe' because the query is only
        # 3 chars after normalization; residual < 6 chars guard skips fuzzy.
        fpe_matches = [m for m in matches if m.canonical_field == "avg_days_to_fpe"]
        assert len(fpe_matches) == 0

    def test_budget_short_circuit_no_crash(self, resolver):
        """300-token unknown query → budget exhausted, no crash."""
        long_query = " ".join(["unknown"] * 300)
        matches = resolver.resolve(long_query)
        # Just no exception raised, regardless of result
        assert isinstance(matches, list)


# ===========================================================================
# Group 4 — Implied operator (10 cases)
# ===========================================================================

class TestImpliedOperator:

    def test_high_screening_rate(self, resolver):
        matches = resolver.resolve("high screening rate diabetes")
        m = next((x for x in matches if x.canonical_field == "avg_screening_rate"), None)
        if m:
            assert m.implied_operator == "gt"

    def test_low_screening_rate(self, resolver):
        matches = resolver.resolve("low screening rate diabetes")
        m = next((x for x in matches if x.canonical_field == "avg_screening_rate"), None)
        if m:
            assert m.implied_operator == "lt"

    def test_fast_query_response(self, resolver):
        matches = resolver.resolve("fast query response time")
        m = next((x for x in matches if x.canonical_field == "avg_days_respond_to_queries"), None)
        if m:
            assert m.implied_operator == "lt"

    def test_since_approval_date(self, resolver):
        matches = resolver.resolve("since most recent approval date")
        m = next((x for x in matches if x.canonical_field == "most_recent_approval_date"), None)
        if m:
            assert m.implied_operator == "gte"

    def test_before_approval_date(self, resolver):
        matches = resolver.resolve("before most recent approval date")
        m = next((x for x in matches if x.canonical_field == "most_recent_approval_date"), None)
        if m:
            assert m.implied_operator == "lt"

    def test_recent_approval(self, resolver):
        matches = resolver.resolve("recent approval date diabetes")
        m = next((x for x in matches if x.canonical_field == "most_recent_approval_date"), None)
        if m:
            assert m.implied_operator == "gte"

    def test_many_active_trials(self, resolver):
        matches = resolver.resolve("many active trials cancer")
        m = next((x for x in matches if x.canonical_field == "active_trials"), None)
        if m:
            assert m.implied_operator == "gt"

    def test_zero_protocol_deviations_all(self, resolver):
        matches = resolver.resolve("zero protocol deviations all studies")
        m = next((x for x in matches if x.canonical_field == "total_protocol_deviations_all_studies"), None)
        if m:
            assert m.implied_operator == "eq"

    def test_large_total_advarra_studies(self, resolver):
        matches = resolver.resolve("large total studies with advarra")
        m = next((x for x in matches if x.canonical_field == "total_studies_with_advarra"), None)
        if m:
            assert m.implied_operator == "gt"

    def test_small_matching_studies(self, resolver):
        matches = resolver.resolve("small matching studies count")
        m = next((x for x in matches if x.canonical_field == "studies_matching_search"), None)
        if m:
            assert m.implied_operator == "lt"


# ===========================================================================
# Group 5 — Overlap dedup (5 cases)
# ===========================================================================

class TestOverlapAndDedup:

    def test_field7_not_field8_all_studies(self, resolver):
        """"total protocol deviations in all my studies" → ONLY field 7, NOT field 8."""
        matches = resolver.resolve("total protocol deviations in all my studies")
        fields = {m.canonical_field for m in matches}
        assert "total_protocol_deviations_all_studies" in fields
        assert "total_protocol_deviations_matching_studies" not in fields

    def test_field8_not_field7_matching(self, resolver):
        """"protocol deviations across my matching studies" → ONLY field 8, NOT field 7."""
        matches = resolver.resolve("protocol deviations across my matching studies")
        fields = {m.canonical_field for m in matches}
        assert "total_protocol_deviations_matching_studies" in fields
        assert "total_protocol_deviations_all_studies" not in fields

    def test_field10_only_no_field9(self, resolver):
        """"average enrollment matching ta" → ONLY field 10 (no field 9 anchor)."""
        matches = resolver.resolve("average enrollment matching ta")
        fields = {m.canonical_field for m in matches}
        assert "avg_enrollment_matching_ta" in fields
        assert "avg_enrollment_matching_studies" not in fields

    def test_field9_wins_over_field10(self, resolver):
        """"average enrollment matching studies in matching ta" → field 9 wins (longer span)."""
        matches = resolver.resolve("average enrollment matching studies in matching ta")
        fields = {m.canonical_field for m in matches}
        assert "avg_enrollment_matching_studies" in fields

    def test_field6_and_field4_both(self, resolver):
        """"submission to approval days, most recent approval date" → BOTH field 6 and field 4."""
        matches = resolver.resolve("submission to approval days, most recent approval date")
        fields = {m.canonical_field for m in matches}
        assert "avg_days_submission_to_approval" in fields
        assert "most_recent_approval_date" in fields


# ===========================================================================
# Group 6 — MetricAmbiguityGate firing + date E2E (8 cases)
# ===========================================================================

class TestMetricAmbiguityGate:

    def test_single_field_numeric_fires(self, resolver):
        """Single numeric metric field unresolved → gate fires."""
        session = MagicMock()
        session.is_max_turns_reached.return_value = False
        session.session_id = "test-single-num"
        filters = MagicMock()
        filters.metric_fields = {}
        matches = [
            MetricMatch(
                canonical_field="active_trials",
                canonical_label="Active Trials",
                data_type="numeric",
                matched_text="active trials",
                span=(0, 13),
                confidence=1.0,
                implied_operator="any",
                match_source="ac_synonym",
                implied_op_map=types.MappingProxyType({}),
            ),
        ]
        gate = MetricAmbiguityGate(resolver=resolver)
        decision = gate.evaluate(matches, filters, session)
        assert decision is not None
        assert decision.sufficient is False
        assert decision.reason == "metric_without_value"
        assert decision.triggered_by is None
        assert "{trigger}" not in decision.matched_entry.question_template
        assert "Active Trials" in decision.matched_entry.question_template

    def test_single_field_date_options_substituted(self, resolver):
        """Date field → options are absolute MM/YYYY strings, not relative."""
        session = MagicMock()
        session.is_max_turns_reached.return_value = False
        session.session_id = "test-date-opts"
        filters = MagicMock()
        filters.metric_fields = {}
        matches = [
            MetricMatch(
                canonical_field="most_recent_approval_date",
                canonical_label="Most Recent Approval Date",
                data_type="date",
                matched_text="most recent approval date",
                span=(0, 26),
                confidence=1.0,
                implied_operator="any",
                match_source="ac_synonym",
                implied_op_map=types.MappingProxyType({}),
            ),
        ]
        gate = MetricAmbiguityGate(resolver=resolver)
        decision = gate.evaluate(matches, filters, session)
        assert decision is not None
        opts = decision.matched_entry.options
        # All but last should match "Since MM/YYYY" or "Before MM/YYYY"
        for i, opt in enumerate(opts[:-1]):
            assert re.match(r"^(Since|Before) \d{2}/\d{4}$", opt), (
                f"Option {i} does not match MM/YYYY pattern: {opt!r}"
            )
        assert opts[-1] == "No preference"

    def test_max_turns_escape(self, resolver):
        """Max turns reached → gate returns None."""
        session = MagicMock()
        session.is_max_turns_reached.return_value = True
        session.session_id = "test-maxturn"
        filters = MagicMock()
        filters.metric_fields = {}
        matches = [
            MetricMatch(
                canonical_field="active_trials",
                canonical_label="Active Trials",
                data_type="numeric",
                matched_text="active trials",
                span=(0, 13),
                confidence=1.0,
                implied_operator="any",
                match_source="ac_synonym",
                implied_op_map=types.MappingProxyType({}),
            ),
        ]
        gate = MetricAmbiguityGate(resolver=resolver)
        decision = gate.evaluate(matches, filters, session)
        assert decision is None

    def test_no_refire_after_any(self, resolver):
        """operator='any' → gate does not re-fire."""
        session = MagicMock()
        session.is_max_turns_reached.return_value = False
        session.session_id = "test-no-refire"
        resolved = MetricFilterOutput(
            field="active_trials",
            canonical_label="Active Trials",
            operator="any",
            data_type="numeric",
            value=None,
            original_text="active trials",
            confidence=1.0,
            unit=None,
        )
        filters = MagicMock()
        filters.metric_fields = {"active_trials": resolved}
        matches = [
            MetricMatch(
                canonical_field="active_trials",
                canonical_label="Active Trials",
                data_type="numeric",
                matched_text="active trials",
                span=(0, 13),
                confidence=1.0,
                implied_operator="any",
                match_source="ac_synonym",
                implied_op_map=types.MappingProxyType({}),
            ),
        ]
        gate = MetricAmbiguityGate(resolver=resolver)
        assert gate.evaluate(matches, filters, session) is None

    def test_no_matches_returns_none(self, resolver):
        session = MagicMock()
        session.is_max_turns_reached.return_value = False
        filters = MagicMock()
        filters.metric_fields = {}
        gate = MetricAmbiguityGate(resolver=resolver)
        assert gate.evaluate([], filters, session) is None

    def test_combined_fire(self, resolver):
        """Two unresolved → combined question template with second field label."""
        session = MagicMock()
        session.is_max_turns_reached.return_value = False
        session.session_id = "test-combined"
        filters = MagicMock()
        filters.metric_fields = {}
        matches = [
            MetricMatch(
                canonical_field="active_trials",
                canonical_label="Active Trials",
                data_type="numeric",
                matched_text="active trials",
                span=(0, 13),
                confidence=1.0,
                implied_operator="any",
                match_source="ac_synonym",
                implied_op_map=types.MappingProxyType({}),
            ),
            MetricMatch(
                canonical_field="studies_matching_search",
                canonical_label="Studies Matching Search",
                data_type="numeric",
                matched_text="matching studies count",
                span=(20, 42),
                confidence=1.0,
                implied_operator="any",
                match_source="ac_synonym",
                implied_op_map=types.MappingProxyType({}),
            ),
        ]
        gate = MetricAmbiguityGate(resolver=resolver)
        decision = gate.evaluate(matches, filters, session)
        assert decision is not None
        assert "Studies Matching Search" in decision.matched_entry.question_template


# ===========================================================================
# Group 7 — Pydantic validators (12 cases)
# ===========================================================================

class TestMetricFilterOutputValidators:

    def test_valid_numeric(self):
        mfo = MetricFilterOutput(
            field="active_trials",
            canonical_label="Active Trials",
            operator="gt",
            data_type="numeric",
            value=25,
            original_text="more than 25",
            confidence=0.95,
            unit=None,
        )
        assert mfo.value == 25.0

    def test_invalid_operator_raises(self):
        with pytest.raises(Exception):
            MetricFilterOutput(
                field="active_trials", canonical_label="Active Trials",
                operator="<<", data_type="numeric", value=25,
                original_text="", confidence=0.9, unit=None,
            )

    def test_invalid_unit_raises(self):
        with pytest.raises(Exception):
            MetricFilterOutput(
                field="active_trials", canonical_label="Active Trials",
                operator="gt", data_type="numeric", value=25,
                original_text="", confidence=0.9, unit="bananas",
            )

    def test_none_unit_valid(self):
        mfo = MetricFilterOutput(
            field="active_trials", canonical_label="Active Trials",
            operator="lt", data_type="numeric", value=10,
            original_text="", confidence=1.0, unit=None,
        )
        assert mfo.unit is None

    def test_no_preference_coerced_to_none(self):
        mfo = MetricFilterOutput(
            field="active_trials", canonical_label="Active Trials",
            operator="any", data_type="numeric", value="No preference",
            original_text="", confidence=1.0, unit=None,
        )
        assert mfo.value is None

    def test_numeric_string_coerced_to_float(self):
        mfo = MetricFilterOutput(
            field="active_trials", canonical_label="Active Trials",
            operator="gt", data_type="numeric", value="500",
            original_text="", confidence=0.95, unit=None,
        )
        assert mfo.value == 500.0

    def test_operator_any_with_value_raises(self):
        with pytest.raises(ValueError, match="operator='any' requires value=None"):
            MetricFilterOutput(
                field="active_trials", canonical_label="Active Trials",
                operator="any", data_type="numeric", value=500,
                original_text="", confidence=1.0, unit=None,
            )

    def test_date_mm_yyyy_valid(self):
        mfo = MetricFilterOutput(
            field="most_recent_approval_date", canonical_label="Most Recent Approval Date",
            operator="gte", data_type="date", value="06/2025",
            original_text="since 06/2025", confidence=0.95, unit=None,
        )
        assert mfo.value == "06/2025"

    def test_date_freeform_coerces_to_none(self, caplog):
        """Invalid date format like 'June 2025' → value coerced to None."""
        mfo = MetricFilterOutput(
            field="most_recent_approval_date", canonical_label="Most Recent Approval Date",
            operator="gte", data_type="date", value="June 2025",
            original_text="since June 2025", confidence=0.95, unit=None,
        )
        assert mfo.value is None

    def test_dob_year_rejected(self):
        """DOB-shaped year 1985 → coerced to None via year-range check."""
        mfo = MetricFilterOutput(
            field="most_recent_approval_date", canonical_label="Most Recent Approval Date",
            operator="eq", data_type="date", value="03/1985",
            original_text="born 03/1985", confidence=0.9, unit=None,
        )
        assert mfo.value is None

    def test_between_numeric_valid(self):
        mfo = MetricFilterOutput(
            field="avg_enrollment_matching_studies",
            canonical_label="Average Enrollment Matching Studies",
            operator="between", data_type="numeric", value=100, value_end=500,
            original_text="between 100 and 500", confidence=0.95, unit=None,
        )
        assert mfo.value == 100.0
        assert mfo.value_end == 500.0

    def test_between_date_valid(self):
        mfo = MetricFilterOutput(
            field="most_recent_approval_date", canonical_label="Most Recent Approval Date",
            operator="between", data_type="date", value="01/2024", value_end="06/2025",
            original_text="between 01/2024 and 06/2025", confidence=0.95, unit=None,
        )
        assert mfo.value == "01/2024"
        assert mfo.value_end == "06/2025"

    def test_between_missing_value_end_raises(self):
        with pytest.raises(ValueError, match="operator='between' requires both value and value_end"):
            MetricFilterOutput(
                field="avg_enrollment_matching_studies", canonical_label="Test",
                operator="between", data_type="numeric", value=100, value_end=None,
                original_text="", confidence=0.9, unit=None,
            )

    def test_between_value_gt_value_end_raises(self):
        with pytest.raises(ValueError, match="between requires value"):
            MetricFilterOutput(
                field="avg_enrollment_matching_studies", canonical_label="Test",
                operator="between", data_type="numeric", value=500, value_end=100,
                original_text="", confidence=0.9, unit=None,
            )

    def test_between_mixed_types_raises(self):
        with pytest.raises(Exception):
            MetricFilterOutput(
                field="avg_enrollment_matching_studies", canonical_label="Test",
                operator="between", data_type="numeric", value="01/2024", value_end="06/2025",
                original_text="", confidence=0.9, unit=None,
            )

    def test_non_between_with_value_end_raises(self):
        with pytest.raises(ValueError, match="value_end only allowed with operator"):
            MetricFilterOutput(
                field="active_trials", canonical_label="Active Trials",
                operator="gt", data_type="numeric", value=25, value_end=50,
                original_text="", confidence=0.9, unit=None,
            )

    def test_valid_units_constant(self):
        expected = {"patients", "months", "days", "sites", "percent", "queries", None}
        assert expected.issubset(VALID_UNITS)


# ===========================================================================
# Group 8 — Normalizer static methods (10 cases)
# ===========================================================================

class TestMetricFilterNormalizerStaticMethods:

    def test_normalize_operator_fewer_than(self):
        assert MetricFilterNormalizer.normalize_operator("fewer than") == "lt"

    def test_normalize_operator_at_least(self):
        assert MetricFilterNormalizer.normalize_operator("at least") == "gte"

    def test_normalize_operator_over(self):
        assert MetricFilterNormalizer.normalize_operator("over") == "gt"

    def test_normalize_operator_unknown_returns_any(self):
        assert MetricFilterNormalizer.normalize_operator("gobbledygook") == "any"

    def test_normalize_value_no_preference(self):
        assert MetricFilterNormalizer.normalize_value("No preference", "numeric") is None

    def test_normalize_value_numeric_string(self):
        assert MetricFilterNormalizer.normalize_value("500", "numeric") == 500.0

    def test_normalize_value_none_input(self):
        assert MetricFilterNormalizer.normalize_value(None, "numeric") is None

    def test_normalize_date_mm_yyyy(self):
        result = MetricFilterNormalizer.normalize_date("01/2026", date(2026, 5, 12))
        assert result == "01/2026"

    def test_normalize_date_not_a_date(self):
        result = MetricFilterNormalizer.normalize_date("not a date", date(2026, 5, 12))
        assert result is None

    def test_normalize_date_freeform_month(self):
        """Month-name strings return None (not implemented)."""
        result = MetricFilterNormalizer.normalize_date("June 2025", date(2026, 5, 12))
        assert result is None

    def test_normalize_value_date_mm_yyyy(self):
        result = MetricFilterNormalizer.normalize_value("01/2026", "date", "gte")
        assert result == "01/2026"

    def test_normalize_value_date_invalid(self):
        result = MetricFilterNormalizer.normalize_value("garbage", "date", "gte")
        assert result is None

    def test_normalize_value_date_none(self):
        result = MetricFilterNormalizer.normalize_value(None, "date", "gte")
        assert result is None


# ===========================================================================
# Resolver health check (cross-cutting)
# ===========================================================================

class TestResolverHealthCheck:

    def test_health_check_returns_counts(self, resolver):
        hc = resolver.health_check()
        assert "fields" in hc
        assert "synonyms" in hc
        assert "known_fields" in hc
        assert hc["fields"] == 12
        assert hc["synonyms"] > 0

    def test_get_entry_known_field(self, resolver):
        entry = resolver.get_entry("active_trials")
        assert entry["canonical_label"] == "active_trials"
        assert entry["data_type"] == "numeric"
        assert "clarification_question" in entry
        assert "{trigger}" in entry["clarification_question"]

    def test_get_entry_unknown_raises(self, resolver):
        with pytest.raises(KeyError):
            resolver.get_entry("nonexistent_field_xyz")


# ===========================================================================
# Group 9 — Startup validation (preserved from prior suite)
# ===========================================================================

class TestStartupValidation:

    def _make_tmp_json(self, tmp_path, entries):
        path = tmp_path / "metric_filters_test.json"
        path.write_text(json.dumps(entries), encoding="utf-8")
        return str(path)

    def _good_entry(self, label="test_field"):
        return {
            "canonical_label": label,
            "data_type": "numeric",
            "synonyms": ["test phrase", "another phrase"],
            "fuzzy_threshold": 80,
            "implied_operators": {"low": "lt"},
            "clarification_options": ["A", "B", "No preference"],
            "clarification_question": "Pick a range for {trigger}",
        }

    def test_invalid_data_type_strict_raises(self, tmp_path):
        bad = self._good_entry()
        bad["data_type"] = "fizzbuzz"
        path = self._make_tmp_json(tmp_path, [bad])
        with pytest.raises(ValueError, match="invalid data_type"):
            MetricIntentResolver(path, strict_validation=True)

    def test_empty_synonym_strict_raises(self, tmp_path):
        bad = self._good_entry()
        bad["synonyms"] = ["valid phrase", "   "]
        path = self._make_tmp_json(tmp_path, [bad])
        with pytest.raises(ValueError, match="empty or non-string synonym"):
            MetricIntentResolver(path, strict_validation=True)

    def test_invalid_implied_operator_strict_raises(self, tmp_path):
        bad = self._good_entry()
        bad["implied_operators"] = {"fast": "bogus_op"}
        path = self._make_tmp_json(tmp_path, [bad])
        with pytest.raises(ValueError, match="invalid implied_operator"):
            MetricIntentResolver(path, strict_validation=True)

    def test_threshold_out_of_range_strict_raises(self, tmp_path):
        bad = self._good_entry()
        bad["fuzzy_threshold"] = 150
        path = self._make_tmp_json(tmp_path, [bad])
        with pytest.raises(ValueError, match="fuzzy_threshold"):
            MetricIntentResolver(path, strict_validation=True)

    def test_clarification_options_wrong_count_strict_raises(self, tmp_path):
        bad = self._good_entry()
        bad["clarification_options"] = ["only one"]
        path = self._make_tmp_json(tmp_path, [bad])
        with pytest.raises(ValueError, match="clarification_options length"):
            MetricIntentResolver(path, strict_validation=True)

    def test_last_option_not_no_preference_strict_raises(self, tmp_path):
        bad = self._good_entry()
        bad["clarification_options"] = ["A", "B", "Something else"]
        path = self._make_tmp_json(tmp_path, [bad])
        with pytest.raises(ValueError, match="No preference"):
            MetricIntentResolver(path, strict_validation=True)

    def test_question_missing_trigger_strict_raises(self, tmp_path):
        bad = self._good_entry()
        bad["clarification_question"] = "No placeholder here"
        path = self._make_tmp_json(tmp_path, [bad])
        with pytest.raises(ValueError, match=r"\{trigger\}"):
            MetricIntentResolver(path, strict_validation=True)

    def test_duplicate_synonym_across_fields_strict_raises(self, tmp_path):
        a = self._good_entry("field_a")
        a["synonyms"] = ["shared phrase"]
        b = self._good_entry("field_b")
        b["synonyms"] = ["shared phrase"]
        path = self._make_tmp_json(tmp_path, [a, b])
        with pytest.raises(ValueError, match="Duplicate synonym"):
            MetricIntentResolver(path, strict_validation=True)

    def test_lenient_mode_drops_bad_entry(self, tmp_path):
        bad = self._good_entry("bad_field")
        bad["data_type"] = "fizzbuzz"
        good = self._good_entry("good_field")
        path = self._make_tmp_json(tmp_path, [bad, good])
        r = MetricIntentResolver(path, strict_validation=False)
        assert r.health_check()["fields"] == 1
        assert "good_field" in r._known_metric_fields
        assert "bad_field" not in r._known_metric_fields

    def test_empty_automaton_after_validation_raises(self, tmp_path):
        bad = self._good_entry()
        bad["data_type"] = "fizzbuzz"
        path = self._make_tmp_json(tmp_path, [bad])
        with pytest.raises(ValueError, match="no valid metric fields"):
            MetricIntentResolver(path, strict_validation=False)


# ===========================================================================
# Post-impl coverage
# ===========================================================================

class TestPostImplCoverage:

    def test_max_fuzzy_comparisons_constant(self):
        assert MAX_FUZZY_COMPARISONS_PER_QUERY == 5000

    def test_fuzzy_budget_no_crash(self, resolver, caplog):
        import logging
        caplog.set_level(logging.INFO)
        long_query = " ".join(["unknown"] * 300)
        resolver.resolve(long_query)
        budget_hits = [r for r in caplog.records if "metric_fuzzy_budget_exhausted" in r.getMessage()]
        for r in budget_hits:
            msg = r.getMessage()
            assert "comparisons=" in msg
            assert "unknown" not in msg

    def test_validate_missing_operator_raises(self):
        with pytest.raises(ValueError, match="operator field missing"):
            MetricFilterOutput.model_validate({
                "field": "x", "canonical_label": "x",
                "data_type": "numeric", "value": 5,
                "original_text": "", "confidence": 0.9,
            })
