"""Tests for MetricFieldAssembler."""

import types
import pytest

from src.extractors.metric_assembler import MetricFieldAssembler
from src.normalizers.metric import MetricMatch


@pytest.fixture(scope="module")
def assembler() -> MetricFieldAssembler:
    return MetricFieldAssembler()


def _match(
    canonical_field: str,
    data_type: str,
    span: tuple[int, int],
    implied_operator: str = "any",
    canonical_label: str = "",
    matched_text: str = "",
) -> MetricMatch:
    return MetricMatch(
        canonical_field=canonical_field,
        canonical_label=canonical_label or canonical_field.replace("_", " ").title(),
        matched_text=matched_text or canonical_field,
        span=span,
        data_type=data_type,
        match_source="ac_synonym",
        confidence=1.0,
        implied_operator=implied_operator,
        implied_op_map=types.MappingProxyType({}),
    )


def test_numeric_with_implied_gt_and_explicit_value(assembler):
    """active_trials with implied gt and value 10 → operator=gt, value=10.0"""
    query = "more than 10 active trials"
    # "active trials" starts at index 13
    m = _match(
        canonical_field="active_trials",
        data_type="numeric",
        span=(13, 26),
        implied_operator="gt",
        canonical_label="Active Trials",
        matched_text="active trials",
    )
    result = assembler.assemble([m], query, has_qualifying_snomed=True)
    assert "active_trials" in result.metric_fields
    out = result.metric_fields["active_trials"]
    assert out.operator == "gt"
    assert out.value == 10.0
    assert result.snomed_required_unmet == []


def test_date_field_with_value(assembler):
    """date field with 06/2024 value in query → operator=gte, value=06/2024"""
    query = "approval since 06/2024 latest approval date"
    # "latest approval date" starts at index 23
    m = _match(
        canonical_field="most_recent_approval_date",
        data_type="date",
        span=(23, 43),
        implied_operator="gte",
        canonical_label="Most Recent Approval Date",
        matched_text="latest approval date",
    )
    result = assembler.assemble([m], query, has_qualifying_snomed=True)
    assert "most_recent_approval_date" in result.metric_fields
    out = result.metric_fields["most_recent_approval_date"]
    assert out.operator == "gte"
    assert out.value == "06/2024"


def test_snomed_required_absent(assembler):
    """studies_matching_search with no qualifying SNOMED → field excluded, flagged unmet"""
    query = "studies matching my search"
    m = _match(
        canonical_field="studies_matching_search",
        data_type="numeric",
        span=(0, 24),
        implied_operator="any",
        canonical_label="Studies Matching Search",
        matched_text="studies matching",
    )
    result = assembler.assemble([m], query, has_qualifying_snomed=False)
    assert result.metric_fields == {}
    assert result.snomed_required_unmet == ["studies_matching_search"]


def test_snomed_required_present(assembler):
    """studies_matching_search with qualifying SNOMED → field included, no unmet"""
    query = "studies matching my search"
    m = _match(
        canonical_field="studies_matching_search",
        data_type="numeric",
        span=(0, 24),
        implied_operator="any",
        canonical_label="Studies Matching Search",
        matched_text="studies matching",
    )
    result = assembler.assemble([m], query, has_qualifying_snomed=True)
    assert "studies_matching_search" in result.metric_fields
    assert result.snomed_required_unmet == []
