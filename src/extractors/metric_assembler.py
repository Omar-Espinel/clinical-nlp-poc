"""
MetricFieldAssembler: converts MetricMatch list to dict[str, MetricFilterOutput].

No LLM. Direct conversion using MetricIntentResolver already-resolved data.
MetricIntentResolver has done all synonym matching and operator inference.
This module handles value extraction and output assembly only.

SNOMED-required metric enforcement:
  studies_matching_search and avg_enrollment_matching_studies require at
  least one qualifying SNOMED match in the query. If requirement not met:
  field excluded from output, added to snomed_required_unmet list for
  gate handling.

HIPAA: matched_text and value strings never logged.
"""

from __future__ import annotations
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from pydantic import ValidationError

from src.normalizers.metric import (
    MetricMatch,
    MetricFilterOutput,
    MetricFilterNormalizer,
)

logger = logging.getLogger(__name__)


SNOMED_REQUIRED_FIELDS: frozenset[str] = frozenset({
    "studies_matching_search",
    "avg_enrollment_matching_studies",
})

VALUE_WINDOW_CHARS: int = 30

_NUMERIC_PATTERN = re.compile(r'\b(\d{1,6}(?:\.\d{1,3})?)\b')
_DATE_MMYYYY = re.compile(r'\b(\d{1,2})[/\-](\d{4})\b')
_DATE_YYYYMM = re.compile(r'\b(\d{4})[/\-](\d{1,2})\b')


@dataclass
class AssemblerResult:
    metric_fields: dict[str, MetricFilterOutput] = field(default_factory=dict)
    snomed_required_unmet: list[str] = field(default_factory=list)


class MetricFieldAssembler:

    def _extract_numeric_near_span(
        self, query: str, span: tuple[int, int]
    ) -> Optional[float]:
        before_start = max(0, span[0] - VALUE_WINDOW_CHARS)
        after_end = min(len(query), span[1] + VALUE_WINDOW_CHARS)
        before = query[before_start:span[0]]
        after = query[span[1]:after_end]
        for window in (after, before):
            m = _NUMERIC_PATTERN.search(window)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    pass
        return None

    def _extract_date_near_span(
        self, query: str, span: tuple[int, int]
    ) -> Optional[str]:
        before_start = max(0, span[0] - VALUE_WINDOW_CHARS)
        after_end = min(len(query), span[1] + VALUE_WINDOW_CHARS)
        before = query[before_start:span[0]]
        after = query[span[1]:after_end]
        for window in (after, before):
            m = _DATE_MMYYYY.search(window)
            if m:
                raw = f"{m.group(1).zfill(2)}/{m.group(2)}"
                normalized = MetricFilterNormalizer.normalize_value(
                    raw, "date", "any"
                )
                if normalized is not None:
                    return str(normalized)
            m = _DATE_YYYYMM.search(window)
            if m:
                raw = f"{m.group(2).zfill(2)}/{m.group(1)}"
                normalized = MetricFilterNormalizer.normalize_value(
                    raw, "date", "any"
                )
                if normalized is not None:
                    return str(normalized)
        return None

    def assemble(
        self,
        metric_matches: list[MetricMatch],
        canonical_query: str,
        has_qualifying_snomed: bool,
    ) -> AssemblerResult:
        result = AssemblerResult()

        for match in metric_matches:
            if (
                match.canonical_field in SNOMED_REQUIRED_FIELDS
                and not has_qualifying_snomed
            ):
                result.snomed_required_unmet.append(match.canonical_field)
                continue

            operator = match.implied_operator

            raw_value: Optional[object] = None
            if match.data_type == "numeric":
                raw_value = self._extract_numeric_near_span(
                    canonical_query, match.span
                )
            elif match.data_type == "date":
                raw_value = self._extract_date_near_span(
                    canonical_query, match.span
                )

            if operator == "any" and raw_value is not None:
                operator = "eq"

            if raw_value is None:
                value = None
            else:
                value = MetricFilterNormalizer.normalize_value(
                    str(raw_value), match.data_type, operator
                )

            try:
                mfo = MetricFilterOutput(
                    field=match.canonical_field,
                    canonical_label=match.canonical_label,
                    operator=operator,
                    data_type=match.data_type,
                    value=value,
                    value_end=None,
                    original_text=match.matched_text,
                    confidence=match.confidence,
                    unit=None,
                )
                result.metric_fields[match.canonical_field] = mfo
            except ValidationError as exc:
                logger.info(
                    "metric_assembler: validation skip error_count=%d",
                    exc.error_count(),
                )
                continue

        logger.info(
            "metric_assembler: assembled=%d snomed_unmet=%d",
            len(result.metric_fields),
            len(result.snomed_required_unmet),
        )
        return result
