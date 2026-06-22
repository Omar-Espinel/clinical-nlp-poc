"""Deterministic filter extraction for clinical research queries.

No LLM. No external API. Fully deterministic. Thread-safe.
All components are read-only after __init__.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from src.normalizers.metric import MetricMatch, MetricFilterOutput

# Forward-only typing import — avoids the runtime circular dep with extractors.
# (DeterministicFilterExtractor lazy-imports the concrete classes in __init__.)
if False:  # TYPE_CHECKING shim that doesn't drag the dep at runtime
    from src.extractors.names import AmbiguousNameMatch  # noqa: F401

logger = logging.getLogger(__name__)


class FilterField(BaseModel):
    model_config = ConfigDict(frozen=True)
    value: Optional[str]
    confidence: float


class StateFilter(BaseModel):
    model_config = ConfigDict(frozen=True)
    values: list[str]
    confidence: float
    is_region: bool


class PhaseFilter(BaseModel):
    """Multi-value phase filter — supports conjunctions like 'Phase 2 or 3'.

    Shape: {"values": ["Phase 2", "Phase 3"], "confidence": 0.95}
    Replaces the old FilterField for phase throughout the pipeline.
    """
    model_config = ConfigDict(frozen=True)
    values: list[str]
    confidence: float


class ExtractedFilters(BaseModel):
    model_config = ConfigDict(frozen=True)
    investigator_name: FilterField
    site_name: FilterField
    city: FilterField
    state: StateFilter
    phase: PhaseFilter
    raw_response_length: int
    metric_fields: dict[str, MetricFilterOutput] = Field(default_factory=dict)


@dataclass
class ExtractionResult:
    """Carries ExtractedFilters plus auxiliary data from deterministic extraction.

    Replaces the LLM raw response. Eliminates thread-local side effects.
    Element type is `AmbiguousNameMatch` (from src.extractors.names) but
    declared as `list` here to avoid an import cycle.
    """
    filters: ExtractedFilters
    ambiguous_names: list = field(default_factory=list)
    snomed_required_unmet: list[str] = field(default_factory=list)


class DeterministicFilterExtractor:
    """Replaces the LLM-based FilterExtractor.

    Same public extract() interface. Returns ExtractionResult instead of
    ExtractedFilters directly. Pipeline unwraps filters from result.filters.

    No LLM. No external API calls. Fully deterministic. Thread-safe.
    All components are read-only after __init__.
    """

    def __init__(
        self,
        known_metric_fields: Optional[frozenset[str]] = None,
        geo_normalizer=None,
        institution_keywords_path: Optional[str] = None,
        geo_json_path: Optional[str] = None,
        snomed_known_terms: frozenset[str] = frozenset(),
    ) -> None:
        from src.extractors.phase import PhaseExtractor
        from src.extractors.names import NameExtractor
        from src.extractors.metric_assembler import MetricFieldAssembler

        if institution_keywords_path is None or geo_json_path is None:
            raise ValueError(
                "DeterministicFilterExtractor requires "
                "institution_keywords_path and geo_json_path"
            )

        self._phase = PhaseExtractor()
        self._names = NameExtractor(
            institution_keywords_path=institution_keywords_path,
            geo_json_path=geo_json_path,
            snomed_known_terms=snomed_known_terms,
        )
        self._metric_assembler = MetricFieldAssembler()
        self._known_metric_fields = known_metric_fields
        self._geo = geo_normalizer  # shared reference; not owned

        logger.info(
            "DeterministicFilterExtractor initialized: known_metric_fields=%d",
            len(known_metric_fields) if known_metric_fields else 0,
        )

    def extract(
        self,
        canonical_query: str,
        metric_matches: Optional[list[MetricMatch]] = None,
        qualifying_snomed_count: int = 0,
    ) -> ExtractionResult:
        """Deterministic extraction pipeline.

        Returns ExtractionResult — pipeline reads .filters for ExtractedFilters,
        .ambiguous_names for name gate, .snomed_required_unmet for metric gate.

        Thread-safe: no instance state modified. All results in return value.
        HIPAA: canonical_query never logged.
        """
        # Step 1: phase
        phase_result = self._phase.extract(canonical_query)

        # Step 2: geo spans (for name exclusion)
        geo_spans = self._names.find_geo_spans(canonical_query)

        # Step 3: metric spans (for name exclusion)
        metric_spans = [m.span for m in (metric_matches or [])]

        # Step 4: name extraction on residual text
        phase_span = [phase_result.span] if phase_result.span else []
        safe_excluded = []
        for gs in geo_spans:
            geo_text = canonical_query[gs[0]:gs[1]].lower()
            # Keep a geo span out of the exclusion list if its text is a
            # whole-word component (prefix, suffix, interior, or the whole)
            # of a known multiword site, so e.g. "Toronto" does not block the
            # "University of Toronto" site match.
            is_site_component = any(
                mw == geo_text
                or mw.startswith(geo_text + " ")
                or mw.endswith(" " + geo_text)
                or (" " + geo_text + " ") in mw
                for mw in self._names._multiword_set
            )
            if not is_site_component:
                safe_excluded.append(gs)
        excluded = safe_excluded + metric_spans + phase_span
        name_result = self._names.extract(canonical_query, excluded_spans=excluded)

        # Step 5: geo signal extraction for city/state
        geo_result = self._names._extract_city_state(canonical_query)

        # Step 6: metric assembly
        has_snomed = qualifying_snomed_count > 0
        assembler_result = self._metric_assembler.assemble(
            metric_matches or [],
            canonical_query,
            has_snomed,
        )

        # Step 7: build StateFilter
        if geo_result.is_region and geo_result.region_states:
            state_filter = StateFilter(values=geo_result.region_states, confidence=0.95, is_region=True)
        elif geo_result.is_region and geo_result.city_raw:
            if self._geo is not None and geo_result.original_region_term:
                geo_norm = self._geo.normalize(geo_result.original_region_term, None)
                state_filter = StateFilter(values=geo_norm.states if geo_norm.states else [], confidence=geo_norm.confidence, is_region=True)
            else:
                state_filter = StateFilter(values=[], confidence=0.0, is_region=True)
        elif geo_result.is_region:
            # Region matched but contributes no positive states (e.g. a fully
            # negated region like "not in New England") — still flag is_region.
            state_filter = StateFilter(values=[], confidence=0.95, is_region=True)
        elif geo_result.state_raw:
            state_filter = StateFilter(values=[geo_result.state_raw], confidence=0.90, is_region=False)
        else:
            state_filter = StateFilter(values=[], confidence=0.0, is_region=False)

        # Step 8: assemble ExtractedFilters
        # Expand conjunction phases (e.g. "Phase 2/3") into multi-value PhaseFilter
        if phase_result.value is not None:
            phase_values = phase_result.values if phase_result.values else [phase_result.value]
        else:
            phase_values = []
        phase_filter = PhaseFilter(
            values=phase_values,
            confidence=phase_result.confidence,
        )

        filters = ExtractedFilters(
            investigator_name=FilterField(
                value=name_result.investigator_name,
                confidence=name_result.investigator_confidence,
            ),
            site_name=FilterField(
                value=name_result.site_name,
                confidence=name_result.site_confidence,
            ),
            city=FilterField(
                value=geo_result.city_raw,
                confidence=0.90 if geo_result.city_raw else 0.0,
            ),
            state=state_filter,
            phase=phase_filter,
            raw_response_length=len(canonical_query),
            metric_fields=assembler_result.metric_fields,
        )

        city_found = geo_result.city_raw is not None
        state_found = geo_result.state_raw is not None or geo_result.is_region
        is_region = geo_result.is_region
        region_states = len(geo_result.region_states)
        logger.info(
            "deterministic_extractor: phase_found=%s "
            "inv_conf=%.2f site_conf=%.2f city_found=%s state_found=%s "
            "is_region=%s region_states=%d ambiguous_names=%d metric_fields=%d",
            phase_result.value is not None,
            name_result.investigator_confidence,
            name_result.site_confidence,
            city_found,
            state_found,
            is_region,
            region_states,
            len(name_result.ambiguous_names),
            len(assembler_result.metric_fields),
        )

        return ExtractionResult(
            filters=filters,
            ambiguous_names=name_result.ambiguous_names,
            snomed_required_unmet=assembler_result.snomed_required_unmet,
        )
