"""Assembles final NLPOutput from all pipeline component results."""

import time
from typing import Optional

from pydantic import BaseModel, ConfigDict

from src.extractor import ExtractionResult, FilterField
from src.snomed_resolver import SNOMEDMatch
from src.geo_normalizer import GeoResult


class SNOMEDTermOutput(BaseModel):
    """A single resolved SNOMED CT term in the final output."""
    model_config = ConfigDict(frozen=True)
    code: str
    display: str
    match_type: str
    confidence: float
    original_text: str
    negated: bool


class FilterFieldOutput(BaseModel):
    """A single-value structured filter field (investigator, site, city, phase)."""
    model_config = ConfigDict(frozen=True)
    value: Optional[str]
    confidence: float


class StateFilterOutput(BaseModel):
    """State filter field — holds one state for city queries or many for regional queries."""
    model_config = ConfigDict(frozen=True)
    values: list[str]
    confidence: float
    is_region: bool


class FiltersOutput(BaseModel):
    """All structured filters in the final output."""
    model_config = ConfigDict(frozen=True)
    investigator_name: FilterFieldOutput
    site_name: FilterFieldOutput
    city: FilterFieldOutput
    state: StateFilterOutput
    phase: FilterFieldOutput


class MetadataOutput(BaseModel):
    """Processing metadata in the final output."""
    model_config = ConfigDict(frozen=True)
    processing_time_ms: int
    total_snomed_matches: int
    snomed_match_types: dict[str, int]
    negated_terms_excluded: int


class NLPOutput(BaseModel):
    """Complete NLP pipeline output."""
    model_config = ConfigDict(frozen=True)
    snomed_terms: list[SNOMEDTermOutput]
    filters: FiltersOutput
    metadata: MetadataOutput


_INVALID_STATE_VALUES = frozenset({"null", "none", "n/a", "na", "unknown", ""})


class ResponseAssembler:
    """Assembles the final NLPOutput from all pipeline component results."""

    MIN_CONFIDENCE = 0.60

    def assemble(
        self,
        extraction: ExtractionResult,
        snomed_matches: list[SNOMEDMatch],
        geo: GeoResult,
        start_time: float,
    ) -> NLPOutput:
        """Build NLPOutput from extraction, SNOMED matches, geo result, and timing.

        Negated SNOMED matches are excluded from snomed_terms and counted separately.
        Non-negated matches below MIN_CONFIDENCE are also excluded.
        Geo values replace raw extraction city/state when geo.confidence >= 0.60.
        For multi-state regions, state.values contains all covered states.
        """
        included: list[SNOMEDTermOutput] = []
        negated_excluded = 0
        match_type_counts: dict[str, int] = {}

        snomed_matches = sorted(snomed_matches, key=lambda m: m.confidence, reverse=True)

        for match in snomed_matches:
            if match.negated:
                negated_excluded += 1
                continue
            if match.confidence < self.MIN_CONFIDENCE:
                continue
            included.append(
                SNOMEDTermOutput(
                    code=match.code,
                    display=match.display,
                    match_type=match.match_type,
                    confidence=match.confidence,
                    original_text=match.original_text,
                    negated=match.negated,
                )
            )
            match_type_counts[match.match_type] = match_type_counts.get(match.match_type, 0) + 1

        seen_codes: set[str] = set()
        deduped: list[SNOMEDTermOutput] = []
        for term in included:
            if term.code not in seen_codes:
                seen_codes.add(term.code)
                deduped.append(term)
        included = deduped

        # Geo integration
        if geo.confidence >= self.MIN_CONFIDENCE:
            city_value = geo.city
            city_conf = geo.confidence
            state_values = geo.states
            state_conf = geo.confidence
            state_is_region = geo.is_region
        else:
            city_value = extraction.city.value
            city_conf = extraction.city.confidence
            raw_state = extraction.state.value
            if raw_state and raw_state.strip().lower() not in _INVALID_STATE_VALUES:
                state_values = [raw_state]
            else:
                state_values = []
            state_conf = extraction.state.confidence
            state_is_region = False

        filters = FiltersOutput(
            investigator_name=FilterFieldOutput(
                value=extraction.investigator_name.value,
                confidence=extraction.investigator_name.confidence,
            ),
            site_name=FilterFieldOutput(
                value=extraction.site_name.value,
                confidence=extraction.site_name.confidence,
            ),
            city=FilterFieldOutput(value=city_value, confidence=city_conf),
            state=StateFilterOutput(
                values=state_values,
                confidence=state_conf,
                is_region=state_is_region,
            ),
            phase=FilterFieldOutput(
                value=extraction.phase.value,
                confidence=extraction.phase.confidence,
            ),
        )

        processing_time_ms = int((time.time() - start_time) * 1000)

        metadata = MetadataOutput(
            processing_time_ms=processing_time_ms,
            total_snomed_matches=len(included),
            snomed_match_types=match_type_counts,
            negated_terms_excluded=negated_excluded,
        )

        return NLPOutput(
            snomed_terms=included,
            filters=filters,
            metadata=metadata,
        )
