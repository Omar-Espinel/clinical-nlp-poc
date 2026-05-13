"""Assembles final NLPOutput or ClarificationOutput from all pipeline component results."""

import html
import time
from typing import TYPE_CHECKING, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.filter_extractor import ExtractedFilters
from src.snomed_search.base import SNOMEDMatch
from src.normalizers.geo import GeoResult
from src.normalizers.metric import MetricFilterOutput
from src.sufficiency_gate import AmbiguousEntry, SufficiencyDecision

if TYPE_CHECKING:
    from src.conversation import ConversationSession


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
    """Complete NLP pipeline output — search result path."""
    model_config = ConfigDict(frozen=True)
    type: Literal["search"] = "search"
    snomed_terms: list[SNOMEDTermOutput]
    filters: FiltersOutput
    metric_filters: list[MetricFilterOutput] = Field(default_factory=list)
    metadata: MetadataOutput


class ClarificationOutput(BaseModel):
    """Output for a clarification turn — pipeline needs more info from the user."""
    model_config = ConfigDict(frozen=True)
    type: Literal["clarification"] = "clarification"
    question: str            # html.escape'd
    options: list[str]       # each html.escape'd
    canonical_query: str     # html.escape'd; for transparency
    turn_number: int         # 1-indexed clarification turn
    max_turns: int = 3
    metadata: MetadataOutput


_INVALID_STATE_VALUES = frozenset({"null", "none", "n/a", "na", "unknown", ""})


def render_question(
    entry: AmbiguousEntry,
    trigger: str,
    session: "ConversationSession",
) -> str:
    """Module-level helper: render the clarification question from a registry entry.

    Rendering is a presentation concern; it lives here rather than in sufficiency_gate.py.
    HIPAA: do NOT log the return value (it contains user-derived content).
    """
    prior_filters_str = session.summarize_known_filters()
    prefix = f"You're searching in {prior_filters_str} — " if prior_filters_str else ""
    template = entry.question_template
    return template.format(trigger=trigger, prior_filters=prefix).strip()


class ResponseAssembler:
    """Assembles the final NLPOutput or ClarificationOutput from pipeline component results."""

    MIN_CONFIDENCE = 0.60

    def build_clarification(
        self,
        decision: SufficiencyDecision,
        session: "ConversationSession",
        start_time: float,
    ) -> ClarificationOutput:
        """Build a ClarificationOutput from a non-sufficient SufficiencyDecision.

        All user-rendered strings are html.escape'd before being placed in the model.
        Called for both ambiguous_trigger and filters_without_condition paths.
        """
        entry = decision.matched_entry  # guaranteed non-None when sufficient=False
        trigger = decision.triggered_by or "your query"
        question_raw = render_question(entry, trigger, session)
        return ClarificationOutput(
            question=html.escape(question_raw),
            options=[html.escape(o) for o in entry.options],
            canonical_query=html.escape(session.canonical_query),
            turn_number=session.clarification_turn_count() + 1,
            max_turns=session.max_clarification_turns,
            metadata=MetadataOutput(
                processing_time_ms=int((time.perf_counter() - start_time) * 1000),
                total_snomed_matches=0,
                snomed_match_types={},
                negated_terms_excluded=0,
            ),
        )

    def assemble(
        self,
        filters: ExtractedFilters,
        snomed_matches: list[SNOMEDMatch],
        geo: GeoResult,
        start_time: float,
        metric_filters: Optional[list[MetricFilterOutput]] = None,
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
            city_value = filters.city.value
            city_conf = filters.city.confidence
            raw_state_values = filters.state.values
            if raw_state_values:
                # Filter out invalid sentinel strings
                state_values = [
                    s for s in raw_state_values
                    if s and s.strip().lower() not in _INVALID_STATE_VALUES
                ]
            else:
                state_values = []
            state_conf = filters.state.confidence
            state_is_region = filters.state.is_region

        filters_out = FiltersOutput(
            investigator_name=FilterFieldOutput(
                value=filters.investigator_name.value,
                confidence=filters.investigator_name.confidence,
            ),
            site_name=FilterFieldOutput(
                value=filters.site_name.value,
                confidence=filters.site_name.confidence,
            ),
            city=FilterFieldOutput(value=city_value, confidence=city_conf),
            state=StateFilterOutput(
                values=state_values,
                confidence=state_conf,
                is_region=state_is_region,
            ),
            phase=FilterFieldOutput(
                value=filters.phase.value,
                confidence=filters.phase.confidence,
            ),
        )

        processing_time_ms = int((time.perf_counter() - start_time) * 1000)

        metadata = MetadataOutput(
            processing_time_ms=processing_time_ms,
            total_snomed_matches=len(included),
            snomed_match_types=match_type_counts,
            negated_terms_excluded=negated_excluded,
        )

        return NLPOutput(
            snomed_terms=included,
            filters=filters_out,
            metric_filters=metric_filters or [],
            metadata=metadata,
        )
