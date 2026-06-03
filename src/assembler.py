"""Assembles final NLPOutput or ClarificationOutput from all pipeline component results."""

import csv
import html
import json
import re
import time
from pathlib import Path
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


class PhaseFilterOutput(BaseModel):
    """Phase filter field — supports multi-value conjunctions ('Phase 2 or 3').

    Shape: {"values": ["Phase 2", "Phase 3"], "confidence": 0.95}
    """
    model_config = ConfigDict(frozen=True)
    values: list[str]
    confidence: float


class FiltersOutput(BaseModel):
    """All structured filters in the final output."""
    model_config = ConfigDict(frozen=True)
    investigator_name: FilterFieldOutput
    site_name: FilterFieldOutput
    city: FilterFieldOutput
    state: StateFilterOutput
    phase: PhaseFilterOutput


class MetadataOutput(BaseModel):
    """Processing metadata in the final output."""
    model_config = ConfigDict(frozen=True)
    processing_time_ms: int
    total_snomed_matches: int
    snomed_match_types: dict[str, int]
    negated_terms_excluded: int


class FlagItem(BaseModel):
    """A single warning flag for the query summary."""
    model_config = ConfigDict(frozen=True)
    # field: one of the known category strings — server-defined, not user-derived
    field: str
    # message: a server-defined constant; NEVER derived from user input
    message: str
    confidence: float


class QuerySummary(BaseModel):
    """Human-readable interpretation of the assembled query — purely additive, never gates results."""
    model_config = ConfigDict(frozen=True)
    # label: HTML-escaped interpretation string built from resolved terms/filters; never raw query text
    label: str
    interpreted_terms: list[str]
    interpreted_filters: dict[str, str]
    flags: list[FlagItem]
    # unrecognized_terms: HTML-escaped tokens that survived all known-term/geo/phase filters
    unrecognized_terms: list[str]
    has_warnings: bool
    flag_count: int
    unrecognized_term_count: int


class NLPOutput(BaseModel):
    """Complete NLP pipeline output — search result path."""
    model_config = ConfigDict(frozen=True)
    type: Literal["search"] = "search"
    snomed_terms: list[SNOMEDTermOutput]
    filters: FiltersOutput
    metric_filters: list[MetricFilterOutput] = Field(default_factory=list)
    metadata: MetadataOutput
    # query_summary: purely additive — existing consumers can safely ignore this field
    query_summary: Optional[QuerySummary] = None


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

# ---------------------------------------------------------------------------
# QuerySummary assembly helpers — loaded lazily at first call
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent

# Phase token patterns — tokens like "phase", "phase 1", "i", "ii", "iii", "iv",
# "1", "2", "3", "4" that should not count as unrecognized terms.
_PHASE_TOKEN_PATTERN = re.compile(
    r"^(phase|[1-4]|i{1,3}v?|vi{0,3})$", re.IGNORECASE
)

# Word pattern for unrecognized-term scan — same bounded pattern as preflight
_WORD_PATTERN_SUMMARY = re.compile(r"\b[A-Za-z][A-Za-z'\-]{1,30}\b")

# Function words from preflight — reused here; defined inline to avoid circular import.
# These match the _FUNCTION_WORDS set in src/extractors/preflight.py exactly.
_PREFLIGHT_FUNCTION_WORDS: frozenset[str] = frozenset({
    "with", "from", "that", "this", "have", "been", "will", "what", "when",
    "where", "which", "there", "their", "about", "more", "also", "into",
    "than", "then", "some", "could", "would", "should", "other", "after",
    "find", "show", "pull", "give", "search", "studies", "trials", "research",
    "study",
})


def _load_geo_keys() -> frozenset[str]:
    """Return all top-level city/state/region keys from geo_canonical.json.

    Used to skip geo terms when scanning for unrecognized tokens.
    Never logs file content.
    """
    try:
        with open(_PROJECT_ROOT / "data" / "geo_canonical.json", encoding="utf-8") as fh:
            data = json.load(fh)
        keys: set[str] = set()
        for section_key in ("cities", "states", "regions"):
            for k in data.get(section_key, {}):
                # Split multi-word keys so individual tokens are also covered
                keys.add(k.lower())
                for word in k.lower().split():
                    keys.add(word)
        return frozenset(keys)
    except Exception:
        return frozenset()


def _load_snomed_known_terms() -> frozenset[str]:
    """Return all preferred_term + synonym tokens from snomed_clinical_trials.csv.

    Mirrors the logic in NLPPipeline._build_known_terms_from_csv.
    Never logs file content.
    """
    try:
        terms: set[str] = set()
        with open(
            _PROJECT_ROOT / "data" / "snomed_clinical_trials.csv", encoding="utf-8"
        ) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                preferred = (row.get("preferred_term") or "").strip().lower()
                if preferred:
                    terms.add(preferred)
                    # Also add individual words from multi-word terms
                    for word in preferred.split():
                        if len(word) >= 4:
                            terms.add(word)
                syns_raw = (row.get("synonyms") or "").strip().lower()
                if syns_raw:
                    for syn in syns_raw.split("|"):
                        syn_clean = syn.strip()
                        if syn_clean:
                            terms.add(syn_clean)
                            for word in syn_clean.split():
                                if len(word) >= 4:
                                    terms.add(word)
        return frozenset(t for t in terms if len(t) >= 4)
    except Exception:
        return frozenset()


# Lazy-loaded singletons — populated on first assemble_query_summary() call
_GEO_KEYS: Optional[frozenset[str]] = None
_SNOMED_KNOWN_TERMS: Optional[frozenset[str]] = None


def _get_geo_keys() -> frozenset[str]:
    global _GEO_KEYS
    if _GEO_KEYS is None:
        _GEO_KEYS = _load_geo_keys()
    return _GEO_KEYS


def _get_snomed_known_terms() -> frozenset[str]:
    global _SNOMED_KNOWN_TERMS
    if _SNOMED_KNOWN_TERMS is None:
        _SNOMED_KNOWN_TERMS = _load_snomed_known_terms()
    return _SNOMED_KNOWN_TERMS


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
        canonical_query: Optional[str] = None,
        sufficiency_decision: Optional[SufficiencyDecision] = None,
    ) -> NLPOutput:
        """Build NLPOutput from extraction, SNOMED matches, geo result, and timing.

        Negated SNOMED matches are excluded from snomed_terms and counted separately.
        Non-negated matches below MIN_CONFIDENCE are also excluded.
        Geo values replace raw extraction city/state when geo.confidence >= 0.60.
        For multi-state regions, state.values contains all covered states.
        canonical_query and sufficiency_decision are used to build the optional
        query_summary — they do not affect any other assembled fields.
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
            phase=PhaseFilterOutput(
                values=filters.phase.values,
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

        # Build query_summary when canonical_query is provided; purely additive.
        query_summary: Optional[QuerySummary] = None
        if canonical_query is not None:
            query_summary = self.assemble_query_summary(
                snomed_matches=snomed_matches,
                filters_out=filters_out,
                sufficiency_decision=sufficiency_decision,
                canonical_query=canonical_query,
            )

        return NLPOutput(
            snomed_terms=included,
            filters=filters_out,
            metric_filters=metric_filters or [],
            metadata=metadata,
            query_summary=query_summary,
        )

    def assemble_query_summary(
        self,
        snomed_matches: list[SNOMEDMatch],
        filters_out: FiltersOutput,
        sufficiency_decision: Optional[SufficiencyDecision],
        canonical_query: str,
    ) -> QuerySummary:
        """Build QuerySummary from qualifying SNOMED matches, filters, and canonical query.

        All user-derived strings are html.escape'd before inclusion.
        FlagItem.message values are server-defined constants — no escaping needed.
        HIPAA: canonical_query, filter values, SNOMED display strings, and
        unrecognized_terms must NEVER appear in any log statement from this method.
        """
        # ── Qualifying SNOMED matches (confidence >= 0.60, not negated) ────────
        qualifying: list[SNOMEDMatch] = [
            m for m in snomed_matches
            if not m.negated and m.confidence >= self.MIN_CONFIDENCE
        ]
        # Deduplicate by code (same order as assemble())
        seen_codes: set[str] = set()
        deduped_qualifying: list[SNOMEDMatch] = []
        for m in qualifying:
            if m.code not in seen_codes:
                seen_codes.add(m.code)
                deduped_qualifying.append(m)

        # ── interpreted_terms: HTML-escaped SNOMED display values ───────────────
        # User-derived SNOMED display strings require html.escape before inclusion.
        interpreted_terms: list[str] = [
            html.escape(m.display) for m in deduped_qualifying
        ]

        # ── interpreted_filters: non-null filter fields, HTML-escaped ───────────
        interpreted_filters: dict[str, str] = {}
        phase_vals = filters_out.phase.values
        if phase_vals:
            interpreted_filters["phase"] = html.escape(", ".join(phase_vals))
        if filters_out.investigator_name.value:
            interpreted_filters["investigator"] = html.escape(
                filters_out.investigator_name.value
            )
        if filters_out.site_name.value:
            interpreted_filters["site"] = html.escape(filters_out.site_name.value)
        if filters_out.city.value:
            interpreted_filters["city"] = html.escape(filters_out.city.value)
        state_vals = filters_out.state.values
        if state_vals:
            interpreted_filters["state"] = html.escape(", ".join(state_vals))

        # ── label: join SNOMED display + filter values in fixed order ────────────
        # Fixed order: SNOMED terms · phase · investigator · site · city · state
        label_parts: list[str] = list(interpreted_terms)
        for key in ("phase", "investigator", "site", "city", "state"):
            if key in interpreted_filters:
                label_parts.append(interpreted_filters[key])
        label: str = " · ".join(label_parts)

        # ── flags ────────────────────────────────────────────────────────────────
        flags: list[FlagItem] = []

        # Low-confidence investigator (field value present but confidence < 0.75)
        inv = filters_out.investigator_name
        if inv.value is not None and inv.confidence < 0.75:
            flags.append(
                FlagItem(
                    field="investigator",
                    message="Matched with low confidence",
                    confidence=inv.confidence,
                )
            )

        # Low-confidence site (field value present but confidence < 0.75)
        site = filters_out.site_name
        if site.value is not None and site.confidence < 0.75:
            flags.append(
                FlagItem(
                    field="site",
                    message="Matched with low confidence",
                    confidence=site.confidence,
                )
            )

        # Low-confidence SNOMED matches (0.60 <= confidence < 0.72, not negated)
        snomed_flag_count = 0
        for m in deduped_qualifying:
            if snomed_flag_count >= 3:
                break
            if m.confidence < 0.72:
                flags.append(
                    FlagItem(
                        field="snomed",
                        message="Matched with low confidence",
                        confidence=m.confidence,
                    )
                )
                snomed_flag_count += 1

        # Ambiguous name flags from SufficiencyDecision (Phase 2 passthrough)
        if (
            sufficiency_decision is not None
            and sufficiency_decision.ambiguous_name_flags
        ):
            for name in sufficiency_decision.ambiguous_name_flags:
                flags.append(
                    FlagItem(
                        field="investigator_or_site",
                        # Server-defined constant; name is NOT included in the message
                        message="Role unclear — investigator or site?",
                        confidence=0.0,
                    )
                )

        # ── unrecognized_terms: noun-like tokens not in any known set ────────────
        # Scan canonical_query for tokens that are not known SNOMED terms, geo terms,
        # phase tokens, or preflight function words. Tokens appearing in any SNOMED
        # match's original_text are also excluded (already recognized).
        # HIPAA: these tokens are user-derived — never log them.
        geo_keys = _get_geo_keys()
        snomed_known = _get_snomed_known_terms()

        # Collect all original_text substrings from SNOMED matches for exclusion
        snomed_match_texts: set[str] = {
            m.original_text.lower() for m in snomed_matches if m.original_text
        }

        unrecognized_terms: list[str] = []
        for token_match in _WORD_PATTERN_SUMMARY.finditer(canonical_query):
            if len(unrecognized_terms) >= 5:
                break
            token = token_match.group()
            if len(token) < 4:
                continue
            token_lower = token.lower()
            # Skip preflight function words
            if token_lower in _PREFLIGHT_FUNCTION_WORDS:
                continue
            # Skip geo terms
            if token_lower in geo_keys:
                continue
            # Skip phase tokens ("phase", "1"-"4", roman numerals)
            if _PHASE_TOKEN_PATTERN.match(token_lower):
                continue
            # Skip known SNOMED terms
            if token_lower in snomed_known:
                continue
            # Skip if this token appears in any SNOMED match's original_text
            if any(token_lower in ot for ot in snomed_match_texts):
                continue
            # Survived — HTML-escape and cap at 50 chars
            escaped = html.escape(token)[:50]
            unrecognized_terms.append(escaped)

        has_warnings: bool = len(flags) > 0 or len(unrecognized_terms) > 0

        return QuerySummary(
            label=label,
            interpreted_terms=interpreted_terms,
            interpreted_filters=interpreted_filters,
            flags=flags,
            unrecognized_terms=unrecognized_terms,
            has_warnings=has_warnings,
            flag_count=len(flags),
            unrecognized_term_count=len(unrecognized_terms),
        )
