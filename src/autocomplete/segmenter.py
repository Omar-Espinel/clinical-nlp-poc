from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class ContextType(Enum):
    PHASE = "phase"
    GEO = "geo"
    SNOMED = "snomed"


PHASE_PATTERN = re.compile(r'\bPhase\s+\d+(?:/\d+)*\b', re.IGNORECASE)

_PHASES_LITERAL = frozenset({
    "phase 1", "phase 2", "phase 3", "phase 4",
    "phase 1/2", "phase 2/3", "phase 1/2/3",
    "phase i", "phase ii", "phase iii", "phase iv",
})

# Structural delimiters that separate committed context from the active tail.
# " for " is intentionally excluded — "trials for diabetes" is a valid
# clinical construction and would cause false splits if included.
# Patterns require surrounding whitespace (except comma) to avoid matching
# inside words. Order does not affect correctness because we find the
# rightmost match across all patterns.
_BOUNDARY_PATTERNS: list[re.Pattern] = [
    re.compile(r',\s*', re.IGNORECASE),
    re.compile(r'\s+in\s+', re.IGNORECASE),
    re.compile(r'\s+by\s+', re.IGNORECASE),
    re.compile(r'\s+at\s+', re.IGNORECASE),
    re.compile(r'\s+excluding\s+', re.IGNORECASE),
    re.compile(r'\s+near\s+', re.IGNORECASE),
]

# Compound phase patterns: "Phase 2 and 3", "Phase 2 or 3",
# "Phases 2 and 3". Captured groups are the individual phase numbers
# or roman numerals. Used by _detect_compound_phase().
_COMPOUND_PHASE_PATTERN = re.compile(
    r'\bPhases?\s+(\d+(?:/\d+)*)\s+(?:and|or)\s+(\d+(?:/\d+)*)\b',
    re.IGNORECASE,
)

# Noise words that appear at the end of an active prefix but carry no
# autocomplete signal. Stripped before sending to tiers.
# Anchored to end-of-string with \s* to handle trailing whitespace.
# Word-boundary \b prevents stripping "only" inside "only child syndrome".
_NOISE_SUFFIX_PATTERN = re.compile(
    r'\s*\b(only|sites?\s+only|just|please|us\s+sites?\s+only|'
    r'sites?|locations?)\s*$',
    re.IGNORECASE,
)

# Conservative filler prefixes stripped before segmentation analysis.
# These are kept verbatim in committed_prefix_raw for completion assembly.
_FILLER_PREFIX = re.compile(
    r'^(studies\s+on|research\s+on|trials\s+on|a\s+study\s+in|'
    r'a\s+trial\s+in|find\s+me|looking\s+for|find|show\s+me|'
    r'search\s+for)\s+',
    re.IGNORECASE,
)

# Characters stripped when normalising text for known-term matching.
_STRIP_PUNCT = ",.;: "


@dataclass(frozen=True)
class SegmentResult:
    context_tokens: list[str]
    active_prefix: str
    context_types: list[ContextType]
    last_token_complete: bool
    # Verbatim slice of the original query that precedes active_prefix.
    # Used by _format_suggestion to reconstruct the full completion string
    # while preserving the user's original casing and punctuation.
    # Never logged anywhere in the system.
    committed_prefix_raw: str = field(default="")


@dataclass(frozen=True)
class NormalizedPrefix:
    """Result of active prefix normalization.

    search_text is what the tiers actually search on — noise stripped,
    compound phase reduced to first component for tier search.

    is_compound_phase signals the orchestrator to expand phase results:
    when True, compound_phases contains the individual canonical phase
    strings (e.g. ["Phase 2", "Phase 3"]) and the orchestrator must
    generate one suggestion per phase rather than one suggestion total.

    original is preserved verbatim for completion assembly — the
    orchestrator replaces only the active_prefix portion of the
    completion with the canonical phase string, keeping committed
    context intact.

    HIPAA: this dataclass is never logged. Only its boolean fields
    and lengths are safe to log.
    """
    search_text: str
    original: str
    is_compound_phase: bool = False
    compound_phases: list[str] = field(default_factory=list)


class QuerySegmenter:
    """Splits a partial query into committed context and active prefix.

    Uses right-anchored segmentation with two strategies:

    Strategy A — rightmost structural boundary:
        Scans for the rightmost delimiter (, in by at excluding near).
        Everything left of it is committed context; everything right is
        the active prefix being typed.

    Strategy B — last-token split on known terms:
        Used when no structural boundary exists. Checks whether the query
        minus its last whitespace-delimited token matches a known SNOMED
        term, geo key, or phase literal. If yes, the last token is the
        active prefix.

    Falls back to treating the full string as the active prefix when
    neither strategy finds a committed boundary — this preserves the
    original behaviour for bare single-term queries like "esophagus".

    geo_keys must exclude 2-char abbreviations because they are too
    short to reliably distinguish from medical abbreviations.

    HIPAA: query content is never logged anywhere in this class.
    """

    def __init__(
        self,
        geo_keys: frozenset[str],
        *,
        snomed_known_terms: frozenset[str] | None = None,
    ) -> None:
        self._geo_keys = frozenset(k for k in geo_keys if len(k) >= 3)
        self._geo_keys_sorted = sorted(self._geo_keys, key=len, reverse=True)

        # Normalised lowercase SNOMED display terms for committed-context
        # detection. Stored as a frozenset for O(1) whole-phrase lookup.
        # Multi-word terms are kept whole so "cancer" does not falsely match
        # inside "lung cancer" — only exact normalised phrases match.
        self._snomed_terms: frozenset[str] = (
            frozenset(t.lower().strip() for t in snomed_known_terms)
            if snomed_known_terms
            else frozenset()
        )

        # Pre-built set of all individual words that appear in any known
        # SNOMED term. Used in _extract_context_signals for a fast first-pass
        # filter before attempting whole-phrase substring detection.
        # This avoids iterating the full 90k term set on every segment call.
        self._snomed_word_set: frozenset[str] = frozenset(
            word
            for term in self._snomed_terms
            for word in term.split()
        )

    def segment(self, query: str) -> SegmentResult:
        """Split query into committed context and active prefix.

        Returns SegmentResult with empty active_prefix when the derived
        active prefix is under 3 chars, signalling the orchestrator to
        return no suggestions.

        HIPAA: query content is not logged.
        """
        original = query.strip()

        # Strip leading filler words for segmentation analysis only.
        # The raw filler text is preserved in committed_prefix_raw so the
        # completion assembler can reconstruct the full string verbatim.
        filler_match = _FILLER_PREFIX.match(original)
        filler_raw = filler_match.group(0) if filler_match else ""
        working = original[len(filler_raw):]

        # Strategy A: rightmost structural boundary delimiter.
        committed_raw, active_prefix, boundary_found = (
            self._split_at_rightmost_boundary(working)
        )

        # Strategy B: last-token split when no structural boundary found.
        if not boundary_found:
            committed_raw, active_prefix = self._split_at_last_token(working)

        full_committed_raw = filler_raw + committed_raw

        context_tokens, context_types = self._extract_context_signals(
            committed_raw
        )

        active_prefix = active_prefix.strip()
        last_token_complete = active_prefix.lower() in self._geo_keys

        if len(active_prefix) < 3:
            return SegmentResult(
                context_tokens=context_tokens,
                active_prefix="",
                context_types=context_types,
                last_token_complete=last_token_complete,
                committed_prefix_raw=full_committed_raw,
            )

        return SegmentResult(
            context_tokens=context_tokens,
            active_prefix=active_prefix,
            context_types=context_types,
            last_token_complete=last_token_complete,
            committed_prefix_raw=full_committed_raw,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _split_at_rightmost_boundary(
        self, text: str
    ) -> tuple[str, str, bool]:
        """Find the rightmost structural delimiter and split there.

        Returns (committed_raw, active_prefix, found).
        committed_raw includes the delimiter text verbatim so the
        completion assembler can reconstruct the full string exactly.
        """
        best_end: int = -1

        for pattern in _BOUNDARY_PATTERNS:
            for m in pattern.finditer(text):
                if m.end() > best_end:
                    best_end = m.end()

        if best_end == -1:
            return "", text, False

        committed_raw = text[:best_end]
        active_prefix = text[best_end:]
        return committed_raw, active_prefix, True

    def _split_at_last_token(self, text: str) -> tuple[str, str]:
        """Fall back to last-whitespace-token split when no delimiter found.

        Checks whether text minus its last word matches a known SNOMED
        term, geo key, or phase literal. If yes, the last word alone is
        the active prefix. Also checks whether text minus its last TWO
        words matches, to handle cases like "Myocardial Infarction Michig"
        where the committed term is two words.

        When no known-term match is found, returns the full text as the
        active prefix with an empty committed string — preserving the
        original behaviour for bare single-term queries.
        """
        # Whole-text guard: if the entire text is itself a known term or
        # phase literal (e.g. a bare "Phase 3" with nothing typed after it),
        # treat it all as committed context with no active prefix. Preserves
        # the pre-existing behaviour where a completed bare phase/term yields
        # no active prefix and therefore no suggestions.
        if self._is_known_term(text):
            return text, ""

        parts = text.rsplit(None, 1)
        if len(parts) < 2:
            return "", text

        candidate_committed = parts[0].strip()
        candidate_active = parts[1]   # always the single last token

        if self._is_known_term(candidate_committed):
            return parts[0] + " ", candidate_active

        # Two-word look-back: check if text minus last TWO words is a known
        # term. candidate_active remains the single last token only.
        parts2 = candidate_committed.rsplit(None, 1)
        if len(parts2) == 2:
            two_word_committed = parts2[0].strip()
            if self._is_known_term(two_word_committed):
                return parts2[0] + " ", candidate_active

        return "", text

    def _is_known_term(self, text: str) -> bool:
        """Return True if text matches a known SNOMED, geo, or phase token.

        Strips trailing punctuation before matching so "esophagus,"
        matches "esophagus". Uses O(1) frozenset lookup — not substring
        search — so there is no false match of short terms inside longer
        ones.
        """
        normalised = text.lower().strip().rstrip(_STRIP_PUNCT)
        if normalised in self._snomed_terms:
            return True
        if normalised in self._geo_keys:
            return True
        if normalised in _PHASES_LITERAL:
            return True
        return False

    def _extract_context_signals(
        self, committed_text: str
    ) -> tuple[list[str], list[ContextType]]:
        """Extract phase, geo, and SNOMED signals from committed text.

        Reads committed_text to produce context_tokens and context_types
        for the ranker's context_bonus. Does not mutate the text.

        SNOMED detection uses a two-step approach to avoid iterating
        90k terms on every call:
          Step 1 — fast word-set check: any word from _snomed_word_set
                   present in the committed text words?
          Step 2 — only if step 1 hits: scan _snomed_terms for a whole-
                   phrase match using word-boundary-safe comparison.
        This keeps the common case (no SNOMED word found) at O(words_in_committed)
        instead of O(90k).

        HIPAA: committed_text is not logged.
        """
        context_tokens: list[str] = []
        context_types: list[ContextType] = []

        for m in PHASE_PATTERN.finditer(committed_text):
            context_tokens.append(m.group())
            context_types.append(ContextType.PHASE)

        normalised_committed = committed_text.lower().rstrip(_STRIP_PUNCT)

        for key in self._geo_keys_sorted:
            if key in normalised_committed:
                context_tokens.append(key)
                context_types.append(ContextType.GEO)

        # Fast pre-filter: skip SNOMED scan entirely if no SNOMED word
        # appears in the committed text at all.
        committed_words = set(normalised_committed.split())
        if committed_words & self._snomed_word_set:
            for term in self._snomed_terms:
                # Whole-phrase match: term must appear as a substring bounded
                # by start/end or non-word characters to prevent "cancer"
                # matching inside "lung cancer statistics".
                pattern = r'(?<!\w)' + re.escape(term) + r'(?!\w)'
                if re.search(pattern, normalised_committed):
                    context_tokens.append(term)
                    context_types.append(ContextType.SNOMED)
                    break  # one SNOMED signal is sufficient for ranker bonus

        return context_tokens, context_types

    def normalize_active_prefix(self, active_prefix: str) -> NormalizedPrefix:
        """Normalize active prefix for tier search.

        Two normalizations applied in order:

        1. Compound phase detection: "Phase 2 and 3" or "Phase 2 or 3"
           is recognized as a multi-value phase expression. Returns
           is_compound_phase=True with compound_phases=["Phase 2","Phase 3"]
           and search_text="Phase 2" (first component for tier search).
           The orchestrator expands this into one suggestion per phase.

        2. Noise suffix stripping: trailing noise words (only, sites only,
           US Sites only, just, please) are stripped from the active prefix
           before tier search. "gout US Sites only" → search_text="gout".
           The original is preserved for completion reconstruction.

        If neither normalization applies, search_text == original == active_prefix.

        HIPAA: active_prefix content is not logged.
        """
        # Step 1: compound phase detection.
        m = _COMPOUND_PHASE_PATTERN.search(active_prefix)
        if m:
            part1 = m.group(1)
            part2 = m.group(2)
            phase1 = f"Phase {part1}"
            phase2 = f"Phase {part2}"
            return NormalizedPrefix(
                search_text=phase1,
                original=active_prefix,
                is_compound_phase=True,
                compound_phases=[phase1, phase2],
            )

        # Step 2: noise suffix stripping.
        stripped = _NOISE_SUFFIX_PATTERN.sub("", active_prefix).strip()
        if stripped and stripped.lower() != active_prefix.lower().strip():
            return NormalizedPrefix(
                search_text=stripped,
                original=active_prefix,
                is_compound_phase=False,
                compound_phases=[],
            )

        return NormalizedPrefix(
            search_text=active_prefix,
            original=active_prefix,
            is_compound_phase=False,
            compound_phases=[],
        )
