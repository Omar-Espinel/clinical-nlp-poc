"""
PhaseExtractor: deterministic phase detection from clinical query text.

No LLM. No external dependencies beyond stdlib re and rapidfuzz.
All regex patterns pre-compiled at MODULE LOAD TIME — zero runtime
regex compilation. All quantifiers explicitly bounded (ReDoS prevention).
Max query length is 500 chars (enforced by Preprocessor upstream).

HIPAA: phase value is structured output. Not logged per consistency rule.
"""

from __future__ import annotations
import re
import logging
from dataclasses import dataclass
from typing import Optional

from rapidfuzz.fuzz import token_sort_ratio

logger = logging.getLogger(__name__)


@dataclass
class PhaseResult:
    value: Optional[str]
    confidence: float
    span: Optional[tuple[int, int]]
    # Multi-value support for conjunctions ("Phase 2 or 3", "Phase 2/3").
    # When populated, contains all resolved phase values.
    # When None, callers should fall back to [value] if value is not None.
    values: Optional[list[str]] = None


# (compiled_pattern, normalized_value, confidence)
# Order matters — more specific patterns must precede general ones.
_EXACT_PATTERNS: list[tuple[re.Pattern, str, float]] = [
    (re.compile(r'\bphase\s*(?:iii|3)\b', re.IGNORECASE), "Phase 3", 0.95),
    (re.compile(r'\bp[\-\s]?3\b', re.IGNORECASE), "Phase 3", 0.95),
    (re.compile(r'\bphase\s*(?:ii|2)b\b', re.IGNORECASE), "Phase 2b", 0.95),
    (re.compile(r'\bphase\s*(?:ii|2)a\b', re.IGNORECASE), "Phase 2a", 0.95),
    (re.compile(r'\bphase\s*(?:ii|2)\/(?:iii|3)\b', re.IGNORECASE), "Phase 2/3", 0.95),
    (re.compile(r'\bphase\s*(?:i|1)\/(?:ii|2)\b', re.IGNORECASE), "Phase 1/2", 0.95),
    (re.compile(r'\bphase\s*(?:ii|2)\b', re.IGNORECASE), "Phase 2", 0.95),
    (re.compile(r'\bp[\-\s]?2\b', re.IGNORECASE), "Phase 2", 0.95),
    (re.compile(r'\bphase\s*(?:ib|1b)\b', re.IGNORECASE), "Phase 1b", 0.95),
    (re.compile(r'\bphase\s*(?:i|1)(?![\/\w])\b', re.IGNORECASE), "Phase 1", 0.95),
    (re.compile(r'\bp[\-\s]?1\b', re.IGNORECASE), "Phase 1", 0.95),
    (re.compile(r'\bphase\s*(?:iv|4)\b', re.IGNORECASE), "Phase 4", 0.95),
    (re.compile(r'\bp[\-\s]?4\b', re.IGNORECASE), "Phase 4", 0.95),
]

# Conjunction patterns: "Phase 2 or 3", "Phase 2 and 3", "Phase 2/3",
# "Phase II or III", etc.  Each maps to a (phase_a, phase_b) pair.
# Ordered most-specific first.  Captured groups: (phase_num_a, phase_num_b).
_CONJUNCTION_PATTERNS: list[tuple[re.Pattern, str, str, float]] = [
    # Phase 2/3  or Phase II/III  (slash — already in _EXACT_PATTERNS as single value;
    # here we expand to TWO values instead)
    (re.compile(r'\bphase\s*(?:ii|2)\s*/\s*(?:iii|3)\b', re.IGNORECASE), "Phase 2", "Phase 3", 0.95),
    (re.compile(r'\bphase\s*(?:i|1)\s*/\s*(?:ii|2)\b', re.IGNORECASE), "Phase 1", "Phase 2", 0.95),
    # Phase 2 or 3 / Phase II or III
    (re.compile(r'\bphase\s*(?:ii|2)\s+or\s+(?:iii|3)\b', re.IGNORECASE), "Phase 2", "Phase 3", 0.95),
    (re.compile(r'\bphase\s*(?:i|1)\s+or\s+(?:ii|2)\b', re.IGNORECASE), "Phase 1", "Phase 2", 0.95),
    (re.compile(r'\bphase\s*(?:iii|3)\s+or\s+(?:iv|4)\b', re.IGNORECASE), "Phase 3", "Phase 4", 0.95),
    # Phase 2 and 3 / Phase II and III
    (re.compile(r'\bphase\s*(?:ii|2)\s+and\s+(?:iii|3)\b', re.IGNORECASE), "Phase 2", "Phase 3", 0.95),
    (re.compile(r'\bphase\s*(?:i|1)\s+and\s+(?:ii|2)\b', re.IGNORECASE), "Phase 1", "Phase 2", 0.95),
    (re.compile(r'\bphase\s*(?:iii|3)\s+and\s+(?:iv|4)\b', re.IGNORECASE), "Phase 3", "Phase 4", 0.95),
]

_ALIAS_PATTERNS: list[tuple[re.Pattern, str, float]] = [
    (re.compile(r'\bpivotal\b', re.IGNORECASE), "Phase 3", 0.85),
    (re.compile(r'\bregistrational\b', re.IGNORECASE), "Phase 3", 0.85),
    (re.compile(r'\bfirst[\-\s]in[\-\s]human\b', re.IGNORECASE), "Phase 1", 0.85),
    (re.compile(r'\bfih\b', re.IGNORECASE), "Phase 1", 0.85),
    (re.compile(r'\bdose[\-\s]escalation\b', re.IGNORECASE), "Phase 1", 0.85),
    (re.compile(r'\bdose[\-\s]expansion\b', re.IGNORECASE), "Phase 1/2", 0.85),
    (re.compile(r'\bproof[\-\s]of[\-\s]concept\b', re.IGNORECASE), "Phase 2", 0.85),
    (re.compile(r'\bpoc[\-\s]trial\b', re.IGNORECASE), "Phase 2", 0.85),
    (re.compile(r'\bearly[\-\s]phase[\-\s](?:i|1)\b', re.IGNORECASE), "Phase 1", 0.85),
    (re.compile(r'\blate[\-\s]phase[\-\s](?:ii|2)\b', re.IGNORECASE), "Phase 2", 0.85),
]

_ALL_PATTERNS: list[tuple[re.Pattern, str, float]] = _EXACT_PATTERNS + _ALIAS_PATTERNS

_FUZZY_PHASE_ALIASES: list[str] = [
    "phase 1", "phase 2", "phase 3", "phase 4",
    "phase 1/2", "phase 2/3", "pivotal", "first in human",
    "dose escalation", "proof of concept",
]

FUZZY_PHASE_THRESHOLD: int = 80

_TOKEN_PATTERN = re.compile(r'\S+')


class PhaseExtractor:
    def __init__(self) -> None:
        pass

    def extract(self, query: str) -> PhaseResult:
        if not query:
            return PhaseResult(value=None, confidence=0.0, span=None)

        # Check conjunction patterns first (more specific than single-phase patterns)
        for pat, phase_a, phase_b, conf in _CONJUNCTION_PATTERNS:
            m = pat.search(query)
            if m:
                return PhaseResult(
                    value=phase_a,
                    confidence=conf,
                    span=m.span(),
                    values=[phase_a, phase_b],
                )

        for pat, value, conf in _ALL_PATTERNS:
            m = pat.search(query)
            if m:
                return PhaseResult(value=value, confidence=conf, span=m.span())

        return self._fuzzy_extract(query)

    def _fuzzy_extract(self, query: str) -> PhaseResult:
        tokens = list(_TOKEN_PATTERN.finditer(query))
        if not tokens:
            return PhaseResult(value=None, confidence=0.0, span=None)

        windows: list[tuple[str, tuple[int, int]]] = []
        for i, tok in enumerate(tokens):
            windows.append((tok.group(), tok.span()))
            if i + 1 < len(tokens):
                nxt = tokens[i + 1]
                start, end = tok.start(), nxt.end()
                windows.append((query[start:end], (start, end)))

        best_score = 0
        best_window: Optional[tuple[str, tuple[int, int]]] = None
        best_alias: Optional[str] = None

        for win_text, win_span in windows:
            already_exact = False
            for pat, _, _ in _ALL_PATTERNS:
                if pat.search(win_text):
                    already_exact = True
                    break
            if already_exact:
                continue
            for alias in _FUZZY_PHASE_ALIASES:
                score = token_sort_ratio(win_text.lower(), alias.lower())
                if score >= FUZZY_PHASE_THRESHOLD and score > best_score:
                    best_score = score
                    best_window = (win_text, win_span)
                    best_alias = alias

        if best_window is None or best_alias is None:
            return PhaseResult(value=None, confidence=0.0, span=None)

        # FIX 5: record original_span BEFORE building working query.
        # Return original_span verbatim — discard working-copy offsets.
        original_span = best_window[1]

        working = query[:original_span[0]] + best_alias + query[original_span[1]:]

        for pat, value, _ in _ALL_PATTERNS:
            if pat.search(working):
                return PhaseResult(value=value, confidence=0.70, span=original_span)

        return PhaseResult(value=None, confidence=0.0, span=None)
