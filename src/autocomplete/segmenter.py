from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class ContextType(Enum):
    PHASE = "phase"
    GEO = "geo"


PHASE_PATTERN = re.compile(r'\bPhase\s+\d+(?:/\d+)*\b', re.IGNORECASE)

_PHASES_LITERAL = frozenset({
    "phase 1", "phase 2", "phase 3", "phase 4",
    "phase 1/2", "phase 2/3", "phase 1/2/3",
    "phase i", "phase ii", "phase iii", "phase iv",
})


@dataclass(frozen=True)
class SegmentResult:
    context_tokens: list[str]
    active_prefix: str
    context_types: list[ContextType]
    last_token_complete: bool


class QuerySegmenter:
    """Splits partial query into context tokens and active prefix.

    Only extracts context from the leading portion of the query.
    Mid-query geo tokens like 'trials in Boston' are handled by the
    downstream filter extractor, not here. The segmenter identifies
    what the user has committed to versus what they are still typing.

    geo_keys must exclude 2-char abbreviations because they are too
    short to reliably distinguish from medical abbreviations.
    """

    def __init__(self, geo_keys: frozenset[str]) -> None:
        self._geo_keys = frozenset(k for k in geo_keys if len(k) >= 3)
        self._geo_keys_sorted = sorted(self._geo_keys, key=len, reverse=True)

    def segment(self, query: str) -> SegmentResult:
        """Split query into context tokens and active prefix.

        Extracts phase patterns first since they can appear anywhere,
        then checks if the remaining string starts with a known geo key,
        then treats the remainder as the active prefix.
        Returns SegmentResult with empty active_prefix when prefix is
        under 3 chars, signalling the orchestrator to return no suggestions.
        HIPAA: query content is not logged.
        """
        remainder = query.strip()
        context_tokens: list[str] = []
        context_types: list[ContextType] = []

        phase_matches = list(PHASE_PATTERN.finditer(remainder))
        for m in reversed(phase_matches):
            context_tokens.insert(0, m.group())
            context_types.insert(0, ContextType.PHASE)
            remainder = (remainder[:m.start()] + remainder[m.end():]).strip()

        geo_found = True
        while geo_found:
            geo_found = False
            remainder_lower = remainder.lower()
            for key in self._geo_keys_sorted:
                if remainder_lower.startswith(key):
                    after = remainder[len(key):]
                    if after == "" or after[0] == " ":
                        context_tokens.append(remainder[:len(key)])
                        context_types.append(ContextType.GEO)
                        remainder = after.strip()
                        geo_found = True
                        break

        active_prefix = remainder.strip()
        last_token_complete = active_prefix.lower() in self._geo_keys

        if len(active_prefix) < 3:
            return SegmentResult(
                context_tokens=context_tokens,
                active_prefix="",
                context_types=context_types,
                last_token_complete=last_token_complete,
            )

        return SegmentResult(
            context_tokens=context_tokens,
            active_prefix=active_prefix,
            context_types=context_types,
            last_token_complete=last_token_complete,
        )
