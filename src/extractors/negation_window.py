"""
Filter-level negation window checker.

Distinct from the NegEx SNOMED annotator in src/snomed_search/negation.py.
This module checks whether a geo or filter extraction span falls
inside a negation window in the query, so negated locations and
investigators are silently dropped rather than returned as positive
filters.

HIPAA: this module receives query strings but logs nothing.
"""
from __future__ import annotations
import re

_NEG_CUE_PATTERN = re.compile(
    r'\b(?:not|neither|nor|except|excluding|outside|avoid|avoiding'
    r'|without|rather\s{1,3}than|other\s{1,3}than|instead\s{1,3}of)\b',
    re.IGNORECASE,
)

def is_negated_span(query: str, span_start: int,
                    window_tokens: int = 8) -> bool:
    """Return True if span_start falls inside a negation window.

    Scans backward up to window_tokens whitespace-delimited tokens
    from span_start for a negation cue. Also detects the
    "neither...nor" pattern by checking whether "neither" appears
    before span_start in the same clause (no intervening sentence
    boundary).

    Bounded scan — safe against adversarial input.
    """
    prefix = query[:span_start]
    prefix_tokens = prefix.split()
    window = " ".join(prefix_tokens[-window_tokens:])
    if _NEG_CUE_PATTERN.search(window):
        return True
    neither_m = re.search(r'\bneither\b', prefix, re.IGNORECASE)
    if neither_m:
        between = query[neither_m.end():span_start]
        if not re.search(r'[.;!?]', between):
            return True
    return False
