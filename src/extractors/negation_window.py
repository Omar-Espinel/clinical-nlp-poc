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

    Negation does not cross clause boundaries: the backward scan is
    scoped to the current clause (after the nearest preceding comma,
    semicolon, or sentence-ending punctuation), so a "not"/"neither"
    governing an earlier clause does not suppress this span.

    Bounded scan — safe against adversarial input.
    """
    prefix = query[:span_start]
    boundary = max(
        prefix.rfind(','), prefix.rfind(';'), prefix.rfind('.'),
        prefix.rfind('!'), prefix.rfind('?'),
    )
    clause = prefix[boundary + 1:] if boundary != -1 else prefix
    clause_tokens = clause.split()
    window = " ".join(clause_tokens[-window_tokens:])
    if _NEG_CUE_PATTERN.search(window):
        return True
    neither_m = re.search(r'\bneither\b', clause, re.IGNORECASE)
    if neither_m:
        return True
    return False
