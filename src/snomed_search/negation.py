from __future__ import annotations

import logging
import re
from dataclasses import replace
from typing import Optional

from src.snomed_search.base import SNOMEDMatch

logger = logging.getLogger(__name__)

NEGATION_CUES_PRE: list[str] = [
    "no", "not", "without", "denies", "denied", "rules out", "ruled out",
    "history of", "h/o", "free of", "absence of", "absent", "neither",
    "negative for", "no evidence of", "no signs of",
]
NEGATION_CUES_POST: list[str] = [
    "unlikely", "ruled out", "negative", "denied",
]
PSEUDO_NEGATION_PHRASES: list[str] = [
    # Longer phrases must appear before shorter sub-phrases they contain.
    "no contraindication for",
    "no change in",
    "no further",
    "not only",
]
# Comma intentionally absent: clinical syntax chains conditions with commas inside the
# same negated clause (e.g. "no history of cardiac issues, diabetes, stroke").
SCAN_STOP_PUNCT: frozenset[str] = frozenset(".;!?\n")
WINDOW_SIZE: int = 5


class NegationAnnotator:
    """
    NegEx-style rule-based negation detector.
    Operates on character-offset SNOMEDMatch.span values.
    No LLM. Deterministic. Stateless — safe for concurrent calls.

    Comma is intentionally NOT a scan-stop character (see proposal §3.7.5).
    Sentence-boundary stops: period, semicolon, !, ?, newline.
    """

    def annotate(
        self,
        query: str,
        matches: list[SNOMEDMatch],
    ) -> list[SNOMEDMatch]:
        """
        Returns a new list where each SNOMEDMatch.negated is set per NegEx rules.
        Input SNOMEDMatch objects are frozen dataclasses; new objects created via replace().
        Does not mutate the input list or any input SNOMEDMatch.
        """
        tokens: list[tuple[str, int, int]] = self._tokenize_with_offsets(query)
        logger.debug(
            "NegationAnnotator.annotate: %d tokens, %d matches",
            len(tokens),
            len(matches),
        )
        annotated: list[SNOMEDMatch] = []
        for m in matches:
            other_spans = [o.span for o in matches if o is not m]
            is_neg = self._is_negated(query, tokens, m, other_spans)
            annotated.append(replace(m, negated=is_neg))
        return annotated

    def _is_negated(
        self,
        query: str,
        tokens: list[tuple[str, int, int]],
        match: SNOMEDMatch,
        other_match_spans: list[tuple[int, int]],
    ) -> bool:
        """
        Core NegEx logic for a single SNOMEDMatch.

        Steps:
          1. Locate the match-start token via span[0].
          2. Check pseudo-negation overlap FIRST — suppresses negation classification.
          3. Pre-window scan: tokens [max(0, anchor-WINDOW_SIZE)..anchor).
          4. Post-window scan: tokens (match_last_token+1..match_last_token+WINDOW_SIZE].

        other_match_spans: spans of sibling matches that act as scan barriers.
        Negation scope does not cross an intervening SNOMEDMatch (standard NegEx rule).

        Returns False if span is not aligned to any token boundary.
        """
        match_start_token_idx: Optional[int] = self._find_token_index(
            tokens, match.span[0]
        )
        if match_start_token_idx is None:
            return False

        if self._has_pseudo_negation_near(query, match.span):
            return False

        if self._scan_window_for_cue(
            tokens=tokens,
            anchor_idx=match_start_token_idx,
            cues=NEGATION_CUES_PRE,
            direction="pre",
            barrier_spans=other_match_spans,
        ):
            return True

        match_end_token_idx = self._find_token_index(tokens, match.span[1] - 1)
        if match_end_token_idx is None:
            match_end_token_idx = match_start_token_idx

        if self._scan_window_for_cue(
            tokens=tokens,
            anchor_idx=match_end_token_idx,
            cues=NEGATION_CUES_POST,
            direction="post",
            barrier_spans=other_match_spans,
        ):
            return True

        return False

    def _tokenize_with_offsets(self, text: str) -> list[tuple[str, int, int]]:
        """
        Returns list of (token_text, char_start, char_end_exclusive).
        Splits on whitespace only; punctuation is kept attached to its token.
        Stop-punct detection inspects the last character of each token.

        Uses re.finditer(r'\\S+') to avoid the cursor-drift bug that text.split() +
        text.index(token, cursor) produces on repeated tokens (M1 fix).
        """
        return [
            (m.group(), m.start(), m.end())
            for m in re.finditer(r'\S+', text)
        ]

    def _find_token_index(
        self,
        tokens: list[tuple[str, int, int]],
        char_offset: int,
    ) -> Optional[int]:
        """
        Returns the index of the token whose span contains char_offset.
        A token (text, start, end) contains char_offset iff start <= char_offset < end.
        Returns None if no token contains the offset.
        """
        for idx, (_text, start, end) in enumerate(tokens):
            if start <= char_offset < end:
                return idx
        return None

    def _has_pseudo_negation_near(
        self,
        query: str,
        match_span: tuple[int, int],
    ) -> bool:
        """
        Returns True if any PSEUDO_NEGATION_PHRASES appears in
        query[max(0, match_span[0]-60) : match_span[1]].

        60-char lookback is generous enough to catch multi-word pseudo-negation
        phrases appearing several tokens before the match.
        """
        lookback_start = max(0, match_span[0] - 60)
        context_region = query[lookback_start: match_span[1]].lower()
        for phrase in PSEUDO_NEGATION_PHRASES:
            if phrase in context_region:
                return True
        return False

    def _scan_window_for_cue(
        self,
        tokens: list[tuple[str, int, int]],
        anchor_idx: int,
        cues: list[str],
        direction: str,
        barrier_spans: Optional[list[tuple[int, int]]] = None,
    ) -> bool:
        """
        Scans up to WINDOW_SIZE tokens in `direction` from anchor_idx.
        Stops before a token whose last character is in SCAN_STOP_PUNCT.
        Stops before a token that falls inside any span in barrier_spans —
        an intervening SNOMEDMatch resets negation scope (standard NegEx rule).

        Pre direction: scans anchor_idx-1, anchor_idx-2, ... (closest-first so
        stop-punct at the nearest sentence boundary halts correctly). Texts are
        reversed back to left-to-right order before phrase matching.

        Post direction: scans anchor_idx+1, anchor_idx+2, ...

        Multi-word cues: substring match on joined window text.
        Single-word cues: whole-word regex match.

        B1 fix: build pre-window indices as a plain list rather than a raw range
        so anchor_idx=0 yields an empty list reliably.
        """
        if direction == "pre":
            window_token_indices = list(
                range(max(0, anchor_idx - WINDOW_SIZE), anchor_idx)
            )[::-1]
        else:
            window_token_indices = range(
                anchor_idx + 1, min(len(tokens), anchor_idx + WINDOW_SIZE + 1)
            )

        barrier_spans = barrier_spans or []

        window_token_texts: list[str] = []
        for idx in window_token_indices:
            token_text, tok_start, tok_end = tokens[idx]
            if token_text[-1] in SCAN_STOP_PUNCT:
                break
            # An intervening match from the same query resets negation scope.
            if any(bs <= tok_start and tok_end <= be for bs, be in barrier_spans):
                break
            window_token_texts.append(token_text.lower())

        if not window_token_texts:
            return False

        if direction == "pre":
            window_token_texts = list(reversed(window_token_texts))

        window_text = " ".join(window_token_texts)

        for cue in cues:
            if " " in cue:
                if cue in window_text:
                    return True
            else:
                if re.search(rf"\b{re.escape(cue)}\b", window_text):
                    return True

        return False
