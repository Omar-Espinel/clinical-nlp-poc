"""NGramLookupStrategy: token-window n-gram SNOMED search (exact + synonym only)."""

import logging
import re

import pandas as pd

from src.snomed_search.base import SNOMEDMatch

logger = logging.getLogger(__name__)


class NGramLookupStrategy:
    """
    Candidate extraction: enumerate all 1..max_n token windows; check
    exact_index then synonym_index per window. No fuzzy or semantic stages.

    All instance attributes after __init__ are read-only. Thread-safe.
    """

    name = "ngram_lookup"

    def __init__(self, dictionary_path: str, max_n: int = 4, **kwargs) -> None:
        self._exact_index: dict[str, dict] = {}
        self._synonym_index: dict[str, dict] = {}
        self._max_n = max_n
        self._ready = False
        self._load_csv(dictionary_path)
        self._ready = True
        logger.info(
            "NGramLookupStrategy: loaded %d exact terms, %d synonyms",
            len(self._exact_index), len(self._synonym_index),
        )

    def _load_csv(self, csv_path: str) -> None:
        df = pd.read_csv(csv_path, dtype=str).fillna("")
        for _, row in df.iterrows():
            concept_id = row["concept_id"].strip()
            preferred_term = row["preferred_term"].strip().lower()
            synonyms_raw = row["synonyms"].strip()
            record = {"concept_id": concept_id, "preferred_term": preferred_term}
            self._exact_index[preferred_term] = record
            for syn in synonyms_raw.split("|"):
                syn_clean = syn.strip().lower()
                if syn_clean:
                    self._synonym_index[syn_clean] = record

    def search(self, query: str) -> list[SNOMEDMatch]:
        """
        Candidate extraction: enumerate all 1..max_n token windows;
        check exact_index then synonym_index per window.
        """
        tokens = self._tokenize_with_offsets(query)
        hits: list[SNOMEDMatch] = []

        for n in range(self._max_n, 0, -1):
            for i in range(len(tokens) - n + 1):
                window = tokens[i: i + n]
                window_text = " ".join(t[0] for t in window).lower()
                span_start = window[0][1]
                span_end = window[-1][2]

                if window_text in self._exact_index:
                    record = self._exact_index[window_text]
                    hits.append(SNOMEDMatch(
                        code=record["concept_id"],
                        display=record["preferred_term"],
                        match_type="exact",
                        confidence=0.99,
                        original_text=query[span_start:span_end],
                        span=(span_start, span_end),
                        negated=False,
                    ))
                    continue

                if window_text in self._synonym_index:
                    record = self._synonym_index[window_text]
                    hits.append(SNOMEDMatch(
                        code=record["concept_id"],
                        display=record["preferred_term"],
                        match_type="synonym",
                        confidence=0.97,
                        original_text=query[span_start:span_end],
                        span=(span_start, span_end),
                        negated=False,
                    ))

        logger.debug("NGramLookupStrategy: %d candidates before dedup", len(hits))
        return self._dedup_longest_match(hits)

    def _tokenize_with_offsets(self, query: str) -> list[tuple[str, int, int]]:
        return [
            (m.group(), m.start(), m.end())
            for m in re.finditer(r'\S+', query)
        ]

    def _dedup_longest_match(self, hits: list[SNOMEDMatch]) -> list[SNOMEDMatch]:
        """
        Keep longest non-contained span per character region.
        Sort: span_length DESC, confidence DESC, span_start ASC.
        Drop any hit whose span is entirely contained within a kept hit's span.
        """
        if not hits:
            return []

        sorted_hits = sorted(
            hits,
            key=lambda m: (-(m.span[1] - m.span[0]), -m.confidence, m.span[0]),
        )

        kept: list[SNOMEDMatch] = []
        kept_spans: list[tuple[int, int]] = []

        for candidate in sorted_hits:
            cs, ce = candidate.span
            contained = any(ks <= cs and ce <= ke for ks, ke in kept_spans)
            if not contained:
                kept.append(candidate)
                kept_spans.append((cs, ce))

        return kept

    def health_check(self) -> dict:
        return {
            "ready": self._ready,
            "name": self.name,
            "dictionary_size": len(self._exact_index),
        }
