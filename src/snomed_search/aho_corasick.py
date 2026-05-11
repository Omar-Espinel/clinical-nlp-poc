"""AhoCorasickStrategy: substring-scan SNOMED search via Aho-Corasick automaton."""

import logging
import re

import pandas as pd

from src.exceptions import StrategyError
from src.snomed_search.base import SNOMEDMatch

logger = logging.getLogger(__name__)

try:
    import ahocorasick as _ahocorasick
    _AC_AVAILABLE = True
except ImportError:
    _AC_AVAILABLE = False


class AhoCorasickStrategy:
    """
    Candidate extraction via Aho-Corasick multi-pattern substring scan.
    Exact hits (preferred_term) get confidence 0.99; synonym hits get 0.97.
    Word-boundary post-filter prevents sub-word false positives.

    All instance attributes after __init__ are read-only.
    Thread-safe: automaton.iter() is stateless per call.
    """

    name = "aho_corasick"

    def __init__(self, dictionary_path: str, **kwargs) -> None:
        if not _AC_AVAILABLE:
            raise StrategyError("pyahocorasick not installed")

        self._automaton = _ahocorasick.Automaton()
        self._ready = False
        self._load_csv(dictionary_path)
        self._automaton.make_automaton()
        self._ready = True
        logger.info(
            "AhoCorasickStrategy: automaton built with %d entries",
            len(self._automaton),
        )

    def _load_csv(self, csv_path: str) -> None:
        df = pd.read_csv(csv_path, dtype=str).fillna("")
        entry_count = 0
        for _, row in df.iterrows():
            concept_id = row["concept_id"].strip()
            preferred_term = row["preferred_term"].strip().lower()
            synonyms_raw = row["synonyms"].strip()

            value = (concept_id, preferred_term, "exact", len(preferred_term))
            # push_unique deduplicates; keep highest-priority value if collision
            if preferred_term not in self._automaton:
                self._automaton.add_word(preferred_term, value)
                entry_count += 1

            for syn in synonyms_raw.split("|"):
                syn_clean = syn.strip().lower()
                if syn_clean and syn_clean not in self._automaton:
                    self._automaton.add_word(
                        syn_clean,
                        (concept_id, preferred_term, "synonym", len(syn_clean)),
                    )
                    entry_count += 1

        logger.info("AhoCorasickStrategy: loaded %d dictionary entries", entry_count)

    def search(self, query: str) -> list[SNOMEDMatch]:
        """
        Candidate extraction: substring iteration via Aho-Corasick automaton.
        Word-boundary post-filter prevents 'cancer' matching inside 'pancreatic'.
        """
        query_lower = query.lower()
        hits: list[SNOMEDMatch] = []

        for end_inclusive, (concept_id, display, match_type, term_len) in self._automaton.iter(query_lower):
            start = end_inclusive - term_len + 1
            end = end_inclusive + 1

            if not self._is_word_bounded(query_lower, start, end):
                continue

            confidence = 0.99 if match_type == "exact" else 0.97
            hits.append(SNOMEDMatch(
                code=concept_id,
                display=display,
                match_type=match_type,
                confidence=confidence,
                original_text=query[start:end],
                span=(start, end),
                negated=False,
            ))

        logger.debug("AhoCorasickStrategy: %d candidates before dedup", len(hits))
        return self._dedup_longest_match(hits)

    def _is_word_bounded(self, text: str, start: int, end: int) -> bool:
        left_ok = start == 0 or not text[start - 1].isalnum()
        right_ok = end == len(text) or not text[end].isalnum()
        return left_ok and right_ok

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
            "dictionary_size": len(self._automaton),
        }
