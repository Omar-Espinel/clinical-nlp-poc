"""
PreflightMandatoryCheck: fast pre-extraction gate (<2ms).

Runs BEFORE parallel extraction launch in pipeline.py.
Rejects queries with zero mandatory term signals before expensive
SNOMED embedding search is launched.

DECOUPLING: Completely decoupled from strategy internals. Receives a
frozenset[str] of known SNOMED terms at init time. Never imports or
references any SNOMED strategy class.

REJECTION RULE: Reject ONLY when ALL seven signals are absent:
  A. SNOMED term signals (from known_terms frozenset)
  B. Person prefix signals
  C. Person suffix signals
  D. Institution keyword signals
  E. Context signals (by, at, investigator, site, pi)
  F. Token-based ambiguity signal (FIX 2): case-insensitive token NOT
     in _SECONDARY_TOKENS skiplist, NOT in any signal A-E category,
     and NOT part of an unambiguous multi-word geo phrase.
  G. Capitalized token signal (Fix 6, Phase 1): a token of length >= 4
     that starts with an uppercase letter AND is not in _FUNCTION_WORDS
     AND is not a geo stoplist term. Person prefix/suffix signals (B, C)
     are preserved independently; phase tokens (Phase 1/2/3/4 or I/II/III/IV)
     also fire Signal G. No SNOMED lookup is performed here.

HIPAA: query text never logged. Only signal_count and passed are logged.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# FIX 2: skiplist of clinical/structural words that should NOT count as
# mandatory term candidates. Token length filter (>= 4) handles many
# very short noise tokens; this list catches longer noise.
_SECONDARY_TOKENS: frozenset[str] = frozenset({
    # Clinical / structural
    "phase", "trial", "trials", "study", "studies", "research", "clinical",
    "active", "ongoing", "recruiting", "enrolling", "completed",
    "approved", "approval", "approvals", "pending", "open", "closed",
    "patient", "patients", "subject", "subjects", "enrollment",
    "date", "dates", "time", "times", "year", "years",
    "month", "months", "week", "weeks", "day", "days",
    # Articles / prepositions / conjunctions (>=4 chars)
    "with", "and", "from", "into", "between", "during", "while",
    "this", "that", "these", "those", "what", "which", "where",
    "when", "have", "been",
    # Quantifiers / comparators
    "more", "less", "than", "least", "most", "above", "below", "over",
    "under", "since", "before", "after", "exactly", "equal", "around",
    "many", "much", "some", "each", "every", "none",
    "preference", "regardless",
    # Adjectives
    "fast", "slow", "high", "small", "tall", "short", "big", "tiny",
    "early", "late", "first", "second", "third", "fourth", "fifth",
    "deep", "shallow", "deviation", "deviations",
    "primary", "secondary", "main", "best", "worst",
    "good", "great", "bad",  "poor",
    "available",
    # Common verbs / request / output words
    "find", "show", "list", "search", "filter", "give", "want", "need",
    "looking", "rephrase", "result", "results", "output", "report",
    # Misc generic
    "type", "types", "kind", "kinds", "group", "groups",
    "criteria", "field", "fields", "value", "values",
})

# Tokens with length below this threshold cannot fire Signal F or G.
# 4 chars is enough to skip noise like "fpe", "low", "abc".
MIN_TOKEN_LEN_FOR_F: int = 4

# Cap on SNOMED known_terms scanned per query (performance).
_SNOMED_SCAN_CAP: int = 500

# Word pattern: bounded letter/apostrophe token (ReDoS-safe)
_WORD_PATTERN = re.compile(r"\b[A-Za-z][A-Za-z'\-]{1,30}\b")

# Signal G: function-word skiplist — capitalized tokens that are NOT name/subject signals.
# These are common English words that happen to be capitalized at the start of a sentence
# or in title-case queries, and should NOT count as a capitalized token signal.
_FUNCTION_WORDS: frozenset[str] = frozenset({
    "with", "from", "that", "this", "have", "been", "will", "what", "when",
    "where", "which", "there", "their", "about", "more", "also", "into",
    "than", "then", "some", "could", "would", "should", "other", "after",
    "find", "show", "pull", "give", "search", "studies", "trials", "research",
    "study",
})

_NON_CLINICAL_TOPIC_TOKENS: frozenset[str] = frozenset({
    "restaurant", "restaurants", "food", "hotel", "hotels",
    "weather", "traffic", "directions", "parking", "menu",
    "dining", "cuisine", "recipe", "recipes", "entertainment",
    "shopping", "sports", "news", "politics", "finance",
})

# Structural/administrative words that appear in SNOMED terms (e.g. "clinical
# trial" / 169230002) but carry zero real medical intent on their own.
# Used only in the non-clinical-topic gate: a SNOMED hit whose matched term
# is composed entirely of these tokens is treated as noise, not clinical
# signal, so it cannot mask an otherwise non-clinical query.
_CLINICAL_NOISE_TOKENS: frozenset[str] = frozenset({
    "clinical", "trial", "trials", "clinic", "clinics",
    "study", "studies", "studied",
    "research", "researching",
    "office", "offices",
    "trialing",
})



@dataclass
class PreflightResult:
    passed: bool
    signal_count: int


class PreflightMandatoryCheck:

    def __init__(
        self,
        known_terms: frozenset[str],
        person_prefixes: frozenset[str],
        person_suffixes: frozenset[str],
        institution_keywords: frozenset[str],
        context_signals: frozenset[str],
        geo_multiword_keys: frozenset[str],
    ) -> None:
        """known_terms: SNOMED preferred_terms + synonyms. Only terms with
        len >= 4 chars are used at scan time. Capped at 500.
        geo_multiword_keys: phrases that, when matched, mean unambiguous geo.
        """
        # Take terms of len >= 4 and cap at _SNOMED_SCAN_CAP. Use sorted
        # deterministic sampling (alphabetical) so behavior is reproducible.
        eligible = sorted(t for t in known_terms if len(t) >= 4)
        self._snomed_sample: frozenset[str] = frozenset(
            eligible[:_SNOMED_SCAN_CAP]
        )

        self._person_prefixes = frozenset(p.lower() for p in person_prefixes)
        self._person_suffixes = frozenset(p.lower() for p in person_suffixes)
        self._institution_keywords = frozenset(
            k.lower() for k in institution_keywords
        )
        self._context_signals = frozenset(s.lower() for s in context_signals)
        self._geo_multiword_keys = frozenset(
            k.lower() for k in geo_multiword_keys
        )

        # Pre-build per-prefix/suffix/context whole-word regex helpers
        # for fast multi-word context phrases like "led by", "at the".
        self._multi_word_context = tuple(
            s for s in self._context_signals if " " in s
        )

    # ----------------- internal helpers -----------------

    def _signal_a_snomed(self, query_lower: str) -> bool:
        for term in self._snomed_sample:
            if re.search(r"\b" + re.escape(term) + r"\b", query_lower):
                return True
        return False

    def _signal_a_snomed_real(self, query_lower: str) -> bool:
        """Like _signal_a_snomed but excludes SNOMED terms whose every token
        is in _CLINICAL_NOISE_TOKENS (e.g. "clinical trial").  Used only by
        the non-clinical-topic gate so that stop-concepts like 169230002 cannot
        mask a non-clinical query.
        """
        for term in self._snomed_sample:
            if not re.search(r"\b" + re.escape(term) + r"\b", query_lower):
                continue
            # A term counts as real only when at least one of its tokens is
            # not a structural noise word.
            term_tokens = re.findall(r"[a-z]+", term)
            if any(tok not in _CLINICAL_NOISE_TOKENS for tok in term_tokens):
                return True
        return False

    def _signal_b_prefix(self, query_lower: str) -> bool:
        for prefix in self._person_prefixes:
            if re.search(r"\b" + re.escape(prefix) + r"\b", query_lower):
                return True
        return False

    def _signal_c_suffix(self, query_lower: str) -> bool:
        for suffix in self._person_suffixes:
            if re.search(r"\b" + re.escape(suffix) + r"\b", query_lower):
                return True
        return False

    def _signal_d_institution(self, query_lower: str) -> bool:
        for keyword in self._institution_keywords:
            # Substring check — institution keywords can be phrases too
            if keyword in query_lower:
                return True
        return False

    def _signal_e_context(self, query_lower: str) -> bool:
        for signal in self._context_signals:
            if " " in signal:
                if signal in query_lower:
                    return True
            else:
                if re.search(r"\b" + re.escape(signal) + r"\b", query_lower):
                    return True
        return False

    def _is_token_unambiguous_geo(
        self,
        matches: list[re.Match],
        idx: int,
    ) -> bool:
        """Return True if the token at idx is itself in geo_multiword_keys
        OR forms part of a 2- or 3-token window matching a multi-word geo key.
        """
        n = len(matches)
        # Single-token check (handles test-setup overloads like "midwest")
        if matches[idx].group().lower() in self._geo_multiword_keys:
            return True
        # 2-token windows
        for s, e in (
            (max(0, idx - 1), idx + 1),
            (idx, min(n, idx + 2)),
        ):
            if e - s == 2:
                phrase = " ".join(
                    matches[j].group().lower() for j in range(s, e)
                )
                if phrase in self._geo_multiword_keys:
                    return True
        # 3-token windows
        for s, e in (
            (max(0, idx - 2), idx + 1),
            (max(0, idx - 1), min(n, idx + 2)),
            (idx, min(n, idx + 3)),
        ):
            if e - s == 3:
                phrase = " ".join(
                    matches[j].group().lower() for j in range(s, e)
                )
                if phrase in self._geo_multiword_keys:
                    return True
        return False

    def _signal_g_capitalized(self, query: str, query_lower: str) -> bool:
        """Fix 6 Signal D (new addition): a capitalized token of length >= 4 that is
        NOT in _FUNCTION_WORDS and NOT an unambiguous geo stoplist term.

        Capitalization is checked against the original query (not lowercased) so that
        genuine proper nouns (e.g. "Gout", "Holmes") fire even in mixed-case queries.
        No SNOMED lookup is performed here.
        """
        matches = list(_WORD_PATTERN.finditer(query))
        for idx, m in enumerate(matches):
            token = m.group()
            if len(token) < MIN_TOKEN_LEN_FOR_F:
                continue
            # Must start with uppercase letter
            if not token[0].isupper():
                continue
            token_lower = token.lower()
            # Skip function words
            if token_lower in _FUNCTION_WORDS:
                continue
            # Skip unambiguous geo terms
            if self._is_token_unambiguous_geo(matches, idx):
                continue
            # Survived — fire Signal G (spec's Signal D)
            return True
        return False

    def _signal_f_token(self, query: str, query_lower: str) -> bool:
        """FIX 2: token-based ambiguity signal. Case-insensitive scan with
        _SECONDARY_TOKENS skiplist, length filter, and multi-word geo window.
        """
        matches = list(_WORD_PATTERN.finditer(query))
        for idx, m in enumerate(matches):
            token = m.group()
            token_lower = token.lower()
            # Length filter — skip very short tokens
            if len(token_lower) < MIN_TOKEN_LEN_FOR_F:
                continue
            # Secondary-token skiplist
            if token_lower in _SECONDARY_TOKENS:
                continue
            # Skip if already counted in signals A-E (avoid double-counting)
            if token_lower in self._person_prefixes:
                continue
            if token_lower in self._person_suffixes:
                continue
            if token_lower in self._institution_keywords:
                continue
            if token_lower in self._context_signals:
                continue
            if token_lower in self._snomed_sample:
                continue
            # Unambiguous geo? Skip.
            if self._is_token_unambiguous_geo(matches, idx):
                continue
            # Survived all filters — fire signal F
            return True
        return False

    # ----------------- public API -----------------

    def evaluate(self, query: str, session_id: str = "") -> PreflightResult:
        query_lower = query.lower()
        signal_count = 0

        sig_a_known_term = self._signal_a_snomed(query_lower)
        if sig_a_known_term:
            signal_count += 1
        sig_b_person_prefix = self._signal_b_prefix(query_lower)
        if sig_b_person_prefix:
            signal_count += 1
        sig_c_person_suffix = self._signal_c_suffix(query_lower)
        if sig_c_person_suffix:
            signal_count += 1
        if self._signal_d_institution(query_lower):
            signal_count += 1
        if self._signal_e_context(query_lower):
            signal_count += 1
        if self._signal_f_token(query, query_lower):
            signal_count += 1
        if self._signal_g_capitalized(query, query_lower):
            signal_count += 1

        sig_person = sig_b_person_prefix or sig_c_person_suffix
        query_tokens = re.findall(r"[a-z']+", query_lower)
        # For the non-clinical-topic gate, ignore SNOMED hits that consist
        # solely of structural noise tokens (e.g. "clinical trial" / 169230002)
        # — they must not protect a non-clinical query from rejection.
        sig_a_real = self._signal_a_snomed_real(query_lower)
        if (
            not sig_a_real
            and not sig_person
            and any(tok in _NON_CLINICAL_TOPIC_TOKENS for tok in query_tokens)
        ):
            logger.info(
                "preflight: rejection_reason=non_clinical_topic session_id=%s",
                session_id,
            )
            return PreflightResult(passed=False, signal_count=signal_count)

        result = PreflightResult(
            passed=signal_count > 0,
            signal_count=signal_count,
        )
        logger.info(
            "preflight: passed=%s signal_count=%d session_id=%s",
            result.passed, result.signal_count, session_id,
        )
        return result
