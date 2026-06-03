"""
NameExtractor: deterministic investigator_name and site_name detection.
No LLM. Rule-based signals with confidence scoring.
Security: all regex use bounded quantifiers (ReDoS prevention), MAX_NAME_LENGTH=100.
HIPAA: name values never logged.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants (exported for pipeline.py and preflight.py)
# ---------------------------------------------------------------------------
MAX_NAME_LENGTH: int = 100
CONFIDENCE_THRESHOLD: float = 0.60
FIRM_THRESHOLD: float = 0.75
INSTITUTION_FUZZY_THRESHOLD: int = 82

# Phrases stripped from the START of the query before name extraction.
# Applied greedily left-to-right (case-insensitive) until no prefix matches.
# Ordered longest-first so multi-word phrases win over their prefixes.
_REQUEST_PREFIX_PHRASES: tuple[str, ...] = (
    "can you",
    "find me",
    "pull up",
    "show me",
    "i need",
    "i want",
    "looking for",
    "please",
    "get me",
    "give me",
    "search for",
    "do you have",
    "are there",
    "find",
    "show",
    "pull",
)
# Pre-compiled patterns anchored to start of string for each prefix phrase.
# Sorted longest-first (already above) and compiled once at module load.
_REQUEST_PREFIX_PATTERNS: tuple[re.Pattern, ...] = tuple(
    re.compile(r'(?i)^' + re.escape(p) + r'\b')
    for p in _REQUEST_PREFIX_PHRASES
)

_LEADING_PUNCT_WS = re.compile(r'^[\s\W]+')

PERSON_PREFIXES: frozenset[str] = frozenset({
    "dr", "dr.", "prof", "prof.", "professor", "pi", "physician"
})
PERSON_SUFFIXES: frozenset[str] = frozenset({
    "md", "m.d.", "phd", "ph.d.", "do", "d.o.", "rn", "mph",
    "facp", "facs", "fnp", "np"
})
CONTEXT_PERSON: frozenset[str] = frozenset({
    "by", "investigator", "pi", "led by", "run by",
    "researcher", "principal investigator"
})
CONTEXT_SITE: frozenset[str] = frozenset({
    "at", "at the", "site", "location", "facility"
})

# ---------------------------------------------------------------------------
# Canonical display table for known multiword sites
# ---------------------------------------------------------------------------
_CANONICAL_DISPLAY_TABLE: dict[str, str] = {
    "johns hopkins": "Johns Hopkins",
    "mayo clinic": "Mayo Clinic",
    "cleveland clinic": "Cleveland Clinic",
    "memorial sloan kettering": "Memorial Sloan Kettering",
    "md anderson": "MD Anderson",
    "mass general": "Mass General",
    "massachusetts general": "Massachusetts General",
    "cedars sinai": "Cedars-Sinai",
    "cedars-sinai": "Cedars-Sinai",
    "mount sinai": "Mount Sinai",
    "stanford medicine": "Stanford Medicine",
    "ucsf": "UCSF",
    "ucla health": "UCLA Health",
    "duke university": "Duke University",
    "yale new haven": "Yale New Haven",
    "vanderbilt university": "Vanderbilt University",
    "emory university": "Emory University",
    "northwestern university": "Northwestern University",
    "university of michigan": "University of Michigan",
    "university of toronto": "University of Toronto",
    "university health network": "University Health Network",
    "boston children's": "Boston Children's",
    "children's hospital": "Children's Hospital",
    "veterans affairs": "Veterans Affairs",
}

# ---------------------------------------------------------------------------
# Compiled patterns (module-level, one-shot)
# NB: applied against title-normalised query so [A-Z][a-z] patterns work
# ---------------------------------------------------------------------------
# Prefix pattern: dr/prof followed by 1-3 name tokens.
# Applied on norm_query (already title-cased). Uses IGNORECASE so "Dr" matches.
_PERSON_PREFIX_PATTERN = re.compile(
    r'\b(Dr\.?|Prof\.?|Professor)\s+([A-Z][a-z]{1,30}(?:\s+[A-Z][a-z]{1,30}){0,2})\b'
)
# Suffix pattern: applied on norm_query; suffix must be a separate token (whitespace/comma required)
_PERSON_SUFFIX_PATTERN = re.compile(
    r'\b([A-Z][a-z]{1,30}(?:\s+[A-Z][a-z]{1,30}){0,2})(?:\s*,\s*|\s+)(M\.?D\.?|Ph\.?D\.?|D\.?O\.?|R\.?N\.?|M\.?P\.?H\.?)\b',
    re.IGNORECASE,
)
# Title-case multi-token pattern for STEP 5 — max 2 tokens to avoid long spurious matches
_TITLE_CASE_PATTERN = re.compile(
    r'\b([A-Z][a-z]{1,30}(?:\s+[A-Z][a-z]{1,30}){1,1})\b'
)
_TITLE_TOKEN = re.compile(r'[A-Z][a-z]{1,30}')
_WORD_TOKEN = re.compile(r"[\w']{2,30}")

# Common non-name words that appear title-cased but are not person names
_STOPWORDS: frozenset[str] = frozenset({
    "Phase", "Trial", "Trials", "Diabetes", "Cancer", "Study", "Clinical",
    "Research", "Data", "Patient", "Patients", "Treatment", "Disease",
    "Drug", "Dose", "Safety", "Efficacy", "Protocol", "Arm", "Cohort",
    "Stage", "Grade", "Type", "Form", "Group", "Center", "Hospital",
    "Institute", "University", "College", "School", "Medical", "Health",
    "Care", "System", "Network", "Foundation", "Society", "Association",
    "Inc", "Llc", "Corp", "Ltd",
    "In", "On", "At", "To", "Of", "Or", "And", "But", "For", "With", "By",
    "From", "Into", "Upon", "About", "After", "Before", "During", "Since",
    "Until", "While", "Where", "When", "Why", "How", "What", "Which", "Who",
    "Whom", "Whose", "This", "That", "These", "Those", "Is", "Are", "Was",
    "Were", "Be", "Been", "Being", "Have", "Has", "Had", "Do", "Does", "Did",
    "Will", "Would", "Should", "Could", "May", "Might", "Must", "Can", "The",
    "A", "An", "As", "If", "So", "No", "Not", "All", "Any", "Some", "Few",
    "Many", "More", "Most", "Other", "Such", "Own", "Same", "Studies", "Find",
    "Show", "List",
})

# ---------------------------------------------------------------------------
# Structured-parse role sets
# ---------------------------------------------------------------------------
_ROLE_PERSON: frozenset[str] = frozenset({
    "investigator", "inv", "doctor", "dr", "pi", "physician"
})
_ROLE_SITE: frozenset[str] = frozenset({
    "site", "institution", "inst", "hospital", "center", "clinic", "facility"
})


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class AmbiguousNameMatch:
    text: str
    span: tuple[int, int]
    possible_types: list[str]


@dataclass
class NameResult:
    investigator_name: Optional[str]
    investigator_confidence: float
    site_name: Optional[str]
    site_confidence: float
    ambiguous_names: list[AmbiguousNameMatch] = field(default_factory=list)
    excluded_spans: list[tuple[int, int]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _spans_overlap(
    span: tuple[int, int],
    existing: list[tuple[int, int]],
) -> bool:
    s, e = span
    for es, ee in existing:
        if s < ee and e > es:
            return True
    return False


def _title_normalize(query: str) -> str:
    """
    Title-case each word token so that pattern matching (which relies on
    [A-Z][a-z]) works regardless of the original case of the query.
    Apostrophe-containing tokens (children's) are handled gracefully.
    Span positions are preserved (same length as original).
    """
    def _cap_word(m: re.Match) -> str:
        w = m.group(0)
        return w[0].upper() + w[1:].lower() if w else w

    return re.sub(r"[A-Za-z']+", _cap_word, query)


def _extract_name_from_normalized(
    norm_query: str, orig_query: str, start: int
) -> tuple[str, int, int]:
    """
    Given a position in the (length-identical) normalised query, extract
    up to 3 word tokens from that position and return (display_name, abs_start, abs_end).
    Display name is derived from the normalised query (title-cased already).
    """
    after = norm_query[start:]
    tokens = _TITLE_TOKEN.findall(after)[:3]
    if not tokens:
        return ("", start, start)
    name = " ".join(tokens)
    # find first token position
    m = re.search(re.escape(tokens[0]), after)
    if m is None:
        return ("", start, start)
    abs_start = start + m.start()
    abs_end = abs_start + len(name)
    return (name, abs_start, abs_end)


# ---------------------------------------------------------------------------
# NameExtractor
# ---------------------------------------------------------------------------
class NameExtractor:
    def __init__(
        self,
        institution_keywords_path: str,
        geo_json_path: str,
        snomed_known_terms: frozenset[str] = frozenset(),
    ) -> None:
        # Load and validate institution_keywords.json
        with open(institution_keywords_path, encoding="utf-8") as fh:
            inst = json.load(fh)

        for key in ("primary_keywords", "legal_suffixes", "known_multiword_sites"):
            if key not in inst:
                raise ValueError(f"institution_keywords.json missing required key: '{key}'")
            if not inst[key]:
                raise ValueError(f"institution_keywords.json key '{key}' must be non-empty list")
            for item in inst[key]:
                if not isinstance(item, str) or not item.strip():
                    raise ValueError(
                        f"institution_keywords.json key '{key}' contains invalid entry"
                    )

        self._primary_set: frozenset[str] = frozenset(
            k.lower() for k in inst["primary_keywords"]
        )
        self._suffix_set: frozenset[str] = frozenset(
            k.lower() for k in inst["legal_suffixes"]
        )
        self._multiword_set: frozenset[str] = frozenset(
            k.lower() for k in inst["known_multiword_sites"]
        )

        # Build canonical display (filtered to keys in _multiword_set)
        self._canonical_display: dict[str, str] = {
            k: v for k, v in _CANONICAL_DISPLAY_TABLE.items()
            if k in self._multiword_set
        }

        # Load geo_canonical.json
        with open(geo_json_path, encoding="utf-8") as fh:
            geo = json.load(fh)

        cities_dict: dict = geo.get("cities", {})
        regions_dict: dict = geo.get("regions", {})
        states_dict: dict = geo.get("states", {})

        self._geo_city_keys: frozenset[str] = frozenset(cities_dict.keys())
        self._geo_region_keys: frozenset[str] = frozenset(regions_dict.keys())
        self._cities_data: dict = cities_dict
        self._regions_data: dict = regions_dict

        # Single-word geo keys (no space) from cities + regions
        self._geo_single_word_keys: frozenset[str] = frozenset(
            k for k in (self._geo_city_keys | self._geo_region_keys)
            if " " not in k
        )

        # FIX 4: state keys (lowercase names + 2-letter abbreviations)
        self._state_keys: frozenset[str] = frozenset(states_dict.keys())
        self._state_canonical: dict[str, str] = states_dict

        # Separate 2-letter uppercase abbreviation keys for case-sensitive matching
        self._state_abbrev_keys: frozenset[str] = frozenset(
            k for k in states_dict.keys() if len(k) <= 2 and k.isupper()
        )

        # All geo terms (single-word) that should suppress ambiguous extraction
        self._all_geo_single: frozenset[str] = (
            self._geo_single_word_keys
            | frozenset(k for k in self._state_keys if " " not in k)
        )

        self._snomed_single_tokens: frozenset[str] = frozenset(
            term for term in snomed_known_terms
            if " " not in term and len(term) >= 5
        )
        self._snomed_multi_tokens: frozenset[tuple] = frozenset(
            tuple(term.split())
            for term in snomed_known_terms
            if 2 <= len(term.split()) <= 3
        )

    # ------------------------------------------------------------------
    # _is_snomed_token_sequence
    # ------------------------------------------------------------------
    def _is_snomed_token_sequence(self, tokens) -> bool:
        """True if the token sequence exactly matches a known SNOMED term (single or 2-3 token)."""
        toks = tuple(t.lower() for t in tokens)
        if len(toks) == 1:
            return toks[0] in self._snomed_single_tokens
        if 2 <= len(toks) <= 3:
            return toks in self._snomed_multi_tokens
        return False

    # ------------------------------------------------------------------
    # _is_institution_context
    # ------------------------------------------------------------------
    def _is_institution_context(self, query_lower: str, suffix_start: int) -> bool:
        window_start = max(0, suffix_start - 40)
        window_end = min(len(query_lower), suffix_start + 40)
        window = query_lower[window_start:window_end]
        for mw in self._multiword_set:
            if re.search(r'\b' + re.escape(mw) + r'\b', window):
                return True
        return False

    # ------------------------------------------------------------------
    # _matches_institution_keyword
    # ------------------------------------------------------------------
    def _matches_institution_keyword(self, text: str) -> bool:
        text_lower = text.lower()
        # Substring match against primary and suffix sets
        for kw in self._primary_set | self._suffix_set:
            if kw in text_lower:
                return True
        # Fuzzy match per word against primary set
        for word in text_lower.split():
            for kw in self._primary_set:
                if fuzz.token_sort_ratio(word, kw) >= INSTITUTION_FUZZY_THRESHOLD:
                    return True
        return False

    # ------------------------------------------------------------------
    # _try_structured_parse
    # ------------------------------------------------------------------
    def _try_structured_parse(self, query: str) -> Optional[NameResult]:
        segments = [s.strip() for s in query.split(",")]
        if len(segments) < 2:
            return None

        inv_name: Optional[str] = None
        site_name: Optional[str] = None
        inv_conf = 0.0
        site_conf = 0.0
        role_matched = 0

        for seg in segments:
            tokens = seg.strip().split()
            if not tokens:
                continue
            role_token = tokens[-1].lower().rstrip(".")
            name_tokens = tokens[:-1]
            name_candidate = " ".join(name_tokens).strip()

            if role_token in _ROLE_PERSON:
                role_matched += 1
                if name_candidate and len(name_candidate) <= MAX_NAME_LENGTH:
                    # Title-case the name candidate
                    inv_name = name_candidate.title()
                    inv_conf = 0.95
            elif role_token in _ROLE_SITE:
                role_matched += 1
                if name_candidate and len(name_candidate) <= MAX_NAME_LENGTH:
                    site_name = name_candidate.title()
                    site_conf = 0.95

        if role_matched == 0:
            return None

        # If only one role was matched and there are 2 segments, the other → site
        if role_matched == 1 and len(segments) == 2 and site_name is None:
            for seg in segments:
                tokens = seg.strip().split()
                if not tokens:
                    continue
                role_token = tokens[-1].lower().rstrip(".")
                name_tokens = tokens[:-1]
                name_candidate = " ".join(name_tokens).strip()
                if role_token not in _ROLE_PERSON and role_token not in _ROLE_SITE:
                    if name_candidate and len(name_candidate) <= MAX_NAME_LENGTH:
                        site_name = seg.strip().title()
                        site_conf = 0.95

        return NameResult(
            investigator_name=inv_name,
            investigator_confidence=inv_conf,
            site_name=site_name,
            site_confidence=site_conf,
        )

    # ------------------------------------------------------------------
    # find_geo_spans
    # ------------------------------------------------------------------
    def find_geo_spans(self, query: str) -> list[tuple[int, int]]:
        all_geo_keys = self._geo_city_keys | self._geo_region_keys | self._state_keys
        sorted_keys = sorted(all_geo_keys, key=len, reverse=True)
        found_spans: list[tuple[int, int]] = []

        for key in sorted_keys:
            # For uppercase 2-letter abbreviations: case-sensitive
            if key in self._state_abbrev_keys:
                pattern = r'\b' + re.escape(key) + r'\b'
                for m in re.finditer(pattern, query):
                    span = (m.start(), m.end())
                    if not _spans_overlap(span, found_spans):
                        found_spans.append(span)
            else:
                pattern = r'\b' + re.escape(key) + r'\b'
                for m in re.finditer(pattern, query, re.IGNORECASE):
                    span = (m.start(), m.end())
                    if not _spans_overlap(span, found_spans):
                        found_spans.append(span)

        return found_spans

    # ------------------------------------------------------------------
    # _extract_city_state  (FIX 4)
    # ------------------------------------------------------------------
    def _extract_city_state(
        self, query: str
    ) -> tuple[Optional[str], Optional[str]]:
        q_lower = query.lower()

        # City: longest-first scan of city keys
        city_canonical: Optional[str] = None
        for key in sorted(self._geo_city_keys, key=len, reverse=True):
            pattern = r'\b' + re.escape(key) + r'\b'
            if re.search(pattern, q_lower):
                city_canonical = self._cities_data[key]["canonical"]
                break

        # Fallback to regions
        if city_canonical is None:
            for key in sorted(self._geo_region_keys, key=len, reverse=True):
                pattern = r'\b' + re.escape(key) + r'\b'
                if re.search(pattern, q_lower):
                    region_info = self._regions_data[key]
                    city_canonical = region_info.get("primary_city")
                    break

        # State: scan state_keys longest-first
        state_canonical: Optional[str] = None
        for key in sorted(self._state_keys, key=len, reverse=True):
            if key in self._state_abbrev_keys:
                # Case-sensitive, word-boundary
                pattern = r'\b' + re.escape(key) + r'\b'
                if re.search(pattern, query):
                    state_canonical = self._state_canonical[key]
                    break
            else:
                pattern = r'\b' + re.escape(key) + r'\b'
                if re.search(pattern, q_lower):
                    state_canonical = self._state_canonical[key]
                    break

        return (city_canonical, state_canonical)

    # ------------------------------------------------------------------
    # extract  — main decision tree
    # ------------------------------------------------------------------
    @staticmethod
    def _strip_request_prefixes(query: str) -> str:
        """Strip filler request phrases from the START of the query (Fix 2).

        Applied greedily left-to-right until no prefix phrase matches.
        Trailing whitespace and punctuation are stripped after each removal.
        This is LOCAL to NameExtractor — canonical_query is never modified.
        """
        changed = True
        while changed:
            changed = False
            for pat in _REQUEST_PREFIX_PATTERNS:
                m = pat.match(query)
                if m:
                    query = query[m.end():]
                    # Strip leading whitespace and punctuation
                    query = _LEADING_PUNCT_WS.sub('', query)
                    changed = True
                    break  # restart from beginning after each strip
        return query

    def extract(
        self,
        query: str,
        excluded_spans: Optional[list[tuple[int, int]]] = None,
    ) -> NameResult:
        if excluded_spans is None:
            excluded_spans = []

        # Strip filler request prefixes before any name extraction (Fix 2).
        # This is local to NameExtractor — does NOT modify canonical_query.
        query = self._strip_request_prefixes(query)

        # Track already-extracted spans
        extracted_spans: list[tuple[int, int]] = list(excluded_spans)

        inv_name: Optional[str] = None
        inv_conf: float = 0.0
        site_name: Optional[str] = None
        site_conf: float = 0.0
        ambiguous: list[AmbiguousNameMatch] = []

        # -------------------------------------------------------------------
        # STEP 0: structured parse (e.g. "johnson investigator, northwestern site")
        # -------------------------------------------------------------------
        structured = self._try_structured_parse(query)
        if structured is not None:
            logger.info(
                "name_extractor: inv_conf=%.2f site_conf=%.2f ambiguous_count=%d",
                structured.investigator_confidence,
                structured.site_confidence,
                len(structured.ambiguous_names),
            )
            return structured

        query_lower = query.lower()

        # Build a title-normalised version of the query (same length) for pattern matching.
        # This lets us apply [A-Z][a-z] patterns even when input is all-lowercase.
        norm_query = _title_normalize(query)

        # -------------------------------------------------------------------
        # STEP 0b (FIX 3): multiword institution scan (on lowered query)
        # -------------------------------------------------------------------
        for mw_key in sorted(self._multiword_set, key=len, reverse=True):
            pattern = r'\b' + re.escape(mw_key) + r'\b'
            m = re.search(pattern, query_lower)
            if m:
                span = (m.start(), m.end())
                if not _spans_overlap(span, extracted_spans):
                    site_name = self._canonical_display.get(mw_key, mw_key.title())
                    site_conf = 0.90
                    extracted_spans.append(span)
                    break

        # If no multiword, scan primary keywords; look for preceding words in the
        # normalised query (so "mayo hospital" → preceding title token "Mayo")
        if site_name is None:
            # Sort primary keywords: multi-word first, then by length descending
            sorted_primary = sorted(
                self._primary_set,
                key=lambda k: (0 if " " in k else 1, -len(k)),
            )
            for kw in sorted_primary:
                kw_pattern = r'\b' + re.escape(kw) + r'\b'
                m = re.search(kw_pattern, query_lower)
                if m:
                    span = (m.start(), m.end())
                    if _spans_overlap(span, extracted_spans):
                        continue
                    # Look backward up to 3 title-case tokens before match
                    # Use normalised query so title tokens are visible
                    prefix_norm = norm_query[: m.start()]
                    preceding = _TITLE_TOKEN.findall(prefix_norm)[-3:]
                    if preceding:
                        candidate_site = " ".join(preceding) + " " + kw.title()
                    else:
                        candidate_site = kw.title()
                    site_name = candidate_site
                    site_conf = 0.85
                    extracted_spans.append(span)
                    break

        # -------------------------------------------------------------------
        # STEP 1: Person prefix signals (on normalised query)
        # Capture prefix + following tokens, then trim stopwords from right
        # -------------------------------------------------------------------
        for m in _PERSON_PREFIX_PATTERN.finditer(norm_query):
            raw_cand = m.group(2)
            # Trim trailing stopword tokens
            tokens = raw_cand.split()
            while tokens and tokens[-1] in _STOPWORDS:
                tokens.pop()
            if not tokens:
                continue
            trimmed_tokens = []
            for i, tok in enumerate(tokens):
                if tok.lower() in self._snomed_single_tokens:
                    break
                if i + 1 < len(tokens) and (tok.lower(), tokens[i + 1].lower()) in self._snomed_multi_tokens:
                    break
                trimmed_tokens.append(tok)
            if not trimmed_tokens:
                continue
            tokens = trimmed_tokens
            cand = " ".join(tokens)
            # Recompute span to match trimmed name
            name_start = m.start(2)
            name_end = name_start + len(cand)
            span = (m.start(), name_end)
            if _spans_overlap(span, extracted_spans):
                continue
            if len(cand) <= MAX_NAME_LENGTH:
                if inv_conf < 0.95:
                    inv_name = cand
                    inv_conf = 0.95
                extracted_spans.append(span)

        # -------------------------------------------------------------------
        # STEP 2: Person suffix signals (on normalised query — IGNORECASE)
        # -------------------------------------------------------------------
        for m in _PERSON_SUFFIX_PATTERN.finditer(norm_query):
            span = (m.start(), m.end())
            if _spans_overlap(span, extracted_spans):
                continue
            cand = m.group(1)
            # Trim trailing stopwords from name candidate
            tokens = cand.split()
            while tokens and tokens[-1] in _STOPWORDS:
                tokens.pop()
            if not tokens:
                continue
            cand = " ".join(tokens)
            suffix_start = m.start(2)
            if self._is_institution_context(query_lower, suffix_start):
                continue
            if len(cand) <= MAX_NAME_LENGTH:
                if inv_conf < 0.93:
                    inv_name = cand
                    inv_conf = 0.93
                extracted_spans.append(span)

        # -------------------------------------------------------------------
        # STEP 3: Context person signals (search on original, extract from norm)
        # -------------------------------------------------------------------
        sorted_ctx_person = sorted(CONTEXT_PERSON, key=len, reverse=True)
        for ctx in sorted_ctx_person:
            ctx_pattern = r'\b' + re.escape(ctx) + r'\b'
            for m in re.finditer(ctx_pattern, query_lower):
                after_pos = m.end()
                # Extract next 1-3 title-case tokens from normalised query
                after_norm = norm_query[after_pos:]
                title_tokens = _TITLE_TOKEN.findall(after_norm)[:3]
                if not title_tokens:
                    continue
                trimmed_tokens = []
                for i, tok in enumerate(title_tokens):
                    if tok.lower() in self._snomed_single_tokens:
                        break
                    if i + 1 < len(title_tokens) and (tok.lower(), title_tokens[i + 1].lower()) in self._snomed_multi_tokens:
                        break
                    trimmed_tokens.append(tok)
                if not trimmed_tokens:
                    continue
                title_tokens = trimmed_tokens
                candidate = " ".join(title_tokens)
                # find span in norm (same positions as original)
                tok_m = re.search(re.escape(title_tokens[0]), after_norm)
                if tok_m is None:
                    continue
                cand_start = after_pos + tok_m.start()
                cand_end = cand_start + len(candidate)
                span = (cand_start, cand_end)
                if _spans_overlap(span, extracted_spans):
                    continue
                if len(candidate) > MAX_NAME_LENGTH:
                    continue
                if self._matches_institution_keyword(candidate):
                    if site_conf < 0.85:
                        site_name = candidate
                        site_conf = 0.85
                    extracted_spans.append(span)
                elif len(title_tokens) >= 2:
                    if inv_conf < 0.85:
                        inv_name = candidate
                        inv_conf = 0.85
                    extracted_spans.append(span)
                else:
                    # single token
                    if inv_conf < 0.80:
                        inv_name = candidate
                        inv_conf = 0.80
                    extracted_spans.append(span)

        # -------------------------------------------------------------------
        # STEP 4: Context site signals (search on original, extract from norm)
        # -------------------------------------------------------------------
        sorted_ctx_site = sorted(CONTEXT_SITE, key=len, reverse=True)
        for ctx in sorted_ctx_site:
            ctx_pattern = r'\b' + re.escape(ctx) + r'\b'
            for m in re.finditer(ctx_pattern, query_lower):
                after_pos = m.end()
                after_norm = norm_query[after_pos:].lstrip()
                after_lower = query_lower[after_pos:].lstrip()
                lstrip_offset = len(norm_query[after_pos:]) - len(after_norm)
                actual_after_pos = after_pos + lstrip_offset

                if not after_norm:
                    continue

                # Try title-case tokens first
                title_tokens = _TITLE_TOKEN.findall(after_norm)[:4]
                # Fallback: any word tokens (any case, from lower)
                word_tokens = _WORD_TOKEN.findall(after_lower)[:3]

                if title_tokens:
                    candidate = " ".join(title_tokens)
                    cand_lower_text = candidate.lower()
                    first_tok = title_tokens[0]
                else:
                    if not word_tokens:
                        continue
                    candidate = " ".join(t.title() for t in word_tokens)
                    cand_lower_text = " ".join(word_tokens)
                    first_tok = word_tokens[0]

                # Locate span in original query
                tok_m = re.search(re.escape(first_tok), norm_query[actual_after_pos:], re.IGNORECASE)
                if tok_m is None:
                    continue
                abs_start = actual_after_pos + tok_m.start()
                abs_end = abs_start + len(candidate)
                span = (abs_start, abs_end)
                if _spans_overlap(span, extracted_spans):
                    continue
                if len(candidate) > MAX_NAME_LENGTH:
                    continue

                # Classify
                mw_hit = False
                for mw_key in sorted(self._multiword_set, key=len, reverse=True):
                    if mw_key in cand_lower_text:
                        if site_conf < 0.95:
                            site_name = self._canonical_display.get(mw_key, mw_key.title())
                            site_conf = 0.95
                        extracted_spans.append(span)
                        mw_hit = True
                        break
                if mw_hit:
                    continue

                if self._matches_institution_keyword(candidate):
                    if site_conf < 0.90:
                        site_name = candidate
                        site_conf = 0.90
                    extracted_spans.append(span)
                elif len(title_tokens) >= 2 or (not title_tokens and len(word_tokens) >= 2):
                    if site_conf < 0.80:
                        site_name = candidate
                        site_conf = 0.80
                    extracted_spans.append(span)
                else:
                    # Single token — firm-extract if not geo, else skip
                    single_lower = cand_lower_text.strip()
                    if single_lower in self._all_geo_single:
                        pass  # skip geo tokens
                    else:
                        # Firm-extract as site at 0.80
                        if site_conf < 0.80:
                            site_name = candidate
                            site_conf = 0.80
                        extracted_spans.append(span)

        # -------------------------------------------------------------------
        # STEP 5: Residual title-case scan (on normalised query)
        # -------------------------------------------------------------------
        for m in _TITLE_CASE_PATTERN.finditer(norm_query):
            span = (m.start(), m.end())
            if _spans_overlap(span, extracted_spans):
                continue
            candidate = m.group(1)
            cand_lower_text = candidate.lower()
            if len(candidate) > MAX_NAME_LENGTH:
                continue
            tokens_count = len(candidate.split())

            if self._is_snomed_token_sequence(candidate.split()):
                continue
            if tokens_count >= 2:
                leading_tokens = candidate.split()[:3]
                if self._is_snomed_token_sequence(leading_tokens):
                    continue

            if cand_lower_text in self._multiword_set:
                if site_conf < 0.90:
                    site_name = self._canonical_display.get(cand_lower_text, cand_lower_text.title())
                    site_conf = 0.90
                extracted_spans.append(span)
            elif self._matches_institution_keyword(candidate):
                if site_conf < 0.75:
                    site_name = candidate
                    site_conf = 0.75
                extracted_spans.append(span)
            elif tokens_count >= 2:
                parts = candidate.split()
                # Skip if any token is a stopword or all tokens are geo
                has_stopword = any(t in _STOPWORDS for t in parts)
                all_geo = all(t.lower() in self._all_geo_single for t in parts)
                if has_stopword or all_geo:
                    pass
                else:
                    # Emit each non-stopword, non-geo token as a separate ambiguous entry
                    pos = span[0]
                    for part in parts:
                        part_lower = part.lower()
                        if part_lower in _STOPWORDS or part_lower in self._all_geo_single:
                            pos += len(part) + 1
                            continue
                        part_span = (pos, pos + len(part))
                        ambiguous.append(
                            AmbiguousNameMatch(
                                text=part,
                                span=part_span,
                                possible_types=["investigator", "site"],
                            )
                        )
                        pos += len(part) + 1
            # Single title-case word → SKIP (too ambiguous)

        # -------------------------------------------------------------------
        # Conflict resolution & dedup
        # -------------------------------------------------------------------
        extracted_texts = []
        if inv_name:
            extracted_texts.append(inv_name.lower())
        if site_name:
            extracted_texts.append(site_name.lower())

        filtered_ambiguous: list[AmbiguousNameMatch] = []
        seen_texts: set[str] = set()
        for am in ambiguous:
            at_lower = am.text.lower()
            if any(at_lower in et for et in extracted_texts):
                continue
            if at_lower in seen_texts:
                continue
            seen_texts.add(at_lower)
            filtered_ambiguous.append(am)

        # -------------------------------------------------------------------
        # Final length check
        # -------------------------------------------------------------------
        if inv_name and len(inv_name) > MAX_NAME_LENGTH:
            inv_name = None
            inv_conf = 0.0
        if site_name and len(site_name) > MAX_NAME_LENGTH:
            site_name = None
            site_conf = 0.0

        # -------------------------------------------------------------------
        # Logging (HIPAA: no name values)
        # -------------------------------------------------------------------
        logger.info(
            "name_extractor: inv_conf=%.2f site_conf=%.2f ambiguous_count=%d",
            inv_conf,
            site_conf,
            len(filtered_ambiguous),
        )

        return NameResult(
            investigator_name=inv_name,
            investigator_confidence=inv_conf,
            site_name=site_name,
            site_confidence=site_conf,
            ambiguous_names=filtered_ambiguous,
            excluded_spans=list(excluded_spans),
        )
