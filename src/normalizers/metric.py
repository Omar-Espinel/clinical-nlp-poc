"""Metric filter normalizer — AC automaton + fuzzy fallback for 12 clinical metric fields."""

from __future__ import annotations

import json
import logging
import os
import re
import types
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Optional, Union

import ahocorasick
from rapidfuzz.fuzz import token_sort_ratio
from pydantic import BaseModel, ConfigDict, field_validator, model_validator, ValidationInfo

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

VALID_OPERATORS: frozenset[str] = frozenset({"lt", "lte", "gt", "gte", "eq", "between", "any"})

# rev2 Sec-SB1: year-range floor for normalize_date (DOB-leak mitigation).
# IRB approvals predating 2000 are out of scope for current trials.
MIN_VALID_YEAR: int = 2000

VALID_UNITS: frozenset[Optional[str]] = frozenset(
    {"patients", "months", "days", "sites", "percent", "queries", None}
)

MAX_FUZZY_COMPARISONS_PER_QUERY: int = 5000

_OP_WORD_MAP: dict[str, str] = {
    "less than": "lt",
    "fewer than": "lt",
    "under": "lt",
    "below": "lt",
    "<": "lt",
    "at most": "lte",
    "up to": "lte",
    "no more than": "lte",
    "≤": "lte",
    "<=": "lte",
    "more than": "gt",
    "greater than": "gt",
    "over": "gt",
    "above": "gt",
    ">": "gt",
    "at least": "gte",
    "minimum": "gte",
    "≥": "gte",
    ">=": "gte",
    "exactly": "eq",
    "equal to": "eq",
    "=": "eq",
    "between": "between",
    "from": "between",
    "any": "any",
    "no preference": "any",
    "regardless": "any",
}

# Sorted by phrase length descending so multi-word keys win in window scan
_OP_WORD_MAP_SORTED: list[tuple[str, str]] = sorted(
    _OP_WORD_MAP.items(), key=lambda kv: len(kv[0]), reverse=True
)


# ── MetricMatch dataclass ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class MetricMatch:
    canonical_field: str
    canonical_label: str
    matched_text: str
    span: tuple[int, int]
    data_type: str
    match_source: str
    confidence: float
    implied_operator: str
    implied_op_map: types.MappingProxyType


# ── MetricFilterOutput Pydantic model ─────────────────────────────────────────

class MetricFilterOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    # Field order is mandatory — validators cross-reference via info.data
    field: str
    canonical_label: str
    operator: str
    data_type: str
    value: Optional[Union[float, str]]
    # rev2 §3.3b: value_end supports the `between` operator for both numerics and dates.
    # MUST be None for any non-`between` operator; MUST be non-None when operator='between'.
    value_end: Optional[Union[float, str]] = None
    original_text: str
    confidence: float
    unit: Optional[str] = None

    @field_validator("operator")
    @classmethod
    def _validate_operator(cls, v: str) -> str:
        if v not in VALID_OPERATORS:
            raise ValueError(f"operator must be one of {VALID_OPERATORS}, got {v!r}")
        return v

    @field_validator("unit")
    @classmethod
    def _validate_unit(cls, v: Optional[str]) -> Optional[str]:
        if v not in VALID_UNITS:
            raise ValueError(f"unit must be one of {VALID_UNITS}, got {v!r}")
        return v

    @staticmethod
    def _coerce_value_field(v: Any, data_type: str, operator: str) -> Optional[Union[float, str]]:
        """Shared coercion body for both `value` and `value_end` field validators.

        - None passes through as None.
        - operator='any' requires the coerced value to be None.
        - Sentinel strings ("null", "", "none", "no preference") coerce to None.
        - data_type='numeric': float(v) or None on parse error.
        - data_type='date': MM/YYYY-format string via normalize_date or None.
        """
        if v is None:
            return None
        # S10: operator=any ↔ value=None invariant — also applies to value_end.
        if operator == "any":
            if isinstance(v, str) and v.strip().lower() in {"null", "", "none", "no preference"}:
                return None
            raise ValueError("operator='any' requires value=None")
        if isinstance(v, str) and v.strip().lower() in {"null", "", "none", "no preference"}:
            return None
        if data_type == "numeric":
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
        # rev2 §3.2: explicit date branch with isinstance guard (QA-M1) and
        # operator='any' short-circuit (Sec-SM1, handled above).
        if data_type == "date":
            if not isinstance(v, str):
                # LLM occasionally returns a number; reject — date values must be string-shaped.
                return None
            parsed = MetricFilterNormalizer.normalize_date(v, date.today())
            if parsed is None:
                return None
            return parsed
        return v

    @field_validator("value", mode="before")
    @classmethod
    def _validate_value(cls, v, info: ValidationInfo) -> Optional[Union[float, str]]:
        if "operator" not in info.data:
            raise ValueError("operator field missing — value validation requires operator")
        if "data_type" not in info.data:
            raise ValueError("data_type field missing — value validation requires data_type")
        operator = info.data["operator"]
        data_type = info.data["data_type"]
        return cls._coerce_value_field(v, data_type, operator)

    @field_validator("value_end", mode="before")
    @classmethod
    def _validate_value_end(cls, v, info: ValidationInfo) -> Optional[Union[float, str]]:
        # rev2 §3.3b: value_end uses the same coercion path as value.
        # Cross-field invariants enforced separately in _validate_between_pair.
        if v is None:
            return None
        if "operator" not in info.data:
            raise ValueError("operator field missing — value_end validation requires operator")
        if "data_type" not in info.data:
            raise ValueError("data_type field missing — value_end validation requires data_type")
        operator = info.data["operator"]
        data_type = info.data["data_type"]
        # For operator='any' the model-level validator already enforces value=None;
        # value_end with operator='any' is also disallowed via the between-pair invariant.
        if operator == "any":
            # Treat sentinel strings as None; otherwise let the between-pair validator reject.
            if isinstance(v, str) and v.strip().lower() in {"null", "", "none", "no preference"}:
                return None
            return None
        return cls._coerce_value_field(v, data_type, operator)

    @model_validator(mode="after")
    def _validate_between_pair(self) -> "MetricFilterOutput":
        """rev2 §3.3b: enforce operator/value/value_end invariants.

        - operator='between' requires both value and value_end of matching data_type
          with value <= value_end ordering.
        - Any other operator requires value_end is None.
        """
        if self.operator == "between":
            if self.value is None or self.value_end is None:
                raise ValueError("operator='between' requires both value and value_end")
            if self.data_type == "numeric":
                if not isinstance(self.value, (int, float)) or not isinstance(self.value_end, (int, float)):
                    raise ValueError(
                        "between with data_type=numeric requires numeric value AND value_end"
                    )
                if self.value > self.value_end:
                    raise ValueError(
                        f"between requires value ({self.value}) <= value_end ({self.value_end})"
                    )
            elif self.data_type == "date":
                if not isinstance(self.value, str) or not isinstance(self.value_end, str):
                    raise ValueError(
                        "between with data_type=date requires MM/YYYY string value AND value_end"
                    )
                # Both are guaranteed MM/YYYY post-_coerce_value_field. Compare as (year, month).
                try:
                    v_yyyy, v_mm = int(self.value[3:]), int(self.value[:2])
                    ve_yyyy, ve_mm = int(self.value_end[3:]), int(self.value_end[:2])
                except (ValueError, IndexError) as exc:
                    raise ValueError(
                        "between with data_type=date requires MM/YYYY-formatted strings"
                    ) from exc
                if (v_yyyy, v_mm) > (ve_yyyy, ve_mm):
                    raise ValueError(
                        f"between requires value <= value_end "
                        f"(got {self.value} > {self.value_end})"
                    )
        else:
            if self.value_end is not None:
                raise ValueError(
                    f"value_end only allowed with operator='between' "
                    f"(got operator={self.operator!r})"
                )
        return self


# ── MetricFilterNormalizer static helpers ─────────────────────────────────────

class MetricFilterNormalizer:
    """Static normalization helpers for metric filter fields."""

    @staticmethod
    def normalize_operator(text: str) -> str:
        """Map a natural-language word or phrase to an operator string.

        Returns "any" if no match found.
        """
        if not text:
            return "any"
        normalized = text.strip().lower()
        for phrase, op in _OP_WORD_MAP_SORTED:
            if phrase in normalized:
                return op
        return "any"

    @staticmethod
    def normalize_date(raw_value: str, current_date: date) -> Optional[str]:
        """Parse a date string into MM/YYYY format. Returns None if unparseable.

        rev2 §3.9 (Sec-SB1): rejects years outside [MIN_VALID_YEAR, current_date.year + 1]
        to mitigate DOB-leak risk (e.g. "patient born 03/1985" would otherwise survive).
        """
        if not raw_value:
            return None
        raw = raw_value.strip()
        mm: Optional[int] = None
        yyyy: Optional[int] = None
        # MM/YYYY
        m = re.fullmatch(r"(\d{1,2})/(\d{4})", raw)
        if m:
            mm, yyyy = int(m.group(1)), int(m.group(2))
        # YYYY-MM
        if mm is None:
            m = re.fullmatch(r"(\d{4})-(\d{2})", raw)
            if m:
                yyyy, mm = int(m.group(1)), int(m.group(2))
        # YYYY-MM-DD
        if mm is None:
            m = re.fullmatch(r"(\d{4})-(\d{2})-\d{2}", raw)
            if m:
                yyyy, mm = int(m.group(1)), int(m.group(2))
        if mm is None or yyyy is None:
            return None
        if not (1 <= mm <= 12):
            return None
        # rev2 §3.9: reject implausible years (DOB-leak mitigation, Sec-SB1).
        if yyyy < MIN_VALID_YEAR or yyyy > current_date.year + 1:
            return None
        return f"{mm:02d}/{yyyy}"

    @staticmethod
    def normalize_value(
        raw: Optional[str],
        data_type: str,
        operator: str = "any",
    ) -> Optional[Union[float, str]]:
        """Coerce raw string to typed value.

        Returns None for "No preference" (any case) or unparseable numeric.
        """
        if raw is None:
            return None
        if raw.strip().lower() in {"no preference", "null", "", "none"}:
            return None
        if data_type == "numeric":
            try:
                return float(raw.strip())
            except (TypeError, ValueError):
                return None
        if data_type == "date":
            return MetricFilterNormalizer.normalize_date(raw, date.today())
        if data_type == "categorical":
            return raw.strip()
        return raw.strip()


# ── Internal helpers ──────────────────────────────────────────────────────────

def _normalize_query(text: str) -> str:
    """Lowercase and replace non-word chars with spaces (one-to-one for ASCII)."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def _is_word_bounded(query: str, start: int, end: int) -> bool:
    """Return True if the span [start, end) is on word boundaries."""
    before_ok = start == 0 or not query[start - 1].isalnum()
    after_ok = end == len(query) or not query[end].isalnum()
    return before_ok and after_ok


def _ngrams_with_offsets(
    tokens: list[str], token_offsets: list[int], n: int
) -> list[tuple[str, int, int]]:
    """Yield (ngram_text, char_start, char_end) for contiguous n-grams."""
    result = []
    for i in range(len(tokens) - n + 1):
        gram = " ".join(tokens[i : i + n])
        char_start = token_offsets[i]
        char_end = token_offsets[i + n - 1] + len(tokens[i + n - 1])
        result.append((gram, char_start, char_end))
    return result


def _token_offsets(span_text: str, span_offset: int) -> tuple[list[str], list[int]]:
    """Return (tokens, absolute_char_offsets) for tokens in span_text."""
    tokens = []
    offsets = []
    for m in re.finditer(r"\S+", span_text):
        tokens.append(m.group())
        offsets.append(span_offset + m.start())
    return tokens, offsets


# ── MetricIntentResolver ──────────────────────────────────────────────────────

class MetricIntentResolver:
    """Resolve metric field intent from a clinical query using AC + fuzzy fallback."""

    def __init__(
        self,
        dictionary_path: str,
        strict_validation: bool = True,
    ) -> None:
        # Env var overrides only when explicitly set; otherwise the constructor
        # parameter wins. This makes tests instantiable with strict_validation=False
        # regardless of the deploy-time default of METRIC_STRICT_VALIDATION=true.
        env_strict = os.environ.get("METRIC_STRICT_VALIDATION")
        if env_strict is not None and env_strict.lower() in {"true", "false"}:
            strict = env_strict.lower() != "false"
        else:
            strict = strict_validation

        with open(dictionary_path, encoding="utf-8") as fh:
            entries: list[dict] = json.load(fh)

        entries, synonym_duplicates_dropped = self._validate_entries(entries, strict)

        # S13: empty-automaton guard — refuse to start with zero valid fields
        # (matches AmbiguousTermsRegistry pattern in sufficiency_gate.py).
        if not entries:
            raise ValueError(
                f"MetricIntentResolver: no valid metric fields after validation "
                f"(strict={strict}). Check {dictionary_path}."
            )

        self._entries: dict[str, dict] = {e["canonical_label"]: e for e in entries}
        self._automaton = ahocorasick.Automaton()
        self._fuzzy_index: list[tuple[str, str, str, int]] = []

        seen_synonyms: dict[str, str] = {}
        for entry in entries:
            canonical_field = entry["canonical_label"]
            label = canonical_field.replace("_", " ").title()
            data_type = entry["data_type"]
            threshold = entry["fuzzy_threshold"]
            implied_op_dict: dict[str, str] = entry.get("implied_operators", {})

            for synonym in entry["synonyms"]:
                normalized_syn = synonym.lower()
                if normalized_syn in seen_synonyms:
                    continue
                seen_synonyms[normalized_syn] = canonical_field
                self._automaton.add_word(
                    normalized_syn,
                    {
                        "canonical_field": canonical_field,
                        "label": label,
                        "data_type": data_type,
                        "fuzzy_threshold": threshold,
                        "implied_op_map": types.MappingProxyType(implied_op_dict),
                        "pattern_len": len(normalized_syn),
                    },
                )
                self._fuzzy_index.append((normalized_syn, canonical_field, label, threshold))

        self._automaton.make_automaton()
        self._known_metric_fields: frozenset[str] = frozenset(self._entries.keys())

        logger.info(
            "MetricIntentResolver loaded: fields=%d synonyms=%d synonym_duplicates_dropped=%d strict=%s",
            len(self._entries),
            len(self._fuzzy_index),
            synonym_duplicates_dropped,
            str(strict).lower(),
        )

    def get_entry(self, field_key: str) -> dict:
        """Return the raw JSON entry dict for field_key. Raises KeyError if unknown."""
        return self._entries[field_key]

    def health_check(self) -> dict:
        """Return counts only — no synonym lists."""
        return {
            "fields": len(self._entries),
            "synonyms": len(self._fuzzy_index),
            "known_fields": len(self._known_metric_fields),
        }

    @staticmethod
    def _validate_entries(
        entries: list[dict], strict: bool
    ) -> tuple[list[dict], int]:
        """Validate all entries; return (valid_entries, synonym_duplicates_dropped)."""
        valid_entries: list[dict] = []
        all_synonyms: dict[str, str] = {}
        synonym_duplicates_dropped = 0

        for entry in entries:
            field_key = entry.get("canonical_label", "<unknown>")
            errors: list[str] = []

            # data_type
            if entry.get("data_type") not in {"numeric", "date", "categorical"}:
                errors.append(f"invalid data_type={entry.get('data_type')!r}")

            # synonyms — each must be a non-empty string
            for syn in entry.get("synonyms", []):
                if not isinstance(syn, str) or not syn.strip():
                    errors.append("empty or non-string synonym found")
                    break

            # implied_operators values
            for op_val in entry.get("implied_operators", {}).values():
                if op_val not in {"lt", "lte", "gt", "gte", "eq", "between", "any"}:
                    errors.append(f"invalid implied_operator value={op_val!r}")
                    break

            # fuzzy_threshold
            threshold = entry.get("fuzzy_threshold")
            if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 100):
                errors.append(f"fuzzy_threshold={threshold!r} out of [0, 100]")

            # clarification_options: 3–5 entries; last must be "No preference"
            opts = entry.get("clarification_options", [])
            if not (3 <= len(opts) <= 5):
                errors.append(f"clarification_options length={len(opts)} not in [3,5]")
            elif opts[-1] != "No preference":
                errors.append("last clarification_option must be 'No preference'")

            # clarification_question must contain {trigger}
            question = entry.get("clarification_question", "")
            if "{trigger}" not in question:
                errors.append("clarification_question missing {trigger}")

            if errors:
                msg = f"MetricIntentResolver: field={field_key!r} validation errors: {errors}"
                if strict:
                    raise ValueError(msg)
                logger.warning("%s", msg)
                continue

            # Cross-field synonym duplicate check
            field_synonyms_clean: list[str] = []
            for syn in entry.get("synonyms", []):
                norm = syn.lower()
                if norm in all_synonyms:
                    synonym_duplicates_dropped += 1
                    logger.warning(
                        "MetricIntentResolver: synonym_duplicates_dropped=1 field=%s",
                        field_key,
                    )
                    if strict:
                        raise ValueError(
                            f"Duplicate synonym {norm!r} in field={field_key!r}; "
                            f"already in field={all_synonyms[norm]!r}"
                        )
                else:
                    all_synonyms[norm] = field_key
                    field_synonyms_clean.append(syn)

            # Build a clean copy with deduplicated synonyms
            clean_entry = dict(entry)
            clean_entry["synonyms"] = field_synonyms_clean
            valid_entries.append(clean_entry)

        return valid_entries, synonym_duplicates_dropped

    def resolve(self, canonical: str) -> list[MetricMatch]:
        """Return detected metric matches sorted by span start."""
        query = _normalize_query(canonical)

        # Step 2 — AC scan: collect ALL hits first
        raw_hits: list[tuple[int, int, dict]] = []
        for end_idx, payload in self._automaton.iter(query):
            term_len = payload["pattern_len"]
            start = end_idx - term_len + 1
            end = end_idx + 1
            if not _is_word_bounded(query, start, end):
                continue
            raw_hits.append((start, end, payload))

        # Overlap resolution: longer span wins; tie-break by earlier start
        raw_hits.sort(key=lambda h: (-(h[1] - h[0]), h[0]))
        accepted: list[tuple[int, int, dict]] = []
        accepted_spans: list[tuple[int, int]] = []
        for start, end, payload in raw_hits:
            overlaps = any(
                not (end <= aks or start >= ake)
                for aks, ake in accepted_spans
            )
            if not overlaps:
                accepted.append((start, end, payload))
                accepted_spans.append((start, end))

        # Step 3 — Residual spans
        residual: list[tuple[int, int]] = []
        prev_end = 0
        for start, end, _ in sorted(accepted, key=lambda h: h[0]):
            if prev_end < start:
                residual.append((prev_end, start))
            prev_end = end
        if prev_end < len(query):
            residual.append((prev_end, len(query)))

        # Build initial candidates from AC hits
        all_candidates: list[MetricMatch] = []
        for start, end, payload in accepted:
            matched_text = canonical[start:end]
            implied_operator = self._scan_implied_op(query, start, end, payload["implied_op_map"])
            all_candidates.append(
                MetricMatch(
                    canonical_field=payload["canonical_field"],
                    canonical_label=payload["label"],
                    matched_text=matched_text,
                    span=(start, end),
                    data_type=payload["data_type"],
                    match_source="ac_synonym",
                    confidence=1.0,
                    implied_operator=implied_operator,
                    implied_op_map=payload["implied_op_map"],
                )
            )

        # Step 4 — Fuzzy fallback on residual spans
        matched_fields = {p["canonical_field"] for _, _, p in accepted}
        fuzzy_comparisons = 0
        budget_exhausted = False

        for res_start, res_end in residual:
            span_text = query[res_start:res_end]
            if len(span_text.strip()) < 6:
                continue
            tokens, tok_offsets = _token_offsets(span_text, res_start)
            if not tokens:
                continue

            for n in range(2, 6):
                if len(tokens) < n:
                    continue
                ngrams = _ngrams_with_offsets(tokens, [o - res_start for o in tok_offsets], n)
                for ngram_text, ngram_char_start, ngram_char_end in ngrams:
                    for synonym, cf, label, threshold in self._fuzzy_index:
                        if cf in matched_fields:
                            continue
                        if budget_exhausted:
                            break
                        fuzzy_comparisons += 1
                        if fuzzy_comparisons > MAX_FUZZY_COMPARISONS_PER_QUERY:
                            budget_exhausted = True
                            logger.info(
                                "metric_fuzzy_budget_exhausted=true comparisons=%d",
                                fuzzy_comparisons,
                            )
                            break
                        score = token_sort_ratio(ngram_text, synonym)
                        if score >= threshold:
                            abs_start = res_start + ngram_char_start
                            abs_end = res_start + ngram_char_end
                            matched_text = canonical[abs_start:abs_end]
                            entry = self._entries.get(cf, {})
                            implied_op_map = types.MappingProxyType(
                                entry.get("implied_operators", {})
                            )
                            implied_operator = self._scan_implied_op(
                                query, abs_start, abs_end, implied_op_map
                            )
                            all_candidates.append(
                                MetricMatch(
                                    canonical_field=cf,
                                    canonical_label=label,
                                    matched_text=matched_text,
                                    span=(abs_start, abs_end),
                                    data_type=entry.get("data_type", "numeric"),
                                    match_source="fuzzy",
                                    confidence=round(score / 100 * 0.90, 4),
                                    implied_operator=implied_operator,
                                    implied_op_map=implied_op_map,
                                )
                            )
                    if budget_exhausted:
                        break
                if budget_exhausted:
                    break
            if budget_exhausted:
                break

        # Step 5 — Dedup by canonical_field, keep max confidence
        best: dict[str, MetricMatch] = {}
        for m in all_candidates:
            if m.canonical_field not in best or m.confidence > best[m.canonical_field].confidence:
                best[m.canonical_field] = m

        # Fix 4 — Context window validation: a match is only valid when at least one
        # qualifying signal is present within ±10 tokens of the matched region:
        #   (a) an explicit numeric value (digit sequence, incl. ordinals like "1st"),
        #   (b) a comparator phrase from _CONTEXT_COMPARATORS (or field implied_op_map key),
        #   (c) the metric field's canonical_label verbatim (case-insensitive).
        # Applied to fuzzy matches and short single-token AC matches that lack a
        # clear measurement context. Exact multi-word AC synonym matches are kept as-is
        # because the user explicitly referenced the metric field by name.
        validated: list[MetricMatch] = []
        tokens_list = query.split()
        for m in best.values():
            # Exact multi-word AC matches are trusted as intentional references
            if m.match_source == "ac_synonym" and len(m.matched_text.split()) >= 2:
                validated.append(m)
                continue
            if self._has_metric_context(query, tokens_list, m):
                validated.append(m)

        return sorted(validated, key=lambda m: m.span[0])

    # Comparator phrases for Fix 4 context window check (normalized lowercase)
    _CONTEXT_COMPARATORS: frozenset[str] = frozenset({
        "more than", "less than", "at least", "under", "over",
        "fewer than", "greater than", "no more than", "no fewer than",
        "up to", "minimum", "maximum",
    })

    @staticmethod
    def _has_metric_context(
        query: str,
        tokens_list: list[str],
        m: MetricMatch,
    ) -> bool:
        """Return True if a qualifying context signal exists within ±10 tokens.

        Signal (a): any digit sequence (including ordinals like "1st", "2nd").
        Signal (b): a comparator phrase from _CONTEXT_COMPARATORS.
        Signal (c): the canonical_label of the metric field verbatim.
        """
        # Locate the approximate token index for the matched span
        span_start, span_end = m.span
        # Build a token-position map for window extraction
        char_pos = 0
        token_positions: list[tuple[int, int]] = []
        for tok in tokens_list:
            idx = query.find(tok, char_pos)
            if idx == -1:
                idx = char_pos
            token_positions.append((idx, idx + len(tok)))
            char_pos = idx + len(tok)

        # Find which token(s) overlap the span
        matched_indices: list[int] = []
        for i, (ts, te) in enumerate(token_positions):
            if ts < span_end and te > span_start:
                matched_indices.append(i)

        if not matched_indices:
            # Fallback: use full query for context check
            center = 0
        else:
            center = matched_indices[0]

        window_start = max(0, center - 10)
        window_end = min(len(tokens_list), center + 10 + 1)
        window_tokens = tokens_list[window_start:window_end]
        window_text = " ".join(window_tokens).lower()

        # Signal (a): digit sequence (incl. ordinals)
        if re.search(r'\d', window_text):
            return True

        # Signal (b): comparator phrase
        for phrase in MetricIntentResolver._CONTEXT_COMPARATORS:
            if phrase in window_text:
                return True

        # Signal (b2): field-specific implied operator keys also qualify as context signals.
        # This preserves matches triggered by domain-specific intensifiers like
        # "fast" → lt, "slow" → gt, "high" → gt, "low" → lt which are semantically
        # equivalent to a comparator directive for that field.
        for key in m.implied_op_map:
            if key and re.search(r'\b' + re.escape(key) + r'\b', window_text):
                return True

        # Signal (c): canonical_label verbatim (case-insensitive)
        if m.canonical_label.lower() in window_text:
            return True

        return False

    @staticmethod
    def _scan_implied_op(
        query: str,
        start: int,
        end: int,
        implied_op_map: types.MappingProxyType,
    ) -> str:
        """Scan ±20-char window for implied operator keys (longest key wins)."""
        window_start = max(0, start - 20)
        window_end = min(len(query), end + 20)
        window = query[window_start:window_end]
        for key in sorted(implied_op_map.keys(), key=len, reverse=True):
            if key in window:
                return implied_op_map[key]
        return "any"
