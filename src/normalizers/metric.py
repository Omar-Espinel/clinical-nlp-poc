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
from typing import Optional, Union

import ahocorasick
from rapidfuzz.fuzz import token_sort_ratio
from pydantic import BaseModel, ConfigDict, field_validator, ValidationInfo

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

VALID_OPERATORS: frozenset[str] = frozenset({"lt", "lte", "gt", "gte", "eq", "between", "any"})

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

    @field_validator("value", mode="before")
    @classmethod
    def _validate_value(cls, v, info: ValidationInfo) -> Optional[Union[float, str]]:
        if "operator" not in info.data:
            raise ValueError("operator field missing — value validation requires operator")
        if "data_type" not in info.data:
            raise ValueError("data_type field missing — value validation requires data_type")
        operator = info.data["operator"]
        data_type = info.data["data_type"]
        if v is None:
            if operator not in {"any"}:
                return None
            return None
        # S10: operator=any ↔ value=None invariant
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
        return v


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
        """Parse a date string into MM/YYYY format. Returns None if unparseable."""
        if not raw_value:
            return None
        raw = raw_value.strip()
        # MM/YYYY
        m = re.fullmatch(r"(\d{1,2})/(\d{4})", raw)
        if m:
            mm, yyyy = int(m.group(1)), int(m.group(2))
            if 1 <= mm <= 12:
                return f"{mm:02d}/{yyyy}"
        # YYYY-MM
        m = re.fullmatch(r"(\d{4})-(\d{2})", raw)
        if m:
            yyyy, mm = int(m.group(1)), int(m.group(2))
            if 1 <= mm <= 12:
                return f"{mm:02d}/{yyyy}"
        # YYYY-MM-DD
        m = re.fullmatch(r"(\d{4})-(\d{2})-\d{2}", raw)
        if m:
            yyyy, mm = int(m.group(1)), int(m.group(2))
            if 1 <= mm <= 12:
                return f"{mm:02d}/{yyyy}"
        return None

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

        return sorted(best.values(), key=lambda m: m.span[0])

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
