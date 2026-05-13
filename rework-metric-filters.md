# rework-metric-filters.md
## Design Spec — Metric Filter Recognition v2
**Status:** ARCHITECT DRAFT · 2026-05-12 · **Rev 2 — Post-Review** <!-- rev2 -->
**Author:** Architect agent  
**Implements:** AC automaton + fuzzy fallback for 12 clinical-trial operational metric fields  
**Target branch:** `aho-corasick-NLP`

---

## 1. Summary & Flow Diagram

Adds a deterministic pre-extraction pass that recognizes clinical-trial operational metrics (enrollment count, response times, etc.) from the canonical query before the LLM filter-extraction step. A single Aho-Corasick automaton scans all 12 fields in one pass; rapidfuzz covers residual spans for spelling variants. Recognized fields without values trigger a `MetricAmbiguityGate` clarification (parallel to the existing `EmbeddingAmbiguityGate`). Resolved fields flow into `ExtractedFilters.metric_fields` and appear in `NLPOutput`.

```
canonical query
     │
     ├── [Step 3b NEW] MetricIntentResolver.resolve(canonical)
     │         └── AC scan → fuzzy fallback → list[MetricMatch]
     │
     ├── [Step 4]  fut_filters = executor.submit(extract, canonical, metric_matches)
     │             FilterExtractor.extract() appends metric_section to SYSTEM_PROMPT
     │             when metric_matches non-empty; parses metric_fields from LLM response
     │
     │   ... [Steps 5, 5b, 6 unchanged] ...
     │
     ├── [Step 7b NEW] MetricAmbiguityGate.evaluate(metric_matches, filters, session)
     │         ├── if unresolved field → ClarificationOutput (early return)
     │         └── else None → fall through
     │
     ├── [Step 8]  GeoNormalizer (unchanged)
     │
     └── [Step 9]  assembler.assemble(..., metric_filters=resolved_metric_filters)
                   → NLPOutput with metric_filters: list[MetricFilterOutput]
```

---

## 2. `data/metric_filters.json` — Schema + 12 Fields

### Schema per entry
```json
{
  "canonical_label": "string — snake_case field key",
  "data_type": "numeric | date | categorical",
  "synonyms": ["list of trigger phrases"],
  "clarification_question": "template string containing {trigger}",
  "clarification_options": ["3–5 strings; last MUST be 'No preference'"],
  "implied_operators": {"trigger_word": "lt|lte|gt|gte|eq|between|any"},
  "fuzzy_threshold": 75
}
```

**Decision:** `fuzzy_threshold` is per-field (not global) because operational metrics have
very different synonym density. Numeric fields use 80; date-range fields use 75.

<!-- rev2 --> **Startup validation (`_validate_entries()`):** `MetricIntentResolver.__init__` calls `_validate_entries()` before building the automaton. Rules (strict mode raises `ValueError`; lenient mode logs and skips):
- `data_type` ∈ `{"numeric", "date", "categorical"}`
- every synonym is a non-empty string
- every `implied_operators` value ∈ `{"lt", "lte", "gt", "gte", "eq", "between", "any"}`
- `fuzzy_threshold` ∈ [0, 100]
- `clarification_options` has 3–5 entries; last entry is exactly `"No preference"`
- `clarification_question` contains the literal `{trigger}` substring
- after normalizing all synonyms to lowercase, no synonym appears in more than one field (duplicate → raise/skip per strict mode; count logged as `synonym_duplicates_dropped=N`)

Startup log line format: `"MetricIntentResolver loaded: fields=%d synonyms=%d synonym_duplicates_dropped=%d strict=%s"`

<!-- rev2 --> **Startup failure mode:** `MetricIntentResolver.__init__(dictionary_path, strict_validation: bool = True)`. Env var `METRIC_STRICT_VALIDATION` (default `"true"`) overrides the constructor default. Strict → `raise ValueError` on schema violation (CI/test mode). Lenient → `logger.warning` + skip field. Mirror `AmbiguousTermsRegistry` pattern.

### 12 Fields <!-- review-hot -->

```json
[
  {
    "canonical_label": "enrollment_count",
    "data_type": "numeric",
    "synonyms": [
      "enrollment count", "enrolled patients", "number of patients enrolled",
      "patient enrollment", "participants enrolled", "total enrollment",
      "enrollment target", "enrollment size", "number enrolled",
      "enrollment number", "enrolling patients"
    ],
    "clarification_question": "What enrollment count range are you targeting for {trigger}?",
    "clarification_options": [
      "Under 50 patients",
      "50–200 patients",
      "200–500 patients",
      "Over 500 patients",
      "No preference"
    ],
    "implied_operators": {
      "large": "gt", "small": "lt", "fewer than": "lt", "more than": "gt",
      "at least": "gte", "up to": "lte", "between": "between"
    },
    "fuzzy_threshold": 80
  },
  {
    "canonical_label": "sites_count",
    "data_type": "numeric",
    "synonyms": [
      "sites count", "number of sites", "site count", "number of study sites",
      "participating sites", "active sites", "investigational sites",
      "clinical sites", "research sites", "multi-site", "multisite"
    ],
    "clarification_question": "How many study sites are you looking for in {trigger}?",
    "clarification_options": [
      "Single site",
      "2–10 sites",
      "11–50 sites",
      "Over 50 sites",
      "No preference"
    ],
    "implied_operators": {
      "large": "gt", "small": "lt", "few": "lt", "many": "gt",
      "single": "eq", "multiple": "gt", "fewer than": "lt", "more than": "gt"
    },
    "fuzzy_threshold": 80
  },
  {
    "canonical_label": "avg_days_to_first_response",
    "data_type": "numeric",
    "synonyms": [
      "days to first response", "time to first response", "response time",
      "initial response time", "site response time", "days to respond",
      "average response time", "avg response time", "first response days",
      "response latency", "site activation time"
    ],
    "clarification_question": "What site response time do you need for {trigger}?",
    "clarification_options": [
      "Under 5 days",
      "5–10 days",
      "11–20 days",
      "Over 20 days",
      "No preference"
    ],
    "implied_operators": {
      "fast": "lt", "quick": "lt", "rapid": "lt", "slow": "gt",
      "under": "lt", "below": "lt", "within": "lte",
      "over": "gt", "above": "gt", "more than": "gt", "fewer than": "lt"
    },
    "fuzzy_threshold": 75
  },
  {
    "canonical_label": "study_duration_months",
    "data_type": "numeric",
    "synonyms": [
      "study duration", "trial duration", "study length", "trial length",
      "duration of study", "months of study", "study period", "trial period",
      "length of trial", "study timeline", "trial timeline", "duration months"
    ],
    "clarification_question": "What study duration are you looking for in {trigger}?",
    "clarification_options": [
      "Under 6 months",
      "6–12 months",
      "12–24 months",
      "Over 24 months",
      "No preference"
    ],
    "implied_operators": {
      "short": "lt", "long": "gt", "brief": "lt", "extended": "gt",
      "under": "lt", "over": "gt", "at least": "gte", "up to": "lte"
    },
    "fuzzy_threshold": 78
  },
  {
    "canonical_label": "screen_failure_rate",
    "data_type": "numeric",
    "synonyms": [
      "screen failure rate", "screening failure rate", "screen fail rate",
      "failed screening", "screening failures", "screen failure percentage",
      "screen fail percentage", "failed screens", "screen failure ratio"
    ],
    "clarification_question": "What screen failure rate threshold matters for {trigger}?",
    "clarification_options": [
      "Under 20%",
      "20–40%",
      "Over 40%",
      "No preference"
    ],
    "implied_operators": {
      "high": "gt", "low": "lt", "under": "lt", "over": "gt",
      "below": "lt", "above": "gt", "less than": "lt", "more than": "gt"
    },
    "fuzzy_threshold": 80
  },
  {
    "canonical_label": "enrollment_rate_per_month",
    "data_type": "numeric",
    "synonyms": [
      "enrollment rate", "monthly enrollment", "enrollment per month",
      "patients per month", "accrual rate", "monthly accrual", "accrual per month",
      "enrollment velocity", "recruitment rate", "monthly recruitment"
    ],
    "clarification_question": "What monthly enrollment rate do you need for {trigger}?",
    "clarification_options": [
      "Under 5 patients/month",
      "5–15 patients/month",
      "16–30 patients/month",
      "Over 30 patients/month",
      "No preference"
    ],
    "implied_operators": {
      "fast": "gt", "slow": "lt", "high": "gt", "low": "lt",
      "at least": "gte", "at most": "lte", "under": "lt", "over": "gt"
    },
    "fuzzy_threshold": 78
  },
  {
    "canonical_label": "dropout_rate",
    "data_type": "numeric",
    "synonyms": [
      "dropout rate", "drop-out rate", "drop out rate", "attrition rate",
      "withdrawal rate", "discontinuation rate", "lost to follow-up rate",
      "patient attrition", "study withdrawal rate", "early termination rate"
    ],
    "clarification_question": "What dropout rate constraint matters for {trigger}?",
    "clarification_options": [
      "Under 10%",
      "10–20%",
      "Over 20%",
      "No preference"
    ],
    "implied_operators": {
      "high": "gt", "low": "lt", "under": "lt", "over": "gt",
      "below": "lt", "above": "gt"
    },
    "fuzzy_threshold": 80
  },
  {
    "canonical_label": "protocol_amendment_count",
    "data_type": "numeric",
    "synonyms": [
      "protocol amendments", "protocol amendment count", "number of amendments",
      "amendments count", "protocol changes", "protocol revisions",
      "amendment frequency", "number of protocol revisions"
    ],
    "clarification_question": "How many protocol amendments are acceptable for {trigger}?",
    "clarification_options": [
      "Zero amendments",
      "1–2 amendments",
      "3 or more amendments",
      "No preference"
    ],
    "implied_operators": {
      "few": "lt", "many": "gt", "no": "eq", "zero": "eq",
      "minimal": "lt", "frequent": "gt", "more than": "gt", "fewer than": "lt"
    },
    "fuzzy_threshold": 78
  },
  {
    "canonical_label": "irb_approval_days",
    "data_type": "numeric",
    "synonyms": [
      "irb approval time", "irb turnaround", "irb review time", "ethics approval time",
      "irb days", "days to irb approval", "irb processing time",
      "ethics board approval time", "irb review days", "irb turnaround time"
    ],
    "clarification_question": "What IRB approval timeline do you need for {trigger}?",
    "clarification_options": [
      "Under 30 days",
      "30–60 days",
      "Over 60 days",
      "No preference"
    ],
    "implied_operators": {
      "fast": "lt", "quick": "lt", "slow": "gt", "under": "lt",
      "within": "lte", "over": "gt", "more than": "gt"
    },
    "fuzzy_threshold": 75
  },
  {
    "canonical_label": "contract_execution_days",
    "data_type": "numeric",
    "synonyms": [
      "contract execution time", "contract turnaround", "cta execution time",
      "cta turnaround", "contract days", "days to contract execution",
      "site contract time", "clinical trial agreement time", "contract cycle time",
      "cta cycle time", "contract processing time"
    ],
    "clarification_question": "What contract execution timeline matters for {trigger}?",
    "clarification_options": [
      "Under 30 days",
      "30–60 days",
      "Over 60 days",
      "No preference"
    ],
    "implied_operators": {
      "fast": "lt", "quick": "lt", "slow": "gt", "under": "lt",
      "within": "lte", "over": "gt", "more than": "gt"
    },
    "fuzzy_threshold": 75
  },
  {
    "canonical_label": "data_query_rate",
    "data_type": "numeric",
    "synonyms": [
      "data query rate", "query rate", "data queries", "number of queries",
      "query frequency", "data error rate", "data quality rate",
      "outstanding queries", "query volume", "edc query rate"
    ],
    "clarification_question": "What data query rate threshold matters for {trigger}?",
    "clarification_options": [
      "Under 5 queries per patient",
      "5–15 queries per patient",
      "Over 15 queries per patient",
      "No preference"
    ],
    "implied_operators": {
      "high": "gt", "low": "lt", "many": "gt", "few": "lt",
      "under": "lt", "over": "gt", "less than": "lt", "more than": "gt"
    },
    "fuzzy_threshold": 78
  },
  {
    "canonical_label": "site_initiation_visit_days",
    "data_type": "numeric",
    "synonyms": [
      "site initiation visit", "siv time", "siv days", "site initiation time",
      "initiation visit days", "days to site initiation", "site startup time",
      "site activation days", "startup visit time", "site readiness days",
      "time to site initiation"
    ],
    "clarification_question": "What site initiation timeline do you need for {trigger}?",
    "clarification_options": [
      "Under 30 days",
      "30–60 days",
      "Over 60 days",
      "No preference"
    ],
    "implied_operators": {
      "fast": "lt", "quick": "lt", "slow": "gt", "under": "lt",
      "within": "lte", "over": "gt", "more than": "gt"
    },
    "fuzzy_threshold": 75
  }
]
```

**Decision:** All 12 fields are `numeric` (no `date` or `categorical` in v1). Date fields deferred; the schema supports them for forward compatibility.

---

## 3. `MetricMatch` Dataclass

File: `src/normalizers/metric.py`

```python
import types

@dataclass(frozen=True)
class MetricMatch:
    canonical_field: str        # key from metric_filters.json
    canonical_label: str        # human display label
    matched_text: str           # substring from ORIGINAL canonical at same indices (NOT logged)
    span: tuple[int, int]       # (start, end) in normalized query
    data_type: str              # "numeric" | "date" | "categorical"
    match_source: str           # "ac_exact" | "ac_synonym" | "fuzzy"
    confidence: float           # 1.0 (AC) or ≤ 0.90 (fuzzy)
    implied_operator: str       # "lt"|"lte"|"gt"|"gte"|"eq"|"between"|"any" — default "any"
    implied_op_map: types.MappingProxyType  # from JSON — wraps dict at construction; immutable <!-- rev2 -->
```

<!-- rev2 --> **`implied_op_map` immutability:** pass `types.MappingProxyType(raw_dict)` when constructing each `MetricMatch`. The frozen dataclass prevents reassignment of the attribute; `MappingProxyType` prevents mutation of the dict contents. This replaces the plain `dict[str, str]` annotation from rev 1 (M13).

<!-- rev2 --> **`matched_text` span recovery:** `MetricIntentResolver.resolve()` computes spans in the *normalized* query (lowercase, punct-stripped). Because normalization replaces non-word characters with spaces (one-to-one substitution for ASCII text), character indices are preserved in the original canonical. `matched_text` is therefore recovered as `original_canonical[start:end]` (where `start`/`end` are the normalized-query span indices), which is an acceptable approximation for ASCII-dominant clinical text. See §18 Known Limitations.

---

## 4. `MetricFilterOutput` — Pydantic V2 Model <!-- review-hot -->

Field order is mandatory — validator on `value` uses `info.data` which requires `operator` and `data_type` to be declared before it.

<!-- rev2 --> **`unit` allowlist:** `unit` is restricted to a static set; it is server-derived from the field's `data_type` and JSON config, **never** extracted verbatim from the LLM response. A Pydantic V2 `field_validator` rejects any value outside the allowlist.

<!-- rev2 --> **`operator`/`data_type` required in validator:** `_validate_value` raises `ValueError` if `operator` or `data_type` is absent from `info.data` (not silently defaulted).

```python
from __future__ import annotations
from typing import Optional, Literal
from pydantic import BaseModel, ConfigDict, field_validator, ValidationInfo

VALID_OPERATORS = frozenset({"lt", "lte", "gt", "gte", "eq", "between", "any"})

# Server-derived unit allowlist — never sourced from LLM output <!-- rev2 -->
VALID_UNITS: frozenset[str | None] = frozenset(
    {"patients", "months", "days", "sites", "percent", "queries", None}
)

class MetricFilterOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    # Field order matters for validator cross-field access via info.data
    field: str                        # canonical_field key (e.g. "enrollment_count")
    canonical_label: str              # display label
    operator: str                     # "lt"|"lte"|"gt"|"gte"|"eq"|"between"|"any"
    data_type: str                    # "numeric"|"date"|"categorical"
    value: Optional[float | str]      # None = unresolved; str for date/categorical
    original_text: str                # matched phrase from query (NOT logged)
    confidence: float                 # 0.0–1.0
    unit: Optional[str] = None        # server-derived from data_type/JSON config (NOT from LLM)

    @field_validator("operator")
    @classmethod
    def _validate_operator(cls, v: str) -> str:
        if v not in VALID_OPERATORS:
            raise ValueError(f"operator must be one of {VALID_OPERATORS}, got {v!r}")
        return v

    @field_validator("unit")  # <!-- rev2 -->
    @classmethod
    def _validate_unit(cls, v: Optional[str]) -> Optional[str]:
        if v not in VALID_UNITS:
            raise ValueError(f"unit must be one of {VALID_UNITS}, got {v!r}")
        return v

    @field_validator("value", mode="before")
    @classmethod
    def _validate_value(cls, v, info: ValidationInfo) -> Optional[float | str]:
        # <!-- rev2 --> Raise if prerequisite fields are absent — do not silently default
        if "operator" not in info.data:
            raise ValueError("operator field missing — value validation requires operator")
        if "data_type" not in info.data:
            raise ValueError("data_type field missing — value validation requires data_type")
        operator = info.data["operator"]
        data_type = info.data["data_type"]
        if v is None:
            return None
        if isinstance(v, str) and v.strip().lower() in {"null", "", "none", "no preference"}:
            return None
        if data_type == "numeric":
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
        return v
```

**Decision:** `value` validator coerces `"No preference"` to `None`. This is the mechanism for the "No preference" flow (see §14).

---

## 5. `MetricFilterNormalizer` — Static Method Signatures

```python
class MetricFilterNormalizer:
    """Static normalization helpers for metric filter fields."""

    @staticmethod
    def normalize_operator(text: str) -> str:
        """Map a natural-language word or phrase to an operator string.

        Returns "any" if no match found.
        Checked against the field's implied_op_map; this method is the fallback
        for LLM-extracted operator_text that doesn't match the AC window scan.
        """
        ...

    @staticmethod
    def normalize_date(text: str) -> Optional[str]:
        """Parse a date string into ISO-8601 format (YYYY-MM-DD).

        Returns None if unparseable. Reserved for future date-type fields.
        """
        ...

    @staticmethod
    def normalize_value(
        raw: Optional[str],
        data_type: str,
        operator: str = "any",
    ) -> Optional[float | str]:
        """Coerce raw LLM string to typed value.

        - numeric: float() conversion; returns None on failure
        - "No preference": always returns None (triggers operator="any")
        - date: delegates to normalize_date()
        - categorical: strip whitespace only
        """
        ...
```

### Operator word → op mapping table (used in `normalize_operator` and window scan)

| Word/Phrase | Operator |
|---|---|
| less than, fewer than, under, below, <  | `lt` |
| at most, up to, no more than, ≤, <=     | `lte` |
| more than, greater than, over, above, > | `gt` |
| at least, minimum, ≥, >=               | `gte` |
| exactly, equal to, =                    | `eq` |
| between, from … to, … to …             | `between` |
| any, no preference, regardless          | `any` |

**Decision:** `normalize_operator` uses this table as a fallback only. The primary op-assignment is the per-field `implied_operators` map from JSON, scanned in a ±20-char window around the AC match span.

---

## 6. `MetricIntentResolver` — Class Shape & Algorithm

```python
import types

class MetricIntentResolver:
    def __init__(
        self,
        metric_filters_path: str,
        strict_validation: bool = True,   # <!-- rev2 --> mirrors AmbiguousTermsRegistry
    ) -> None:
        # strict_validation overridden by env var METRIC_STRICT_VALIDATION (default "true")
        self._entries: dict[str, dict]      # keyed by canonical_field
        self._automaton: ahocorasick.Automaton
        self._fuzzy_index: list[tuple[str, str, str, int]]  # (synonym, canonical_field, label, threshold)
        self._known_metric_fields: frozenset[str]            # <!-- rev2 --> for FilterExtractor allowlist check

    def get_entry(self, field_key: str) -> dict:  # <!-- rev2 --> replaces _load_metric_entry
        """Return the raw JSON entry dict for field_key. Raises KeyError if unknown."""
        return self._entries[field_key]
```

<!-- rev2 --> **Init pseudocode:** load JSON → call `_validate_entries(entries)` → for each entry, for each synonym:
```python
automaton.add_word(
    synonym.lower(),
    {
        "canonical_field": canonical_field,
        "label": label,
        "data_type": data_type,
        "fuzzy_threshold": fuzzy_threshold,
        "implied_op_map": types.MappingProxyType(implied_operators_dict),
        "pattern_len": len(synonym),   # <!-- rev2 B1 --> used in Step 2 span recovery
    }
)
```
Call `make_automaton()`. Store `self._entries = {e["canonical_label"]: e for e in entries}`. Build `_fuzzy_index`. Set `self._known_metric_fields = frozenset(self._entries.keys())`.

<!-- rev2 --> **No DEBUG logs containing user-derived strings.** No `logger.debug()` statements containing `matched_text`, `original_text`, raw LLM response, or window strings anywhere in this class. INFO logs only; counts/IDs/enums only — same convention as existing `aho_corasick.py` and `pipeline.py`.

### `resolve(canonical: str) -> list[MetricMatch]`

**Step 1 — Normalize.** `query = re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', ' ', canonical.lower())).strip()`

**Step 2 — AC scan + collect-all-then-resolve (overlap resolution).**

```python
# Collect ALL raw hits first; do NOT accept one-by-one
raw_hits: list[tuple[int, int, dict]] = []
for end_idx, payload in automaton.iter(query):
    term_len = payload["pattern_len"]   # <!-- rev2 B1 --> stored in payload at add_word time
    start = end_idx - term_len + 1
    end = end_idx + 1
    if not _is_word_bounded(query, start, end):
        continue
    raw_hits.append((start, end, payload))

# Overlap resolution: longer span wins; tie-break by confidence DESC
raw_hits.sort(key=lambda h: (-(h[1] - h[0]), h[0]))
accepted: list[tuple[int, int, dict]] = []
accepted_spans: list[tuple[int, int]] = []
for start, end, payload in raw_hits:
    # reject if overlaps any already-accepted span (not just contained)
    overlaps = any(
        not (end <= aks or start >= ake)
        for aks, ake in accepted_spans
    )
    if not overlaps:
        accepted.append((start, end, payload))
        accepted_spans.append((start, end))
```

<!-- rev2 --> **`matched_text` recovery from original canonical (B2/B7):** after the AC scan, recover `matched_text` from the *original* canonical string at the same character indices:
```python
# original_canonical is the un-normalized canonical passed into resolve()
matched_text = original_canonical[start:end]
# Normalization is punct→space (one-to-one for ASCII), so indices are valid.
# See §18 Known Limitations for edge cases.
```

**Decision:** overlap rejection is stricter than SNOMED AC (which uses containment only). For metric fields, partial overlaps also rejected because two metrics sharing a word should not both fire on the same span. See §18 Known Limitations for overlapping cross-field synonyms.

**Step 3 — Residual spans.**

```python
# Sort accepted by span start; find gaps between them
residual: list[tuple[int, int]] = []
prev_end = 0
for start, end, _ in sorted(accepted, key=lambda h: h[0]):
    if prev_end < start:
        residual.append((prev_end, start))
    prev_end = end
if prev_end < len(query):
    residual.append((prev_end, len(query)))
```

**Step 4 — Fuzzy fallback on residual spans only.**
- Already-matched canonical_fields: `matched_fields = {p["canonical_field"] for _, _, p in accepted}`
- For each residual span with length ≥ 6 chars: generate contiguous n-grams n ∈ [2, 5] (token-level).
- <!-- rev2 S13 --> **Fuzzy budget:** `MAX_FUZZY_COMPARISONS_PER_QUERY = 5000`. Maintain a running counter across all n-gram × synonym comparisons. When budget is exhausted, break the inner loops and log `logger.info("metric_fuzzy_budget_exhausted=true session_id=%s", ...)`. Counts only; no user-derived strings logged.
- For each n-gram: `score = token_sort_ratio(ngram, synonym)` for all `(synonym, canonical_field, label, threshold)` in `_fuzzy_index` where `canonical_field not in matched_fields`.
- <!-- rev2 M14 --> **`matched_text` for fuzzy hits:** use the n-gram window *as it appears in order in the canonical* (not the synonym, which may be token-reordered by `token_sort_ratio`). `matched_text = original_canonical[res_start + ngram_char_start : res_start + ngram_char_end]`.
- Keep if `score >= field_threshold` (from JSON per-field). `confidence = score / 100 * 0.90`.

**Step 5 — Dedup by canonical_field, keep max confidence.**

```python
best: dict[str, MetricMatch] = {}
for m in all_candidates:
    if m.canonical_field not in best or m.confidence > best[m.canonical_field].confidence:
        best[m.canonical_field] = m
```

**Step 6 — Implied-operator window scan.** For each accepted match, extract `query[max(0, start-20):min(len(query), end+20)]`. <!-- rev2 M15 --> Iterate `implied_op_map` keys **sorted by length descending** so multi-word keys (e.g. `"at least"`) win over prefix-matched single words (e.g. `"at"`). Assign `implied_operator`; default `"any"`.

Return `sorted(best.values(), key=lambda m: m.span[0])`.

**Confidence values:** AC hits = `1.0`. Fuzzy cap = `score/100 × 0.90`. No explicit floor — per-field `fuzzy_threshold` is the gate.

**Known limitation (document in context.md):** Negation ("not fast") in the ±20-char window is not handled in v1. Implied operator scan will incorrectly assign "lt" from "fast" even if preceded by "not". Out of scope for v1.

---

## 7. `MetricAmbiguityGate` — Class Shape <!-- review-hot -->

Location: `src/sufficiency_gate.py` (same file as `EmbeddingAmbiguityGate`).

<!-- rev2 B4 --> **`MetricAmbiguityGate` takes a `MetricIntentResolver` instance** — no separate JSON load. The gate calls `self._resolver.get_entry(field_key)` to retrieve clarification options and question templates at evaluation time.

```python
class MetricAmbiguityGate:
    """Step 7b — fire when a metric field is recognized but value is unresolved."""

    def __init__(self, resolver: MetricIntentResolver) -> None:  # <!-- rev2 B4 -->
        self._resolver = resolver

    def evaluate(
        self,
        metric_matches: list[MetricMatch],
        extracted_filters: ExtractedFilters,
        session: ConversationSession,
    ) -> Optional[SufficiencyDecision]:
        ...
```

### Firing condition
<!-- rev2 B6 --> Fire when ANY `MetricMatch` has no corresponding resolved value in `extracted_filters.metric_fields`, **with the following predicate** (updated to prevent double-clarification on "No preference"):

```python
unresolved = [
    m for m in metric_matches
    if (
        m.canonical_field not in extracted_filters.metric_fields
        or (
            extracted_filters.metric_fields[m.canonical_field].value is None
            and extracted_filters.metric_fields[m.canonical_field].operator != "any"  # <!-- rev2 B6 -->
        )
    )
]
```

A field with `operator == "any"` and `value is None` means the user already answered "No preference" — do NOT re-fire on it.

### Max-turns escape
`if session.is_max_turns_reached(): return None` — fall through, do not clarify.

### AmbiguousEntry synthesis

**Decision:** One combined question per gate activation, options from the FIRST unresolved field only (cap at 5 including "No preference").

```python
if not unresolved:
    return None

first = unresolved[0]

# <!-- rev2 B4 --> Use resolver.get_entry() — no separate JSON load
entry_json = self._resolver.get_entry(first.canonical_field)

# Combined question template if >1 unresolved field
if len(unresolved) > 1:
    fields_str = ", ".join(m.canonical_label for m in unresolved[1:4])
    question_template = (
        "I found {trigger} and other metric criteria ("
        + fields_str
        + "). " + entry_json["clarification_question"]
    )
else:
    question_template = entry_json["clarification_question"]

# <!-- rev2 M19 --> No padding logic — startup validation guarantees 3–5 options
# AmbiguousEntry.options validator enforces 3–5; startup _validate_entries enforces it too
options = entry_json["clarification_options"][:5]

# <!-- rev2 S1/S2 --> Use canonical_field as trigger (server-defined key, not user content)
# triggered_by=None → APPEND mode in compute_canonical_query (no substitution)
# AmbiguousEntry.trigger = canonical_field (server-defined JSON key)
entry = AmbiguousEntry(
    trigger=first.canonical_field,        # <!-- rev2 S1/S2 --> server-defined key, not user text
    category="metric",
    question_template=question_template,  # MUST contain literal {trigger}
    options=options,
    override_terms=frozenset(),           # no override needed for metrics
    max_options=5,
)

return SufficiencyDecision(
    sufficient=False,
    reason="metric_without_value",
    triggered_by=None,          # <!-- rev2 S1/S2/B5 --> APPEND mode; no user text in triggered_by
    matched_entry=entry,
)
```

**CRITICAL:** `question_template` MUST contain literal `{trigger}`. All 12 JSON entries are authored to contain it. `{trigger}` in the rendered question will be substituted with `entry.trigger` (= `first.canonical_field`, a server-defined string).

<!-- rev2 --> **`triggered_by=None` (APPEND mode):** Setting `triggered_by=None` means `compute_canonical_query` (see `conversation.py:170`) follows the `else` branch — it appends the user's clarification answer to the canonical rather than substituting. This is the same pattern used by `EmbeddingAmbiguityGate`. The original metric keyword (e.g. `"fast"`) stays in the canonical; on turn 2 the `MetricIntentResolver` re-detects it, the user's answer (e.g. `"Under 5 days"`) is appended, and the LLM extracts a concrete value. The gate then sees `operator != "any"` and `value is not None`, so it does not re-fire. See §14 for the full flow.

**`add "metric_without_value"` to the reason enum comment block** in `SufficiencyDecision` docstring.

<!-- rev2 S3 --> **No DEBUG logs containing user-derived strings.** No `logger.debug()` calls containing `matched_text`, `original_text`, raw LLM response, or window strings in this class. INFO logs only; counts/IDs/enums only.

---

## 8. `FilterExtractor` Changes

### 8a. `extract()` new signature
```python
def extract(
    self,
    canonical_query: str,
    metric_matches: list[MetricMatch] = None,  # <!-- rev2 M16 --> pass directly; test `if metric_matches:` inside
) -> ExtractedFilters:
```
Backwards-compatible. `metric_matches=None` → existing behavior unchanged. Test internally with `if metric_matches:` (not `metric_matches or None`).

### 8b. Dynamic prompt construction (Option ii — minimal change) <!-- review-hot -->

**Decision:** Keep `SYSTEM_PROMPT` as `ClassVar[str]` base. Concatenate `metric_section` in `extract()` before calling `provider.complete()`.

```python
def extract(self, canonical_query, metric_matches=None):
    system = self.SYSTEM_PROMPT
    if metric_matches:
        system = system + "\n\n" + self._build_metric_section(metric_matches)
    raw = self._provider.complete(system_prompt=system, ...)
```

<!-- rev2 S3 --> No `logger.debug()` calls containing user-derived strings in `extract()`. INFO only; counts only.

### 8c. `_build_metric_section(metric_matches) -> str`

<!-- rev2 S6 --> Before building the prompt, assert every match's `canonical_field` is in the known allowlist. Drop unknown fields silently with a count log:
```python
known = self._known_metric_fields  # frozenset captured from MetricIntentResolver at FilterExtractor init
safe_matches = [m for m in metric_matches if m.canonical_field in known]
if len(safe_matches) < len(metric_matches):
    logger.info(
        "_build_metric_section: dropped unknown fields count=%d",
        len(metric_matches) - len(safe_matches),
    )
```

<!-- rev2 M12 --> The prompt block must show **both** the snake_case field key and canonical label, and include a JSON example with snake_case keys verbatim. Parser silently drops unknown keys (logged as count only, never the key value):

```
ADDITIONAL TASK — Metric fields detected:

For each field below, extract from the query:

Fields:
  - Field key: enrollment_count (Enrollment Count)
  - Field key: avg_days_to_first_response (Average Days to First Response)
  [... one line per safe_match ...]

Respond with a JSON object containing a "metric_fields" key, using EXACTLY the
snake_case field keys shown above. Example:

{
  "metric_fields": {
    "avg_days_to_first_response": {
      "operator_text": "under",
      "value": "5",
      "value_end": null
    }
  }
}

Include "metric_fields" in your JSON response even if all values are null.
Do NOT invent values not present in the query.
Do NOT use camelCase keys.
```

### 8d. Response parsing for `metric_fields`

In `_validate()`, after building the base `ExtractedFilters`:
- Check `parsed.get("metric_fields", {})` — if empty dict or key absent, `metric_fields = {}`.
- For each `(field_key, raw_entry)` in parsed `metric_fields`:
  - Skip if `field_key` not in expected set (from `metric_matches`). Log count only.
  - Call `MetricFilterNormalizer.normalize_operator(raw_entry.get("operator_text"))` → `operator`.
  - If operator is `"any"` and the MetricMatch's `implied_operator != "any"`, use `implied_operator`.
  - Call `MetricFilterNormalizer.normalize_value(raw_entry.get("value"), data_type, operator)` → `value`.
  - <!-- rev2 --> `unit` is server-derived from the field's `data_type` entry in JSON config — **never** from LLM response.
  - Build `MetricFilterOutput(field=field_key, ..., value=value, operator=operator, unit=server_unit, ...)`.
- Return extended `ExtractedFilters` with `metric_fields=<dict>`.

<!-- rev2 S4 --> **Validation error handling:** catch `ValidationError` and log only `e.error_count()` and field-key count. Never log `str(e)` or `repr(e)` (they may contain field values).

```python
except ValidationError as exc:
    logger.info(
        "_validate metric_fields: skipped field error_count=%d",
        exc.error_count(),
        # NOT logged: str(exc), repr(exc), field values
    )
    continue
```

<!-- rev2 S3 --> No `logger.debug()` calls containing user-derived strings in this method. INFO only; counts/IDs/enums only.

---

## 9. `ExtractedFilters` Extension

```python
class ExtractedFilters(BaseModel):
    model_config = ConfigDict(frozen=True)
    investigator_name: FilterField
    site_name: FilterField
    city: FilterField
    state: StateFilter
    phase: FilterField
    raw_response_length: int
    metric_fields: dict[str, MetricFilterOutput] = Field(default_factory=dict)
```

**Decision:** `Field(default_factory=dict)` not `{}` — Pydantic V2 `frozen=True` models require `default_factory` for mutable defaults. The dict itself is not mutated after construction (frozen model prevents it); the factory just avoids the shared-default trap.

<!-- rev2 B8 --> **`_empty_filters()` in `pipeline.py`** must include `metric_fields={}` in its returned `ExtractedFilters`. This is required so legacy `run()` shim and `_assemble_best_effort_from_session` do not `AttributeError` on `.metric_fields`.

---

## 10. `ResponseAssembler.assemble` Extension

New signature:
```python
def assemble(
    self,
    filters: ExtractedFilters,
    snomed_matches: list[SNOMEDMatch],
    geo: GeoResult,
    start_time: float,
    metric_filters: Optional[list[MetricFilterOutput]] = None,
) -> NLPOutput:
```

Pass `metric_filters or []` into `NLPOutput`. Backwards-compatible (default `None`).

---

## 11. `NLPOutput` Extension

```python
class NLPOutput(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["search"] = "search"
    snomed_terms: list[SNOMEDTermOutput]
    filters: FiltersOutput
    metric_filters: list[MetricFilterOutput] = []   # NEW — append after filters
    metadata: MetadataOutput
```

**Decision:** `metric_filters` placed after `filters` and before `metadata` to group filter-family fields together. `= []` is safe in Pydantic V2 frozen models because it's a literal default (Pydantic copies it).

---

## 12. `pipeline.py` Integration

### 12a. New import
```python
from src.normalizers.metric import MetricIntentResolver, MetricMatch
from src.sufficiency_gate import MetricAmbiguityGate
```

### 12b. New module-level constant
```python
LOG_PATH_METRIC_AMBIGUITY = "metric_ambiguity_clarification"
```

### 12c. Init — add resolver and gate (after existing components, before executor)
```python
DEFAULT_METRIC_FILTERS_PATH = str(PROJECT_ROOT / "data" / "metric_filters.json")

# In __init__:
self._metric_resolver = MetricIntentResolver(
    metric_filters_path or DEFAULT_METRIC_FILTERS_PATH
)
# <!-- rev2 B4 --> MetricAmbiguityGate takes resolver instance — no separate path
self._metric_gate = MetricAmbiguityGate(self._metric_resolver)
```

Add `metric_filters_path: Optional[str] = None` to `__init__` signature.

### 12d. Step 3b — insert BEFORE `ThreadPoolExecutor.submit` calls (current line ~287)

```python
# ── Step 3b: Metric intent resolution (deterministic, pre-LLM) ───────────
metric_matches: list[MetricMatch] = self._metric_resolver.resolve(canonical)
logger.info(
    "run_with_session step=3b metric_matches=%d session_id=%s",
    len(metric_matches), session.session_id,
    # NOT logged: matched_text, canonical
)
```

### 12e. Step 4 — change filter future submission

<!-- rev2 M16 --> Pass `metric_matches` directly — do not collapse empty list to None:
```python
fut_filters = self._executor.submit(
    self._extractor.extract, canonical, metric_matches
)
```
`FilterExtractor.extract()` tests `if metric_matches:` internally.

### 12f. Step 7b — insert AFTER post_extraction_check block, BEFORE Step 8 geo

```python
# ── Step 7b: Metric ambiguity gate ───────────────────────────────────────
if metric_matches:
    metric_decision = self._metric_gate.evaluate(metric_matches, filters, session)
    if metric_decision is not None:
        log_path = LOG_PATH_METRIC_AMBIGUITY
        clarification = self._assembler.build_clarification(metric_decision, session, start)
        session.append_turn(Turn(
            turn_index=len(session.turns),
            user_input=user_text,
            canonical_query=canonical,
            decision=metric_decision,
            filters=filters,
            snomed_matches=snomed_matches,
            geo=None,
            timestamp=time.time(),
        ))
        self._log_turn(
            session, metric_decision, log_path, start,
            snomed_count=len(snomed_matches),
            filter_count=filter_set_count,
            metric_match_count=len(metric_matches),
            metric_resolved_count=0,
        )
        return clarification
```

### 12g. Step 9 — assemble call change

```python
resolved_metric_filters = list(filters.metric_fields.values()) if filters.metric_fields else []

output: NLPOutput = self._assembler.assemble(
    filters=filters,
    snomed_matches=snomed_matches,
    geo=geo,
    start_time=start,
    metric_filters=resolved_metric_filters,
)
```

### 12h. `_log_turn` extension <!-- review-hot -->

**Decision:** extend minimally with 2 new `int` params appended after existing params. Update ALL 4 call sites.

```python
def _log_turn(
    self,
    session: ConversationSession,
    decision: SufficiencyDecision,
    log_path: str,
    start: float,
    snomed_count: int = 0,
    filter_count: int = 0,
    metric_match_count: int = 0,       # NEW — count only, never matched_text
    metric_resolved_count: int = 0,    # NEW — count only, never field values
) -> None:
```

Add to `log_record` dict:
```python
"metric_match_count": metric_match_count,
"metric_resolved_count": metric_resolved_count,
```

**HIPAA:** only counts logged. `matched_text` is never logged. Field values never logged.

**4 call sites to update** (all pass `0, 0` for the new params unless step 7b/10):
1. Pre-extraction clarification return (~line 221) — `metric_match_count=0, metric_resolved_count=0`
2. Step 5b embedding gate return (~line 359) — `metric_match_count=len(metric_matches), metric_resolved_count=0`
3. Step 7 post-extraction safety (~line 408) — `metric_match_count=len(metric_matches), metric_resolved_count=len(filters.metric_fields)`
4. Step 10 search path (~line 446) — `metric_match_count=len(metric_matches), metric_resolved_count=len(resolved_metric_filters)`

### 12i. `_count_set_filters` and `_any_filter_set` extension <!-- rev2 M17/M18/S23 -->

<!-- rev2 --> These helpers live in `src/sufficiency_gate.py` (NOT inlined in pipeline.py). Update both functions there:

```python
def _count_set_filters(filters) -> int:
    """Return the count of non-empty filter fields.

    Includes resolved metric fields (value not None, operator not 'any').
    """
    count = 0
    if filters.investigator_name.value: count += 1
    if filters.site_name.value:         count += 1
    if filters.city.value:              count += 1
    if filters.state.values:            count += 1
    if filters.phase.value:             count += 1
    # NEW: each metric field with a resolved value counts as one filter
    for mf in getattr(filters, "metric_fields", {}).values():
        if mf.operator != "any" and mf.value is not None:  # <!-- rev2 M17 -->
            count += 1
    return count


def _any_filter_set(filters) -> bool:
    """Return True if at least one filter field has a non-empty value.

    Includes resolved metric fields.
    """
    if (
        filters.investigator_name.value
        or filters.site_name.value
        or filters.city.value
        or filters.state.values
        or filters.phase.value
    ):
        return True
    # NEW: check metric fields <!-- rev2 M18 -->
    return any(
        mf.operator != "any" and mf.value is not None
        for mf in getattr(filters, "metric_fields", {}).values()
    )
```

`post_extraction_check` in `SufficiencyGate` already calls `_count_set_filters` and `_any_filter_set` — no further change needed there; the unified count is picked up automatically.

### 12j. `_run_extraction_path` signature <!-- rev2 M21/B8 -->

```python
def _run_extraction_path(
    self,
    canonical: str,
    user_text: str,
    decision: SufficiencyDecision,
    session: ConversationSession,
    start: float,
    metric_matches: list[MetricMatch],   # <!-- rev2 M21 --> new parameter
) -> NLPOutput:
```

Step 3b in `run_with_session` passes `metric_matches` into `_run_extraction_path`. The legacy `_assemble_best_effort_from_session` passes `[]` explicitly — it **does not** support metric clarification (clean boundary; documented here). No `MetricIntentResolver.resolve()` call inside `_assemble_best_effort_from_session`.

---

## 13. `app.py` Rendering

### 13a. Operator display map (module-level constant)
```python
METRIC_OP_DISPLAY = {
    "lt": "<", "lte": "≤", "gt": ">", "gte": "≥",
    "eq": "=", "between": "between", "any": "any",
}
```

### 13b. Render block in `_render_nlp_output()` — append after existing filters section

```python
if output.metric_filters:
    st.subheader("📊 Metric Filters")
    for mf in output.metric_filters:
        op_symbol = METRIC_OP_DISPLAY.get(mf.operator, mf.operator)
        label = _safe(mf.canonical_label)
        if mf.value is not None:
            val_str = _safe(str(mf.value))
            unit_str = f" {_safe(mf.unit)}" if mf.unit else ""
            st.markdown(
                f"**{label}:** {op_symbol} {val_str}{unit_str}",
            )  # <!-- rev2 Sec-12 --> unsafe_allow_html removed; default is False
        else:
            st.markdown(f"**{label}:** any")  # <!-- rev2 Sec-12 --> no unsafe_allow_html
        conf_val = min(max(float(mf.confidence), 0.0), 1.0)
        st.progress(conf_val, text=f"{conf_val * 100:.0f}% confidence")
        st.divider()
```

**Use existing `_safe()` at `app.py:51`** (`html.escape(str(value))`). Do NOT replace or reimplement.

<!-- rev2 --> **XSS note:** `st.markdown()` without `unsafe_allow_html=True` renders content as escaped markdown text per Streamlit contract. `_safe()` provides defense-in-depth. `model_dump()` / `st.json()` render as escaped JSON text — no `unsafe_allow_html` path.

---

## 14. "No preference" Flow — APPEND Mode <!-- rev2 -->

<!-- rev2 B5/S1/S2 --> **Decision change from Rev 1:** Metric clarifications now use `triggered_by=None` (APPEND mode), matching the `EmbeddingAmbiguityGate` pattern. The rev 1 approach of substituting `first.matched_text` via `triggered_by` is withdrawn because: (a) `matched_text` is user-derived content in `triggered_by` (HIPAA risk via Turn serialization — S1/S2); (b) the substituted canonical could produce malformed queries (B5).

**Precise turn-2 flow for "No preference":**

1. **Turn 1:** User sends `"fast response time cancer"`.
2. **Step 3b:** `MetricIntentResolver` matches `avg_days_to_first_response` (via `"response time"` synonym); `implied_operator="lt"` (from `"fast"` in window).
3. **Step 7b:** `MetricAmbiguityGate` fires. `triggered_by=None`. `AmbiguousEntry.trigger="avg_days_to_first_response"` (server key). Clarification question rendered to user with options including `"No preference"`.
4. **Turn 2:** User selects `"No preference"`.
5. **`compute_canonical_query`:** `triggered_by=None` → APPEND branch → `canonical = "fast response time cancer No preference"`.
6. **Step 3b (turn 2):** `MetricIntentResolver` re-detects `avg_days_to_first_response`. Window scan still finds `"fast"` → `implied_operator="lt"`.
7. **Step 4 (turn 2):** LLM sees the appended `"No preference"` text. System prompt instructs: emit `operator="any"` when user expressed no preference. LLM emits `metric_fields: { "avg_days_to_first_response": { "operator_text": "any", "value": null } }`.
8. **Step 8d:** `normalize_value("No preference" or null, "numeric")` → `None`. `normalize_operator("any")` → `"any"`. `MetricFilterOutput` built with `operator="any", value=None`.
9. **Step 7b (turn 2):** Unresolved predicate: `operator == "any"` → field is excluded from `unresolved` list → gate does NOT re-fire.
10. Pipeline continues to search path.

**LLM system prompt addition for "no preference" detection (add to §8c block):**
```
If the user expressed "No preference" for a field, emit operator_text="any" and value=null for that field.
```

---

## 15. `src/normalizers/base.py` Extension

Add `MetricFilterNormalizer` Protocol matching the static-method pattern:

```python
class MetricFilterNormalizerProtocol(Protocol):
    """Protocol for metric filter normalizers."""

    @staticmethod
    def normalize_operator(text: str) -> str: ...

    @staticmethod
    def normalize_value(raw: Optional[str], data_type: str, operator: str) -> Optional[float | str]: ...
```

**Decision:** Protocol is for future extensibility. The concrete `MetricFilterNormalizer` class lives in `src/normalizers/metric.py` and does not need to inherit from this — duck-typing is sufficient for the pipeline.

---

## 16. Test Plan — `tests/test_metric_filters.py`

### Group 1 — AC Exact Match
- "enrollment count" → `canonical_field="enrollment_count"`, `confidence=1.0`, `match_source="ac_exact"`
- "sites count" → `canonical_field="sites_count"`
- Multi-field: "enrollment count and response time" → 2 matches, correct spans, no overlap

### Group 2 — AC Synonym Match
- "accrual rate" → `canonical_field="enrollment_rate_per_month"`, `confidence=1.0`
- "patient attrition" → `canonical_field="dropout_rate"`
- "cta turnaround" → `canonical_field="contract_execution_days"`

### Group 3 — Fuzzy Match (residual span only)
- "enrolment count" (typo) → fuzzy hit on `enrollment_count`, `confidence ≤ 0.90`
- "screenfailure rate" (merged) → fuzzy on `screen_failure_rate`
- True negative: "phase 3 diabetes cancer" → empty list (no metric match)

### Group 4 — Implied Operator Assignment
- "fast response time" → `implied_operator="lt"` on `avg_days_to_first_response`
- "large enrollment count" → `implied_operator="gt"` on `enrollment_count`
- No context → `implied_operator="any"`
- <!-- rev2 M15 --> "at least 50 sites" → `implied_operator="gte"` (not "any" from partial "at" match)

### Group 5 — Overlap / Dedup
- "enrollment rate per month" — should match `enrollment_rate_per_month`, NOT both `enrollment_count` and `enrollment_rate_per_month`
- Fuzzy skips already-AC-matched fields
- `resolve()` returns max-confidence match per canonical_field
- <!-- rev2 B3 --> **Overlap cross-field test (explicit assertion):** "total enrollment rate" → longest span wins (e.g. `enrollment_rate_per_month` if longer synonym matches); the shorter overlapping `enrollment_count` match is silently dropped. Assert only ONE result returned. This is expected behavior per §18 Known Limitations.

### Group 6 — `MetricFilterOutput` Validators
- `operator="invalid"` → `ValueError`
- `value="No preference"` → `value=None`
- `value="abc"` with `data_type="numeric"` → `value=None`
- `value="5.5"` with `data_type="numeric"` → `value=5.5`
- <!-- rev2 S9/S11 --> `unit="injections"` → `ValueError` (not in allowlist)
- <!-- rev2 S9/S11 --> `unit="days"` → accepted
- <!-- rev2 M9 --> construct with `operator` absent from payload → `ValueError("operator field missing")`

### Group 7 — `MetricFilterNormalizer`
- `normalize_operator("fewer than")` → `"lt"`
- `normalize_operator("at least")` → `"gte"`
- `normalize_operator("gibberish")` → `"any"`
- `normalize_value("No preference", "numeric")` → `None`
- `normalize_value("42", "numeric")` → `42.0`

### Group 8 — `MetricAmbiguityGate`
- Unresolved field → returns `SufficiencyDecision(sufficient=False, reason="metric_without_value")`
- <!-- rev2 S1/S2 --> `triggered_by` is `None` (not user text)
- <!-- rev2 B4 --> `entry.trigger` equals `first.canonical_field` (server-defined key)
- Combined question template contains literal `{trigger}` placeholder (assert `"{trigger}" in entry.question_template`)
- `options` length 3–5 (validator enforced)
- Max-turns reached → returns `None`
- All fields resolved → returns `None`
- <!-- rev2 B6 --> `operator="any"` + `value=None` already in `metric_fields` → gate does NOT re-fire
- <!-- rev2 M19 --> No padding code path: `options` from JSON always has 3–5 entries (startup validates)

### Group 9 — `_validate_entries` Startup Validation <!-- rev2 S16/S17 -->
- Field with `data_type="text"` → raises `ValueError` (strict mode)
- Field with empty synonym string `""` → raises `ValueError`
- Field with `implied_operators: {"over": "large"}` → raises `ValueError` (invalid op value)
- Field with `clarification_options: ["A", "B"]` → raises `ValueError` (fewer than 3)
- Field with `clarification_options` where last entry is not `"No preference"` → raises `ValueError`
- Field with `clarification_question` not containing `{trigger}` → raises `ValueError`
- Duplicate synonym across two fields → drops duplicate, logs `synonym_duplicates_dropped=1`
- Lenient mode (`strict_validation=False`): bad field skipped, valid fields loaded

---

## 17. `tests/batch_test_cases.csv` — 6 New Rows

Add columns `expected_metric_field`, `expected_metric_operator`, `expected_metric_value` (blank = don't check).

| id | category | description | input | expected_type | expected_metric_field | expected_metric_operator | expected_metric_value |
|---|---|---|---|---|---|---|---|
| 101 | metric_filter | enrollment count exact | diabetes trials enrollment count under 100 | search | enrollment_count | lt | 100 |
| 102 | metric_filter | response time implied op | fast response time sites Phase 3 cancer | clarification | avg_days_to_first_response | lt | |
| 103 | metric_filter | no metric match | Dr. Smith Phase 2 cardiovascular New York | search | | | |
| 104 | metric_filter | multi-field metric | enrollment count over 200 and dropout rate under 15% diabetes | search | enrollment_count | gt | 200 |
| 105 | metric_filter | fuzzy synonym | accrual rate diabetes phase 3 | search | enrollment_rate_per_month | any | |
| 106 | metric_filter | multi-turn no preference | fast response time cancer>>>No preference | search | avg_days_to_first_response | any | |

Row 106 uses `>>>` separator (confirmed). First turn fires MetricAmbiguityGate; second turn "No preference" appended to canonical → `operator="any"`, `value=None`, gate does not re-fire.

---

## 18. `context.md` Additions

### Post-Build Changes Log entry (append after existing entries)

```
### Metric filter recognition — AC automaton + fuzzy fallback (2026-05-XX)

Adds recognition of 12 clinical-trial operational metric fields (enrollment count,
sites count, response times, etc.) via a single Aho-Corasick automaton (Step 3b)
with rapidfuzz fallback on residual spans (Step 4 extended). Recognized fields
without extracted values trigger MetricAmbiguityGate (Step 7b) for clarification.
Resolved metrics appear in NLPOutput.metric_filters.

New files: data/metric_filters.json, src/normalizers/metric.py, tests/test_metric_filters.py
Modified: src/filter_extractor.py, src/sufficiency_gate.py, src/assembler.py,
          src/pipeline.py, src/normalizers/base.py, app.py, tests/batch_test_cases.csv

Design: rework-metric-filters.md
```

### Known Limitations entry (append to Known POC gaps section)

- Metric AC automaton synonym dedup: if two different fields share a synonym phrase, the first-loaded entry wins (pyahocorasick `add_word` silently ignores duplicate keys). Synonym phrases must be globally unique across all 12 fields. Startup `_validate_entries()` enforces this with a `synonym_duplicates_dropped` counter.
- Implied-operator negation not handled: "not fast" in the ±20-char window still assigns `lt` from "fast". Out of scope for v1.
- <!-- rev2 B2/B7 --> Normalized-to-original index mapping: `matched_text` is recovered from the original canonical using span indices computed on the normalized (lowercase, punct-stripped) query. Normalization replaces `[^\w\s]` with spaces (one-to-one for ASCII), so indices are valid for ASCII-dominant clinical text. Non-ASCII punctuation or multi-byte characters may shift indices; out of scope for v1.
- <!-- rev2 B3 --> Overlapping cross-field synonyms favor the longest match; the shorter overlapping match is silently dropped. Design favors precision over recall. Example: "total enrollment rate" may match `enrollment_rate_per_month` only if that synonym is longer; `enrollment_count` match on the overlapping span is dropped. See Group 5 test assertion.
- <!-- rev2 S9/S11 --> `st.json()` renders `model_dump()` as escaped JSON text (no `unsafe_allow_html`) — XSS-safe by Streamlit contract.
- <!-- rev2 M23/S24 --> NEVER-LOGGED additions: `MetricMatch.matched_text`, `MetricFilterOutput.original_text` join the existing NEVER-LOGGED list in context.md.
- <!-- rev2 --> `"metric_ambiguity_clarification"` added to the per-turn log `path` enum list.

---

## 19. Decisions Taken / Open Questions

**Decisions taken:**

| # | Decision | Rationale |
|---|---|---|
| D1 | All 12 fields `data_type=numeric` in v1 | Simplifies normalizer; date fields defer |
| D2 | Fuzzy threshold per-field (75–80) | Different synonym density per field domain |
| D3 | Overlap rejection = any overlap, not just containment | Prevents co-firing of related metric synonyms |
| D4 | "No preference" via APPEND mode (`triggered_by=None`) | Removes HIPAA risk; eliminates B5 canonical-merge nonsense; mirrors EmbeddingAmbiguityGate |
| D5 | `SYSTEM_PROMPT` stays ClassVar; `metric_section` concatenated in `extract()` | Minimal change; no method-ref breakage in existing call sites |
| D6 | Resolved metric fields count toward `_count_set_filters` | A metric filter is a filter; prevents false clinical-intent rejection |
| D7 | `_log_turn` extended with 2 int params (minimal) | Preserves HIPAA hygiene; no lists, no text |
| D8 | `MetricAmbiguityGate` in `src/sufficiency_gate.py` | Same module as `EmbeddingAmbiguityGate`; no new cross-module dependency |
| D9 | Combined question uses options from first unresolved field only | Gate generates valid AmbiguousEntry (3–5 options enforced by validator) |
| D10 | `MetricAmbiguityGate` takes `MetricIntentResolver` instance | No duplicate JSON load; single source of truth for entries |
| D11 | `implied_op_map` wrapped in `MappingProxyType` | Frozen dataclass + immutable mapping prevents all mutation paths |
| D12 | `unit` server-derived from field config; validated against allowlist | Prevents LLM-injected arbitrary strings flowing into UI / model_dump |
| D13 | Legacy `_assemble_best_effort_from_session` passes `[]` for metric_matches | Clean boundary; shim does not support metric clarification |
| D14 | `_count_set_filters` / `_any_filter_set` updated in `sufficiency_gate.py` | Single source of truth; `post_extraction_check` picks up change automatically |

**Contentious decisions (mark for reviewer challenge):**

- **D3 (overlap = any overlap):** Stricter than SNOMED AC. Could produce false negatives if a legitimate two-field query has adjacent metric phrases. Reviewers should verify with multi-field test cases. Documented as known limitation.
- **D4 (APPEND mode):** Relies on LLM emitting `operator="any"` when it sees "No preference" text. The system prompt explicitly instructs this. If LLM omits `metric_fields` key entirely, the field stays unresolved and gate re-fires (acceptable: user sees same clarification again). If LLM emits the key with `operator="any"`, gate correctly suppresses.
- **D6 (metric filter → `_count_set_filters`):** Means a query with only a metric filter and no SNOMED passes the clinical-intent gate. Correct behavior but reviewers should confirm it does not weaken the injection guard.

**Open questions (need orchestrator decision):**

- None blocking. All preflight findings resolved in spec. Dev agents may proceed.

---

## 20. Rev 2 Errata Summary

| ID | Resolution |
|----|-----------|
| B1 | Added `pattern_len: int` to AC payload dict in `add_word` call. Step 2 pseudocode now uses `payload["pattern_len"]` instead of `len(payload["synonym"])`. |
| B2/B7 | Specified that `matched_text` is recovered from the *original* canonical at normalized-query span indices. Documented acceptable approximation in §18 Known Limitations. |
| B3 | Accepted limitation. Added explicit test in Group 5 asserting longest-match wins and second field is dropped. Added Known Limitations bullet in §18. |
| B4 | `_load_metric_entry` removed. `MetricIntentResolver.get_entry(field_key)` specified. `MetricAmbiguityGate.__init__` takes resolver instance; calls `self._resolver.get_entry()`. |
| B5 | Switched metric clarifications to `triggered_by=None` (APPEND mode). §14 rewritten with precise turn-by-turn flow. LLM prompt addition for "no preference" → `operator="any"` documented. |
| B6 | Unresolved predicate updated: `operator != "any"` added so fields with `operator="any"` + `value=None` are not counted as unresolved. Combined with B5 fix. |
| B8/M22 | `_empty_filters()` specified to include `metric_fields={}`. `_run_extraction_path` given `metric_matches: list[MetricMatch]` parameter. Legacy `_assemble_best_effort_from_session` passes `[]` explicitly (no metric clarification support — clean boundary). |
| S1/S2 | `triggered_by=None` for all metric clarifications. `AmbiguousEntry.trigger=first.canonical_field` (server-defined key). No user-derived content in triggered_by or trigger fields. |
| S3 | Added explicit no-DEBUG-with-user-strings rule to §6, §7, §8. INFO logs only; counts/IDs/enums only. |
| S4 | `FilterExtractor` metric parsing catches `ValidationError` and logs only `e.error_count()`. Never logs `str(e)` or `repr(e)`. |
| S6 | `_build_metric_section` asserts `canonical_field in self._known_metric_fields` before building prompt. Drops unknown fields with count log. |
| S9/S11 | `unit` field validated against static allowlist `VALID_UNITS`. `unit` is server-derived from field config, never from LLM. `st.json()` XSS-safety noted in §18. |
| S13 | `MAX_FUZZY_COMPARISONS_PER_QUERY = 5000` budget added to fuzzy fallback. Short-circuits inner loops when exhausted; logs count metric. |
| S14/S15 | `MetricIntentResolver.__init__` has `strict_validation: bool = True` parameter. Env var `METRIC_STRICT_VALIDATION` (default `"true"`). Strict → raise; lenient → warn+skip. Mirrors `AmbiguousTermsRegistry`. |
| S16/S17 | `_validate_entries()` specified with exact rules: data_type allowlist, non-empty synonyms, implied_operator values, fuzzy_threshold range, options 3–5 with last="No preference", question contains `{trigger}`, cross-field synonym duplicate check with `synonym_duplicates_dropped` counter. |
| M9 | `_validate_value` raises `ValueError("operator field missing")` / `ValueError("data_type field missing")` if keys absent from `info.data`. No silent defaults. |
| M10 | Resolved by B5/B6: `operator="any"` predicate handles LLM omitting `metric_fields` key (field stays absent → treated as unresolved for subsequent turn, or already-"any" → gate suppresses). |
| M12 | `_build_metric_section` prompt shows `"Field key: snake_case_key (Label)"` format and includes a JSON example with snake_case keys verbatim. Unknown keys dropped silently (count logged). |
| M13 | `implied_op_map` changed from `dict[str, str]` to `types.MappingProxyType`. Wrapped at `MetricMatch` construction. |
| M14 | Fuzzy `matched_text` uses the n-gram window in-order from the original canonical (not token-sort-reordered synonym). |
| M15 | Implied-operator window scan iterates `implied_op_map` keys sorted by length descending. |
| M16 | `metric_matches` passed directly (not `metric_matches or None`). `FilterExtractor.extract()` tests `if metric_matches:` internally. |
| M17/M18/S23 | `_count_set_filters` and `_any_filter_set` updated in `sufficiency_gate.py` (not pipeline.py). Both now count metric fields with `operator != "any"` and `value is not None`. |
| M19 | Padding logic removed. Startup `_validate_entries()` guarantees 3–5 options per field; `AmbiguousEntry.options` validator enforces at construction. |
| M20 | `SufficiencyDecision.reason` — `"metric_without_value"` added to reason enum comment block. Dev should add a `field_validator` enforcing known reason values for forward safety (noted as implementation detail). |
| M21 | `_run_extraction_path` signature updated with `metric_matches: list[MetricMatch]` parameter. |
| M23/S24 | `MetricMatch.matched_text` and `MetricFilterOutput.original_text` added to context.md NEVER-LOGGED list in §18. |
| Sec-12 | `unsafe_allow_html=True` removed from both `st.markdown` calls in §13b render block. |
| §17 Row 106 | `>>>` separator confirmed. |
| §18 | `"metric_ambiguity_clarification"` added to per-turn log `path` enum list. |
