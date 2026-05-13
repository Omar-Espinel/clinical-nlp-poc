# rework-metric-filters-v2 — Implementation Spec

**Status:** REV2 — post QA + Security adversarial review, user-signed off
**Owner:** Architect
**Date:** 2026-05-13
**Source of truth:** this document; replaces `data/metric_filters.json` and `tests/test_metric_filters.py` in full

<!-- rev2 -->
**Rev2 changelog (post-review):**
- C1: `value_end` added to `MetricFilterOutput` (user decision: enable `between` operator for dates AND numerics)
- C2: Year-range validation added to `normalize_date` (DOB-leak mitigation, Sec-SB1)
- C3: Char-count error fixed in field #9/#10 design notes + Group 5 test case (QA-B1)
- C4: Field #6 `override_terms` removed — concept doesn't exist in metric registry; AC longer-wins suffices (QA-M7)
- C5: Date E2E test added to Group 6 test plan (QA-B3)
- C6: `isinstance(v, str)` guard added to §3.2 code (QA-M1)
- C7: `operator="any"` date-branch guard added (Sec-SM1)
- Q1 user decision: field #11 = screening velocity (rate of candidates screened), NOT screen-failure rate
- Q3 user decision: ADD `value_end` to schema — full `between` support for dates and numerics
- Q9: rapidfuzz version pinning note added to test plan

---

## 1. Overview & Motivation

The current 12 entries in `data/metric_filters.json` are generic CTMS KPIs (enrollment_count, sites_count, screen_failure_rate, etc.) authored before the actual Advarra schema was confirmed. The product team has now defined 12 concrete metrics that map directly to the Advarra search-results widget (counts of studies, average days for query/approval/FPE workflows, protocol deviations, screening rate, etc.). None of the existing 12 are reused; 1 of the new 12 is a date (`most_recent_approval_date`), which exercises a code path (`data_type=="date"`) that exists in the normalizer skeleton but was never wired end-to-end.

**Strategy: full replace, no deprecation cycle.** This is a POC. No external consumer depends on the old field names. A single commit replaces the JSON and the test file. Stage the JSON+pipeline+app changes together so the existing test suite never sees an inconsistent state.

**Out of scope:**
- Standard filters (investigator_name, site_name, city, state, phase) — already handled by `FilterExtractor.SYSTEM_PROMPT`, unchanged.
- SNOMED concepts.
- AC automaton overlap algorithm, fuzzy budget, HIPAA log policy — unchanged.
- Adding new operators or units to `VALID_OPERATORS` / `VALID_UNITS` — see §3 for the one exception (date-range serialization).

---

## 2. Field Definitions

Conventions applied across all 12 entries:
- `clarification_question` MUST contain literal `{trigger}` (validator at `src/normalizers/metric.py:374-377`).
- `clarification_options` MUST be 3–5 entries, last MUST be exactly `"No preference"`.
- Synonyms lead with the user-facing label lowercased, then add ≤12 paraphrases. AC automaton runs on lowercased text — case doesn't matter at runtime but keep entries lowercase for readability.
- `fuzzy_threshold` 80 for short/distinctive labels, 75 for multi-word phrases prone to typos.
- `implied_operators`: only words/phrases NOT already in `_OP_WORD_MAP` (`metric.py:32-60`) need to appear here. Field-specific qualitative cues (`high`, `low`, `fast`, `quick`, `extensive`) belong in the field's `implied_operators`. Numeric/positional operators (`over`, `under`, `more than`, `at least`, `between`) are picked up by `MetricFilterNormalizer.normalize_operator()` post-LLM-parse from `operator_text`. The window scan in `MetricIntentResolver._scan_implied_op` (`metric.py:539-553`) only fires for words in the per-field map, so it's fine to leave numeric phrasings out and let the LLM emit them via `operator_text`.

### 1. `total_studies_with_advarra`
- **Type:** numeric
- **User-facing name:** "Total Studies with Advarra"
- **Synonyms (12):** `total studies with advarra`, `studies with advarra`, `total advarra studies`, `advarra study count`, `advarra studies count`, `total studies at advarra`, `number of advarra studies`, `advarra portfolio size`, `studies in advarra`, `advarra trial count`, `total trials with advarra`, `advarra study portfolio`
- **Implied operators:** `{"large": "gt", "small": "lt", "many": "gt", "few": "lt", "no": "eq", "zero": "eq"}`
- **Clarification question:** `"What total Advarra study count are you looking for in {trigger}?"`
- **Clarification options:** `["Under 100", "100–500", "500–1000", "Over 1000", "No preference"]`
- **Override terms (manual):** *(none — see Design Notes)*
- **Fuzzy threshold:** 78
- **Design notes:** Highest risk of AC collision with `studies_matching_search`. Disambiguator: every synonym in field #1 contains the literal token `advarra` (or `at advarra`), every synonym in #2 contains `matching`/`match`/`my search`. Do NOT add bare `total studies` here — it would collide with #2's spirit and lose the Advarra-specific anchor.

### 2. `studies_matching_search`
- **Type:** numeric
- **User-facing name:** "Studies Matching Search"
- **Synonyms (12):** `studies matching search`, `studies matching my search`, `matching studies`, `studies matching the search`, `search match count`, `matched studies count`, `studies that match`, `studies in my search`, `studies in this search`, `studies matching criteria`, `studies matching my filters`, `matching study count`
- **Implied operators:** `{"large": "gt", "small": "lt", "no": "eq", "zero": "eq"}`
- **Clarification question:** `"How many matching studies are you targeting in {trigger}?"`
- **Clarification options:** `["Under 10", "10–50", "50–200", "Over 200", "No preference"]`
- **Override terms (manual):** *(none)*
- **Fuzzy threshold:** 78
- **Design notes:** Synonyms all carry `match*` as the anchor token. Bare `studies` is deliberately omitted — it's too generic; the LLM's filter section will not interpret it as a metric trigger absent context. Note "matching" overlaps with field #8 (`...matching_studies`) and #9 (`...matching_studies`); the cross-field overlap rule (longer span wins) protects us: #8 synonym `protocol deviations in matching studies` (38 chars) outranks #2's `matching studies` (16 chars).

### 3. `active_trials`
- **Type:** numeric
- **User-facing name:** "Active Trials"
- **Synonyms (10):** `active trials`, `active studies`, `ongoing trials`, `ongoing studies`, `currently active trials`, `currently enrolling trials`, `live trials`, `live studies`, `trials in progress`, `studies in progress`
- **Implied operators:** `{"high": "gt", "low": "lt", "many": "gt", "few": "lt", "no": "eq", "zero": "eq"}`
- **Clarification question:** `"How many active trials matter for {trigger}?"`
- **Clarification options:** `["Under 5", "5–25", "25–100", "Over 100", "No preference"]`
- **Override terms (manual):** *(none)*
- **Fuzzy threshold:** 80
- **Design notes:** Distinct anchor `active`/`ongoing`/`live`/`in progress`. No collision with #1 or #2 because none of those use these adjectives.

### 4. `most_recent_approval_date`  **← date type**
- **Type:** date (MM/YYYY)
- **User-facing name:** "Most Recent Approval Date"
- **Synonyms (12):** `most recent approval date`, `most recent approval`, `latest approval date`, `latest approval`, `last approval date`, `last approval`, `recent approval date`, `most recent ibr approval`, `most recent irb approval`, `latest irb approval`, `most recent approval month`, `last approved date`
- **Implied operators:** `{"since": "gte", "after": "gt", "before": "lt", "on or after": "gte", "on or before": "lte", "in": "eq", "between": "between", "recent": "gte", "recently": "gte"}`
- **Clarification question:** `"What approval recency are you looking for in {trigger}?"`
- **Clarification options (relative):** `["Last 6 months", "Last 12 months", "Last 2 years", "Older than 2 years", "No preference"]`
- **Override terms (manual):** *(none)*
- **Fuzzy threshold:** 78
- **Design notes:**
  - `irb approval` appears in synonyms — there's no IRB-approval-days field in this rewrite, so no collision. But verify the LLM is told `most_recent_approval_date` is a DATE, not a count of days. The current `_build_metric_section` (`filter_extractor.py:162-213`) passes `canonical_field (canonical_label)` only; the LLM cannot distinguish numeric from date from the prompt. **See §3 for required `_build_metric_section` change.**
  - Clarification options are relative-window strings. The LLM must convert them to MM/YYYY when echoed back as the next-turn answer. Simpler path: pre-compute the absolute MM/YYYY at gate-fire time using `date.today()` and offer absolute options like `"Since 11/2025"`, `"Since 05/2025"`, `"Since 05/2024"`. This bypasses LLM math entirely. **Recommendation: use absolute MM/YYYY options computed at gate time** — see §3.4.

### 5. `avg_days_respond_to_queries`
- **Type:** numeric
- **User-facing name:** "Average Days to Respond to Queries"
- **Synonyms (12):** `average days to respond to queries`, `avg days to respond to queries`, `average query response time`, `average days respond to queries`, `query response days`, `days to respond to queries`, `query response time`, `avg query response time`, `query turnaround`, `query turnaround time`, `query response latency`, `time to respond to queries`
- **Implied operators:** `{"fast": "lt", "quick": "lt", "rapid": "lt", "slow": "gt", "high": "gt", "low": "lt"}`
- **Clarification question:** `"What query response time threshold matters for {trigger}?"`
- **Clarification options:** `["Under 3 days", "3–7 days", "7–14 days", "Over 14 days", "No preference"]`
- **Override terms (manual):** *(none)*
- **Fuzzy threshold:** 75
- **Design notes:** No collision with other fields. `query` is the anchor token and appears nowhere else.

### 6. `avg_days_submission_to_approval`
- **Type:** numeric
- **User-facing name:** "Average Days, Submission to Approval"
- **Synonyms (12):** `average days submission to approval`, `avg days submission to approval`, `days submission to approval`, `submission to approval time`, `submission to approval days`, `average approval turnaround`, `approval turnaround time`, `submission approval cycle`, `days to approval from submission`, `submission to irb approval`, `time from submission to approval`, `avg submission to approval`
- **Implied operators:** `{"fast": "lt", "quick": "lt", "slow": "gt", "high": "gt", "low": "lt"}`
- **Clarification question:** `"What submission-to-approval timeline are you targeting in {trigger}?"`
- **Clarification options:** `["Under 14 days", "14–30 days", "30–60 days", "Over 60 days", "No preference"]`
- **Override terms (manual):** *(none — see Design Notes)* <!-- rev2: removed; override_terms concept doesn't exist in metric registry schema -->
- **Fuzzy threshold:** 75
- **Design notes:** Shares the token `approval` with #4. Cross-field overlap test: query `"submission to approval days"` (27 chars) vs the longest field-4 synonym `"most recent ibr approval"` (24 chars) — #6's anchor wins on length under AC longer-wins. <!-- rev2: removed override_terms block. The metric registry has no override-terms mechanism (only the ambiguous-terms registry does, see QA-M7). AC longer-wins is sufficient because every field-6 synonym is ≥27 chars and every field-4 synonym is ≤27 chars with the discriminating anchor `approval date`/`approval month` absent. A regression test in Group 5 verifies this. -->

### 7. `total_protocol_deviations_all_studies`
- **Type:** numeric
- **User-facing name:** "Total Protocol Deviations, all Studies"
- **Synonyms (12):** `total protocol deviations all studies`, `protocol deviations in all my studies`, `protocol deviations across all studies`, `total protocol deviations`, `protocol deviations all studies`, `deviation count all studies`, `total deviations across studies`, `all study protocol deviations`, `protocol deviation total`, `total pds all studies`, `protocol deviations in all studies`, `total deviation count`
- **Implied operators:** `{"high": "gt", "low": "lt", "many": "gt", "few": "lt", "no": "eq", "zero": "eq"}`
- **Clarification question:** `"What total protocol deviation count matters for {trigger}?"`
- **Clarification options:** `["0", "1–10", "10–50", "Over 50", "No preference"]`
- **Override terms (manual):** *(none — field-8 synonyms are longer; AC overlap wins for the matching-scoped case)*
- **Fuzzy threshold:** 78
- **Design notes:** This is the highest-risk overlap pair in the registry. Disambiguator: #7 always carries `all studies` or `total ... studies`. #8 always carries `matching studies` or `matching` as the scoping qualifier. The synonyms below are picked so that NO synonym appears verbatim in both fields. Confirm at startup via the duplicate-synonym validator (`_validate_entries`, `metric.py:386-403`).

### 8. `total_protocol_deviations_matching_studies`
- **Type:** numeric
- **User-facing name:** "Total Protocol Deviations, Matching Studies"
- **Synonyms (12):** `total protocol deviations matching studies`, `protocol deviations in matching studies`, `protocol deviations across my matching studies`, `protocol deviations matching studies`, `deviation count matching studies`, `protocol deviations for matching studies`, `pds in matching studies`, `matching studies protocol deviations`, `total deviations matching studies`, `protocol deviation total matching`, `matching study protocol deviations`, `deviations in matching studies`
- **Implied operators:** same as #7
- **Clarification question:** `"What protocol deviation count for matching studies are you targeting in {trigger}?"`
- **Clarification options:** `["0", "1–5", "5–20", "Over 20", "No preference"]`
- **Override terms (manual):** *(none)*
- **Fuzzy threshold:** 78
- **Design notes:** See #7. Verify via test: query `"how many protocol deviations in all my studies"` → field #7 only; query `"protocol deviations in my matching studies"` → field #8 only.

### 9. `avg_enrollment_matching_studies`
- **Type:** numeric
- **User-facing name:** "Average Enrollment, Matching Studies"
- **Synonyms (10):** `average enrollment matching studies`, `avg enrollment matching studies`, `mean enrollment matching studies`, `average enrollment in matching studies`, `enrollment average matching studies`, `average matched enrollment`, `avg matched enrollment`, `mean matched enrollment`, `enrollment per matching study`, `matching study enrollment average`
- **Implied operators:** `{"high": "gt", "low": "lt", "large": "gt", "small": "lt"}`
- **Clarification question:** `"What average enrollment per matching study are you targeting in {trigger}?"`
- **Clarification options:** `["Under 25", "25–100", "100–500", "Over 500", "No preference"]`
- **Override terms (manual):** *(none)*
- **Fuzzy threshold:** 78
- **Design notes:** Triple-overlap zone with #10 and #11. All field-9 synonyms include both `enrollment` AND `matching studies` (or `matched`). Field #10 synonyms include `enrollment` AND `ta`/`therapeutic area`. Field #11 has no `enrollment`.
  <!-- rev2: corrected char-count claim per QA-B1 -->
  **AC overlap behavior between #9 and #10:** field-9 synonym `"average enrollment matching studies"` (37 chars) is LONGER than field-10 `"average enrollment matching ta"` (32 chars). For ambiguous compound queries containing both anchors, field #9 wins AC overlap by length. Disambiguation in practice requires distinct surface forms — see Group 5 test case rewritten to use `"high average enrollment in matching ta"` (38 chars including "in matching ta" anchor, beats field-9 anchors).

### 10. `avg_enrollment_matching_ta`
- **Type:** numeric
- **User-facing name:** "Average Enrollment, Matching TA"
- **Synonyms (12):** `average enrollment matching ta`, `avg enrollment matching ta`, `average enrollment matching therapeutic area`, `avg enrollment matching therapeutic area`, `average enrollment ta`, `average ta enrollment`, `avg ta enrollment`, `mean ta enrollment`, `enrollment by therapeutic area`, `enrollment per ta`, `enrollment for matching therapeutic area`, `enrollment in matching ta`
- **Implied operators:** same as #9
- **Clarification question:** `"What average enrollment for the matching therapeutic area are you targeting in {trigger}?"`
- **Clarification options:** `["Under 25", "25–100", "100–500", "Over 500", "No preference"]`
- **Override terms (manual):** *(none)*
- **Fuzzy threshold:** 78
- **Design notes:** "TA" is a 2-character abbreviation; the AC automaton respects whole-word boundaries (`_is_word_bounded`, `metric.py:218-222`), so `ta` will only match as a standalone token, not embedded in `data`/`status`/etc. Verify with a unit test.

### 11. `avg_screening_rate`
- **Type:** numeric
- **User-facing name:** "Average Screening Rate"
- **Synonyms (10):** `average screening rate`, `avg screening rate`, `mean screening rate`, `screening rate average`, `screening rate`, `patient screening rate`, `average screening rate per site`, `screening velocity`, `screening throughput`, `screen rate`
- **Implied operators:** `{"high": "gt", "low": "lt", "fast": "gt", "slow": "lt"}`
- **Clarification question:** `"What screening rate threshold matters for {trigger}?"`
- **Clarification options:** `["Under 1 per week", "1–5 per week", "5–10 per week", "Over 10 per week", "No preference"]`
- **Override terms (manual):** *(none)*
- **Fuzzy threshold:** 78
- **Design notes:** `screening rate` (15 chars) is the minimum-anchor synonym. Does NOT overlap with any other field. Note: not the same as the old `screen_failure_rate` — confirm with product that screening rate means "rate at which candidates are screened," not "rate of screen failures."

### 12. `avg_days_to_fpe`
- **Type:** numeric
- **User-facing name:** "Average Days to FPE (First Patient Enrolled)"
- **Synonyms (12):** `average days to fpe`, `avg days to fpe`, `days to fpe`, `days to first patient enrolled`, `time to fpe`, `time to first patient enrolled`, `first patient enrolled days`, `fpe time`, `fpe days`, `days to first patient`, `average days to first patient enrolled`, `time to first enrollment`
- **Implied operators:** `{"fast": "lt", "quick": "lt", "rapid": "lt", "slow": "gt", "high": "gt", "low": "lt"}`
- **Clarification question:** `"What time to first patient enrolled matters for {trigger}?"`
- **Clarification options:** `["Under 30 days", "30–90 days", "90–180 days", "Over 180 days", "No preference"]`
- **Override terms (manual):** *(none)*
- **Fuzzy threshold:** 75
- **Design notes:** `fpe` is a 3-letter token; word-boundary safe. No collision risk with the rest of the registry.

---

## 3. Schema / Code Impacts

### 3.1 `data/metric_filters.json` (full replace)
Replace all 12 entries. Top-level shape is unchanged (list of dicts).

### 3.2 `MetricFilterOutput._validate_value` — date path verification
**Current behavior** (`src/normalizers/metric.py:112-137`, line numbers approximate):
- `data_type == "numeric"` → `float(v)` coercion.
- Anything else (including `data_type == "date"`) → falls through `return v` unchanged.

**Gap:** date values are passed through as strings without parsing. A literal `"01/2026"` survives validation; a literal `"January 2026"` ALSO survives (it's a non-empty string and operator isn't `"any"`).

<!-- rev2: added isinstance guard (QA-M1) + operator=any short-circuit (Sec-SM1) -->
**Fix required:** add an explicit `data_type == "date"` branch in `_validate_value`, with type guard and `any`-operator short-circuit:
```python
if data_type == "date":
    if operator == "any":
        # operator=any ⇔ value=None invariant; trust the operator-validator's downstream check
        return None
    if not isinstance(v, str):
        # LLM occasionally returns a number; reject — date values must be string-shaped
        return None
    parsed = MetricFilterNormalizer.normalize_date(v, date.today())
    if parsed is None:
        # invalid date string → coerce to None so the gate re-fires
        return None
    return parsed
```
This ensures every stored date value is MM/YYYY-formatted or None. File:line: `src/normalizers/metric.py` (insert before the final `return v` in `_validate_value`).

### 3.3 `VALID_UNITS`
Current: `{"patients", "months", "days", "sites", "percent", "queries", None}` (`metric.py:26-28`).

**No change required.** For the new 12 fields:
- All 11 numeric fields → `unit=None`.
- `most_recent_approval_date` → `unit=None`.

The current server-derived `server_unit = None` in `FilterExtractor._validate` (`filter_extractor.py:308`) already enforces this. Keep as-is.

<!-- rev2: NEW SECTION per user Q3 decision -->
### 3.3b `MetricFilterOutput.value_end` — `between` operator support (NEW)

**Gap:** the LLM prompt example in `_build_metric_section` (`filter_extractor.py:195`) emits `value_end` as part of the JSON schema, but `MetricFilterOutput` has only `value` — no `value_end`. Pydantic silently drops the field at instantiation, so `between` queries (`"approved between 06/2024 and 12/2024"`, `"enrollment between 100 and 500"`) cannot be stored. The operator/value invariant validator (`_validate_value`) only checks `operator="any" ⇔ value=None`, not the between-pair contract.

**Fix required (user-approved, Q3):** add `value_end` to `MetricFilterOutput` and wire a cross-field validator.

1. **Add field** (in `MetricFilterOutput`, `metric.py` near line 93):
```python
value_end: Optional[Union[float, str]] = None
```

2. **Add validator** (Pydantic V2 `@model_validator(mode="after")` on `MetricFilterOutput`):
```python
@model_validator(mode="after")
def _validate_between_pair(self) -> "MetricFilterOutput":
    if self.operator == "between":
        if self.value is None or self.value_end is None:
            raise ValueError("operator='between' requires both value and value_end")
        # date/numeric type-symmetry: both must coerce same way
        if self.data_type == "numeric":
            if not isinstance(self.value, (int, float)) or not isinstance(self.value_end, (int, float)):
                raise ValueError("between with data_type=numeric requires numeric value AND value_end")
            if self.value > self.value_end:
                raise ValueError(f"between requires value ({self.value}) <= value_end ({self.value_end})")
        elif self.data_type == "date":
            # both must be MM/YYYY strings after _validate_value coercion
            if not isinstance(self.value, str) or not isinstance(self.value_end, str):
                raise ValueError("between with data_type=date requires MM/YYYY string value AND value_end")
            # parse-compare: convert MM/YYYY → (year, month) tuple for ordering
            v_yyyy, v_mm = int(self.value[3:]), int(self.value[:2])
            ve_yyyy, ve_mm = int(self.value_end[3:]), int(self.value_end[:2])
            if (v_yyyy, v_mm) > (ve_yyyy, ve_mm):
                raise ValueError(f"between requires value <= value_end (got {self.value} > {self.value_end})")
    else:
        # non-between: value_end MUST be None
        if self.value_end is not None:
            raise ValueError(f"value_end only allowed with operator='between' (got operator={self.operator!r})")
    return self
```

3. **Extend `_validate_value`** to also coerce `value_end` when present. Simplest: apply the same `data_type`-aware coercion via a small helper:
```python
@staticmethod
def _coerce_value_field(v: Any, data_type: str, operator: str) -> Any:
    # extracted from existing _validate_value body — shared by value and value_end
    ...
```
Call it from both `_validate_value` and a new `_validate_value_end` (`@field_validator("value_end", mode="before")`).

4. **LLM prompt** — keep `value_end` in the JSON example (now backed by real field). Document in the system prompt:
> `When operator is "between", emit both value (lower bound) and value_end (upper bound). For all other operators, value_end MUST be null.`

5. **Assembler / app rendering** — `app.py` metric-render block (`app.py:180-193`) must handle `between` display. Proposed format: `**{label}:** between {value} and {value_end}{unit}`. File:line: `app.py:182` (extend `op_symbol` branch).

6. **MetricMatch flow:** the resolver `MetricMatch` doesn't currently carry `value_end` because it only extracts trigger spans, not values. No change to `MetricMatch`. `value_end` lives only on `MetricFilterOutput` (post-LLM-parse).

7. **Test cases:** add `between`-pair tests for both data_types (Group 7 Pydantic validators in §4.1).

### 3.4 LLM prompt — date awareness
**Gap:** `_build_metric_section` (`filter_extractor.py:162-213`, line range approximate) does NOT communicate `data_type` to the LLM. For field #4 the LLM will produce `value="last 6 months"` or similar prose, which `normalize_value("last 6 months", "date", op)` returns None for.

**Fix required:** include `data_type` in the per-field LLM hint:
```python
fields_block = "\n".join(
    f"  - Field key: {m.canonical_field} ({m.canonical_label}) — data_type={m.data_type}"
    for m in safe_matches
)
```
Plus add two rules near the end of the metric section:
> `If data_type=date, value MUST be in MM/YYYY format (e.g. "01/2026"). Convert relative phrases like "since June 2025" to "06/2025". If you cannot determine an explicit month/year, emit value=null.`
> <!-- rev2 Q3 -->
> `When operator is "between", emit both value (lower bound) and value_end (upper bound) — same data_type. For all other operators, value_end MUST be null.`

File:line: `src/filter_extractor.py` `_build_metric_section` body. Dev should visually locate, not rely on the spec's line numbers.

### 3.5 `MetricAmbiguityGate` — date-option resolution at gate-fire time
**Gap:** the JSON-stored options for field #4 are relative-window strings (`"Last 6 months"`). When the user picks one, the canonical-merge APPEND path produces `"... Last 6 months"` and the LLM must convert that back to MM/YYYY in the next turn. That round-trip is fragile.

**Recommendation:** in `MetricAmbiguityGate.evaluate` (`sufficiency_gate.py:996-1081`), if `first.data_type == "date"`, compute absolute MM/YYYY options using `date.today()` and substitute them at fire time before building the `AmbiguousEntry`. Example:
```python
if first.data_type == "date":
    today = date.today()
    options = [
        f"Since {_months_ago(today, 6):%m/%Y}",
        f"Since {_months_ago(today, 12):%m/%Y}",
        f"Since {_months_ago(today, 24):%m/%Y}",
        f"Before {_months_ago(today, 24):%m/%Y}",
        "No preference",
    ]
```
Where `_months_ago` is a small helper added to `sufficiency_gate.py` or `metric.py`. The user picks an absolute MM/YYYY; the next-turn canonical contains a directly parseable date string; `_validate_value` (after §3.2 fix) coerces it to `MM/YYYY`. **File:line:** `src/sufficiency_gate.py:1054-1055`.

Alternative (less invasive): keep relative options in JSON, accept the LLM round-trip, and add a regression test for the `>>>"Last 6 months"` answer flow. **Recommendation: do the absolute substitution** — it removes one moving part.

### 3.6 `assembler.py` and `app.py` — date display
`app.py:180-193` renders `f"**{label}:** {op_symbol} {val_str}{unit_str}"`. For `mf.value = "01/2026"`, `str(mf.value)` is `"01/2026"`. Renders correctly with no change. The `op_symbol` map (`METRIC_OP_DISPLAY`) presumably contains `gte`/`lte`/etc. — verify it covers `"between"` for date ranges. Quick read of `app.py` would show this; flagging for QA verification.

For `assembler.py`: `metric_filters` is a plain list pass-through (`assembler.py:240`). No change.

### 3.7 `MetricMatch.data_type` propagation
`metric.py:288, 460, 517` already carry `data_type` from JSON entry → MetricMatch → MetricFilterOutput. The pipeline is data-type-agnostic post-parse. Just need §3.2 + §3.3b + §3.4 + §3.5 + §3.9 fixes.

### 3.8 Pipeline log fields
No change required. `LOG_PATH_METRIC_AMBIGUITY` already exists. Metric count fields in `_log_turn` are field-type-agnostic. `value_end` is NOT logged (same HIPAA tier as `value` — server-formatted but flagged for review since dates could theoretically carry user-derived content if the LLM mis-extracts).

<!-- rev2: NEW SECTION for Sec-SB1 mitigation -->
### 3.9 `normalize_date` year-range hardening (Sec-SB1)

**Gap:** `normalize_date` (`metric.py:160-183`, line range approximate) accepts any 4-digit year. A user query smuggling `"patient born 03/1985"` could land `03/1985` in `MetricFilterOutput.value` for field #4, which IS rendered in the UI and stored in the output schema.

**Fix required:** add a year-range check before returning the parsed MM/YYYY:
```python
MIN_VALID_YEAR = 2000  # IRB approvals predating 2000 are out of scope for current trials
def normalize_date(raw_value: str, current_date: date) -> Optional[str]:
    # ... existing parsing ...
    if mm < 1 or mm > 12:
        return None
    if yyyy < MIN_VALID_YEAR or yyyy > current_date.year + 1:
        return None  # rev2: DOB-leak mitigation — reject implausible years
    return f"{mm:02d}/{yyyy}"
```

`MIN_VALID_YEAR = 2000` is conservative; product can tighten later if real IRB approval dates from 2000–2005 are uncommon. The `+ 1` buffer on the upper bound allows future-dated approvals (rare but legal). File:line: `src/normalizers/metric.py` in `normalize_date`.

---

## 4. Test Plan

### 4.1 `tests/test_metric_filters.py` — full rewrite

Replace the 8 current groups (43 cases) with the following 9 groups (~58 cases). Module-scoped `resolver` fixture points at the new JSON; strict_validation=True.

**Group 1 — AC exact match (12 cases, one per field):**

| Test | Query | Expected field | Implied op |
|---|---|---|---|
| `test_field1_exact` | `"total studies with advarra over 1000"` | `total_studies_with_advarra` | `any` (numeric `over` parsed by LLM, not implied map) |
| `test_field2_exact` | `"matching studies count under 50"` | `studies_matching_search` | `any` |
| `test_field3_exact` | `"active trials more than 25"` | `active_trials` | `any` |
| `test_field4_exact` | `"most recent approval date since 2025"` | `most_recent_approval_date` | `gte` |
| `test_field5_exact` | `"query response time fast"` | `avg_days_respond_to_queries` | `lt` |
| `test_field6_exact` | `"submission to approval time slow"` | `avg_days_submission_to_approval` | `gt` |
| `test_field7_exact` | `"total protocol deviations all studies high"` | `total_protocol_deviations_all_studies` | `gt` |
| `test_field8_exact` | `"protocol deviations in matching studies few"` | `total_protocol_deviations_matching_studies` | `lt` |
| `test_field9_exact` | `"average enrollment matching studies large"` | `avg_enrollment_matching_studies` | `gt` |
| `test_field10_exact` | `"average enrollment matching ta low"` | `avg_enrollment_matching_ta` | `lt` |
| `test_field11_exact` | `"average screening rate high"` | `avg_screening_rate` | `gt` |
| `test_field12_exact` | `"days to fpe under 60"` | `avg_days_to_fpe` | `any` (LLM parses "under") |

**Group 2 — AC synonym match (12 cases, alternate synonym per field):**
- `test_field1_synonym`: `"number of advarra studies"` → field 1
- `test_field2_synonym`: `"studies matching my filters"` → field 2
- `test_field3_synonym`: `"ongoing trials live status"` → field 3
- `test_field4_synonym`: `"latest irb approval"` → field 4
- `test_field5_synonym`: `"query turnaround time"` → field 5
- `test_field6_synonym`: `"submission to irb approval"` → field 6
- `test_field7_synonym`: `"pds in all studies high"` → field 7 (tests acronym)
- `test_field8_synonym`: `"deviations in matching studies"` → field 8
- `test_field9_synonym`: `"average matched enrollment"` → field 9
- `test_field10_synonym`: `"avg ta enrollment"` → field 10
- `test_field11_synonym`: `"screening velocity"` → field 11
- `test_field12_synonym`: `"time to first patient enrolled"` → field 12

**Group 3 — Fuzzy fallback (6 cases):**
- Typo in `advarra` → `"advara studies count"` → field 1 (or no match — assert confidence ≤ 0.90 when found)
- Typo `enrolment` (British) → field 9 / 10 candidates
- Typo `screning rate` → field 11
- Typo `protocl deviations all studies` → field 7
- Single-token query `"fpe"` → no match (residual < 6 chars guard or n-gram floor)
- Long unknown query (300 tokens of `"unknown"`) → budget short-circuit, no crash

**Group 4 — Implied operator (10 cases):**

| Trigger word | Field | Expected op |
|---|---|---|
| `high` + screening rate | 11 | `gt` |
| `low` + screening rate | 11 | `lt` |
| `fast` + query response | 5 | `lt` |
| `since` + approval date | 4 | `gte` |
| `before` + approval date | 4 | `lt` |
| `recent` + approval | 4 | `gte` |
| `many` + active trials | 3 | `gt` |
| `zero` + protocol deviations all studies | 7 | `eq` |
| `large` + total advarra studies | 1 | `gt` |
| `small` + matching studies | 2 | `lt` |

**Group 5 — Overlap dedup (5 cases):** <!-- rev2: fixed field 9/10 case per QA-B1 -->
- `"total protocol deviations in all my studies"` → ONLY field 7, NOT field 8
- `"protocol deviations across my matching studies"` → ONLY field 8, NOT field 7
- `"average enrollment matching ta"` → ONLY field 10 (no ambiguity — field 9 anchor `matching studies` is not present in this query)
- `"average enrollment matching studies in matching ta"` → field 9 wins by AC longer-span (37 chars > 32 chars for field 10's anchor); document expected behavior
- `"submission to approval days, most recent approval date"` → BOTH field 6 and field 4 in the registered match list (different spans, no overlap)
- `"total studies with advarra matching my search"` → BOTH field 1 and field 2 (different anchors, no overlap)

**Group 6 — MetricAmbiguityGate firing + date E2E (8 cases):** <!-- rev2: added E2E cases per QA-B3 -->
- Single-field fire for each data_type: one numeric (`"active trials"`), one date (`"most recent approval date"`)
- Combined fire: `"active trials and matching studies count"` → 2 unresolved → combined template with second canonical_label appended
- Max-turns escape: session.is_max_turns_reached=True → returns None
- No-refire-after-any: pre-populate `filters.metric_fields["active_trials"]` with `operator="any", value=None` → gate returns None
- Date-field option substitution: single-field fire on field 4 → assert options[:-1] match `re.compile(r"^(Since|Before) \d{2}/\d{4}$")` pattern AND `options[-1] == "No preference"`
- **NEW Date E2E #1:** `"diabetes most recent approval date since 06/2025"` → run full `NLPPipeline.run_with_session` with mock LLM that returns `{"most_recent_approval_date": {"operator": "gte", "value": "06/2025", "value_end": null, "unit": null}}` → assert `NLPOutput.metric_filters[0].value == "06/2025"`, `operator == "gte"`. Verifies AC match → LLM prompt → normalize → MetricFilterOutput → assembler path.
- **NEW Date E2E #2:** `"diabetes most recent approval date between 01/2024 and 06/2025"` → mock LLM returns `{"most_recent_approval_date": {"operator": "between", "value": "01/2024", "value_end": "06/2025", "unit": null}}` → assert both `value` and `value_end` populated; renders correctly in `app.py`.
- **NEW Date E2E #3 (security):** mock LLM returns `value="03/1985"` (DOB-shaped) → `_validate_value` coerces to None via `normalize_date` year-range check → gate re-fires. Verifies §3.9 mitigation.

**Group 7 — Pydantic validators (12 cases):** <!-- rev2: added between + value_end cases per Q3 user decision -->
- valid numeric, invalid operator, invalid unit, unit=None valid, "No preference" → value=None, numeric coerce-from-string, operator=any+value!=None raises
- **NEW: valid date MM/YYYY** (`{operator: "gte", value: "06/2025"}`)
- **NEW: invalid date freeform → coerced to None** (`{operator: "gte", value: "June 2025"}`)
- **NEW: DOB-year rejection** (`{operator: "eq", value: "03/1985"}` → `value` coerced to None via §3.9 year-range check)
- **NEW: between numeric pair valid** (`{operator: "between", value: 100, value_end: 500}`)
- **NEW: between date pair valid** (`{operator: "between", value: "01/2024", value_end: "06/2025"}`)
- **NEW: between missing value_end raises** (`{operator: "between", value: 100, value_end: null}` → ValueError)
- **NEW: between value > value_end raises** (`{operator: "between", value: 500, value_end: 100}` → ValueError)
- **NEW: between mixed types raises** (`{operator: "between", value: 100, value_end: "06/2025"}` → ValueError)
- **NEW: non-between with value_end raises** (`{operator: "gt", value: 100, value_end: 500}` → ValueError)

**Group 8 — Normalizer static methods (10 cases — keep current group 7):**
- Add: `normalize_value("01/2026", "date", "gte")` → `"01/2026"`
- Add: `normalize_value("not a date", "date", "gte")` → `None`
- Add: `normalize_value("June 2025", "date", "gte")` → `None` (or document month-name parsing as future work)

**Group 9 — Startup validation (keep current group 9 unchanged):**
- All 11 strict-mode rejection tests + lenient-mode tests + empty-automaton test. JSON shape didn't change.

**Cross-cutting:**
- Update `test_health_check_returns_counts` expected: `hc["fields"] == 12` (unchanged) but now passes against the new JSON.
- Update `test_get_entry_known_field` to use `"active_trials"` (or any new key).
- <!-- rev2: Q9 -->Pin `rapidfuzz==3.10.0` in `requirements.txt` (already pinned per CONTEXT.md tech stack) — Group 3 fuzzy thresholds (75–80) are tuned to this version. If `rapidfuzz` is updated, re-tune thresholds. Add a `conftest.py` assertion `importlib.metadata.version("rapidfuzz") == "3.10.0"` to fail fast on accidental upgrades.

### 4.2 `tests/batch_test_cases.csv` — replace rows TC101–TC106

| id | category | input | expected_metric_field | expected_metric_operator | expected_type |
|---|---|---|---|---|---|
| TC101 | metric-exact | `"phase 3 diabetes active trials over 25"` | `active_trials` | `gt` | (search) |
| TC102 | metric-fuzzy | `"advara studies count cancer trials"` | `total_studies_with_advarra` | (blank — no value) | (search or clarification) |
| TC103 | metric-clarification | `"matching studies for diabetes"` | `studies_matching_search` | (blank) | clarification |
| TC104 | metric-multi-clarification | `"fast query response high screening rate diabetes phase 2"` | `avg_days_respond_to_queries` | (blank) | clarification |
| TC105 | metric-date | `"diabetes most recent approval date since 2025"` | `most_recent_approval_date` | `gte` | (search) |
| TC106 | metric-multi-turn | `"average enrollment matching ta diabetes>>>100–500"` | `avg_enrollment_matching_ta` | `between` (or `any`) | search |

Note: TC106 depends on whether the LLM successfully parses "100–500" as a between range. If empirically flaky, change to `>>>No preference` and expect operator=any.

---

## 5. HIPAA / Security Review Checklist

1. **Synonym PHI risk:** None of the 12 fields' synonyms contain patient-identifying terms (no names, no DOB tokens, no MRN-shaped strings). `total_studies_with_advarra` and similar contain the organization name "advarra" which is a vendor name, not PHI. **No new risk.**
2. **`MetricMatch.matched_text` / `MetricFilterOutput.original_text`** — these will now hold strings like `"most recent approval date since 2025"` for field 4. The existing rule (`CONTEXT.md` line 473, 491) that these must NEVER be logged remains correct. Verify the AppendOnly turn log in `_log_turn` only emits counts. Confirmed: `pipeline.py:282-297` logs counts only. No code change.
3. **`canonical_label` log safety:** server-defined in JSON; safe to log. No change.
4. **Date type and PHI:** `most_recent_approval_date` stores MM/YYYY of a study's IRB approval — that's regulatory metadata, not patient data. However, a misclassified query like `"patient born 03/1985"` could land a `03/1985` string into `MetricFilterOutput.value` if the LLM incorrectly extracts it. Mitigation: `_validate_value` (§3.2) only accepts properly-formatted MM/YYYY, but it doesn't reject `"03/1985"`. **Recommended belt-and-suspenders:** validate year is between, say, 2000 and `date.today().year + 1` in `normalize_date` to filter implausible date-of-birth-shaped values. File:line: `src/normalizers/metric.py:160-183`.
5. **Override-terms scan in `_derive_overrides` (`sufficiency_gate.py:612-675`)** — does NOT apply to metric fields. The metric registry has its own validation path (`_validate_entries` in `metric.py`). No new override surface introduced.
6. **Display sanitization:** `app.py:181, 187, 188` calls `_safe()` on `canonical_label`, `value`, and `unit`. `value` for date fields is server-formatted MM/YYYY (no user content). No new escape risk.

---

## 6. Migration Steps (Dev order) <!-- rev2: expanded for value_end + year-range -->

Execute as a single atomic commit (`metric-filters-v2`):

1. **Replace `data/metric_filters.json`** with the 12 new entries from §2.
2. **Modify `src/normalizers/metric.py`:**
   - Add `value_end: Optional[Union[float, str]] = None` to `MetricFilterOutput` (§3.3b).
   - Add `_validate_between_pair` `@model_validator(mode="after")` cross-field validator (§3.3b).
   - Extract `_coerce_value_field` helper and add `_validate_value_end` `@field_validator` (§3.3b).
   - Add `data_type == "date"` branch in `_validate_value` with `isinstance` + `any`-operator guards (§3.2).
   - Add year-range check to `normalize_date` (§3.9, Sec-SB1).
3. **Modify `src/filter_extractor.py:_build_metric_section`** — include `data_type=...` per field, MM/YYYY rule, between/value_end rule (§3.4).
4. **Modify `src/sufficiency_gate.py:MetricAmbiguityGate.evaluate`** — date-aware option substitution; add `_months_ago` helper (§3.5).
5. **Modify `app.py`** — extend metric-render block to handle `between` (`f"between {value} and {value_end}{unit}"`) per §3.3b step 5.
6. **Replace `tests/test_metric_filters.py`** with the rewrite per §4.1.
7. **Update `tests/batch_test_cases.csv`** rows TC101–TC106 per §4.2.
8. **Add `conftest.py` rapidfuzz version assertion** (§4.1 cross-cutting).
9. Run `pytest tests/test_metric_filters.py` — must pass 100%.
10. Run `python tests/batch_eval.py --limit 6 --strategy hybrid_cascade` filtered to the 6 metric rows — confirm pass.
11. Run `pytest tests/test_sufficiency_gate.py tests/test_conversation.py tests/test_ambiguity_coverage.py` — must remain green.
12. Smoke-test in Streamlit:
   - `"active trials in cancer trials"` → metric clarification fires
   - `"diabetes studies with most recent approval date since 06/2025"` → search path with date filter rendered
   - `"diabetes enrollment between 100 and 500"` → search path with between-range numeric filter

Steps 1–8 form one commit. CI gates on step 9+10.

---

## 7. Open Questions — RESOLVED <!-- rev2 -->

1. **Screening rate semantics:** ✅ **User confirmed: rate of candidates screened (velocity), NOT failure rate.** Field #11 keeps the velocity synonyms and per-week clarification options as drafted.
2. **Date clarification options:** ✅ **Both reviewers + spec recommend absolute MM/YYYY substitution at gate-fire (§3.5).** Implementing.
3. **"between" operator:** ✅ **User decision: ADD `value_end` to schema.** Full `between` support for both numerics and dates. See §3.3b for implementation. LLM prompt's `value_end` example is now backed by a real field. Cross-field validator enforces operator/value/value_end invariants.
4. **METRIC_STRICT_VALIDATION default:** ✅ Both reviewers recommend `true` for CI and tests (already in spec). Production deploys may opt into `false` via env var.
5. **"TA" word-boundary safety:** ✅ Security review confirmed `_is_word_bounded` handles `(TA)`, `TA;`, etc. correctly (non-alnum chars at boundaries). Adding unit test as belt-and-suspenders in Group 1.
6. **MetricAmbiguityGate 4-field cap:** ✅ Acceptable as-is. Logged as known UX limitation; revisit if user feedback surfaces it.
7. **Field #6 override-terms:** ✅ **REMOVED.** The metric registry has no override-terms mechanism — only the ambiguous-terms registry does. AC longer-wins between field-6 anchor (≥27 chars) and field-4 synonyms (≤24 chars) is sufficient. Group 5 regression test verifies.
8. **`metric_resolved_count` for date:** ✅ Verified. No special-casing needed (`value != None && operator != "any"` works for dates).
9. **rapidfuzz version-sensitive fuzzy tests:** ✅ Pin enforced via `conftest.py` version assertion; thresholds remain. See §4.1 cross-cutting.
10. **AC scan injection surface:** ✅ Security review walked all 77 preprocessor patterns — no bypass. Preprocessor fires before resolver; metric synonyms are vendor-neutral phrases with no overlap to blocked patterns.

---

## 8. Reviewer Findings Disposition <!-- rev2: NEW SECTION -->

| Finding | Source | Disposition |
|---|---|---|
| C1 — `value_end` inconsistency | QA-B2 + Sec-SB2 | **ADDRESSED §3.3b** — added `value_end` to schema, cross-field validator, LLM prompt rule, app rendering. |
| C2 — DOB-leak via `normalize_date` | QA-M4 + Sec-SB1 | **ADDRESSED §3.9** — year-range check (`MIN_VALID_YEAR=2000`, max=today.year+1). |
| C3 — char-count error #9/#10 | QA-B1/M5 | **ADDRESSED** — Design Notes corrected; Group 5 test case rewritten. |
| C4 — override_terms doesn't exist | QA-M7 | **ADDRESSED** — removed from field #6; rely on AC longer-wins. |
| C5 — date E2E test missing | QA-B3 | **ADDRESSED** — Group 6 expanded with 3 E2E cases (gte, between, year-range rejection). |
| C6 — `isinstance(v, str)` guard | QA-M1 | **ADDRESSED §3.2** — guard added. |
| C7 — `operator="any"` date guard | Sec-SM1 | **ADDRESSED §3.2** — short-circuit added. |
| QA-M2 — line number drift | QA | **NOTED** — spec instructs Dev to use visual search, not copy-paste line numbers. |
| QA-M6 — field 7/8 synonym dedup | QA | **NOTED** — startup validation in `_validate_entries` already raises on cross-field synonym duplicates (strict mode). QA spot-checked 7/8 synonyms — no verbatim duplicates. |
| QA-N1 — METRIC_OP_DISPLAY has between | QA | **VERIFIED** — `app.py:47` already maps `between → "between"`. No change. |
| Sec-SM2/SM3 — §3.4/§3.5 implementation pending | Sec | **EXPECTED** — Dev work, not spec defect. Migration steps cover. |
