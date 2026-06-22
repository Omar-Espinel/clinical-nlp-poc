# Clinical Research NLP — Complete AI Context Document

## 1. Project Identity

| Attribute | Value |
|---|---|
| **Name** | Clinical Research NLP |
| **Purpose** | Free-text clinical research queries → structured SNOMED concepts + filters + metric fields |
| **Runtime** | Fully deterministic (no LLM since v2.1, June 2026) |
| **Stack** | Python 3.13, Streamlit 1.40.0, FastAPI, Pydantic V2, pgvector/PostgreSQL |
| **Status** | Active, v2.1 |
| **Organization** | Advarra (clinical research IRB/regulatory) |
| **Downstream** | Feeds a search system (siteid-app) |

---

## 2. Core Intent

> Takes free-text clinical research queries like "Phase 3 diabetes trials in Boston by Dr. Smith" and returns structured search results: SNOMED CT concepts, filters (investigator, site, city, state, phase), metric fields, and a passive `query_summary` label. If the query fails safety or preflight checks, it returns a clarification question instead.

**Key design decisions:**
- **No LLM** — the entire pipeline is deterministic rule-based (phase extraction via regex, name extraction via signals + stoplists, SNOMED search via algorithmic cascade)
- **Multi-turn** — up to 3 clarification turns per session; session state is serializable
- **HIPAA-compliant logging** — NEVER log raw query, canonical query, filter values, or SNOMED displays; only log counts, confidence scores, decision enums, codes, and latency
- **Additive `query_summary`** — passive interpretation label with confidence flags and unrecognized terms; does not gate results

---

## 3. Project Structure

```
clinical-nlp-poc/
├── api.py               # FastAPI REST API (5 routes)
├── app.py               # Streamlit chat UI
├── src/
│   ├── pipeline.py      # NLPPipeline orchestrator (716 lines)
│   ├── preprocessor.py  # ~77 regex safety patterns (178 lines)
│   ├── sufficiency_gate.py  # AmbiguousEntry, registry, gates (1321 lines)
│   ├── filter_extractor.py  # DeterministicFilterExtractor (242 lines)
│   ├── assembler.py     # Pydantic V2 output models + assembly (568 lines)
│   ├── conversation.py  # Multi-turn session state (302 lines)
│   ├── exceptions.py    # Custom exceptions
│   ├── extractors/
│   │   ├── names.py     # NameExtractor — investigator/site detection (984 lines)
│   │   ├── phase.py     # PhaseExtractor — regex-based phase detection (181 lines)
│   │   ├── preflight.py # PreflightMandatoryCheck — intent gate (360 lines)
│   │   ├── metric_assembler.py  # MetricFieldAssembler (159 lines)
│   │   └── negation_window.py   # Token window utilities for negation
│   ├── autocomplete/
│   │   ├── orchestrator.py  # AutocompleteOrchestrator (262 lines)
│   │   ├── index.py     # AutocompleteIndex — 3-tier search (330 lines)
│   │   ├── ranker.py    # Dedup + hierarchy collapse (125 lines)
│   │   ├── segmenter.py # QuerySegmenter — context vs active prefix
│   │   └── cache.py     # Dict cache with FIFO eviction
│   ├── normalizers/
│   │   ├── geo.py       # GeoNormalizer — city/state/region (166 lines)
│   │   ├── metric.py    # MetricIntentResolver — 12 Aho-Corasick fields (746 lines)
│   │   └── base.py      # MetricFilterOutput Pydantic model
│   └── snomed_search/
│       ├── base.py      # SNOMEDMatch + SNOMEDSearchStrategy Protocol (64 lines)
│       ├── hybrid_cascade.py  # 4-tier cascade — exact→synonym→fuzzy→semantic (671 lines)
│       ├── pgvector_cascade.py  # Default — PostgreSQL+pgvector (613 lines)
│       ├── registry.py  # Strategy registry + auto-fallback (127 lines)
│       ├── aho_corasick.py  # Alternative strategy
│       ├── ngram_lookup.py  # Alternative strategy
│       ├── negation.py  # NegEx deterministic negation (280 lines)
│       ├── fhir_fallback.py  # OLS4/BioPortal API fallback
│       └── __init__.py
├── data/
│   ├── snomed_clinical_trials.csv   # 121 SNOMED concepts with synonyms
│   ├── ambiguous_terms.json         # 44 ambiguous triggers + options
│   ├── metric_filters.json          # 12 Advarra metric fields
│   ├── geo_canonical.json           # 200+ cities/states/59 regions
│   ├── clinical_aliases.json        # 87 lay-term→SNOMED mappings
│   └── institution_keywords.json    # Hospital/site keywords
├── tests/                 # 372+ tests (pytest)
├── qa_testing/            # QA agent with 202 test cases
├── db/migrations/         # PostgreSQL schema
└── scripts/               # build_snomed_index.py
```

---

## 4. Pipeline Data Flow (10 Steps)

```
User Query
  │
  ▼ Step 1: Preprocessor (BLOCKING)
  │   3–500 char check, null-byte strip, ~77 regex safety patterns
  │   (injection, exfiltration, self-harm, chemical weapons, etc.)
  │   Returns PreprocessedInput or raises PreprocessorError
  │
  ▼ Step 2: ConversationSession
  │   compute_canonical_query() — substitute or append
  │   assert_safe() on merged canonical (defense-in-depth)
  │   set_canonical_query() — mutate session AFTER safety pass
  │
  ▼ Step 3: SufficiencyGate (NON-BLOCKING since v2.1)
  │   find_trigger() in ambiguous_terms.json + auto-derived CSV tokens
  │   If trigger found with allow_parent_search=True → ok_parent_snomed_used
  │     (injects parent SNOMED match, skips clarification)
  │   If trigger found without parent → ok_no_trigger (pass-through)
  │   If max_turns_reached → sufficient=True (escape valve)
  │   EmbeddingAmbiguityGate and MetricAmbiguityGate classes exist but UNUSED
  │
  ▼ Step 3b: MetricIntentResolver
  │   Aho-Corasick scan for 12 Advarra metric fields
  │   Context-gated: fuzzy/single-token matches need numeric/comparator/label ±10 tokens
  │   Multi-word AC synonyms exempt from context gate
  │
  ▼ Step 3c: PreflightMandatoryCheck (BLOCKING)
  │   7 signals: SNOMED term (A), person prefix (B), person suffix (C),
  │   institution keyword (D), context signal (E), token-based (F),
  │   capitalized non-function-word ≥4 chars (G / Signal D)
  │   Rejects if ALL signals absent (passes if ANY fires)
  │   Also rejects non-clinical topics (restaurant, weather, etc.)
  │
  ▼ Step 4: Parallel Extraction (ThreadPoolExecutor, 15s timeout)
  │   ┌─ DeterministicFilterExtractor.extract()
  │   │   ├── PhaseExtractor — regex patterns + fuzzy fallback
  │   │   ├── NameExtractor — signal-based (prefix/suffix/context/structured parse)
  │   │   │     Strips conversational openers ("can you", "find me", …)
  │   │   │     SNOMED-aware stop logic (v2.1 Fix 1)
  │   │   └── MetricFieldAssembler — from MetricMatch to MetricFilterOutput
  │   └─ SNOMED Search Strategy (default: pgvector_cascade)
  │        • pgvector_cascade: SQL cascade exact→synonym→fuzzy→semantic→cache→API fallback
  │        • hybrid_cascade fallback: in-memory exact→synonym→fuzzy→semantic
  │
  ▼ Step 5: NegationAnnotator
  │   NegEx-based: pre-window scan + post-window scan + conjunction propagation
  │   Sentence-boundary stops: . ; ! ? \n (comma is NOT a stop)
  │   5-token window, pseudo-negation suppression ("no contraindication for")
  │
  ▼ Step 6: Clinical-Intent Gate
  │   Requires ≥1 qualifying SNOMED match (confidence ≥0.60, not negated) OR ≥1 filter
  │   Otherwise raises PreprocessorError
  │
  ▼ Step 7: Post-Extraction Check (NON-BLOCKING after v2.1)
  │   Branch A: name ambiguity → carry flags, pass through
  │   Branch B: SNOMED-required metric unmet → clarify
  │   Branch C: missing mandatory term → clarify
  │   Unresolved metric fields → operator="any", value=None
  │
  ▼ Step 8: GeoNormalizer
  │   Canonical city/state/region lookup via rapidfuzz
  │   Region expansion (e.g., "Bay Area" → multiple states)
  │
  ▼ Step 9: assert_safe() belt-and-suspenders
  │
  ▼ Step 10: ResponseAssembler
  │   Pydantic V2 assembly of NLPOutput (type:"search")
  │   HTML-escapes all user-derived strings
  │   Builds optional query_summary (additive passive label)
  │   ClarificationOutput only for safety/preflight rejections
  │
  ▼ Output: NLPOutput (search) or ClarificationOutput
```

---

## 5. Data Models (Pydantic V2)

### NLPOutput (`type: "search"`)
```python
{
  "type": "search",
  "snomed_terms": [
    {"code": "44054006", "display": "type 2 diabetes", "match_type": "exact",
     "confidence": 0.99, "original_text": "type 2 diabetes", "negated": false}
  ],
  "filters": {
    "investigator_name": {"value": "Dr. Smith", "confidence": 0.95},
    "site_name": {"value": null, "confidence": 0.0},
    "city": {"value": "Boston", "confidence": 0.99},
    "state": {"values": ["Massachusetts"], "confidence": 0.99, "is_region": false},
    "phase": {"values": ["Phase 3"], "confidence": 0.95}
  },
  "metric_filters": [
    {"field": "active_trials", "canonical_label": "Active Trials",
     "operator": "gt", "data_type": "numeric", "value": 5.0,
     "value_end": null, "original_text": "active trials",
     "confidence": 0.95, "unit": null}
  ],
  "metadata": {"processing_time_ms": 843, "total_snomed_matches": 1,
               "snomed_match_types": {"exact": 1}, "negated_terms_excluded": 0},
  "query_summary": {
    "label": "Type 2 Diabetes Mellitus · Phase 3 · Boston · Massachusetts",
    "interpreted_terms": ["type 2 diabetes"],
    "interpreted_filters": {"phase": "Phase 3", "city": "Boston", "state": "Massachusetts"},
    "flags": [],
    "unrecognized_terms": [],
    "has_warnings": false,
    "flag_count": 0,
    "unrecognized_term_count": 0
  }
}
```

### ClarificationOutput (`type: "clarification"`)
```python
{
  "type": "clarification",
  "question": "Which type of cancer are you looking for?",
  "options": ["Lung Cancer", "Breast Cancer", "Colorectal Cancer"],
  "canonical_query": "cancer trials",
  "turn_number": 1,
  "max_turns": 3,
  "metadata": {"processing_time_ms": 12, ...}
}
```

### Key Internal Models
- **`SNOMEDMatch`**: frozen dataclass with `code`, `display`, `match_type`, `confidence`, `original_text`, `span`, `negated`
- **`MetricMatch`**: frozen dataclass with `canonical_field`, `canonical_label`, `matched_text`, `span`, `data_type`, `match_source`, `confidence`, `implied_operator`, `implied_op_map`
- **`ExtractedFilters`**: Pydantic with `investigator_name`, `site_name`, `city` (FilterField), `state` (StateFilter with `values`, `is_region`), `phase` (PhaseFilter with multi-value `values`), `metric_fields` (dict)
- **`SufficiencyDecision`**: Pydantic with `sufficient`, `reason` (validated against `_KNOWN_REASONS`), `triggered_by`, `matched_entry`, `ambiguous_name_flags`
- **`AmbiguousEntry`**: Pydantic with `trigger`, `category`, `question_template`, `options`, `override_terms`, `snomed_parent_code`, `allow_parent_search`
- **`Turn`**: Pydantic with `turn_index`, `user_input`, `canonical_query`, `decision`, `filters`, `snomed_matches`, `geo`, `timestamp`
- **`ConversationSession`**: Mutable Pydantic with `session_id`, `turns[]`, `canonical_query`, `max_clarification_turns`
- **`QuerySummary`**: Frozen Pydantic with `label`, `interpreted_terms`, `interpreted_filters`, `flags[]`, `unrecognized_terms[]`, `has_warnings`, `flag_count`, `unrecognized_term_count`

---

## 6. Key Constants

| Constant | File | Value | Impact |
|---|---|---|---|
| `MIN_CONFIDENCE` | `assembler.py` | 0.60 | SNOMED/Geo inclusion threshold |
| `SEMANTIC_THRESHOLD` | `hybrid_cascade.py` | 0.82 | Semantic search FP/FN tradeoff |
| `PARALLEL_TIMEOUT` | `pipeline.py` | 15.0s | Hard cap on parallel extraction |
| `MAX_TURNS` | `conversation.py` | 3 | Max clarifications per session |
| `MIN_VALID_YEAR` | `metric.py` | 2000 | DOB-leak mitigation |
| `LAYER2_LOW_THRESHOLD` | `sufficiency_gate.py` | 0.42 | Floor for embedding candidates |
| `FUZZY_CUTOFF_DEFAULT` | `hybrid_cascade.py` | 88 | rapidfuzz score cutoff |
| `MAX_NAME_LENGTH` | `names.py` | 100 | Name token cap |
| `INSTITUTION_FUZZY_THRESHOLD` | `names.py` | 82 | Site name fuzzy match |
| `MAX_FUZZY_COMPARISONS_PER_QUERY` | `metric.py` | 5000 | Metric fuzzy budget |
| `WINDOW_SIZE` | `negation.py` | 5 | NegEx pre/post token window |

---

## 7. Environment Variables

| Env Var | Default | Required | Purpose |
|---|---|---|---|
| `DATABASE_URL` | — | For pgvector | PostgreSQL connection string |
| `SNOMED_SEARCH_STRATEGY` | `pgvector_cascade` | No | Override search strategy |
| `API_KEY` | unset | Optional | API auth (skipped if unset) |
| `BIOPORTAL_API_KEY` | unset | Optional | FHIR fallback API key |
| `AMBIG_STRICT_VALIDATION` | `false` | No | Strict mode for registry |
| `METRIC_STRICT_VALIDATION` | `true` | No | Strict mode for metrics |
| `SKIP_AUTH` | `false` | No | Dev-mode auth bypass |
| `ALLOWED_ORIGINS` | localhost:3000 | No | CORS origins |
| `AUTOCOMPLETE_FUZZY_THRESHOLD` | 70 | No | Autocomplete fuzzy cutoff |
| `AUTOCOMPLETE_SEMANTIC_THRESHOLD` | 0.65 | No | Autocomplete semantic floor |
| `AMBIG_DERIVED_DEBUG` | `false` | No | Dev-only derived trigger debug |

---

## 8. Component Details

### 8a. Preprocessor (`src/preprocessor.py`)
- 77 injection/safety regex patterns organized into groups: prompt injection, code/SQL injection, data exfiltration, social engineering, LLM special tokens, SSTI, eval, HTTP header injection, XML injection, path traversal, harmful content/WMD, controlled substances, child safety, self-harm, cybercrime
- Null-byte strip (`\x00`)
- Length gate: 3–1000 chars
- `assert_safe()` — defense-in-depth on merged canonical queries (no length enforcement)
- Returns `PreprocessedInput(text, original, char_count)`
- Raises `PreprocessorError` with safe messages only

### 8b. ConversationSession (`src/conversation.py`)
- **`compute_canonical_query(user_input)`** — pure function (no mutation):
  - No prior turns → return user_input
  - Answering clarification (last decision non-sufficient with `triggered_by`) → substitute trigger word with user_input, or append if trigger not literally present
  - Free-text follow-up → append
- **`set_canonical_query(query)`** — mutation, called only after `assert_safe()` passes
- **`summarize_known_filters()`** — for clarification question template `{prior_filters}` substitution
- **`clarification_turn_count()`** — counts non-sufficient decisions
- **`is_max_turns_reached()`** — `>= 3`
- **`summary_for_logging()`** — LOG-SAFE fields only (session_id, counts)
- **`to_dict()` / `from_dict()`** — serialization (NEVER pass to logger)
- HIPAA: `to_dict()` and `repr(session)` must NEVER appear in log statements

### 8c. SufficiencyGate (`src/sufficiency_gate.py`)

**AmbiguousTermsRegistry:**
- Loads `data/ambiguous_terms.json` (44 entries: 30 clinical with `allow_parent_search: true`, 14 anatomical with `false`)
- Validates each option against SNOMED strategy (must resolve at ≥0.70 confidence)
- Derives `override_terms` from CSV scan for each trigger (prevents compound queries like "lung cancer" from firing bare "lung" trigger)
- **Auto-derived triggers** from CSV tokens: tokens appearing in ≥2 distinct preferred_terms (min length 4) become ambiguous entries. Tokens that ARE a full preferred_term ("leukemia") are excluded (Fix 3)
- Merges JSON + derived entries; JSON always wins on collision
- `find_trigger(query)` → `TriggerResult(trigger, entry, use_parent_snomed)` — first non-overridden trigger wins

**SufficiencyGate.evaluate():**
1. Max-turns escape → `sufficient=True`
2. Registry trigger hit → if `allow_parent_search` and `snomed_parent_code` → `ok_parent_snomed_used` (injects parent SNOMED match). Otherwise → `ok_no_trigger` (pass-through, no clarification)
3. Default → `ok_no_trigger`

**post_extraction_check():**
1. Branch A: name ambiguity → carry flags as `ambiguous_name_flags`, pass through
2. Branch B: SNOMED-required metric unmet without qualifying SNOMED → clarify
3. Branch C1/C1b/C2: qualifying SNOMED / investigator/site / filter present → pass through
4. Branch C3: nothing → clarify

**EmbeddingAmbiguityGate** and **MetricAmbiguityGate** classes are retained in the file but **NOT used by the pipeline** since v2.1.

### 8d. PreflightMandatoryCheck (`src/extractors/preflight.py`)
Seven signals checked independently; passes if ANY fires:
- **Signal A**: SNOMED known term present (from CSV, capped at 500 terms)
- **Signal B**: Person prefix (`dr`, `prof`, `pi`, etc.)
- **Signal C**: Person suffix (`md`, `phd`, `rn`, etc.)
- **Signal D**: Institution keyword (`hospital`, `clinic`, `university`, etc.)
- **Signal E**: Context signal (`by`, `at`, `investigator`, `site`, `led by`)
- **Signal F**: Token-based — any token NOT in `_SECONDARY_TOKENS` skiplist, NOT in signals A-E, NOT multi-word geo
- **Signal G (Signal D in spec)**: Capitalized token ≥4 chars, not a function word, not a geo stoplist term
- Also rejects non-clinical topics (restaurant, weather, hotel, etc.)

### 8e. PhaseExtractor (`src/extractors/phase.py`)
- Pre-compiled regex patterns for exact phase matching (Phase 1-4, 1a, 1b, 2a, 2b, 1/2, 2/3)
- Conjunction patterns for "Phase 2 or 3", "Phase 2 and 3", "Phase 2/3" → multiple values
- Alias patterns for "early phase", "pivotal", "registrational", "first in human", "dose escalation", etc.
- Fuzzy fallback via rapidfuzz `token_sort_ratio` (threshold 80)
- Typo correction: "fase"→"phase", "phaze"→"phase", etc.
- Returns `PhaseResult(value, confidence, span, values)` where `values` is a list for conjunctions

### 8f. NameExtractor (`src/extractors/names.py`)
**5-step rule-based extraction (no LLM):**
1. **STEP 0**: Structured parse — comma-separated "name investigator, name site" format
2. **STEP 0b**: Multiword institution scan — checks known sites (Mayo Clinic, Johns Hopkins, etc.)
3. **STEP 1**: Person prefix signals — `dr`, `prof` + following tokens (SNOMED-aware truncation)
4. **STEP 2**: Person suffix signals — `md`, `phd`, etc. (with institution context suppression)
5. **STEP 3**: Context person signals — `by`, `investigator`, `led by` + following title-case tokens
6. **STEP 4**: Context site signals — `at`, `site`, `facility` + following tokens
7. **STEP 5**: Residual title-case scan — catches remaining 2-token title-case sequences

**Key features:**
- Strips conversational openers before extraction (local to NameExtractor, canonical_query untouched)
- SNOMED-aware stop logic: truncates name candidates at SNOMED term boundaries (hypertension, myocardial infarction, etc.)
- 1-token ambiguous → flagged but not blocked (v2.1 pass-through)
- Confidence scoring per step (0.95 for prefix/suffix, 0.85 for context, 0.75 for residual)
- Negation-aware: negated spans are skipped

### 8g. MetricIntentResolver (`src/normalizers/metric.py`)
- Aho-Corasick automaton built from 12 Advarra metric field definitions (with multiple synonyms each)
- Overlap resolution: longer span wins
- Fuzzy fallback on residual spans via rapidfuzz `token_sort_ratio` (budget: 5000 comparisons/query)
- **Context gate (Fix 4)**: fuzzy/single-token AC matches are only valid when a numeric, comparator phrase, or canonical_label is within ±10 tokens (multi-word AC synonyms exempt)
- Operator inference from implied operator maps in JSON (e.g., "fast"→lt, "high"→gt, "between"→between)
- Date normalization to MM/YYYY with DOB-leak mitigation (rejects years < 2000 or > current+1)

**12 Metric Fields** (from `data/metric_filters.json`):
1. `total_studies_with_advarra` (numeric)
2. `studies_matching_search` (numeric, SNOMED-required)
3. `active_trials` (numeric)
4. `most_recent_approval_date` (date)
5. `avg_days_respond_to_queries` (numeric)
6. `avg_days_submission_to_approval` (numeric)
7. `total_protocol_deviations_all_studies` (numeric)
8. `total_protocol_deviations_matching_studies` (numeric)
9. `avg_enrollment_matching_studies` (numeric, SNOMED-required)
10. `avg_enrollment_matching_ta` (numeric)
11. `avg_screening_rate` (numeric)
12. `avg_days_to_fpe` (numeric)

### 8h. GeoNormalizer (`src/normalizers/geo.py`)
- Loads `data/geo_canonical.json` (200+ cities, 50 states, 59 regions)
- Exact lookup then rapidfuzz `token_sort_ratio` (threshold 82) fuzzy fallback
- Region expansion: "Bay Area" → ["California"], "New England" → ["Connecticut", "Maine", "Massachusetts", "New Hampshire", "Rhode Island", "Vermont"]
- Returns `GeoResult(city, states[], country, confidence, is_region)`

### 8i. SNOMED Search Strategies (`src/snomed_search/`)

**Protocol** (`base.py`):
- `SNOMEDMatch(code, display, match_type, confidence, original_text, span, negated)` — frozen dataclass
- `SNOMEDSearchStrategy` — Protocol with `name`, `__init__(dictionary_path)`, `search(query)`, `health_check()`

**HybridCascadeStrategy** (`hybrid_cascade.py`):
- 4-stage cascade: exact → synonym → fuzzy → semantic
- Builds in-memory `_exact_index` (preferred_term→record), `_synonym_index` (synonym→record), `_alias_dict` (lay-term→preferred_term from `clinical_aliases.json`)
- Stages 1+2: enumerate 1..4 token windows over query
- Stages 3+4: operate only on residual character spans NOT already matched
- Fuzzy: rapidfuzz `fuzz.ratio` per window (cutoff 88), per-anchor best-match
- Semantic: sentence-transformers + ChromaDB (fallback: numpy cosine similarity, threshold 0.82)
- Dedup: longest-match-wins (spans contained within larger spans are dropped)
- `get_top_neighbors(query, n, low_threshold)` — for EmbeddingAmbiguityGate; returns neighbors above threshold

**PgVectorCascadeStrategy** (`pgvector_cascade.py`):
- **Default strategy** (since 2026-05-27)
- PostgreSQL + pgvector: exact→synonym→fuzzy (pg_trgm)→semantic (pgvector cosine)→cache→API fallback (OLS4/BioPortal)
- Connection pool: 1–5 connections; parameterised SQL only
- `_build_autocomplete_indices(csv_path)` — builds lightweight in-memory maps for autocomplete (since 2026-06-02)
- `get_top_neighbors(query, n, low_threshold)` — SQL query with pgvector `<=>` operator
- Statement timeouts: 1s for health check count, 5s for get_top_neighbors

**Registry** (`registry.py`):
- `DEFAULT_STRATEGY = "pgvector_cascade"`
- Auto-falls back to `hybrid_cascade` on implicit-default init failure (logs WARNING)
- Explicit override (`SNOMED_SEARCH_STRATEGY` env var or `name=` kwarg) is honoured verbatim — no silent swap

### 8j. NegationAnnotator (`src/snomed_search/negation.py`)
- NegEx-style: pre-window (5 tokens) + post-window (5 tokens)
- Pre-cues: "no", "not", "without", "denies", "history of", "no evidence of", etc.
- Post-cues: "unlikely", "ruled out", "negative", "denied"
- Pseudo-negation suppression: "no contraindication for", "no change in", "not only"
- Sentence-boundary stops: `. ; ! ? \n` (comma intentionally NOT a stop)
- Intervening SNOMED match resets negation scope
- Pass 2: conjunction propagation — "not X or Y" → both X and Y flagged as negated

### 8k. Autocomplete (`src/autocomplete/`)
**Endpoint:** `GET /v1/autocomplete?q=<prefix>&limit=<1-10>`
**Architecture:** 3 tiers running in parallel via `asyncio.gather` + `run_in_executor`:
- **Tier 1** (prefix/exact): Dict scan over `_exact_index` + `_synonym_index` + `_alias_dict` + geo keys + phase list
- **Tier 2** (fuzzy): rapidfuzz WRatio with first-char prefilter (<300ms)
- **Tier 3** (semantic): sentence-transformers cosine similarity (<200ms)
- Results flow through: dedup (by SNOMED code, then by display) → score (tier × category × specificity × context bonuses) → hierarchy collapse (parent suppressed if child within 0.05 raw_score) → top-7
- **QuerySegmenter**: splits query into committed context tokens (Phase 3, geo, SNOMED) + active prefix
- **Cache**: SHA-256 keyed, FIFO eviction at 10k entries
- **Rate limit**: 30 req/10s per IP
- **50ms timing floor** to prevent timing oracle attacks
- HIPAA: only prefix_length, tier_used, suggestion_count, latency_ms logged

### 8l. ResponseAssembler (`src/assembler.py`)
- Filters SNOMED matches by MIN_CONFIDENCE (0.60), excludes negated, deduplicates by code
- Geo integration: if geo confidence ≥ 0.60, geo values replace raw extraction values
- HTML-escapes ALL user-derived strings (display, filter values, unrecognized terms)
- `assemble_query_summary()` builds passive interpretation label:
  - Fixed order: SNOMED terms · phase · investigator · site · city · state
  - Flags: low-confidence investigator (<0.75), low-confidence site (<0.75), low-confidence SNOMED (0.60–0.72, capped at 3), ambiguous name flags
  - Unrecognized terms: tokens not in known SNOMED set, geo keys, phase tokens, or function words; capped at 5×50 chars
  - HIPAA: NEVER log unrecognized_terms

---

## 9. HIPAA & Security Architecture

### Per-Turn Security Stack
1. Preprocessor ~77 regex patterns (blocking)
2. assert_safe() on merged canonical (defense-in-depth)
3. PreflightMandatoryCheck (blocking — rejects non-clinical queries)
4. HTML-escaping in ResponseAssembler (all user-derived strings)
5. Rate limiting: 5/60s burst + 30/session (app.py); unlimited for API

### HIPAA Log Hygiene
- **NEVER logged**: raw query, canonical query, filter values, SNOMED display strings, query_summary content, unrecognized_terms, matched_text, original_text, option strings, trigger values
- **Logged**: counts, lengths, confidence scores, SNOMED codes (public identifiers), decision enums, session_id, turn counts, latency, error type names, IP hashes
- **Audit tool**: grep for `to_dict()` or `repr(session)` in logs; use `session.summary_for_logging()` instead
- DOB-leak mitigation: MIN_VALID_YEAR = 2000 for date normalization
- IP hash truncated to 16 bits for autocomplete rate logs

---

## 10. Search Strategies Comparison

| Strategy | Hybrid Cascade | pgvector_cascade (DEFAULT) |
|---|---|---|
| **Scope** | 121 in-memory CSV terms | 90,904 DB concepts |
| **Storage** | Dict + ChromaDB/numpy | PostgreSQL + pgvector |
| **Cascade** | exact→synonym→fuzzy→semantic | exact→synonym→fuzzy→semantic→cache→API fallback |
| **Autocomplete maps** | Built in `__init__` | Built in `__init__` (since 2026-06-02) |
| **get_top_neighbors** | ChromaDB/numpy | SQL with pgvector `<=>` |
| **Fallback** | None (is the fallback) | Falls back to hybrid_cascade on init failure |
| **Health check** | Dict size, semi-available | DB ping + model check + concept count |
| **Dependencies** | sentence-transformers, ChromaDB, numpy | psycopg2, pgvector, sentence-transformers |
| **When used** | Auto-fallback, or explicit `hybrid_cascade` | Default, or explicit `pgvector_cascade` |

---

## 11. Clarification Flow (v2.1)

As of v2.1, `type: "clarification"` is returned ONLY for:
1. **Preprocessor safety rejection**: injection patterns detected → safe error message
2. **Preflight mandatory check failure**: no clinical signal found → "Your search needs at least one of: a medical condition, investigator name, or research site"
3. **Post-extraction Branch B**: SNOMED-required metric field without qualifying SNOMED
4. **Post-extraction Branch C3**: absolutely nothing extracted (no SNOMED, no filters)

Everything else passes through as `type: "search"` with optional `query_summary` flags for:
- Low-confidence investigator/site matches
- Low-confidence SNOMED matches
- Ambiguous names ("investigator_or_site" flag)
- Unrecognized terms

---

## 12. Response API Envelope

```json
{
  "result": NLPOutput | ClarificationOutput,
  "session_id": "abc123...",
  "processing_time_ms": 843,
  "api_version": "2.1"
}
```

**API Routes** (from `api.py`):
- `GET /health/live` — liveness probe (no auth)
- `GET /health/ready` — readiness probe (no auth, 503 until pipeline initialized)
- `POST /v1/query` — main NLP endpoint (auth required)
- `DELETE /v1/session/{session_id}` — clear session (auth required)
- `GET /v1/autocomplete` — clinical query autocomplete (auth required, rate limited)
- Auth: X-API-Key header middleware; skipped if API_KEY env var unset or SKIP_AUTH=true
- CORS: configurable ALLOWED_ORIGINS
- Session store: in-memory OrderedDict with 30-min TTL, max 1000 sessions

---

## 13. Streamlit UI (`app.py`)
- Chat interface with turn history
- Two-column display: SNOMED terms (left) + filters (right)
- Metric filters section below
- Query summary interpretation label with confidence flags
- "Edit this search" button that pre-fills canonical query for refinement
- Per-session rate limiting: 5/60s burst, 30 total
- "New Search" button in sidebar resets conversation
- JSON expander with `original_text` scrubbed for HIPAA safety

---

## 14. Test Suite

| Test File | Tests | Scope |
|---|---|---|
| `test_phase1_extraction_fixes.py` | 25 | Fixes 1-6: parent_search, opener stripping, derived entries, metric context, phase conjunctions, Signal D |
| `test_phase2_gate_removal.py` | 8 | All queries return `type:"search"` |
| `test_query_summary.py` | 12 | QuerySummary assembly |
| `test_sufficiency_gate.py` | 10 | SufficiencyGate cases |
| `test_parent_snomed_fix.py` | 10 | Parent SNOMED resolution |
| `test_metric_filters.py` | 58 | Metric field detection |
| `test_conversation.py` | 13 | Session state management |
| `test_ambiguity_coverage.py` | — | CSV-derived trigger coverage |
| `test_name_extractor.py` | — | Name extraction + SNOMED stops |
| `test_negation.py` | 9 | NegEx detection |
| `test_phase_extractor.py` | — | Phase extraction |
| `test_preflight.py` | — | Preflight checks |
| `test_snomed_strategies.py` | — | Strategy parity |
| `test_pgvector_cascade.py` | — | Live-DB tests (3 pre-existing failures on Windows) |
| `autocomplete/test_*.py` | 42 | Autocomplete pipeline |
| **Total** | **372+** | Excluding pgvector, autocomplete, qa_testing |

**Run:** `pytest tests/ --ignore=tests/test_pgvector_cascade.py --ignore=tests/autocomplete --ignore=qa_testing`

---

## 15. Data Files

| File | Format | Contents |
|---|---|---|
| `snomed_clinical_trials.csv` | CSV (121 rows) | `concept_id`, `preferred_term`, `synonyms` (pipe-delimited) |
| `ambiguous_terms.json` | JSON (44 keys) | `trigger`, `category`, `question_template`, `options[]`, `snomed_parent_code`, `allow_parent_search` |
| `metric_filters.json` | JSON (12 entries) | `canonical_label`, `data_type`, `synonyms[]`, `clarification_question`, `clarification_options[]`, `implied_operators{}`, `fuzzy_threshold` |
| `geo_canonical.json` | JSON | `cities{}`, `states{}`, `regions{}` with canonical names, state associations, region_states |
| `clinical_aliases.json` | JSON (87 entries) | lay-term → preferred_term mappings ("heart attack" → "myocardial infarction") |
| `institution_keywords.json` | JSON | `primary_keywords[]`, `legal_suffixes[]`, `known_multiword_sites[]` |

---

## 16. Database Schema (`db/migrations/001_create_snomed_schema.sql`)

```
Schema: clinical_nlp
Tables:
  - concepts (concept_id TEXT PK, preferred_term TEXT, source TEXT)
  - synonyms (id SERIAL PK, concept_id FK, synonym_text TEXT)
  - embeddings (id SERIAL PK, concept_id FK UNIQUE, embedding vector(384))
  - cache (term TEXT PK, concept_id TEXT, source TEXT, confidence REAL, cached_at TIMESTAMP, ttl_days INT)
  - metadata (key TEXT PK, value TEXT)
Indexes:
  - pgvector HNSW index on embeddings.embedding
  - trigram indexes on concepts.preferred_term, synonyms.synonym_text
  - B-tree on synonyms.concept_id
Functions:
  - cleanup_expired_cache() → void
```

---

## 17. Key Extension Points

1. **New SNOMED strategy**: Implement `SNOMEDSearchStrategy` protocol in `src/snomed_search/` — register in `registry.py`
2. **New filter field**: Add to `ExtractedFilters` in `filter_extractor.py`, update `NameExtractor` or create new extractor, update `ResponseAssembler.assemble()`, add to `FiltersOutput`
3. **New metric field**: Add entry to `data/metric_filters.json` with synonyms, operators, options
4. **New ambiguous term**: Add entry to `data/ambiguous_terms.json` with options, parent SNOMED code
5. **New SNOMED term**: Add row to `data/snomed_clinical_trials.csv`
6. **New geo location**: Add entry to `data/geo_canonical.json`

---

## 18. Build History (2026)

| Date | Change | Tests |
|---|---|---|
| 2026-05-27 | pgvector default + auto-fallback + get_top_neighbors | 211 passed |
| 2026-05-28 | NameExtractor SNOMED-aware stop logic | 329 passed, 3 failed |
| 2026-06-01 | Parent SNOMED resolution + depression fix | 328 passed |
| 2026-06-01 | Autocomplete endpoint (GET /v1/autocomplete) | 368 passed |
| 2026-06-02 | Alias dictionary externalized + pgvector autocomplete maps | 42 autocomplete tests |
| 2026-06-03 | **v2.1**: Gate removal + QuerySummary + extraction fixes | 372 passed |
