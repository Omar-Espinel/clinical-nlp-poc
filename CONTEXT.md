# Clinical Research NLP — Architecture Context

## Status: ACTIVE · v2 rework merged · LLM-free (fully deterministic runtime)
## Last Updated: 2026-06-01
## Platform: Python 3.13 · Streamlit · FastAPI

---

## Core Intent
Takes free-text clinical research queries and returns structured search results (SNOMED concepts + filters) or clarification questions. This is a multi-turn POC for Advarra, feeding a downstream search system.

---

## High-Level Architecture
The system uses a **multi-turn pipeline** that is fully deterministic at runtime (no LLM) with a sufficiency gate to minimize latency and external dependencies.

1.  **Preprocessor:** 3–500 char check, null-byte strip, ~77 regex safety patterns.
2.  **ConversationSession:** Merges user input into a `canonical_query` (substitute or append), then re-runs `assert_safe` on the merged query.
3.  **SufficiencyGate (Layer 1):** Deterministic trigger detection for ambiguous terms (e.g., "cancer") from `ambiguous_terms.json` and auto-derived CSV tokens.
4.  **Metric Resolver:** Aho-Corasick scan for 12 Advarra-specific metric fields.
5.  **Preflight Mandatory Check:** Rejects early if no medical-condition / investigator / site signal is present.
6.  **Extraction Path (Parallel):**
    *   **DeterministicFilterExtractor:** Rule-based extraction — `PhaseExtractor` + `NameExtractor` + `MetricFieldAssembler` (investigator, site, city, state, phase). No LLM.
    *   **SNOMED Strategy:** Algorithmic search (default: `pgvector_cascade`; auto-falls back to Hybrid Cascade — Exact → Synonym → Fuzzy → Semantic).
7.  **NegationAnnotator:** NegEx-based deterministic negation detection.
8.  **SufficiencyGate (Layer 2):** Embedding-based ambiguity detection (nearest-neighbor spread).
9.  **Clinical-Intent Gate:** Rejects if 0 SNOMED matches AND 0 filters set.
10. **Post-Extraction Safety + Metric Ambiguity Gates:** Final clarification checks against extracted results.
11. **GeoNormalizer:** Canonical city/state/region lookup.
12. **ResponseAssembler:** Re-runs `assert_safe`, then Pydantic V2 assembly of `NLPOutput` or `ClarificationOutput`.

---

## Component Responsibilities

| Component | Responsibility | Key Logic |
|---|---|---|
| `Pipeline` | Orchestrator | `run_with_session()`, parallel execution, error handling. |
| `Preprocessor` | Safety | Input sanitization, injection defense, length limits. |
| `SufficiencyGate` | Ambiguity | Registry-based triggers (L1) and embedding signals (L2). |
| `DeterministicFilterExtractor` | Extraction | Rule-based structured filter parsing (phase, names, metrics). No LLM. |
| `SNOMED Search` | Mapping | Pluggable strategies for term-to-code resolution. |
| `Negation` | Context | NegEx cues for excluding negated clinical terms. |
| `Normalizers` | Data | Geo (rapidfuzz) and Metric (Aho-Corasick) normalization. |
| `Conversation` | State | Turn history, canonical query merging, session serialization. |

---

## Primary Data Models (Contract)

### NLPOutput (Type: "search")
```python
{
  "type": "search",
  "snomed_terms": [{"code": str, "display": str, "match_type": str, "confidence": float, "negated": bool}],
  "filters": {
    "investigator_name": FilterField, "site_name": FilterField,
    "city": FilterField, "phase": FilterField,
    "state": {"values": list[str], "confidence": float, "is_region": bool}
  },
  "metric_filters": list[MetricFilterOutput],
  "metadata": {"processing_time_ms": int, "total_snomed_matches": int}
}
```

### ClarificationOutput (Type: "clarification")
```python
{
  "type": "clarification",
  "question": str,
  "options": list[str],
  "canonical_query": str,
  "turn_number": int,
  "max_turns": 3
}
```

---

## Security & HIPAA Mandates

### Per-Turn Security Stack
1. `Preprocessor.process()`: ~77 regex patterns (injection, harmful content).
2. `Preprocessor.assert_safe()`: Defense-in-depth on merged canonical query.
3. **Clinical-Intent Gate:** Prevents non-clinical queries from reaching full processing.
4. **HTML Escaping:** `ResponseAssembler` escapes all user-derived strings.
5. **Rate Limiting:** 5/60s and 30/session (app.py).

### HIPAA Log Hygiene
*   **NEVER logged:** Raw query, canonical query, filter values, SNOMED displays.
*   **LOGGED:** Counts, lengths, confidence scores, SNOMED codes, decision enums, latency.
*   **Audit Tool:** Grep for `to_dict()` or `repr(session)` in logs (use `session.summary_for_logging()`).

---

## Critical Constants & Environment

| Constant | Value | Impact |
|---|---|---|
| `MIN_CONFIDENCE` | `0.60` | SNOMED/Geo inclusion threshold |
| `SEMANTIC_THRESHOLD` | `0.82` | Semantic search FP/FN tradeoff |
| `PARALLEL_TIMEOUT` | `15.0s` | Hard cap on deterministic extraction + SNOMED search |
| `MAX_TURNS` | `3` | Max clarifications per session |
| `MIN_VALID_YEAR` | `2000` | DOB-leak mitigation for dates |

**Env Vars:** `SNOMED_SEARCH_STRATEGY` (default: `pgvector_cascade`), `DATABASE_URL` (required when pgvector_cascade is active), `API_KEY` (optional — API auth; auth skipped if unset), `BIOPORTAL_API_KEY` (optional — FHIR fallback), `AMBIG_STRICT_VALIDATION` (default: `false`), `METRIC_STRICT_VALIDATION` (default: `true`). No LLM/Groq key is used — the runtime is fully deterministic.

---

## Key Extension Points
*   **SNOMED Strategies:** Implement `SNOMEDSearchStrategy` protocol in `src/snomed_search/`.
*   **Filters:** Add fields to `ExtractedFilters` in `src/filter_extractor.py` and update assembler.

> **Note:** LLM integration was removed. There is no `src/llm_provider/` and no Groq/LLM runtime dependency — filter extraction is fully rule-based.

---

## Data Inventory
*   `data/snomed_clinical_trials.csv`: 121 terms used for algorithmic matching (116 original + 5 psychiatric/depression concepts added 2026-06-01).
*   `data/geo_canonical.json`: 200+ cities/regions for normalization.
*   `data/ambiguous_terms.json`: Triggers and options for Layer 1 sufficiency.
*   `data/metric_filters.json`: 12 Advarra metric field definitions for Aho-Corasick scan.

---

## Full File Inventory
See `README.md` for a high-level overview. Project root contains `api.py` (FastAPI), `app.py` (Streamlit), `src/` (core logic), `tests/` (pytest suite), and `qa_testing/` (QA agent).

*For detailed history and past changes, see `HISTORY.md`.*

---

## pgvector_cascade Strategy (DEFAULT as of 2026-05-27)
- PostgreSQL + pgvector backend for SNOMED matching at 80k–150k concept scale; the restore-shipped index (`restore_db.bat`) verifies 90,904 concepts. (The 122,410 figure in the 2026-05-27 log entry below refers to an earlier production index snapshot.)
- Schema: `clinical_nlp` (isolated). Migration: `db/migrations/001_create_snomed_schema.sql`
- Build: `python scripts/build_snomed_index.py` (see `docs/DATA_SOURCES.md`)
- Now the registry's `DEFAULT_STRATEGY`. On implicit-default init failure (DB unreachable, dependency missing, health-check fail), the registry auto-falls back to `hybrid_cascade` and logs a WARNING with `error_type`. An explicit `SNOMED_SEARCH_STRATEGY` env var override is honoured verbatim — no silent swap.
- Deployment note: see how-to block below.

> **Parent monorepo step (manual):** In `siteid-app/docker-compose.yml`, change the `db.image` from `postgres:16` to `pgvector/pgvector:pg16`. This is a one-line change; no data migration needed since pgvector/pgvector:pg16 is a drop-in replacement that adds the vector extension. After changing, run `docker compose up -d --force-recreate db` from the monorepo root.

---

## Post-Build Changes Log

### 2026-06-02 — Alias dictionary externalized + pgvector autocomplete gap fixed

**What:** (1) The 87-entry lay-term→SNOMED `ALIAS_DICTIONARY` (previously a hardcoded literal in `hybrid_cascade.py`) is now loaded from `data/clinical_aliases.json`, matching the `data/*.json` convention. (2) `PgVectorCascadeStrategy` — the runtime default — now builds the lightweight in-memory `_exact_index`/`_synonym_index`/`_alias_dict` from the CSV path it already receives and sets `semantic_available = True`.

**Why:** The autocomplete feature reads `_exact_index`/`_synonym_index`/`_alias_dict`/`semantic_available` off the active strategy. pgvector_cascade never built them, so on the **default** strategy all three autocomplete tiers were dead for clinical terms — Tier 1 (prefix/alias) and Tier 2 (fuzzy, which scans the same keys) returned nothing, and Tier 3 (semantic) was gated off by `semantic_available` defaulting to `False`. Only geo/phase suggestions survived. The 90k pgvector SQL search path is unaffected; these maps serve autocomplete only.

**Files touched:**
- `data/clinical_aliases.json` — new; 87 alias→target entries (verbatim from the old literal, no entries lost).
- `src/snomed_search/hybrid_cascade.py` — `ALIAS_DICTIONARY` literal replaced with `_load_alias_dictionary()` JSON loader; module-level `ALIAS_DICTIONARY` name preserved so the `snomed_resolver.py` re-export still works.
- `src/snomed_search/pgvector_cascade.py` — `_build_autocomplete_indices(csv_path)` added (stdlib `csv`, no pandas), called at end of `__init__`; `semantic_available = True` set. try/except keeps `__init__` from ever failing on a missing/empty CSV (degraded empty maps + WARNING).

**Critical constraints honoured:** pgvector search/health_check/get_top_neighbors logic unchanged; no pandas import added to pgvector; HIPAA log lines emit only counts and error-type names; degraded path is non-fatal.

**Test status after change:** Autocomplete suite 42 passed. Isolated checks: alias JSON loads identically across both modules (87 entries, re-export intact); pgvector imports without a DB; `_build_autocomplete_indices` yields exact=104/synonyms=540/aliases=86 on the live CSV and 0/0 (no raise) on a bad path. Full regression and live-DB pgvector verification not re-run (no Postgres in this environment).

---

### 2026-06-01 — Autocomplete endpoint (GET /v1/autocomplete)

**What:** Added a Google-style clinical query autocomplete feature. New `GET /v1/autocomplete?q=<prefix>&limit=<1-10>` endpoint returns up to 7 ranked suggestions across SNOMED terms, geo (cities/states/regions), and phases. Activates at ≥3 chars (server-enforced), typo-tolerant via rapidfuzz, lay-term aware (e.g. "heart attack" → "Myocardial Infarction"). Reuses the already-initialized SNOMED strategy's existing data structures — zero new ML models or indexes.

**Why:** Companion to the 2026-06-01 parent-SNOMED change: rather than only reacting to ambiguous terms with clarification, autocomplete proactively guides users toward specific terms as they type.

**Architecture:** Three tiers run in parallel via `asyncio.gather` + `run_in_executor`: Tier 1 prefix/exact (dict scan over `_exact_index`/`_synonym_index`/`_alias_dict` + geo + phases, <50ms), Tier 2 rapidfuzz WRatio with first-char prefilter (<300ms), Tier 3 semantic via `strategy.get_top_neighbors()` with numpy matmul fallback (<200ms). Results flow through dedup → score → hierarchy collapse → top-7. `QuerySegmenter` separates committed context tokens (Phase 3, geo) from the active prefix; `completion` is assembled downstream in `_format_suggestion`, keeping `AutocompleteSuggestion` a pure frozen dataclass.

**Files touched:**
- `src/autocomplete/` (new package) — `segmenter.py` (context vs. active-prefix split), `index.py` (`AutocompleteIndex` 3-tier search + `AutocompleteSuggestion` frozen dataclass), `ranker.py` (dedup, tier/category/specificity/context scoring, hierarchy collapse), `cache.py` (dict store, SHA-256 keys, FIFO eviction at 10k), `orchestrator.py` (`AutocompleteOrchestrator`, 50ms timing-oracle floor via `asyncio.sleep`, injection gate, parallel tier coordination).
- `src/pipeline.py` — added `snomed_strategy` property so the orchestrator reuses the initialized strategy (no second `get_strategy()` → no doubled startup/RAM).
- `api.py` — 7 changes: imports (incl. `Query` — required by the endpoint, omitted from the original brief); `autocomplete` global; lifespan init with graceful `None`-on-failure (→ 503); per-IP rate limiter (30 req / 10s rolling window, `_check_autocomplete_rate`); `AutocompleteSuggestionItem`/`AutocompleteResponse` models; `GET /v1/autocomplete` route (X-API-Key via existing middleware, no middleware change); docstring route table entry.
- `requirements.txt` — added only `pytest-asyncio>=0.24.0`.
- `tests/autocomplete/` (new) — `test_segmenter.py`, `test_index.py`, `test_ranker.py`, `test_orchestrator.py`, `test_autocomplete_endpoint.py` (42 tests).

**Critical constraints honoured:**
- HIPAA: new log lines emit only `prefix_length`, `tier_used`, `suggestion_count`, `latency_ms`, error type names, and a 16-bit IP hash — never query text, display strings, or SNOMED codes.
- No new ML models/indexes; existing strategy data structures referenced, not duplicated. No new runtime dependency (only the `pytest-asyncio` dev dep).
- `AutocompleteSuggestion` is frozen; `completion` assembled in `_format_suggestion` (no `object.__setattr__`/`replace`). `asyncio.get_running_loop()` (not deprecated `get_event_loop`). 50ms floor uses `asyncio.sleep`. Geo path uses `Path(__file__).parent`. Existing routes/middleware/models unchanged.

**Two deviations from the brief (both required for the brief's own success criteria):**
1. `ranker.py` hierarchy collapse compares **raw_score** proximity (window unchanged at 0.05), not the display-boosted ranking score. The brief's own `test_hierarchy_collapse_removes_parent` fails otherwise — a child's specificity boost widened the computed-score gap past 0.05, sparing the parent, which inverts the intent.
2. `test_503_when_not_initialized` sets `autocomplete = None` **after** `TestClient` startup. `lifespan` succeeds in this environment and re-initializes it, so the brief's pre-startup patch never exercised the 503 path. Endpoint code was already correct.

**Test status after change:** Autocomplete suite 42 passed / 0 failed. Full regression (excluding the live-DB `tests/test_pgvector_cascade.py` and the standalone `qa_testing/` harness) 368 passed / 0 failed / 0 skipped.

---

### 2026-06-01 — Parent SNOMED resolution + depression fix

**What:** Ambiguous terms that have a valid broad SNOMED parent concept now proceed as a search result instead of always forcing clarification. A new `TriggerResult` frozen dataclass replaces the bare tuple return of `find_trigger()`. New reason `"ok_parent_snomed_used"` added to `_KNOWN_REASONS`. The "depression" entry's options were corrected from wrong neurological conditions to clinically appropriate psychiatric options.

**Why:** "cancer trials in Boston" was forcing a clarification question even though it maps validly to SNOMED 363346000 (Malignant neoplastic disease). The fix allows broad terms to pass through to extraction while keeping clarification for terms with no useful parent (anatomical terms: brain, blood, skin, etc.). Autocomplete (a separate feature) will guide users toward specific terms proactively.

**Files touched:**
- `src/sufficiency_gate.py` — `TriggerResult` frozen dataclass added; `AmbiguousEntry` gains `snomed_parent_code: Optional[str]` and `allow_parent_search: bool`; `"ok_parent_snomed_used"` added to `_KNOWN_REASONS`; `find_trigger()` return type changed from `Optional[tuple]` to `Optional[TriggerResult]`; `SufficiencyGate.evaluate()` branches on `hit.use_parent_snomed`.
- `src/pipeline.py` — `LOG_PATH_PARENT_SNOMED_RESOLVED` constant; `run_with_session()` new `elif decision.reason == "ok_parent_snomed_used"` branch that constructs a `SNOMEDMatch` and calls `_run_extraction_path` with `pre_resolved_snomed`; `_run_extraction_path()` accepts `pre_resolved_snomed: Optional[list[SNOMEDMatch]]` and prepends it (dedup by code).
- `data/ambiguous_terms.json` — all entries gain `snomed_parent_code` and `allow_parent_search` fields; 30 clinical terms set to `allow_parent_search: true`; 14 anatomical terms set to `allow_parent_search: false`; depression options corrected to `["Major Depressive Disorder", "Bipolar Disorder", "Treatment Resistant Depression", "Postpartum Depression", "Persistent Depressive Disorder"]`.
- `data/snomed_clinical_trials.csv` — 5 psychiatric/depression SNOMED concepts added (35489007, 13746004, 58703003, 38451003, 310495003) so the new depression options resolve at ≥0.7 confidence during registry validation. CSV grows from 116 → 121 rows.
- `tests/test_parent_snomed_fix.py` — 10 new tests covering the parent-SNOMED path, TriggerResult flag propagation, SufficiencyDecision validation, anatomical-term fallback, and depression options correctness.
- `tests/test_sufficiency_gate.py` — 3 existing tests updated (test_case2, test_case5, test_case6) to assert the new correct behavior for cancer/heart triggers.

**Critical constraints honoured:**
- HIPAA: no new logging of query text, filter values, or SNOMED display strings; only reason enum, category, session_id, and SNOMED codes (public identifiers) logged.
- Backward compatible: new `AmbiguousEntry` fields are `Optional` with safe defaults; existing JSON entries without them continue to validate.
- `TriggerResult` is a frozen dataclass (not Pydantic) to avoid circular deps with `AmbiguousEntry`.
- `allow_parent_search: false` entries (brain, blood, skin, bone, etc.) still force clarification — anatomical terms have no useful broad SNOMED parent.

**Test status after change:** 328 passed, 0 failures, 14 skipped.

---

### 2026-05-28 — NameExtractor SNOMED-aware stop logic

**What:** NameExtractor now accepts the CSV-derived `snomed_known_terms` frozenset and uses it to truncate name candidates at SNOMED-term boundaries. Single-token stops (e.g. "hypertension", min length 5, no spaces) and 2–3-token stops (e.g. "myocardial infarction", "hodgkin lymphoma") are checked at extraction time so clinical terms can no longer be swallowed into a person or site name.

**Why:** Deterministic name extraction was greedy — phrases like "investigator johnson hypertension" or "trials by johnson myocardial infarction" were producing names like "Johnson Hypertension" / "Johnson Myocardial Infarction" because STEP 1/3 (signal-anchored capture) and STEP 5 (residual title-case sweep) had no notion of what was a clinical concept vs a proper noun. Pipeline already builds `_known_terms` from the SNOMED CSV for preflight; the fix threads that same frozenset into NameExtractor and consults it as a stop list. Eponymous terms (e.g. "hodgkin" alone, "parkinson" alone) are deliberately NOT in the single-token set — the CSV stores them as "hodgkin lymphoma" / "parkinson disease", so a person actually named Hodgkin or Parkinson is still extractable when followed by non-SNOMED context.

**Files touched:**
- `src/pipeline.py` — builds `_known_terms` from CSV before constructing `DeterministicFilterExtractor` and passes it as `snomed_known_terms=_known_terms`. (The frozenset is also still used by the existing preflight Signal-F logic further down — same source, two consumers.)
- `src/filter_extractor.py` — `DeterministicFilterExtractor.__init__` accepts `snomed_known_terms: frozenset[str] = frozenset()` and forwards it to `NameExtractor`.
- `src/extractors/names.py` — `NameExtractor.__init__` accepts `snomed_known_terms`, builds `self._snomed_single_tokens` (single-word terms, len ≥ 5) and `self._snomed_multi_tokens` (2–3 word tuples). New method `_is_snomed_token_sequence(tokens)` does exact-match-only lookup (no substring match). STEP 1 and STEP 3 truncate the title-case token list at the first SNOMED-stop boundary; STEP 5 skips candidates whose full sequence or leading 2–3 tokens are a SNOMED multi-token.
- `tests/test_name_extractor.py` — fixture updated to pass a minimal `snomed_known_terms` set; 6 new tests cover STEP 3 single/multi-token stops, STEP 5 multi-token skip, Parkinson/Hodgkin standalone (not stopped), and Hodgkin+Lymphoma (stopped).

**Critical constraints honoured:**
- Parameter is keyword-only with `frozenset()` default — no signature break for any caller that doesn't thread the set through.
- Exact-match-only semantics — substring matches do NOT trigger a stop, so "parkinson" alone never blocks extraction even though "parkinson disease" is a SNOMED term.
- No change to `_is_snomed_token_sequence` callable surface or NameExtractor's public output schema.
- No new comments beyond docstrings (project style: comments explain WHY, not WHAT).

**Test status after change:** 329 passed, 3 failed. The 3 failures are the pre-existing `tests/test_pgvector_cascade.py` cases noted in the 2026-05-27 entry (latency threshold + recall assertions against the live 122k DB) — unrelated to this work.

---

### 2026-05-27 — pgvector wiring + auto-fallback default (+ Layer 2 parity)

**What:** Made `pgvector_cascade` the production default with automatic fallback to `hybrid_cascade` on init failure. Added `get_top_neighbors` to `PgVectorCascadeStrategy` so the Layer 2 `EmbeddingAmbiguityGate` continues to function when pgvector is the active strategy (it duck-types on `hasattr(strategy, "get_top_neighbors")`).

**Why:** Original brief asked to wire the 122k pgvector index for runtime. Most of the spec was already implemented (registry registration, full cascade impl with parameterised SQL and HIPAA-clean logging, /v1/query route, X-API-Key middleware). Two real gaps: (1) pgvector lacked `get_top_neighbors` → silent Layer 2 regression when selected, (2) no auto-fallback if DB unreachable. Per-request strategy override was *not* added — gates and registries hold strategy refs at construction time and per-request switching would have required deeper architectural surgery the user opted out of.

**Files touched:**
- `src/snomed_search/pgvector_cascade.py` — added `get_top_neighbors`; rewrote health_check to (a) close the transaction opened by `SELECT 1` before returning the connection to the pool, (b) include `concept_count` from `SELECT COUNT(*)`, (c) use psycopg2 connection-level `commit()/rollback()` so SET LOCAL statement_timeout is scoped to a single tracked transaction. Added defensive `conn.rollback()` at the top of `search()` to clear any inherited pool state before `set_session`.
- `src/snomed_search/registry.py` — `DEFAULT_STRATEGY` → `pgvector_cascade`; `get_strategy()` now distinguishes explicit (name kwarg or env var set) from implicit (neither). Implicit-default failures fall back to `hybrid_cascade` with a logged WARNING. Implicit-default *unregistered* strategy (e.g. psycopg2 missing → import in try/except skipped) also falls back. Explicit overrides re-raise unchanged.
- `api.py` — top-of-file module docstring documenting all four routes (`/health/live`, `/health/ready`, `/v1/query`, `/v1/session/{id}`), auth model, and strategy-selection contract.
- `.env.example` — `DATABASE_URL` example aligned to localhost; optional `SNOMED_SEARCH_STRATEGY` override commented in.
- `tests/test_ambiguity_coverage.py` — pinned `_make_registry` to hybrid_cascade explicitly; the test exercises CSV-driven ambiguous-term registry validation and the 122k pgvector DB doesn't index the 99-term CSV options.
- `tests/test_snomed_strategies.py` — updated two `DEFAULT_STRATEGY == "hybrid_cascade"` assertions to the new default; one test now passes `name="hybrid_cascade"` explicitly to keep its assertion stable.

**Critical constraints honoured:**
- Protocol (`base.py`) UNCHANGED — still sync `search(query) -> list[SNOMEDMatch]`.
- `/v1/query` request/response schema UNCHANGED — Streamlit and existing API consumers see no break.
- `pipeline.py` UNCHANGED — strategy is still bound at construction via `SNOMED_SEARCH_STRATEGY` env var. No per-request override added.
- All SQL parameterised with `%s`; embedding vector bound, never interpolated.
- HIPAA log discipline preserved — no query text, no preferred_term, no concept_ids in INFO/WARN paths.
- 5s statement timeout in `get_top_neighbors` (1s in health_check count) scoped via `SET LOCAL` inside a tracked transaction; cannot leak to recycled pool connections.

**Test status after change:** 211 passed, 0 regressions from this work. Three pre-existing failures in `tests/test_pgvector_cascade.py` (`test_fuzzy_match_handles_typos`, `test_semantic_match_handles_paraphrases`, `test_semantic_search_latency_benchmark`) — these tests were skipped in earlier CI runs because `DATABASE_URL` was unset; they now run against the live 122k DB and reveal test-quality issues independent of this change (10ms latency threshold unrealistic on Windows; fuzzy/semantic recall assertions based on a small seed but running against the production index). These need triage in a separate pass.
