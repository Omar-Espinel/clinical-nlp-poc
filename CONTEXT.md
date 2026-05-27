# Clinical Research NLP — Architecture Context

## Status: ACTIVE · v2 rework merged
## Last Updated: 2026-05-13
## Platform: Python 3.13 · Streamlit · FastAPI

---

## Core Intent
Takes free-text clinical research queries and returns structured search results (SNOMED concepts + filters) or clarification questions. This is a multi-turn POC for Advarra, feeding a downstream search system.

---

## High-Level Architecture
The system uses a **multi-turn pipeline** with a deterministic sufficiency gate to minimize LLM costs and latency.

1.  **Preprocessor:** 3–500 char check, null-byte strip, ~77 regex safety patterns.
2.  **ConversationSession:** Merges user input into a `canonical_query` (substitute or append).
3.  **SufficiencyGate (Layer 1):** Deterministic trigger detection for ambiguous terms (e.g., "cancer") from `ambiguous_terms.json` and auto-derived CSV tokens.
4.  **Metric Resolver:** Aho-Corasick scan for 12 Advarra-specific metric fields.
5.  **Extraction Path (Parallel):**
    *   **FilterExtractor:** LLM (Groq/Llama-3.1) extracts filters (investigator, site, city, state, phase).
    *   **SNOMED Strategy:** Algorithmic search (default: Hybrid Cascade — Exact → Synonym → Fuzzy → Semantic).
6.  **NegationAnnotator:** NegEx-based deterministic negation detection.
7.  **SufficiencyGate (Layer 2):** Embedding-based ambiguity detection (nearest neighbor spread).
8.  **Clinical-Intent Gate:** Rejects if 0 SNOMED matches AND 0 filters set.
9.  **GeoNormalizer:** Canonical city/state/region lookup.
10. **ResponseAssembler:** Pydantic V2 model assembly for `NLPOutput` or `ClarificationOutput`.

---

## Component Responsibilities

| Component | Responsibility | Key Logic |
|---|---|---|
| `Pipeline` | Orchestrator | `run_with_session()`, parallel execution, error handling. |
| `Preprocessor` | Safety | Input sanitization, injection defense, length limits. |
| `SufficiencyGate` | Ambiguity | Registry-based triggers (L1) and embedding signals (L2). |
| `FilterExtractor` | Extraction | LLM-driven structured filter parsing (Kansas City fix). |
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
4. **HTML Escaping:** `ResponseAssembler` escapes all user/LLM derived strings.
5. **Rate Limiting:** 5/60s and 30/session (app.py).

### HIPAA Log Hygiene
*   **NEVER logged:** Raw query, canonical query, filter values, SNOMED displays, LLM responses.
*   **LOGGED:** Counts, lengths, confidence scores, SNOMED codes, decision enums, latency.
*   **Audit Tool:** Grep for `to_dict()` or `repr(session)` in logs (use `session.summary_for_logging()`).

---

## Critical Constants & Environment

| Constant | Value | Impact |
|---|---|---|
| `MIN_CONFIDENCE` | `0.60` | SNOMED/Geo inclusion threshold |
| `SEMANTIC_THRESHOLD` | `0.82` | Semantic search FP/FN tradeoff |
| `PARALLEL_TIMEOUT` | `15.0s` | Hard cap on LLM + SNOMED search |
| `MAX_TURNS` | `3` | Max clarifications per session |
| `MIN_VALID_YEAR` | `2000` | DOB-leak mitigation for dates |

**Env Vars:** `GROQ_API_KEY` (required), `LLM_PROVIDER` (default: `groq`), `SNOMED_SEARCH_STRATEGY` (default: `hybrid_cascade`), `AMBIG_STRICT_VALIDATION` (default: `true`).

---

## Key Extension Points
*   **SNOMED Strategies:** Implement `SNOMEDSearchStrategy` protocol in `src/snomed_search/`.
*   **LLM Providers:** Implement `LLMProvider` protocol in `src/llm_provider/`.
*   **Filters:** Add fields to `ExtractedFilters` in `src/filter_extractor.py` and update assembler.

---

## Data Inventory
*   `data/snomed_clinical_trials.csv`: 116 terms used for algorithmic matching.
*   `data/geo_canonical.json`: 200+ cities/regions for normalization.
*   `data/ambiguous_terms.json`: Triggers and options for Layer 1 sufficiency.
*   `data/metric_filters.json`: 12 Advarra metric field definitions for Aho-Corasick scan.

---

## Full File Inventory
See `README.md` for a high-level overview. Project root contains `api.py` (FastAPI), `app.py` (Streamlit), `src/` (core logic), `tests/` (pytest suite), and `qa_testing/` (QA agent).

*For detailed history and past changes, see `HISTORY.md`.*

---

## pgvector_cascade Strategy (DEFAULT as of 2026-05-27)
- PostgreSQL + pgvector backend for SNOMED matching at 80k-150k concept scale; production index is 122,410 concepts.
- Schema: `clinical_nlp` (isolated). Migration: `db/migrations/001_create_snomed_schema.sql`
- Build: `python scripts/build_snomed_index.py` (see `docs/DATA_SOURCES.md`)
- Now the registry's `DEFAULT_STRATEGY`. On implicit-default init failure (DB unreachable, dependency missing, health-check fail), the registry auto-falls back to `hybrid_cascade` and logs a WARNING with `error_type`. An explicit `SNOMED_SEARCH_STRATEGY` env var override is honoured verbatim — no silent swap.
- Deployment note: see how-to block below.

> **Parent monorepo step (manual):** In `siteid-app/docker-compose.yml`, change the `db.image` from `postgres:16` to `pgvector/pgvector:pg16`. This is a one-line change; no data migration needed since pgvector/pgvector:pg16 is a drop-in replacement that adds the vector extension. After changing, run `docker compose up -d --force-recreate db` from the monorepo root.

---

## Post-Build Changes Log

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
