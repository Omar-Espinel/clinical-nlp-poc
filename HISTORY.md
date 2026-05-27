# Project History & Change Log

## Post-Build Changes Log

### REST API service — FastAPI migration (2026-05-13)

Added `api.py` as a FastAPI entry point alongside the existing Streamlit `app.py`.
The `src/` pipeline is completely unchanged. `api.py` exposes:

- `GET  /health/live`              — liveness probe (no auth required)
- `GET  /health/ready`             — readiness probe (no auth required)
- `POST /v1/query`                 — main query endpoint (X-API-Key auth)
- `DELETE /v1/session/{session_id}` — session cleanup (X-API-Key auth)

Session state is held in a TTL-based in-memory store (TTLSessionStore, 30min TTL,
1000 session cap with LRU eviction). Known limitation: sessions are lost on
container restart and not shared across workers. Production fix: replace with Redis.

Security additions over the Streamlit app.py:
- API key middleware (X-API-Key header, env-configured)
- CORS allowlist driven by ALLOWED_ORIGINS env var
- UUID4 session_id format validation before any store lookup
- PreprocessorError message allowlist (prevents internal wording leaks)
- Non-root Docker user (nlp group)
- Docker HEALTHCHECK on /health/live
- uvicorn --limit-concurrency flag
- asyncio.Lock on session store reads and writes
- Pipeline CPU work dispatched via run_in_executor (non-blocking event loop)
- Fast-fail lifespan: pipeline init failure raises immediately, killing the
  process so Docker/orchestrator restarts and ops gets alerted

load_dotenv() is called at the top of api.py so the .env file is loaded
before NLPPipeline() reads GROQ_API_KEY from os.environ. NLPPipeline() is
called with no arguments — it reads the key internally via get_provider().

New env vars: API_KEY, ALLOWED_ORIGINS (both optional with safe defaults).
New requirements: fastapi>=0.115.0, uvicorn[standard]>=0.32.0, python-multipart>=0.0.9.

### Metric filter recognition v2 — Advarra-specific field replace (2026-05-13)

Full replacement of all 12 metric fields with Advarra-confirmed metrics mapped to the search-results widget. None of the original 12 reused. Single commit replaces JSON, test file, batch CSV rows, and adds date-aware gate logic.

**New 12 fields:**
1. `total_studies_with_advarra`
2. `studies_matching_search`
3. `active_trials`
4. `most_recent_approval_date`
5. `avg_days_respond_to_queries`
6. `avg_days_submission_to_approval`
7. `total_protocol_deviations_all_studies`
8. `total_protocol_deviations_matching_studies`
9. `avg_enrollment_matching_studies`
10. `avg_enrollment_matching_ta`
11. `avg_screening_rate`
12. `avg_days_to_fpe`

**Schema changes:**
- `MetricFilterOutput.value_end` added for `between` operator support.
- `normalize_date` year-range check (`MIN_VALID_YEAR=2000`) mitigates DOB-leak risk.

**Gate changes:**
- `MetricAmbiguityGate.evaluate` date-aware option substitution using `_months_ago`.

**Spec:** `rework-metric-filters-v2.md`.

### Metric filter recognition v1 — AC automaton + fuzzy fallback (2026-05-12)

Adds a deterministic pre-extraction pass for clinical-trial operational metrics. Aho-Corasick automaton + rapidfuzz fallback.

### Ambiguity coverage v2 — auto-derived triggers + embedding gate (2026-05-11)

- Layer 1: Auto-derived triggers from SNOMED CSV.
- Layer 2: `EmbeddingAmbiguityGate` (pipeline step 5b).
- B6 fix: literal replacement in `pattern.sub`.
- Spec: `rework-ambiguity-coverage.md`.

### Clarification UI: buttons → text (2026-05-11)

Removed `st.button` widgets; user replies in `st.chat_input`. Fixed double-escaping bug.

### v2 Rework — Architecture and pipeline split (2026-05-08)

Full implementation of `rework-nlp-proposal.md` and `rework-nlp-impl-spec.md`.

---

## Legacy (pre-2026-05-08)

The pre-rework pipeline used `src/extractor.py` for combined LLM extraction and `src/snomed_resolver.py` for a 4-step cascade.

- **Initial build** — single-turn pipeline.
- **Security audit** — harmful-content patterns, HIPAA log hygiene.
- **Multi-state geo** — `StateFilterOutput(values, confidence, is_region)`.
- **Python 3.13 compat** — torch/numpy updates, numpy cosine fallback.
- **Security hardening** — 37+ patterns, `\ASYSTEM` anchor.
- **QA test agent** — `qa_testing/test_agent.py`.
