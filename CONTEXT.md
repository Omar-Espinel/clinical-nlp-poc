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
