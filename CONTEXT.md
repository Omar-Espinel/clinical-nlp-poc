# Clinical Research NLP — Project Context

## Status: ACTIVE / COMPLETE INITIAL BUILD · Rework spec drafted
## Last Updated: 2026-05-06
## Platform: Python 3.13 · Windows 11 · Streamlit

---

## What This Project Does

Takes a free-text natural language query from a clinical researcher and returns:
1. **SNOMED CT concepts** — code, display name, confidence score, match type, negation flag
2. **Structured search filters** — investigator name, site name, city, state(s), study phase

Example input:
> "Dr. Johnson's Phase 3 type 2 diabetes research in NYC, glucose management trials at Mount Sinai"

Example output (abbreviated):
```json
{
  "snomed_terms": [
    {"code": "44054006", "display": "type 2 diabetes mellitus", "confidence": 0.99, "match_type": "exact"},
    {"code": "182021000", "display": "glucose monitoring", "confidence": 0.97, "match_type": "synonym"}
  ],
  "filters": {
    "investigator_name": {"value": "Johnson", "confidence": 0.93},
    "site_name": {"value": "Mount Sinai", "confidence": 0.94},
    "city": {"value": "New York City", "confidence": 0.99},
    "state": {"values": ["New York"], "confidence": 0.99, "is_region": false},
    "phase": {"value": "Phase 3", "confidence": 0.99}
  }
}
```

This is a **proof-of-concept** for Advarra. It is not connected to any clinical trial database — it only parses and structures the query. The output is intended to feed a downstream search/filter system.

---

## Project Location

```
C:\Claude work\advarra\clinical-nlp-poc\
```

---

## Complete File Inventory

```
clinical-nlp-poc/
├── CONTEXT.md                    ← YOU ARE HERE — read this before touching anything
├── app.py                        ← Streamlit UI (290 lines)
├── requirements.txt              ← Pinned dependencies
├── README.md                     ← HuggingFace Spaces config + user docs
├── DEPLOYMENT.md                 ← Step-by-step local + HF deploy guide
├── .env.example                  ← Template: copy to .env and add GROQ_API_KEY
├── .env                          ← NEVER COMMIT — contains live API key
├── .gitignore                    ← Includes .env, __pycache__, chroma_db, etc.
├── src/
│   ├── __init__.py               ← Empty
│   ├── preprocessor.py           ← Input validation, injection/harmful content detection (~168 lines)
│   ├── extractor.py              ← Groq LLM call + JSON parsing (275 lines)
│   ├── snomed_resolver.py        ← 4-step SNOMED matching cascade (372 lines)
│   ├── geo_normalizer.py         ← City/state/region normalization (166 lines)
│   ├── assembler.py              ← Pydantic v2 output assembly (155 lines)
│   └── pipeline.py               ← Single orchestration entry point (115 lines)
├── data/
│   ├── snomed_clinical_trials.csv ← 116 SNOMED concepts with synonyms
│   └── geo_canonical.json         ← 200+ cities, all US states + CA provinces, 59 regions
├── tests/
│   ├── test_cases.json           ← 20 structured regression tests
│   ├── run_tests.py              ← Regression test runner (exit 0 if ≥15 pass)
│   ├── batch_test_cases.csv      ← 100 evaluation scenarios (edit to customize)
│   └── batch_eval.py             ← Batch runner: outputs metrics CSV + summary table (343 lines)
├── qa_testing/                    ← QA-driven test agent + large case sets (added 2026-05-06)
│   ├── test_agent.py             ← Rate-limited test runner with markdown report (497 lines)
│   ├── test_cases_202.json       ← 202-case curated test set (default input)
│   └── test_cases_1000.json      ← 1000-case extended test set
├── rework-nlp-proposal.md         ← Architectural proposal for v2 pipeline (added 2026-05-06)
└── rework-nlp-impl-spec.md        ← Pseudocode implementation spec for v2 (added 2026-05-06)
```

---

## How to Run

```bash
# Install dependencies (Python 3.13 required)
pip install -r requirements.txt

# Add API key
cp .env.example .env
# Edit .env: GROQ_API_KEY=your_key_here

# Start the app
streamlit run app.py
# → http://localhost:8501

# Run 20 regression tests
python tests/run_tests.py

# Run 100 batch evaluation cases
python tests/batch_eval.py
python tests/batch_eval.py --limit 10     # quick smoke test
python tests/batch_eval.py --input tests/batch_test_cases.csv --output tests/results/

# Run the QA test agent (202-case default, rate-limited for Groq free tier)
python qa_testing/test_agent.py
python qa_testing/test_agent.py --limit 20            # smoke test
python qa_testing/test_agent.py --category injection  # single category
python qa_testing/test_agent.py --input qa_testing/test_cases_1000.json
```

---

## Technology Stack

| Package | Version | Purpose |
|---|---|---|
| streamlit | 1.40.0 | Web UI |
| groq | 0.11.0 | LLM API client |
| sentence-transformers | 3.2.1 | Embedding model (all-MiniLM-L6-v2) |
| rapidfuzz | 3.10.0 | Fuzzy string matching |
| pydantic | 2.9.2 | Output data models (V2 syntax only) |
| pandas | 2.2.3 | CSV loading |
| torch | 2.6.0 | Required by sentence-transformers |
| numpy | 2.1.0 | Required by sentence-transformers + fallback semantic search |
| python-dotenv | 1.0.1 | .env loading |
| httpx | 0.27.2 | HTTP client |
| chromadb | *(optional)* | Vector DB — soft dependency, see below |

**IMPORTANT — chromadb is NOT in requirements.txt.** It was removed because `chroma-hnswlib` requires Microsoft C++ Build Tools to compile on Windows and has no Python 3.13 wheel. The code tries to `import chromadb` at runtime; if it fails, it falls back to pure-numpy cosine similarity (same quality for our ~116-term dataset). If a teammate has C++ Build Tools installed, they can `pip install chromadb==0.6.3` and it will be used automatically. No code changes needed.

**IMPORTANT — Python 3.13 compatibility.** The pinned versions of torch (2.6.0) and numpy (2.1.0) are specifically chosen for Python 3.13. The original spec had torch==2.2.2 and numpy==1.26.4 which only support up to Python 3.12.

**IMPORTANT — Pydantic V2 only.** All models use `model_config = ConfigDict(...)`, `model_dump()`, never `.dict()` or `class Config`. Do not revert to V1 syntax.

---

## Architecture: Full Pipeline Data Flow

```
User query (raw string)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│ src/preprocessor.py :: Preprocessor.process()        │
│  • Length check (3–500 chars)                        │
│  • ~77 regex patterns (injection, harmful content)   │
│  → PreprocessedInput(text, original, char_count)    │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│ src/extractor.py :: Extractor.extract()              │
│  • Calls Groq API: llama-3.1-8b-instant              │
│  • Temperature 0.0, max_tokens 1024, timeout 10s     │
│  • 3 retries with exponential backoff (1s, 2s, 4s)  │
│  • Parses JSON response into typed dataclasses       │
│  • Validates required keys before returning          │
│  → ExtractionResult(medical_terms, investigator,     │
│                     site, city, state, phase,        │
│                     raw_response)                    │
└───────────┬─────────────────────┬───────────────────┘
            │                     │
            ▼                     ▼
┌───────────────────┐   ┌─────────────────────────────┐
│ src/snomed_       │   │ src/geo_normalizer.py ::     │
│ resolver.py ::    │   │ GeoNormalizer.normalize()    │
│ SNOMEDResolver    │   │  • Exact city key lookup     │
│ .resolve()        │   │  • Exact region key lookup   │
│  4-step cascade:  │   │  • Fuzzy city match (≥82)    │
│  1. exact_match   │   │  • State abbreviation lookup │
│  2. synonym_match │   │  → GeoResult(                │
│  3. fuzzy (≥88)   │   │      city: Optional[str],    │
│  4. semantic      │   │      states: list[str],  ←KEY│
│  → list[SNOMED    │   │      is_region: bool,        │
│       Match]      │   │      confidence: float)      │
└──────────┬────────┘   └──────────────┬──────────────┘
           │                           │
           └─────────────┬─────────────┘
                         ▼
┌─────────────────────────────────────────────────────┐
│ src/pipeline.py :: NLPPipeline.run()                 │
│  • Clinical intent validation (SECURITY):            │
│    if 0 medical_terms AND 0 non-null filters →       │
│    raise PreprocessorError("No clinical content")   │
│  • Orchestrates all components above                 │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│ src/assembler.py :: ResponseAssembler.assemble()     │
│  • Excludes negated SNOMED terms from output         │
│  • Counts negated_terms_excluded in metadata         │
│  • Applies geo if confidence ≥ 0.60                  │
│  • Uses Pydantic V2 models (frozen=True)             │
│  → NLPOutput (see Output Schema below)              │
└─────────────────────────────────────────────────────┘
```

---

## Output Schema (current)

```python
NLPOutput
├── snomed_terms: list[SNOMEDTermOutput]
│   └── SNOMEDTermOutput
│       ├── code: str                 # SNOMED concept ID e.g. "44054006"
│       ├── display: str              # lowercase preferred_term from CSV
│       ├── match_type: str           # "exact" | "synonym" | "fuzzy" | "semantic"
│       ├── confidence: float         # 0.0–1.0
│       ├── original_text: str        # what the LLM extracted
│       └── negated: bool             # always False here (negated ones excluded)
│
├── filters: FiltersOutput
│   ├── investigator_name: FilterFieldOutput  → value: Optional[str], confidence: float
│   ├── site_name: FilterFieldOutput          → value: Optional[str], confidence: float
│   ├── city: FilterFieldOutput               → value: Optional[str], confidence: float
│   ├── state: StateFilterOutput              ← DIFFERENT from other filters
│   │   ├── values: list[str]    # 1 item for city/state queries, 15 items for "east coast"
│   │   ├── confidence: float
│   │   └── is_region: bool      # True when values has multiple states
│   └── phase: FilterFieldOutput              → value: Optional[str], confidence: float
│
└── metadata: MetadataOutput
    ├── processing_time_ms: int
    ├── total_snomed_matches: int
    ├── snomed_match_types: dict[str, int]   # {"exact": 1, "synonym": 2, ...}
    └── negated_terms_excluded: int
```

**CRITICAL: `state` uses `StateFilterOutput` with `values: list[str]`, NOT `FilterFieldOutput` with `value: Optional[str]`.** Any code that reads state must use `output.filters.state.values` (a list), never `.value`. This was changed after initial build to support multi-state regional queries.

---

## Component Reference

### src/preprocessor.py
**Class:** `Preprocessor`
**Key method:** `process(raw_input: str) -> PreprocessedInput`

Blocks:
- Queries shorter than 3 or longer than 500 characters
- ~77 regex patterns in 10 groups: prompt injection, code/script injection, data exfiltration, social engineering, LLM special tokens ([INST]/<<SYS>>), template/SSTI injection, HTTP header injection, XML tag injection, path traversal (Unix + Windows), SQL injection, WMD/weapon synthesis (including verb-noun attacks: make/build/create + weapon/bomb/firearm), controlled-substance manufacturing (including cook/make/grow + drug nouns), child safety violations, self-harm guides (including how to commit suicide, ways to end my/your life), poison-as-attack-verb, cybercrime (including write/create/develop + malware/exploit)
- Null bytes (`\x00`) stripped as the first step in `process()`, before length check and injection check
- SYSTEM prefix pattern (`\ASYSTEM`) compiled separately to anchor at absolute string start
- Raises `PreprocessorError` with user-safe messages (no internal details)

Error messages (exact strings, tests may depend on these):
- `"Query must be at least 3 characters"`
- `"Query must be under 500 characters"`
- `"Invalid query detected"`
- `"No clinical content found. Please enter a query about a medical condition, investigator, research site, location, or study phase."` ← raised in pipeline.py, not preprocessor

---

### src/extractor.py
**Class:** `Extractor`
**Key method:** `extract(preprocessed: PreprocessedInput) -> ExtractionResult`

- Groq model: `llama-3.1-8b-instant`, temperature 0.0
- System prompt is ~80 lines in the module, defines abbreviation expansion, phase normalization, negation detection, confidence scoring
- Catches `groq.RateLimitError` specifically; all other exceptions get 3 retries
- JSON parse failures raise `ExtractionError` with a generic user-safe message
- **Raw LLM response is NOT logged** (HIPAA) — only error type and response length

Abbreviations expanded by the LLM system prompt (not by code):
T2DM, T1DM, NSCLC, SCLC, HCC, CRC, RA, COPD, CKD, HF, CHF, HTN, MI, AFib, AF, AFL, PAD, DVT, PE, NHL, HL, AML, CML, ALL, CLL, MM, NASH, NAFLD, IBD, UC, CD, PSA, AD, PD, ALS, SLE, SSc, AS, PsA, GBM, MS

Phase normalization by LLM:
- "phase iii", "p3", "phase-3", "pivotal" → "Phase 3"
- "first in human", "fih" → "Phase 1"
- "phase 1/2", "p1/2", "phase i/ii" → "Phase 1/2"
- "phase 2b" → "Phase 2b"

---

### src/snomed_resolver.py
**Class:** `SNOMEDResolver`
**Key method:** `resolve(term: str, negated: bool) -> Optional[SNOMEDMatch]`

Resolution cascade (tries each in order, returns first match):
1. **Exact match** — lowercase term vs `exact_index` keys (confidence 0.99)
2. **Synonym/alias match** — validated alias dict first (0.97), then `synonym_index` (0.95)
3. **Fuzzy match** — rapidfuzz `token_sort_ratio` against all keys, score_cutoff=88 (confidence = score/100)
4. **Semantic match** — ChromaDB (if available) or numpy cosine similarity, threshold 0.82

**Alias dictionary** — ~90 entries mapping common terms to CSV preferred_terms. ALL alias targets must exist as `preferred_term` values in the CSV. Validated at load time; invalid aliases are logged as WARNING and skipped (no crash). The "ms", "ra", "tb" abbreviations are intentionally NOT in the alias dict because they're ambiguous — the LLM handles them with context.

**Semantic index** — Built at `__init__` time using sentence-transformers `all-MiniLM-L6-v2`. Two backends tried in order:
1. ChromaDB EphemeralClient (collection name: `"snomed_clinical_trials_v1"`) — faster for large datasets
2. Numpy cosine similarity fallback — works without C++ Build Tools, used when chromadb import fails

If both fail, `semantic_available = False` and resolution continues with exact/synonym/fuzzy only.

**Confidence threshold:** matches below 0.60 are discarded (not returned).

**Logging:** every resolution is logged at INFO level:
```
Resolved "type 2 diabetes" → 44054006 via exact match
Resolved "glucose management" → 182021000 via synonym (0.97)
No match found for "xyz" — skipping
```

---

### src/geo_normalizer.py
**Class:** `GeoNormalizer`
**Key method:** `normalize(city: Optional[str], state: Optional[str]) -> GeoResult`

**IMPORTANT CHANGE from initial design:** `GeoResult.state: Optional[str]` was changed to `GeoResult.states: list[str]` to support multi-state regional queries.

Lookup order for city input:
1. Exact key match in `cities` dict → confidence 1.0
2. Exact key match in `regions` dict → confidence 0.95, `is_region=True`
3. Fuzzy match against `cities` keys using token_sort_ratio, cutoff=82 → confidence = (score/100) × 0.90

Region behavior:
- **Single-state metro regions** (bay area, silicon valley, research triangle, etc.): `states = [state]`, `city = primary_city`
- **Multi-state geographic regions** (east coast, west coast, midwest, etc.): `states = [all covered states]`, `city = None`

State input: if provided, looked up via abbreviation or full name. For non-region results, user-supplied state overrides inferred state. For multi-state regions, user-supplied state does NOT override `region_states` (the region definition is more informative).

---

### src/assembler.py
**Class:** `ResponseAssembler`
**Key method:** `assemble(...) -> NLPOutput`

- Sorts `snomed_matches` by confidence descending before processing
- Excludes negated SNOMED matches from `snomed_terms` (counts them in `negated_terms_excluded`)
- Deduplicates `snomed_terms` by SNOMED code, keeping the highest-confidence match per code
- Applies geo values if `geo.confidence >= 0.60`, otherwise falls back to raw LLM-extracted values
- State fallback validates against `_INVALID_STATE_VALUES` frozenset before wrapping in list — prevents literal `"null"` string from leaking into `StateFilterOutput`
- All output models are Pydantic V2 with `frozen=True`

---

### src/pipeline.py
**Class:** `NLPPipeline`
**Key method:** `run(raw_query: str) -> NLPOutput`

Single entry point used by both `app.py` and `tests/run_tests.py`. Initializes all components once in `__init__`. The `run()` method:
1. Calls preprocessor
2. Calls extractor
3. **Clinical intent validation** (security layer): raises `PreprocessorError` if extraction yields 0 medical terms AND 0 non-null filters
4. Resolves SNOMED terms
5. Normalizes geo
6. Assembles output

**HIPAA log hygiene:** pipeline logs only counts and confidence scores, never raw query text or extracted field values.

---

### app.py
**Entry point:** `main()` via `if __name__ == "__main__"`

Key behaviors:
- `@st.cache_resource` on `load_pipeline()` — pipeline loads once, shared across all sessions
- Spinner on first load (30–60s for model download)
- Session-level rate limiting: max 5 queries per 60-second window, max 30 per session total (`_check_rate_limit()`)
- PHI disclaimers shown in sidebar and inline above query box
- State column renders as bulleted list when `is_region=True` and `len(values) > 1`
- All LLM-derived display values sanitized with `html.escape()` via `_safe()` before rendering
- Debug expander removed (was showing internal match breakdown — not appropriate for end users)
- Error handling: `PreprocessorError` → `st.warning`, `ExtractionError` → `st.error`, all others → generic `st.error` with no internal details

---

## Data Files

### data/snomed_clinical_trials.csv
**Columns:** `concept_id`, `preferred_term`, `synonyms` (pipe-separated)
**Rows:** 116 (including some duplicate preferred_terms intentional for different synonym sets)
**Coverage:** cancers (15+ types), cardiovascular, metabolic/endocrine, neurological, inflammatory/autoimmune, infectious diseases, clinical procedures, imaging, measurements

**Rule:** Every target value in the alias dictionary in `snomed_resolver.py` MUST appear as a `preferred_term` in this CSV. To add a new term:
1. Add row to CSV with unique `concept_id`, `preferred_term` (lowercase), `synonyms` (pipe-separated lowercase)
2. Optionally add alias entries in `snomed_resolver.py::ALIAS_DICTIONARY`
3. Rebuild the semantic index (happens automatically at next startup)

### data/geo_canonical.json
**Structure:**
```json
{
  "cities": {
    "lookup_key_lowercase": {"canonical": "Official Name", "state": "Full State Name", "country": "US"}
  },
  "states": {
    "ABBREVIATION": "Full State Name",
    "full name lowercase": "Full State Name"
  },
  "regions": {
    "region name lowercase": {
      "type": "region",
      "region_states": ["State1", "State2", ...],   ← for multi-state regions
      "country": "US"
    },
    "metro area name": {
      "type": "region",
      "primary_city": "City Name",                  ← for single-state metro regions
      "state": "State Name",
      "country": "US"
    }
  }
}
```

**Counts:** ~200 city entries (including all aliases), 50 US states + DC + Canadian provinces, 59 region entries (25 multi-state, 34 metro/single-state)

**Multi-state regions defined:** east coast (15 states), west coast (3), northeast (9), southeast (9), southwest (5), mid-atlantic (7), mountain west (8), deep south (5), great lakes (8), great plains (7), upper midwest (5), northwest (4), midwest (12), the south (14), new england (6), pacific northwest (3), tri-state/tri state/tristate (3: NY/NJ/CT)

**To add a city:** add a lowercase key entry to `cities`. Add all known aliases as separate keys pointing to the same canonical entry.

---

## Testing

### tests/run_tests.py (regression suite)
- 20 hardcoded test cases in `tests/test_cases.json`
- Covers: conditions, abbreviations, city aliases, phase variants, investigator names, site names, negation, typos, informal queries, multi-condition, Canadian cities, region resolution, all-fields-present
- State comparison: checks if expected state appears anywhere in `state.values` list (handles multi-state)
- Pass threshold: 15/20 (exit code 0), below 15 (exit code 1)
- Fuzzy match threshold for filters: 88

### tests/batch_eval.py (batch evaluation)
- Reads from `tests/batch_test_cases.csv` (100 cases, edit freely)
- CSV columns: `id`, `category`, `description`, `input`, `expected_snomed_codes` (pipe-sep), `expected_snomed_absent` (pipe-sep), `expected_city`, `expected_state`, `expected_phase`, `expected_investigator_name`, `expected_site_name`
- All `expected_*` columns are optional — blank means "don't check this field"
- Writes timestamped results CSV to `tests/results/`
- Prints terminal summary: overall pass rate, per-category breakdown, per-filter accuracy bars, SNOMED recall, avg/min/max processing time
- Use `--limit N` for quick smoke test

### qa_testing/test_agent.py (QA test agent)
- Reads JSON test cases (default `qa_testing/test_cases_202.json`; 1000-case set also available)
- Case schema: `id`, `input`, `category`, `description`, `expected.snomed_codes`, `expected.filters.{city,state,phase}`, `expected.should_reject`
- Categories handled: `valid`, `injection`, `harmful`, `edge_case`, `missing_condition`, `invalid_nonsense` — categories in `NO_API_CATEGORIES = {"injection", "harmful", "edge_case"}` are expected to be blocked by the preprocessor and never hit the Groq API
- **Rate-limit strategy** for Groq free tier (30 req/min, 500 req/day): no-API categories run first and instantly; API-needing cases throttled by token-bucket `RateLimiter` to `--rpm` (default 25); on 429 → exponential backoff up to 60s for `MAX_RETRIES=3` attempts, then case marked `RATE_LIMITED` (skipped, not failed)
- Filter comparison: `rapidfuzz.token_sort_ratio` ≥ 88; state checked via membership in `state.values` list
- SNOMED comparison: actual codes deduped before checking expected codes are all present (handles P4 duplicate-code semantics)
- Writes a `results.md` markdown report per run with summary metrics, failed cases, and rate-limited cases
- Flags: `--input`, `--limit`, `--category`, `--output`, `--rpm`

---

## Security & HIPAA Considerations

### What's implemented
1. **Two-layer input defense:**
   - Layer 1 (preprocessor): ~70 regex patterns block prompt injection, SQL injection, script injection, data exfiltration phrases, social engineering, LLM token injection, template/SSTI injection, path traversal, XML injection, HTTP header injection, WMD/weapon synthesis queries, controlled-substance manufacturing, child safety violations, self-harm guides, and cybercrime. Null bytes stripped before any check; SYSTEM prefix blocked at absolute string start.
   - Layer 2 (pipeline): clinical intent validation — rejects queries with 0 medical terms AND 0 filters after LLM extraction
2. **Log hygiene (HIPAA):** Query text is never logged. Extracted filter values are never logged. Only counts, lengths, confidence scores, and SNOMED codes logged.
3. **Error message sanitization:** All user-facing error messages are generic. Internal error details never reach the UI.
4. **Output sanitization:** All LLM-derived text rendered in UI goes through `html.escape()` before embedding in markdown.
5. **Session rate limiting:** Max 5 queries/60s, max 30/session (app.py).
6. **PHI disclaimers:** Shown in sidebar and inline — users instructed not to input patient names, MRNs, DOB, or other PHI.
7. **API key:** Only via `GROQ_API_KEY` env var or Streamlit secrets. Never hardcoded. Never logged.

### What this system does NOT do
- Store queries or results anywhere (all in-memory, session state only)
- Authenticate users
- Connect to any clinical trial database
- Process actual patient records

### Known security gaps (for future improvement)
- Rate limiting is session-state based — can be bypassed by opening new tabs. Production should use server-side rate limiting.
- No authentication/authorization layer
- No audit logging of queries (for compliance, may be needed in production)

---

## Known Limitations & Future Improvements

### Current limitations
1. **SNOMED coverage is curated, not complete** — 116 concepts covers common clinical trial use cases but not rare diseases. To expand: add rows to `snomed_clinical_trials.csv` and optionally update the alias dictionary.
2. **No user authentication** — anyone with the URL can use it
3. **Session-only rate limiting** — easily bypassed by new tab
4. **Groq free tier** — 30 req/min, 500/day. For production use, upgrade to paid tier.
5. **chromadb not available on Python 3.13 without C++ Build Tools** — falls back to numpy cosine similarity (functionally equivalent for our dataset size)
6. **LLM non-determinism** — temperature=0.0 minimizes but doesn't eliminate variation across retries
7. **No caching of query results** — identical queries hit the API every time

### Suggested next improvements
- Add more SNOMED concepts to the CSV (rare diseases, specific drug names, biomarkers)
- Add a feedback mechanism so users can flag incorrect extractions
- Cache common query results (Redis or simple dict with TTL)
- Add server-side rate limiting (e.g., via nginx or a middleware layer)
- Expand the LLM system prompt to handle more abbreviation edge cases
- Add support for date ranges ("trials from 2020–2023")
- Add support for patient population filters ("pediatric", "elderly", "> 65 years")
- Connect to ClinicalTrials.gov API to return actual matching trials

---

## Critical Constants (do not change without updating all usages)

| Constant | Value | Location | Impact if changed |
|---|---|---|---|
| ChromaDB collection name | `"snomed_clinical_trials_v1"` | snomed_resolver.py | Must match everywhere |
| Groq model | `"llama-3.1-8b-instant"` | extractor.py | Different behavior, cost |
| Embedding model | `"all-MiniLM-L6-v2"` | snomed_resolver.py | Vector space changes, re-index |
| SNOMED confidence threshold | `0.60` | assembler.py (MIN_CONFIDENCE), snomed_resolver.py | Changes what gets included |
| Semantic similarity threshold | `0.82` | snomed_resolver.py (SEMANTIC_THRESHOLD) | False positive/negative tradeoff |
| SNOMED fuzzy cutoff | `88` | snomed_resolver.py | False positive/negative tradeoff |
| Geo fuzzy cutoff | `82` | geo_normalizer.py | Handles typos like "Bostun" → Boston |
| Geo confidence threshold | `0.60` | assembler.py | Below this, falls back to raw LLM extraction |
| Batch eval fuzzy threshold | `88` | batch_eval.py, run_tests.py | Test pass/fail sensitivity |

---

## Component Interfaces (exact signatures)

```python
# preprocessor.py
Preprocessor.process(raw_input: str) -> PreprocessedInput
PreprocessedInput(text: str, original: str, char_count: int)

# extractor.py
Extractor.extract(preprocessed: PreprocessedInput) -> ExtractionResult
MedicalTerm(term: str, negated: bool, confidence: float)
FilterField(value: Optional[str], confidence: float)
ExtractionResult(medical_terms: list[MedicalTerm],
                 investigator_name: FilterField, site_name: FilterField,
                 city: FilterField, state: FilterField, phase: FilterField,
                 raw_response: str)

# snomed_resolver.py
SNOMEDResolver.resolve(term: str, negated: bool) -> Optional[SNOMEDMatch]
SNOMEDMatch(code: str, display: str, match_type: str, confidence: float,
            original_text: str, negated: bool)

# geo_normalizer.py
GeoNormalizer.normalize(city: Optional[str], state: Optional[str]) -> GeoResult
GeoResult(city: Optional[str], states: list[str], country: Optional[str],
          confidence: float, is_region: bool,
          original_city: Optional[str], original_state: Optional[str])

# assembler.py (Pydantic V2, all frozen=True)
ResponseAssembler.assemble(extraction: ExtractionResult,
                           snomed_matches: list[SNOMEDMatch],
                           geo: GeoResult,
                           start_time: float) -> NLPOutput

# pipeline.py — the only entry point app.py and tests should use
NLPPipeline(groq_api_key: str,
            snomed_csv_path: Optional[str] = None,
            geo_json_path: Optional[str] = None)
NLPPipeline.run(raw_query: str) -> NLPOutput
  # raises PreprocessorError or ExtractionError on failure
```

---

## How to Resume or Extend This Work

### If resuming after interruption
1. Read this entire CONTEXT.md
2. Check that all files listed in the inventory exist
3. Run `python tests/batch_eval.py --limit 5` to verify the pipeline is working
4. Proceed with your task

### If adding a new SNOMED concept
1. Add a row to `data/snomed_clinical_trials.csv`: `concept_id,preferred_term,synonyms`
2. If the concept needs a common-language alias, add to `ALIAS_DICTIONARY` in `src/snomed_resolver.py` — the target value must exactly match the `preferred_term` you just added
3. No other changes needed — the index rebuilds at startup

### If modifying the output schema
1. Update `GeoResult` in `geo_normalizer.py` if geo fields change
2. Update Pydantic models in `assembler.py`
3. Update `assemble()` method in `assembler.py`
4. Update `render_filters_column()` in `app.py`
5. Update `run_one()` and `print_summary()` in `tests/batch_eval.py`
6. Update `run_tests.py` if filter comparison logic changes
7. Use Pydantic V2 syntax throughout

### If changing the LLM
1. Update `MODEL` constant in `extractor.py`
2. Review the system prompt — some models need different formatting instructions
3. Update `README.md` and `DEPLOYMENT.md`
4. Re-run batch evaluation to check quality

### If deploying to HuggingFace Spaces
1. All files except `.env` go into the Space
2. Add `GROQ_API_KEY` as a Space Secret (Settings → Secrets)
3. `README.md` already has the required HF YAML front-matter at line 1
4. chromadb will NOT be available on HF free tier (no C++ Build Tools) — numpy fallback kicks in automatically
5. First cold start takes 2–5 minutes for package install + model download

---

## Dependency Tree (key relationships)

```
app.py
  └── src/pipeline.py (NLPPipeline)
        ├── src/preprocessor.py (Preprocessor)
        ├── src/extractor.py (Extractor) ← needs GROQ_API_KEY
        ├── src/snomed_resolver.py (SNOMEDResolver)
        │     ├── data/snomed_clinical_trials.csv
        │     ├── sentence-transformers (all-MiniLM-L6-v2)
        │     └── chromadb [optional] or numpy [fallback]
        ├── src/geo_normalizer.py (GeoNormalizer)
        │     └── data/geo_canonical.json
        └── src/assembler.py (ResponseAssembler)

tests/run_tests.py → src/pipeline.py (same path as app.py)
tests/batch_eval.py → src/pipeline.py (same path as app.py)
```

---

## Post-Build Changes Log

### Security audit (after initial build)
- `preprocessor.py`: Added 22 new patterns covering data exfiltration, social engineering
- `pipeline.py`: Added clinical intent validation after LLM extraction (primary defense against non-clinical queries)
- `pipeline.py`: Removed extracted field values from all log lines (HIPAA)
- `extractor.py`: Removed partial LLM response content from error logs
- `app.py`: Added PHI disclaimer in sidebar and inline; added session rate limiting; sanitized all LLM-derived display values with `html.escape()`; removed unused `last_raw_response` session state key; removed debug expander

### Multi-state geo support
- `geo_canonical.json`: Added 25 multi-state region entries with `region_states` arrays; updated existing regions (new england, midwest, the south, pacific northwest, tri-state) from single-state to multi-state
- `geo_normalizer.py`: Changed `GeoResult.state: Optional[str]` → `states: list[str]`; updated `normalize()` to return full state lists for regions
- `assembler.py`: Added `StateFilterOutput(values: list[str], confidence, is_region)` replacing `FilterFieldOutput` for the state field
- `app.py`: State column now renders as bulleted list for regional queries
- `tests/run_tests.py`: Updated state comparison to check membership in list

### Python 3.13 compatibility
- `requirements.txt`: `torch 2.2.2 → 2.6.0`, `numpy 1.26.4 → 2.1.0`, `chromadb 0.5.3 → removed`, `sentence-transformers 2.6.1 → 3.2.1`
- `snomed_resolver.py`: chromadb now a soft dependency with numpy cosine similarity fallback; two-stage `_build_chroma_index()` tries chromadb first, then numpy; `_semantic_match()` handles both backends

### Batch evaluation framework
- `tests/batch_eval.py`: Full batch runner with per-case results, SNOMED precision/recall, per-filter accuracy, per-category breakdown, timing stats; writes timestamped CSV to `tests/results/`
- `tests/batch_test_cases.csv`: 100 test cases covering all categories

### Security hardening & pipeline correctness (2026-05-05)
Driven by 1004-case batch evaluation identifying 86 non-SNOMED-coverage failures. Changes reviewed and approved by Architect → QA → Security before implementation.

**P1 — Harmful content blocking (`src/preprocessor.py`)**
- Added 22 new patterns to `_INJECTION_PATTERNS` covering: WMD/weapon synthesis (`nerve agent`, `sarin`, `bioweapon`, `dirty bomb`, `chemical weapon`, explosive synthesis, weapon synthesis), controlled-substance manufacturing (`methamphetamine`, `manufactur*`/`synthesiz*` + drug nouns), child safety (`child exploitation`, `human trafficking`, `child abuse material`), self-harm (`suicide method`, `self-harm guide`, `how to kill myself/yourself`), cybercrime (`ransomware`, `dark web drug`, `malware creat*`)
- Attack vector blocked: queries pairing harmful content with a valid clinical term via AND (e.g. "synthesize nerve agent AND heart failure phase 3") — the preprocessor now catches these before the LLM call

**P1 — Injection pattern hardening (`src/preprocessor.py`)**
- Added 15 new injection patterns: LLM special tokens (`[\s*/?INST\s*]` with whitespace tolerance, `<<SYS>>`, `<</SYS>>`), template/SSTI (`${...}`), code eval (`eval(`), HTTP header injection (`%0a`/`%0d` URL-encoded newlines), XML closing tags (`</tag\s*>`), self-closing tags, path traversal (`\.\.[\\/]` — catches both Unix `../` and Windows `..\`), SQL tautologies (`AND 1=1`, `OR 1=1`, `SLEEP(N)`, `UNION SELECT`)
- Added SYSTEM prefix pattern (`\ASYSTEM\s*[:\n]`) compiled separately with `re.IGNORECASE` using `\A` absolute-start anchor (not `^` with MULTILINE)
- Null bytes (`\x00`) stripped as the VERY FIRST step in `process()`, before `_validate_length` and before injection check; also stripped in `_sanitize()` as belt-and-suspenders
- Residual accepted risk (POC): leet speak substitution, unicode homoglyphs, heavily-spaced characters

**P2 — Null string state leak fix (`src/assembler.py`)**
- Added module-level `_INVALID_STATE_VALUES = frozenset({"null", "none", "n/a", "na", "unknown", ""})` before `ResponseAssembler`
- In `assemble()` geo fallback branch: replaced `state_values = [extraction.state.value] if extraction.state.value else []` with a guard that checks `raw_state.strip().lower() not in _INVALID_STATE_VALUES`, preventing the literal string `"null"` (returned by the LLM instead of JSON null) from appearing as `state: ['null']` in output

**P3 — Kansas City state disambiguation + site-name state extraction (`src/extractor.py`)**
- Root cause confirmed: `geo_canonical.json` correctly mapped "kansas city" → Missouri; the bug was the LLM extracting "Kansas" from the city name, then the geo_normalizer's "explicit state always wins" rule overriding the correct Missouri
- Fixed in `SYSTEM_PROMPT` rule 6: added IMPORTANT block instructing the LLM to extract state ONLY from explicit state mentions, not from city names ("Kansas City" ≠ Kansas, "Oklahoma City" ≠ Oklahoma, "New York" city ≠ New York state) and not from institution names ("Massachusetts General Hospital" does not imply state=Massachusetts when a different city is specified)

**P4 — Duplicate SNOMED codes (`src/assembler.py`)**
- `snomed_matches` now sorted by confidence descending BEFORE the main loop in `assemble()`, ensuring the highest-confidence match wins deduplication
- After the loop and before geo integration: deduplicate `included` by SNOMED code using a `seen_codes` set, keeping first (highest-confidence) occurrence per code
- `metadata.total_snomed_matches` correctly reflects unique codes after deduplication

### Verb-noun attack pattern coverage (2026-05-05)
Gap identified: all prior harmful content patterns were noun compounds (e.g. `weapon synthesis`, `chemical weapon`). Queries pairing a harmful *action verb* with a weapon/drug/person noun bypassed all patterns (e.g. "diabetes studies and how to make weapons").

**7 new patterns added to `_INJECTION_PATTERNS` in `src/preprocessor.py`** (total now ~77):
- `(make|build|create|construct|fashion) + (weapon|bomb|firearm|rifle|pistol|explosive|ied)` — closes verb-noun weapon gap
- `(cook|make|grow|produce|bake) + (meth|methamphetamine|heroin|cocaine|crack|fentanyl|lsd|ecstasy)` — closes drug production verb gap (`manufactur*`/`synthesiz*` were already covered)
- `how to commit suicide` — extends self-harm coverage beyond `suicide method` / `how to kill myself`
- `ways to (end|take) (my|your) life` — additional self-harm instructional phrase
- `how to (hurt|harm|injure) (myself|yourself)` — self-harm verb variant
- `poison + (person|someone|people|victim|target|individual)` — closes poison-as-attack-verb gap (existing pattern only covered `poison water/food supply`)
- `(write|create|build|develop|code) + (malware|ransomware|botnet|exploit)` — closes cybercrime verb gap; `virus` deliberately excluded to avoid blocking HIV/influenza/viral vector clinical terms

### QA test agent + extended case sets (2026-05-06)
New `qa_testing/` directory introduces a rate-limit-aware test runner separate from the existing `tests/` suite.

- `qa_testing/test_agent.py` — runs JSON test cases against `NLPPipeline`, throttles API calls to stay under Groq's 30 req/min free-tier limit, retries on 429 with exponential backoff up to 60s, marks cases `RATE_LIMITED` (skipped) rather than failing them after `MAX_RETRIES=3`. Runs no-API categories (`injection`, `harmful`, `edge_case`) first and instantly. Writes a markdown `results.md` per run.
- `qa_testing/test_cases_202.json` — 202 curated cases, default input
- `qa_testing/test_cases_1000.json` — 1000 cases for extended evaluation
- Case schema is JSON (not CSV like `batch_test_cases.csv`): each case has `expected.snomed_codes` (list), `expected.filters.{city,state,phase}`, and `expected.should_reject` (bool) for cases the preprocessor must block.

### Rework design docs (2026-05-06)
Two architectural specs added at the repo root for a planned v2 of the pipeline. Neither document changes the current implementation — they are forward-looking design only.

- `rework-nlp-proposal.md` — architectural proposal (revision 2, post adversarial QA + Security review). Key proposed changes: pre-extraction `SufficiencyGate` over the raw query (deterministic registry lookup) so insufficient queries cost zero LLM tokens; LLM extracts **filters only** (no SNOMED); SNOMED becomes a pluggable strategy behind a `SNOMEDSearchStrategy` Protocol (Aho-Corasick / n-gram / hybrid candidates); `LLMProvider` abstraction to remove Groq lock-in; NegEx-style algorithmic negation; parallel filter-extraction + SNOMED via `ThreadPoolExecutor`; multi-turn clarification capped at 3 turns; output becomes a discriminated union `NLPOutput | ClarificationOutput`. `state.values: list[str]` schema is preserved.
- `rework-nlp-impl-spec.md` — pseudocode-level implementation spec for the proposal (1949 lines). Defines `AmbiguousTermsRegistry`, `AmbiguousEntry`, `SufficiencyGate`, and the `data/ambiguous_terms.json` artifact. Pydantic V2, `frozen=True`, with strict-vs-graceful validation modes for rolling deploys.

**When implementing the rework**, treat `rework-nlp-proposal.md` as authoritative; `rework-nlp-impl-spec.md` deepens it with file paths, imports, constants, and class signatures. Look for `# OBSOLETE-AT-SCALE: <reason>` markers as the cleanup convention.

### v2 Rework — Architecture and pipeline split (2026-05-08)

Full implementation of the rework design described in `rework-nlp-proposal.md` and `rework-nlp-impl-spec.md`. The pipeline now supports multi-turn clarification and returns a discriminated union output type.

**Core architectural changes:**
- Pre-extraction sufficiency gate (`SufficiencyGate` + `AmbiguousTermsRegistry`): registry-driven deterministic check on the raw query before any LLM call; zero-token-cost for ambiguous queries that need clarification.
- LLM extraction is now **filters-only** (`FilterExtractor`): medical terms are no longer extracted by the LLM; SNOMED matching is fully algorithmic.
- Pluggable SNOMED strategies behind `SNOMEDSearchStrategy` Protocol: `hybrid_cascade` (default), `aho_corasick`, `ngram_lookup`. Swapped via `SNOMED_SEARCH_STRATEGY` env var.
- `LLMProvider` abstraction (`src/llm_provider/`): removes Groq lock-in; default provider is `groq`.
- NegEx algorithmic negation (`src/snomed_search/negation.py`): replaces LLM-based negation flag.
- `ConversationSession` multi-turn state (`src/conversation.py`): immutable `Turn` records, clarification count gate, HIPAA-safe `summary_for_logging()`.
- Parallel filter-extraction + SNOMED search via `ThreadPoolExecutor` with shared 15s timeout budget.
- Discriminated union output: `NLPOutput` (type="search") or `ClarificationOutput` (type="clarification").
- `app.py` rewritten as a chat UI: `st.chat_input` / `st.chat_message` per turn; clarification options rendered as clickable `st.button` widgets; "New Search" sidebar button resets `ConversationSession`.

**New files added:**
- `data/ambiguous_terms.json` — registry of ambiguous trigger terms with clarification question templates and SNOMED-validated option lists.
- `src/exceptions.py` — `LLMProviderError`, `PipelineError`, `StrategyError`.
- `src/sufficiency_gate.py` — `SufficiencyGate`, `AmbiguousTermsRegistry`, `AmbiguousEntry`, `SufficiencyDecision`.
- `src/conversation.py` — `ConversationSession`, `Turn`.
- `src/llm_provider/` — `base.py` (Protocol), `groq_provider.py`, `registry.py`, `__init__.py`.
- `src/snomed_search/` — `base.py` (Protocol + `SNOMEDMatch`), `hybrid_cascade.py`, `aho_corasick.py`, `ngram_lookup.py`, `negation.py`, `registry.py`, `__init__.py`.
- `src/normalizers/` — `base.py`, `geo.py` (moved from `src/geo_normalizer.py`), `__init__.py`.
- `src/filter_extractor.py` — `FilterExtractor`, `ExtractedFilters`, `FilterField`, `StateFilter`.

**OBSOLETE-AT-SCALE shims (kept for backward compatibility, marked for future removal):**
- `src/extractor.py` — original combined LLM extractor (medical terms + filters); still used by `tests/run_tests.py` imports.
- `src/snomed_resolver.py` — original 4-step cascade resolver; superseded by `src/snomed_search/hybrid_cascade.py`.
- `src/geo_normalizer.py` — original geo normalizer; superseded by `src/normalizers/geo.py`.
- `pipeline.run()` — single-turn shim wrapping `run_with_session()` with a fresh session; kept for `tests/run_tests.py`.

**New environment variables:**
- `LLM_PROVIDER` (default `groq`) — selects the LLM provider via `src/llm_provider/registry.py`.
- `SNOMED_SEARCH_STRATEGY` (default `hybrid_cascade`) — selects the SNOMED strategy via `src/snomed_search/registry.py`; also settable per `batch_eval.py --strategy` run.
- `AMBIG_STRICT_VALIDATION` (default `true`) — controls whether `AmbiguousTermsRegistry` raises on invalid option references at startup.

**Known limitation:** the depression trigger in `ambiguous_terms.json` maps to neuro-adjacent SNOMED options because psychiatric concepts (F32/F33 ICD equivalents) are not yet in `data/snomed_clinical_trials.csv`. Future improvement: expand the CSV with DSM-5 aligned concepts.
