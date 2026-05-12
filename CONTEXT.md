# Clinical Research NLP — Project Context

## Status: ACTIVE · v2 rework merged
## Last Updated: 2026-05-11
## Platform: Python 3.13 · Windows 11 · Streamlit

---

## What This Project Does

Takes a free-text natural language query from a clinical researcher and returns either:
1. **A search result** — SNOMED CT concepts + structured filters (investigator, site, city, state, phase), or
2. **A clarification question** — when the query contains an ambiguous trigger (e.g. bare "cancer") or filters but no clinical condition. Up to 3 clarification turns per session.

Example input:
> "Dr. Johnson's Phase 3 type 2 diabetes research in NYC, glucose management trials at Mount Sinai"

Example output (search path, abbreviated):
```json
{
  "type": "search",
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

Example output (clarification path):
```json
{
  "type": "clarification",
  "question": "Which type of cancer are you looking for?",
  "options": ["Lung Cancer", "Breast Cancer", "Colorectal Cancer", "Lymphoma", "Melanoma"],
  "canonical_query": "cancer in Boston phase 3",
  "turn_number": 1,
  "max_turns": 3
}
```

This is a **proof-of-concept** for Advarra. Output is intended to feed a downstream search/filter system. No real PHI or clinical-trial DB is touched.

---

## Project Location

```
C:\Claude work\advarra\clinical-nlp-poc\
```

---

## Complete File Inventory

```
clinical-nlp-poc/
├── context.md                       ← YOU ARE HERE — read before touching anything
├── app.py                           ← Streamlit chat UI (431 lines)
├── requirements.txt                 ← Pinned dependencies (incl. pyahocorasick)
├── README.md                        ← HF Spaces config + user docs
├── DEPLOYMENT.md                    ← Local + HF deploy guide
├── .env / .env.example              ← GROQ_API_KEY only; .env never committed
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── preprocessor.py              ← ~77 regex injection/harmful patterns + assert_safe (178 lines)
│   ├── pipeline.py                  ← NLPPipeline.run_with_session() — orchestrator (486 lines)
│   ├── conversation.py              ← ConversationSession, Turn (295 lines)
│   ├── sufficiency_gate.py          ← SufficiencyGate, AmbiguousTermsRegistry, AmbiguousEntry,
│   │                                  DEFAULT_CONDITION_PROMPT (569 lines)
│   ├── filter_extractor.py          ← Filter-only LLM extraction (197 lines)
│   ├── assembler.py                 ← NLPOutput | ClarificationOutput (238 lines)
│   ├── exceptions.py                ← LLMProviderError, PipelineError, StrategyError,
│   │                                  ExtractionError (42 lines)
│   │
│   ├── llm_provider/                ← LLM provider abstraction
│   │   ├── base.py                  ← LLMProvider Protocol
│   │   ├── groq_provider.py         ← GroqProvider (only file importing `groq`)
│   │   └── registry.py              ← get_provider() factory, env LLM_PROVIDER
│   │
│   ├── snomed_search/               ← Pluggable SNOMED strategies
│   │   ├── base.py                  ← SNOMEDSearchStrategy Protocol + SNOMEDMatch dataclass
│   │   ├── hybrid_cascade.py        ← exact → synonym → fuzzy → semantic (default, 593 lines)
│   │   ├── aho_corasick.py          ← AhoCorasickStrategy (136 lines)
│   │   ├── ngram_lookup.py          ← NGramLookupStrategy (128 lines)
│   │   ├── negation.py              ← NegEx NegationAnnotator (231 lines)
│   │   └── registry.py              ← get_strategy() factory, env SNOMED_SEARCH_STRATEGY
│   │
│   ├── normalizers/
│   │   ├── base.py                  ← FilterNormalizer Protocol
│   │   └── geo.py                   ← GeoNormalizer + GeoResult (166 lines)
│   │
│   ├── extractor.py                 ← OBSOLETE-AT-SCALE shim (legacy combined extractor)
│   ├── snomed_resolver.py           ← OBSOLETE-AT-SCALE shim (legacy 4-step cascade)
│   └── geo_normalizer.py            ← OBSOLETE-AT-SCALE re-export shim
│
├── data/
│   ├── snomed_clinical_trials.csv   ← 116 SNOMED concepts with synonyms
│   ├── geo_canonical.json           ← 200+ cities, US states + CA provinces, 59 regions
│   └── ambiguous_terms.json         ← Triggers → clarification options (validated at startup)
│
├── tests/
│   ├── run_tests.py                 ← OBSOLETE-AT-SCALE: 20-case regression suite
│   ├── test_cases.json              ← 20 regression cases
│   ├── batch_eval.py                ← Batch runner: --strategy, --legacy, --limit (520 lines)
│   ├── batch_test_cases.csv         ← 100 evaluation scenarios
│   ├── test_sufficiency_gate.py     ← Gate unit tests (506 lines)
│   ├── test_conversation.py         ← Multi-turn regression (614 lines)
│   ├── test_snomed_strategies.py    ← Cross-strategy parity (386 lines)
│   ├── test_negation.py             ← NegEx tests (169 lines)
│   ├── test_llm_provider.py         ← Provider abstraction tests with mock (163 lines)
│   └── test_ambiguity_coverage.py   ← Layer 1 + Layer 2 + B6 fix tests (434 lines)
│
├── qa_testing/                       ← QA-driven test agent + large case sets
│   ├── test_agent.py                ← Rate-limited runner with markdown report (497 lines)
│   ├── test_cases_202.json          ← 202-case curated set (default)
│   └── test_cases_1000.json         ← 1000-case extended set
│
├── rework-nlp-proposal.md           ← Architectural proposal (revision 2) for v2 pipeline
├── rework-nlp-impl-spec.md          ← Pseudocode implementation spec for v2
├── modified-proposal.md             ← Predecessor proposal (historical)
├── response-branch-spec.md          ← Branch-specific spec (historical)
├── README response.md               ← Branch-specific README (historical)
├── test_results.md                  ← Latest QA agent results dump
├── results-review.md                ← Review notes on test results
├── gen_block_diagram.py             ← Helper: generates block-diagram.png
├── gen_flow_diagram.py              ← Helper: generates flow-diagram.png
├── block-diagram.png / flow-diagram.png  ← Architecture diagrams
```

---

## How to Run

```bash
# Install (Python 3.13 required)
pip install -r requirements.txt

# Add API key
cp .env.example .env
# Edit .env: GROQ_API_KEY=your_key_here

# Start the chat UI
streamlit run app.py    # → http://localhost:8501

# Legacy regression tests (kept for compatibility; OBSOLETE-AT-SCALE)
python tests/run_tests.py

# Batch evaluation
python tests/batch_eval.py
python tests/batch_eval.py --limit 10
python tests/batch_eval.py --strategy aho_corasick     # try alt strategy
python tests/batch_eval.py --legacy                    # bypass multi-turn

# v2 unit tests (pytest)
pytest tests/test_sufficiency_gate.py tests/test_conversation.py \
       tests/test_snomed_strategies.py tests/test_negation.py \
       tests/test_llm_provider.py tests/test_ambiguity_coverage.py

# QA agent (rate-limited for Groq free tier)
python qa_testing/test_agent.py
python qa_testing/test_agent.py --limit 20
python qa_testing/test_agent.py --category injection
python qa_testing/test_agent.py --input qa_testing/test_cases_1000.json
```

---

## Technology Stack

| Package | Version | Purpose |
|---|---|---|
| streamlit | 1.40.0 | Chat UI |
| groq | 0.11.0 | LLM API client (used only inside `groq_provider.py`) |
| sentence-transformers | 3.2.1 | Embedding model `all-MiniLM-L6-v2` |
| rapidfuzz | 3.10.0 | Fuzzy string matching |
| pyahocorasick | ≥ 2.0.0 | AhoCorasickStrategy automaton |
| pydantic | 2.9.2 | All output models (V2 syntax only) |
| pandas | 2.2.3 | CSV loading |
| torch | 2.6.0 | Required by sentence-transformers |
| numpy | 2.1.0 | Required by sentence-transformers + semantic-search fallback |
| python-dotenv | 1.0.1 | .env loading |
| httpx | 0.27.2 | HTTP client |
| chromadb | *(optional)* | Vector DB — soft dependency, see below |

**chromadb is NOT in requirements.txt.** `chroma-hnswlib` needs MS C++ Build Tools on Windows and has no Py 3.13 wheel. The code tries `import chromadb` at runtime; on failure it falls back to numpy cosine similarity (same quality for our ~116-term dataset). If a teammate installs `chromadb==0.6.3`, it's used automatically — no code changes.

**Python 3.13 compatibility.** `torch==2.6.0` and `numpy==2.1.0` are chosen for Py 3.13. Original spec had `torch==2.2.2`/`numpy==1.26.4` which only support up to Py 3.12.

**Pydantic V2 only.** `model_config = ConfigDict(...)`, `model_dump()`. Never `.dict()` or `class Config`.

---

## Architecture: Full Pipeline (v2)

```
User query (raw string)              ┌─────────────────────────────────────────┐
        │                            │ ConversationSession (st.session_state)  │
        ▼                            │  • session_id (uuid4)                   │
┌─────────────────────────────────┐  │  • turns: list[Turn]                    │
│ Preprocessor.process()          │  │  • canonical_query (merged)             │
│  • Length 3–500                 │  │  • is_max_turns_reached() — hard cap 3  │
│  • ~77 injection/harmful regex  │  └─────────────────────────────────────────┘
│  • Null-byte strip first        │                  ▲       ▲
└──────────┬──────────────────────┘                  │       │ (read/append)
           │                                         │       │
           ▼                                         │       │
session.compute_canonical_query() — pure merge ──────┘       │
substitute-or-append (clarification answer substitutes       │
the previous trigger; free-text appends)                     │
           │                                                 │
           ▼                                                 │
Preprocessor.assert_safe(canonical)  — defense-in-depth      │
           │                                                 │
           ▼                                                 │
session.set_canonical_query(canonical)  — mutator ───────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ SufficiencyGate.evaluate(canonical, session)  — DETERMINISTIC, no LLM   │
│  1. max-turns escape valve → sufficient=True                            │
│  2. AmbiguousTermsRegistry.find_trigger() → if hit & no override:       │
│     sufficient=False, reason="ambiguous_trigger"                        │
│  3. default → sufficient=True                                           │
└────────────┬────────────────────────────────────────────────────────────┘
             │
             │  sufficient=False ─────────┐
             │                            ▼
             │                ResponseAssembler.build_clarification()
             │                            │
             │                            ▼
             │                ClarificationOutput → UI (chat message + buttons)
             │
             │  sufficient=True
             ▼
┌────────────────────────────┐    ┌───────────────────────────────────────┐
│ Path A — LLM filter        │    │ Path B — Algorithmic SNOMED           │
│ FilterExtractor.extract()  │    │ SNOMEDSearchStrategy.search()         │
│   via LLMProvider          │    │   default: HybridCascadeStrategy      │
│ → ExtractedFilters         │    │   alternatives: aho_corasick,         │
│   (investigator, site,     │    │                 ngram_lookup          │
│    city, state, phase)     │    │ → list[SNOMEDMatch] with char spans   │
└────────────┬───────────────┘    └────────────────┬──────────────────────┘
             │                                     │
             └──────────────┬──────────────────────┘
                            │  ThreadPoolExecutor.submit()
                            │  joined via futures.wait(ALL_COMPLETED, timeout=15s)
                            │  on timeout → PipelineError("extraction_timeout")
                            ▼
            NegationAnnotator.annotate()  — NegEx, deterministic
            (pre/post cues, 5-token window, pseudo-negation suppressed,
             comma does NOT stop scan; period/;/!/?/newline do)
                            │
                            ▼
            EmbeddingAmbiguityGate.evaluate()  — Step 5b, deterministic
            Signal A: 0 high-conf + ≥3 mid-band embedding neighbors (0.42..0.60)
            Signal B: ≥3 high-conf matches with conf spread < 0.08
                → ClarificationOutput (triggered_by=None, append-mode)
            else → None, fall through
            (no-op for strategies without get_top_neighbors)
                            │
                            ▼
            Clinical-intent gate (every turn)
            if 0 qualifying SNOMED (conf ≥ 0.60, not negated) AND 0 filters set
                → PreprocessorError("No clinical content found.")
                            │
                            ▼
            SufficiencyGate.post_extraction_check()
            if filters set AND 0 qualifying SNOMED matches
                → sufficient=False, reason="filters_without_condition"
                → ClarificationOutput (DEFAULT_CONDITION_PROMPT)
                            │
                            ▼
            GeoNormalizer.normalize(city, state_value_for_geo(state_filter))
            (state.values: list[str] schema preserved for multi-state regions)
                            │
                            ▼
            Preprocessor.assert_safe(canonical)  — belt-and-suspenders
                            │
                            ▼
            ResponseAssembler.assemble() → NLPOutput(type="search", ...)
                            │
                            ▼
            session.append_turn(...)  — APPEND ONLY AFTER successful assembly
                            │
                            ▼
            UI renders results in st.chat_message
```

**Discriminated union output:** every pipeline call returns `NLPOutput | ClarificationOutput`, distinguished by the `type` literal field.

**Per-turn structured log line (JSON):** `ts`, `session_id`, `turn_index`, `clarification_count`, `path` (`sufficiency_clarification` | `max_turns` | `embedding_ambiguity_clarification` | `clinical_intent` | `post_extraction_safety` | `search`), `decision_reason`, `snomed_match_count`, `snomed_strategy`, `filter_count`, `llm_provider`, `processing_time_ms`. **Never logged:** any raw text, canonical query, filter values, SNOMED display strings, LLM response, trigger values, option text.

---

## Output Schema

```python
ResponseOutput = Annotated[Union[NLPOutput, ClarificationOutput], Field(discriminator="type")]

NLPOutput
├── type: Literal["search"] = "search"
├── snomed_terms: list[SNOMEDTermOutput]
│   └── SNOMEDTermOutput
│       ├── code: str            # SNOMED concept ID
│       ├── display: str         # lowercase preferred_term
│       ├── match_type: str      # "exact" | "synonym" | "fuzzy" | "semantic" | "ngram"
│       ├── confidence: float    # 0.0–1.0
│       ├── original_text: str
│       └── negated: bool        # always False here; negated matches excluded
├── filters: FiltersOutput
│   ├── investigator_name: FilterFieldOutput  → value: Optional[str], confidence: float
│   ├── site_name: FilterFieldOutput
│   ├── city: FilterFieldOutput
│   ├── state: StateFilterOutput              ← DIFFERENT from other filters
│   │   ├── values: list[str]    # 1 for city/state; many for "east coast"
│   │   ├── confidence: float
│   │   └── is_region: bool
│   └── phase: FilterFieldOutput
└── metadata: MetadataOutput
    ├── processing_time_ms: int
    ├── total_snomed_matches: int
    ├── snomed_match_types: dict[str, int]
    └── negated_terms_excluded: int

ClarificationOutput
├── type: Literal["clarification"] = "clarification"
├── question: str               # html.escape'd
├── options: list[str]          # each html.escape'd
├── canonical_query: str        # html.escape'd; for transparency
├── turn_number: int            # 1-indexed clarification turn
├── max_turns: int = 3
└── metadata: MetadataOutput
```

**CRITICAL:** `state` uses `StateFilterOutput` with `values: list[str]`, NOT a single `.value`. Always read `output.filters.state.values`.

---

## Component Reference

### `src/preprocessor.py` — `Preprocessor`
- `process(raw: str) -> PreprocessedInput` — length check (3–500), null-byte strip, ~77 injection/harmful regex patterns, SYSTEM-prefix anchor (`\A`).
- `assert_safe(text: str) -> None` — runs the injection-pattern subset only (no length check) on a merged canonical query. Used by pipeline as defense-in-depth.
- Raises `PreprocessorError` with user-safe messages: `"Query must be at least 3 characters"`, `"Query must be under 500 characters"`, `"Invalid query detected"`, `"No clinical content found. Please enter a query about a medical condition, investigator, research site, location, or study phase."` (last is raised in pipeline, not preprocessor).
- Pattern coverage: prompt injection, code/script injection, data exfiltration, social engineering, LLM special tokens (`[INST]`, `<<SYS>>`), template/SSTI, HTTP header injection, XML tags, path traversal (Unix + Windows), SQL injection, WMD/weapon synthesis (verb-noun: make/build/create + weapon/bomb/firearm), controlled-substance manufacturing (cook/make/grow + drug nouns), child safety, self-harm (incl. "how to commit suicide", "ways to end my/your life"), poison-as-attack, cybercrime (write/create/develop + malware/exploit).

### `src/conversation.py` — `ConversationSession`, `Turn`
- Pydantic V2. `Turn` is `frozen=True`; `ConversationSession` is `frozen=False`.
- `Turn`: `turn_index`, `user_input`, `canonical_query`, `decision: Optional[SufficiencyDecision]`, `filters: Optional[ExtractedFilters]`, `snomed_matches: list[SNOMEDMatch]`, `geo: Optional[GeoResult]`, `timestamp`. Clarification turns have filters/geo None and empty snomed_matches.
- Methods: `new()` (uuid4 + ts), `append_turn(t)`, `compute_canonical_query(input) -> str` (pure), `set_canonical_query(q)` (mutator), `update_canonical_query(input)` (compute+set convenience), `clarification_turn_count()`, `is_max_turns_reached()` (≥ `max_clarification_turns`, default 3), `summarize_known_filters()`, `to_dict()`, `from_dict(d, on_error="raise"|"new_session")`, `summary_for_logging()` (the ONLY safe method for logging session state — `to_dict()`/`repr()` MUST NEVER be logged).
- Canonical-query merge: if last turn was a clarification with `triggered_by=T`, substitute T with the new input (case-insensitive whole-word `\b`, **literal replacement via `lambda _m: user_input` — B6 fix**); else append. M2 split: `compute_*` runs before `assert_safe()`; `set_*` only after it passes.

### `src/sufficiency_gate.py` — `SufficiencyGate`, `AmbiguousTermsRegistry`, `AmbiguousEntry`, `SufficiencyDecision`, `EmbeddingAmbiguityGate`, `DEFAULT_CONDITION_PROMPT`, `_count_set_filters()`, `_any_filter_set()`
- `SufficiencyDecision` (frozen): `sufficient: bool`, `reason: str` (enum: `ok_no_trigger`, `ambiguous_trigger`, `max_turns_reached`, `ok_post_extraction`, `filters_without_condition`, `legacy_bypass`), `triggered_by: Optional[str]`, `matched_entry: Optional[AmbiguousEntry]`.
- `AmbiguousTermsRegistry.__init__(path, snomed_strategy, snomed_csv_path, strict_validation)` — validates EVERY option in `ambiguous_terms.json` resolves through the strategy at confidence ≥ 0.85. `strict_validation=True` → `sys.exit(1)` on failure (default, for POC/CI). `False` → log warn, drop trigger (for production rolling deploys). Empty registry after validation → raises `ValueError` even when lenient.
- **Layer 1 derived triggers** (post-JSON-load, in `__init__`): `_build_derived_entries(csv_path)` scans `preferred_term` column for single tokens passing length (≥`MIN_DERIVED_TOKEN_LEN=4`), stopword (27 entries incl. "human"), and frequency (≥`MIN_DERIVED_TERM_FREQUENCY=2`) gates. Synthesizes one `AmbiguousEntry` per qualifying token with matching CSV rows as options. `_merge_entries(json, derived)` unions; **hand-curated JSON wins on key collision** (logged). Options use raw lowercase CSV values (no `.title()` — preserves acronyms). Derived `override_terms` include CSV preferred_terms + synonyms containing the token (suppresses compound queries like "lung cancer" from firing the bare "lung" trigger). Derive failures gracefully fall back to JSON-only regardless of `strict_validation`.
- Trigger detection: single compiled regex with longest-first alternation, `\b` whole-word, `re.IGNORECASE`. `find_trigger(query)` returns `(trigger, entry)` for first hit whose override terms are absent.
- Override-terms auto-derivation: every `option` lowercased + every CSV `preferred_term` containing the trigger as a whole-word substring. **Self-defeat guard:** drop any override equal to the trigger itself. Manual `manual_override_terms` from JSON are unioned in.
- `SufficiencyGate.evaluate(canonical, session)`: 3 rules — max-turns → registry trigger → default sufficient. Does NOT call LLM or SNOMED.
- `SufficiencyGate.post_extraction_check(matches, filters)`: returns `filters_without_condition` (with `DEFAULT_CONDITION_PROMPT` as matched_entry) iff ≥1 filter is set AND 0 SNOMED matches qualify (conf ≥ 0.60 and not negated).
- `DEFAULT_CONDITION_PROMPT`: built lazily on first `SufficiencyGate.__init__` (NOT module-import time — `snomed_csv_path` is passed in). Top-5 categories derived from CSV preferred-terms clustered against `_CATEGORY_SEEDS` (Cancer, Diabetes, Heart Disease, Autoimmune, Neurological). Hard fallback on CSV failure: `["Cancer", "Diabetes", "Heart Disease", "Autoimmune", "Other"]`. Memoized on the class.
- `EmbeddingAmbiguityGate(strategy)` — **Layer 2**, pipeline step 5b. Duck-types `hasattr(strategy, "get_top_neighbors")` at init; gate silently no-ops on strategies without an embedder (`aho_corasick`, `ngram_lookup`). `evaluate(canonical, snomed_matches) -> Optional[SufficiencyDecision]` MUST receive post-`NegationAnnotator` matches (filters negated internally). **Signal A** (rare-term): 0 high-conf (`< MIN_CONFIDENCE`) AND ≥`LAYER2_MIN_NEIGHBORS=3` mid-band embedding neighbors in `[LAYER2_LOW_THRESHOLD=0.42, MIN_CONFIDENCE)` → clarification with neighbors as options. **Signal B** (genuine ambiguity): ≥`LAYER2_MIN_GENUINE_AMBIG=3` high-conf matches AND `max(conf) - min(conf) < LAYER2_SPREAD_THRESHOLD=0.08` → clarification with top-5 matches as options. Returns `None` (gate falls through to clinical-intent gate) otherwise. **Always sets `triggered_by=None`** → canonical merge uses append-mode (no lossy single-token substitution).

### `src/llm_provider/`
- `base.py` — `LLMProvider` Protocol (`@runtime_checkable`): `name`, `model_id`, `complete(system_prompt, user_prompt, max_tokens, temperature, timeout, json_mode) -> str`. Raises `LLMProviderError` on unrecoverable failure.
- `groq_provider.py` — `GroqProvider`. Default model `llama-3.1-8b-instant`. Retry policy: transient (network timeout, 429, 5xx) retried up to 3× with `[1s, 2s, 4s]` backoff; permanent (401/404/400) raises immediately. **Only file in the repo importing `groq`.**
- `registry.py` — `PROVIDER_REGISTRY = {"groq": GroqProvider}`, `DEFAULT_PROVIDER = "groq"`. `get_provider(name=None, **kwargs)` reads `LLM_PROVIDER` env var if name omitted.

### `src/filter_extractor.py` — `FilterExtractor`, `ExtractedFilters`, `FilterField`, `StateFilter`
- LLM extracts **filters only** — no SNOMED, no medical terms, no negation. Pydantic V2 models with `frozen=True`.
- System prompt enforces Kansas-City disambiguation (state extracted only from explicit mentions, not from city or institution names) and phase normalization rules ported from the legacy extractor.
- `extract(canonical: str) -> ExtractedFilters` — calls `provider.complete(... json_mode=True)`, parses JSON, validates required keys, sanitizes literal `"null"` strings, raises `ExtractionError` on parse failure.

### `src/snomed_search/`
- `base.py` — `SNOMEDSearchStrategy` Protocol (`@runtime_checkable`) + `SNOMEDMatch` dataclass. `SNOMEDMatch.span: tuple[int, int]` is REQUIRED with `__post_init__` validation (no None, start ≥ 0, end > start).
- `hybrid_cascade.py` — default. Cascade: exact → synonym → fuzzy (rapidfuzz token_sort_ratio, cutoff 88) → semantic (sentence-transformer + chromadb-or-numpy, threshold 0.82). Fuzzy/semantic operate on residual unmatched character spans via `_compute_residual_spans()`. Ports `ALIAS_DICTIONARY` from `snomed_resolver.py` shim during transition. Also exposes `get_top_neighbors(query, n=15, low_threshold=0.42) -> list[SNOMEDMatch]` (not part of the Protocol) for `EmbeddingAmbiguityGate` to reuse the same embedder/index for Layer 2. Returns `[]` and emits one INFO log if semantic unavailable.
- `aho_corasick.py` — substring scan via `pyahocorasick` automaton, word-boundary post-filter, longest-match dedup. Microsecond scale.
- `ngram_lookup.py` — 1..max_n token contiguous windows looked up in exact + synonym indexes. Pure stdlib.
- `negation.py` — `NegationAnnotator.annotate(query, matches) -> list[SNOMEDMatch]`. NegEx pre-cues (no, not, without, denies, ...), post-cues (unlikely, ruled out, ...), pseudo-negation suppressors ("no contraindication for", "no change in", ...). Window 5 tokens. Stop chars: `.;!?\n` — **comma does NOT stop scan** (clinical syntax chains).
- `registry.py` — `STRATEGY_REGISTRY = {"hybrid_cascade": ..., "aho_corasick": ..., "ngram_lookup": ...}`, `DEFAULT_STRATEGY = "hybrid_cascade"`. `get_strategy(name=None)` reads `SNOMED_SEARCH_STRATEGY` env var; runs `health_check()` and raises `StrategyError` if `ready=False`.

### `src/normalizers/`
- `base.py` — `FilterNormalizer` Protocol.
- `geo.py` — `GeoNormalizer.normalize(city, state) -> GeoResult(city, states: list[str], country, confidence, is_region, original_city, original_state)`. Lookup order: exact city → exact region → fuzzy city (rapidfuzz token_sort_ratio cutoff 82, confidence = score/100 × 0.90). Multi-state regions return all covered states; single-state metro regions return `[state]` with `city=primary_city`. User-supplied state overrides inferred for non-regions; does NOT override `region_states` for multi-state regions.

### `src/assembler.py` — `ResponseAssembler`, `render_question()`
- `assemble(filters, snomed_matches, geo, start_time) -> NLPOutput`: sorts SNOMED by confidence desc, drops negated, dedups by code keeping highest confidence, drops matches below MIN_CONFIDENCE (0.60), applies geo if `geo.confidence ≥ 0.60` (else falls back to raw filter values, with `_INVALID_STATE_VALUES` frozenset blocking literal `"null"` from leaking).
- `build_clarification(decision, session, start_time) -> ClarificationOutput`: calls module-level `render_question(entry, trigger, session)` with `prior_filters` substitution from `session.summarize_known_filters()`. All output strings `html.escape`'d.
- Every output model is Pydantic V2 with `frozen=True`.

### `src/pipeline.py` — `NLPPipeline`
- **Primary entry point:** `run_with_session(raw_query, session) -> NLPOutput | ClarificationOutput`. 10 steps (incl. step 5b `EmbeddingAmbiguityGate`) per the diagram above.
- **Legacy shim:** `run(raw_query) -> NLPOutput` (OBSOLETE-AT-SCALE). Wraps `run_with_session` in a fresh single-turn session. If a `ClarificationOutput` is returned (gate fired), `_assemble_best_effort_from_session` bypasses the gate and runs the extraction path so legacy callers always get an `NLPOutput`.
- Init order: SNOMED strategy → LLM provider → `AmbiguousTermsRegistry` (needs strategy; auto-derives Layer 1 entries from CSV) → `SufficiencyGate` → `EmbeddingAmbiguityGate` (needs strategy) → `FilterExtractor` → `GeoNormalizer` → `NegationAnnotator` → `Preprocessor` → `ResponseAssembler` → `ThreadPoolExecutor(max_workers=2)`.
- Step 5b firing path appends a `Turn` with `decision=embed_decision, filters=None, geo=None, snomed_matches=<original>` and emits `LOG_PATH_EMBEDDING_AMBIGUITY` log line. Mutation discipline preserved: turn appended after successful `build_clarification`.
- Parallel join uses `futures.wait([...], timeout=15.0, return_when=ALL_COMPLETED)`. Timeout cancels both futures, raises `PipelineError("extraction_timeout")`. `LLMProviderError` propagates with provider name logged but never `original_error.message`. Unknown exceptions become `PipelineError("strategy_unavailable")`.
- Mutation discipline: session.append_turn() runs ONLY after successful clarification build or final assembly. A failure in any earlier step leaves the session unchanged.

### `app.py` — Streamlit chat UI
- `@st.cache_resource` on `load_pipeline()` — pipeline initializes once per process.
- `st.session_state["conversation"]` holds the `ConversationSession`. "New Search" sidebar button replaces it with `ConversationSession.new()`.
- Per-turn rendering: `st.chat_message("user")` for `turn.user_input`; `st.chat_message("assistant")` renders either the search NLPOutput or the clarification question + a bulleted hint list of options. The user types their answer (free text — option text or anything else) into the existing `st.chat_input` at the bottom of the page; no per-option buttons.
- Rate limiting (session-state based): max 5 queries / 60s window, max 30 / session. Clarifications count toward limits.
- Escaping ownership: `ResponseAssembler.build_clarification()` (`src/assembler.py:122-124`) `html.escape`'s `question`, each `option`, and `canonical_query` before placing them into `ClarificationOutput`. `app.py` therefore renders those fields WITHOUT `_safe()` and WITHOUT `unsafe_allow_html=True`. The `_safe()` helper is still used in `_render_nlp_output()` for SNOMED/filter display strings (which are NOT pre-escaped) and at the user-input `st.chat_message` line. PHI disclaimer in sidebar + inline.
- Errors render as assistant chat messages, not banners. `PreprocessorError` → warning style; `ExtractionError`, `LLMProviderError`, `PipelineError` → error style with generic text. Log lines use `session.summary_for_logging()` only.

### OBSOLETE-AT-SCALE shims
- `src/extractor.py` — original combined extractor (medical terms + filters). Still imported by `tests/run_tests.py`.
- `src/snomed_resolver.py` — original 4-step cascade. Provides `ALIAS_DICTIONARY` used by `hybrid_cascade.py` during the transition (moves into `hybrid_cascade.py` at cleanup).
- `src/geo_normalizer.py` — 2-line re-export of `src/normalizers/geo.py`.
- `NLPPipeline.run()` + `_assemble_best_effort_from_session()` — single-turn shim around `run_with_session`.
- `tests/run_tests.py` — pre-multi-turn regression suite.

---

## Data Files

### `data/snomed_clinical_trials.csv`
Columns `concept_id`, `preferred_term`, `synonyms` (pipe-separated). 116 rows. Coverage: cancers (15+), cardiovascular, metabolic/endocrine, neurological, inflammatory/autoimmune, infectious diseases, procedures, imaging, measurements. **Rule:** every option in `ambiguous_terms.json` MUST resolve through the configured strategy at conf ≥ 0.85 at startup — validated by `AmbiguousTermsRegistry`.

### `data/geo_canonical.json`
~200 city entries (incl. aliases), 50 US states + DC + Canadian provinces, 59 regions (25 multi-state, 34 metro/single-state). Multi-state regions: east coast (15 states), west coast (3), northeast (9), southeast (9), southwest (5), mid-atlantic (7), mountain west (8), deep south (5), great lakes (8), great plains (7), upper midwest (5), northwest (4), midwest (12), the south (14), new england (6), pacific northwest (3), tri-state/tri state/tristate (NY/NJ/CT).

### `data/ambiguous_terms.json`
Schema per entry: `category`, `question_template` (`{trigger}` and `{prior_filters}` placeholders), `options` (3–5 SNOMED-resolvable strings), `manual_override_terms` (optional), `max_options`. Seeded triggers: `cancer`, `tumor`, `oncology`, `diabetes`, `diabetic`, `heart`, `cardiac`, `cardiovascular`, `arthritis`, `autoimmune`, `neurological`/`neuro`/`brain`, `liver`/`hepatic`, `lung disease`/`pulmonary`, `depression`, `infection`. **Known limitation:** the `depression` trigger maps to neuro-adjacent SNOMED options because psychiatric DSM-5 concepts aren't yet in the CSV.

---

## Testing

### `tests/run_tests.py` (legacy regression — OBSOLETE-AT-SCALE)
20 hardcoded cases in `tests/test_cases.json`. State comparison uses membership in `state.values` list. Pass threshold: 15/20.

### `tests/batch_eval.py`
Reads `tests/batch_test_cases.csv` (100 cases). CSV cols: `id`, `category`, `description`, `input`, `expected_snomed_codes` (pipe-sep), `expected_snomed_absent` (pipe-sep), `expected_city`, `expected_state`, `expected_phase`, `expected_investigator_name`, `expected_site_name`. Optional cols (blank = don't check): `expected_type` (search | clarification), `expected_clarification_field`, `expected_options_contain`. Multi-turn cases use `>>>` separator: `cancer>>>Lung Cancer`. Flags: `--limit N`, `--strategy <name>`, `--legacy` (bypass multi-turn). Writes timestamped CSV to `tests/results/`. Prints overall pass rate, per-category, per-filter accuracy, SNOMED recall, timing.

### `tests/test_*.py` (pytest)
- `test_sufficiency_gate.py` — gate rules, override behavior, self-defeat guard, max-turns escape, partial-word non-match.
- `test_conversation.py` — multi-turn flows (C1-C13): basic clarification, substitute-on-answer, append-on-free-text, max-turns escape, malicious 2nd-turn payload blocked, round-trip serialization, corruption recovery, plural-trigger false negative (documented).
- `test_snomed_strategies.py` — top-5 SNOMED codes per strategy with set Jaccard ≥ 0.80 on ≥80% of `batch_test_cases.csv`. Span validity, thread-safety (50 concurrent search calls), health_check.
- `test_negation.py` — pre/post cues, sentence-boundary stop, comma chain pass-through, pseudo-negation suppression, window edge.
- `test_llm_provider.py` — retry budget, transient vs permanent error policy, mock provider, factory error message.

### `qa_testing/test_agent.py`
JSON test cases (default `qa_testing/test_cases_202.json`; 1000-case set available). Schema: `id`, `input`, `category`, `description`, `expected.snomed_codes`, `expected.filters.{city,state,phase}`, `expected.should_reject`. Categories: `valid`, `injection`, `harmful`, `edge_case`, `missing_condition`, `invalid_nonsense`. `NO_API_CATEGORIES = {"injection", "harmful", "edge_case"}` are expected to be blocked by the preprocessor and run instantly without an API call. Rate-limited token-bucket runner (default `--rpm 25`) for the rest, with exponential backoff on 429 up to 60s and `MAX_RETRIES=3`. Cases that exhaust retries are `RATE_LIMITED` (skipped, not failed). Flags: `--input`, `--limit`, `--category`, `--output`, `--rpm`. Writes a `results.md` markdown report.

---

## Security & HIPAA

### Per-turn security stack
1. **Preprocessor.process(raw_user_input)** — ~77 regex patterns + length + null-byte strip.
2. **Preprocessor.assert_safe(canonical)** — defense-in-depth on the merged canonical query (post-substitute/append).
3. **Pre-extraction SufficiencyGate** — runs BEFORE any LLM call. Insufficient queries cost zero tokens.
4. **Clinical-intent gate** — 0 qualifying SNOMED + 0 filters → reject.
5. **Post-extraction safety check** — filters set but 0 SNOMED → clarification.
6. **Rate limiting** — 5/60s and 30/session (app.py).
7. **Max-turns cap** — 3 clarifications per session, hard.
8. **assert_safe again** — belt-and-suspenders before final assembly.
9. **Output sanitization** — `html.escape()` on every user-derived/LLM-derived display string.

### HIPAA log hygiene (enforced project-wide)
- Per-turn structured JSON log line emitted by `pipeline._log_turn`.
- **NEVER logged:** raw query text, canonical query, user_input, preprocessed text, filter values, SNOMED display strings, LLM response, `decision.triggered_by` values, option text, `original_error.message` from `LLMProviderError`.
- **LOGGED:** counts, lengths, confidence scores, SNOMED codes, decision reason enums, latency, session_id, turn_index, clarification_count, strategy/provider names.
- Convention: any log statement involving session state uses `session.summary_for_logging()`. **Reviewers should grep for `to_dict()` and `repr(session)` in log lines.**

### API key handling
Only via `GROQ_API_KEY` env var or Streamlit secrets. Never hardcoded, never logged.

### Known POC gaps
- Rate limiting is session-state based — bypassable by opening new tabs. Production needs server-side per-IP rate limits.
- No authentication / authorization.
- No audit logging for compliance.
- Background-thread leak under sustained extraction-timeout DoS — `ThreadPoolExecutor.cancel()` doesn't truly cancel running threads.
- `chromadb` not on Py 3.13 without C++ Build Tools — numpy fallback in use.
- `pyahocorasick` C extension — Py 3.13 wheels exist for major platforms; without them, fall back to `ngram_lookup` or `hybrid_cascade`.

---

## Critical Constants

| Constant | Value | Location | Impact if changed |
|---|---|---|---|
| Groq model | `"llama-3.1-8b-instant"` | `llm_provider/groq_provider.py` | Cost, latency, quality |
| Embedding model | `"all-MiniLM-L6-v2"` | `snomed_search/hybrid_cascade.py` | Re-index required |
| SNOMED MIN_CONFIDENCE | `0.60` | `assembler.py`, `pipeline.py` step 6 | What's included in output |
| Semantic threshold | `0.82` | `snomed_search/hybrid_cascade.py` | FP/FN tradeoff |
| SNOMED fuzzy cutoff | `88` | `snomed_search/hybrid_cascade.py` | FP/FN tradeoff |
| Geo fuzzy cutoff | `82` | `normalizers/geo.py` | Typo tolerance |
| Geo confidence threshold | `0.60` | `assembler.py` | Below → raw LLM fallback |
| `OPTION_MIN_CONFIDENCE` | `0.85` | `sufficiency_gate.py` | Registry option validation gate |
| `AMBIG_JSON_MIN/MAX_OPTIONS` | `3 / 5` | `sufficiency_gate.py` | Registry schema constraint |
| `PARALLEL_TIMEOUT_SECONDS` | `15.0` | `pipeline.py` | Hard timeout for both futures |
| `THREAD_POOL_MAX_WORKERS` | `2` | `pipeline.py` | One per pipeline instance |
| NegEx `WINDOW_SIZE` | `5` (tokens) | `snomed_search/negation.py` | Negation scan range |
| `max_clarification_turns` | `3` | `conversation.py` | Hard cap per session |
| Batch eval fuzzy threshold | `88` | `batch_eval.py`, `run_tests.py` | Test sensitivity |
| `MIN_DERIVED_TOKEN_LEN` | `4` | `sufficiency_gate.py` | Layer 1 token min length |
| `MIN_DERIVED_TERM_FREQUENCY` | `2` | `sufficiency_gate.py` | Token must appear in ≥N preferred_terms |
| `LAYER2_LOW_THRESHOLD` | `0.42` | `sufficiency_gate.py` | Embedding mid-band floor |
| `LAYER2_MIN_NEIGHBORS` | `3` | `sufficiency_gate.py` | Min mid-band neighbors for Signal A |
| `LAYER2_MIN_GENUINE_AMBIG` | `3` | `sufficiency_gate.py` | Min high-conf matches for Signal B |
| `LAYER2_SPREAD_THRESHOLD` | `0.08` | `sufficiency_gate.py` | Max conf-spread for Signal B |

---

## Component Interfaces

```python
# preprocessor.py
Preprocessor.process(raw: str) -> PreprocessedInput
Preprocessor.assert_safe(text: str) -> None     # raises PreprocessorError
PreprocessedInput(text: str, original: str, char_count: int)

# llm_provider/base.py
class LLMProvider(Protocol):
    name: str
    model_id: str
    def complete(self, system_prompt, user_prompt, max_tokens=512,
                 temperature=0.0, timeout=10.0, json_mode=True) -> str: ...
    # raises LLMProviderError

# filter_extractor.py
FilterExtractor(provider: LLMProvider)
FilterExtractor.extract(canonical: str) -> ExtractedFilters
ExtractedFilters(investigator_name: FilterField, site_name: FilterField,
                 city: FilterField, state: StateFilter, phase: FilterField,
                 raw_response_length: int)
FilterField(value: Optional[str], confidence: float)
StateFilter(values: list[str], confidence: float, is_region: bool)

# snomed_search/base.py
class SNOMEDSearchStrategy(Protocol):
    name: str
    def __init__(self, dictionary_path: str, **kwargs): ...
    def search(self, query: str) -> list[SNOMEDMatch]: ...
    def health_check(self) -> dict: ...
@dataclass(frozen=True)
class SNOMEDMatch:
    code: str; display: str; match_type: str; confidence: float
    original_text: str; span: tuple[int, int]; negated: bool = False

# snomed_search/negation.py
NegationAnnotator.annotate(query: str, matches: list[SNOMEDMatch]) -> list[SNOMEDMatch]

# snomed_search/hybrid_cascade.py — extension method (NOT in Protocol)
HybridCascadeStrategy.get_top_neighbors(
    query: str, n: int = 15, low_threshold: float = 0.42,
) -> list[SNOMEDMatch]
# Returns [] and emits one INFO log if semantic unavailable. Used by EmbeddingAmbiguityGate.

# sufficiency_gate.py
class SufficiencyGate:
    def __init__(self, registry: AmbiguousTermsRegistry, snomed_csv_path: str): ...
    def evaluate(self, canonical: str, session: ConversationSession) -> SufficiencyDecision: ...
    def post_extraction_check(self, matches: list[SNOMEDMatch],
                              filters: ExtractedFilters) -> SufficiencyDecision: ...
class AmbiguousTermsRegistry:
    def __init__(self, path: str, snomed_strategy: SNOMEDSearchStrategy,
                 snomed_csv_path: str, strict_validation: bool = True): ...
    # __init__ auto-derives Layer 1 entries from snomed_csv_path; JSON wins on collision
    def find_trigger(self, query: str) -> Optional[tuple[str, AmbiguousEntry]]: ...
class EmbeddingAmbiguityGate:
    # Layer 2 — pipeline step 5b
    def __init__(self, strategy: SNOMEDSearchStrategy): ...
    # duck-types hasattr(strategy, "get_top_neighbors"); silently no-ops without it
    def evaluate(self, canonical: str,
                 snomed_matches: list[SNOMEDMatch]) -> Optional[SufficiencyDecision]: ...
    # snomed_matches MUST be post-NegationAnnotator
    # Returns None to fall through to clinical-intent gate

# conversation.py
ConversationSession.new() -> ConversationSession
ConversationSession.append_turn(t: Turn) -> None
ConversationSession.compute_canonical_query(user_input: str) -> str
ConversationSession.set_canonical_query(q: str) -> None
ConversationSession.update_canonical_query(user_input: str) -> str   # compute + set
ConversationSession.clarification_turn_count() -> int
ConversationSession.is_max_turns_reached() -> bool
ConversationSession.summarize_known_filters() -> str
ConversationSession.summary_for_logging() -> dict      # PHI-safe; ONLY method safe to log
ConversationSession.to_dict() -> dict                  # SERIALIZATION only; NEVER log
ConversationSession.from_dict(d: dict, on_error: str = "raise") -> ConversationSession

# normalizers/geo.py
GeoNormalizer.normalize(city: Optional[str], state: Optional[str]) -> GeoResult
GeoResult(city: Optional[str], states: list[str], country: Optional[str],
          confidence: float, is_region: bool,
          original_city: Optional[str], original_state: Optional[str])

# assembler.py (Pydantic V2, all frozen=True)
ResponseAssembler.assemble(filters, snomed_matches, geo, start_time) -> NLPOutput
ResponseAssembler.build_clarification(decision, session, start_time) -> ClarificationOutput
render_question(entry: AmbiguousEntry, trigger: str,
                session: ConversationSession) -> str

# pipeline.py — the only entry point app.py uses
NLPPipeline(llm_provider=None, snomed_strategy=None,
            ambiguous_terms_path=None, snomed_csv_path=None, geo_json_path=None,
            strict_validation=None, groq_api_key=None)
NLPPipeline.run_with_session(raw_query: str, session: ConversationSession)
    -> Union[NLPOutput, ClarificationOutput]
NLPPipeline.run(raw_query: str) -> NLPOutput        # OBSOLETE-AT-SCALE shim
```

---

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | — (required) | Groq API authentication |
| `LLM_PROVIDER` | `groq` | Selects provider via `llm_provider/registry.py` |
| `SNOMED_SEARCH_STRATEGY` | `hybrid_cascade` | Selects strategy via `snomed_search/registry.py` |
| `AMBIG_STRICT_VALIDATION` | `true` | `true` = `sys.exit(1)` on invalid registry options; `false` = log warn and drop (rolling-deploy graceful degrade) |

---

## Dependency Tree

```
app.py
  └── src/pipeline.py (NLPPipeline.run_with_session)
        ├── src/preprocessor.py
        ├── src/conversation.py (ConversationSession, Turn)
        ├── src/sufficiency_gate.py (SufficiencyGate, AmbiguousTermsRegistry,
        │     DEFAULT_CONDITION_PROMPT)
        │     ├── data/ambiguous_terms.json
        │     ├── data/snomed_clinical_trials.csv (for option validation
        │     │                                     + default prompt)
        │     └── src/snomed_search/* (for option validation)
        ├── src/llm_provider/registry.py → groq_provider.py
        │     ← needs GROQ_API_KEY
        ├── src/filter_extractor.py (FilterExtractor)
        │     └── src/llm_provider/base.LLMProvider
        ├── src/snomed_search/registry.py → {hybrid_cascade | aho_corasick | ngram_lookup}
        │     ├── data/snomed_clinical_trials.csv
        │     ├── sentence-transformers (hybrid_cascade only)
        │     ├── pyahocorasick (aho_corasick only)
        │     └── chromadb [optional] or numpy [fallback] (hybrid_cascade only)
        ├── src/snomed_search/negation.py (NegationAnnotator)
        ├── src/normalizers/geo.py (GeoNormalizer)
        │     └── data/geo_canonical.json
        └── src/assembler.py (ResponseAssembler, render_question)

tests/run_tests.py     → src/pipeline.py.run() shim  (OBSOLETE-AT-SCALE)
tests/batch_eval.py    → src/pipeline.py (supports --strategy, --legacy)
qa_testing/test_agent.py → src/pipeline.py
```

---

## How to Resume or Extend

### If resuming after interruption
1. Read this entire context.md.
2. Verify files exist per the inventory; verify `data/ambiguous_terms.json` is present.
3. Run `python tests/batch_eval.py --limit 5` to verify the pipeline.
4. Run `pytest tests/test_sufficiency_gate.py tests/test_conversation.py` for quick health.

### Adding a new SNOMED concept
1. Add a row to `data/snomed_clinical_trials.csv`: `concept_id,preferred_term,synonyms`.
2. If common-language alias is needed, add to `ALIAS_DICTIONARY` in `src/snomed_resolver.py` (shim) — target must equal a CSV `preferred_term`.
3. No other code changes — indexes rebuild at startup.

### Adding a new ambiguous trigger
1. Add an entry to `data/ambiguous_terms.json` (3–5 options).
2. Every option must resolve at conf ≥ 0.85 via the default strategy — if not, either add CSV terms first OR run with `AMBIG_STRICT_VALIDATION=false` and accept that the trigger will be skipped.
3. Override terms auto-derive from the CSV; supply `manual_override_terms` for archaic synonyms.

### Adding a new SNOMED strategy
1. Implement `SNOMEDSearchStrategy` in `src/snomed_search/<my_strategy>.py`. Populate `SNOMEDMatch.span` for every match. Document candidate-extraction in `search()` docstring.
2. Register in `src/snomed_search/registry.py`.
3. Add to `tests/test_snomed_strategies.py` parity suite.
4. `python tests/batch_eval.py --strategy my_strategy` to evaluate.

### Adding a new LLM provider
1. Implement `LLMProvider` in `src/llm_provider/<name>_provider.py`. Honor retry contract (transient retried 3× with `[1s,2s,4s]`; permanent raises immediately).
2. Register in `src/llm_provider/registry.py`.
3. Re-tune `FilterExtractor.SYSTEM_PROMPT` if needed; re-run batch eval.
4. Set `LLM_PROVIDER` env var.

### Modifying the output schema
1. Update `GeoResult` in `normalizers/geo.py` if geo fields change.
2. Update Pydantic models in `assembler.py` (`NLPOutput`, `ClarificationOutput`, `FiltersOutput`, etc.).
3. Update `assemble()` / `build_clarification()`.
4. Update `app.py` rendering.
5. Update `batch_eval.py` comparisons.
6. Pydantic V2 syntax throughout (`model_config = ConfigDict(...)`, `model_dump()`).

### Deploying to HuggingFace Spaces
1. All files except `.env` go into the Space.
2. `GROQ_API_KEY` as Space Secret.
3. `README.md` already has HF YAML front-matter.
4. chromadb won't install on HF free (no C++ Build Tools) — numpy fallback kicks in.
5. `pyahocorasick` wheel install will determine `aho_corasick` strategy availability.
6. First cold start: 2–5 min for install + model download.

---

## Post-Build Changes Log

### Clarification UI: buttons → text (2026-05-11)

Removed per-option `st.button` widgets from the clarification render in `app.py`. The clarification turn now shows the question + a passive bulleted hint list of options; the user replies in the existing `st.chat_input` (free text — they can type one of the listed options verbatim, a refinement, or anything else, all of which route through the same `Preprocessor` → `SufficiencyGate` → canonical-query merge path).

**Scope:** `app.py` only. No pipeline / schema / test changes. Existing `tests/test_conversation.py`, `tests/batch_eval.py`, `qa_testing/test_agent.py` continue to pass because they always fed option text directly as the next `user_input` string — they never simulated Streamlit button clicks.

**Latent bug fixed inline:** the previous render code called `_safe()` on `clarif.question` and `clarif.options` AND passed `unsafe_allow_html=True` to `st.markdown`. Those strings are already `html.escape`'d in `ResponseAssembler.build_clarification()` (`src/assembler.py:122-124`), so `_safe()` at the render site was double-escaping (visible for strings containing `& < >`, e.g. `Hodgkin's & Non-Hodgkin's` → `Hodgkin&amp;#39;s &amp;amp; Non-Hodgkin&amp;#39;s`). Render site now trusts the pre-escaped strings and renders with plain `st.markdown(...)`. Dropping `unsafe_allow_html=True` is also a defense-in-depth win — already-escaped entities render correctly without the flag, and the flag's only effect was to allow raw HTML tags through unsanitized.

**Escaping convention going forward:** `ResponseAssembler` owns escaping for `ClarificationOutput` fields. `app.py` does NOT re-escape those fields. For `NLPOutput` rendering (`_render_nlp_output()`), values are NOT pre-escaped, so `_safe()` is still applied at the render site there.

**Removed code:** the per-option `st.columns` + `st.button` loop in `_render_turn_result()` (lines 238–249) and the `pending_input` session-state consumer in `main()` (lines 345–349). Both `st.session_state["pending_input"]` and the per-option `st.rerun()` are no longer used anywhere in the app.

**Spec:** `rework-buttons-to-text.md` (Decision C revised post Security review; see the `<!-- override -->` block in section 3 for the double-escape + flag-removal rationale).

**Known pre-existing dead code (NOT removed in this work item):** `_run_pipeline_turn` (lines ~171–214) and `_sync_turn_outputs` (lines ~299–305) in `app.py` have zero callers. They were dead before this change. Candidate for a separate cleanup PR.

### Ambiguity coverage v2 — auto-derived triggers + embedding gate (2026-05-11)

Closes the coverage gap where bare anatomy terms ("kidney", "lung", "bone") were rejected by the clinical-intent gate with "No clinical content found." Two layers added.

**Layer 1 — Auto-derived triggers from SNOMED CSV** (in `src/sufficiency_gate.py`):
- At `AmbiguousTermsRegistry.__init__`, scan `data/snomed_clinical_trials.csv` `preferred_term` column. Extract single-token anatomy/system words that pass length (≥4), stopword (27 entries incl. "human"), and frequency (≥2 distinct preferred_terms) gates. Synthesize an `AmbiguousEntry` per qualifying token with the matching CSV rows as options.
- New helpers: `_extract_anatomy_tokens`, `_select_derived_options`, `_build_derived_entries`, `_merge_entries`. JSON entries always win on key collision.
- Options use raw lowercase CSV values — no `.title()` (preserves acronyms like "hiv").
- `override_terms` for derived entries auto-include preferred_terms + synonyms containing the token (suppresses compound queries like "lung cancer" from firing the bare "lung" trigger).
- Derive failures gracefully fall back to JSON-only regardless of `AMBIG_STRICT_VALIDATION`.

**Layer 2 — `EmbeddingAmbiguityGate`** (new class in `src/sufficiency_gate.py`, pipeline step 5b):
- Inserted between `NegationAnnotator` (step 5) and clinical-intent gate (step 6) in `_run_extraction_path`.
- Signal A: 0 high-conf SNOMED + ≥3 mid-band embedding neighbors in `[0.42, 0.60)` → clarification.
- Signal B: ≥3 high-conf matches with confidence spread < `0.08` → clarification (genuine ambiguity).
- Reuses `HybridCascadeStrategy._embedder` via new `get_top_neighbors(query, n=15, low_threshold=0.42) -> list[SNOMEDMatch]`. Strategies without the method (aho_corasick, ngram_lookup) are duck-type detected; gate silently no-ops.
- Always sets `triggered_by=None` → canonical merge uses append mode (preserves user phrasing; no lossy single-token substitution).
- Filters negated matches before computing signals (M4).
- New log path enum `LOG_PATH_EMBEDDING_AMBIGUITY = "embedding_ambiguity_clarification"`.

**Pre-existing bug fixed (B6):** `src/conversation.py:174` used `pattern.sub(user_input, ...)` where the replacement arg interprets `re` backreferences (`\1`, `\g<name>`). User input containing `\1` raised `re.error`. Patched to `pattern.sub(lambda _m: user_input, ..., count=1)` — replacement is now literal.

**Design**: `rework-ambiguity-coverage.md` (revision 2, post adversarial QA + Security review). 6 blockers + 8 majors addressed before implementation. Inline `<!-- rev2 -->` markers identify post-review changes.

**Tests added:** `tests/test_ambiguity_coverage.py` (17 cases, all pass). Existing suite (`test_sufficiency_gate.py`, `test_conversation.py`, `test_negation.py`, `test_llm_provider.py`) continues to pass: 85/85 total.

**New constants:** `MIN_DERIVED_TOKEN_LEN=4`, `MIN_DERIVED_TERM_FREQUENCY=2`, `LAYER2_LOW_THRESHOLD=0.42`, `LAYER2_MIN_NEIGHBORS=3`, `LAYER2_MIN_GENUINE_AMBIG=3`, `LAYER2_SPREAD_THRESHOLD=0.08`. `MIN_CONFIDENCE=0.60` is imported from `assembler.py` — single source of truth.

**Known limitation:** Single-CSV-row anatomy tokens (e.g. "kidney" appears in only "malignant neoplasm of kidney") fall to Layer 2 — they don't generate a Layer 1 entry. Layer 2's embedding gate handles them at the cost of one nearest-neighbor lookup per low-confidence query.

### v2 Rework — Architecture and pipeline split (2026-05-08)

Full implementation of `rework-nlp-proposal.md` (revision 2) and `rework-nlp-impl-spec.md`. Pipeline now supports multi-turn clarification and returns a discriminated union output type.

**Core changes:**
- Pre-extraction `SufficiencyGate` + registry-driven `AmbiguousTermsRegistry` — zero-token-cost for ambiguous queries.
- LLM extraction is now filters-only (`FilterExtractor`); SNOMED is fully algorithmic.
- Pluggable `SNOMEDSearchStrategy` Protocol with 3 implementations (`hybrid_cascade` default).
- `LLMProvider` abstraction (`src/llm_provider/`) — removes Groq lock-in.
- NegEx `NegationAnnotator` replaces LLM-based negation flag.
- `ConversationSession` multi-turn state with `Turn` records and PHI-safe `summary_for_logging()`.
- Parallel filter-extraction + SNOMED via `ThreadPoolExecutor` with shared 15s timeout.
- Discriminated union output: `NLPOutput(type="search") | ClarificationOutput(type="clarification")`.
- `app.py` rewritten as a chat UI with `st.chat_input` / `st.chat_message` and clickable clarification options.

**New env vars:** `LLM_PROVIDER` (default `groq`), `SNOMED_SEARCH_STRATEGY` (default `hybrid_cascade`), `AMBIG_STRICT_VALIDATION` (default `true`).

**OBSOLETE-AT-SCALE shims kept for compatibility:** `src/extractor.py`, `src/snomed_resolver.py`, `src/geo_normalizer.py`, `NLPPipeline.run()`, `tests/run_tests.py`.

**Known limitation:** the `depression` trigger maps to neuro-adjacent SNOMED options because psychiatric (DSM-5) concepts are not yet in `data/snomed_clinical_trials.csv`.

---

## Legacy (pre-2026-05-08) — historical reference only

The pre-rework pipeline (single-turn) used `src/extractor.py` for combined SNOMED + filter LLM extraction, `src/snomed_resolver.py` for a 4-step cascade resolver, and `src/geo_normalizer.py` directly. The OBSOLETE-AT-SCALE shims preserve those import paths so `tests/run_tests.py` and any external callers continue to work. Key v1 milestones:

- **Initial build** — single-turn pipeline; Pydantic V2; 116 SNOMED concepts; geo_canonical.json with cities, states, regions.
- **Security audit** — 22 patterns added (data exfil, social engineering); pipeline clinical-intent validation; HIPAA log hygiene.
- **Multi-state geo** — `GeoResult.state` → `states: list[str]`; `StateFilterOutput(values, confidence, is_region)`.
- **Python 3.13 compat** — `torch 2.6.0`, `numpy 2.1.0`; chromadb removed from requirements; numpy cosine fallback added.
- **Batch evaluation** — `tests/batch_eval.py` + 100-case CSV.
- **Security hardening 2026-05-05** — 22 harmful-content patterns, 15 injection patterns, null-byte strip first, `\ASYSTEM` anchor, Kansas-City fix in extractor prompt, duplicate-SNOMED-code dedup by highest confidence, `_INVALID_STATE_VALUES` guard.
- **Verb-noun attack coverage 2026-05-05** — 7 patterns for make/build + weapon, cook/grow + drug, write/develop + malware, etc.
- **QA test agent 2026-05-06** — `qa_testing/test_agent.py` + 202/1000-case JSON sets with token-bucket rate limiter for Groq free tier.
- **Rework design docs 2026-05-06** — `rework-nlp-proposal.md` + `rework-nlp-impl-spec.md` drafted.
