# Pipeline Architecture — Deep Dive

## Full 10-Step Flow

```
User query (raw string)              ┌─────────────────────────────────────────┐
        │                            │ ConversationSession (st.session_state)  │
        ▼                            │  • session_id (uuid4)                   │
┌─────────────────────────────────┐  │  • turns: list[Turn]                    │
│ Preprocessor.process()          │  │  • canonical_query (merged)             │
└──────────┬──────────────────────┘  └─────────────────────────────────────────┘
           ▼
session.compute_canonical_query() — pure merge
           ▼
Preprocessor.assert_safe(canonical)  — defense-in-depth
           ▼
session.set_canonical_query(canonical)  — mutator
           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ SufficiencyGate.evaluate(canonical, session)  — DETERMINISTIC, no LLM   │
└────────────┬────────────────────────────────────────────────────────────┘
             │  sufficient=True
             ▼
┌────────────────────────────┐    ┌───────────────────────────────────────┐
│ Path A — LLM filter        │    │ Path B — Algorithmic SNOMED           │
│ FilterExtractor.extract()  │    │ SNOMEDSearchStrategy.search()         │
└────────────┬───────────────┘    └────────────────┬──────────────────────┘
             │                                     │
             └──────────────┬──────────────────────┘
                            │  ThreadPoolExecutor (15s timeout)
                            ▼
            NegationAnnotator.annotate()  — NegEx
                            │
                            ▼
            EmbeddingAmbiguityGate.evaluate()  — Step 5b
                            │
                            ▼
            Clinical-intent gate (Step 6)
                            │
                            ▼
            SufficiencyGate.post_extraction_check() (Step 7)
                            │
                            ▼
            GeoNormalizer.normalize()
                            │
                            ▼
            ResponseAssembler.assemble()
                            │
                            ▼
            session.append_turn(...)
```

## Component Details

### 1. Preprocessor (`src/preprocessor.py`)
- **Responsibility:** First line of defense.
- **Rules:** Length 3–500, null-byte strip, ~77 regex patterns (injection, harmful, weapon/drug synthesis).
- **Fail Fast:** Raises `PreprocessorError` with user-safe messages.

### 2. SufficiencyGate (`src/sufficiency_gate.py`)
- **Layer 1:** Registry-based trigger detection. `AmbiguousTermsRegistry` validates options at startup.
- **Layer 2:** `EmbeddingAmbiguityGate` uses nearest-neighbor spread to detect semantic ambiguity.
- **Post-Extraction:** `post_extraction_check` ensures filters aren't orphaned (filters set but 0 clinical terms).

### 3. Extraction Path
- **LLM Provider:** Abstraction in `src/llm_provider/`. Groq is default.
- **FilterExtractor:** LLM extracts site/investigator/geo/phase. Strict system prompt prevents city->state inference.
- **SNOMED Strategies:** `hybrid_cascade` (Exact -> Synonym -> Fuzzy -> Semantic).

### 4. Normalizers
- **Geo:** `GeoNormalizer` handles fuzzy city matching and region expansion (e.g., "East Coast" -> 15 states).
- **Metric:** Aho-Corasick automaton scans for operational metrics (e.g., "enrollment rate").

### 5. Conversation State
- `ConversationSession` handles multi-turn state.
- **Canonical Merging:** Substitute trigger word if clarification answer; otherwise append.
- **Safety:** Turn only appended after successful assembly.
