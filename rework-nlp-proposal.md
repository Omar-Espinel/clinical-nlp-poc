# Rework Proposal — NLP Pipeline v2 (revision 2)
**Branch: `response` · Repo: clinical-nlp-poc · 2026-05-06**
*Architectural spec — pseudocode only · post adversarial QA + Security review*

---

## 0. Why this proposal exists (delta from `modified-proposal.md`)

`modified-proposal.md` solved the right product problem (multi-turn clarification for ambiguous queries) but the architecture is now being revised based on three constraints that surfaced after that proposal:

1. **Cost minimization is now a primary goal.** Each Groq call costs tokens; current pipeline spends them even on queries that will be rejected.
2. **SNOMED algorithm experimentation is required.** Omar wants to A/B test multiple SNOMED matching algorithms (Aho-Corasick, n-gram, hybrid). The existing `snomed_resolver.py` is a single hardcoded strategy — not swappable.
3. **LLM-provider lock-in is a risk.** Today Groq, tomorrow Claude / OpenAI / local model. The current extractor is provider-coupled.

| Decision | `modified-proposal.md` | This proposal |
|---|---|---|
| Sufficiency check position | After extraction (uses resolver confidence) | **Before** extraction (deterministic registry lookup) |
| Sufficiency check mechanism | Python over LLM output | Python over raw query string |
| Clarification text source | Templated lookup | Templated lookup (unchanged — kept) |
| Extractor scope | LLM extracts SNOMED + filters together | LLM extracts **filters only**; SNOMED is algorithmic |
| SNOMED resolution | Single hardcoded resolver | **Pluggable strategy** behind `SNOMEDSearchStrategy` Protocol |
| LLM provider | Hardcoded Groq | Behind `LLMProvider` abstraction |
| Negation detection | LLM-driven | **NegEx-style algorithmic** (deterministic) |
| Filter + SNOMED execution | Sequential | **Parallel** via thread pool |
| Multi-turn session | New (`ConversationSession`) | Reused — refined |

**Revision 2 changes (after adversarial QA + Security review):** see §14 Revision Log at the bottom for the full audit of which findings were accepted, rejected, or deferred.

**What survives unchanged from `modified-proposal.md`:**
- `ConversationSession` design
- Per-turn security loop
- HIPAA log hygiene
- Pydantic V2 throughout
- `state.values: list[str]` schema
- OBSOLETE-AT-SCALE marker convention
- Discriminated union output (`NLPOutput | ClarificationOutput`)
- Templated clarification questions

**What's discarded:**
- Post-extraction `SufficiencyEvaluator` (replaced by pre-extraction gate)
- LLM doing SNOMED concept identification (now algorithmic)
- Combined extractor prompt (split into smaller filter-only prompt)
- Sufficiency rules in any LLM system prompt
- LLM-driven negation detection (now NegEx)

---

## 1. Architectural principles

1. **Fail fast, save tokens.** Sufficiency is checked before any LLM call. Insufficient queries cost zero tokens.
2. **Deterministic-first.** LLM is used *only* where natural-language understanding genuinely helps (filter extraction). Everything else is deterministic.
3. **Pluggable SNOMED search.** The `SNOMEDSearchStrategy` Protocol is the keystone. Implementations are drop-in replaceable.
4. **LLM-provider independent.** The filter extractor depends only on `LLMProvider`. Single-class swap to change providers.
5. **Parallel where independent.** Filter extraction and SNOMED search run concurrently via `concurrent.futures.ThreadPoolExecutor`.
6. **Multi-turn but bounded.** Max 3 clarification turns per query. Hard cap, no override.
7. **Backwards-compatible with grep-able cleanup.** `# OBSOLETE-AT-SCALE: <reason>` markers throughout.

---

## 2. New / modified file inventory

| Path | Status | Purpose |
|---|---|---|
| `data/ambiguous_terms.json` | **NEW** | Versioned registry: triggers → clarification + override-terms |
| `src/sufficiency_gate.py` | **NEW** | Deterministic pre-extraction gate + post-extraction safety + DEFAULT_CONDITION_PROMPT |
| `src/conversation.py` | **NEW** | `ConversationSession`, `Turn`, canonical-query merge, serialization, log-safe summary |
| `src/exceptions.py` | **NEW** | `LLMProviderError`, `PipelineError`, `StrategyError` (centralized) |
| `src/llm_provider/` | **NEW PACKAGE** | LLM provider abstraction |
| `src/llm_provider/__init__.py` | NEW | Exports |
| `src/llm_provider/base.py` | NEW | `LLMProvider` Protocol + retry contract |
| `src/llm_provider/groq_provider.py` | NEW | `GroqProvider` (only file with `import groq`) |
| `src/llm_provider/registry.py` | NEW | Provider factory |
| `src/filter_extractor.py` | **NEW** | Replaces `extractor.py` for filter-only extraction |
| `src/snomed_search/` | **NEW PACKAGE** | Pluggable SNOMED matching |
| `src/snomed_search/__init__.py` | NEW | Exports |
| `src/snomed_search/base.py` | NEW | `SNOMEDSearchStrategy` Protocol + `SNOMEDMatch` dataclass |
| `src/snomed_search/aho_corasick.py` | NEW | Aho-Corasick implementation |
| `src/snomed_search/ngram_lookup.py` | NEW | Token n-gram window lookup |
| `src/snomed_search/hybrid_cascade.py` | NEW | Production default — ports existing 4-step cascade |
| `src/snomed_search/negation.py` | NEW | NegEx-style negation annotator |
| `src/snomed_search/registry.py` | NEW | Strategy factory |
| `src/normalizers/` | **NEW PACKAGE** | Geo + extension points |
| `src/normalizers/geo.py` | MOVED | From `src/geo_normalizer.py`; logic unchanged |
| `src/normalizers/base.py` | NEW | `FilterNormalizer` Protocol |
| `src/pipeline.py` | MODIFIED | Adds `run_with_session()`; orchestrates parallel paths |
| `src/assembler.py` | MODIFIED | Adds `ClarificationOutput` + discriminated union + `render_question()` |
| `src/extractor.py` | DEPRECATED | OBSOLETE-AT-SCALE shim |
| `src/snomed_resolver.py` | DEPRECATED | OBSOLETE-AT-SCALE shim wrapping `hybrid_cascade` |
| `src/geo_normalizer.py` | DEPRECATED | OBSOLETE-AT-SCALE re-export shim |
| `app.py` | MODIFIED | Chat UI |
| `tests/test_sufficiency_gate.py` | NEW | Unit tests for gate |
| `tests/test_conversation.py` | NEW | Multi-turn regression |
| `tests/test_snomed_strategies.py` | NEW | Cross-strategy parity |
| `tests/test_negation.py` | NEW | NegEx tests |
| `tests/test_llm_provider.py` | NEW | Provider abstraction with mock |
| `tests/batch_eval.py` | MODIFIED | Multi-turn `>>>`, `expected_type`, `--legacy`, `--strategy` |
| `tests/run_tests.py` | UNCHANGED | Tagged OBSOLETE-AT-SCALE |
| `requirements.txt` | MODIFIED | + `pyahocorasick` |
| `data/snomed_clinical_trials.csv` | UNCHANGED | — |
| `data/geo_canonical.json` | UNCHANGED | — |

---

## 3. Component specifications

### 3.1 `data/ambiguous_terms.json` (NEW)

**Purpose:** Single source of truth for which broad terms trigger clarification, what options to show, and which specific forms override the trigger.

**Schema:**
```json
{
  "<lowercase_trigger_term>": {
    "category": "indication" | "phase" | "geography" | "population",
    "question_template": "Which type of {trigger} are you looking for?",
    "options": ["Lung Cancer", "Breast Cancer", "Colorectal Cancer", "Lymphoma", "Melanoma"],
    "manual_override_terms": ["...optional manual entries..."],
    "max_options": 5
  }
}
```

**Constraints:**
- 3 ≤ `len(options)` ≤ 5
- Every `option` MUST resolve via the configured `SNOMEDSearchStrategy` with confidence ≥ 0.85 at startup. See **Validator failure policy** below.
- `manual_override_terms` is OPTIONAL. Auto-derivation handles common cases; manual entries are for edge cases (e.g., archaic synonyms).
- `question_template` supports `{trigger}` and `{prior_filters}` placeholders. `{prior_filters}` is filled from `ConversationSession.summarize_known_filters()`. If empty, the substring degrades gracefully (template should be written so empty-string substitution still reads correctly).
- Trigger keys are matched **case-insensitive whole-word** via `re.compile(r'\b<trigger>\b', re.IGNORECASE)`. Special regex chars in trigger keys are escaped via `re.escape()` in the loader.

**Override-terms auto-derivation algorithm (§3.1 — exact spec):**

Given a trigger `T` (e.g., `"cancer"`):
1. Start with `auto_overrides = set(opt.lower() for opt in entry.options)` — every clarification option becomes an override (e.g., `"Lung Cancer"` → `"lung cancer"`).
2. Scan `data/snomed_clinical_trials.csv`. For each row, take the `preferred_term` (lowercased). If it contains `T` as a **whole-word substring** (token-level match — `"cancer"` matches `"lung cancer"` and `"non-small cell lung cancer"` but NOT `"carcinoma"`), add it to `auto_overrides`.
3. **Critical filter:** remove any override that equals the trigger itself, case-insensitive: `auto_overrides.discard(T.lower())`. This prevents the **self-defeat bug** where a CSV row `preferred_term="cancer"` would make the bare query `"cancer"` count as its own override and skip clarification.
4. Synonyms from the CSV's `synonyms` column are NOT auto-derived (synonyms tend to be too liberal and produce false negatives). Use `manual_override_terms` for synonym-driven overrides.
5. `final_overrides = auto_overrides | set(opt.lower() for opt in entry.manual_override_terms)`.

**Initial seed (port from Arturo's table):**
- `cancer`, `tumor`, `oncology` → Lung / Breast / Colorectal / Lymphoma / Melanoma
- `diabetes`, `diabetic` → Type 1 / Type 2 / Both
- `heart`, `cardiac`, `cardiovascular` → Heart Failure / AFib / MI / Hypertension / CAD
- `arthritis` → Rheumatoid / Psoriatic / Ankylosing
- `autoimmune` → Lupus / RA / MS / Crohn's / Psoriasis
- `neurological`, `neuro`, `brain` → Alzheimer's / Parkinson's / MS / Epilepsy / ALS
- `liver`, `hepatic` → Hepatitis B / Hepatitis C / NASH
- `lung disease`, `pulmonary` → Lung Cancer / COPD / Asthma / Pulmonary Fibrosis
- `depression` → MDD / Bipolar Disorder / Anxiety
- `infection` → HIV / Hepatitis / Bacterial Infection / Other

**Loader/validator pseudocode:**
```python
class AmbiguousEntry:
    """Pydantic V2 frozen model. Built by the registry at load time."""
    trigger: str                          # lowercase canonical key
    category: str
    question_template: str
    options: list[str]
    override_terms: frozenset[str]        # final, post-derivation
    max_options: int

class AmbiguousTermsRegistry:
    """Init-time validator. Strategy is REQUIRED at construction time —
    options are validated by resolving them through the strategy."""

    def __init__(
        self,
        path: str,
        snomed_strategy: SNOMEDSearchStrategy,   # required, not Optional
        snomed_csv_path: str,
        strict_validation: bool = True,
    ) -> None:
        # strict_validation=True (default): invalid options → ERROR + sys.exit(1)
        # strict_validation=False: invalid options → WARN; entry is dropped from registry,
        #   gate behaves as if the trigger doesn't exist (graceful degrade for rolling deploys)
        self._strategy = snomed_strategy
        self._csv_path = snomed_csv_path
        raw = json.load(open(path))
        self._entries: dict[str, AmbiguousEntry] = {}
        invalid_count = 0
        for trigger, data in raw.items():
            try:
                entry = self._validate_and_build(trigger.lower(), data)
                self._entries[trigger.lower()] = entry
            except ValueError as e:
                invalid_count += 1
                if strict_validation:
                    logger.error(f"ambiguous_terms.json: trigger '{trigger}' invalid: {e}")
                    sys.exit(1)
                else:
                    logger.warning(f"ambiguous_terms.json: trigger '{trigger}' skipped: {e}")
        if not self._entries:
            raise ValueError("AmbiguousTermsRegistry: no valid entries after validation")
        self._compiled_triggers = re.compile(
            r'\b(' + '|'.join(re.escape(t) for t in self._entries) + r')\b',
            re.IGNORECASE,
        )

    def _validate_and_build(self, trigger: str, data: dict) -> AmbiguousEntry:
        # 1. Validate every option resolves ≥ 0.85
        for opt in data['options']:
            matches = self._strategy.search(opt)
            if not matches or max(m.confidence for m in matches) < 0.85:
                max_conf = max((m.confidence for m in matches), default=0.0)
                raise ValueError(
                    f"option '{opt}' did not resolve via {self._strategy.name} "
                    f"(max confidence: {max_conf})"
                )
        # 2. Auto-derive overrides per algorithm above
        overrides = self._derive_overrides(trigger, data['options'], data.get('manual_override_terms', []))
        return AmbiguousEntry(
            trigger=trigger, category=data['category'],
            question_template=data['question_template'],
            options=data['options'], override_terms=frozenset(overrides),
            max_options=data.get('max_options', 5),
        )

    def find_trigger(self, query: str) -> Optional[tuple[str, AmbiguousEntry]]:
        """Returns (trigger, entry) for the first trigger NOT overridden in the query.
        Returns None if no trigger fires.
        Iteration order = insertion order = JSON key order (for determinism)."""
        # find ALL trigger matches; return the first whose overrides aren't present
        for m in self._compiled_triggers.finditer(query.lower()):
            trigger = m.group(1).lower()
            entry = self._entries[trigger]
            override_present = any(
                re.search(rf'\b{re.escape(o)}\b', query, re.IGNORECASE)
                for o in entry.override_terms
            )
            if not override_present:
                return (trigger, entry)
        return None
```

**Validator failure policy (Security finding addressed):**

The default `strict_validation=True` is appropriate for **POC and test environments** — fail-fast catches drift in CI. For **production / multi-replica deployments**, set `strict_validation=False` so a CSV/JSON drift during a rolling deploy degrades gracefully (affected triggers are skipped, gate falls through to default-sufficient) instead of failing liveness probes. Documented behavior, env-var overridable: `AMBIG_STRICT_VALIDATION=false`.

**Operational pre-flight:** A separate `tests/validate_ambiguous_terms.py` script runs strict validation in CI; production servers run lenient. Pre-flight catches drift before deploy.

**Dependencies:** `re`, `json`, `sys`, `logger`, `pydantic`. No LLM. Reads `SNOMEDSearchStrategy` for option validation only.

**Non-responsibilities:**
- Does NOT generate questions dynamically (templates only).
- Does NOT call any LLM.
- Does NOT decide max-turns logic.

**HIPAA logging spec:**
- LOGS at startup: total entries loaded, total override_terms derived, validator pass/fail counts, strategy name used.
- NEVER LOGS: query text, matched trigger value at runtime.

---

### 3.2 `src/sufficiency_gate.py` (NEW)

**Purpose:** Deterministic pre-extraction decision + post-extraction safety check.

**Pydantic V2 models:**
```python
class SufficiencyDecision(BaseModel):
    model_config = ConfigDict(frozen=True)
    sufficient: bool
    reason: str                                  # internal log enum, never user-facing
    triggered_by: Optional[str] = None
    matched_entry: Optional[AmbiguousEntry] = None
```

**Reason enum values (for unambiguous logging):**
- `"ok_no_trigger"` — pre-extraction gate passed
- `"ambiguous_trigger"` — pre-extraction gate fired
- `"max_turns_reached"` — escape valve forced sufficient=True
- `"ok_post_extraction"` — post-extraction check passed
- `"filters_without_condition"` — post-extraction check fired

**DEFAULT_CONDITION_PROMPT (module-level constant — addresses QA blocker):**
```python
DEFAULT_CONDITION_PROMPT = AmbiguousEntry(
    trigger="__default_condition__",
    category="indication",
    question_template="What medical condition or area of research are you interested in?",
    options=[
        # Sourced at module load from top-5 most common SNOMED categories in the CSV.
        # Hard fallback if CSV inspection fails: ["Cancer", "Diabetes", "Heart Disease",
        # "Autoimmune", "Other"]. The actual list is built once at startup.
    ],
    override_terms=frozenset(),
    max_options=5,
)
```

The `options` list is computed once at import time by scanning `snomed_clinical_trials.csv`'s preferred_terms and clustering them by the seed-list categories from `ambiguous_terms.json` (cancer / diabetes / heart / etc.). Hard fallback is hardcoded so the module always loads.

**Class signature:**
```python
class SufficiencyGate:
    def __init__(self, registry: AmbiguousTermsRegistry) -> None:
        self._registry = registry

    def evaluate(
        self,
        canonical_query: str,
        session: ConversationSession,
    ) -> SufficiencyDecision: ...

    def post_extraction_check(
        self,
        snomed_matches: list[SNOMEDMatch],
        filters: ExtractedFilters,
    ) -> SufficiencyDecision: ...
```

**Pre-extraction decision algorithm (only 3 rules — no rule 4):**
```python
def evaluate(self, query: str, session: ConversationSession) -> SufficiencyDecision:
    # Rule 1: max-turns escape valve.
    # Counter is owned by ConversationSession (single source of truth).
    if session.is_max_turns_reached():
        return SufficiencyDecision(sufficient=True, reason="max_turns_reached")

    # Rule 2: registry trigger lookup (with override-term filter).
    hit = self._registry.find_trigger(query)
    if hit is not None:
        trigger, entry = hit
        return SufficiencyDecision(
            sufficient=False,
            reason="ambiguous_trigger",
            triggered_by=trigger,
            matched_entry=entry,
        )

    # Rule 3: default — sufficient. (No "rule 4" here. The "filters but no condition"
    # case is handled by post_extraction_check, AFTER the parallel paths run.
    # That ordering is intentional: we only know there are 0 SNOMED matches after
    # the algorithmic search runs, which is cheap.)
    return SufficiencyDecision(sufficient=True, reason="ok_no_trigger")
```

**Post-extraction safety check:**
```python
def post_extraction_check(
    self,
    snomed_matches: list[SNOMEDMatch],
    filters: ExtractedFilters,
) -> SufficiencyDecision:
    # Discard SNOMED matches below MIN_CONFIDENCE (0.60) for this purpose only;
    # assembler also filters but with its own threshold.
    high_conf_matches = [m for m in snomed_matches if m.confidence >= 0.60 and not m.negated]
    if not high_conf_matches and any_filter_set(filters):
        return SufficiencyDecision(
            sufficient=False,
            reason="filters_without_condition",
            triggered_by=None,
            matched_entry=DEFAULT_CONDITION_PROMPT,
        )
    return SufficiencyDecision(sufficient=True, reason="ok_post_extraction")
```

**Dependencies:** `AmbiguousTermsRegistry`, `ConversationSession`, `SNOMEDMatch`. No LLM.

**Non-responsibilities:**
- Does NOT call the LLM.
- Does NOT call `SNOMEDSearchStrategy` directly (registry validates options at startup).
- Does NOT mutate session state.
- Does NOT perform clinical-intent gating (separate; lives in pipeline.py for backwards compat with current behavior).

**HIPAA logging spec:**
- LOGS: `sufficient` (bool), `reason` (enum string).
- NEVER LOGS: `canonical_query`, `triggered_by` value, `matched_entry.options` content.

---

### 3.3 `src/conversation.py` (NEW — refined from `modified-proposal.md` §3.3)

**Purpose:** Multi-turn state. `st.session_state` today, Redis-ready tomorrow.

**Pydantic V2 models — Optional fields explicit (addresses QA major):**
```python
class Turn(BaseModel):
    model_config = ConfigDict(frozen=True)
    turn_index: int
    user_input: str
    canonical_query: str
    decision: Optional[SufficiencyDecision] = None
    filters: Optional[ExtractedFilters] = None      # None for clarification turns
    snomed_matches: list[SNOMEDMatch] = []          # empty for clarification turns
    geo: Optional[GeoResult] = None                 # None for clarification turns
    timestamp: float

class ConversationSession(BaseModel):
    model_config = ConfigDict(frozen=False)
    session_id: str
    turns: list[Turn] = []
    canonical_query: str = ""
    max_clarification_turns: int = 3
    created_at: float
```

**Methods:**
```python
ConversationSession.new() -> ConversationSession                    # uuid4 hex, timestamp
ConversationSession.append_turn(turn: Turn) -> None
ConversationSession.clarification_turn_count() -> int
    # Counts turns where decision is not None AND decision.sufficient == False.
    # SINGLE SOURCE OF TRUTH for the counter — no other code increments separately.
ConversationSession.is_max_turns_reached() -> bool
    # True iff clarification_turn_count() >= max_clarification_turns.
ConversationSession.last_clarification() -> Optional[SufficiencyDecision]
    # Returns turns[-1].decision if it exists and was not sufficient, else None.
ConversationSession.update_canonical_query(user_input: str) -> str
ConversationSession.summarize_known_filters() -> str
    # Returns "Boston, Phase 3" or "" — for question template substitution.
    # Reads turns[-1].filters if available; ignores filter values from non-search turns.
ConversationSession.to_dict() -> dict
ConversationSession.summary_for_logging() -> dict
    # SAFE for logging. Returns ONLY: session_id, turn_count,
    # clarification_count, max_turns, created_at. NEVER includes
    # canonical_query, user_input, filter values, or SNOMED display strings.

@classmethod
ConversationSession.from_dict(cls, d: dict, on_error: str = "raise") -> "ConversationSession":
    # on_error: "raise" (default — pydantic.ValidationError propagates)
    # on_error="new_session": catches ValidationError, returns ConversationSession.new()
    #   and logs WARN with corruption indicator. Used in production session-load paths.
```

**Canonical query construction (substitute-or-append):**
```python
def update_canonical_query(self, user_input: str) -> str:
    if not self.turns:
        self.canonical_query = user_input
        return self.canonical_query

    last = self.turns[-1]
    last_decision = last.decision

    # Are we answering a clarification?
    if last_decision and not last_decision.sufficient and last_decision.triggered_by:
        trigger = last_decision.triggered_by
        # Substitute trigger with new input (case-insensitive whole-word).
        # NOTE: there is NO LLM-extracted "medical_terms" array in this design.
        # The trigger comes directly from ambiguous_terms.json and is matched against
        # the canonical query that the USER literally typed (post-preprocessor).
        # No normalization step exists between user input and trigger detection,
        # so substitution is well-defined.
        pattern = re.compile(rf'\b{re.escape(trigger)}\b', re.IGNORECASE)
        if pattern.search(self.canonical_query):
            self.canonical_query = pattern.sub(user_input, self.canonical_query)
        else:
            # Defensive fallback: trigger not literally present (rare —
            # could happen if user typed e.g. "cancers" and trigger key is "cancer";
            # whole-word \b doesn't match plurals).
            # Append → next turn's gate re-evaluates.
            self.canonical_query = f"{self.canonical_query} {user_input}".strip()
    else:
        # Free-text follow-up or post-extraction safety clarification → append.
        # (The post-extraction safety case has triggered_by=None.)
        self.canonical_query = f"{self.canonical_query} {user_input}".strip()

    return self.canonical_query
```

**Why no preprocessor re-validation here:**
- `update_canonical_query()` only ever combines strings that have ALREADY been preprocessed (raw user_input passes through preprocessor at the top of `pipeline.run_with_session()` before reaching here).
- Concatenation of two safe strings can only produce patterns that are unions of safe patterns. No regex-level pattern injection arises from concatenation.
- However, as **defense in depth**, the new canonical query IS re-checked by the preprocessor at the start of the NEXT turn (because every turn's user_input → preprocessor → update_canonical_query). And one additional defense is added: `pipeline.run_with_session()` runs the preprocessor's injection-pattern check (only) against the fully-merged canonical query before extraction (see §3.8). This catches any pathological substitution edge cases.

**Serialization:**
```python
def to_dict(self) -> dict:
    return self.model_dump(mode="json")

@classmethod
def from_dict(cls, d: dict, on_error: str = "raise") -> "ConversationSession":
    try:
        return cls.model_validate(d)
    except ValidationError as e:
        if on_error == "new_session":
            logger.warning(f"ConversationSession.from_dict: corrupted state, starting fresh ({type(e).__name__})")
            return cls.new()
        raise
# Round-trip: from_dict(s.to_dict()) == s   (must hold in tests)
```

**Dependencies:** `pydantic`, `re`, `uuid`, `time`, `logger`. No LLM.

**Non-responsibilities:**
- Does NOT decide sufficiency.
- Does NOT call LLM or SNOMED.
- Does NOT enforce rate limits (`app.py`).

**HIPAA logging spec:**
- LOGS: `session_id`, `turn_index`, `clarification_turn_count`, timestamps, on_error recovery events.
- NEVER LOGS: `user_input`, `canonical_query`, filter values, `to_dict()` output.
- **Convention:** any log statement involving session state must use `session.summary_for_logging()`, never `session.to_dict()` or `repr(session)`. Reviewers should grep for `session.to_dict()` in log lines.

---

### 3.4 `src/exceptions.py` (NEW — addresses QA major)

```python
class LLMProviderError(Exception):
    """Raised when the LLM provider encounters an unrecoverable failure
    (after retry budget exhausted, or for permanent errors like invalid API key)."""

    def __init__(
        self,
        message: str,
        provider_name: str,
        original_error: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.provider_name = provider_name
        self.original_error = original_error

    def __str__(self) -> str:
        # User-safe representation; never includes original_error details.
        return f"LLM provider '{self.provider_name}' unavailable"

class PipelineError(Exception):
    """Raised by the pipeline for orchestration failures
    (extraction timeout, parallel-path coordination failure)."""
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code   # enum: "extraction_timeout" | "strategy_unavailable" | ...

class StrategyError(Exception):
    """Raised by SNOMED strategies for init failures (missing dictionary, etc.)."""
    pass
```

---

### 3.5 `src/llm_provider/` (NEW PACKAGE)

**Purpose:** Provider abstraction.

**`base.py`:**
```python
class LLMProvider(Protocol):
    """Every provider implements this. Pipeline depends only on this Protocol."""

    @property
    def name(self) -> str: ...   # "groq" / "claude" / etc.

    @property
    def model_id(self) -> str: ...

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        timeout: float = 10.0,
        json_mode: bool = True,
    ) -> str: ...
    # Returns raw response text. Provider handles retries internally per the contract below.
    # Raises LLMProviderError on unrecoverable failure (retry budget exhausted, or permanent).
```

**Retry contract — every provider implementation MUST honor:**
- Transient errors (retry up to 3 attempts with exponential backoff `[1s, 2s, 4s]`):
  - Network timeout
  - Rate limit (HTTP 429)
  - Server errors (5xx)
- Permanent errors (raised immediately as `LLMProviderError`, no retry):
  - Authentication failure (401)
  - Model not found (404)
  - Malformed request (400)
- After exhausted retries on transient errors: `LLMProviderError` with `original_error` set.
- The `timeout` parameter is the per-attempt timeout (NOT total). Total wall time = `timeout × attempts + backoff_sum`.

**`groq_provider.py` — implementation:**
```python
import groq

class GroqProvider:
    name = "groq"

    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant") -> None:
        self._client = groq.Groq(api_key=api_key)
        self._model = model

    @property
    def model_id(self) -> str:
        return self._model

    def complete(self, system_prompt, user_prompt, max_tokens=512, temperature=0.0,
                 timeout=10.0, json_mode=True) -> str:
        last_error: Optional[Exception] = None
        for attempt, backoff in enumerate([0, 1, 2, 4]):
            if backoff:
                time.sleep(backoff)
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_format={"type": "json_object"} if json_mode else None,
                    timeout=timeout,
                )
                return response.choices[0].message.content
            except (groq.RateLimitError, groq.APITimeoutError, groq.APIConnectionError) as e:
                last_error = e   # transient — retry
                continue
            except (groq.AuthenticationError, groq.NotFoundError, groq.BadRequestError) as e:
                # Permanent — no retry
                raise LLMProviderError(str(e), provider_name=self.name, original_error=e)
        raise LLMProviderError("retry budget exhausted", provider_name=self.name, original_error=last_error)
```

**`registry.py` — factory:**
```python
PROVIDER_REGISTRY: dict[str, type[LLMProvider]] = {
    "groq": GroqProvider,
}
DEFAULT_PROVIDER = "groq"

def get_provider(name: Optional[str] = None, **kwargs) -> LLMProvider:
    name = name or os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER)
    if name not in PROVIDER_REGISTRY:
        raise ValueError(f"Unknown provider: {name!r}. Available: {sorted(PROVIDER_REGISTRY)}")
    return PROVIDER_REGISTRY[name](**kwargs)
```

**Dependencies:** `groq` (only inside `groq_provider.py`).

**Non-responsibilities:**
- Does NOT parse JSON.
- Does NOT know about prompts (just transports).

**HIPAA logging spec:**
- LOGS: provider name, model_id, attempt count, response length, latency.
- NEVER LOGS: `system_prompt`, `user_prompt`, response text, `original_error.message`.

---

### 3.6 `src/filter_extractor.py` (NEW)

**Purpose:** Replaces `extractor.py` for filter-only extraction.

**Pydantic V2 models:**
```python
class FilterField(BaseModel):
    model_config = ConfigDict(frozen=True)
    value: Optional[str]
    confidence: float

class StateFilter(BaseModel):
    model_config = ConfigDict(frozen=True)
    values: list[str]                  # CRITICAL: list, not single value
    confidence: float
    is_region: bool

class ExtractedFilters(BaseModel):
    model_config = ConfigDict(frozen=True)
    investigator_name: FilterField
    site_name: FilterField
    city: FilterField
    state: StateFilter
    phase: FilterField
    raw_response_length: int
```

**System prompt — full spec (addresses Security major: must include disambiguation rules from existing extractor):**
```
You extract structured filter fields from clinical research queries.
Return JSON only. Fields:
- investigator_name (string or null)
- site_name (string or null)
- city (string or null)
- state (string or null) — US state or Canadian province name; null for multi-state regions
- phase (string or null) — normalized
- per-field confidence (float 0.0–1.0)

CRITICAL RULES (do not skip):

1. Do NOT extract medical conditions or diseases. They are handled separately. If the
   query contains medical terms, ignore them — return null for any field they don't fit.

2. State extraction — explicit only:
   - Extract state ONLY from explicit state mentions ("California", "in TX", "Texas trials").
   - Do NOT infer state from a city name. "Kansas City" is in Missouri, NOT Kansas.
     "Oklahoma City" is in Oklahoma but you do not know that here — leave state=null.
     "New York" alone is the city — return city="New York", state=null. The downstream
     geo normalizer resolves this.
   - Do NOT infer state from an institution name. "Massachusetts General Hospital"
     does not imply state=Massachusetts when a different city is specified.
   - For multi-state regions (e.g., "east coast", "midwest"): state=null. The geo
     normalizer expands these.

3. Phase normalization (return EXACTLY one of these forms):
   - "phase iii", "p3", "phase-3", "pivotal" → "Phase 3"
   - "first in human", "fih" → "Phase 1"
   - "phase 1/2", "p1/2", "phase i/ii" → "Phase 1/2"
   - "phase 2b" → "Phase 2b"
   - Generic phases: "Phase 1", "Phase 2", "Phase 3", "Phase 4"

4. Investigator names: lastname-only is acceptable. Title prefixes (Dr., Prof.) stripped.

5. Site names: full institution name as written. Do NOT split city out of site name.

6. Confidence scoring: high (0.9+) for explicit unambiguous mentions. Lower for inferred.
   Null fields have confidence 0.0.

Output JSON schema (all fields required, value may be null):
{
  "investigator_name": {"value": ..., "confidence": ...},
  "site_name":         {"value": ..., "confidence": ...},
  "city":              {"value": ..., "confidence": ...},
  "state":             {"value": ..., "confidence": ...},
  "phase":             {"value": ..., "confidence": ...}
}
```

**Class signature:**
```python
class FilterExtractor:
    SYSTEM_PROMPT: ClassVar[str] = "<above>"

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    def extract(self, canonical_query: str) -> ExtractedFilters:
        raw = self._provider.complete(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=canonical_query,
            max_tokens=512,
            temperature=0.0,
            json_mode=True,
        )
        # Parse JSON; raise ExtractionError on failure with generic user-safe message
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ExtractionError("Could not parse extraction response") from e
        return self._validate(parsed, raw_response_length=len(raw))

    def _validate(self, parsed: dict, raw_response_length: int) -> ExtractedFilters:
        # Sanitize "null" string → None (existing post-build fix preserved)
        # If LLM accidentally includes medical_terms, drop silently.
        # Build StateFilter (single-state from LLM goes into values=[state_name],
        #   is_region=False); multi-state regions arrive via geo normalizer downstream.
        # ...
```

**Parity gate (Security finding):** before merge, batch-evaluate the new `FilterExtractor` against the current `Extractor` on `tests/test_cases.json` (20 cases) — all 20 cases must produce identical filter values. The Kansas City test case is mandatory.

**Dependencies:** `LLMProvider`, `json`, `pydantic`. No groq import.

**Non-responsibilities:**
- Does NOT extract SNOMED concepts.
- Does NOT detect negation.
- Does NOT normalize geo.
- Does NOT decide sufficiency.

**HIPAA logging spec:**
- LOGS: provider, attempts, response length, parse success bool, per-field confidence values.
- NEVER LOGS: query, response text, field values.

---

### 3.7 `src/snomed_search/` (NEW PACKAGE) — KEYSTONE

#### 3.7.1 `base.py` — `SNOMEDSearchStrategy` Protocol + `SNOMEDMatch`

**`SNOMEDMatch` — `span` is REQUIRED (addresses QA blocker):**
```python
@dataclass(frozen=True)
class SNOMEDMatch:
    code: str
    display: str
    match_type: str               # "exact" | "synonym" | "fuzzy" | "semantic" | "ngram"
    confidence: float
    original_text: str
    span: tuple[int, int]         # REQUIRED — (start_char, end_char) offsets in the
                                  # query passed to search(). NegEx requires these.
    negated: bool = False         # populated by NegationAnnotator (default False)
```

**Validation:** `__post_init__` raises `ValueError` if `span[0] < 0`, `span[1] <= span[0]`, or if `span` is `None`. Strategies cannot return matches without spans.

**`SNOMEDSearchStrategy` Protocol:**
```python
class SNOMEDSearchStrategy(Protocol):
    """Every SNOMED search algorithm implements this Protocol.
    Implementations are drop-in replaceable; the pipeline depends ONLY on this Protocol.

    Contract (must hold for every implementation):
    - Read-only after __init__. No state mutation in search() or anywhere else.
      Thread-safe by construction; multiple concurrent search() calls are valid.
    - Deterministic given fixed dictionary + same query. No randomness.
    - Must populate SNOMEDMatch.span with valid char offsets.
    - Returns ALL candidates from the underlying algorithm; does NOT pre-filter
      by MIN_CONFIDENCE (caller filters in assembler).
    - Resource initialization happens in __init__; no lazy loading. No shared mutable
      state across instances. External resources (DB connections, network sockets)
      should use connection pools owned by the strategy.
    - search() must be free of catastrophic backtracking. No regex with unbounded
      quantifiers on user input. Strategies that use regex MUST anchor and bound them.
    - Implementations document their candidate-extraction approach in search()'s
      docstring (see "Candidate extraction" below).
    """

    @property
    def name(self) -> str: ...

    def __init__(self, dictionary_path: str, **kwargs) -> None: ...

    def search(self, query: str) -> list[SNOMEDMatch]: ...

    def health_check(self) -> dict: ...
        # Returns {"ready": bool, "dictionary_size": int, "name": str, ...}
        # Called at startup; ready=False → StrategyError.
```

**Candidate extraction — interface contract (addresses QA blocker):**

Each strategy chooses HOW to find candidate spans within the query (n-gram windows, full-substring scan, embedding nearest-neighbor on tokens). The strategy MUST document its approach in its `search()` docstring. **The parity test in §10 compares the set of TOP-5 SNOMED CODES per strategy** (not the full match set, not match_type, not exact confidence) — this gives strategies room to differ on candidate extraction without failing parity.

#### 3.7.2 `aho_corasick.py`

```python
class AhoCorasickStrategy:
    """Aho-Corasick exact + synonym match. Microsecond-scale.
    Candidate extraction: scans every substring position; word-boundary post-filter."""

    name = "aho_corasick"

    def __init__(self, dictionary_path: str, **kwargs) -> None:
        self._automaton = ahocorasick.Automaton()
        self._build(dictionary_path)
        self._ready = True

    def _build(self, path) -> None:
        # Read snomed_clinical_trials.csv
        # For each row: insert preferred_term and each synonym
        # Value: (concept_id, display, match_type)
        # Confidence: 0.99 for exact, 0.97 for synonym
        # ...
        self._automaton.make_automaton()

    def search(self, query: str) -> list[SNOMEDMatch]:
        """Candidate extraction: substring iteration via Aho-Corasick automaton.
        Word-boundary post-filter prevents 'cancer' matching inside 'pancreatic'."""
        q = query.lower()
        raw_hits = []
        for end_inclusive, (concept_id, display, mtype, term_len) in self._automaton.iter(q):
            start = end_inclusive - term_len + 1
            end_exclusive = end_inclusive + 1
            if not self._is_word_bounded(q, start, end_exclusive):
                continue
            confidence = 0.99 if mtype == "exact" else 0.97
            raw_hits.append(SNOMEDMatch(
                code=concept_id, display=display, match_type=mtype,
                confidence=confidence, original_text=q[start:end_exclusive],
                span=(start, end_exclusive), negated=False,
            ))
        return self._dedup_longest_match(raw_hits)

    def _is_word_bounded(self, text: str, start: int, end: int) -> bool:
        if start > 0 and text[start - 1].isalnum():
            return False
        if end < len(text) and text[end].isalnum():
            return False
        return True

    def _dedup_longest_match(self, hits: list[SNOMEDMatch]) -> list[SNOMEDMatch]:
        """Sort by span length DESC, then drop any later hit whose span is
        entirely contained in (or equal to) an earlier kept hit. Breaks ties
        by higher confidence, then by earlier span start."""
        # ...

    def health_check(self) -> dict:
        return {"ready": self._ready, "name": self.name, "dictionary_size": len(self._automaton)}
```

**Dependencies:** `pyahocorasick` (justified — only well-maintained Aho-Corasick library for Python).

#### 3.7.3 `ngram_lookup.py`

```python
class NGramLookupStrategy:
    """Token n-gram window lookup. Stdlib only.
    Candidate extraction: enumerate all 1-to-max_n-token contiguous windows;
    look each up in exact_index then synonym_index."""

    name = "ngram_lookup"

    def __init__(self, dictionary_path: str, max_n: int = 4, **kwargs) -> None:
        self._exact_index: dict[str, tuple[str, str]] = {}
        self._synonym_index: dict[str, tuple[str, str]] = {}
        self._max_n = max_n
        self._build(dictionary_path)

    def search(self, query: str) -> list[SNOMEDMatch]:
        # Tokenize with offsets
        tokens_with_offsets = self._tokenize_with_offsets(query)
        hits = []
        for n in range(self._max_n, 0, -1):
            for i in range(len(tokens_with_offsets) - n + 1):
                window_tokens = tokens_with_offsets[i:i+n]
                window_str = " ".join(t.text for t in window_tokens).lower()
                start = window_tokens[0].start
                end = window_tokens[-1].end
                if window_str in self._exact_index:
                    code, display = self._exact_index[window_str]
                    hits.append(SNOMEDMatch(code=code, display=display, match_type="exact",
                                            confidence=0.99, original_text=window_str,
                                            span=(start, end), negated=False))
                elif window_str in self._synonym_index:
                    code, display = self._synonym_index[window_str]
                    hits.append(SNOMEDMatch(code=code, display=display, match_type="synonym",
                                            confidence=0.97, original_text=window_str,
                                            span=(start, end), negated=False))
        return self._dedup_longest_match(hits)
```

#### 3.7.4 `hybrid_cascade.py` — production default

```python
class HybridCascadeStrategy:
    """Hybrid cascade: exact → synonym → fuzzy → semantic.
    Candidate extraction: Stage 1+2 use n-gram windows (1..4 tokens).
    Stages 3+4 operate on residual UNMATCHED character spans."""

    name = "hybrid_cascade"

    def __init__(
        self,
        dictionary_path: str,
        fuzzy_cutoff: int = 88,
        semantic_threshold: float = 0.82,
        embedding_model: str = "all-MiniLM-L6-v2",
        **kwargs,
    ) -> None:
        # Build exact_index, synonym_index, alias_dict (port from existing snomed_resolver.py)
        # Build sentence-transformer encoder (cached)
        # Build chromadb collection if available, else numpy embedding matrix
        # ...

    def search(self, query: str) -> list[SNOMEDMatch]:
        hits: list[SNOMEDMatch] = []

        # Stage 1+2 — exact + synonym via n-gram windows
        hits.extend(self._exact_synonym_pass(query))

        # Stage 3 — fuzzy on residual unmatched character spans
        residual = self._compute_residual_spans(query, hits)
        hits.extend(self._fuzzy_pass(query, residual))

        # Stage 4 — semantic on still-residual spans
        residual = self._compute_residual_spans(query, hits)
        hits.extend(self._semantic_pass(query, residual))

        return self._dedup_longest_match(hits)

    def _compute_residual_spans(
        self,
        query: str,
        matched_so_far: list[SNOMEDMatch],
    ) -> list[tuple[int, int]]:
        """Returns list of (start, end) char spans NOT covered by any match.
        Algorithm:
        1. Sort matches by span start ascending.
        2. Walk through; for each match, residual = (cursor, match.start) if non-empty.
        3. Set cursor = max(cursor, match.end).
        4. After loop: residual = (cursor, len(query)) if non-empty.
        5. Filter: drop residuals shorter than 2 characters (noise).
        """
        if not matched_so_far:
            return [(0, len(query))]
        sorted_matches = sorted(matched_so_far, key=lambda m: m.span[0])
        residuals = []
        cursor = 0
        for m in sorted_matches:
            if m.span[0] > cursor:
                residuals.append((cursor, m.span[0]))
            cursor = max(cursor, m.span[1])
        if cursor < len(query):
            residuals.append((cursor, len(query)))
        return [(s, e) for (s, e) in residuals if e - s >= 2]

    def _fuzzy_pass(self, query, residual_spans) -> list[SNOMEDMatch]:
        # For each span: extract substring, run rapidfuzz token_sort_ratio
        # against all preferred_terms with score_cutoff = self._fuzzy_cutoff (88)
        # If match: confidence = score / 100
        # ...

    def _semantic_pass(self, query, residual_spans) -> list[SNOMEDMatch]:
        # For each span: encode via sentence-transformer
        # Query chromadb (or numpy cosine fallback)
        # Threshold: confidence ≥ self._semantic_threshold (0.82)
        # ...
```

**Why this is the default:** matches existing `snomed_resolver.py` behavior, so SNOMED recall doesn't regress on `tests/batch_test_cases.csv`.

#### 3.7.5 `negation.py` — NegEx-style annotator

```python
NEGATION_CUES_PRE = [
    "no", "not", "without", "denies", "denied", "rules out", "ruled out",
    "history of", "h/o", "free of", "absence of", "absent", "neither",
    "negative for", "no evidence of", "no signs of",
]
NEGATION_CUES_POST = ["unlikely", "ruled out", "negative", "denied"]
PSEUDO_NEGATION = [        # phrases containing negation-cue tokens that aren't actually negation
    "no contraindication for", "no change in", "no further", "not only",
]
SCAN_STOP_PUNCT = ".;!?\n"  # period, semicolon, question, exclamation, newline
WINDOW_SIZE = 5             # tokens

class NegationAnnotator:
    def annotate(self, query: str, matches: list[SNOMEDMatch]) -> list[SNOMEDMatch]:
        tokens = self._tokenize_with_offsets(query)
        return [replace(m, negated=self._is_negated(query, tokens, m)) for m in matches]

    def _is_negated(self, query, tokens, match) -> bool:
        # Find the token index of the match start
        match_token_idx = self._find_token_index(tokens, match.span[0])
        if match_token_idx is None:
            return False

        # Check pseudo-negation FIRST — pseudo-negation phrases overlapping the match
        # area suppress negation classification (e.g., "no contraindication for diabetes"
        # should NOT mark diabetes negated).
        if self._has_pseudo_negation_near(query, match):
            return False

        # Pre-window scan: tokens [match_token_idx - WINDOW_SIZE .. match_token_idx).
        # Stop scanning at any SCAN_STOP_PUNCT token boundary.
        # NOTE: comma is NOT a stop character — clinical syntax often uses commas
        # to chain conditions in the same clause. Period/semicolon/newline are the
        # sentence-level stops. Empirically calibrated against test_negation.py cases.
        if self._scan_window_for_cue(tokens, match_token_idx, NEGATION_CUES_PRE,
                                     direction="pre", window=WINDOW_SIZE,
                                     stop_chars=SCAN_STOP_PUNCT):
            return True

        # Post-window scan: tokens (match_token_idx + match_token_count .. + WINDOW_SIZE]
        match_end_token_idx = self._find_token_index(tokens, match.span[1] - 1) or match_token_idx
        if self._scan_window_for_cue(tokens, match_end_token_idx, NEGATION_CUES_POST,
                                     direction="post", window=WINDOW_SIZE,
                                     stop_chars=SCAN_STOP_PUNCT):
            return True

        return False

    def _scan_window_for_cue(self, tokens, anchor_idx, cues, direction, window, stop_chars):
        # If the match is near the start/end of the query, the window is naturally
        # truncated (we don't pad with synthetic tokens).
        # Stop scanning at stop_chars.
        # ...
```

**Comma handling — explicit decision:** comma does NOT stop the negation scan. Rationale: clinical syntax frequently chains conditions in the same clause (`"no history of cardiac issues, diabetes, or stroke"`) where all three are negated. False-positive risk is bounded by the 5-token window.

**Window truncation:** when a match is near the start or end of the query (fewer than `WINDOW_SIZE` tokens of context), the window is truncated to available tokens. No synthetic padding.

#### 3.7.6 `registry.py`

```python
STRATEGY_REGISTRY: dict[str, type[SNOMEDSearchStrategy]] = {
    "aho_corasick":   AhoCorasickStrategy,
    "ngram_lookup":   NGramLookupStrategy,
    "hybrid_cascade": HybridCascadeStrategy,
}
DEFAULT_STRATEGY = "hybrid_cascade"

def get_strategy(
    name: Optional[str] = None,
    dictionary_path: Optional[str] = None,
    **kwargs,
) -> SNOMEDSearchStrategy:
    name = name or os.environ.get("SNOMED_SEARCH_STRATEGY", DEFAULT_STRATEGY)
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {name!r}. Available: {sorted(STRATEGY_REGISTRY)}")
    path = dictionary_path or DEFAULT_DICT_PATH
    strategy = STRATEGY_REGISTRY[name](dictionary_path=path, **kwargs)
    health = strategy.health_check()
    if not health.get("ready"):
        raise StrategyError(f"Strategy {name!r} failed health check: {health}")
    return strategy
```

---

### 3.8 `src/normalizers/` (NEW PACKAGE)

```python
class FilterNormalizer(Protocol):
    @property
    def name(self) -> str: ...
    def normalize(self, raw_value: Any) -> NormalizedFilter: ...
```

`normalizers/geo.py` is `src/geo_normalizer.py` moved into the package — logic identical. `GeoResult.states: list[str]` schema preserved.

`src/geo_normalizer.py` becomes:
```python
# OBSOLETE-AT-SCALE: re-export shim — remove after callers migrate to normalizers.geo
from .normalizers.geo import GeoNormalizer, GeoResult  # noqa
```

---

### 3.9 `src/pipeline.py` (MODIFIED)

**Purpose:** Single entry point. Orchestrates parallel paths. Manages multi-turn loop.

**Pipeline class:**
```python
class NLPPipeline:
    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        snomed_strategy: Optional[SNOMEDSearchStrategy] = None,
        ambiguous_terms_path: Optional[str] = None,
        snomed_csv_path: Optional[str] = None,
        geo_json_path: Optional[str] = None,
        strict_validation: Optional[bool] = None,
    ) -> None:
        self._snomed = snomed_strategy or get_strategy(dictionary_path=snomed_csv_path)
        self._llm = llm_provider or get_provider()
        # Init order: strategy first (registry validates options against it).
        self._registry = AmbiguousTermsRegistry(
            path=ambiguous_terms_path or DEFAULT_AMBIG_PATH,
            snomed_strategy=self._snomed,
            snomed_csv_path=snomed_csv_path or DEFAULT_SNOMED_CSV,
            strict_validation=(
                strict_validation if strict_validation is not None
                else os.environ.get("AMBIG_STRICT_VALIDATION", "true").lower() == "true"
            ),
        )
        self._gate = SufficiencyGate(self._registry)
        self._extractor = FilterExtractor(self._llm)
        self._geo = GeoNormalizer(geo_json_path or DEFAULT_GEO_PATH)
        self._negation = NegationAnnotator()
        self._preprocessor = Preprocessor()
        self._assembler = ResponseAssembler()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="nlp-")

    def run_with_session(
        self,
        raw_query: str,
        session: ConversationSession,
    ) -> Union[NLPOutput, ClarificationOutput]: ...

    # OBSOLETE-AT-SCALE: single-turn entry, kept for tests/run_tests.py
    def run(self, raw_query: str) -> NLPOutput: ...
```

**`run_with_session` flow (post-revision, full ordering):**
```python
def run_with_session(self, raw_query, session) -> Union[NLPOutput, ClarificationOutput]:
    start = time.perf_counter()
    log_path = "unknown"   # one of the enum values below

    # 1. Preprocess raw user input (security gate, every turn)
    preprocessed = self._preprocessor.process(raw_query)
    # PreprocessorError propagates up; UI shows warning.
    # Session state NOT mutated yet — blocked input never poisons session.

    # 2. Update canonical query (substitute-or-append)
    canonical = session.update_canonical_query(preprocessed.text)

    # 2b. Defense-in-depth: re-run injection-pattern check on the merged canonical query.
    # This catches pathological substitutions (very unlikely, but cheap to check).
    self._preprocessor.assert_safe(canonical)   # raises PreprocessorError if patterns match

    # 3. Pre-extraction sufficiency gate
    decision = self._gate.evaluate(canonical, session)
    if not decision.sufficient:
        log_path = "sufficiency_clarification" if decision.reason == "ambiguous_trigger" else "max_turns"
        clarification = self._assembler.build_clarification(decision, session, start)
        # APPEND TURN ONLY AFTER successful build (no exceptions raised by assembler).
        session.append_turn(Turn(
            turn_index=len(session.turns),
            user_input=preprocessed.text,
            canonical_query=canonical,
            decision=decision,
            filters=None,
            snomed_matches=[],
            geo=None,
            timestamp=time.time(),
        ))
        self._log_turn(session, decision, log_path, start)
        return clarification

    # 4. PARALLEL — Path A (filter LLM) + Path B (algorithmic SNOMED)
    fut_filters = self._executor.submit(self._extractor.extract, canonical)
    fut_snomed = self._executor.submit(self._snomed.search, canonical)
    try:
        filters = fut_filters.result(timeout=15.0)
        snomed_matches = fut_snomed.result(timeout=15.0)
    except FuturesTimeout:
        # Both futures cancelled (in the ThreadPoolExecutor sense — submitted but not yet
        # started are cancelled; in-flight threads keep running until completion in the
        # background, but their results are discarded). Session state is NOT modified.
        # POC limitation: thread-pool exhaustion under sustained timeout DoS is documented in §5.
        fut_filters.cancel()
        fut_snomed.cancel()
        raise PipelineError("extraction_timeout")

    # 5. Negation annotation (deterministic, post-search)
    snomed_matches = self._negation.annotate(canonical, snomed_matches)

    # 6. Clinical-intent gate (existing, every turn)
    if not snomed_matches and not any_filter_set(filters):
        log_path = "clinical_intent"
        # Don't append turn — failed at security gate, similar to preprocessor failure.
        raise PreprocessorError("No clinical content found. Please enter a query about a "
                                "medical condition, investigator, research site, location, "
                                "or study phase.")

    # 7. Post-extraction safety check ("filters but no condition")
    post_decision = self._gate.post_extraction_check(snomed_matches, filters)
    if not post_decision.sufficient:
        log_path = "post_extraction_safety"
        clarification = self._assembler.build_clarification(post_decision, session, start)
        session.append_turn(Turn(
            turn_index=len(session.turns),
            user_input=preprocessed.text,
            canonical_query=canonical,
            decision=post_decision,
            filters=filters,
            snomed_matches=snomed_matches,
            geo=None,
            timestamp=time.time(),
        ))
        self._log_turn(session, post_decision, log_path, start)
        return clarification

    # 8. Geo normalization
    geo = self._geo.normalize(filters.city.value, _state_value_for_geo(filters.state))

    # 9. Assemble FIRST, append turn AFTER successful assembly (Security finding addressed)
    output = self._assembler.assemble(filters, snomed_matches, geo, start)

    # 10. Append turn only after successful assembly
    log_path = "search"
    session.append_turn(Turn(
        turn_index=len(session.turns),
        user_input=preprocessed.text,
        canonical_query=canonical,
        decision=decision,
        filters=filters,
        snomed_matches=snomed_matches,
        geo=geo,
        timestamp=time.time(),
    ))
    self._log_turn(session, decision, log_path, start)
    return output


def _state_value_for_geo(state_filter: StateFilter) -> Optional[str]:
    """Convert a StateFilter to a single state string for the legacy geo normalizer.
    For multi-state regions or empty values, returns None — geo normalizer handles
    region expansion based on city alone.
    For a single-state filter, returns the single value."""
    if state_filter.is_region:
        return None
    if len(state_filter.values) == 1:
        return state_filter.values[0]
    return None
```

**Per-turn log path enum (canonical, addresses Security blocker):**
- `"sufficiency_clarification"` — pre-extraction gate fired (ambiguous trigger)
- `"max_turns"` — escape valve forced sufficient=True
- `"clinical_intent"` — clinical-intent gate rejected (0 SNOMED + 0 filters)
- `"post_extraction_safety"` — filters present but no SNOMED matches → "what condition?"
- `"search"` — full pipeline completed; output is `NLPOutput`
- (preprocessor failure raises before reaching the path enum — logged separately)

**Per-turn log line spec (structured):**
```
JSON line:
{
  "ts": "2026-05-06T14:23:45Z",
  "session_id": "<uuid>",
  "turn_index": <n>,
  "clarification_count": <k>,
  "path": "<enum>",
  "decision_reason": "<sufficiency_decision_reason_enum>",
  "snomed_match_count": <n>,
  "snomed_strategy": "<name>",
  "filter_count": <n>,
  "llm_provider": "<name>",
  "processing_time_ms": <ms>
}
```

**NEVER logged:** `raw_query`, `canonical_query`, `user_input`, filter values, SNOMED display strings, LLM response, `triggered_by` value, option text. Reviewers grep log lines for any of these field names.

**Backwards-compat shim:**
```python
# OBSOLETE-AT-SCALE: kept for tests/run_tests.py and tests/batch_eval.py --legacy
def run(self, raw_query: str) -> NLPOutput:
    session = ConversationSession.new()
    result = self.run_with_session(raw_query, session)
    if isinstance(result, ClarificationOutput):
        return self._assemble_best_effort_from_session(session)
    return result
```

**Dependencies:** all components above, `concurrent.futures`, `time`, `os`.

**Non-responsibilities:**
- Does NOT manage rate limiting (`app.py`).
- Does NOT render UI.

---

### 3.10 `src/assembler.py` (MODIFIED)

**Purpose:** Build either `NLPOutput` (search) or `ClarificationOutput` (clarification turn). Discriminated by `type` field.

**Models (Pydantic V2):**
```python
class NLPOutput(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["search"] = "search"
    snomed_terms: list[SNOMEDTermOutput]
    filters: FiltersOutput
    metadata: MetadataOutput

class ClarificationOutput(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["clarification"] = "clarification"
    question: str                          # html.escape'd
    options: list[str]                     # each html.escape'd
    canonical_query: str                   # html.escape'd; for transparency
    turn_number: int                       # 1-indexed clarification turn
    max_turns: int = 3
    metadata: MetadataOutput

ResponseOutput = Annotated[
    Union[NLPOutput, ClarificationOutput],
    Field(discriminator="type"),
]
```

**Methods:**
```python
def render_question(
    entry: AmbiguousEntry,
    trigger: str,
    session: ConversationSession,
) -> str:
    """Module-level helper. Lives here (not in sufficiency_gate.py) because
    rendering is a presentation concern, not a decision concern."""
    prior_filters_str = session.summarize_known_filters()
    prefix = f"You're searching in {prior_filters_str} — " if prior_filters_str else ""
    template = entry.question_template
    return template.format(trigger=trigger, prior_filters=prefix).strip()

class ResponseAssembler:
    def build_clarification(
        self,
        decision: SufficiencyDecision,
        session: ConversationSession,
        start_time: float,
    ) -> ClarificationOutput:
        entry = decision.matched_entry  # not None — guaranteed by decision.sufficient=False
        trigger = decision.triggered_by or "your query"   # post_extraction_safety has no trigger
        question_raw = render_question(entry, trigger, session)
        return ClarificationOutput(
            question=html.escape(question_raw),
            options=[html.escape(o) for o in entry.options],
            canonical_query=html.escape(session.canonical_query),
            turn_number=session.clarification_turn_count() + 1,
            max_turns=session.max_clarification_turns,
            metadata=MetadataOutput(
                processing_time_ms=int((time.perf_counter() - start_time) * 1000),
                total_snomed_matches=0,
                snomed_match_types={},
                negated_terms_excluded=0,
            ),
        )

    def assemble(
        self,
        filters: ExtractedFilters,
        snomed_matches: list[SNOMEDMatch],
        geo: GeoResult,
        start_time: float,
    ) -> NLPOutput:
        # Filter SNOMED matches: confidence ≥ 0.60, not negated.
        # Sort by confidence DESC.
        # Dedup by code (keep highest confidence).
        # Build SNOMEDTermOutput list (each html.escape'd display).
        # Apply geo if confidence ≥ 0.60 (else fall back to raw filter values).
        # Build FiltersOutput with html.escape'd values.
        # Build metadata: counts by match_type, negated_terms_excluded count.
        # ...
```

**Sanitization invariant:** every user-rendered string passes through `html.escape()` before being placed in the output model. SNOMED display strings come from CSV (trusted) but are escaped defensively; clarification options come from JSON registry (trusted) but escaped defensively.

**Dependencies:** `pydantic`, `html`, `time`. No LLM. No SNOMED.

---

### 3.11 `app.py` (MODIFIED)

**Old structure:** single `st.text_area` → `pipeline.run(query)` → render. Marked `# OBSOLETE-AT-SCALE`.

**New structure — chat UI:**
```python
st.session_state["conversation"]       # ConversationSession instance
st.session_state["rate_limit_window"]  # existing — keep

# Render loop
1. Sidebar — PHI disclaimer + "New Search" button (resets conversation)
2. For each turn in session.turns:
     - st.chat_message("user"): render html.escape'd user_input
     - st.chat_message("assistant"):
         if turn.decision is None or turn.decision.sufficient:
             render results (existing logic)
         else:
             render clarification question + clickable buttons
3. If no terminal turn yet: st.chat_input() for next user message
```

**Clickable options:** `st.button(option_text, key=f"opt_{turn_index}_{i}")`. Treat label as next user input → re-run `pipeline.run_with_session()`.

**Rate limiting:** every turn (clarifications included) counts toward 5/60s and 30/session caps.

**Error rendering:** Errors render as assistant chat messages, not banners. Error log calls use `session.summary_for_logging()` only.

**XSS hardening:** `_safe()` (existing `html.escape` wrapper) on every user-input render and every LLM-derived display value.

---

### 3.12 `tests/`

**`tests/test_sufficiency_gate.py`** (NEW):
| Case | Input | Expected |
|---|---|---|
| Strong-match indication | "lung cancer phase 2" | sufficient |
| Bare ambiguous | "cancer" | NOT sufficient, options include "Lung Cancer" |
| Override present | "lung cancer in NYC" | sufficient |
| Override case-insensitive | "Lung Cancer in nyc" | sufficient |
| Filters but ambiguous | "cancer in Boston phase 3" | NOT sufficient |
| Multiple ambiguous | "heart and lung problems" | NOT sufficient, fires first match |
| Max-turns reached | session at 3 clarifications + "cancer" | sufficient (escape valve) |
| Partial-word non-match | "incancentive program" | sufficient |
| Self-defeat protection | trigger="cancer", CSV row preferred_term="cancer" → "cancer" alone NOT overridden | NOT sufficient |
| Empty query | "" | error path (Preprocessor catches first) |

**`tests/test_conversation.py`** (NEW):
| Case | Sequence | Expected |
|---|---|---|
| C1 | `["cancer"]` | turn 1 = clarification, options=["Lung Cancer", "Breast Cancer", ...] |
| C2 | `["cancer", "Lung Cancer"]` | turn 2 = search; SNOMED includes lung cancer code 254637007 (or whatever CSV has); confidence ≥ 0.95 |
| C3 | `["cancer in Boston phase 3", "Lung Cancer"]` | turn 2 = search; canonical = "Lung Cancer in Boston phase 3"; filters.city.value="Boston"; filters.phase.value="Phase 3"; original confidences preserved (extractor unchanged) |
| C4 | `["something something", "junk", "more junk", "still junk"]` | turn 4 = search via max-turns escape valve |
| C5 | `["lung cancer"]` | turn 1 = search (override fires) |
| C6 | `["cancer", "<malicious payload>"]` | turn 2 blocked by preprocessor; session retains turn 1 history |
| C7 | `["cancer", "Lung Cancer", "Lung Cancer", "Lung Cancer"]` | turn 4 forces search via max-turns |
| C8 | `["cancer"]` then `from_dict(to_dict())` | round-trip equality |
| C9 | Two sessions submitting "cancer" in parallel | no state leak |
| C10 | `["What's at Mayo?"]` | turn 1 → preprocessor passes → extractor runs → SNOMED finds 0 matches → post_extraction_safety fires → ClarificationOutput with question="What medical condition or area of research are you interested in?" and options=DEFAULT_CONDITION_PROMPT.options |
| C11 | `["cancers"]` (plural; trigger key is "cancer") | turn 1 = sufficient (whole-word `\bcancer\b` does NOT match "cancers") — known false negative; documented in §13 open risks |
| C12 | `from_dict(corrupted_dict, on_error="new_session")` | returns fresh session, logs WARN |
| C13 | Multi-turn substitution where canonical merge produces a string that triggers an injection pattern | preprocessor's `assert_safe()` on the merged canonical raises PreprocessorError before extraction |

Tests use `MockLLMProvider` and `MockSNOMEDStrategy` — no network.

**`tests/test_snomed_strategies.py`** (NEW):
- For each test case in `tests/batch_test_cases.csv`, run all 3 strategies and compare top-5 SNOMED codes (set Jaccard ≥ 0.80). At most one code may differ.
- `health_check()` returns ready=True after init for each.
- 50 concurrent `search()` calls per strategy → no errors, no race conditions.
- Each strategy's matches have valid `span` offsets (start ≥ 0, end > start, end ≤ len(query)).

**`tests/test_negation.py`** (NEW):
| Case | Query | Expected |
|---|---|---|
| Plain | "type 2 diabetes" | not negated |
| Pre-cue | "no history of diabetes" | negated |
| Pre-cue distance | "the patient has no history of significant cardiac events including diabetes" | depends on token count to "diabetes"; documented expected result based on WINDOW_SIZE=5 |
| Sentence boundary | "no cancer. Diabetes trials" | "diabetes" NOT negated |
| Comma chain (intentional pass-through) | "no history of cardiac issues, diabetes, stroke" | "diabetes" AND "stroke" both negated (comma doesn't stop scan) |
| Post-cue | "diabetes ruled out" | negated |
| Multiple matches | "no cancer but has diabetes" | "cancer" negated, "diabetes" not |
| Pseudo-negation | "no contraindication for diabetes trials" | "diabetes" NOT negated |
| Window edge | "diabetes" | not negated; window truncates without padding |

**`tests/test_llm_provider.py`** (NEW):
- `GroqProvider.complete()` retries 3 times on `RateLimitError`, succeeds.
- `GroqProvider.complete()` raises `LLMProviderError` with `original_error` set after 3 transient failures.
- `GroqProvider.complete()` raises `LLMProviderError` immediately on `AuthenticationError` (no retry).
- `MockLLMProvider` returns canned response.
- `get_provider("nonexistent")` raises `ValueError` listing available names.

**`tests/batch_eval.py`** (MODIFIED):
- New columns: `expected_type` (search | clarification, default search), `expected_clarification_field`, `expected_options_contain` (pipe-separated assertions).
- Multi-turn cases use `>>>` separator: `cancer>>>Lung Cancer`.
- New CLI flags: `--legacy`, `--strategy <name>`.
- New report metric: `clarification_precision`. When `--strategy` is iterated, prints comparative SNOMED recall by strategy.

**Parity gate (precise definition):**
- Run the 3 strategies (`hybrid_cascade`, `aho_corasick`, `ngram_lookup`) over `tests/batch_test_cases.csv`.
- For each case, take the top 5 codes by confidence from each strategy.
- Compute set Jaccard on codes (ignore confidence/match_type).
- A strategy passes parity if ≥80% of cases have Jaccard ≥ 0.80 against `hybrid_cascade`.
- `hybrid_cascade` is also compared against the existing `snomed_resolver.py` output on the 100 cases; recall (cases where ≥1 expected SNOMED code is in top-5) must not regress.

**`tests/run_tests.py`** (UNCHANGED): module docstring tagged `# OBSOLETE-AT-SCALE`.

---

## 4. End-to-end data flow (multi-turn)

```
Turn 0: User types "cancer in Boston phase 3"
        │
        ▼
Preprocessor.process()  — security · ~77 patterns · every turn
        │
        ▼
session.update_canonical_query()
  turn 0: canonical = "cancer in Boston phase 3"
        │
        ▼
preprocessor.assert_safe(canonical)   — defense-in-depth on merged string
        │
        ▼
SufficiencyGate.evaluate()  — DETERMINISTIC
  • is_max_turns_reached() → False
  • registry.find_trigger(...) → "cancer" matches; overrides absent
  → SufficiencyDecision(sufficient=False, reason="ambiguous_trigger",
                         triggered_by="cancer", matched_entry=...)
        │
        ▼
assembler.build_clarification(...)
        │
        ▼
session.append_turn(<clarification turn>)
        │
        ▼
ClarificationOutput → UI: "Which type of cancer?" [Lung] [Breast] [...]
log_path = "sufficiency_clarification"


Turn 1: User clicks "Lung Cancer"
        │
        ▼
Preprocessor.process("Lung Cancer")
        │
        ▼
session.update_canonical_query("Lung Cancer")
  • last decision.triggered_by = "cancer" → substitute
  • canonical = "Lung Cancer in Boston phase 3"
        │
        ▼
preprocessor.assert_safe(canonical)
        │
        ▼
SufficiencyGate.evaluate()
  • registry.find_trigger() → "cancer" matches BUT override "lung cancer" present → None
  → SufficiencyDecision(sufficient=True, reason="ok_no_trigger")
        │
        ▼
        ┌────────────┴────────────┐
        ▼                         ▼
   ┌────────────────────┐    ┌──────────────────────────┐
   │ Path A: LLM        │    │ Path B: Algorithmic      │
   │ FilterExtractor    │    │ SNOMEDSearchStrategy     │
   │  via LLMProvider   │    │ (configurable)           │
   │ → ExtractedFilters │    │ → list[SNOMEDMatch]      │
   │ (city=Boston,      │    │ ("lung cancer" → 0.99,   │
   │  phase=Phase 3)    │    │  span=(0,11))            │
   └─────────┬──────────┘    └────────────┬─────────────┘
             │                            │
             └─────────────┬──────────────┘
                           ▼  [JOIN — both futures completed within 15s]
                           ▼
NegationAnnotator.annotate()  — DETERMINISTIC
  → "Lung Cancer" not negated
                           │
                           ▼
Clinical-intent gate (existing)
  → snomed_count=1, filter_count=2 → passes
                           │
                           ▼
SufficiencyGate.post_extraction_check()
  → high-conf SNOMED match present → sufficient=True (reason="ok_post_extraction")
                           │
                           ▼
GeoNormalizer.normalize("Boston", _state_value_for_geo(state_filter))
  → city=Boston, states=["Massachusetts"], confidence=1.0
                           │
                           ▼
ResponseAssembler.assemble()  → NLPOutput(type="search", ...)
                           │
                           ▼
session.append_turn(<search turn>)   ← APPENDS AFTER successful assembly
log_path = "search"
                           │
                           ▼
return NLPOutput → UI renders results
```

---

## 5. Security model (per-turn)

Every turn MUST go through:

1. **Preprocessor on raw user_input** — ~77 patterns (unchanged).
2. **Preprocessor `assert_safe()` on merged canonical query** — defense-in-depth check that re-runs the injection-pattern subset (NOT length checks) on the post-substitution canonical query.
3. **Pre-extraction sufficiency gate** — runs BEFORE any LLM call.
4. **Clinical-intent gate** — existing every-turn check.
5. **Post-extraction safety check** — "filters but no condition".
6. **Rate-limit accounting** — 5/60s and 30/session caps (existing).
7. **Max-turns cap** — 3 per query, hard cap.
8. **Output sanitization** — `html.escape()` everywhere.

### Threats and mitigations

| Threat | Mitigation |
|---|---|
| Context-grinding across turns | Every turn re-runs preprocessor on user_input AND assert_safe on canonical |
| Free LLM calls via clarification loop | Max 3 clarifications + 5/60s + 30/session caps |
| Option-injection via JSON tampering | `ambiguous_terms.json` in-repo, version-controlled, validator at startup |
| Prompt-injection via LLM round-tripping | LLM output never re-fed to LLM in subsequent turns |
| SNOMED dictionary poisoning | CSV in-repo, version-controlled; strategies validate at startup |
| Session-state leak between users | session_id is uuid4; `app.py` keys session state by session_id |
| LLM provider compromise | Provider abstraction; failure mode is `LLMProviderError`, never silent fallback |
| Algorithmic SNOMED false positives | confidence thresholds gate inclusion; negation annotator excludes inverted hits |
| Race in parallel paths | Strategies read-only after init; `LLMProvider` request-scoped |
| ThreadPoolExecutor exhaustion (timeout DoS) | `max_workers=2` per pipeline; one pipeline per process via `@st.cache_resource`. **POC limitation:** sustained timeout DoS could leave threads running in background; documented; production deploy should add per-IP rate limit (already a known gap) |
| Extraction timeout | Hard 15s timeout on each future. On timeout → `PipelineError` with generic UI message; both futures cancelled; session unchanged |
| Catastrophic backtracking in a SNOMED strategy | Protocol contract forbids unbounded regex on user input; strategies must be free of catastrophic backtracking. `hybrid_cascade` (the default) uses no regex on user input — exact dict lookup, fuzzy token_sort_ratio (rapidfuzz, no backtracking), embedding similarity (no regex). New strategies must document compliance |
| Session corruption (Redis tampering, file truncation) | `from_dict(d, on_error="new_session")` recovers; logs WARN |
| PHI leak via logging | `summary_for_logging()` is the ONLY safe method to log session state. `to_dict()` is for serialization, not logging |
| Substitution introduces new injection pattern | `assert_safe()` on merged canonical query (defense-in-depth) |

### Why pre-extraction gate doesn't reduce security

The preprocessor runs first, on the raw user input. The sufficiency gate is a router downstream of security. A malicious query that passes the preprocessor (= passes security) and triggers the gate ends up in clarification — same surface as a successful extraction-then-clarify path. No new attack surface.

---

## 6. LLM-portability checklist

When Groq is replaced:

- [ ] Add `<provider>_provider.py` in `src/llm_provider/` implementing `LLMProvider`
- [ ] Register in `PROVIDER_REGISTRY`
- [ ] Set `LLM_PROVIDER` env var
- [ ] Re-tune `FilterExtractor.SYSTEM_PROMPT` for new model's prompt style (only one prompt in the system)
- [ ] Run `tests/batch_eval.py` to revalidate filter extraction quality
- [ ] Run `tests/test_llm_provider.py` with the new provider's mock
- [ ] Update CONTEXT.md tech stack table

What does NOT change:
- ✅ `SufficiencyGate`, `AmbiguousTermsRegistry` — pure Python
- ✅ Any `SNOMEDSearchStrategy` — provider-agnostic
- ✅ `NegationAnnotator`, `GeoNormalizer` — pure Python
- ✅ `pipeline.py` — depends only on Protocols
- ✅ `app.py` — provider-agnostic

---

## 7. SNOMED-search-strategy plug-in guide

Adding a new SNOMED matching algorithm:

1. **Create** `src/snomed_search/<my_strategy>.py` implementing the `SNOMEDSearchStrategy` Protocol. Required: `name`, `__init__(dictionary_path, **kwargs)`, `search(query) -> list[SNOMEDMatch]`, `health_check()`. Document candidate-extraction approach in `search()` docstring. Populate `SNOMEDMatch.span` for every match.
2. **Register** in `src/snomed_search/registry.py`: `STRATEGY_REGISTRY["my_strategy"] = MyStrategy`.
3. **Test** by adding to `tests/test_snomed_strategies.py` parity suite.
4. **Run** the batch eval: `python tests/batch_eval.py --strategy my_strategy`.
5. **Adopt as default** (optional): change `DEFAULT_STRATEGY` in `registry.py`.

**Constraints every strategy must satisfy:**
- Deterministic given fixed dictionary + same query.
- Thread-safe after `__init__` (mutate nothing).
- Returns `SNOMEDMatch` with valid char `span` offsets.
- No catastrophic backtracking; no unbounded regex on user input.
- `health_check()` returns `ready=True` only if dictionary is loaded and indexes built.
- Does NOT pre-filter by MIN_CONFIDENCE; assembler filters.

---

## 8. Deployment-portability checklist

For multi-replica / containerized deploy:

- [ ] Replace `st.session_state["conversation"] = session` with a session store (Redis / DynamoDB / Postgres). `to_dict()` / `from_dict()` define the boundary.
- [ ] Use `from_dict(d, on_error="new_session")` in production session loads.
- [ ] Add session-store TTL (1 hour idle).
- [ ] Move rate-limit counters from session-state to Redis.
- [ ] Server-side per-IP rate limit.
- [ ] Externalize SNOMED dictionary (S3 / GCS) — pre-fetch at container init.
- [ ] Set `AMBIG_STRICT_VALIDATION=false` in production for graceful degrade.
- [ ] Add health-check endpoint that returns 200 only if `AmbiguousTermsRegistry` validated all options under strict mode (pre-flight CI test, not production gate).

What does NOT change for multi-replica:
- ✅ Pipeline code, sufficiency logic, conversation schema, strategy interface, provider interface

---

## 9. OBSOLETE-AT-SCALE marker convention

```python
# OBSOLETE-AT-SCALE: <one-line reason and what to do at cleanup>
```

**Cleanup at scale-up:**
```
grep -rn "OBSOLETE-AT-SCALE" .
```

**Expected matches:**
1. `NLPPipeline.run(raw_query)` in `src/pipeline.py`
2. `NLPPipeline._assemble_best_effort_from_session()` in `src/pipeline.py`
3. The `--legacy` flag in `tests/batch_eval.py`
4. `tests/run_tests.py` module docstring
5. Single-shot input rendering function in `app.py`
6. `src/extractor.py` shim
7. `src/snomed_resolver.py` shim
8. `src/geo_normalizer.py` re-export shim
9. CSV rows in `tests/batch_test_cases.csv` with empty `expected_type`

---

## 10. Implementation order

1. **`src/exceptions.py`** — `LLMProviderError`, `PipelineError`, `StrategyError`. Acceptance: importable.
2. **`src/llm_provider/`** — Protocol + GroqProvider + registry. Acceptance: `test_llm_provider.py` passes.
3. **`src/snomed_search/base.py`** — `SNOMEDSearchStrategy` Protocol + `SNOMEDMatch` (with span enforcement). Acceptance: type-checks.
4. **`src/snomed_search/hybrid_cascade.py`** — port existing resolver. Acceptance: parity test against existing `snomed_resolver.py` on `tests/batch_test_cases.csv` — recall non-regression.
5. **`src/snomed_search/negation.py`** — NegEx. Acceptance: `test_negation.py` (9 cases) passes.
6. **`src/snomed_search/registry.py`** — factory. Acceptance: env var selects strategy; bad name → `ValueError`.
7. **`data/ambiguous_terms.json`** + **`AmbiguousTermsRegistry`** loader. Acceptance: every option resolves via `HybridCascadeStrategy` ≥ 0.85; auto-derived overrides exclude trigger itself; `strict_validation=False` gracefully degrades.
8. **`src/sufficiency_gate.py`** — `SufficiencyGate` + DEFAULT_CONDITION_PROMPT. Acceptance: `test_sufficiency_gate.py` (10 cases) passes.
9. **`src/conversation.py`** — `Turn`, `ConversationSession`, serialization, `summary_for_logging()`, `from_dict(on_error=)`. Acceptance: round-trip equality; corruption recovery.
10. **`src/filter_extractor.py`** — `FilterExtractor`. Acceptance: parity with current `Extractor` on `tests/test_cases.json` (20 cases) for filter fields only; Kansas City test case mandatory.
11. **`src/normalizers/geo.py`** — move existing. Acceptance: existing geo tests pass.
12. **`src/assembler.py`** — `ClarificationOutput`, discriminated union, `render_question()`. Acceptance: pydantic parses both shapes.
13. **`src/pipeline.py`** — `run_with_session()`. Acceptance: `test_conversation.py` (13 cases) passes.
14. **`src/snomed_search/aho_corasick.py`** + **`ngram_lookup.py`**. Acceptance: parity test (top-5 Jaccard ≥ 0.80 on ≥80% of cases).
15. **`tests/batch_eval.py`** — multi-turn `>>>` + `--legacy` + `--strategy`. Acceptance: 100-case eval runs under all three strategies; reports comparative metrics.
16. **`app.py`** — chat UI rewrite.
17. **CONTEXT.md** — append "Post-Build Changes Log" entry.

**Parity gates (must pass before merge):**
- `hybrid_cascade` SNOMED recall on `batch_test_cases.csv` ≥ existing `snomed_resolver.py`.
- All three strategies: top-5 code Jaccard ≥ 0.80 on ≥80% of cases (against `hybrid_cascade`).
- NegEx vs LLM-detected negation agreement ≥ 90% on `test_negation.py` cases.
- Pipeline latency p50 (accepted query) ≤ existing pipeline p50.
- `FilterExtractor` parity with current `Extractor` on `tests/test_cases.json` (filter fields only).

---

## 11. What is explicitly out of scope

- ❌ The `advarra-siteiq-nlp.html` file
- ❌ Sufficiency logic in any LLM prompt
- ❌ LLM-generated clarification text
- ❌ LLM-driven negation detection
- ❌ Production session store (Redis) — designed for via `to_dict`/`from_dict`
- ❌ User authentication and per-IP rate limits — known POC gaps
- ❌ ClinicalTrials.gov integration
- ❌ Streaming responses
- ❌ Caching of LLM responses
- ❌ Asyncio rewrite (POC keeps `concurrent.futures.ThreadPoolExecutor`)

---

## 12. File-touch summary

| File | Lines added | Lines removed | Notes |
|---|---|---|---|
| `data/ambiguous_terms.json` | ~120 | 0 | New |
| `src/exceptions.py` | ~40 | 0 | New |
| `src/sufficiency_gate.py` | ~180 | 0 | New (incl. DEFAULT_CONDITION_PROMPT) |
| `src/conversation.py` | ~220 | 0 | New (incl. summary_for_logging, from_dict on_error) |
| `src/llm_provider/base.py` | ~50 | 0 | New |
| `src/llm_provider/groq_provider.py` | ~110 | 0 | New |
| `src/llm_provider/registry.py` | ~40 | 0 | New |
| `src/filter_extractor.py` | ~200 | 0 | New (full prompt incl. state-extraction rules) |
| `src/snomed_search/base.py` | ~80 | 0 | New (incl. span enforcement) |
| `src/snomed_search/aho_corasick.py` | ~160 | 0 | New |
| `src/snomed_search/ngram_lookup.py` | ~130 | 0 | New |
| `src/snomed_search/hybrid_cascade.py` | ~300 | 0 | New (incl. _compute_residual_spans) |
| `src/snomed_search/negation.py` | ~110 | 0 | New (incl. pseudo-negation) |
| `src/snomed_search/registry.py` | ~40 | 0 | New |
| `src/normalizers/geo.py` | ~166 (moved) | ~166 (from src/) | Move |
| `src/normalizers/base.py` | ~30 | 0 | New |
| `src/pipeline.py` | ~200 | ~80 | Refactor (incl. assert_safe, _state_value_for_geo) |
| `src/assembler.py` | ~120 | ~10 | Add ClarificationOutput, render_question |
| `src/extractor.py` | ~10 | ~265 | Shim |
| `src/snomed_resolver.py` | ~10 | ~362 | Shim |
| `src/geo_normalizer.py` | ~3 | ~166 | Re-export |
| `app.py` | ~180 | ~120 | Chat UI |
| `tests/test_sufficiency_gate.py` | ~170 | 0 | New |
| `tests/test_conversation.py` | ~280 | 0 | New |
| `tests/test_snomed_strategies.py` | ~200 | 0 | New |
| `tests/test_negation.py` | ~140 | 0 | New |
| `tests/test_llm_provider.py` | ~100 | 0 | New |
| `tests/batch_eval.py` | ~100 | ~10 | Multi-turn + --legacy + --strategy |
| `tests/run_tests.py` | ~5 | 0 | OBSOLETE-AT-SCALE marker |
| `requirements.txt` | ~1 | 0 | + pyahocorasick |
| `CONTEXT.md` | ~80 | 0 | Post-Build entry |

**Totals:** ~3300 lines added, ~1100 removed. Effort: 5-7 days dev + 2 days QA.

---

## 13. Self-audit (architect → architect)

### Pass

- ✅ All file paths align with existing repo structure
- ✅ Pydantic V2 syntax throughout
- ✅ `state.values: list[str]` schema preserved
- ✅ Multi-turn security loop intact: preprocessor + assert_safe + clinical-intent gate every turn
- ✅ HIPAA log hygiene: every component enumerates LOGS / NEVER LOGS; structured JSON log line for audit
- ✅ LLM-portable: only one prompt; only one provider import
- ✅ SNOMED-pluggable: tight Protocol with span enforcement and thread-safety contract
- ✅ Each §3.x lists non-responsibilities
- ✅ Backwards-compat shims tagged OBSOLETE-AT-SCALE
- ✅ One new dependency (`pyahocorasick`) — justified
- ✅ `summary_for_logging()` prevents PHI leak via session serialization
- ✅ Init order is unambiguous: strategy → registry → gate → pipeline
- ✅ Every helper function defined (`render_question`, `_state_value_for_geo`, `_compute_residual_spans`, `LLMProviderError`, `DEFAULT_CONDITION_PROMPT`)
- ✅ Override-terms self-defeat bug fixed (auto-derivation excludes trigger itself)
- ✅ Session append happens AFTER successful assembly
- ✅ Future timeout cancels both, no partial results, generic error
- ✅ `from_dict()` corruption recovery via `on_error="new_session"`

### Caught and corrected during revision (post-QA/Security review)

- ⚠️ **Init order ambiguity (QA blocker):** Clarified `AmbiguousTermsRegistry.__init__(path, snomed_strategy, snomed_csv_path, strict_validation)` requires strategy. Pipeline init order is now: strategy → registry → gate.
- ⚠️ **`SNOMEDMatch.span` not enforced (QA blocker):** Made `span` a required field with `__post_init__` validation. Protocol docstring requires it.
- ⚠️ **Strategy candidate extraction unspecified (QA blocker):** Each strategy now documents its approach in `search()` docstring; parity test compares only top-5 codes (not exact match sets).
- ⚠️ **Per-turn log path enum unclear (Security blocker):** Defined the 6 path values explicitly: `sufficiency_clarification`, `max_turns`, `clinical_intent`, `post_extraction_safety`, `search`, plus preprocessor failure (logged separately). Structured JSON log line.
- ⚠️ **Undefined helpers (QA majors):** Defined `LLMProviderError` (in new `src/exceptions.py`), `DEFAULT_CONDITION_PROMPT` (in `sufficiency_gate.py`), `_compute_residual_spans` (in `hybrid_cascade.py`), `render_question` (in `assembler.py`), `_state_value_for_geo` (in `pipeline.py`).
- ⚠️ **Turn fields not Optional (QA major):** Marked `filters`, `geo` as Optional with default None.
- ⚠️ **Override-terms self-defeat bug (QA major):** Auto-derivation now explicitly excludes any term equal to the trigger itself.
- ⚠️ **Session append before assembly (Security major):** Reordered; session.append_turn() now happens AFTER successful build_clarification or assemble.
- ⚠️ **`to_dict()` PHI leak risk (Security major):** Added `summary_for_logging()` method; documented logging convention.
- ⚠️ **`FilterExtractor` lost state-extraction rules (Security major):** Embedded full disambiguation rules from existing `extractor.py` in §3.6 prompt spec, including Kansas City and institution-name rules. Added parity gate against current extractor.
- ⚠️ **`LLMProvider` retry contract undefined (Security major):** Specified transient vs permanent error policy with explicit error categories.
- ⚠️ **`from_dict` corruption handling (QA major):** Added `on_error="new_session"` mode for production session-load paths.
- ⚠️ **Future timeout behavior unspecified (QA major):** Both futures cancelled on timeout; PipelineError raised; session unchanged. Background thread leak documented as POC limitation.
- ⚠️ **Negation comma handling (QA major):** Explicitly documented that comma does NOT stop scan; rationale (clinical syntax chains conditions in commas). Added pseudo-negation cue list.
- ⚠️ **Validator `sys.exit(1)` deployment DoS (Security major):** Added `strict_validation` parameter. Default True for POC/CI, env-overridable to False for production rolling deploys.
- ⚠️ **Defense-in-depth: re-validate canonical query (Security minor):** Added `preprocessor.assert_safe(canonical)` after `update_canonical_query()`.
- ⚠️ **Test specificity (QA minor):** C3 and C10 now have concrete expected values; parity gate has explicit Jaccard rule.
- ⚠️ **Self-defeat test case (QA minor):** Added test case 9 to `test_sufficiency_gate.py`.

### Reviewer findings explicitly defended (not adopted)

- **Security: "canonical-query substitution against LLM-extracted normalized term."** Reviewer assumed the LLM extracts `medical_terms` and the trigger is matched against LLM output. In the new design, LLM does NOT extract medical terms; the trigger comes directly from `ambiguous_terms.json` and is matched against the canonical query that the user literally typed. Substitution is well-defined. Documented in §3.3.
- **Security: "structured JSON logging."** Adopted in §3.8 spec.
- **Security minor: "thread-safety claim too informal."** Tightened in §3.6.1 Protocol contract.

### Open risks acknowledged

1. **Algorithmic SNOMED extraction may have lower recall than LLM extraction on idiomatic phrasings.** Parity gate in §10 catches regressions; if observed, expand alias dict before merge.
2. **NegEx may have false negatives/positives on complex syntax.** Pseudo-negation cue list mitigates known false positives; `test_negation.py` covers these. Window size tunable.
3. **Parallel path latency depends on slowest of the two.** If LLM rate-limited, no improvement over serial.
4. **`pyahocorasick` C extension** — wheels exist for Py 3.13 on major platforms; users without can fall back to `ngram_lookup` or `hybrid_cascade`.
5. **Override-derivation depends on CSV preferred_term quality.** Future improvement: derive from CSV synonyms too (currently disabled to avoid false positives).
6. **Streamlit `st.session_state` per-process.** Documented in §8.
7. **Plural form not auto-overridden** (test C11): "cancers" doesn't whole-word-match the "cancer" trigger key (`\bcancer\b`). Mitigation: register "cancers" as separate trigger key OR add lemmatization. Acceptable for POC.
8. **Background thread leak under sustained extraction-timeout DoS.** ThreadPoolExecutor doesn't truly cancel running threads. Documented; production should add per-IP rate limit.
9. **Post-extraction safety check duplicates work** (we paid for the LLM call). Intentional: pre-extraction gate catches ambiguous *terms*; post-extraction catches *missing* terms. Future improvement: heuristic noun-phrase check pre-extraction.

**Deliverable status:** Revision 2 ready. All BLOCKERS and MAJORS from QA + Security review addressed or defended. Proceeding to dev pseudocode phase.

---

## 14. Revision Log

### Revision 2 (this version)

- Addressed 4 QA blockers, 9 QA majors, 6 QA minors (all real issues; some minors deferred with rationale).
- Addressed 3 Security blockers (1 was a misread of the architecture, defended; 2 fixed), 6 Security majors (5 fixed, 1 — structured logging — adopted as enhancement), 4 Security minors (3 adopted, 1 — strict validator behavior — softened with config flag).
- Total fixes integrated: 18 architectural changes across 9 sections.
- New module added: `src/exceptions.py` (centralized exception types).
- New session method: `summary_for_logging()` (PHI-safe logging boundary).
- New pipeline step: `preprocessor.assert_safe(canonical)` (defense-in-depth).

### Revision 1

- Initial architecture proposal (replaced post-extraction sufficiency from `modified-proposal.md` with pre-extraction gate; split extractor into filter LLM + algorithmic SNOMED; introduced pluggable `SNOMEDSearchStrategy`; introduced `LLMProvider` abstraction).

### Errata addressed during dev pseudocode pass (clarifications, not architecture changes)

The dev pass identified five spec ambiguities. These are clarifications, not redesigns; the architect accepts them in-place:

1. **`Preprocessor.assert_safe(text: str) -> None`** — new method on the existing `Preprocessor` class. Runs the injection-pattern subset of `_INJECTION_PATTERNS` (NOT the length checks; canonical queries from substitute/append may legitimately exceed the 500-char turn cap). Raises `PreprocessorError("Invalid query detected")` on match. Implementation is a 5-line method that reuses `_check_injection()`. Listed in §3.9 step 2b but the new method itself was implicit in the proposal — now explicit.
2. **`any_filter_set(filters: ExtractedFilters) -> bool`** — module-level helper, lives in `src/sufficiency_gate.py` (next to its primary user) and is `from .sufficiency_gate import any_filter_set`-imported by `src/pipeline.py`. Public, not underscore-prefixed.
3. **`AmbiguousTermsRegistry` and `AmbiguousEntry` colocation** — both classes live in `src/sufficiency_gate.py` alongside `SufficiencyGate` and `DEFAULT_CONDITION_PROMPT`. No separate module; the file inventory table in §2 implies this but did not state it.
4. **`ALIAS_DICTIONARY` during shim period** — stays in `src/snomed_resolver.py` (now a shim) during the OBSOLETE-AT-SCALE transition. `HybridCascadeStrategy` imports it via `from src.snomed_resolver import ALIAS_DICTIONARY`. At cleanup, `ALIAS_DICTIONARY` moves into `hybrid_cascade.py` and the import disappears with the shim.
5. **`DEFAULT_CONDITION_PROMPT` CSV path** — built lazily on first `SufficiencyGate.__init__` call (NOT at module import time, which lacks path context). The gate receives `snomed_csv_path` from the pipeline and builds DEFAULT_CONDITION_PROMPT from that CSV's most-frequent preferred-term categories on first init, then memoizes it on a class-level cache for subsequent gates. Hard fallback options (`["Cancer", "Diabetes", "Heart Disease", "Autoimmune", "Other"]`) apply if CSV inspection fails. This avoids the import-time path-injection problem the dev flagged.

### Errata addressed during dev/QA loop (round 1)

QA review of the dev's pseudocode surfaced 5 blockers + 8 majors. Dev patched all 13 in-place. One fix introduces new method signatures the architect must reflect here:

6. **`ConversationSession.update_canonical_query()` split into pure compute + mutator (M2 fix).** Original signature mutated session state before the defense-in-depth `Preprocessor.assert_safe()` could run, leaving partial session state on `PreprocessorError`. The fix splits the method:
   - `compute_canonical_query(user_input: str) -> str` — pure, returns the merged canonical query without mutating `self.canonical_query`. Substitute-or-append logic identical to the original.
   - `set_canonical_query(query: str) -> None` — mutator, writes the value to `self.canonical_query`. Called only after `assert_safe()` passes.
   - The original `update_canonical_query()` is retained as a convenience that calls both for callers that don't need the assert_safe guard (tests, legacy `run()` shim). New code uses the split pair.
   - Pipeline order in `run_with_session()`: `canonical = session.compute_canonical_query(preprocessed.text)` → `preprocessor.assert_safe(canonical)` → `session.set_canonical_query(canonical)`.
   - Invariant preserved: session is mutated only after every gate that could raise has passed.
