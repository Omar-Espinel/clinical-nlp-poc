# Ambiguity Coverage Rework — Design Document
**Branch: `Arch-changes-with-response` · Repo: clinical-nlp-poc · 2026-05-11**
*Architect design only — pseudocode throughout · No source changes in this file*

---

## 1. Why This Exists

<!-- rev2: reframed per minor note — "kidney" is the canonical Layer 2 example; Layer 1 handles ≥2-frequency anatomy tokens -->
The `data/ambiguous_terms.json` registry is hand-curated and has systematic coverage gaps. A user typing
bare "kidney" receives "No clinical content found" because: (a) "kidney" is not a registry trigger,
(b) "kidney" does not appear as a `preferred_term` in `snomed_clinical_trials.csv` (only as a sub-word
inside "malignant neoplasm of kidney"), and (c) the algorithmic SNOMED search produces no match at or above
the pipeline's `MIN_CONFIDENCE=0.60` threshold for this one-word query. "kidney" is the canonical example
for **Layer 2** — an embedding-based fallback that fires when the registry and cascade both fail. Layer 1
handles the complementary class: anatomy/system tokens that appear in **≥ 2** distinct CSV preferred_terms
(e.g., "lung" appears in "malignant neoplasm of lung", "non-small cell lung carcinoma", "small cell lung
carcinoma" etc. — Layer 1 auto-derives a trigger for these at startup).

The same dead-end hits "bone", "skin", "thyroid", "stomach", "GI", "ENT", and any other anatomy/system
word that is linguistically broader than the SNOMED preferred terms. This document specifies two
complementary layers: **Layer 1** auto-derives trigger entries from the SNOMED CSV at startup so the
registry stays current as the CSV grows, and **Layer 2** adds an embedding-based ambiguity fallback that
fires between SNOMED search and the clinical-intent rejection, surfacing mid-confidence neighbors as
clarification options for terms that neither the registry nor the cascade can resolve with high confidence.

---

## 2. Layer 1 — Auto-Derived Triggers from SNOMED at Startup

### 2.1 Goal

Scan `data/snomed_clinical_trials.csv` during `AmbiguousTermsRegistry.__init__` and synthesize
`AmbiguousEntry` objects for anatomy/system tokens that appear inside multiple preferred terms but are not
themselves preferred terms. Merge these into the in-memory registry. Hand-curated JSON entries **always win**
on key collision.

### 2.2 Token-Extraction Rules (Open Question 1 — Resolved)

A candidate token is extracted from each `preferred_term` by:

1. Lowercase the preferred_term.
2. Split on whitespace (same as `re.finditer(r'\S+', ...)` used in `hybrid_cascade.py`'s
   `_tokenize_with_offsets`).
3. Drop tokens that match the stopword list `_ANATOMY_STOPWORDS` (see §2.3).
4. Drop tokens shorter than `MIN_DERIVED_TOKEN_LEN = 4` characters (eliminates "of", "in", "the", two-letter
   abbreviations, numbers like "19").
5. Single-token only: multi-token anatomy phrases (e.g. "thyroid gland") are reduced to each constituent
   token individually. This keeps the trigger detection simple and consistent with the existing whole-word
   `\b...\b` regex approach.
6. Whole-word match only: the trigger must fire on whole-word boundaries — same contract as existing
   hand-curated triggers.
7. Case: all derived trigger keys are stored and matched lowercase, exactly as existing entries
   (`trigger_key = raw_trigger.lower().strip()`).

### 2.3 Stopword List

New module-level constant in `sufficiency_gate.py`:

```
_ANATOMY_STOPWORDS: frozenset[str] = frozenset({
    "malignant", "neoplasm", "disease", "disorder", "syndrome",
    "carcinoma", "adenocarcinoma", "sarcoma", "infarction", "failure",
    "infection", "deficiency", "insufficiency", "procedure", "therapy",
    "treatment", "measurement", "monitoring", "imaging", "acute",
    "chronic", "primary", "secondary", "idiopathic", "congenital",
    "type", "cell", "small", "large", "mixed", "multiple", "lateral",
    "diffuse", "bilateral", "unilateral", "stage", "grade", "level",
    "blood", "arterial", "venous", "systemic", "peripheral",
    "obstructive", "restrictive", "progressive", "benign", "solid",
    "squamous", "gland", "vessel", "node", "tissue", "junction",
    "uteri", "mellitus", "erythematosus", "spondylitis",
    "human",   <!-- rev2: added per minor note — "human" in "human immunodeficiency virus infection" -->
})
```

Rationale: these words appear inside many preferred terms but would create overly broad triggers that fire
everywhere. The list can be extended by maintainers; it is not authoritative, just a startup filter.
"human" is explicitly included because it appears inside "human immunodeficiency virus infection" and would
otherwise produce a spuriously broad trigger. Additional tokens found during batch evaluation are added here.

### 2.4 Frequency Threshold (Open Question 1, 2 — Resolved)

A token qualifies as a derived trigger if it appears as a whole-word sub-token of **≥ 2 distinct** CSV
`preferred_term` rows (after deduplication by `preferred_term` text — some concept IDs repeat).

New constant:

```
MIN_DERIVED_TERM_FREQUENCY = 2  # token must appear in ≥ 2 distinct preferred_terms
```

The "kidney" case: "kidney" appears in only **1** distinct preferred_term ("malignant neoplasm of kidney").
Frequency = 1 < threshold. Therefore, Layer 1 alone does not synthesize a "kidney" trigger. This is
intentional — Layer 2 (§3) handles single-match cases via the embedding fallback. The threshold of 2 ensures
synthesized clarification entries always have ≥ 2 candidate options from the CSV (see §2.6 option-selection
rules). Lowering to 1 would require the synonym/embedding fallback path to populate options, complicating
the design.

<!-- rev2: synonym-rescue path removed per M1 recommendation (b) — Layer 2 is the correct home for single-CSV-row tokens -->
**Synonym-Rescue Path — Removed:**
The previous design included an exception for tokens that appear in exactly 1 `preferred_term` but also
appear in CSV `synonyms` for a different concept. This path has been **removed**. The synonym index
(a flat `set[str]`) loses concept-association, making the rescue logic unsound. Single-CSV-row tokens like
"kidney", "bone", "breast", "colon", "stomach", "thyroid", "prostate" fall through Layer 1 and are handled
by Layer 2 — which is the intended design.

**Concrete qualifying examples from current CSV (Layer 1 — frequency ≥ 2):**
- "lung" → "malignant neoplasm of lung", "non-small cell lung carcinoma", "small cell lung carcinoma",
  "lung disorder" → frequency ≥ 2 → **qualifies for Layer 1**
- "kidney" → "malignant neoplasm of kidney" (1 preferred_term) → frequency 1 → **Layer 2 handles**
- "bone" → 1 preferred_term → **Layer 2 handles**
- "breast" → 1 preferred_term → **Layer 2 handles**
- "colon" → 1 preferred_term → **Layer 2 handles**
- "stomach" → 1 preferred_term → **Layer 2 handles**
- "thyroid" → 1 preferred_term → **Layer 2 handles**
- "prostate" → 1 preferred_term → **Layer 2 handles**

*Layer 1 primarily benefits system-level words ("lung", "heart", "liver", "skin", "neuro") where the CSV
has multiple distinct concepts, while organ-specific single-concept words are handled by Layer 2.*

### 2.5 Algorithm Pseudocode

New private method `_build_derived_entries` on `AmbiguousTermsRegistry`:

```
FUNCTION _build_derived_entries(csv_path: str) -> dict[str, AmbiguousEntry]:
    """Scan preferred_terms; derive triggers with frequency >= threshold.
    Returns dict keyed by lowercase trigger. Does NOT validate via SNOMED strategy
    (options ARE preferred_terms, guaranteed to resolve at 1.0 exact match).
    Synonym-rescue path removed — Layer 2 handles single-preferred_term tokens.
    """

    df = read_csv(csv_path, dtype=str).fillna("")
    # Deduplicate by preferred_term text to avoid counting concept_id duplicates twice
    unique_terms = df["preferred_term"].str.strip().str.lower().drop_duplicates().tolist()

    # Step 1: tokenize all preferred_terms; collect candidate tokens with their source rows
    token_to_preferred_terms: dict[str, list[str]] = defaultdict(list)

    FOR preferred_term IN unique_terms:
        raw_tokens = re.findall(r'\S+', preferred_term)
        FOR token IN raw_tokens:
            token_lower = token.lower().strip(".,;:'\"")
            IF len(token_lower) < MIN_DERIVED_TOKEN_LEN: continue
            IF token_lower IN _ANATOMY_STOPWORDS: continue
            IF token_lower NOT IN token_to_preferred_terms[preferred_term]:
                token_to_preferred_terms[token_lower].append(preferred_term)

    # Step 2: apply frequency threshold
    qualified_tokens = {
        token: sources
        FOR token, sources IN token_to_preferred_terms.items()
        IF len(sources) >= MIN_DERIVED_TERM_FREQUENCY
    }

    # Step 3: for each qualified token, select options and build AmbiguousEntry
    derived: dict[str, AmbiguousEntry] = {}
    FOR token, source_preferred_terms IN qualified_tokens.items():
        options = _select_derived_options(token, source_preferred_terms, df)
        IF len(options) < AMBIG_JSON_MIN_OPTIONS:
            # Not enough distinct preferred_terms → skip (Layer 2 will handle)
            logger.info("derived trigger skipped: too few options (len=%d token_len=%d)", len(options), len(token))
            continue
        question = _build_derived_question(token)
        derived[token] = AmbiguousEntry(
            trigger=token,
            category="indication",
            question_template=question,
            options=options,
            override_terms=frozenset(opt.lower() for opt in options),
            max_options=AMBIG_JSON_MAX_OPTIONS,
        )
        logger.info("auto-derived trigger: token_len=%d option_count=%d", len(token), len(options))

    RETURN derived
```

### 2.6 Option-Selection Rules (Open Question 2 — Resolved)

When a token maps to more than `AMBIG_JSON_MAX_OPTIONS` (5) source preferred_terms, the selection follows
this priority ranking:

1. **Prefer shorter preferred_terms** — shorter = broader, closer to a category rather than a specific
   histological variant. Rationale: "malignant neoplasm of lung" is more useful as a clarification option
   than "non-small cell lung carcinoma" for a user who typed "lung".
2. **Tiebreak: prefer lexicographically earlier** — stable, deterministic across restarts.
3. **Cap at `AMBIG_JSON_MAX_OPTIONS` = 5.**

<!-- rev2: .title() replaced with raw CSV value per B4 — display is lowercase as stored in CSV -->
```
FUNCTION _select_derived_options(
    token: str,
    source_preferred_terms: list[str],  # already deduplicated
    df: DataFrame,
) -> list[str]:
    """Return 3-5 display-ready option strings.
    Options are returned as-is from the CSV (lowercase, per CSV convention).
    Casing matches NLPOutput.snomed_terms[].display — lowercase throughout.
    Selection: shortest preferred_terms first; cap at MAX_OPTIONS.
    Returns empty list if < MIN_OPTIONS source_preferred_terms.
    """
    IF len(source_preferred_terms) < AMBIG_JSON_MIN_OPTIONS:
        RETURN []

    sorted_terms = sorted(source_preferred_terms, key=lambda t: (len(t), t))
    selected = sorted_terms[:AMBIG_JSON_MAX_OPTIONS]

    # Return raw CSV values — lowercase as stored. No .title() call.  <!-- rev2 -->
    RETURN list(selected)
```

Note: options use the CSV `preferred_term` value **as stored** (lowercase). This matches the casing
convention used by the existing `NLPOutput.snomed_terms[].display` field. The display layer in `app.py`
handles capitalization via CSS if needed. The `override_terms` in the `AmbiguousEntry` store lowercased
values, so casing does not affect override detection.

### 2.7 Question Template (Open Question 3 — Resolved)

Auto-derived entries use a standardized question template:

```
FUNCTION _build_derived_question(token: str) -> str:
    """Produce question template for an auto-derived trigger.
    Uses the same {trigger} and {prior_filters} placeholder convention as hand-curated entries.
    """
    RETURN "Which type of {trigger} condition are you looking for?"
```

This is intentionally more generic than some hand-curated templates (e.g., "Which oncology area are you
interested in?") because derived tokens are unknown at design time. The phrase "condition" makes it
grammatically valid for anatomy words: "Which type of lung condition are you looking for?" reads naturally.
Maintainers can override individual auto-derived entries by adding them to `ambiguous_terms.json`.

### 2.8 Merge Behavior (Open Question 4 — Resolved)

**In-memory merge only. No disk write.**

Rationale:
- The SNOMED CSV is the authoritative source; a derived `derived_registry.json` on disk would drift and
  require its own validation lifecycle.
- Hand-curated JSON is the human-editable source of truth; do not pollute it with auto-derived content.
- For inspection during development, the registry logs derived entry count at INFO level (no option text).
  Developers who need to see the full derived set can set `AMBIG_DERIVED_DEBUG=true` (new env var).
  Implementation must check `os.environ.get("ENV") == "dev"` before honoring the flag; in non-dev
  environments it is silently ignored regardless of its value. <!-- rev2: dev guard noted per minor note -->

Merge algorithm:

```
FUNCTION _merge_entries(
    json_entries: dict[str, AmbiguousEntry],   # from ambiguous_terms.json
    derived_entries: dict[str, AmbiguousEntry],
) -> dict[str, AmbiguousEntry]:
    """Hand-curated wins on collision. Log count+checksum (not trigger text)."""
    merged = dict(derived_entries)  # start with derived (lower priority)
    collision_count = 0
    FOR trigger, entry IN json_entries.items():
        IF trigger IN merged:
            collision_count += 1
        merged[trigger] = entry  # overwrite: hand-curated always wins
    # <!-- rev2: startup health log with checksum per M2 -->
    logger.info(
        "AmbiguousTermsRegistry merge: json=%d derived=%d collisions=%d total=%d checksum=%d",
        len(json_entries), len(derived_entries), collision_count, len(merged),
        len(derived_entries) + len(json_entries) + collision_count,  # simple checksum
    )
    RETURN merged
```

**Startup health log (M2):** On every startup the merge logs `json`, `derived`, `collisions`, `total`,
and a simple checksum (`len(derived) + len(json) + collisions`). This gives operators a quick signal if
the CSV changes cause unexpected shifts in derived-trigger counts. The trigger keys themselves are NOT logged.

**Optional runtime health check:** On the first runtime hit of any derived trigger, verify the underlying
option still resolves through the SNOMED strategy at ≥ 0.85 confidence. If not, log WARN and continue.
This guards against CSV edits that invalidate previously derived preferred_term options.

### 2.9 Validation Bypass for Auto-Derived Entries

The existing `_validate_and_build` method calls `self._strategy.search(opt)` for each option and requires
`max_conf >= OPTION_MIN_CONFIDENCE = 0.85`. Auto-derived entries use **preferred_terms directly from the
CSV** as options. Since the SNOMED strategy builds its `_exact_index` from these very same preferred_terms
(exact match → confidence 0.99), every derived option is trivially valid.

Therefore, auto-derived entries **skip the SNOMED resolution check** and are constructed directly via the
`AmbiguousEntry` constructor. They do NOT go through `_validate_and_build`. The rationale is logged:

```
logger.info(
    "auto-derived entry for token_len=%d: skipping SNOMED validation "
    "(options are CSV preferred_terms, guaranteed 0.99)",
    len(token)
)
```

### 2.10 Edge Cases

<!-- rev2: AMBIG_STRICT_VALIDATION and derived-entry interaction made explicit per M5 -->

| Edge Case | Behavior |
|---|---|
| CSV read fails during derived entry build | Log warning; `derived_entries = {}`; continue with JSON-only registry. Never exit(1) for derived build failure. Behavior is identical to `AMBIG_STRICT_VALIDATION=False` for the derived path. |
| Derived token equals an existing JSON trigger | JSON wins (merge §2.8). Logged as collision. |
| Derived token qualifies but option selection yields < 3 entries | Skip trigger; log at INFO. Layer 2 handles the residual. |
| Token in `_ANATOMY_STOPWORDS` appears only in short preferred terms (e.g., "cell") | Stopword filter prevents it from being treated as a trigger. |
| `_ANATOMY_STOPWORDS` does NOT contain a problematic token | Runtime: the derived entry will fire, but options (being preferred_terms) are still valid SNOMED. Not a correctness risk. |
| `MIN_DERIVED_TOKEN_LEN` increased by operator (e.g., to 5) | Tokens like "lung" (4 chars) would no longer qualify. Not a risk during normal ops; documented constant. |
| `AMBIG_STRICT_VALIDATION=True` with derived entries | Derived entries are **not** subject to `strict_validation` — they are CSV-sourced and cannot be invalid by construction. `strict_validation` applies only to JSON-loaded entries. If `_build_derived_entries` itself raises, behavior is: log warn, `derived_entries = {}`, continue. `strict_validation` mode is unaffected. |

### 2.11 Changes to `AmbiguousTermsRegistry.__init__` (lines 204–291 of `sufficiency_gate.py`)

After the existing JSON load + validation loop, add:

```
# ── Auto-derive triggers from SNOMED CSV ──────────────────────────────────────
# Runs AFTER JSON validation so collision detection is correct.
try:
    derived_entries = self._build_derived_entries(snomed_csv_path)
except Exception as exc:
    logger.warning(
        "AmbiguousTermsRegistry: derived entry build failed (%s) — using JSON-only",
        type(exc).__name__,
    )
    derived_entries = {}

# Merge: JSON wins on collision
merged_entries = _merge_entries(self._entries, derived_entries)
self._entries = merged_entries
```

Then the existing regex compilation block (lines 277–282) runs on the merged `_entries` unchanged.

---

## 3. Layer 2 — Embedding-Based Ambiguity Fallback

### 3.1 Goal

When the SNOMED cascade returns no high-confidence match for a term that is clearly medical (e.g., "kidney"),
the embedding model already loaded by `HybridCascadeStrategy` can find semantically close neighbors. Layer 2
intercepts the "no match" signal before the clinical-intent rejection at step 6 in `pipeline.py` and
surfaces the neighbors as a `ClarificationOutput`.

### 3.2 Trigger Conditions

<!-- rev2: LAYER2_MIN_NEIGHBORS raised from 2 to 3 per B1; Signal A definition updated accordingly -->
Layer 2 fires when **either** of two signals is present. All signal computations operate on
**post-`NegationAnnotator` matches** (see §4.1) and **filter out negated matches first**:

```
qualifying = [m for m in snomed_matches if m.confidence >= MIN_CONFIDENCE and not m.negated]
mid_band   = [m for m in snomed_matches if
              LAYER2_LOW_THRESHOLD <= m.confidence < MIN_CONFIDENCE
              and not m.negated]
```

**Signal A — Zero high-confidence, sufficient mid-confidence:**

Layer 2 fires when `len(qualifying) == 0` AND `len(mid_band) >= LAYER2_MIN_NEIGHBORS`.

<!-- rev2: LAYER2_HIGH_THRESHOLD eliminated per B5; MIN_CONFIDENCE is the single source of truth -->
<!-- rev2: LAYER2_MIN_NEIGHBORS = 3 per B1 -->
New constants: <!-- rev2 -->

```
# LAYER2_HIGH_THRESHOLD removed — use MIN_CONFIDENCE = 0.60 from assembler.py directly.
# Single source of truth: import MIN_CONFIDENCE from src.assembler (or move to shared constants).
LAYER2_LOW_THRESHOLD      = 0.42   # floor for "plausible but uncertain" embeddings
LAYER2_MIN_NEIGHBORS      = 3      # genuine ambiguity requires ≥3 mid-band options  <!-- rev2 -->
```

Rationale for `LAYER2_MIN_NEIGHBORS = 3`: Signal B already defines "genuine ambiguity" as ≥3 competing
options. Signal A should be consistent — 2 mid-band hits is too easy to trip on partially-matching
terminology. In practice "kidney" produces ≥3 mid-band neighbors (empirically validated in §7.3 smoke set:
"kidney" must return ≥3 options in the pre-merge gate).

Rationale for 0.42: the sentence-transformer model `all-MiniLM-L6-v2` produces cosine similarities in a
compressed range. Empirically, "kidney" encodes at ~0.52–0.60 similarity to "malignant neoplasm of kidney".
A floor of 0.42 captures genuine semantic neighbors while excluding noise (random words produce < 0.35).
This threshold should be tuned after batch evaluation; the name `LAYER2_LOW_THRESHOLD` makes it
grep-able for future adjustment.

**Signal B — Multiple high-confidence matches with genuine ambiguity (spread too narrow):**

```
qualifying = [m for m in snomed_matches if m.confidence >= MIN_CONFIDENCE and not m.negated]
```

Layer 2 fires when `len(qualifying) >= LAYER2_MIN_GENUINE_AMBIG` AND the confidence spread across
distinct concept codes is below `LAYER2_SPREAD_THRESHOLD`:

```
LAYER2_MIN_GENUINE_AMBIG  = 3      # at least 3 competing matches
LAYER2_SPREAD_THRESHOLD   = 0.08   # max - min confidence across qualifying matches
```

This handles "MS" → matches "multiple sclerosis" and "multiple myeloma" and "myocardial infarction" all
near 0.88–0.92 — no clear winner. The 0.08 spread threshold means if the top match is ≥ 0.08 above the
next one, there IS a winner and Layer 2 does not fire.

### 3.3 Embedding Access

<!-- rev2: n parameter for get_top_neighbors defined as AMBIG_JSON_MAX_OPTIONS * 3 = 15 per B3 -->
Layer 2 does **not** load a new model. It calls the embedding model already held by `HybridCascadeStrategy`
via a new method on that class:

```
FUNCTION HybridCascadeStrategy.get_top_neighbors(
    query_text: str,
    n: int,             # caller passes AMBIG_JSON_MAX_OPTIONS * 3 = 15  <!-- rev2 -->
    low_threshold: float,
    high_threshold: float,
) -> list[tuple[str, str, float]]:
    """
    Returns list of (concept_id, display_term, similarity_score) for
    the top-n preferred_terms whose similarity to query_text falls in
    [low_threshold, high_threshold).

    n = AMBIG_JSON_MAX_OPTIONS * 3 (= 15): over-fetch to allow per-code dedup
    (different concepts can have multiple synonym rows) without under-filling
    the final 3–5 option list after _deduplicate_neighbors.

    Uses existing _embedder and _np_embeddings/_collection.
    Returns [] if semantic_available is False.
    On first call when semantic_available is False: emits one INFO log.  <!-- rev2: per SEC-MIN9 -->
    HIPAA: query_text NOT logged.
    """
    IF NOT self.semantic_available OR self._embedder is None:
        RETURN []

    query_vec = self._embedder.encode([query_text], show_progress_bar=False)
    # ... cosine search against existing index (numpy or chromadb) ...
    # Filter to [low_threshold, high_threshold) band; return top-n by similarity desc.
    # Implementation mirrors existing _semantic_match_single but returns N results
    # and does NOT create SNOMEDMatch objects (caller is Layer 2 gate, not pipeline output)
```

This is a new public method on `HybridCascadeStrategy` — not part of the `SNOMEDSearchStrategy` Protocol
(other strategies do not have an embedding model). Layer 2 accesses it only if the configured strategy is
`HybridCascadeStrategy` or exposes a `get_top_neighbors` attribute (duck-type check).

### 3.4 Layer 2 Gate: New Class `EmbeddingAmbiguityGate`

New class in `src/sufficiency_gate.py`:

```
class EmbeddingAmbiguityGate:
    """
    Fires between SNOMED search (step 4/5) and clinical-intent rejection (step 6).
    Uses the embedding model already loaded by the SNOMED strategy.
    Does NOT call LLM. Does NOT mutate session.
    Thread-safe post-init: all state is read-only after construction.
    """

    def __init__(
        self,
        snomed_strategy: SNOMEDSearchStrategy,
        low_threshold: float = LAYER2_LOW_THRESHOLD,
        # high_threshold: uses MIN_CONFIDENCE imported from assembler  <!-- rev2: B5 -->
        min_neighbors: int = LAYER2_MIN_NEIGHBORS,
        min_genuine_ambig: int = LAYER2_MIN_GENUINE_AMBIG,
        spread_threshold: float = LAYER2_SPREAD_THRESHOLD,
    ) -> None:
        # Duck-type check: does the strategy expose get_top_neighbors?
        self._has_embeddings: bool = hasattr(snomed_strategy, "get_top_neighbors")
        self._strategy = snomed_strategy
        self._low = low_threshold
        self._high = MIN_CONFIDENCE  # single source of truth  <!-- rev2 -->
        self._min_neighbors = min_neighbors
        self._min_genuine_ambig = min_genuine_ambig
        self._spread = spread_threshold

    def evaluate(
        self,
        canonical_query: str,
        snomed_matches: list[SNOMEDMatch],
        # snomed_matches MUST be post-NegationAnnotator (see §4.1)  <!-- rev2: M4 -->
    ) -> Optional[SufficiencyDecision]:
        """
        Returns a SufficiencyDecision(sufficient=False, reason="embedding_ambiguity")
        if Layer 2 should fire, else None (pass-through).

        Precondition: snomed_matches are post-NegationAnnotator; .negated is populated.
        HIPAA: canonical_query NOT logged. Only match counts and confidence scores logged.
        """
        IF NOT self._has_embeddings:
            RETURN None  # graceful no-op if strategy lacks embeddings

        # <!-- rev2: M4 — filter negated matches before any signal computation -->
        qualifying = [m for m in snomed_matches if m.confidence >= self._high and not m.negated]
        mid_band   = [m for m in snomed_matches if self._low <= m.confidence < self._high and not m.negated]

        signal_a = (len(qualifying) == 0 and len(mid_band) >= self._min_neighbors)

        signal_b = False
        IF len(qualifying) >= self._min_genuine_ambig:
            confs = [m.confidence for m in qualifying]
            IF (max(confs) - min(confs)) < self._spread:
                signal_b = True

        IF NOT (signal_a OR signal_b):
            RETURN None

        # Collect neighbor options
        # For signal A: use mid_band matches as candidates (they are already found)
        # For signal B: use qualifying matches
        candidates = mid_band if signal_a else qualifying

        options = _deduplicate_neighbors(candidates, max_options=AMBIG_JSON_MAX_OPTIONS)
        IF len(options) < AMBIG_JSON_MIN_OPTIONS:
            # Too few distinct neighbors → not enough to offer a meaningful choice
            logger.info(
                "EmbeddingAmbiguityGate: insufficient options (count=%d) — pass-through",
                len(options)
            )
            RETURN None

        # Build a synthetic AmbiguousEntry for the assembler
        # Options flow through ResponseAssembler.build_clarification() which applies
        # html.escape() per existing assembler.py code — no additional sanitization needed.  <!-- rev2: M7 -->
        entry = AmbiguousEntry(
            trigger="__embedding_fallback__",
            category="indication",
            question_template="Which condition are you looking for? {prior_filters}",
            options=options,
            override_terms=frozenset(opt.lower() for opt in options),
            max_options=AMBIG_JSON_MAX_OPTIONS,
        )

        logger.info(
            "EmbeddingAmbiguityGate fired: signal=%s candidates=%d options=%d qualifying=%d",
            "A" if signal_a else "B",
            len(candidates), len(options), len(qualifying),
            # NOT logged: canonical_query, option text, candidate display strings
        )

        # <!-- rev2: B2 — triggered_by=None always (APPEND mode); single-token substitute branch removed -->
        RETURN SufficiencyDecision(
            sufficient=False,
            reason="embedding_ambiguity",
            triggered_by=None,   # APPEND mode always — see §3.6
            matched_entry=entry,
        )
```

New `reason` enum value: `"embedding_ambiguity"`. Extend the enum comment in `SufficiencyDecision`.

### 3.5 Neighbor Deduplication (Open Question 6 — Resolved)

<!-- rev2: n parameter for get_top_neighbors = 15 (AMBIG_JSON_MAX_OPTIONS * 3) per B3; .title() removed per B4 -->
```
FUNCTION _deduplicate_neighbors(
    candidates: list[SNOMEDMatch],
    max_options: int,
) -> list[str]:
    """
    Deduplicate by concept code (highest-confidence wins per code).
    Sort descending by confidence. Tied confidence: dict insertion order is
    the tie-breaker (Python 3.7+ dict preserves insertion order).  <!-- rev2: SEC-MIN10 -->
    Return display strings as-is from the CSV (lowercase per CSV convention).
    Capped at max_options (3-5).
    """
    best_per_code: dict[str, SNOMEDMatch] = {}
    FOR m IN candidates:
        IF m.code NOT IN best_per_code OR m.confidence > best_per_code[m.code].confidence:
            best_per_code[m.code] = m
    sorted_matches = sorted(best_per_code.values(), key=lambda m: -m.confidence)
    selected = sorted_matches[:max_options]
    # Return m.display as-is — lowercase, matching NLPOutput.snomed_terms[].display convention.
    # No .title() call.  <!-- rev2: B4 -->
    RETURN [m.display for m in selected]
```

`get_top_neighbors` is called with `n = AMBIG_JSON_MAX_OPTIONS * 3 = 15`. Over-fetching allows per-code
dedup to fill the final 3–5 option list even when multiple synonym rows map to the same concept code.

Maximum option count is `AMBIG_JSON_MAX_OPTIONS = 5`, minimum `AMBIG_JSON_MIN_OPTIONS = 3`. If fewer than 3
distinct concept codes are present in the candidates, Layer 2 does not fire (returns `None` as shown above).

### 3.6 Canonical Query Merge Behavior for Layer 2 Selections (Open Question 8 — Resolved)

<!-- rev2: B2 — APPEND mode always; single-token substitute branch deleted entirely -->
The existing `ConversationSession.compute_canonical_query` works by:
- If the last turn had `decision.triggered_by = T`, substitute T with the new user input (whole-word,
  case-insensitive regex substitution).
- Else: append.

For Layer 2, **`triggered_by = None` always**. There is no single trigger word to substitute. Layer 2
always uses **APPEND mode**.

Rationale: single-token substitution like `"kidney"` → `"malignant neoplasm of kidney"` loses the user's
original framing. The registry-trigger model (Layer 1) is the correct place for substitute-mode because the
trigger word is the entire query intent. Layer 2 is fuzzy/probabilistic and should preserve user input via
append — the user's original term stays in the canonical query.

The result for "kidney" → user picks option → canonical becomes `"kidney malignant neoplasm of kidney"`.
This is slightly redundant but acceptable because:
1. On the next turn, the SNOMED strategy will find "malignant neoplasm of kidney" (exact match, 0.99).
2. The filter extractor (LLM path) is robust to redundant words.
3. Append-mode bloat is bounded by the 500-char preprocessor + 3-turn cap (monitored, not enforced).

The previous design's single-token `triggered_by` branch (conditional on `len(canonical_query.split()) == 1`)
has been **deleted**.

### 3.7 Failure Mode: Embedding Model Unavailable (Open Question 11 — Resolved)

**Layer 2 silently no-ops.** It does not raise. The `EmbeddingAmbiguityGate.__init__` sets
`self._has_embeddings = False` if `get_top_neighbors` is absent, and `evaluate()` returns `None`
immediately. On the first occurrence where embeddings are unavailable, one INFO log is emitted
(not repeated per-call). The pipeline then proceeds to the clinical-intent gate, which may reject the
query — the same behavior as today. This is the correct tradeoff for a POC: embedding is a graceful
enhancement, not a required safety gate.

---

## 4. Pipeline Integration

### 4.1 Updated Step Ordering (Open Question 10 — Resolved)

<!-- rev2: M4 — negation flow made explicit; Layer 2 receives post-NegationAnnotator matches -->
Layer 2 fires **between step 5 (negation annotation) and step 6 (clinical-intent gate)** in
`_run_extraction_path`. Specifically, it is inserted as **Step 5b**.

Justification:
- It must be **after step 5** (negation annotation): `snomed_matches` passed to
  `EmbeddingAmbiguityGate.evaluate()` are post-`NegationAnnotator`. The `.negated` field is populated on
  each match at step 5. The gate filters negated matches out immediately:
  `qualifying = [m for m in matches if not m.negated]`. This is the source-of-truth filter for both
  Signal A and Signal B.
- It must be **before step 6** (clinical-intent gate) because the purpose of Layer 2 is to intercept the
  query before it is rejected by the clinical-intent gate.
- It must be **after step 4** (parallel extraction) because it needs the `snomed_matches` list.
- It does NOT fire before the pre-extraction `SufficiencyGate` (step 3) because the registry check is
  cheaper (pure string regex) and handles the known-trigger cases already. Layer 2 is the fallback.

**Updated step sequence in `_run_extraction_path` (starting at step 5):**

```
Step 4:  Parallel extraction (filter LLM + SNOMED search)
Step 5:  NegationAnnotator.annotate()     [existing, line ~329]
         → snomed_matches[i].negated is now populated
Step 5b: EmbeddingAmbiguityGate.evaluate(canonical, snomed_matches)  ← NEW
         snomed_matches are post-NegationAnnotator; gate filters negated internally
         IF decision is not None (gate fired):
             build clarification, append turn, return ClarificationOutput
Step 6:  Clinical-intent gate             [existing, line ~343]
         IF 0 qualifying AND 0 filters: raise PreprocessorError
Step 7:  SufficiencyGate.post_extraction_check()   [existing, line ~365]
Step 8:  GeoNormalizer                    [existing, line ~388]
Step 9:  assert_safe                      [existing, line ~399]
Step 10: assemble + append turn           [existing, line ~402]
```

### 4.2 New Log Path Enum Value

```python
LOG_PATH_EMBEDDING_AMBIGUITY = "embedding_ambiguity_clarification"
```

This is added to `pipeline.py` alongside the existing `LOG_PATH_*` constants. The name
`embedding_ambiguity_clarification` is preferred over shorter alternatives for consistency with existing
log path naming conventions.

### 4.3 Exact `pipeline.py` Changes (Pseudocode)

In `NLPPipeline.__init__`, after step 4 (remaining components), add:

```
# Step 4e: Embedding ambiguity gate (Layer 2)
self._embedding_gate = EmbeddingAmbiguityGate(snomed_strategy=self._snomed)
```

In `_run_extraction_path`, between the negation annotation block (step 5, line ~329) and the
clinical-intent gate (step 6, line ~343), insert:

```
# ── Step 5b: Embedding-based ambiguity fallback (Layer 2) ────────────────
layer2_decision = self._embedding_gate.evaluate(canonical, snomed_matches)
IF layer2_decision is not None:
    log_path = LOG_PATH_EMBEDDING_AMBIGUITY
    clarification = self._assembler.build_clarification(layer2_decision, session, start)
    session.append_turn(Turn(
        turn_index=len(session.turns),
        user_input=user_text,
        canonical_query=canonical,
        decision=layer2_decision,
        filters=None,         # extraction ran but we are treating this as clarification turn
        snomed_matches=snomed_matches,
        geo=None,
        timestamp=time.time(),
    ))
    self._log_turn(
        session, layer2_decision, log_path, start,
        snomed_count=len(snomed_matches),
        filter_count=0,   # filters ran, but this turn is being surfaced as clarification
    )
    RETURN clarification
```

Note: `filters=None` in the `Turn` is consistent with how clarification turns are stored in the existing
step 3 path (line ~207–218). The SNOMED matches are retained on the turn record for debugging/inspection.

### 4.4 Interaction with `DEFAULT_CONDITION_PROMPT` (Open Question — Resolved)

`DEFAULT_CONDITION_PROMPT` fires at step 7 (`post_extraction_check`) when filters are set but SNOMED
returns 0 qualifying matches. Layer 2 (step 5b) fires only on 0 qualifying OR genuine ambiguity. These
are distinct:

- Pure "kidney" with no filters: Layer 2 fires at step 5b (if mid-band neighbors present); step 7 never
  reached.
- "kidney at Mayo": Layer 2 fires at step 5b if no high-conf match. The `filters.site_name` will be set
  but this is irrelevant — Layer 2 has priority.
- "Mayo Clinic phase 3": no SNOMED match of any confidence → Layer 2 mid-band check may fail (no medical
  terms at all) → step 6 clinical-intent check → step 7 `DEFAULT_CONDITION_PROMPT` fires because filter
  is set.

The existing `DEFAULT_CONDITION_PROMPT` path is unaffected.

### 4.5 New Constants Summary

<!-- rev2: LAYER2_HIGH_THRESHOLD removed; LAYER2_MIN_NEIGHBORS updated to 3 -->
All new constants go in `src/sufficiency_gate.py` except `LOG_PATH_EMBEDDING_AMBIGUITY` (in `pipeline.py`).
`MIN_CONFIDENCE` is imported from `src/assembler.py` (or moved to a shared `src/constants.py` module if
the implementer prefers a single import point — either is acceptable, but do NOT duplicate the value):

```python
# Layer 1 constants (sufficiency_gate.py)
MIN_DERIVED_TOKEN_LEN       = 4
MIN_DERIVED_TERM_FREQUENCY  = 2

# Layer 2 constants (sufficiency_gate.py)  <!-- rev2 -->
# LAYER2_HIGH_THRESHOLD removed — use MIN_CONFIDENCE (= 0.60) from assembler.py
LAYER2_LOW_THRESHOLD        = 0.42
LAYER2_MIN_NEIGHBORS        = 3      # raised from 2 per B1  <!-- rev2 -->
LAYER2_MIN_GENUINE_AMBIG    = 3
LAYER2_SPREAD_THRESHOLD     = 0.08

# pipeline.py
LOG_PATH_EMBEDDING_AMBIGUITY = "embedding_ambiguity_clarification"
```

---

## 5. Schema Additions

### 5.1 New `reason` Enum Value on `SufficiencyDecision`

The `reason` field comment in `SufficiencyDecision` (line 180–186 of `sufficiency_gate.py`) gains:

```
#   "embedding_ambiguity" — Layer 2 embedding gate fired (step 5b)
```

No structural change to the Pydantic model itself.

### 5.2 New `EmbeddingAmbiguityGate` (No New Pydantic Model)

`EmbeddingAmbiguityGate` is a plain Python class, not a Pydantic model. It returns the existing
`SufficiencyDecision` with a new reason value. No new output types.

### 5.3 `AmbiguousEntry` — No Schema Change

<!-- rev2: trust model note added per M8 -->
Auto-derived entries use exactly the same `AmbiguousEntry` frozen Pydantic V2 model. The
`trigger="__embedding_fallback__"` sentinel is a naming convention, not a new field.

**Trust model:** `data/snomed_clinical_trials.csv` is in-repo and version-controlled. Layer 1 trusts CSV
row contents the same way Layer 1's existing `ALIAS_DICTIONARY` trusts hardcoded values. Supply-chain
integrity (ensuring the CSV has not been tampered with) is the deploy-pipeline's responsibility and is not
enforced by runtime code. This mirrors the trust model of every other in-repo static data asset.

### 5.4 New Method on `HybridCascadeStrategy`

<!-- rev2: n default updated to 15 = AMBIG_JSON_MAX_OPTIONS * 3 per B3 -->
```python
def get_top_neighbors(
    self,
    query_text: str,
    n: int = 15,   # AMBIG_JSON_MAX_OPTIONS * 3; over-fetch for dedup  <!-- rev2 -->
    low_threshold: float = 0.42,
    high_threshold: float = MIN_CONFIDENCE,  # imported; not duplicated  <!-- rev2: B5 -->
) -> list[tuple[str, str, float]]:
    """
    Returns (concept_id, preferred_term, similarity) tuples for the top-n
    preferred_terms whose cosine similarity to query_text falls in [low_threshold, high_threshold).
    Returns [] if semantic_available is False.
    This method is NOT part of the SNOMEDSearchStrategy Protocol.
    Thread-safe: uses same read-only embedder/index as search().
    HIPAA: query_text NOT logged.
    """
```

This is added to `hybrid_cascade.py`. Since it is not part of the Protocol, other strategy implementations
(aho_corasick, ngram_lookup) do not need to implement it. The duck-type check in `EmbeddingAmbiguityGate`
handles this gracefully.

---

## 6. Backwards Compatibility

### 6.1 What Does NOT Break

| Component | Impact |
|---|---|
| `AmbiguousTermsRegistry.find_trigger()` | No change to signature or behavior. Auto-derived entries participate in the existing regex-based trigger detection. |
| `SufficiencyGate.evaluate()` | No change. |
| `SufficiencyGate.post_extraction_check()` | No change. |
| `DEFAULT_CONDITION_PROMPT` | No change. |
| All existing `ambiguous_terms.json` entries | Hand-curated entries always win on key collision (§2.8). |
| `HybridCascadeStrategy.search()` | No change to existing method. |
| `AhoCorasickStrategy`, `NGramLookupStrategy` | No change. `EmbeddingAmbiguityGate` gracefully no-ops for these. |
| `NLPOutput` / `ClarificationOutput` schema | No change. |
| `batch_eval.py` | No change. Layer 2 clarifications appear as `type=clarification` in output — already handled by `expected_type` column. |
| `tests/run_tests.py` (OBSOLETE-AT-SCALE shim) | No change — it bypasses multi-turn via `run()`. |

### 6.2 What Changes

| Component | Change | Risk |
|---|---|---|
| `AmbiguousTermsRegistry.__init__` | Adds derived entry build + merge. Startup takes slightly longer (one extra CSV read — same file already read by `_derive_overrides`). | Low. CSV is already in memory; a second `pd.read_csv` is fast. |
| `_run_extraction_path` in `pipeline.py` | Adds step 5b between existing steps 5 and 6. | Low. Returns early (clarification) or passes through (None). |
| `SufficiencyDecision.reason` comment | Adds `"embedding_ambiguity"` to the enum comment. | None — comment only. |
| `HybridCascadeStrategy` | Adds `get_top_neighbors` method. | Low. New method; no mutation of existing methods. |
| Log line `path` enum | New value `"embedding_ambiguity_clarification"`. Downstream log parsers expecting the existing 5 values would need updating. | Medium. Documented in context.md; parsers must whitelist new value. |
| `tests/test_sufficiency_gate.py` | Must be extended (see §7). | Low for existing tests — no existing test breaks. |

### 6.3 `AMBIG_STRICT_VALIDATION` Interaction

<!-- rev2: M5 — derived entries explicitly not subject to strict_validation -->
Auto-derived entries do NOT go through `strict_validation`. They are built separately and merged before
the regex compilation. If derived entry build fails entirely (exception), the registry falls back to
JSON-only — `strict_validation` mode is unaffected. Derived entries are an enhancement, not a contract.
The `strict_validation` flag governs only JSON-loaded entries.

### 6.4 Single-CSV-Read Optimization (Suggested)

Currently `AmbiguousTermsRegistry._derive_overrides` and the new `_build_derived_entries` each call
`pd.read_csv`. The implementer should refactor to read the CSV once in `__init__` and pass the DataFrame
to both methods. This is a quality-of-life change, not a correctness change.

### 6.5 Backwards-Compatibility Fix: `re.sub` Regex-Replacement Injection <!-- rev2: B6 -->

**Pre-existing bug in `src/conversation.py:174`** (not introduced by this design, but inherited by the
Layer 2 code path):

`pattern.sub(user_input, self.canonical_query)` passes user input directly as the `re.sub` replacement
argument. The `re` module interprets backreference patterns (`\1`, `\g<name>`, etc.) in the replacement
string. User input containing `\1` raises `re.error`, breaking the pipeline.

**Fix:** replace with the lambda form, which treats the replacement as a literal:

```python
# Before (buggy):
result = pattern.sub(user_input, self.canonical_query)

# After (safe):  <!-- rev2 -->
result = pattern.sub(lambda _m: user_input, self.canonical_query, count=1)
```

The lambda form bypasses backreference interpretation entirely. The `count=1` is consistent with the
existing whole-word substitution intent (one trigger word per canonical query).

This fix should be applied in the same change set as the Layer 2 work because Layer 2 relies on the same
`compute_canonical_query` code path (APPEND mode traverses a different branch, but the bug remains latent
in the same function and is easiest to fix in context).

---

## 7. Test Plan

### 7.1 Parity Gate — Existing Tests Must Not Break

<!-- rev2: M6 — hard numbers specified for parity gate -->
Before implementing, run the full existing test suite and record baselines:

```
pytest tests/test_sufficiency_gate.py tests/test_conversation.py \
       tests/test_snomed_strategies.py tests/test_negation.py tests/test_llm_provider.py
```

**Hard pass criteria (M6):**
- 100% of existing tests in `test_sufficiency_gate.py`, `test_conversation.py`,
  `test_snomed_strategies.py`, `test_negation.py`, `test_llm_provider.py` pass post-change. Any failure
  is a regression and blocks merge.
- SNOMED recall on `tests/batch_test_cases.csv` ≥ baseline (run `batch_eval.py --limit 100` before merge;
  record number; post-change run must be ≥ that number). Layer 1/2 only adds clarification paths; it must
  not degrade existing SNOMED search results.
- Phase 4 anatomy smoke set (§7.3): ≥ 8/10 terms fire Layer 2 with ≥ 3 options. Zero terms may return
  empty (0 options) when Layer 2 fires.

### 7.2 New pytest Cases for `tests/test_sufficiency_gate.py`

#### Layer 1 (Auto-Derived Triggers)

```
T-L1-01: test_derived_triggers_loaded
  Assert: after init, registry.entry_count > len(json_entries).
  (Verifies derived build ran and merged at least some entries.)

T-L1-02: test_hand_curated_wins_on_collision
  Add "lung" to a mock ambiguous_terms.json with custom options.
  Assert: registry._entries["lung"].options == mock_options.
  (Verifies JSON wins on collision.)

T-L1-03: test_derived_trigger_fires_for_lung
  Input: "lung phase 2"
  Assert: find_trigger("lung phase 2") returns a non-None (trigger, entry) pair.
  (Verifies a known qualifying derived token fires the registry.)

T-L1-04: test_derived_trigger_suppressed_by_override
  Input: "malignant neoplasm of lung" (a preferred_term, hence an override_term)
  Assert: find_trigger("malignant neoplasm of lung") returns None.
  (Verifies override suppression works for derived entries.)

T-L1-05: test_derived_entry_options_are_lowercase
  For any derived entry, assert all option strings are lowercase (matching CSV convention).
  (Verifies .title() is NOT applied — rev2)  <!-- rev2: B4 -->

T-L1-06: test_anatomy_stopwords_not_derived
  For each token in _ANATOMY_STOPWORDS (including "human"), assert it is NOT a key in
  registry._entries unless also present in ambiguous_terms.json.

T-L1-07: test_derived_build_csv_failure_graceful
  Patch _build_derived_entries to raise. Assert registry still loads from JSON.
  Assert registry.entry_count == len(json_entries).

T-L1-08: test_min_options_not_met_skips_token
  Mock CSV where a token appears in only 1 preferred_term.
  Assert the token is NOT in registry._entries.
```

#### Layer 2 (Embedding Ambiguity Gate)

```
T-L2-01: test_layer2_no_op_when_no_embeddings
  Create EmbeddingAmbiguityGate with a mock strategy that lacks get_top_neighbors.
  Assert: evaluate(query, matches) returns None.

T-L2-02: test_layer2_signal_a_fires
  Create mock snomed_matches with 3 matches in [0.42, 0.60) band, 0 at >= 0.60.
  Assert: evaluate() returns SufficiencyDecision(reason="embedding_ambiguity").
  (Verifies LAYER2_MIN_NEIGHBORS = 3 is the threshold — rev2)  <!-- rev2: B1 -->

T-L2-03: test_layer2_signal_a_does_not_fire_with_two_mid_band
  Mock matches: 2 matches in [0.42, 0.60). Assert: evaluate() returns None.
  (Verifies LAYER2_MIN_NEIGHBORS = 3; 2 is insufficient — rev2)  <!-- rev2: B1 -->

T-L2-04: test_layer2_signal_b_fires_genuine_ambiguity
  Mock 3 qualifying matches (>= 0.60) with spread < 0.08.
  Assert: evaluate() returns SufficiencyDecision(reason="embedding_ambiguity").

T-L2-05: test_layer2_signal_b_no_fire_clear_winner
  Mock 3 qualifying matches; spread = 0.15. Assert: evaluate() returns None.

T-L2-06: test_layer2_triggered_by_always_none
  Input canonical = "kidney" (1 token), signal A fires.
  Assert: decision.triggered_by is None.  (APPEND mode always — rev2: B2)  <!-- rev2 -->

T-L2-07: test_layer2_triggered_by_none_multi_token_query
  Input canonical = "kidney phase 3" (3 tokens), signal A fires.
  Assert: decision.triggered_by is None.

T-L2-08: test_layer2_insufficient_options_no_fire
  Mock candidates that deduplicate to < 3 distinct concept codes.
  Assert: evaluate() returns None.

T-L2-09: test_layer2_negated_matches_excluded
  Mock: 3 matches in [0.42, 0.60) band, all with negated=True.
  Assert: evaluate() returns None (negated matches do not count toward mid_band).
  (Validates M4 negation filtering)  <!-- rev2: M4 -->
```

#### Pipeline Integration

```
T-P-01: test_pipeline_kidney_produces_clarification
  Input: "kidney"
  Expectation: pipeline returns ClarificationOutput (not PreprocessorError).
  (Integration test — requires real SNOMED strategy with embeddings.)

T-P-02: test_pipeline_layer2_step_ordering
  Verify via mock that EmbeddingAmbiguityGate.evaluate is called AFTER
  NegationAnnotator.annotate and BEFORE clinical-intent rejection.

T-P-03: test_pipeline_regex_sub_literal_replacement  <!-- rev2: B6 -->
  Input canonical_query = "kidney phase 3", trigger = "kidney",
  user_input = r"something\1with\g<name>backrefs".
  Assert: compute_canonical_query does NOT raise re.error.
  Assert: result contains the literal backref string, not a substitution.
```

### 7.3 Batch Evaluation (`batch_eval.py`) Impact

<!-- rev2: M3/M6 — anatomy smoke set formalized as pre-merge validation gate -->
**Pre-merge validation gate — anatomy smoke set:**

Run the following 10 known-ambiguous single-token anatomy queries through the pipeline with real SNOMED
embeddings. Each must fire Layer 2 with ≥ 3 options and none may return empty (0 options):

```
kidney, lung, bone, skin, thyroid, brain, liver, heart, blood, eye
```

Expected pass count: **≥ 8/10**. Failing 2 is acceptable (some terms may legitimately resolve with high
confidence or fall below the low threshold — log which 2 fail and document). Failing ≥ 3 is a blocker.

Regression note: "blood" and "eye" are the most likely to not fire Layer 2 (they may resolve with
high-confidence preferred_terms or produce < 3 mid-band neighbors). If they fail, document and adjust
`LAYER2_LOW_THRESHOLD` before merge.

Add rows to `tests/batch_test_cases.csv` for the gap terms:

```
id,category,input,expected_type,expected_clarification_field,description
B-101,missing_condition,kidney,clarification,condition,bare anatomy word
B-102,missing_condition,lung,clarification,condition,bare anatomy word (L1 if ≥2 preferred_terms)
B-103,missing_condition,kidney phase 3,clarification,condition,anatomy + filter
B-104,missing_condition,bone marrow,search,,should match 'stem cell transplant'
B-105,missing_condition,GI,clarification,condition,abbreviation for gastrointestinal
```

The `expected_type=clarification` rows verify that the pipeline now returns a clarification instead of
"No clinical content found." The `expected_clarification_field` column can be mapped to a new check in
`batch_eval.py` comparing `output.type == "clarification"`.

**Regex performance note:** With many derived triggers, the compiled regex alternation may grow large.
Monitor in `batch_eval.py` timing report. If the pattern compilation or match time becomes notable,
refactor to use Aho-Corasick for trigger lookup. Defer until timing data is available.

---

## 8. Open Risks and Accepted Tradeoffs

| Risk | Severity | Mitigation |
|---|---|---|
| `LAYER2_LOW_THRESHOLD = 0.42` may be too permissive for some noise terms | Medium | Configurable constant; tune after running batch eval on 100-case CSV. Initial value 0.42 is conservative but can be raised. |
| Auto-derived triggers may fire for tokens the team does not want (e.g., "human" formerly produced spurious hits) | Low | "human" added to `_ANATOMY_STOPWORDS`. Additional tokens found in batch eval are added there. The JSON-wins collision mechanism allows hand-curation to suppress any unwanted derived trigger. |
| Layer 2 adds latency for queries that go through it (embedding call) | Low | `get_top_neighbors` uses the already-warmed embedding index; no new model load. Expected < 5ms on GPU, < 50ms on CPU for a single query. The 15s parallel timeout (PARALLEL_TIMEOUT_SECONDS) is not affected. |
| `get_top_neighbors` is not on the `SNOMEDSearchStrategy` Protocol — breaks type-safety for non-cascade strategies | Accepted | Duck-type check in `EmbeddingAmbiguityGate.__init__`. Non-cascade strategies do not advertise embeddings. This is a deliberate protocol extension point. |
| New `reason="embedding_ambiguity"` value in structured logs — downstream parsers may break | Low-Medium | Documented. Parsers using `IN` allowlist must add the new value. |
| Layer 2 APPEND mode produces redundant canonical queries | Low | Bounded by 500-char preprocessor + 3-turn cap. Monitored, not enforced. |
| Derived entry build runs on every pipeline startup, adding ~100ms | Low | One extra `pd.read_csv` of 116 rows. Negligible. If it matters, cache derived entries at class level. |
| `AMBIG_DERIVED_DEBUG=true` env var logs derived trigger text — HIPAA risk if used in production | Medium | Implementation must check `os.environ.get("ENV") == "dev"` before honoring the flag. Only trigger token LENGTH is logged in non-dev environments. Flag is silently ignored in production. |
| Tied-confidence matches in dedup produce non-deterministic ordering | Low | Python 3.7+ dict insertion order is the tie-breaker. Deterministic within a process run. |
| CSV supply-chain integrity | Low | In-repo, version-controlled. Same trust level as `ALIAS_DICTIONARY`. Deploy pipeline is responsible for supply-chain verification. |

---

## 9. Implementation Order

### Phase 0: Pre-work (no new features — safety gate for regressions)
1. Run full existing pytest suite; record pass counts as parity baseline.
2. Run `batch_eval.py --limit 100` to confirm current SNOMED recall baseline (record number).
3. Confirm `data/snomed_clinical_trials.csv` is on the path the implementation will use.
4. Apply the `re.sub` lambda fix in `src/conversation.py:174` (§6.5) — low-risk, pre-existing bug.

### Phase 1: Layer 1 — Auto-Derived Triggers (file-first order)

**File 1: `src/sufficiency_gate.py`**
- Add `_ANATOMY_STOPWORDS` constant (including "human").
- Add `MIN_DERIVED_TOKEN_LEN`, `MIN_DERIVED_TERM_FREQUENCY` constants.
- Add `_build_derived_entries(csv_path)` private function (module-level or static).
  Synonym-rescue path NOT implemented — see §2.4.
- Add `_merge_entries(json_entries, derived_entries)` private function with checksum log.
- Add `_select_derived_options(token, source_terms, df)` helper (returns raw CSV values, no `.title()`).
- Add `_build_derived_question(token)` helper.
- Modify `AmbiguousTermsRegistry.__init__`: after JSON load loop, call `_build_derived_entries` and
  `_merge_entries`; update `self._entries` before regex compilation.
- **Parity gate:** all existing tests in `test_sufficiency_gate.py` must still pass.

**File 2: `tests/test_sufficiency_gate.py`**
- Add test cases T-L1-01 through T-L1-08 (T-L1-05 now checks lowercase, not Title Case).

### Phase 2: Layer 2 — Embedding Ambiguity Gate

**File 3: `src/snomed_search/hybrid_cascade.py`**
- Add `get_top_neighbors(query_text, n=15, low_threshold, high_threshold)` method.
- **Parity gate:** all existing tests in `test_snomed_strategies.py` must still pass.

**File 4: `src/sufficiency_gate.py` (continued)**
- Add `LAYER2_*` constants (`LAYER2_HIGH_THRESHOLD` NOT added; use `MIN_CONFIDENCE` import).
- Add `EmbeddingAmbiguityGate` class with negation filtering and `triggered_by=None` always.
- Extend `SufficiencyDecision.reason` comment.

**File 5: `tests/test_sufficiency_gate.py` (continued)**
- Add test cases T-L2-01 through T-L2-09.

### Phase 3: Pipeline Integration

**File 6: `src/pipeline.py`**
- Add `LOG_PATH_EMBEDDING_AMBIGUITY` constant.
- Add `self._embedding_gate = EmbeddingAmbiguityGate(...)` in `__init__`.
- Insert step 5b block in `_run_extraction_path` (after NegationAnnotator, before clinical-intent gate).
- **Parity gate:** full pytest suite passes; `batch_eval.py --limit 100` SNOMED recall ≥ baseline.

**File 7: `tests/test_sufficiency_gate.py` (continued)**
- Add T-P-01, T-P-02, T-P-03.

### Phase 4: Evaluation

- Run anatomy smoke set (§7.3): "kidney", "lung", "bone", "skin", "thyroid", "brain", "liver", "heart",
  "blood", "eye". Require ≥ 8/10 fire Layer 2 with ≥ 3 options. Zero terms may return empty.
- Run `batch_eval.py` on full 100-case CSV. Confirm:
  - SNOMED recall ≥ baseline.
  - Anatomy gap terms now return `type=clarification`.
- Run QA agent: `python qa_testing/test_agent.py --limit 20`. Confirm injection/harmful categories
  still blocked.
- Tune `LAYER2_LOW_THRESHOLD` if mid-band is producing spurious clarifications on valid medical queries.

### Phase 5: Context Update

- Update `context.md` with new constants table entries, new `EmbeddingAmbiguityGate` in the component
  reference, the updated step ordering diagram, and the `re.sub` fix note.

---

## 10. Revision 2 (Post Review) <!-- rev2 -->

**Date:** 2026-05-11  
**Based on:** QA + Security review of Revision 1

### Blockers Addressed

| ID | Summary | Section(s) Changed |
|---|---|---|
| B1 | `LAYER2_MIN_NEIGHBORS` raised from 2 to 3 (genuine ambiguity = ≥3 options, consistent with Signal B). Signal A definition updated to require `len(mid_band) >= 3`. | §3.2, §4.5, T-L2-02, T-L2-03 |
| B2 | Layer 2 uses APPEND mode always. Single-token substitute branch (`triggered_by = canonical_query.strip().lower()` when 1-token query) deleted entirely. `triggered_by=None` for all Layer 2 fires. Rationale documented. | §3.4, §3.6, T-L2-06 |
| B3 | `get_top_neighbors` `n` parameter defined as `AMBIG_JSON_MAX_OPTIONS * 3 = 15`. Rationale: over-fetch to allow per-code dedup without under-filling final 3–5 option list. `_deduplicate_neighbors` signature note updated. | §3.3, §3.5, §5.4 |
| B4 | `.title()` calls removed from `_select_derived_options` and `_deduplicate_neighbors`. Options returned as raw CSV `preferred_term` values (lowercase). Casing convention aligned with `NLPOutput.snomed_terms[].display`. T-L1-05 updated to assert lowercase. | §2.6, §3.5, T-L1-05 |
| B5 | `LAYER2_HIGH_THRESHOLD = 0.60` constant eliminated. All Layer 2 code now references `MIN_CONFIDENCE` imported from `src/assembler.py`. Single source of truth. | §3.2, §3.4, §4.5, §5.4 |
| B6 | Pre-existing `re.sub` injection bug in `src/conversation.py:174` documented and fix specified. Lambda form `pattern.sub(lambda _m: user_input, self.canonical_query, count=1)` prevents backreference interpretation. New §6.5 added. New test T-P-03 added. Fix placed in Phase 0. | §6.5, §9 Phase 0, T-P-03 |

### Majors Addressed

| ID | Summary | Section(s) Changed |
|---|---|---|
| M1 | Synonym-rescue path (concept-association-losing `dict → set` flattening) removed entirely. Single-CSV-row tokens fall through to Layer 2. §2.4 updated, §2.5 pseudocode `synonym_terms` block removed. | §2.4, §2.5 |
| M2 | Startup health log with checksum added to `_merge_entries`. Optional runtime hit verification documented. | §2.8 |
| M3 | §7.3 smoke set formalized as a hard pre-merge validation gate: 10 anatomy terms, ≥8/10 must fire Layer 2 with ≥3 options, zero may return empty. | §7.3 |
| M4 | Negation flow made explicit: `EmbeddingAmbiguityGate.evaluate()` precondition is post-`NegationAnnotator`. Pipeline step 5b annotation updated. Negated matches filtered before Signal A and B computation. New test T-L2-09 added. | §3.2, §3.4, §4.1, T-L2-09 |
| M5 | `AMBIG_STRICT_VALIDATION` interaction with derived entries made explicit: derived entries bypass `strict_validation` by design. CSV failure behavior = `strict_validation=False` semantics. | §2.10, §6.3 |
| M6 | Parity gate quantified with hard numbers: 100% of named test files pass; SNOMED recall ≥ recorded baseline; anatomy smoke set ≥ 8/10. | §7.1 |
| M7 | Documented that Layer 2 synthetic `AmbiguousEntry.options` flow through `ResponseAssembler.build_clarification()` which applies existing `html.escape()` — no new sanitization step required. | §3.4 |
| M8 | §5.3 trust model note added: CSV is in-repo, version-controlled, trusted at same level as `ALIAS_DICTIONARY`. Supply-chain integrity is deploy-pipeline responsibility. | §5.3 |

### Minors Folded In

- §1 rewritten: "kidney" reframed as canonical Layer 2 example; Layer 1 handles ≥2-frequency tokens.
- "human" added to `_ANATOMY_STOPWORDS` (§2.3).
- `AMBIG_DERIVED_DEBUG` dev guard noted (§2.8): must check `ENV == "dev"`.
- Log path `embedding_ambiguity_clarification` kept unchanged (consistency wins).
- Append-mode bloat noted as bounded by 500-char preprocessor + 3-turn cap (§3.6).
- Tied-confidence tie-breaker documented: dict insertion order (§3.5, §8).
- Embedding-unavailable INFO log on first occurrence noted (§3.3, §3.7).
- Regex perf with many triggers: defer; monitor in batch_eval timing report (§7.3).

*End of design document.*
