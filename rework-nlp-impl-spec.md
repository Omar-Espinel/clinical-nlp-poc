# NLP Pipeline v2 — Implementation Pseudocode Spec
**Branch: `response` · 2026-05-06 · DEV spec (post-architect revision 2)**

Authority: `rework-nlp-proposal.md` (revision 2). This spec deepens that document.
Pseudocode only — not working code.

---

## 1. `AmbiguousTermsRegistry` — `data/ambiguous_terms.json` + `src/sufficiency_gate.py`

**File paths:**
- `data/ambiguous_terms.json` — data artifact (see schema below)
- `src/sufficiency_gate.py` — `AmbiguousEntry`, `AmbiguousTermsRegistry`, `SufficiencyGate`

**Imports:**
```python
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator

from src.snomed_search.base import SNOMEDSearchStrategy
```

**Constants:**
```python
AMBIG_JSON_MIN_OPTIONS = 3
AMBIG_JSON_MAX_OPTIONS = 5
OPTION_MIN_CONFIDENCE  = 0.85    # options must resolve at this confidence or higher
```

**Pydantic V2 models:**
```python
class AmbiguousEntry(BaseModel):
    model_config = ConfigDict(frozen=True)
    trigger: str                   # lowercase canonical key (e.g. "cancer")
    category: str                  # "indication" | "phase" | "geography" | "population"
    question_template: str         # supports {trigger} and {prior_filters} placeholders
    options: list[str]             # 3-5 display-ready option strings (e.g. "Lung Cancer")
    override_terms: frozenset[str] # post-derivation, all lowercase; prevents clarification skip
    max_options: int               # informational; assembler renders up to this many

    @field_validator("options")
    @classmethod
    def _options_length(cls, v: list[str]) -> list[str]:
        if not (AMBIG_JSON_MIN_OPTIONS <= len(v) <= AMBIG_JSON_MAX_OPTIONS):
            raise ValueError(
                f"options must have {AMBIG_JSON_MIN_OPTIONS}-{AMBIG_JSON_MAX_OPTIONS} entries, "
                f"got {len(v)}"
            )
        return v
```

**`AmbiguousTermsRegistry` class:**
```python
class AmbiguousTermsRegistry:
    """
    Loads data/ambiguous_terms.json, validates every option against the SNOMED strategy,
    derives override_terms for each trigger, and compiles a combined regex for fast
    whole-word trigger detection at runtime.

    Thread-safety: all mutable state is built in __init__ and then read-only.
    search() / find_trigger() are pure reads; safe for concurrent calls.
    """

    def __init__(
        self,
        path: str,
        snomed_strategy: SNOMEDSearchStrategy,
        snomed_csv_path: str,
        strict_validation: bool = True,
    ) -> None:
        # strict_validation=True  → invalid option → logger.error + sys.exit(1)
        # strict_validation=False → invalid option → logger.warning; entry dropped;
        #   gate behaves as if trigger doesn't exist (graceful degrade for rolling deploys)
        self._strategy      = snomed_strategy
        self._csv_path      = snomed_csv_path
        self._strict        = strict_validation
        self._entries: dict[str, AmbiguousEntry] = {}
        self._compiled_triggers: re.Pattern              # set after loop

        # --- Load raw JSON ---
        try:
            raw: dict = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("AmbiguousTermsRegistry: cannot load %s: %s", path, type(exc).__name__)
            # NOT logged: exc message (may contain filesystem path details)
            sys.exit(1)

        invalid_count = 0
        for raw_trigger, data in raw.items():
            trigger_key = raw_trigger.lower().strip()
            try:
                entry = self._validate_and_build(trigger_key, data)
                self._entries[trigger_key] = entry
            except ValueError as exc:
                invalid_count += 1
                if self._strict:
                    logger.error(
                        "ambiguous_terms.json: trigger '%s' failed validation: %s",
                        trigger_key, str(exc),
                        # NOT logged: option text, full data dict
                    )
                    sys.exit(1)
                else:
                    logger.warning(
                        "ambiguous_terms.json: trigger '%s' skipped (strict=False): %s",
                        trigger_key, str(exc),
                    )

        # B3 FIX: guard against empty registry BEFORE attempting regex compilation.
        # An empty alternation r'\b()\b' is malformed and raises re.error.
        # This check must come first — even strict_validation=False can drain all entries.
        if not self._entries:
            # Every entry was invalid — fail regardless of strict_validation.
            # An empty registry means the gate never fires; harmful silence.
            raise ValueError(
                "AmbiguousTermsRegistry: no valid entries after validation "
                f"(total invalid: {invalid_count})"
            )

        # --- Compile combined trigger regex ---
        # Sorted longest-first so longer triggers win on overlap (e.g. "lung disease"
        # before "lung") — re alternation takes first match; order matters.
        sorted_triggers = sorted(self._entries.keys(), key=len, reverse=True)
        pattern_body = "|".join(re.escape(t) for t in sorted_triggers)
        self._compiled_triggers = re.compile(
            rf"\b({pattern_body})\b",
            re.IGNORECASE,
        )

        logger.info(
            "AmbiguousTermsRegistry loaded: %d triggers, strategy=%s, strict=%s",
            len(self._entries), self._strategy.name, strict_validation,
            # NOT logged: trigger keys, option text
        )

    def _validate_and_build(self, trigger: str, data: dict) -> AmbiguousEntry:
        """
        Full validation + derivation for a single JSON entry.
        Raises ValueError with reason (no internal details that would leak to logs).
        """
        # --- Step 1: structural validation ---
        required_keys = {"category", "question_template", "options"}
        missing = required_keys - set(data.keys())
        if missing:
            raise ValueError(f"missing required keys: {sorted(missing)}")

        options: list[str] = data["options"]
        if not (AMBIG_JSON_MIN_OPTIONS <= len(options) <= AMBIG_JSON_MAX_OPTIONS):
            raise ValueError(
                f"options count {len(options)} outside [{AMBIG_JSON_MIN_OPTIONS}, {AMBIG_JSON_MAX_OPTIONS}]"
            )

        # --- Step 2: SNOMED resolution check for each option ---
        for opt in options:
            matches = self._strategy.search(opt)
            max_conf = max((m.confidence for m in matches), default=0.0)
            if max_conf < OPTION_MIN_CONFIDENCE:
                raise ValueError(
                    f"option did not resolve at ≥{OPTION_MIN_CONFIDENCE} "
                    f"(max_conf={max_conf:.3f}, strategy={self._strategy.name})"
                    # NOT logged or included: opt text (option text is considered content data)
                )

        # --- Step 3: derive override_terms ---
        manual_overrides: list[str] = data.get("manual_override_terms", [])
        override_terms = self._derive_overrides(trigger, options, manual_overrides)

        # --- Step 4: build frozen model (Pydantic validator re-checks options length) ---
        return AmbiguousEntry(
            trigger=trigger,
            category=data["category"],
            question_template=data["question_template"],
            options=options,
            override_terms=frozenset(override_terms),
            max_options=int(data.get("max_options", AMBIG_JSON_MAX_OPTIONS)),
        )

    def _derive_overrides(
        self,
        trigger: str,
        options: list[str],
        manual_override_terms: list[str],
    ) -> set[str]:
        """
        Auto-derivation algorithm (spec §3.1):

        1. Seed with lowercased option strings.
        2. Scan CSV preferred_terms for whole-word containment of trigger.
        3. Remove the trigger itself (self-defeat fix).
        4. Union with manual_override_terms (lowercased).

        Returns a set[str] of lowercase override terms.
        """
        auto_overrides: set[str] = set(opt.lower() for opt in options)

        # --- CSV scan ---
        try:
            df = pd.read_csv(self._csv_path, dtype=str).fillna("")
        except (OSError, pd.errors.ParserError) as exc:
            # CSV unreadable at derivation time — proceed with options-only overrides.
            # Logged as warning, not error: the strategy already loaded the CSV at init.
            logger.warning(
                "_derive_overrides: CSV read failed (%s); using options-only overrides",
                type(exc).__name__,
                # NOT logged: exc message (may contain path / data details)
            )
            df = pd.DataFrame(columns=["preferred_term"])

        trigger_word_pattern = re.compile(rf"\b{re.escape(trigger)}\b", re.IGNORECASE)
        for preferred_term in df["preferred_term"].str.strip().str.lower():
            if preferred_term and trigger_word_pattern.search(preferred_term):
                auto_overrides.add(preferred_term)

        # --- Self-defeat fix: remove the bare trigger from BOTH sets ---
        # Without this, a CSV row with preferred_term="cancer" would add "cancer"
        # to override_terms, making the bare query "cancer" appear self-overridden
        # and skip clarification entirely.
        auto_overrides.discard(trigger.lower())

        # M3 FIX: also filter the bare trigger from manual_override_terms BEFORE
        # the union. The original code only discarded from auto_overrides; if
        # manual_override_terms contained the trigger string directly, it would
        # survive into final_overrides via the union and re-introduce self-defeat.
        filtered_manual = {
            t.lower().strip()
            for t in manual_override_terms
            if t.lower().strip() != trigger.lower()
        }

        # --- Union with filtered manual overrides ---
        final_overrides = auto_overrides | filtered_manual

        logger.info(
            "_derive_overrides: trigger='%s' → %d overrides (auto=%d, manual=%d)",
            trigger, len(final_overrides), len(auto_overrides), len(manual_override_terms),
            # NOT logged: the override term strings themselves
        )
        return final_overrides

    def find_trigger(self, query: str) -> Optional[tuple[str, AmbiguousEntry]]:
        """
        Returns (trigger, entry) for the FIRST trigger in insertion/JSON order that:
          a) appears as a whole-word match in the query, AND
          b) none of its override_terms appear as whole-word matches in the query.

        Returns None if no trigger fires.
        Iteration order = self._entries insertion order = JSON key order (deterministic).

        Thread-safe: no mutation; _compiled_triggers and _entries are read-only post-init.

        HIPAA: query text is NOT logged anywhere in this method.
        """
        query_lower = query.lower()

        for match_obj in self._compiled_triggers.finditer(query_lower):
            trigger: str = match_obj.group(1).lower()
            entry: AmbiguousEntry = self._entries[trigger]

            # Check whether any override_term is present as whole word
            override_present = False
            for override_term in entry.override_terms:
                if re.search(
                    rf"\b{re.escape(override_term)}\b",
                    query,
                    re.IGNORECASE,
                ):
                    override_present = True
                    break   # one override is enough to suppress this trigger

            if not override_present:
                # Do NOT log the trigger value (it may be a common medical word)
                logger.debug(
                    "find_trigger: fired (trigger redacted), entry.category=%s",
                    entry.category,
                )
                return (trigger, entry)

        return None
```

---

## 2. `SufficiencyGate` — `src/sufficiency_gate.py`

**File:** `src/sufficiency_gate.py`

**Imports:**
```python
import logging
import re
from typing import Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict

from src.sufficiency_gate import AmbiguousEntry, AmbiguousTermsRegistry
from src.snomed_search.base import SNOMEDMatch
from src.filter_extractor import ExtractedFilters
# ConversationSession imported here creates a circular dependency risk;
# use TYPE_CHECKING guard:
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.conversation import ConversationSession
```

**Constants (module-level):**
```python
_SNOMED_CSV_DEFAULT = Path(__file__).parent.parent / "data" / "snomed_clinical_trials.csv"

# Hard-coded fallback for DEFAULT_CONDITION_PROMPT when CSV inspection fails
_FALLBACK_CONDITION_OPTIONS = ["Cancer", "Diabetes", "Heart Disease", "Autoimmune", "Other"]

# Category-seed keywords used for clustering CSV terms into DEFAULT_CONDITION_PROMPT options
_CATEGORY_SEEDS: dict[str, list[str]] = {
    "Cancer":       ["cancer", "carcinoma", "leukemia", "lymphoma", "melanoma", "tumor"],
    "Diabetes":     ["diabetes", "diabetic", "glucose", "insulin"],
    "Heart Disease":["cardiac", "heart", "cardiovascular", "coronary", "myocardial"],
    "Autoimmune":   ["lupus", "rheumatoid", "sclerosis", "inflammatory", "autoimmune"],
    "Neurological": ["alzheimer", "parkinson", "epilepsy", "neurolog", "brain"],
}
```

**`_build_default_condition_options` helper (module-level function, NOT called at import):**
```python
def _build_default_condition_options(csv_path: Path) -> list[str]:
    """
    Scan the SNOMED CSV and return the top-5 most-represented category labels
    from _CATEGORY_SEEDS. Counts how many preferred_terms contain any seed keyword
    (case-insensitive substring, NOT whole-word — intentionally liberal here).

    Returns a list of category names (keys from _CATEGORY_SEEDS) with count > 0,
    sorted descending by count, capped at 5.

    Hard fallback: if CSV unreadable or all counts are zero → _FALLBACK_CONDITION_OPTIONS.

    M8: this function is no longer called at import time. It is called lazily from
    SufficiencyGate.__init__() on the first instantiation and the result is cached
    at the class level. See SufficiencyGate.__init__ docstring.
    """
    try:
        df = pd.read_csv(csv_path, dtype=str).fillna("")
        terms_lower: list[str] = df["preferred_term"].str.lower().tolist()
    except Exception:
        logger.warning(
            "_build_default_condition_options: CSV unreadable; using hard fallback",
            # NOT logged: exception message
        )
        return list(_FALLBACK_CONDITION_OPTIONS)

    category_counts: dict[str, int] = {}
    for category_label, seeds in _CATEGORY_SEEDS.items():
        count = 0
        for term in terms_lower:
            if any(seed in term for seed in seeds):
                count += 1
        if count > 0:
            category_counts[category_label] = count

    if not category_counts:
        return list(_FALLBACK_CONDITION_OPTIONS)

    # Sort by count descending; take top 5
    sorted_categories = sorted(category_counts.items(), key=lambda kv: kv[1], reverse=True)
    top_labels = [label for label, _ in sorted_categories[:5]]

    # Pad to at least 3 with fallbacks if needed
    if len(top_labels) < 3:
        extras = [o for o in _FALLBACK_CONDITION_OPTIONS if o not in top_labels]
        top_labels.extend(extras[:3 - len(top_labels)])

    logger.info(
        "_build_default_condition_options: built %d options from CSV",
        len(top_labels),
        # NOT logged: the option strings themselves
    )
    return top_labels
```

**`SufficiencyDecision` model:**
```python
class SufficiencyDecision(BaseModel):
    model_config = ConfigDict(frozen=True)
    sufficient: bool
    reason: str           # enum: "ok_no_trigger" | "ambiguous_trigger" | "max_turns_reached"
                          #       | "ok_post_extraction" | "filters_without_condition"
    triggered_by: Optional[str] = None        # trigger key (not logged)
    matched_entry: Optional[AmbiguousEntry] = None
```

**`SufficiencyGate` class:**
```python
class SufficiencyGate:
    """
    Deterministic pre-extraction + post-extraction sufficiency decisions.
    No LLM calls. No state mutation. Thread-safe post-init.
    """

    # M8 FIX: DEFAULT_CONDITION_PROMPT is built LAZILY on first SufficiencyGate
    # instantiation, not at module import time. Per architect's erratum (rev 2 §14
    # point 5). The CSV path comes from gate construction (snomed_csv_path parameter),
    # not from a module-level constant. Class-level attribute caches the result so
    # multiple SufficiencyGate instances (e.g., in tests) share one build.
    _default_condition_prompt_cache: Optional[AmbiguousEntry] = None

    def __init__(
        self,
        registry: AmbiguousTermsRegistry,
        snomed_csv_path: Optional[str] = None,
    ) -> None:
        self._registry = registry

        # Lazy build of DEFAULT_CONDITION_PROMPT (class-level memoization)
        if SufficiencyGate._default_condition_prompt_cache is None:
            csv_path = Path(snomed_csv_path) if snomed_csv_path else _SNOMED_CSV_DEFAULT
            options = _build_default_condition_options(csv_path)
            SufficiencyGate._default_condition_prompt_cache = AmbiguousEntry(
                trigger="__default_condition__",
                category="indication",
                question_template=(
                    "What medical condition or area of research are you interested in?"
                ),
                options=options,
                override_terms=frozenset(),  # post-extraction prompt — no override suppression
                max_options=5,
            )
        self._default_condition_prompt: AmbiguousEntry = (
            SufficiencyGate._default_condition_prompt_cache
        )

    def evaluate(
        self,
        canonical_query: str,
        session: "ConversationSession",
    ) -> SufficiencyDecision:
        """
        Pre-extraction gate. Three rules only (no rule 4 — see proposal §3.2).

        HIPAA: canonical_query NOT logged. Only sufficient bool and reason enum are logged.
        """
        # Rule 1: max-turns escape valve.
        # session.is_max_turns_reached() is the single source of truth for the counter.
        if session.is_max_turns_reached():
            logger.info(
                "SufficiencyGate.evaluate: sufficient=True reason=max_turns_reached "
                "session_id=%s turn_count=%d",
                session.session_id,
                len(session.turns),
                # NOT logged: canonical_query, turn content
            )
            return SufficiencyDecision(sufficient=True, reason="max_turns_reached")

        # Rule 2: registry trigger lookup (with override-term filtering).
        hit = self._registry.find_trigger(canonical_query)
        if hit is not None:
            trigger, entry = hit
            logger.info(
                "SufficiencyGate.evaluate: sufficient=False reason=ambiguous_trigger "
                "category=%s session_id=%s",
                entry.category,
                session.session_id,
                # NOT logged: trigger value, canonical_query, entry.options
            )
            return SufficiencyDecision(
                sufficient=False,
                reason="ambiguous_trigger",
                triggered_by=trigger,
                matched_entry=entry,
            )

        # Rule 3: default — sufficient.
        logger.info(
            "SufficiencyGate.evaluate: sufficient=True reason=ok_no_trigger session_id=%s",
            session.session_id,
        )
        return SufficiencyDecision(sufficient=True, reason="ok_no_trigger")

    def post_extraction_check(
        self,
        snomed_matches: list[SNOMEDMatch],
        filters: ExtractedFilters,
    ) -> SufficiencyDecision:
        """
        Post-extraction safety check.
        Fires when the algorithmic SNOMED search found zero high-confidence matches
        but the LLM extracted at least one non-null filter field.

        This catches "What's at Mayo?" — geographic/site interest with no condition.

        Threshold: SNOMEDMatch.confidence >= 0.60 AND not negated.
        Uses same threshold as assembler MIN_CONFIDENCE.
        """
        high_conf_matches = [
            m for m in snomed_matches
            if m.confidence >= 0.60 and not m.negated
        ]

        if not high_conf_matches and _any_filter_set(filters):
            logger.info(
                "SufficiencyGate.post_extraction_check: sufficient=False "
                "reason=filters_without_condition filter_count=%d",
                _count_set_filters(filters),
                # NOT logged: filter values, snomed_matches display strings
            )
            return SufficiencyDecision(
                sufficient=False,
                reason="filters_without_condition",
                triggered_by=None,
                matched_entry=self._default_condition_prompt,  # M8: lazy instance attr
            )

        logger.info(
            "SufficiencyGate.post_extraction_check: sufficient=True "
            "reason=ok_post_extraction snomed_count=%d",
            len(high_conf_matches),
        )
        return SufficiencyDecision(sufficient=True, reason="ok_post_extraction")


def _any_filter_set(filters: ExtractedFilters) -> bool:
    """True if at least one non-null, non-empty filter field has been extracted."""
    return bool(
        filters.investigator_name.value
        or filters.site_name.value
        or filters.city.value
        or filters.state.values          # list[str] — non-empty list is truthy
        or filters.phase.value
    )

def _count_set_filters(filters: ExtractedFilters) -> int:
    """Count of non-null filter fields. Used only in log lines."""
    return sum([
        bool(filters.investigator_name.value),
        bool(filters.site_name.value),
        bool(filters.city.value),
        bool(filters.state.values),
        bool(filters.phase.value),
    ])
```

---

## 3. `HybridCascadeStrategy` — `src/snomed_search/hybrid_cascade.py`

**File:** `src/snomed_search/hybrid_cascade.py`

**Imports:**
```python
import logging
import re                             # M1: needed for re.finditer in _tokenize_with_offsets
from dataclasses import replace
from pathlib import Path
from typing import Optional

import pandas as pd
from rapidfuzz import fuzz, process as rf_process

from src.snomed_search.base import SNOMEDMatch, SNOMEDSearchStrategy
from src.snomed_resolver import ALIAS_DICTIONARY   # preserved alias dict
```

**Constants:**
```python
FUZZY_CUTOFF_DEFAULT    = 88
SEMANTIC_THRESHOLD_DEFAULT = 0.82
EMBEDDING_MODEL_DEFAULT = "all-MiniLM-L6-v2"
MIN_RESIDUAL_LEN        = 2    # residual spans shorter than this are dropped as noise
MAX_NGRAM_WINDOW        = 4    # token window for exact/synonym pass
```

**Thread-safety note (inline):**
```
All instance attributes after __init__ are read-only:
  - _exact_index, _synonym_index, _alias_dict: dict lookups, no mutation
  - _embedder: SentenceTransformer.encode() is documented thread-safe
  - _collection (ChromaDB EphemeralClient): query() is thread-safe per ChromaDB docs
  - _np_embeddings: numpy array, read-only after construction
  - _np_terms: list, read-only after construction
  - _fuzzy_cutoff, _semantic_threshold: primitive floats

No locks required. Multiple concurrent search() calls are safe.
```

**Class:**
```python
class HybridCascadeStrategy:
    """
    Hybrid cascade: exact → synonym → fuzzy → semantic.
    Production default. Ports existing snomed_resolver.py behavior.

    Candidate extraction:
      Stages 1+2 (exact/synonym): enumerate all 1..MAX_NGRAM_WINDOW token windows.
      Stages 3+4 (fuzzy/semantic): operate ONLY on character spans NOT already matched
        (residual spans). This prevents fuzzy/semantic from re-matching terms that
        already have a higher-quality exact/synonym hit.
    """

    name = "hybrid_cascade"

    def __init__(
        self,
        dictionary_path: str,
        fuzzy_cutoff: int = FUZZY_CUTOFF_DEFAULT,
        semantic_threshold: float = SEMANTIC_THRESHOLD_DEFAULT,
        embedding_model: str = EMBEDDING_MODEL_DEFAULT,
        **kwargs,
    ) -> None:
        self._fuzzy_cutoff      = fuzzy_cutoff
        self._semantic_threshold = semantic_threshold
        self._exact_index:   dict[str, dict] = {}
        self._synonym_index: dict[str, dict] = {}
        self._alias_dict:    dict[str, str]  = {}    # validated alias → preferred_term
        self.semantic_available = False
        self._collection    = None    # ChromaDB collection or None
        self._embedder      = None    # SentenceTransformer or None
        self._np_embeddings = None    # numpy float32 matrix or None
        self._np_terms:     list[str] = []

        self._load_csv(dictionary_path)
        self._validate_aliases()
        self._build_semantic_index(embedding_model)
        self._ready = True

    def _load_csv(self, csv_path: str) -> None:
        """Port of SNOMEDResolver._load_csv(). Builds exact_index and synonym_index."""
        df = pd.read_csv(csv_path, dtype=str).fillna("")
        for _, row in df.iterrows():
            concept_id    = row["concept_id"].strip()
            preferred_term = row["preferred_term"].strip().lower()
            synonyms_raw  = row["synonyms"].strip()
            record = {"concept_id": concept_id, "preferred_term": preferred_term}
            self._exact_index[preferred_term] = record
            for syn in synonyms_raw.split("|"):
                syn_clean = syn.strip().lower()
                if syn_clean:
                    self._synonym_index[syn_clean] = record
        logger.info(
            "HybridCascadeStrategy: loaded %d terms, %d synonyms",
            len(self._exact_index), len(self._synonym_index),
        )

    def _validate_aliases(self) -> None:
        """
        Port of SNOMEDResolver._validate_aliases().
        Uses module-level ALIAS_DICTIONARY from snomed_resolver.py (shared source of truth).
        Invalid alias targets (not in exact_index) are skipped with a WARNING.
        NOT logged: alias key strings (they are medical terms; HIPAA-adjacent).
        """
        for alias, target in ALIAS_DICTIONARY.items():
            alias_key  = alias.lower().strip()
            target_key = target.lower().strip()
            if target_key in self._exact_index:
                self._alias_dict[alias_key] = target_key
            else:
                logger.warning(
                    "HybridCascadeStrategy: alias target not in CSV — alias skipped "
                    "(alias_len=%d, target_len=%d)",
                    len(alias_key), len(target_key),
                    # NOT logged: alias or target strings
                )

    def _build_semantic_index(self, embedding_model: str) -> None:
        """
        Port of SNOMEDResolver._build_chroma_index().
        Tries ChromaDB first; falls back to numpy cosine similarity.
        Never raises — all exceptions handled internally.
        Sets self.semantic_available = True on success of either backend.
        """
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(embedding_model)
        except Exception as exc:
            self.semantic_available = False
            logger.warning(
                "HybridCascadeStrategy: sentence-transformers unavailable — "
                "semantic disabled (%s)", type(exc).__name__,
            )
            return

        terms = list(self._exact_index.keys())

        # --- Attempt ChromaDB ---
        try:
            import chromadb
            client = chromadb.EphemeralClient()
            self._collection = client.create_collection(
                name="snomed_clinical_trials_v1",
                metadata={"hnsw:space": "cosine"},
            )
            ids = [
                self._exact_index[t]["concept_id"] + "_" + str(i)
                for i, t in enumerate(terms)
            ]
            embeddings = self._embedder.encode(terms, show_progress_bar=False).tolist()
            self._collection.add(documents=terms, embeddings=embeddings, ids=ids)
            self.semantic_available = True
            self._np_embeddings = None    # chromadb path: numpy matrix not needed
            logger.info(
                "HybridCascadeStrategy: ChromaDB index built (%d terms)", len(terms)
            )
            return
        except Exception as exc:
            logger.warning(
                "HybridCascadeStrategy: ChromaDB unavailable (%s) — numpy fallback",
                type(exc).__name__,
            )

        # --- Numpy fallback ---
        try:
            import numpy as np
            raw_embeddings = self._embedder.encode(terms, show_progress_bar=False)
            norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)    # avoid divide-by-zero
            self._np_embeddings = (raw_embeddings / norms).astype("float32")
            self._np_terms = terms
            self.semantic_available = True
            logger.info(
                "HybridCascadeStrategy: numpy index built (%d terms)", len(terms)
            )
        except Exception as exc:
            self.semantic_available = False
            logger.warning(
                "HybridCascadeStrategy: numpy fallback also failed (%s) — "
                "semantic disabled", type(exc).__name__,
            )

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def search(self, query: str) -> list[SNOMEDMatch]:
        """
        Candidate extraction (documented per SNOMEDSearchStrategy contract):
          Stage 1+2: enumerate all 1..MAX_NGRAM_WINDOW token windows over query.
          Stage 3:   rapidfuzz token_sort_ratio on residual character spans.
          Stage 4:   sentence-transformer cosine similarity on still-residual spans.
        Returns ALL candidates from stages 1-4; does NOT pre-filter by MIN_CONFIDENCE.
        """
        hits: list[SNOMEDMatch] = []

        # Stage 1+2: exact + synonym via token n-gram windows
        exact_syn_hits = self._exact_synonym_pass(query)
        hits.extend(exact_syn_hits)

        # Stage 3: fuzzy on residual spans
        after_stage_2 = self._compute_residual_spans(query, hits)
        fuzzy_hits = self._fuzzy_pass(query, after_stage_2)
        hits.extend(fuzzy_hits)

        # Stage 4: semantic on still-residual spans
        after_stage_3 = self._compute_residual_spans(query, hits)
        semantic_hits = self._semantic_pass(query, after_stage_3)
        hits.extend(semantic_hits)

        return self._dedup_longest_match(hits)

    def health_check(self) -> dict:
        return {
            "ready": self._ready,
            "name": self.name,
            "dictionary_size": len(self._exact_index),
            "synonym_size": len(self._synonym_index),
            "semantic_available": self.semantic_available,
        }

    # -------------------------------------------------------------------------
    # Stage 1+2: exact + synonym via n-gram windows
    # -------------------------------------------------------------------------

    def _exact_synonym_pass(self, query: str) -> list[SNOMEDMatch]:
        """
        Tokenize query with character offsets.
        Enumerate all 1..MAX_NGRAM_WINDOW token windows.
        For each window: exact_index lookup, then synonym_index lookup (with alias resolution).
        Add match on first hit per window; do NOT add both exact and synonym for same window.
        """
        tokens = self._tokenize_with_offsets(query)
        hits: list[SNOMEDMatch] = []

        for n in range(MAX_NGRAM_WINDOW, 0, -1):
            # Enumerate windows of size n (larger windows first to bias longest-match)
            for i in range(len(tokens) - n + 1):
                window = tokens[i : i + n]
                window_text = " ".join(t[0] for t in window).lower()
                span_start = window[0][1]   # char start of first token
                span_end   = window[-1][2]  # char end of last token (exclusive)

                # --- Exact lookup ---
                if window_text in self._exact_index:
                    record = self._exact_index[window_text]
                    hits.append(SNOMEDMatch(
                        code=record["concept_id"],
                        display=record["preferred_term"],
                        match_type="exact",
                        confidence=0.99,
                        original_text=window_text,
                        span=(span_start, span_end),
                        negated=False,
                    ))
                    continue   # don't also check synonym for same window

                # --- Alias lookup → resolves to exact_index record ---
                alias_target = self._alias_dict.get(window_text)
                if alias_target:
                    record = self._exact_index[alias_target]
                    hits.append(SNOMEDMatch(
                        code=record["concept_id"],
                        display=record["preferred_term"],
                        match_type="synonym",
                        confidence=0.97,
                        original_text=window_text,
                        span=(span_start, span_end),
                        negated=False,
                    ))
                    continue

                # --- Raw synonym lookup ---
                if window_text in self._synonym_index:
                    record = self._synonym_index[window_text]
                    hits.append(SNOMEDMatch(
                        code=record["concept_id"],
                        display=record["preferred_term"],
                        match_type="synonym",
                        confidence=0.95,
                        original_text=window_text,
                        span=(span_start, span_end),
                        negated=False,
                    ))

        return hits

    def _tokenize_with_offsets(self, query: str) -> list[tuple[str, int, int]]:
        """
        Returns list of (token_text, char_start, char_end_exclusive).
        Splits on whitespace sequences. Preserves original character offsets in query.
        Does NOT lowercase tokens here; lowercasing happens at comparison site.

        M1 FIX: replaced query.split() + query.index(token, cursor) with
        re.finditer(r'\\S+', query). The old approach fails for tokens with attached
        punctuation or repeated substrings — index() may find the wrong occurrence.
        re.finditer yields (match.group(), match.start(), match.end()) directly, which
        is both correct and O(n) without substring scanning.
        """
        return [
            (m.group(), m.start(), m.end())
            for m in re.finditer(r'\S+', query)
        ]

    # -------------------------------------------------------------------------
    # Residual span computation
    # -------------------------------------------------------------------------

    def _compute_residual_spans(
        self,
        query: str,
        matched_so_far: list[SNOMEDMatch],
    ) -> list[tuple[int, int]]:
        """
        Returns character spans NOT covered by any match in matched_so_far.
        Algorithm (from architect spec §3.7.4):
          1. Sort matches by span[0] ascending.
          2. Walk matches; for each: residual = (cursor, match.span[0]) if non-empty.
          3. Advance cursor = max(cursor, match.span[1]).
          4. After loop: residual = (cursor, len(query)) if non-empty.
          5. Drop residuals with length < MIN_RESIDUAL_LEN (noise filter).
        """
        if not matched_so_far:
            return [(0, len(query))] if len(query) >= MIN_RESIDUAL_LEN else []

        sorted_matches = sorted(matched_so_far, key=lambda m: m.span[0])
        residuals: list[tuple[int, int]] = []
        cursor = 0

        for m in sorted_matches:
            if m.span[0] > cursor:
                gap_start = cursor
                gap_end   = m.span[0]
                if gap_end - gap_start >= MIN_RESIDUAL_LEN:
                    residuals.append((gap_start, gap_end))
            cursor = max(cursor, m.span[1])

        tail_start = cursor
        tail_end   = len(query)
        if tail_end - tail_start >= MIN_RESIDUAL_LEN:
            residuals.append((tail_start, tail_end))

        return residuals

    # -------------------------------------------------------------------------
    # Stage 3: fuzzy pass on residual spans
    # -------------------------------------------------------------------------

    def _fuzzy_pass(
        self,
        query: str,
        residual_spans: list[tuple[int, int]],
    ) -> list[SNOMEDMatch]:
        """
        For each residual span:
          1. Extract substring from query.
          2. Run rapidfuzz token_sort_ratio against ALL preferred_terms (exact_index keys).
          3. If best score >= self._fuzzy_cutoff (88): create SNOMEDMatch with
             confidence = score / 100.0, match_type = "fuzzy".
          4. span offset = (span_start, span_end) of the residual character span.

        NOT logged: extracted substring text.
        """
        all_preferred_terms = list(self._exact_index.keys())
        hits: list[SNOMEDMatch] = []

        for span_start, span_end in residual_spans:
            substring = query[span_start:span_end].strip()
            if not substring:
                continue

            result = rf_process.extractOne(
                substring.lower(),
                all_preferred_terms,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=self._fuzzy_cutoff,
            )
            if result is None:
                continue

            matched_term, score, _idx = result
            record = self._exact_index.get(matched_term)
            if record is None:
                continue   # defensive: matched_term must be in exact_index

            hits.append(SNOMEDMatch(
                code=record["concept_id"],
                display=record["preferred_term"],
                match_type="fuzzy",
                confidence=round(score / 100.0, 4),
                original_text=substring,
                span=(span_start, span_end),
                negated=False,
            ))

        return hits

    # -------------------------------------------------------------------------
    # Stage 4: semantic pass on residual spans
    # -------------------------------------------------------------------------

    def _semantic_pass(
        self,
        query: str,
        residual_spans: list[tuple[int, int]],
    ) -> list[SNOMEDMatch]:
        """
        For each residual span:
          1. Extract substring; encode with sentence-transformer.
          2. Query ChromaDB (if available) or compute numpy cosine similarity.
          3. If best similarity >= self._semantic_threshold (0.82): create SNOMEDMatch.
          4. span offset = residual span bounds.

        Returns [] immediately if self.semantic_available is False.
        All exceptions caught internally — failure returns [] without raising.
        NOT logged: substring text.
        """
        if not self.semantic_available or self._embedder is None:
            return []

        hits: list[SNOMEDMatch] = []

        for span_start, span_end in residual_spans:
            substring = query[span_start:span_end].strip()
            if not substring:
                continue

            match = self._semantic_match_single(substring, span_start, span_end)
            if match is not None:
                hits.append(match)

        return hits

    def _semantic_match_single(
        self,
        substring: str,
        span_start: int,
        span_end: int,
    ) -> Optional[SNOMEDMatch]:
        """Single-span semantic lookup. Returns None on any failure or below threshold."""
        try:
            # --- ChromaDB path ---
            if self._collection is not None:
                embedding = self._embedder.encode(
                    [substring], show_progress_bar=False
                ).tolist()
                results = self._collection.query(
                    query_embeddings=embedding,
                    n_results=1,
                    include=["documents", "distances"],
                )
                if not results["documents"] or not results["documents"][0]:
                    return None
                doc        = results["documents"][0][0]
                similarity = 1.0 - results["distances"][0][0]
                if similarity < self._semantic_threshold:
                    return None
                record = self._exact_index.get(doc)
                if record is None:
                    return None
                return SNOMEDMatch(
                    code=record["concept_id"],
                    display=record["preferred_term"],
                    match_type="semantic",
                    confidence=round(similarity, 4),
                    original_text=substring,
                    span=(span_start, span_end),
                    negated=False,
                )

            # --- Numpy fallback path ---
            if self._np_embeddings is not None and self._np_terms:
                import numpy as np
                query_vec = self._embedder.encode([substring], show_progress_bar=False)
                norm = float(np.linalg.norm(query_vec))
                if norm == 0.0:
                    return None
                query_norm   = (query_vec / norm).astype("float32")
                similarities = self._np_embeddings @ query_norm.T
                best_idx     = int(np.argmax(similarities))
                similarity   = float(similarities[best_idx])
                if similarity < self._semantic_threshold:
                    return None
                matched_term = self._np_terms[best_idx]
                record = self._exact_index.get(matched_term)
                if record is None:
                    return None
                return SNOMEDMatch(
                    code=record["concept_id"],
                    display=record["preferred_term"],
                    match_type="semantic",
                    confidence=round(similarity, 4),
                    original_text=substring,
                    span=(span_start, span_end),
                    negated=False,
                )

        except Exception as exc:
            logger.warning(
                "_semantic_match_single: exception (%s) — skipping span",
                type(exc).__name__,
                # NOT logged: substring, span offsets
            )
        return None

    # -------------------------------------------------------------------------
    # Deduplication: longest-match-wins
    # -------------------------------------------------------------------------

    def _dedup_longest_match(self, hits: list[SNOMEDMatch]) -> list[SNOMEDMatch]:
        """
        Given all candidates from stages 1-4, return deduplicated list where:
          - For any two hits where hit_A.span fully contains hit_B.span (or equals it),
            keep hit_A (the longer / outer span) and discard hit_B.
          - Tie-break by confidence DESC, then span_start ASC (earlier span preferred).
          - Two hits with overlapping but non-containing spans: both kept (let assembler
            decide via confidence-based dedup by SNOMED code).

        Algorithm:
          1. Sort: primary=span_length DESC, secondary=confidence DESC, tertiary=span[0] ASC.
          2. Walk sorted list. Track kept_spans as list of (start, end).
          3. For each candidate: if its span [s, e) is fully contained within any kept span,
             discard it. Otherwise keep it and add (s, e) to kept_spans.
        """
        if not hits:
            return []

        sorted_hits = sorted(
            hits,
            key=lambda m: (-(m.span[1] - m.span[0]), -m.confidence, m.span[0]),
        )

        kept: list[SNOMEDMatch] = []
        kept_spans: list[tuple[int, int]] = []

        for candidate in sorted_hits:
            c_start, c_end = candidate.span
            contained = False
            for k_start, k_end in kept_spans:
                if k_start <= c_start and c_end <= k_end:
                    contained = True
                    break
            if not contained:
                kept.append(candidate)
                kept_spans.append((c_start, c_end))

        return kept
```

---

## 4. `NegationAnnotator` — `src/snomed_search/negation.py`

**File:** `src/snomed_search/negation.py`

**Imports:**
```python
import logging
import re
from dataclasses import replace
from typing import Optional

from src.snomed_search.base import SNOMEDMatch
```

**Constants:**
```python
NEGATION_CUES_PRE: list[str] = [
    "no", "not", "without", "denies", "denied", "rules out", "ruled out",
    "history of", "h/o", "free of", "absence of", "absent", "neither",
    "negative for", "no evidence of", "no signs of",
]
NEGATION_CUES_POST: list[str] = [
    "unlikely", "ruled out", "negative", "denied",
]
PSEUDO_NEGATION_PHRASES: list[str] = [
    # Phrases that contain negation-cue tokens but are NOT negating the clinical term.
    # Order matters: longer phrases must be checked before shorter sub-phrases.
    "no contraindication for",
    "no change in",
    "no further",
    "not only",
]
SCAN_STOP_PUNCT: frozenset[str] = frozenset(".;!?\n")
WINDOW_SIZE: int = 5    # tokens on each side of match
```

**`NegationAnnotator` class:**
```python
class NegationAnnotator:
    """
    NegEx-style rule-based negation detector.
    Operates on character-offset SNOMEDMatch.span values.
    No LLM. Deterministic. Stateless — safe for concurrent calls.

    Comma is intentionally NOT a scan-stop character (see CONTEXT.md rationale and
    proposal §3.7.5). Sentence-boundary stops: period, semicolon, !, ?, newline.
    """

    def annotate(
        self,
        query: str,
        matches: list[SNOMEDMatch],
    ) -> list[SNOMEDMatch]:
        """
        Returns a new list where each SNOMEDMatch.negated is set per NegEx rules.
        Input SNOMEDMatch objects are frozen dataclasses; new objects created via replace().
        """
        tokens: list[tuple[str, int, int]] = self._tokenize_with_offsets(query)
        annotated: list[SNOMEDMatch] = []
        for m in matches:
            is_neg = self._is_negated(query, tokens, m)
            annotated.append(replace(m, negated=is_neg))
        return annotated

    def _is_negated(
        self,
        query: str,
        tokens: list[tuple[str, int, int]],
        match: SNOMEDMatch,
    ) -> bool:
        """
        Core NegEx logic for a single SNOMEDMatch.
        Returns True if the match should be classified as negated.

        Steps:
          1. Find match start token index via span[0].
          2. Check pseudo-negation overlap FIRST — if any pseudo-negation phrase
             overlaps or immediately precedes the match, return False (not negated).
          3. Pre-window scan: check tokens [max(0, anchor-WINDOW_SIZE)..anchor) for cues,
             stopping at SCAN_STOP_PUNCT boundaries before the anchor.
          4. Post-window scan: check tokens (match_last_token+1..match_last_token+WINDOW_SIZE]
             for cues, stopping at SCAN_STOP_PUNCT boundaries after the match.

        Token index None → span not aligned to a token boundary → return False.
        """
        match_start_token_idx: Optional[int] = self._find_token_index(tokens, match.span[0])
        if match_start_token_idx is None:
            return False

        # Step 2: pseudo-negation check
        if self._has_pseudo_negation_near(query, match.span):
            return False

        # Step 3: pre-window scan
        if self._scan_window_for_cue(
            tokens=tokens,
            anchor_idx=match_start_token_idx,
            cues=NEGATION_CUES_PRE,
            direction="pre",
            window=WINDOW_SIZE,
        ):
            return True

        # Step 4: post-window scan
        match_end_token_idx = self._find_token_index(tokens, match.span[1] - 1)
        if match_end_token_idx is None:
            match_end_token_idx = match_start_token_idx   # fallback: single-token match

        if self._scan_window_for_cue(
            tokens=tokens,
            anchor_idx=match_end_token_idx,
            cues=NEGATION_CUES_POST,
            direction="post",
            window=WINDOW_SIZE,
        ):
            return True

        return False

    def _tokenize_with_offsets(self, text: str) -> list[tuple[str, int, int]]:
        """
        Returns list of (token_text, char_start, char_end_exclusive).
        Splits on whitespace only. Preserves original casing for stop-punct check.
        Does NOT split on punctuation — "ruled." is one token; stop-punct detection
        is done by inspecting whether the last char of a token is in SCAN_STOP_PUNCT.

        M1 FIX (same pattern as HybridCascadeStrategy._tokenize_with_offsets):
        replaced text.split() + text.index(token, cursor) with re.finditer(r'\\S+', text).
        See that method's docstring for rationale. Both implementations share the fix;
        a shared module-level helper is left for the developer to extract if desired.
        """
        return [
            (m.group(), m.start(), m.end())
            for m in re.finditer(r'\S+', text)
        ]

    def _find_token_index(
        self,
        tokens: list[tuple[str, int, int]],
        char_offset: int,
    ) -> Optional[int]:
        """
        Returns the index in `tokens` whose span contains `char_offset`.
        A token (text, start, end) contains char_offset iff start <= char_offset < end.
        Returns None if no token contains the offset.

        Linear scan — acceptable for WINDOW_SIZE=5 and typical query lengths (<100 tokens).
        """
        for idx, (text, start, end) in enumerate(tokens):
            if start <= char_offset < end:
                return idx
        return None

    def _has_pseudo_negation_near(
        self,
        query: str,
        match_span: tuple[int, int],
    ) -> bool:
        """
        Returns True if any PSEUDO_NEGATION_PHRASES appears in the query region
        spanning [max(0, match_span[0] - 60) .. match_span[1]].

        The 60-char lookback window is intentionally generous to catch multi-word
        pseudo-negation phrases that may appear several tokens before the match.
        Character-based (not token-based) for simplicity and speed.

        HIPAA: query text NOT logged.
        """
        lookback_start = max(0, match_span[0] - 60)
        context_region = query[lookback_start : match_span[1]].lower()
        for phrase in PSEUDO_NEGATION_PHRASES:
            if phrase in context_region:
                return True
        return False

    def _scan_window_for_cue(
        self,
        tokens: list[tuple[str, int, int]],
        anchor_idx: int,
        cues: list[str],
        direction: str,     # "pre" or "post"
        window: int,
    ) -> bool:
        """
        Scans up to `window` tokens in `direction` from `anchor_idx`.
        Stops before scanning a token whose text ends with a SCAN_STOP_PUNCT character
        (period, semicolon, !, ?, newline).

        For "pre" direction: scans tokens [anchor_idx - window .. anchor_idx).
          Scan order: from anchor inward (anchor_idx-1, anchor_idx-2, ...).
          Stop as soon as a stop-punct token is encountered — tokens further from
          anchor are beyond the sentence boundary.

        For "post" direction: scans tokens (anchor_idx .. anchor_idx + window].
          Scan order: from anchor outward (anchor_idx+1, anchor_idx+2, ...).
          Stop as soon as a stop-punct token is encountered.

        Multi-word cues (e.g. "no history of", "no evidence of"):
          Detected by checking if the cue phrase (lowercased) is a contiguous
          substring of the lowercased window text (joined with spaces).

        Window truncation: when anchor is near start/end, window naturally truncates
        to available tokens. No synthetic padding tokens are added.

        Returns True if any cue is found within the (truncated) window before a stop.
        """
        if direction == "pre":
            # B1 FIX: build the candidate index list as a plain Python list rather than
            # a range, so anchor_idx=0 produces [] (correct: no tokens to the left)
            # instead of the old range(-1, -6, -1) which also happened to be empty
            # but for the wrong reasons and was fragile at other boundary values.
            #
            # Scan order for "pre": closest to anchor first (anchor_idx-1, -2, ...) so
            # that stop-punct early-break correctly halts at the nearest sentence boundary.
            # The collected texts are then reversed before phrase matching to restore
            # natural left-to-right order.
            window_token_indices = list(range(max(0, anchor_idx - window), anchor_idx))[::-1]
        else:  # "post"
            window_token_indices = range(anchor_idx + 1, min(len(tokens), anchor_idx + window + 1))

        window_token_texts: list[str] = []
        for idx in window_token_indices:
            token_text, _start, _end = tokens[idx]
            # Stop-punct check: if the token ends with a stop character,
            # include it in stop check (e.g. "cancer." stops scan here)
            # but do NOT include the stop-punct token itself in the cue window.
            if token_text[-1] in SCAN_STOP_PUNCT:
                break   # sentence boundary — do not scan further
            window_token_texts.append(token_text.lower())

        if not window_token_texts:
            return False

        # Reconstruct window text for multi-word cue matching.
        # For "pre" direction, window_token_texts are in closest-first order;
        # reverse to get natural left-to-right order for phrase matching.
        if direction == "pre":
            window_token_texts = list(reversed(window_token_texts))

        window_text = " ".join(window_token_texts)

        for cue in cues:
            # Single-word cues: whole-word match
            # Multi-word cues: substring match on the joined window text
            if " " in cue:
                if cue in window_text:
                    return True
            else:
                if re.search(rf"\b{re.escape(cue)}\b", window_text):
                    return True

        return False
```

---

## 5. `NLPPipeline.run_with_session()` — `src/pipeline.py`

**File:** `src/pipeline.py`

**Imports:**
```python
import datetime
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, wait as futures_wait, ALL_COMPLETED
from pathlib import Path
from typing import Optional, Union

from src.assembler import ResponseAssembler, ClarificationOutput
from src.assembler import NLPOutput                   # discriminated union return type
from src.conversation import ConversationSession, Turn
from src.exceptions import PipelineError, LLMProviderError
from src.filter_extractor import FilterExtractor, ExtractedFilters
from src.llm_provider.registry import get_provider
from src.normalizers.geo import GeoNormalizer
from src.preprocessor import Preprocessor, PreprocessorError
from src.snomed_search.base import SNOMEDMatch
from src.snomed_search.negation import NegationAnnotator
from src.snomed_search.registry import get_strategy
from src.sufficiency_gate import SufficiencyGate, SufficiencyDecision, _count_set_filters
from src.ambiguous_registry import AmbiguousTermsRegistry
from src.llm_provider.base import LLMProvider
from src.snomed_search.base import SNOMEDSearchStrategy
from src.filter_extractor import StateFilter, FilterField
```

**Constants:**
```python
PROJECT_ROOT       = Path(__file__).parent.parent
DEFAULT_SNOMED_CSV = str(PROJECT_ROOT / "data" / "snomed_clinical_trials.csv")
DEFAULT_GEO_PATH   = str(PROJECT_ROOT / "data" / "geo_canonical.json")
DEFAULT_AMBIG_PATH = str(PROJECT_ROOT / "data" / "ambiguous_terms.json")

PARALLEL_TIMEOUT_SECONDS = 15.0    # shared budget for BOTH futures combined (B2 fix)
THREAD_POOL_MAX_WORKERS  = 2
THREAD_POOL_NAME_PREFIX  = "nlp-"

# Per-turn log path enum — canonical string identifiers
LOG_PATH_SUFFICIENCY_CLARIFICATION = "sufficiency_clarification"
LOG_PATH_MAX_TURNS                 = "max_turns"
LOG_PATH_CLINICAL_INTENT           = "clinical_intent"
LOG_PATH_POST_EXTRACTION_SAFETY    = "post_extraction_safety"
LOG_PATH_SEARCH                    = "search"
```

**`NLPPipeline.__init__`:**
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
        """
        Init order is critical:
          1. SNOMED strategy first — AmbiguousTermsRegistry needs it for option validation.
          2. LLM provider second — independent.
          3. Registry third — depends on strategy.
          4. Gate, extractor, geo, negation, preprocessor, assembler — order flexible.
          5. ThreadPoolExecutor last — after all components ready.
        """
        _snomed_csv = snomed_csv_path or DEFAULT_SNOMED_CSV

        # Step 1: SNOMED strategy
        self._snomed: SNOMEDSearchStrategy = (
            snomed_strategy or get_strategy(dictionary_path=_snomed_csv)
        )

        # Step 2: LLM provider
        self._llm: LLMProvider = llm_provider or get_provider()

        # Step 3: Ambiguous terms registry
        _strict = (
            strict_validation
            if strict_validation is not None
            else os.environ.get("AMBIG_STRICT_VALIDATION", "true").lower() == "true"
        )
        self._registry = AmbiguousTermsRegistry(
            path=ambiguous_terms_path or DEFAULT_AMBIG_PATH,
            snomed_strategy=self._snomed,
            snomed_csv_path=_snomed_csv,
            strict_validation=_strict,
        )

        # Step 4: remaining components
        # M8: pass snomed_csv_path so SufficiencyGate can build DEFAULT_CONDITION_PROMPT lazily
        self._gate        = SufficiencyGate(self._registry, snomed_csv_path=_snomed_csv)
        self._extractor   = FilterExtractor(self._llm)
        self._geo         = GeoNormalizer(geo_json_path or DEFAULT_GEO_PATH)
        self._negation    = NegationAnnotator()
        self._preprocessor = Preprocessor()
        self._assembler   = ResponseAssembler()

        # Step 5: thread pool
        self._executor = ThreadPoolExecutor(
            max_workers=THREAD_POOL_MAX_WORKERS,
            thread_name_prefix=THREAD_POOL_NAME_PREFIX,
        )

        logger.info(
            "NLPPipeline initialized: strategy=%s provider=%s strict_validation=%s",
            self._snomed.name, self._llm.name, _strict,
            # NOT logged: api keys, paths, any content
        )
```

**`NLPPipeline.run_with_session` — 10-step orchestration:**
```python
    def run_with_session(
        self,
        raw_query: str,
        session: ConversationSession,
    ) -> Union[NLPOutput, ClarificationOutput]:
        """
        10-step orchestration. Full error paths documented inline.

        Session is mutated (turn appended) ONLY after successful pipeline completion
        or after a clarification is successfully built. This ensures session state
        is never poisoned by partial pipeline failures.

        HIPAA: raw_query and canonical are NEVER logged. Only counts, confidence
        scores, decision enums, and session metadata are logged.
        """
        start: float = time.perf_counter()
        log_path: str = "unknown"    # updated at each path fork

        # ─── Step 1: Preprocess raw user input ──────────────────────────────
        # Security gate on every turn — even clarification follow-ups.
        # PreprocessorError propagates to caller (app.py renders as st.warning).
        # Session state NOT mutated yet — blocked input never contaminates session.
        try:
            preprocessed = self._preprocessor.process(raw_query)
        except PreprocessorError:
            # Re-raise without wrapping — caller distinguishes PreprocessorError
            # from LLMProviderError / PipelineError.
            raise
        # Logged: char count only
        logger.info(
            "run_with_session step=1 preprocessed char_count=%d session_id=%s",
            preprocessed.char_count, session.session_id,
            # NOT logged: raw_query, preprocessed.text
        )

        # ─── Step 2: Compute canonical query (pure; does NOT mutate session) ──
        # M2 FIX: session.compute_canonical_query() is a PURE computation — it returns
        # the new canonical string without mutating session.canonical_query. Mutation is
        # deferred until AFTER assert_safe passes (Step 2b), preserving the invariant
        # that session state is only updated on fully validated input.
        #
        # conversation.py spec must expose:
        #   compute_canonical_query(user_input: str) -> str  (pure, no side effects)
        #   set_canonical_query(query: str) -> None           (mutator, called below)
        #
        # Chosen over the "revert_canonical_query()" alternative because it keeps
        # the happy-path mutation in one place and never requires rollback logic.
        canonical: str = session.compute_canonical_query(preprocessed.text)
        # No log here — canonical contains user content.

        # ─── Step 2b: Defense-in-depth injection check on merged canonical ──
        # Catches pathological substitution edge cases (e.g., multi-turn injection
        # where each individual turn passed but combination triggers a pattern).
        # assert_safe() is a read-only method on Preprocessor — no new PreprocessedInput.
        self._preprocessor.assert_safe(canonical)
        # Raises PreprocessorError if any injection pattern matches canonical.
        # Session.canonical_query has NOT yet been updated — if this raises, session
        # is in a clean pre-update state. The partial-state risk noted in proposal §13
        # is eliminated by this ordering.

        # Mutation only after assert_safe passes:
        session.set_canonical_query(canonical)

        # ─── Step 3: Pre-extraction sufficiency gate ─────────────────────────
        decision: SufficiencyDecision = self._gate.evaluate(canonical, session)

        if not decision.sufficient:
            # Clarification path — either "ambiguous_trigger" or "max_turns_reached"
            # (max_turns_reached always returns sufficient=True per spec §3.2 Rule 1,
            # so this branch is only reached for ambiguous_trigger).
            log_path = (
                LOG_PATH_SUFFICIENCY_CLARIFICATION
                if decision.reason == "ambiguous_trigger"
                else LOG_PATH_MAX_TURNS
            )
            clarification = self._assembler.build_clarification(decision, session, start)
            # Append turn ONLY after clarification successfully built (no exception from assembler).
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
            self._log_turn(session, decision, log_path, start, snomed_count=0, filter_count=0)
            return clarification

        # ─── Step 4: Parallel paths — filter LLM + algorithmic SNOMED ───────
        # Both futures are submitted before either result is awaited.
        # ThreadPoolExecutor with max_workers=2 means both run concurrently.
        fut_filters = self._executor.submit(self._extractor.extract, canonical)
        fut_snomed  = self._executor.submit(self._snomed.search, canonical)

        filters: ExtractedFilters
        snomed_raw: list[SNOMEDMatch]

        # B2 FIX: use futures_wait() with a SINGLE shared timeout so that both futures
        # together get exactly PARALLEL_TIMEOUT_SECONDS — not PARALLEL_TIMEOUT_SECONDS
        # each (which would allow up to 2× the budget if the first future fully consumes it).
        try:
            done, not_done = futures_wait(
                [fut_filters, fut_snomed],
                timeout=PARALLEL_TIMEOUT_SECONDS,
                return_when=ALL_COMPLETED,
            )
            if not_done:
                # At least one future did not complete within the shared budget.
                # cancel() on in-flight threads is a no-op in Python's ThreadPoolExecutor
                # — threads keep running but results are discarded. Known POC limitation
                # (proposal §5 open risks: thread-pool exhaustion under sustained DoS).
                fut_filters.cancel()
                fut_snomed.cancel()
                # Session NOT mutated — failed step does not poison session.
                logger.error(
                    "run_with_session step=4 TIMEOUT session_id=%s elapsed_ms=%.0f",
                    session.session_id, (time.perf_counter() - start) * 1000,
                )
                raise PipelineError("extraction_timeout")

            # Both futures are done — extract results (may raise if future raised)
            filters    = fut_filters.result()
            snomed_raw = fut_snomed.result()

        except PipelineError:
            raise   # re-raise timeout without wrapping
        except LLMProviderError as exc:
            # Provider-level permanent failure (auth error, invalid API key, etc.)
            # B5/M5 FIX: cancel BOTH futures, not just fut_snomed.
            # The filter future's thread may still be running (or may have raised);
            # cancel() ensures its result slot is released regardless.
            fut_filters.cancel()
            fut_snomed.cancel()
            logger.error(
                "run_with_session step=4 LLMProviderError provider=%s session_id=%s",
                exc.provider_name, session.session_id,
                # NOT logged: exc.original_error.message, any response content
            )
            raise   # propagate; app.py renders as generic st.error
        except Exception as exc:
            fut_filters.cancel()
            fut_snomed.cancel()
            logger.error(
                "run_with_session step=4 unexpected error type=%s session_id=%s",
                type(exc).__name__, session.session_id,
            )
            raise PipelineError("strategy_unavailable") from exc

        # ─── Step 5: Negation annotation ─────────────────────────────────────
        # Deterministic NegEx-style annotation. Mutates negated flag on each match
        # (via dataclasses.replace — originals not mutated; new list returned).
        snomed_matches: list[SNOMEDMatch] = self._negation.annotate(canonical, snomed_raw)

        logger.info(
            "run_with_session step=5 snomed_candidates=%d negated=%d session_id=%s",
            len(snomed_matches),
            sum(1 for m in snomed_matches if m.negated),
            session.session_id,
            # NOT logged: snomed display strings, canonical query
        )

        # ─── Step 6: Clinical-intent gate (existing security layer) ──────────
        # Rejects queries that yield zero SNOMED matches AND zero structured filters
        # after LLM extraction. Primary defense against non-clinical / data-dump
        # queries that slip through the preprocessor.
        #
        # M4 FIX: apply the same MIN_CONFIDENCE=0.60 + not-negated filter used by the
        # assembler BEFORE the intent-gate check. Previously, unfiltered snomed_matches
        # (including 0.62-confidence semantic hits that the assembler would later drop)
        # could pass the gate, leading to output with no SNOMED and no condition info.
        # Using qualifying_matches here ensures the intent gate sees only matches that
        # will actually appear in the final output.
        qualifying_matches = [
            m for m in snomed_matches
            if m.confidence >= 0.60 and not m.negated
        ]
        filter_set_count = _count_set_filters(filters)

        if not qualifying_matches and filter_set_count == 0:
            log_path = LOG_PATH_CLINICAL_INTENT
            # Do NOT append turn — failed at security gate; equivalent to preprocessor failure.
            logger.info(
                "run_with_session step=6 clinical_intent_rejected session_id=%s",
                session.session_id,
            )
            raise PreprocessorError(
                "No clinical content found. Please enter a query about a medical "
                "condition, investigator, research site, location, or study phase."
            )

        logger.info(
            "run_with_session step=6 ok snomed_qualifying=%d filter_count=%d session_id=%s",
            len(qualifying_matches), filter_set_count, session.session_id,
        )

        # ─── Step 7: Post-extraction safety check ────────────────────────────
        # Fires when SNOMED found zero high-confidence matches but filters are present.
        # e.g., "What's at Mayo?" → city/site extracted but no condition → ask condition.
        post_decision: SufficiencyDecision = self._gate.post_extraction_check(
            snomed_matches, filters
        )
        if not post_decision.sufficient:
            log_path = LOG_PATH_POST_EXTRACTION_SAFETY
            clarification = self._assembler.build_clarification(post_decision, session, start)
            # Include filters and snomed_matches in turn for context on next turn.
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
            self._log_turn(
                session, post_decision, log_path, start,
                snomed_count=len(snomed_matches), filter_count=filter_set_count,
            )
            return clarification

        # ─── Step 8: Geo normalization ────────────────────────────────────────
        geo = self._geo.normalize(
            city=filters.city.value,
            state=_state_value_for_geo(filters.state),
        )
        logger.info(
            "run_with_session step=8 geo_confidence=%.3f is_region=%s session_id=%s",
            geo.confidence, geo.is_region, session.session_id,
            # NOT logged: geo.city, geo.states, filter values
        )

        # ─── Step 9: Defense-in-depth: assert_safe on canonical ──────────────
        # Second call (first was step 2b). After parallel paths complete, canonical
        # has not changed, but this call is intentionally kept as belt-and-suspenders
        # confirmation before assembling the final output.
        self._preprocessor.assert_safe(canonical)
        # Raises PreprocessorError — should be unreachable if step 2b passed, but
        # provides defense against any future code path that might mutate canonical.

        # ─── Step 10: Assemble output; append turn AFTER success ─────────────
        # Assembler raises on Pydantic validation failure — should not occur with
        # well-formed inputs, but defensive catch is in the caller (app.py).
        output: NLPOutput = self._assembler.assemble(
            filters=filters,
            snomed_matches=snomed_matches,
            geo=geo,
            start_time=start,
        )

        # Append turn ONLY after successful assembly — no partial state.
        log_path = LOG_PATH_SEARCH
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
        self._log_turn(
            session, decision, log_path, start,
            snomed_count=output.metadata.total_snomed_matches,
            filter_count=filter_set_count,
        )
        return output

    # ─── Log helper ───────────────────────────────────────────────────────────

    def _log_turn(
        self,
        session: ConversationSession,
        decision: SufficiencyDecision,
        log_path: str,
        start: float,
        snomed_count: int = 0,
        filter_count: int = 0,
    ) -> None:
        """
        Emit single structured JSON log line per turn.
        Fields logged: ts, session_id, turn_index, clarification_count, path,
                       decision_reason, snomed_match_count, snomed_strategy,
                       filter_count, llm_provider, processing_time_ms.
        NEVER logged: raw_query, canonical_query, user_input, filter values,
                      SNOMED display strings, LLM response, triggered_by value,
                      option text.
        """
        # M6: datetime imported at module top (not inline here)
        # M7: `decision` is the PRE-extraction SufficiencyDecision for the success path
        # (step 10). On that path decision.reason is always "ok_no_trigger" — this is
        # intentional: the pre-extraction gate passed and no clarification was needed.
        # On the post-extraction-safety path (step 7), _log_turn is called with
        # `post_decision` (whose reason is "filters_without_condition"), not `decision`.
        # The parameter name `decision` here is generic; callers pass the appropriate
        # decision object for each path.
        log_record = {
            "ts": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "session_id": session.session_id,
            "turn_index": len(session.turns) - 1,    # after append
            "clarification_count": session.clarification_turn_count(),
            "path": log_path,
            "decision_reason": decision.reason,
            "snomed_match_count": snomed_count,
            "snomed_strategy": self._snomed.name,
            "filter_count": filter_count,
            "llm_provider": self._llm.name,
            "processing_time_ms": int((time.perf_counter() - start) * 1000),
        }
        logger.info("turn_log %s", json.dumps(log_record))

    # ─── Backwards-compat shim ────────────────────────────────────────────────

    # OBSOLETE-AT-SCALE: single-turn entry, kept for tests/run_tests.py
    def run(self, raw_query: str) -> NLPOutput:
        """
        Wraps run_with_session() with a fresh single-turn session.
        If the pipeline returns a ClarificationOutput (ambiguous query),
        assembles a best-effort NLPOutput from session state.
        This allows legacy tests to pass without multi-turn awareness.
        """
        session = ConversationSession.new()
        result = self.run_with_session(raw_query, session)
        if isinstance(result, ClarificationOutput):
            return self._assemble_best_effort_from_session(session)
        return result

    def _assemble_best_effort_from_session(self, session: ConversationSession) -> NLPOutput:
        """
        Called when run() receives a ClarificationOutput.
        Returns an NLPOutput with empty snomed_terms and whatever filters the session
        has collected (may be none for the first clarification turn).
        Used only by the legacy run() shim — not by multi-turn paths.
        """
        # If last turn has filters, use them; otherwise empty ExtractedFilters.
        last_turn = session.turns[-1] if session.turns else None
        if last_turn and last_turn.filters:
            filters = last_turn.filters
            snomed  = last_turn.snomed_matches or []
            geo     = last_turn.geo or self._geo.normalize(None, None)
        else:
            # Fully empty result — preprocessor passed but no data yet
            filters = _empty_filters()
            snomed  = []
            geo     = self._geo.normalize(None, None)
        return self._assembler.assemble(
            filters=filters,
            snomed_matches=snomed,
            geo=geo,
            start_time=time.perf_counter(),
        )
```

**Helper functions (module-level in pipeline.py):**
```python
def _state_value_for_geo(state_filter: StateFilter) -> Optional[str]:
    """
    Convert StateFilter to a single state string for the legacy GeoNormalizer.
    Multi-state regions or empty → None (geo normalizer handles region expansion
    from city input alone).
    """
    if state_filter.is_region:
        return None
    if len(state_filter.values) == 1:
        return state_filter.values[0]
    return None

# B4 FIX: _count_set_filters is the authoritative definition in sufficiency_gate.py
# (alongside _any_filter_set, per proposal §14 erratum). It is imported above.
# The duplicate that previously appeared here is removed to avoid NameError and drift.

def _empty_filters() -> ExtractedFilters:
    """Return a fully-empty ExtractedFilters for the best-effort shim path.
    M6: FilterField and StateFilter imported at module top; no inline import needed."""
    return ExtractedFilters(
        investigator_name=FilterField(value=None, confidence=0.0),
        site_name=FilterField(value=None, confidence=0.0),
        city=FilterField(value=None, confidence=0.0),
        state=StateFilter(values=[], confidence=0.0, is_region=False),
        phase=FilterField(value=None, confidence=0.0),
        raw_response_length=0,
    )
```

---

## Discrepancies with architect spec

1. **`assert_safe()` method on `Preprocessor`** — The proposal (§3.3 and §3.8) calls
   `self._preprocessor.assert_safe(canonical)`, but the existing `Preprocessor` class
   has no `assert_safe()` method (only `process()` and `_check_injection()`).
   The spec above assumes `assert_safe(text: str) -> None` is a new public method on
   `Preprocessor` that calls `_check_injection()` and raises `PreprocessorError` if True.
   The architect's spec implies this method but never declares it.
   **Action needed:** architect must add `assert_safe()` signature to the `Preprocessor`
   spec (or developer adds it as a trivial one-liner; no ambiguity in behavior).

2. **`_count_set_filters` / `_any_filter_set` placement (B4 resolved)** — Per proposal §14
   erratum, both `_any_filter_set` and `_count_set_filters` are the authoritative definitions
   in `sufficiency_gate.py`. `pipeline.py` imports `_count_set_filters` from there and no
   longer maintains a local duplicate. `_any_filter_set` remains private to `sufficiency_gate.py`
   (used only within that module by `post_extraction_check`). Architect should confirm whether
   `_any_filter_set` should be exported as a public name; the current spec keeps it private.

3. **`DEFAULT_CONDITION_PROMPT` ownership** — The proposal lists `DEFAULT_CONDITION_PROMPT`
   as a constant in `src/sufficiency_gate.py` (§3.2), but `SufficiencyGate` and
   `AmbiguousTermsRegistry` are both specified as being in `src/sufficiency_gate.py`,
   while §3.1 implies `AmbiguousEntry` and the registry belong there too. This spec places
   everything in `src/sufficiency_gate.py` per the architect's §3.2 header. If `AmbiguousEntry`
   and `AmbiguousTermsRegistry` are moved to a separate `src/ambiguous_registry.py`, the
   import structure in this spec will need adjustment (circular import risk: registry imports
   strategy; pipeline imports both).

4. **`snomed_resolver.ALIAS_DICTIONARY` reuse** — This spec imports `ALIAS_DICTIONARY` from
   `src/snomed_resolver.py` into `hybrid_cascade.py` to avoid duplication. The architect's
   spec says "port from existing snomed_resolver.py" but doesn't specify whether the alias
   dict is shared or copied. If `snomed_resolver.py` is later removed (post-OBSOLETE-AT-SCALE
   cleanup), this import breaks. Architect should decide: move `ALIAS_DICTIONARY` to
   `src/snomed_search/hybrid_cascade.py` now, or keep the shared import.

5. **`DEFAULT_CONDITION_PROMPT` lazy build (M8 resolved)** — Per architect's erratum
   (rev 2 §14 point 5), `DEFAULT_CONDITION_PROMPT` is now built lazily on first
   `SufficiencyGate.__init__()` call, with class-level memoization. The CSV path is
   injected via `SufficiencyGate(registry, snomed_csv_path=...)`. The old module-level
   eager build is removed. The `_SNOMED_CSV_DEFAULT` constant remains as a fallback
   when `snomed_csv_path` is not provided to the gate constructor.

6. **`ExtractedFilters.state` type inconsistency in pipeline step 6** — The proposal (§3.8
   step 6) uses `any_filter_set(filters)` which checks `filters.state.values` (a `list[str]`),
   but the original `pipeline.py` clinical-intent gate checks `extraction.state.value`
   (a single `Optional[str]`). This spec uses `.values` (the new schema). Any test that
   checks clinical-intent rejection behavior must account for this schema change.

---

## Likely QA / Security follow-up objections

1. **`assert_safe()` on merged canonical (steps 2b + 9):** QA will ask whether calling it
   twice is redundant (yes, step 9 is belt-and-suspenders), and Security will ask whether
   the injection check on a concatenated string is sufficient — specifically whether
   `"cancer" + " " + "ignore previous instructions"` would be caught. It will, because
   `_check_injection` scans the full string, but QA will want a specific test case for this.

2. **`ThreadPoolExecutor.cancel()` semantics:** Security will note that `cancel()` on
   in-flight threads is a no-op in Python (documented above). Under a sustained timeout
   DoS, the pool fills with stalled LLM threads. The proposal acknowledges this in §5 open
   risks but QA will ask for a concrete mitigation plan (e.g., process-level timeout via
   `signal.alarm` on Linux, or a separate per-provider timeout enforced inside
   `GroqProvider.complete()` via `httpx` connection timeout).

3. **`_derive_overrides` self-defeat fix:** QA will write a test where the SNOMED CSV
   contains `preferred_term="cancer"` (which it does — see `snomed_resolver.py` ALIAS targets).
   This test case is already in `tests/test_sufficiency_gate.py` (architect §3.12), but
   QA will verify the discard happens before the CSV-scan union step (it does in this spec:
   `discard(trigger.lower())` runs after the full CSV scan, covering both the option-seed
   and the CSV-scan additions).

4. **`NegationAnnotator` pseudo-negation window (60-char hardcoded):** Security will ask
   why 60 chars specifically. This spec should document that 60 chars is approximately
   10-12 average English words — large enough to cover all PSEUDO_NEGATION_PHRASES listed
   but small enough to avoid false suppressions from phrases earlier in the sentence.
   QA will want a unit test with a query where a pseudo-negation phrase is exactly 61 chars
   before the match to confirm it does NOT suppress negation.

5. **Session mutation ordering (M2 resolved):** The original step 2 called
   `session.update_canonical_query()` (mutating) before `assert_safe()`. If step 2b raised,
   session held an updated canonical but no appended Turn — partial state. This spec now uses
   `compute_canonical_query()` (pure) → `assert_safe()` → `set_canonical_query()` (mutate only
   after validation). `conversation.py` must expose both methods. No rollback path is needed.
   The partial-state risk from proposal §3.3 is eliminated by this ordering.
