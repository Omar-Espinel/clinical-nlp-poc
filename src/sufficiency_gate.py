"""
src/sufficiency_gate.py — Deterministic pre-extraction + post-extraction sufficiency gate.

Architect decision D3: AmbiguousEntry, AmbiguousTermsRegistry, SufficiencyGate,
DEFAULT_CONDITION_PROMPT, _any_filter_set, _count_set_filters all live here.

Architect decision D5 / M8: DEFAULT_CONDITION_PROMPT is built LAZILY on first
SufficiencyGate.__init__ call, with class-level memoization. NOT at module import time.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator

from src.snomed_search.base import SNOMEDMatch, SNOMEDSearchStrategy

# Circular-import guard: ConversationSession imports SufficiencyDecision from here.
# Only import for type-checking; never at runtime to avoid cycles.
if TYPE_CHECKING:
    from src.conversation import ConversationSession

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AMBIG_JSON_MIN_OPTIONS = 3
AMBIG_JSON_MAX_OPTIONS = 5
OPTION_MIN_CONFIDENCE = 0.85  # every option in ambiguous_terms.json must resolve here

# Layer 1 — auto-derived trigger constants
MIN_DERIVED_TOKEN_LEN = 4
MIN_DERIVED_TERM_FREQUENCY = 2  # token must appear in >= 2 distinct preferred_terms

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
    "human",  # "human" in "human immunodeficiency virus infection"
})

# Layer 2 — embedding ambiguity gate constants
# LAYER2_HIGH_THRESHOLD removed — use MIN_CONFIDENCE (= 0.60) from assembler.py
LAYER2_LOW_THRESHOLD = 0.42        # floor for "plausible but uncertain" embeddings
LAYER2_MIN_NEIGHBORS = 3           # genuine ambiguity requires >= 3 mid-band options
LAYER2_MIN_GENUINE_AMBIG = 3       # at least 3 competing high-conf matches
LAYER2_SPREAD_THRESHOLD = 0.08  # max - min confidence spread for Signal B

# Canonical CSV path relative to this file
_SNOMED_CSV_DEFAULT = Path(__file__).parent.parent / "data" / "snomed_clinical_trials.csv"


# ---------------------------------------------------------------------------
# Layer 1 module-level helpers (auto-derived triggers)
# ---------------------------------------------------------------------------

def _extract_anatomy_tokens(preferred_terms: list[str]) -> dict[str, list[str]]:
    """Scan a list of lowercase preferred_terms and return a mapping of
    {token: [preferred_term_1, preferred_term_2, ...]} for tokens that pass
    the length gate and are not in _ANATOMY_STOPWORDS.

    Used by _build_derived_entries to identify candidate trigger tokens.
    HIPAA: token text NOT logged.
    """
    token_to_terms: dict[str, list[str]] = defaultdict(list)
    for preferred_term in preferred_terms:
        raw_tokens = re.findall(r'\S+', preferred_term)
        seen_in_this_term: set[str] = set()
        for raw_tok in raw_tokens:
            token_lower = raw_tok.lower().strip(".,;:'\"")
            if len(token_lower) < MIN_DERIVED_TOKEN_LEN:
                continue
            if token_lower in _ANATOMY_STOPWORDS:
                continue
            if token_lower in seen_in_this_term:
                continue
            seen_in_this_term.add(token_lower)
            token_to_terms[token_lower].append(preferred_term)
    return dict(token_to_terms)


def _select_derived_options(
    matching_terms: list[str],
    max_options: int = AMBIG_JSON_MAX_OPTIONS,
) -> list[str]:
    """From a list of source preferred_terms, select up to max_options display strings.

    Selection: shortest preferred_terms first; lexicographic tiebreak; cap at max_options.
    Returns RAW CSV values (lowercase as stored). NO .title() call (B4).
    Returns [] if fewer than AMBIG_JSON_MIN_OPTIONS source terms.
    """
    if len(matching_terms) < AMBIG_JSON_MIN_OPTIONS:
        return []
    sorted_terms = sorted(matching_terms, key=lambda t: (len(t), t))
    return sorted_terms[:max_options]


def _build_derived_entries(csv_path: Path) -> dict[str, "AmbiguousEntry"]:
    """Scan preferred_terms in the SNOMED CSV; derive AmbiguousEntry objects for anatomy/
    system tokens that appear in >= MIN_DERIVED_TERM_FREQUENCY distinct preferred_terms.

    Options are the source preferred_terms themselves (guaranteed 0.99 exact match).
    Does NOT call the SNOMED strategy for validation (options are CSV preferred_terms).
    Returns {} on any failure (caller wraps in try/except for graceful degrade).

    HIPAA: token text and option strings NOT logged.
    """
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    # Deduplicate by preferred_term text to avoid double-counting concept_id duplicates
    unique_terms: list[str] = (
        df["preferred_term"].str.strip().str.lower().drop_duplicates().tolist()
    )

    # Step 1: extract candidate tokens → source preferred_terms mapping
    token_to_terms = _extract_anatomy_tokens(unique_terms)

    # Step 2: apply frequency threshold
    qualified: dict[str, list[str]] = {
        token: sources
        for token, sources in token_to_terms.items()
        if len(sources) >= MIN_DERIVED_TERM_FREQUENCY
    }

    # Build a flat set of all synonyms containing each token for override derivation.
    # This ensures compound terms like "lung cancer" (a synonym) suppress the derived
    # "lung" trigger — matching the same suppression logic as _derive_overrides.
    trigger_pattern_cache: dict[str, re.Pattern] = {}

    def _get_synonyms_containing(token: str) -> set[str]:
        """Return all CSV synonyms that contain token as a whole word (lowercase)."""
        if token not in trigger_pattern_cache:
            trigger_pattern_cache[token] = re.compile(
                rf"\b{re.escape(token)}\b", re.IGNORECASE
            )
        pat = trigger_pattern_cache[token]
        result: set[str] = set()
        for _, row in df.iterrows():
            for syn in row["synonyms"].split("|"):
                syn_clean = syn.strip().lower()
                if syn_clean and pat.search(syn_clean):
                    result.add(syn_clean)
        return result

    # Step 3: build AmbiguousEntry per qualified token
    derived: dict[str, AmbiguousEntry] = {}
    for token, source_terms in qualified.items():
        options = _select_derived_options(source_terms, max_options=AMBIG_JSON_MAX_OPTIONS)
        if len(options) < AMBIG_JSON_MIN_OPTIONS:
            logger.info(
                "derived trigger skipped: too few options (count=%d token_len=%d)",
                len(options), len(token),
                # NOT logged: token text, option strings
            )
            continue
        question_template = "Which type of {trigger} condition are you looking for?"
        logger.info(
            "auto-derived entry: token_len=%d option_count=%d "
            "(options are CSV preferred_terms, skipping SNOMED validation)",
            len(token), len(options),
        )
        # Override terms: options + all CSV preferred_terms + synonyms containing the
        # token as a whole word. This suppresses compound queries like "lung cancer"
        # from firing the bare "lung" trigger (mirrors _derive_overrides CSV scan +
        # extends to synonyms to catch alias-based compound queries).
        override_set: set[str] = {opt.lower() for opt in options}
        # Add preferred_terms containing token
        tok_pat = re.compile(rf"\b{re.escape(token)}\b", re.IGNORECASE)
        for pt in unique_terms:
            if tok_pat.search(pt):
                override_set.add(pt.lower())
        # Add synonyms containing token
        override_set.update(_get_synonyms_containing(token))
        # Self-defeat guard: remove bare trigger from override_terms
        override_set.discard(token.lower())

        derived[token] = AmbiguousEntry(
            trigger=token,
            category="indication",
            question_template=question_template,
            options=options,
            override_terms=frozenset(override_set),
            max_options=AMBIG_JSON_MAX_OPTIONS,
        )
    return derived


def _merge_entries(
    json_entries: dict[str, "AmbiguousEntry"],
    derived_entries: dict[str, "AmbiguousEntry"],
) -> dict[str, "AmbiguousEntry"]:
    """Merge hand-curated JSON entries with auto-derived entries.

    Hand-curated entries (json_entries) ALWAYS win on key collision (M5).
    Logs count + simple checksum — no trigger text (HIPAA).
    """
    merged: dict[str, AmbiguousEntry] = dict(derived_entries)  # lower priority
    collision_count = 0
    for trigger, entry in json_entries.items():
        if trigger in merged:
            collision_count += 1
        merged[trigger] = entry  # overwrite: hand-curated always wins
    checksum = len(derived_entries) + len(json_entries) + collision_count
    logger.info(
        "AmbiguousTermsRegistry merge: json=%d derived=%d collisions=%d total=%d checksum=%d",
        len(json_entries), len(derived_entries), collision_count, len(merged), checksum,
        # NOT logged: trigger keys, option text
    )
    return merged


# ---------------------------------------------------------------------------
# Layer 2 module-level helpers (embedding ambiguity gate)
# ---------------------------------------------------------------------------

def _deduplicate_neighbors(
    candidates: list[SNOMEDMatch],
    max_options: int = AMBIG_JSON_MAX_OPTIONS,
) -> list[str]:
    """Deduplicate candidates by concept code (highest-confidence wins per code).

    Sort descending by confidence; cap at max_options.
    Returns display strings as-is from the CSV (lowercase). No .title() call (B4).
    """
    best_per_code: dict[str, SNOMEDMatch] = {}
    for m in candidates:
        if m.code not in best_per_code or m.confidence > best_per_code[m.code].confidence:
            best_per_code[m.code] = m
    sorted_matches = sorted(best_per_code.values(), key=lambda m: -m.confidence)
    selected = sorted_matches[:max_options]
    return [m.display for m in selected]

# Hard-coded fallback for DEFAULT_CONDITION_PROMPT when CSV inspection fails
_FALLBACK_CONDITION_OPTIONS = ["Cancer", "Diabetes", "Heart Disease", "Autoimmune", "Other"]

# Category-seed keywords used for clustering CSV terms into DEFAULT_CONDITION_PROMPT options
_CATEGORY_SEEDS: dict[str, list[str]] = {
    "Cancer": ["cancer", "carcinoma", "leukemia", "lymphoma", "melanoma", "tumor"],
    "Diabetes": ["diabetes", "diabetic", "glucose", "insulin"],
    "Heart Disease": ["cardiac", "heart", "cardiovascular", "coronary", "myocardial"],
    "Autoimmune": ["lupus", "rheumatoid", "sclerosis", "inflammatory", "autoimmune"],
    "Neurological": ["alzheimer", "parkinson", "epilepsy", "neurolog", "brain"],
}


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _build_default_condition_options(csv_path: Path) -> list[str]:
    """Scan the SNOMED CSV and return top-5 most-represented category labels.

    M8: NOT called at import time. Called lazily from SufficiencyGate.__init__()
    on the first instantiation and the result is cached at the class level.

    HIPAA: option strings are not logged.
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
        count = sum(1 for term in terms_lower if any(seed in term for seed in seeds))
        if count > 0:
            category_counts[category_label] = count

    if not category_counts:
        return list(_FALLBACK_CONDITION_OPTIONS)

    sorted_categories = sorted(category_counts.items(), key=lambda kv: kv[1], reverse=True)
    top_labels = [label for label, _ in sorted_categories[:5]]

    # Pad to at least 3 with fallbacks if needed
    if len(top_labels) < 3:
        extras = [o for o in _FALLBACK_CONDITION_OPTIONS if o not in top_labels]
        top_labels.extend(extras[: 3 - len(top_labels)])

    logger.info(
        "_build_default_condition_options: built %d options from CSV",
        len(top_labels),
        # NOT logged: the option strings themselves
    )
    return top_labels


def _any_filter_set(filters) -> bool:
    """Return True if at least one filter field has a non-empty value.

    Duck-typed: works with any object that has the ExtractedFilters field layout.
    - filters.investigator_name.value  (str | None)
    - filters.site_name.value          (str | None)
    - filters.city.value               (str | None)
    - filters.state.values             (list[str])
    - filters.phase.value              (str | None)

    HIPAA: filter values are never logged or inspected beyond truthiness.
    """
    return bool(
        filters.investigator_name.value
        or filters.site_name.value
        or filters.city.value
        or filters.state.values      # list[str] — non-empty list is truthy
        or filters.phase.value
    )


def _count_set_filters(filters) -> int:
    """Return the count of non-empty filter fields (0-5)."""
    count = 0
    if filters.investigator_name.value:
        count += 1
    if filters.site_name.value:
        count += 1
    if filters.city.value:
        count += 1
    if filters.state.values:
        count += 1
    if filters.phase.value:
        count += 1
    return count


# ---------------------------------------------------------------------------
# Pydantic V2 models
# ---------------------------------------------------------------------------

class AmbiguousEntry(BaseModel):
    """Frozen Pydantic V2 model for a single ambiguous-trigger registry entry.

    Built by AmbiguousTermsRegistry at load time; never mutated afterward.
    """

    model_config = ConfigDict(frozen=True)

    trigger: str                    # lowercase canonical key (e.g. "cancer")
    category: str                   # "indication" | "phase" | "geography" | "population"
    question_template: str          # supports {trigger} and {prior_filters} placeholders
    options: list[str]              # 3-5 display-ready option strings (e.g. "Lung Cancer")
    override_terms: frozenset[str]  # post-derivation, all lowercase
    max_options: int                # informational; assembler renders up to this many

    @field_validator("options")
    @classmethod
    def _options_length(cls, v: list[str]) -> list[str]:
        if not (AMBIG_JSON_MIN_OPTIONS <= len(v) <= AMBIG_JSON_MAX_OPTIONS):
            raise ValueError(
                f"options must have {AMBIG_JSON_MIN_OPTIONS}-{AMBIG_JSON_MAX_OPTIONS} entries, "
                f"got {len(v)}"
            )
        return v


class SufficiencyDecision(BaseModel):
    """Frozen Pydantic V2 model carrying the gate's verdict.

    Consumed by src/conversation.py Turn.decision (Wave 3B).  Match this schema exactly.
    """

    model_config = ConfigDict(frozen=True)

    sufficient: bool
    reason: str
    # Reason enum values:
    #   "ok_no_trigger"           — pre-extraction gate passed
    #   "ambiguous_trigger"       — pre-extraction gate fired
    #   "max_turns_reached"       — escape valve forced sufficient=True
    #   "ok_post_extraction"      — post-extraction check passed
    #   "filters_without_condition" — post-extraction check fired
    #   "embedding_ambiguity"     — Layer 2 embedding gate fired (step 5b)

    triggered_by: Optional[str] = None          # trigger key (not logged)
    matched_entry: Optional[AmbiguousEntry] = None


# ---------------------------------------------------------------------------
# AmbiguousTermsRegistry
# ---------------------------------------------------------------------------

class AmbiguousTermsRegistry:
    """Loads data/ambiguous_terms.json, validates every option against the SNOMED strategy,
    derives override_terms for each trigger, and compiles a combined regex for fast
    whole-word trigger detection at runtime.

    Thread-safety: all mutable state is built in __init__ and then read-only.
    find_trigger() is a pure read; safe for concurrent calls.
    """

    def __init__(
        self,
        path: str,
        snomed_strategy: SNOMEDSearchStrategy,
        snomed_csv_path: str,
        strict_validation: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        path:
            Filesystem path to ambiguous_terms.json.
        snomed_strategy:
            A fully-initialized SNOMEDSearchStrategy used to validate options.
        snomed_csv_path:
            Path to snomed_clinical_trials.csv (used for override derivation).
        strict_validation:
            True  → invalid option → logger.error + sys.exit(1)  (CI / test default)
            False → invalid option → logger.warning; entry dropped (rolling-deploy mode)
        """
        self._strategy = snomed_strategy
        self._csv_path = snomed_csv_path
        self._strict = strict_validation
        self._entries: dict[str, AmbiguousEntry] = {}

        # --- Load raw JSON ---
        try:
            raw: dict = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error(
                "AmbiguousTermsRegistry: cannot load JSON (%s)",
                type(exc).__name__,
                # NOT logged: exc message (may contain filesystem path details)
            )
            sys.exit(1)

        invalid_count = 0
        valid_count = 0
        for raw_trigger, data in raw.items():
            trigger_key = raw_trigger.lower().strip()
            try:
                entry = self._validate_and_build(trigger_key, data)
                self._entries[trigger_key] = entry
                valid_count += 1
            except ValueError as exc:
                invalid_count += 1
                if self._strict:
                    logger.error(
                        "ambiguous_terms.json: trigger (len=%d) failed validation: %s",
                        len(trigger_key),
                        str(exc),
                        # NOT logged: trigger key, option text, full data dict
                    )
                    sys.exit(1)
                else:
                    logger.warning(
                        "ambiguous_terms.json: trigger (len=%d) skipped (strict=False): %s",
                        len(trigger_key),
                        str(exc),
                    )

        # B3 FIX: guard against empty registry BEFORE attempting regex compilation.
        # An empty alternation r'\b()\b' is malformed and raises re.error.
        # Even strict_validation=False can drain all entries.
        if not self._entries:
            raise ValueError(
                "AmbiguousTermsRegistry: no valid entries after validation "
                f"(valid={valid_count}, invalid={invalid_count})"
            )

        # ── Auto-derive triggers from SNOMED CSV (Layer 1) ────────────────────
        # Runs AFTER JSON validation so collision detection is correct.
        # Derived entries are NOT subject to strict_validation — they are CSV-sourced.
        json_entries = dict(self._entries)  # snapshot before merge
        try:
            derived_entries = _build_derived_entries(Path(snomed_csv_path))
        except Exception as exc:
            logger.warning(
                "AmbiguousTermsRegistry: derived entry build failed (%s) — using JSON-only",
                type(exc).__name__,
                # NOT logged: exc message
            )
            derived_entries = {}

        # AMBIG_DERIVED_DEBUG: only honored in dev environment
        if (
            os.environ.get("AMBIG_DERIVED_DEBUG", "").lower() == "true"
            and os.environ.get("ENV") == "dev"
        ):
            logger.info(
                "AmbiguousTermsRegistry AMBIG_DERIVED_DEBUG: derived_count=%d",
                len(derived_entries),
            )

        # Merge: JSON wins on collision (M5)
        merged_entries = _merge_entries(json_entries, derived_entries)
        self._entries = merged_entries

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
            "AmbiguousTermsRegistry: %d hand-curated + %d derived (skipped %d collisions)",
            len(json_entries),
            len(derived_entries),
            sum(1 for k in derived_entries if k in json_entries),
            # NOT logged: trigger keys, option text
        )

    def _validate_and_build(self, trigger: str, data: dict) -> AmbiguousEntry:
        """Full validation + derivation for a single JSON entry.

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
                f"options count {len(options)} outside "
                f"[{AMBIG_JSON_MIN_OPTIONS}, {AMBIG_JSON_MAX_OPTIONS}]"
            )

        # --- Step 2: SNOMED resolution check for each option ---
        for opt in options:
            matches = self._strategy.search(opt)
            max_conf = max((m.confidence for m in matches), default=0.0)
            if max_conf < OPTION_MIN_CONFIDENCE:
                raise ValueError(
                    f"option did not resolve at ≥{OPTION_MIN_CONFIDENCE} "
                    f"(max_conf={max_conf:.3f}, strategy={self._strategy.name})"
                    # NOT logged or included: opt text (option text is content data)
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
        """Auto-derivation algorithm (spec §3.1):

        1. Seed with lowercased option strings.
        2. Scan CSV preferred_terms for whole-word containment of trigger.
        3. Remove the trigger itself (self-defeat fix — M3 fix extended to manual_overrides).
        4. Union with filtered manual_override_terms (lowercased).

        Returns a set[str] of lowercase override terms.

        HIPAA: override term strings are not logged.
        """
        auto_overrides: set[str] = {opt.lower() for opt in options}

        # --- CSV scan ---
        try:
            df = pd.read_csv(self._csv_path, dtype=str).fillna("")
        except (OSError, pd.errors.ParserError) as exc:
            logger.warning(
                "_derive_overrides: CSV read failed (%s); using options-only overrides",
                type(exc).__name__,
                # NOT logged: exc message
            )
            df = pd.DataFrame(columns=["preferred_term"])

        trigger_word_pattern = re.compile(
            rf"\b{re.escape(trigger)}\b", re.IGNORECASE
        )
        for preferred_term in df["preferred_term"].str.strip().str.lower():
            if preferred_term and trigger_word_pattern.search(preferred_term):
                auto_overrides.add(preferred_term)

        # --- Self-defeat fix: remove the bare trigger from auto_overrides ---
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

        final_overrides = auto_overrides | filtered_manual

        logger.info(
            "_derive_overrides: trigger_len=%d → total_overrides=%d auto=%d manual_raw=%d",
            len(trigger),
            len(final_overrides),
            len(auto_overrides),
            len(manual_override_terms),
            # NOT logged: trigger value, override term strings
        )
        return final_overrides

    def find_trigger(self, query: str) -> Optional[tuple[str, AmbiguousEntry]]:
        """Return (trigger, entry) for the FIRST trigger not overridden in the query.

        Returns None if no trigger fires or all triggers are suppressed by overrides.

        Iteration order = _entries insertion order = JSON key order (deterministic).
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
                    break  # one override is enough to suppress this trigger

            if not override_present:
                logger.debug(
                    "find_trigger: fired (trigger redacted), entry.category=%s",
                    entry.category,
                    # NOT logged: trigger value (it may be a common medical word)
                )
                return (trigger, entry)

        return None

    @property
    def entry_count(self) -> int:
        """Number of valid entries in the registry."""
        return len(self._entries)


# ---------------------------------------------------------------------------
# SufficiencyGate
# ---------------------------------------------------------------------------

class SufficiencyGate:
    """Deterministic pre-extraction + post-extraction sufficiency decisions.

    No LLM calls. No state mutation. Thread-safe post-init.
    """

    # M8 FIX: DEFAULT_CONDITION_PROMPT is built LAZILY on first SufficiencyGate
    # instantiation, not at module import time. Class-level attribute caches the
    # result so multiple SufficiencyGate instances (e.g., in tests) share one build.
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
                override_terms=frozenset(),  # no override suppression for post-extraction
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
        """Pre-extraction gate. Three rules only (no rule 4 — see proposal §3.2).

        HIPAA: canonical_query NOT logged. Only sufficient bool and reason enum are logged.
        """
        # Rule 1: max-turns escape valve.
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
        filters,
    ) -> SufficiencyDecision:
        """Post-extraction safety check.

        Fires when the algorithmic SNOMED search found zero high-confidence matches
        but the LLM extracted at least one non-null filter field.

        This catches queries like "What's at Mayo?" — geographic/site interest with
        no condition.

        HIPAA: filter values and snomed display strings are NOT logged.
        """
        high_conf_matches = [
            m for m in snomed_matches if m.confidence >= 0.60 and not m.negated
        ]
        filter_count = _count_set_filters(filters)

        if not high_conf_matches and _any_filter_set(filters):
            logger.info(
                "SufficiencyGate.post_extraction_check: sufficient=False "
                "reason=filters_without_condition snomed_count=0 filter_count=%d",
                filter_count,
                # NOT logged: filter values, snomed display
            )
            return SufficiencyDecision(
                sufficient=False,
                reason="filters_without_condition",
                triggered_by=None,
                matched_entry=self._default_condition_prompt,
            )

        logger.info(
            "SufficiencyGate.post_extraction_check: sufficient=True "
            "reason=ok_post_extraction snomed_count=%d filter_count=%d",
            len(high_conf_matches),
            filter_count,
        )
        return SufficiencyDecision(sufficient=True, reason="ok_post_extraction")


# ---------------------------------------------------------------------------
# EmbeddingAmbiguityGate  (Layer 2)
# ---------------------------------------------------------------------------

class EmbeddingAmbiguityGate:
    """Layer 2 embedding-based ambiguity fallback.

    Fires between SNOMED search (step 5 negation) and clinical-intent rejection (step 6).
    Uses the embedding model already loaded by the SNOMED strategy (duck-typed via
    hasattr(strategy, "get_top_neighbors")).

    Does NOT call LLM. Does NOT mutate session. Thread-safe post-init.
    """

    def __init__(self, strategy: SNOMEDSearchStrategy) -> None:
        self._strategy = strategy
        self._has_embeddings: bool = hasattr(strategy, "get_top_neighbors")
        self._unavailable_logged: bool = False
        logger.info(
            "EmbeddingAmbiguityGate initialized: has_embeddings=%s",
            self._has_embeddings,
        )

    def evaluate(
        self,
        canonical: str,
        snomed_matches: list[SNOMEDMatch],
    ) -> Optional[SufficiencyDecision]:
        """Evaluate whether Layer 2 should fire.

        Precondition: snomed_matches MUST be post-NegationAnnotator (M4).
        Negated matches are filtered out internally before any signal computation.

        Returns SufficiencyDecision(sufficient=False, reason="embedding_ambiguity")
        if a signal fires, else None (pipeline proceeds unchanged).

        HIPAA: canonical NOT logged. Only match counts and confidence bounds logged.
        """
        # Import MIN_CONFIDENCE from assembler — single source of truth (B5)
        from src.assembler import ResponseAssembler
        min_confidence: float = ResponseAssembler.MIN_CONFIDENCE

        if not self._has_embeddings:
            if not self._unavailable_logged:
                logger.info(
                    "EmbeddingAmbiguityGate: strategy lacks get_top_neighbors — "
                    "gate is no-op (logged once)"
                )
                self._unavailable_logged = True
            return None

        # Filter negated matches first (M4 precondition enforced here)
        qualifying = [
            m for m in snomed_matches
            if m.confidence >= min_confidence and not m.negated
        ]

        # ── Signal B: multiple high-conf matches with narrow spread (genuine ambiguity)
        signal_b = False
        if len(qualifying) >= LAYER2_MIN_GENUINE_AMBIG:
            confs = [m.confidence for m in qualifying]
            if (max(confs) - min(confs)) < LAYER2_SPREAD_THRESHOLD:
                signal_b = True

        # ── Signal A: zero high-conf matches but sufficient mid-band neighbors
        signal_a = False
        mid_band_candidates: list[SNOMEDMatch] = []
        if len(qualifying) == 0:
            # Call strategy.get_top_neighbors for mid-band candidates
            neighbors = self._strategy.get_top_neighbors(  # type: ignore[attr-defined]
                canonical,
                n=AMBIG_JSON_MAX_OPTIONS * 3,  # 15 — over-fetch for dedup
                low_threshold=LAYER2_LOW_THRESHOLD,
            )
            # Filter to mid-band: [low_threshold, min_confidence)
            mid_band = [
                m for m in neighbors
                if m.confidence < min_confidence and not m.negated
            ]
            # Dedup per concept code, keeping highest confidence
            seen_codes: dict[str, SNOMEDMatch] = {}
            for m in mid_band:
                if m.code not in seen_codes or m.confidence > seen_codes[m.code].confidence:
                    seen_codes[m.code] = m
            deduped = list(seen_codes.values())
            if len(deduped) >= LAYER2_MIN_NEIGHBORS:
                signal_a = True
                mid_band_candidates = deduped

        if not (signal_a or signal_b):
            return None

        # Collect candidates for option building
        if signal_a:
            candidates = mid_band_candidates
        else:
            candidates = qualifying  # Signal B uses high-conf matches

        options = _deduplicate_neighbors(candidates, max_options=AMBIG_JSON_MAX_OPTIONS)
        if len(options) < AMBIG_JSON_MIN_OPTIONS:
            logger.info(
                "EmbeddingAmbiguityGate: insufficient distinct options (count=%d) — pass-through",
                len(options),
            )
            return None

        # Build a synthetic AmbiguousEntry for the assembler
        # All output strings flow through html.escape() in ResponseAssembler.build_clarification
        entry = AmbiguousEntry(
            trigger="<embedding_ambiguity>",
            category="indication",
            question_template="Which condition are you looking for?",
            options=options,
            override_terms=frozenset(opt.lower() for opt in options),
            max_options=AMBIG_JSON_MAX_OPTIONS,
        )

        logger.info(
            "EmbeddingAmbiguityGate fired: signal=%s candidates=%d options=%d qualifying=%d",
            "A" if signal_a else "B",
            len(candidates), len(options), len(qualifying),
            # NOT logged: canonical, option text, candidate display strings
        )

        # triggered_by=None always — APPEND mode (B2: no single trigger word to substitute)
        return SufficiencyDecision(
            sufficient=False,
            reason="embedding_ambiguity",
            triggered_by=None,
            matched_entry=entry,
        )
