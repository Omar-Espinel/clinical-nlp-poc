"""
src/conversation.py — Multi-turn conversation state for the clinical NLP pipeline.

Architect decisions:
- Turn: frozen Pydantic V2 model (immutable record per turn).
- ConversationSession: mutable Pydantic V2 model owning canonical_query, turns list.
- M2 fix: compute_canonical_query (pure) / set_canonical_query (mutator) split so that
  pipeline.run_with_session() can call assert_safe() between compute and set.

HIPAA log hygiene:
- LOGS:   session_id, turn_index, clarification_count, timestamps, on_error events.
- NEVER:  user_input, canonical_query, filter values, to_dict() output,
          decision.triggered_by values, SNOMED display strings.
- All log statements involving session state must use summary_for_logging(), never
  to_dict() or repr(session).
"""

from __future__ import annotations

import logging
import re
import time
from typing import TYPE_CHECKING, Any, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, ValidationError

from src.snomed_search.base import SNOMEDMatch
from src.sufficiency_gate import SufficiencyDecision

# Wave 4 types — not yet implemented.  Import only for type-checking so that
# test code can pass SimpleNamespace / mock objects at runtime without errors.
if TYPE_CHECKING:
    from src.extractor import ExtractedFilters  # noqa: F401 (Wave 4)
    from src.normalizers.geo import GeoResult   # noqa: F401 (Wave 4)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Turn
# ---------------------------------------------------------------------------

class Turn(BaseModel):
    """Immutable record of a single conversation turn.

    Fields that belong to the extraction path (filters, snomed_matches, geo) are
    None / empty for clarification turns.  They will be populated by the Wave 4-5
    pipeline steps.
    """

    model_config = ConfigDict(frozen=True)

    turn_index: int
    user_input: str
    canonical_query: str
    decision: Optional[SufficiencyDecision] = None
    # Wave 4 types not yet available — use Any at runtime.
    filters: Optional[Any] = None          # ExtractedFilters | None
    snomed_matches: list[SNOMEDMatch] = []  # empty for clarification turns
    geo: Optional[Any] = None              # GeoResult | None
    timestamp: float


# ---------------------------------------------------------------------------
# ConversationSession
# ---------------------------------------------------------------------------

class ConversationSession(BaseModel):
    """Mutable multi-turn session.

    Single source of truth for:
    - canonical_query (merged from all turns)
    - clarification_turn_count()
    - is_max_turns_reached()
    """

    model_config = ConfigDict(frozen=False)

    session_id: str
    turns: list[Turn] = []
    canonical_query: str = ""
    max_clarification_turns: int = 3
    created_at: float

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def new(cls) -> "ConversationSession":
        """Create a fresh session with a unique session_id and current timestamp."""
        return cls(
            session_id=uuid4().hex,
            turns=[],
            canonical_query="",
            max_clarification_turns=3,
            created_at=time.time(),
        )

    # ------------------------------------------------------------------
    # Turn management
    # ------------------------------------------------------------------

    def append_turn(self, turn: Turn) -> None:
        """Append an immutable Turn to the session history."""
        self.turns.append(turn)
        logger.debug(
            "append_turn: session_id=%s turn_index=%d",
            self.session_id,
            turn.turn_index,
            # NOT logged: user_input, canonical_query, decision.triggered_by
        )

    # ------------------------------------------------------------------
    # Clarification counting / gate
    # ------------------------------------------------------------------

    def clarification_turn_count(self) -> int:
        """Count turns where decision is not None AND decision.sufficient == False.

        Single source of truth — no other code maintains a separate counter.
        """
        return sum(
            1
            for t in self.turns
            if t.decision is not None and not t.decision.sufficient
        )

    def is_max_turns_reached(self) -> bool:
        """True iff clarification_turn_count() >= max_clarification_turns."""
        return self.clarification_turn_count() >= self.max_clarification_turns

    def last_clarification(self) -> Optional[SufficiencyDecision]:
        """Return the last turn's decision if it was a non-sufficient clarification.

        Returns None if the session has no turns or the last turn was sufficient.
        """
        if not self.turns:
            return None
        last_decision = self.turns[-1].decision
        if last_decision is not None and not last_decision.sufficient:
            return last_decision
        return None

    # ------------------------------------------------------------------
    # Canonical query: M2 fix — compute (pure) / set (mutator) split
    # ------------------------------------------------------------------

    def compute_canonical_query(self, user_input: str) -> str:
        """Pure: compute the merged canonical query WITHOUT mutating self.canonical_query.

        Algorithm (substitute-or-append):
        - No prior turns: return user_input as-is.
        - Last turn had a non-sufficient decision with triggered_by set:
            - If the trigger word appears (whole-word, case-insensitive) in the
              current canonical_query, substitute it with user_input.
            - Otherwise (e.g. plural "cancers" vs trigger "cancer"): append.
        - All other cases (free-text follow-up, post-extraction safety): append.

        PURE: does not modify self.canonical_query.
        """
        if not self.turns:
            return user_input

        last = self.turns[-1]
        last_decision = last.decision

        # Are we answering a clarification?
        if last_decision and not last_decision.sufficient and last_decision.triggered_by:
            trigger = last_decision.triggered_by
            pattern = re.compile(rf"\b{re.escape(trigger)}\b", re.IGNORECASE)
            if pattern.search(self.canonical_query):
                return pattern.sub(user_input, self.canonical_query)
            else:
                # Defensive fallback: trigger not literally present in canonical
                # (e.g. user typed "cancers", trigger key is "cancer" — \b won't match).
                return f"{self.canonical_query} {user_input}".strip()
        else:
            # Free-text follow-up or post-extraction safety (triggered_by=None).
            return f"{self.canonical_query} {user_input}".strip()

    def set_canonical_query(self, query: str) -> None:
        """Mutator: write query to self.canonical_query.

        Called only after assert_safe() passes in run_with_session().
        """
        self.canonical_query = query

    def update_canonical_query(self, user_input: str) -> str:
        """Convenience: compute then set canonical query.

        Retained for legacy run() shim and tests that do not need the
        assert_safe guard in between.  New pipeline code uses the split pair:
            canonical = session.compute_canonical_query(preprocessed.text)
            preprocessor.assert_safe(canonical)
            session.set_canonical_query(canonical)
        """
        canonical = self.compute_canonical_query(user_input)
        self.set_canonical_query(canonical)
        return canonical

    # ------------------------------------------------------------------
    # Filter summary (for clarification question template substitution)
    # ------------------------------------------------------------------

    def summarize_known_filters(self) -> str:
        """Return a short human-readable summary of known filter values.

        Reads turns[-1].filters if available.  Returns a comma-joined string
        such as "Boston, Phase 3" or "" if no filters are set.

        Used in question_template {prior_filters} substitution by the assembler.
        HIPAA: this method returns values; callers must NOT log the return value.
        """
        if not self.turns:
            return ""
        last_filters = self.turns[-1].filters
        if last_filters is None:
            return ""

        parts: list[str] = []
        # Duck-typed access — ExtractedFilters not yet available (Wave 4).
        try:
            if getattr(getattr(last_filters, "city", None), "value", None):
                parts.append(last_filters.city.value)
            state_obj = getattr(last_filters, "state", None)
            if state_obj is not None:
                states = getattr(state_obj, "values", []) or []
                parts.extend(states)
            if getattr(getattr(last_filters, "phase", None), "value", None):
                parts.append(last_filters.phase.value)
            if getattr(getattr(last_filters, "investigator_name", None), "value", None):
                parts.append(last_filters.investigator_name.value)
            if getattr(getattr(last_filters, "site_name", None), "value", None):
                parts.append(last_filters.site_name.value)
        except Exception:
            # Defensive: if duck-typing fails for any reason, return empty
            return ""

        return ", ".join(parts)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize session to a JSON-safe dict.

        HIPAA: do NOT pass the return value to any log statement.
        Use summary_for_logging() for log-safe session info.
        """
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, d: dict, on_error: str = "raise") -> "ConversationSession":
        """Deserialize a session from a dict (e.g. loaded from Redis / st.session_state).

        Parameters
        ----------
        d:
            Dict produced by to_dict().
        on_error:
            "raise"       — ValidationError propagates (default).
            "new_session" — catches ValidationError, logs WARN, returns fresh session.
        """
        try:
            return cls.model_validate(d)
        except ValidationError as exc:
            if on_error == "new_session":
                logger.warning(
                    "ConversationSession.from_dict: corrupted state, starting fresh (%s)",
                    type(exc).__name__,
                    # NOT logged: exc details (may contain user data in validation context)
                )
                return cls.new()
            raise

    # ------------------------------------------------------------------
    # Log-safe summary
    # ------------------------------------------------------------------

    def summary_for_logging(self) -> dict:
        """Return ONLY log-safe fields.

        Safe to pass directly to logger.  Contains NO user_input, canonical_query,
        filter values, SNOMED display strings, or decision.triggered_by.
        """
        return {
            "session_id": self.session_id,
            "turn_count": len(self.turns),
            "clarification_count": self.clarification_turn_count(),
            "max_turns": self.max_clarification_turns,
            "created_at": self.created_at,
        }
