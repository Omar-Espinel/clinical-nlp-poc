"""Orchestrates the full Clinical NLP pipeline (v2 rework)."""

import datetime
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, wait as futures_wait, ALL_COMPLETED
from pathlib import Path
from typing import Optional, Union

from src.assembler import ResponseAssembler, ClarificationOutput, NLPOutput
from src.conversation import ConversationSession, Turn
from src.exceptions import PipelineError, LLMProviderError
from src.filter_extractor import FilterExtractor, ExtractedFilters, FilterField, StateFilter
from src.llm_provider.base import LLMProvider
from src.llm_provider.registry import get_provider
from src.normalizers.geo import GeoNormalizer
from src.normalizers.metric import MetricIntentResolver, MetricMatch, MetricFilterOutput
from src.preprocessor import Preprocessor, PreprocessorError
from src.snomed_search.base import SNOMEDMatch, SNOMEDSearchStrategy
from src.snomed_search.negation import NegationAnnotator
from src.snomed_search.registry import get_strategy
from src.sufficiency_gate import (
    AmbiguousTermsRegistry,
    EmbeddingAmbiguityGate,
    MetricAmbiguityGate,
    SufficiencyGate,
    SufficiencyDecision,
    _count_set_filters,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_SNOMED_CSV          = str(PROJECT_ROOT / "data" / "snomed_clinical_trials.csv")
DEFAULT_GEO_PATH            = str(PROJECT_ROOT / "data" / "geo_canonical.json")
DEFAULT_AMBIG_PATH          = str(PROJECT_ROOT / "data" / "ambiguous_terms.json")
DEFAULT_METRIC_FILTERS_PATH = str(PROJECT_ROOT / "data" / "metric_filters.json")

PARALLEL_TIMEOUT_SECONDS = 15.0   # shared budget for BOTH futures combined (B2 fix)
THREAD_POOL_MAX_WORKERS  = 2
THREAD_POOL_NAME_PREFIX  = "nlp-"

# Per-turn log path enum
LOG_PATH_SUFFICIENCY_CLARIFICATION = "sufficiency_clarification"
LOG_PATH_MAX_TURNS                 = "max_turns"
LOG_PATH_CLINICAL_INTENT           = "clinical_intent"
LOG_PATH_POST_EXTRACTION_SAFETY    = "post_extraction_safety"
LOG_PATH_SEARCH                    = "search"
LOG_PATH_EMBEDDING_AMBIGUITY       = "embedding_ambiguity_clarification"
LOG_PATH_METRIC_AMBIGUITY          = "metric_ambiguity_clarification"


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _state_value_for_geo(state_filter: StateFilter) -> Optional[str]:
    """Convert StateFilter to a single state string for GeoNormalizer.

    Multi-state regions or empty state list → None (geo normalizer handles
    region expansion from city input alone).
    """
    if state_filter.is_region:
        return None
    if len(state_filter.values) == 1:
        return state_filter.values[0]
    return None


def _empty_filters() -> ExtractedFilters:
    """Return a fully-empty ExtractedFilters for the best-effort shim path (rev2 B8)."""
    return ExtractedFilters(
        investigator_name=FilterField(value=None, confidence=0.0),
        site_name=FilterField(value=None, confidence=0.0),
        city=FilterField(value=None, confidence=0.0),
        state=StateFilter(values=[], confidence=0.0, is_region=False),
        phase=FilterField(value=None, confidence=0.0),
        raw_response_length=0,
        metric_fields={},
    )


# ---------------------------------------------------------------------------
# NLPPipeline
# ---------------------------------------------------------------------------

class NLPPipeline:
    """Single entry point for the full Clinical Research NLP pipeline (v2)."""

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        snomed_strategy: Optional[SNOMEDSearchStrategy] = None,
        ambiguous_terms_path: Optional[str] = None,
        snomed_csv_path: Optional[str] = None,
        geo_json_path: Optional[str] = None,
        strict_validation: Optional[bool] = None,
        metric_filters_path: Optional[str] = None,
        metric_strict_validation: Optional[bool] = None,
        # Legacy positional kwarg accepted for backwards compat (tests/run_tests.py)
        groq_api_key: Optional[str] = None,
    ) -> None:
        """Init order is critical:
          1. SNOMED strategy first — AmbiguousTermsRegistry needs it for option validation.
          2. LLM provider second — independent.
          3. Registry third — depends on strategy.
          4. Gate, extractor, geo, negation, preprocessor, assembler — order flexible.
          4f. MetricIntentResolver and MetricAmbiguityGate after other components.
          5. ThreadPoolExecutor last — after all components ready.
        """
        _snomed_csv = snomed_csv_path or DEFAULT_SNOMED_CSV

        # Step 1: SNOMED strategy
        self._snomed: SNOMEDSearchStrategy = (
            snomed_strategy or get_strategy(dictionary_path=_snomed_csv)
        )

        # Step 2: LLM provider — accept legacy groq_api_key kwarg
        if llm_provider is not None:
            self._llm: LLMProvider = llm_provider
        else:
            # Resolve API key: explicit kwarg → GROQ_API_KEY env var → get_provider default
            _api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
            if _api_key:
                self._llm = get_provider(name="groq", api_key=_api_key)
            else:
                self._llm = get_provider()

        # Step 3: Ambiguous terms registry
        _strict: bool = (
            strict_validation
            if strict_validation is not None
            else os.environ.get("AMBIG_STRICT_VALIDATION", "false").lower() == "true"
        )
        self._registry = AmbiguousTermsRegistry(
            path=ambiguous_terms_path or DEFAULT_AMBIG_PATH,
            snomed_strategy=self._snomed,
            snomed_csv_path=_snomed_csv,
            strict_validation=_strict,
        )

        # Step 4: remaining components
        self._gate           = SufficiencyGate(self._registry, snomed_csv_path=_snomed_csv)
        self._geo            = GeoNormalizer(geo_json_path or DEFAULT_GEO_PATH)
        self._negation       = NegationAnnotator()
        self._preprocessor   = Preprocessor()
        self._assembler      = ResponseAssembler()
        # Step 4e: Embedding ambiguity gate (Layer 2)
        self._embedding_gate = EmbeddingAmbiguityGate(self._snomed)

        # Step 4f: Metric intent resolver and gate (§12c)
        _metric_strict: bool = (
            metric_strict_validation
            if metric_strict_validation is not None
            else os.environ.get("METRIC_STRICT_VALIDATION", "true").lower() == "true"
        )
        self._metric_resolver = MetricIntentResolver(
            metric_filters_path or DEFAULT_METRIC_FILTERS_PATH,
            strict_validation=_metric_strict,
        )
        self._metric_gate = MetricAmbiguityGate(self._metric_resolver)

        # FilterExtractor initialized after metric_resolver so we can pass known fields
        self._extractor = FilterExtractor(
            self._llm,
            known_metric_fields=self._metric_resolver._known_metric_fields,
        )

        # Step 5: thread pool
        self._executor = ThreadPoolExecutor(
            max_workers=THREAD_POOL_MAX_WORKERS,
            thread_name_prefix=THREAD_POOL_NAME_PREFIX,
        )

        logger.info(
            "NLPPipeline initialized: strategy=%s provider=%s strict_validation=%s "
            "metric_fields=%d",
            self._snomed.name, self._llm.name, _strict,
            len(self._metric_resolver._entries),
            # NOT logged: api keys, paths, any content
        )

    # -------------------------------------------------------------------------
    # Primary entry point
    # -------------------------------------------------------------------------

    def run_with_session(
        self,
        raw_query: str,
        session: ConversationSession,
    ) -> Union[NLPOutput, ClarificationOutput]:
        """10-step orchestration.

        Session is mutated (turn appended) ONLY after successful pipeline
        completion or after a clarification is successfully built.

        HIPAA: raw_query and canonical are NEVER logged. Only counts, confidence
        scores, decision enums, and session metadata are logged.
        """
        start: float = time.perf_counter()
        log_path: str = "unknown"

        # ── Step 1: Preprocess raw user input ────────────────────────────────
        try:
            preprocessed = self._preprocessor.process(raw_query)
        except PreprocessorError:
            raise
        logger.info(
            "run_with_session step=1 preprocessed char_count=%d session_id=%s",
            preprocessed.char_count, session.session_id,
            # NOT logged: raw_query, preprocessed.text
        )

        # ── Step 2: Compute canonical query (pure — no session mutation yet) ─
        canonical: str = session.compute_canonical_query(preprocessed.text)
        # No log here — canonical contains user content.

        # ── Step 2b: Defense-in-depth injection check on merged canonical ────
        # Catches pathological substitution edge cases where individual turns
        # each passed but the merged canonical triggers a pattern.
        self._preprocessor.assert_safe(canonical)

        # Mutation only after assert_safe passes.
        session.set_canonical_query(canonical)

        # ── Step 3: Pre-extraction sufficiency gate ───────────────────────────
        decision: SufficiencyDecision = self._gate.evaluate(canonical, session)

        if not decision.sufficient:
            log_path = (
                LOG_PATH_SUFFICIENCY_CLARIFICATION
                if decision.reason == "ambiguous_trigger"
                else LOG_PATH_MAX_TURNS
            )
            clarification = self._assembler.build_clarification(decision, session, start)
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
            self._log_turn(
                session, decision, log_path, start,
                snomed_count=0, filter_count=0,
                metric_match_count=0, metric_resolved_count=0,
            )
            return clarification

        # ── Steps 4-10: Extraction, negation, intent gate, geo, assembly ──────
        return self._run_extraction_path(canonical, preprocessed.text, decision, session, start)

    # -------------------------------------------------------------------------
    # Log helper
    # -------------------------------------------------------------------------

    def _log_turn(
        self,
        session: ConversationSession,
        decision: SufficiencyDecision,
        log_path: str,
        start: float,
        snomed_count: int = 0,
        filter_count: int = 0,
        metric_match_count: int = 0,
        metric_resolved_count: int = 0,
    ) -> None:
        """Emit single structured JSON log line per turn.

        Fields: ts, session_id, turn_index, clarification_count, path,
                decision_reason, snomed_match_count, snomed_strategy,
                filter_count, metric_match_count, metric_resolved_count,
                llm_provider, processing_time_ms.
        NEVER logged: raw_query, canonical_query, user_input, filter values,
                      SNOMED display strings, LLM response, triggered_by value,
                      option text, matched_text, original_text.
        """
        log_record = {
            "ts": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "session_id": session.session_id,
            "turn_index": len(session.turns) - 1,
            "clarification_count": session.clarification_turn_count(),
            "path": log_path,
            "decision_reason": decision.reason,
            "snomed_match_count": snomed_count,
            "snomed_strategy": self._snomed.name,
            "filter_count": filter_count,
            "metric_match_count": metric_match_count,
            "metric_resolved_count": metric_resolved_count,
            "llm_provider": self._llm.name,
            "processing_time_ms": int((time.perf_counter() - start) * 1000),
        }
        logger.info("turn_log %s", json.dumps(log_record))

    # -------------------------------------------------------------------------
    # Private: extraction path (steps 4-10)
    # -------------------------------------------------------------------------

    def _run_extraction_path(
        self,
        canonical: str,
        user_text: str,
        decision: SufficiencyDecision,
        session: ConversationSession,
        start: float,
        metric_matches: Optional[list[MetricMatch]] = None,
    ) -> Union[NLPOutput, ClarificationOutput]:
        """Run steps 3b-10: metric resolution, parallel extraction, negation,
        clinical-intent gate, post-extraction safety check, metric gate,
        geo normalization, and final assembly.

        Called by run_with_session() after the sufficiency gate passes, AND
        by run() shim when bypassing the sufficiency gate for ambiguous queries.

        HIPAA: canonical and user_text are NEVER logged.
        """
        log_path: str = "unknown"

        # ── Step 3b: Metric intent resolution (deterministic, pre-LLM) ───────
        if metric_matches is None:
            metric_matches = self._metric_resolver.resolve(canonical)
            logger.info(
                "run_with_session step=3b metric_matches=%d session_id=%s",
                len(metric_matches), session.session_id,
                # NOT logged: matched_text, canonical
            )

        # ── Step 4: Parallel paths — filter LLM + algorithmic SNOMED ─────────
        fut_filters = self._executor.submit(self._extractor.extract, canonical, metric_matches)
        fut_snomed  = self._executor.submit(self._snomed.search, canonical)

        filters: ExtractedFilters
        snomed_raw: list[SNOMEDMatch]

        # B2 FIX: shared timeout budget for both futures combined.
        try:
            done, not_done = futures_wait(
                [fut_filters, fut_snomed],
                timeout=PARALLEL_TIMEOUT_SECONDS,
                return_when=ALL_COMPLETED,
            )
            if not_done:
                fut_filters.cancel()
                fut_snomed.cancel()
                logger.error(
                    "run_with_session step=4 TIMEOUT session_id=%s elapsed_ms=%.0f",
                    session.session_id, (time.perf_counter() - start) * 1000,
                )
                raise PipelineError("extraction_timeout")

            filters    = fut_filters.result()
            snomed_raw = fut_snomed.result()

        except PipelineError:
            raise
        except LLMProviderError as exc:
            fut_filters.cancel()
            fut_snomed.cancel()
            logger.error(
                "run_with_session step=4 LLMProviderError provider=%s session_id=%s",
                exc.provider_name, session.session_id,
                # NOT logged: exc.original_error.message, any response content
            )
            raise
        except Exception as exc:
            fut_filters.cancel()
            fut_snomed.cancel()
            logger.error(
                "run_with_session step=4 unexpected error type=%s session_id=%s",
                type(exc).__name__, session.session_id,
            )
            raise PipelineError("strategy_unavailable") from exc

        # ── Step 5: Negation annotation ───────────────────────────────────────
        snomed_matches: list[SNOMEDMatch] = self._negation.annotate(canonical, snomed_raw)

        logger.info(
            "run_with_session step=5 snomed_candidates=%d negated=%d session_id=%s",
            len(snomed_matches),
            sum(1 for m in snomed_matches if m.negated),
            session.session_id,
            # NOT logged: snomed display strings, canonical query
        )

        # ── Step 5b: Embedding-based ambiguity fallback (Layer 2) ─────────────
        embed_decision = self._embedding_gate.evaluate(canonical, snomed_matches)
        if embed_decision is not None:
            log_path = LOG_PATH_EMBEDDING_AMBIGUITY
            clarification = self._assembler.build_clarification(embed_decision, session, start)
            session.append_turn(Turn(
                turn_index=len(session.turns),
                user_input=user_text,
                canonical_query=canonical,
                decision=embed_decision,
                filters=None,
                snomed_matches=snomed_matches,
                geo=None,
                timestamp=time.time(),
            ))
            self._log_turn(
                session, embed_decision, log_path, start,
                snomed_count=len(snomed_matches),
                filter_count=0,
                metric_match_count=len(metric_matches),
                metric_resolved_count=0,
            )
            return clarification

        # ── Step 6: Clinical-intent gate ─────────────────────────────────────
        # M4 FIX: apply same MIN_CONFIDENCE=0.60 + not-negated filter used by the
        # assembler so the gate sees only matches that will appear in final output.
        qualifying_matches = [
            m for m in snomed_matches
            if m.confidence >= 0.60 and not m.negated
        ]
        filter_set_count = _count_set_filters(filters)

        if not qualifying_matches and filter_set_count == 0:
            log_path = LOG_PATH_CLINICAL_INTENT
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

        # ── Step 7: Post-extraction safety check ─────────────────────────────
        post_decision: SufficiencyDecision = self._gate.post_extraction_check(
            snomed_matches, filters
        )
        if not post_decision.sufficient:
            log_path = LOG_PATH_POST_EXTRACTION_SAFETY
            clarification = self._assembler.build_clarification(post_decision, session, start)
            session.append_turn(Turn(
                turn_index=len(session.turns),
                user_input=user_text,
                canonical_query=canonical,
                decision=post_decision,
                filters=filters,
                snomed_matches=snomed_matches,
                geo=None,
                timestamp=time.time(),
            ))
            self._log_turn(
                session, post_decision, log_path, start,
                snomed_count=len(snomed_matches),
                filter_count=filter_set_count,
                metric_match_count=len(metric_matches),
                metric_resolved_count=len(filters.metric_fields),
            )
            return clarification

        # ── Step 7b: Metric ambiguity gate ───────────────────────────────────
        if metric_matches:
            metric_decision = self._metric_gate.evaluate(metric_matches, filters, session)
            if metric_decision is not None:
                log_path = LOG_PATH_METRIC_AMBIGUITY
                clarification = self._assembler.build_clarification(metric_decision, session, start)
                session.append_turn(Turn(
                    turn_index=len(session.turns),
                    user_input=user_text,
                    canonical_query=canonical,
                    decision=metric_decision,
                    filters=filters,
                    snomed_matches=snomed_matches,
                    geo=None,
                    timestamp=time.time(),
                ))
                self._log_turn(
                    session, metric_decision, log_path, start,
                    snomed_count=len(snomed_matches),
                    filter_count=filter_set_count,
                    metric_match_count=len(metric_matches),
                    metric_resolved_count=0,
                )
                # NOT logged: metric_decision.triggered_by, metric_decision.matched_entry.options
                return clarification

        # ── Step 8: Geo normalization ─────────────────────────────────────────
        geo = self._geo.normalize(
            city=filters.city.value,
            state=_state_value_for_geo(filters.state),
        )
        logger.info(
            "run_with_session step=8 geo_confidence=%.3f is_region=%s session_id=%s",
            geo.confidence, geo.is_region, session.session_id,
            # NOT logged: geo.city, geo.states, filter values
        )

        # ── Step 9: Belt-and-suspenders assert_safe before final assembly ─────
        self._preprocessor.assert_safe(canonical)

        # ── Step 10: Assemble output; append turn AFTER success ───────────────
        resolved_metric_filters = list(filters.metric_fields.values()) if filters.metric_fields else []

        output: NLPOutput = self._assembler.assemble(
            filters=filters,
            snomed_matches=snomed_matches,
            geo=geo,
            start_time=start,
            metric_filters=resolved_metric_filters,
        )

        log_path = LOG_PATH_SEARCH
        session.append_turn(Turn(
            turn_index=len(session.turns),
            user_input=user_text,
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
            metric_match_count=len(metric_matches),
            metric_resolved_count=len(resolved_metric_filters),
        )
        return output

    # -------------------------------------------------------------------------
    # Backwards-compat shim
    # -------------------------------------------------------------------------

    # OBSOLETE-AT-SCALE: single-turn entry, kept for tests/run_tests.py
    def run(self, raw_query: str) -> NLPOutput:
        """Wrap run_with_session() with a fresh single-turn session.

        If the pipeline returns a ClarificationOutput (ambiguous query),
        bypasses the sufficiency gate and runs the extraction path directly
        so legacy single-shot callers always get a best-effort NLPOutput.
        Session is fresh per call — no state leaks between invocations.
        """
        session = ConversationSession.new()
        result = self.run_with_session(raw_query, session)
        if isinstance(result, ClarificationOutput):
            return self._assemble_best_effort_from_session(raw_query, session)
        return result

    def _assemble_best_effort_from_session(
        self, raw_query: str, session: ConversationSession
    ) -> NLPOutput:
        """Called when run() receives a ClarificationOutput (sufficiency gate fired).

        Bypasses the sufficiency gate and runs LLM extraction + SNOMED search
        directly on the canonical query so the legacy single-turn path always
        returns a populated NLPOutput. The clinical-intent gate is NOT bypassed.
        Used only by the legacy run() shim — not by multi-turn paths.
        """
        # The canonical was already set on the session before the gate fired.
        canonical = session.canonical_query or raw_query
        start = time.perf_counter()

        # Retrieve the sufficiency decision stored in the last turn (if any),
        # or synthesise a minimal sufficient-looking one for logging purposes.
        last_turn = session.turns[-1] if session.turns else None
        decision: SufficiencyDecision = (
            last_turn.decision
            if last_turn is not None and last_turn.decision is not None
            else SufficiencyDecision(sufficient=True, reason="legacy_bypass")
        )

        # Legacy shim passes [] for metric_matches — no metric clarification support.
        # Clean boundary: _assemble_best_effort_from_session does not call
        # MetricIntentResolver.resolve() (spec §12j D13).
        result = self._run_extraction_path(
            canonical=canonical,
            user_text=raw_query,
            decision=decision,
            session=session,
            start=start,
            metric_matches=[],
        )
        # _run_extraction_path can itself return a ClarificationOutput only if
        # post_extraction_check fires, which is exceedingly rare for real clinical
        # queries. Fall back to empty output to satisfy NLPOutput return type.
        if isinstance(result, ClarificationOutput):
            return self._assembler.assemble(
                filters=_empty_filters(),
                snomed_matches=[],
                geo=self._geo.normalize(None, None),
                start_time=start,
            )
        return result
