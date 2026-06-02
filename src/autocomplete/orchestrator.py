from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

from src.autocomplete import cache as autocomplete_cache
from src.autocomplete.index import AutocompleteIndex, AutocompleteSuggestion
from src.autocomplete.ranker import rank_and_trim
from src.autocomplete.segmenter import QuerySegmenter
from src.preprocessor import Preprocessor, PreprocessorError

if TYPE_CHECKING:
    from src.snomed_search.base import SNOMEDSearchStrategy

logger = logging.getLogger(__name__)

_MIN_PREFIX_LEN: int = 3
_MAX_PREFIX_LEN: int = 100
_DEFAULT_LIMIT: int = 7
_PARALLEL_TIMEOUT: float = float(os.environ.get("AUTOCOMPLETE_PARALLEL_TIMEOUT", "2.0"))
_MIN_RESPONSE_SECONDS: float = 0.050


class AutocompleteOrchestrator:
    """Coordinates three-tier autocomplete search pipeline.

    Constructed once at server startup from the pipeline's strategy instance.
    The run method is the sole public entry point and is thread-safe because
    all mutable state is local to each invocation.

    HIPAA: raw_prefix is never logged anywhere in this class.
    Only prefix_length, tier_used, suggestion_count, and latency_ms are logged.
    """

    def __init__(self, strategy: "SNOMEDSearchStrategy", geo_data: dict) -> None:
        self._index = AutocompleteIndex(strategy, geo_data)
        self._preprocessor = Preprocessor()

        geo_keys: frozenset[str] = (
            frozenset(geo_data.get("cities", {}).keys())
            | frozenset(geo_data.get("states", {}).keys())
            | frozenset(geo_data.get("regions", {}).keys())
        )

        # Extract SNOMED preferred terms from the strategy's existing exact
        # index so the segmenter can recognise committed SNOMED context in
        # mid-query positions. These are already in memory — no new index,
        # no new model, no duplication.
        # HIPAA: these are public SNOMED display strings, not user content.
        exact_index: dict = getattr(strategy, "_exact_index", {})
        snomed_known_terms: frozenset[str] = frozenset(
            record.get("preferred_term", "").lower().strip()
            for record in exact_index.values()
            if record.get("preferred_term")
        )

        self._segmenter = QuerySegmenter(
            geo_keys, snomed_known_terms=snomed_known_terms
        )

        logger.info(
            "AutocompleteOrchestrator initialized: geo_keys=%d snomed_terms=%d semantic=%s",
            len(geo_keys),
            len(snomed_known_terms),
            getattr(strategy, "semantic_available", "unknown"),
        )

    async def run(self, raw_prefix: str, limit: int = _DEFAULT_LIMIT) -> dict:
        """Run three-tier autocomplete and return API response dict.

        Returns empty suggestions without raising for short input,
        injection attempts, and no-match queries. Never returns a 500
        for valid UTF-8 input.

        HIPAA: raw_prefix is not logged. Only prefix_length is logged.
        """
        start = time.perf_counter()
        _empty: dict = {
            "suggestions": [],
            "processing_time_ms": 0,
            "tier_used": "none",
            "fallback_active": False,
        }

        stripped = raw_prefix.strip()
        if len(stripped) < _MIN_PREFIX_LEN or len(stripped) > _MAX_PREFIX_LEN:
            return _empty

        try:
            self._preprocessor.assert_safe(stripped)
        except PreprocessorError:
            # HIPAA: log only that an injection was rejected and the length, never the content.
            logger.info(
                "AutocompleteOrchestrator: injection rejected prefix_length=%d",
                len(stripped),
            )
            return _empty

        cached = autocomplete_cache.cache_get(stripped, limit)
        if cached is not None:
            await _enforce_min_response(start)
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            return {
                "suggestions": [_format_suggestion(s, context_prefix="") for s in cached],
                "processing_time_ms": max(elapsed_ms, int(_MIN_RESPONSE_SECONDS * 1000)),
                "tier_used": "cache",
                "fallback_active": False,
            }

        segment = self._segmenter.segment(stripped)
        if not segment.active_prefix:
            return _empty

        # Use the verbatim raw slice of the original query that precedes the
        # active prefix. This preserves the user's exact casing, punctuation,
        # and connector words (", phase 3 in ") in the assembled completion.
        # Never logged — only passed to _format_suggestion for response assembly.
        context_prefix = segment.committed_prefix_raw

        # Normalize the active prefix: detect compound phases and strip
        # noise words before sending to the search tiers.
        normalized = self._segmenter.normalize_active_prefix(segment.active_prefix)

        loop = asyncio.get_running_loop()
        tier_used_parts: list[str] = []
        all_suggestions: list[AutocompleteSuggestion] = []

        # Use normalized.search_text for tier search so noise-stripped and
        # compound-reduced text reaches the indexes, not the raw active prefix.
        tier1_fut = loop.run_in_executor(None, self._index.prefix_search, normalized.search_text, limit * 3)
        tier2_fut = loop.run_in_executor(None, self._index.fuzzy_search, normalized.search_text, limit * 3)
        tier3_fut = loop.run_in_executor(None, self._index.semantic_search, normalized.search_text, limit * 3)

        done, pending = await asyncio.wait(
            {tier1_fut, tier2_fut, tier3_fut},
            timeout=_PARALLEL_TIMEOUT,
            return_when=asyncio.ALL_COMPLETED,
        )

        for fut in pending:
            fut.cancel()
            logger.warning("AutocompleteOrchestrator: a tier timed out and was cancelled")

        for fut, label in ((tier1_fut, "tier1"), (tier2_fut, "tier2"), (tier3_fut, "tier3")):
            if fut in done:
                try:
                    tier_results = fut.result()
                    if tier_results:
                        all_suggestions.extend(tier_results)
                        tier_used_parts.append(label)
                except Exception as exc:
                    logger.warning(
                        "AutocompleteOrchestrator: %s raised (%s)",
                        label, type(exc).__name__,
                    )

        ranked = rank_and_trim(all_suggestions, segment, limit)
        autocomplete_cache.cache_store(stripped, limit, ranked)

        tier_used = "+".join(tier_used_parts) if tier_used_parts else "none"

        await _enforce_min_response(start)
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        # HIPAA: only lengths, tier label, count, latency — never text content.
        logger.info(
            "AutocompleteOrchestrator.run: prefix_length=%d tier_used=%s "
            "suggestion_count=%d latency_ms=%d is_compound_phase=%s",
            len(stripped), tier_used, len(ranked), elapsed_ms,
            normalized.is_compound_phase,
        )

        # For compound phase queries expand each ranked suggestion into one
        # completion per phase component. Deduplication is applied after
        # expansion by display string to prevent duplicates when ranked
        # suggestions happen to be phase terms already.
        if normalized.is_compound_phase and normalized.compound_phases:
            expanded: list[dict] = []
            seen_completions: set[str] = set()
            for phase in normalized.compound_phases:
                phase_suggestion = AutocompleteSuggestion(
                    display=phase,
                    snomed_code=None,
                    category="phase",
                    raw_score=1.0,
                    match_type="prefix",
                )
                fmt = _format_suggestion(phase_suggestion, context_prefix=context_prefix)
                if fmt["completion"] not in seen_completions:
                    seen_completions.add(fmt["completion"])
                    expanded.append(fmt)
            # Append any non-phase ranked suggestions after the phase expansions,
            # up to the original limit.
            for s in ranked:
                if len(expanded) >= limit:
                    break
                if s.category != "phase":
                    fmt = _format_suggestion(s, context_prefix=context_prefix)
                    if fmt["completion"] not in seen_completions:
                        seen_completions.add(fmt["completion"])
                        expanded.append(fmt)
            return {
                "suggestions": expanded,
                "processing_time_ms": max(elapsed_ms, int(_MIN_RESPONSE_SECONDS * 1000)),
                "tier_used": tier_used,
                "fallback_active": False,
            }

        return {
            "suggestions": [_format_suggestion(s, context_prefix=context_prefix) for s in ranked],
            "processing_time_ms": max(elapsed_ms, int(_MIN_RESPONSE_SECONDS * 1000)),
            "tier_used": tier_used,
            "fallback_active": False,
        }


async def _enforce_min_response(start: float) -> None:
    """Pad response time to at least _MIN_RESPONSE_SECONDS.

    Prevents timing oracle attacks where the difference in latency
    between cache hits and misses reveals information about index
    contents. Uses asyncio.sleep so the event loop is not blocked.
    """
    elapsed = time.perf_counter() - start
    remaining = _MIN_RESPONSE_SECONDS - elapsed
    if remaining > 0:
        await asyncio.sleep(remaining)


def _format_suggestion(s: AutocompleteSuggestion, context_prefix: str) -> dict:
    """Format suggestion for API response.

    Applies title casing to display because preferred_term values are
    stored lowercase in the CSV. str.title() handles most clinical terms
    correctly. Known limitation: acronyms like HIV and NSCLC become Hiv
    and Nsclc. A curated acronym list can correct this in a future pass
    if needed.

    Assembles completion by prepending context_prefix to the titled
    display. This is done here rather than on the dataclass to keep
    AutocompleteSuggestion pure and frozen.

    HIPAA: this dict is sent to the API consumer only, never logged.
    """
    display_titled = s.display.title()
    # context_prefix is a verbatim query slice that already carries its own
    # trailing delimiter/whitespace (e.g. ", " or " in "); rstrip then rejoin
    # with exactly one space so the completion never has a double space.
    completion = f"{context_prefix.rstrip()} {display_titled}" if context_prefix else display_titled
    return {
        "display":    display_titled,
        "completion": completion,
        "snomed_code": s.snomed_code,
        "category":   s.category,
        "confidence": round(s.raw_score, 3),
        "match_type": s.match_type,
        "deprecated": s.deprecated,
    }
