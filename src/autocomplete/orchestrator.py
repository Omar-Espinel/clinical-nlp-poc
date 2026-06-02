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
        self._segmenter = QuerySegmenter(geo_keys)

        logger.info(
            "AutocompleteOrchestrator initialized: geo_keys=%d semantic=%s",
            len(geo_keys),
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

        context_prefix = " ".join(segment.context_tokens)

        loop = asyncio.get_running_loop()
        tier_used_parts: list[str] = []
        all_suggestions: list[AutocompleteSuggestion] = []

        tier1_fut = loop.run_in_executor(None, self._index.prefix_search, segment.active_prefix, limit * 3)
        tier2_fut = loop.run_in_executor(None, self._index.fuzzy_search, segment.active_prefix, limit * 3)
        tier3_fut = loop.run_in_executor(None, self._index.semantic_search, segment.active_prefix, limit * 3)

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

        # HIPAA: only prefix_length, tier_used, suggestion_count, latency_ms — never query text.
        logger.info(
            "AutocompleteOrchestrator.run: prefix_length=%d tier_used=%s suggestion_count=%d latency_ms=%d",
            len(stripped), tier_used, len(ranked), elapsed_ms,
        )

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
    completion = f"{context_prefix} {display_titled}".strip() if context_prefix else display_titled
    return {
        "display":    display_titled,
        "completion": completion,
        "snomed_code": s.snomed_code,
        "category":   s.category,
        "confidence": round(s.raw_score, 3),
        "match_type": s.match_type,
        "deprecated": s.deprecated,
    }
