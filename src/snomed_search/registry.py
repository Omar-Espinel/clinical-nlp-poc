"""Strategy registry for SNOMED search backends."""

import logging
import os
from pathlib import Path
from typing import Optional

from src.exceptions import StrategyError
from src.snomed_search.base import SNOMEDSearchStrategy
from src.snomed_search.hybrid_cascade import HybridCascadeStrategy
from src.snomed_search.ngram_lookup import NGramLookupStrategy

logger = logging.getLogger(__name__)

STRATEGY_REGISTRY: dict[str, type] = {
    "hybrid_cascade": HybridCascadeStrategy,
    "ngram_lookup": NGramLookupStrategy,
}

try:
    from src.snomed_search.aho_corasick import AhoCorasickStrategy
    STRATEGY_REGISTRY["aho_corasick"] = AhoCorasickStrategy
except ImportError:
    pass  # pyahocorasick unavailable; aho_corasick won't be selectable

try:
    from src.snomed_search.pgvector_cascade import PgVectorCascadeStrategy
    STRATEGY_REGISTRY["pgvector_cascade"] = PgVectorCascadeStrategy
except ImportError:
    pass  # psycopg2 and pgvector unavailable; pgvector_cascade won't be selectable

DEFAULT_STRATEGY = "pgvector_cascade"

DEFAULT_DICT_PATH = str(
    Path(__file__).parent.parent.parent / "data" / "snomed_clinical_trials.csv"
)


def get_strategy(
    name: Optional[str] = None,
    dictionary_path: Optional[str] = None,
    **kwargs,
) -> SNOMEDSearchStrategy:
    """Instantiate and return a ready SNOMED search strategy.

    Reads SNOMED_SEARCH_STRATEGY env var when name is None.
    Raises ValueError for unknown strategy names.
    Raises StrategyError if health_check() reports not ready after init.

    Fallback behaviour (implicit default only):
    - If the resolved default strategy fails to init/health-check and it is NOT
      an explicit user request, logs a WARNING and falls back to hybrid_cascade.
    - If the caller passed name= or set SNOMED_SEARCH_STRATEGY, the exception is
      re-raised unchanged (silent swap would be a footgun).
    """
    explicit = name is not None or "SNOMED_SEARCH_STRATEGY" in os.environ
    resolved_name = name or os.environ.get("SNOMED_SEARCH_STRATEGY", DEFAULT_STRATEGY)

    # Strategy not in registry (e.g., optional dependency like psycopg2 missing
    # caused the import to be skipped in try/except above).  On explicit user
    # request, raise — silent swap would hide a misconfiguration.  On implicit
    # default, fall back to hybrid_cascade so the service still boots.
    if resolved_name not in STRATEGY_REGISTRY:
        if explicit or resolved_name == "hybrid_cascade":
            raise ValueError(
                f"Unknown strategy: {resolved_name!r}. "
                f"Available: {sorted(STRATEGY_REGISTRY)}"
            )
        logger.warning(
            "Default strategy %s not registered (likely missing dependency) — "
            "falling back to hybrid_cascade (99 terms)",
            resolved_name,
        )
        resolved_name = "hybrid_cascade"

    path = dictionary_path or DEFAULT_DICT_PATH

    try:
        strategy = STRATEGY_REGISTRY[resolved_name](dictionary_path=path, **kwargs)
        health = strategy.health_check()
        if not health.get("ready"):
            raise StrategyError(
                f"Strategy {resolved_name!r} health_check failed: {health}"
            )
        concept_size = (
            health.get("concept_count")
            if health.get("concept_count") is not None
            else health.get("dictionary_size", "unknown")
        )
        logger.info(
            "Strategy initialized: name=%s concept_count=%s",
            resolved_name,
            concept_size,
        )
        return strategy

    except Exception:
        if explicit:
            raise
        if resolved_name == "hybrid_cascade":
            raise
        # Implicit default failed — attempt hybrid_cascade fallback
        import sys
        exc_type = sys.exc_info()[0]
        logger.warning(
            "Default strategy %s failed to init (error_type=%s) — "
            "falling back to hybrid_cascade (99 terms)",
            resolved_name,
            exc_type.__name__ if exc_type is not None else "Unknown",
        )
        fallback = STRATEGY_REGISTRY["hybrid_cascade"](dictionary_path=path, **kwargs)
        fallback_health = fallback.health_check()
        if not fallback_health.get("ready"):
            raise StrategyError(
                f"Fallback strategy 'hybrid_cascade' health_check failed: {fallback_health}"
            )
        concept_size = (
            fallback_health.get("concept_count")
            if fallback_health.get("concept_count") is not None
            else fallback_health.get("dictionary_size", "unknown")
        )
        logger.info(
            "Strategy initialized: name=%s concept_count=%s",
            "hybrid_cascade",
            concept_size,
        )
        return fallback
