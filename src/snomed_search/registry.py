"""Strategy registry for SNOMED search backends."""

import os
from pathlib import Path
from typing import Optional

from src.exceptions import StrategyError
from src.snomed_search.base import SNOMEDSearchStrategy
from src.snomed_search.hybrid_cascade import HybridCascadeStrategy
from src.snomed_search.ngram_lookup import NGramLookupStrategy

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

DEFAULT_STRATEGY = "hybrid_cascade"

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
    """
    resolved_name = name or os.environ.get("SNOMED_SEARCH_STRATEGY", DEFAULT_STRATEGY)
    if resolved_name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy: {resolved_name!r}. "
            f"Available: {sorted(STRATEGY_REGISTRY)}"
        )
    path = dictionary_path or DEFAULT_DICT_PATH
    strategy = STRATEGY_REGISTRY[resolved_name](dictionary_path=path, **kwargs)
    health = strategy.health_check()
    if not health.get("ready"):
        raise StrategyError(
            f"Strategy {resolved_name!r} health_check failed: {health}"
        )
    return strategy
