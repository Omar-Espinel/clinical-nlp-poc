from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from rapidfuzz import fuzz, process as rf_process

if TYPE_CHECKING:
    from src.snomed_search.base import SNOMEDSearchStrategy

logger = logging.getLogger(__name__)

AUTOCOMPLETE_FUZZY_THRESHOLD: int = int(
    os.environ.get("AUTOCOMPLETE_FUZZY_THRESHOLD", "70")
)
AUTOCOMPLETE_SEMANTIC_THRESHOLD: float = float(
    os.environ.get("AUTOCOMPLETE_SEMANTIC_THRESHOLD", "0.65")
)

_PHASES: list[str] = [
    "Phase 1", "Phase 2", "Phase 3", "Phase 4",
    "Phase 1/2", "Phase 2/3", "Phase 1/2/3",
]


@dataclass(frozen=True)
class AutocompleteSuggestion:
    """Immutable suggestion record produced by index search methods.

    completion is not a field here because it is assembled in
    _format_suggestion by prepending context_prefix to display.
    Keeping completion out of the dataclass avoids frozen-dataclass
    mutation and keeps this record pure.
    """
    display: str
    snomed_code: str | None
    category: str
    raw_score: float
    match_type: str
    deprecated: bool = False


class AutocompleteIndex:
    """Wraps existing strategy data structures for three-tier autocomplete.

    Constructed once at startup from the pipeline's strategy instance.
    Holds references to existing data, no duplication.
    All search methods are read-only and thread-safe.
    """

    def __init__(self, strategy: "SNOMEDSearchStrategy", geo_data: dict) -> None:
        self._strategy = strategy
        self._geo_data = geo_data

        exact_index: dict = getattr(strategy, "_exact_index", {})
        synonym_index: dict = getattr(strategy, "_synonym_index", {})
        alias_dict: dict = getattr(strategy, "_alias_dict", {})

        self._exact_keys: list[str] = sorted(exact_index.keys())
        self._synonym_keys: list[str] = sorted(synonym_index.keys())
        self._alias_keys: list[str] = sorted(alias_dict.keys())
        self._exact_index = exact_index
        self._synonym_index = synonym_index
        self._alias_dict = alias_dict

        self._geo_city_keys: list[str] = sorted(geo_data.get("cities", {}).keys())
        self._geo_state_keys: list[str] = sorted(geo_data.get("states", {}).keys())
        self._geo_region_keys: list[str] = sorted(geo_data.get("regions", {}).keys())

        self._embedder = getattr(strategy, "_embedder", None)
        self._np_embeddings = getattr(strategy, "_np_embeddings", None)
        self._np_terms: list[str] = getattr(strategy, "_np_terms", [])
        self._collection = getattr(strategy, "_collection", None)
        self._semantic_available: bool = getattr(strategy, "semantic_available", False)

        logger.info(
            "AutocompleteIndex built: exact=%d synonyms=%d aliases=%d geo_cities=%d semantic=%s",
            len(self._exact_keys), len(self._synonym_keys),
            len(self._alias_keys), len(self._geo_city_keys),
            self._semantic_available,
        )

    def prefix_search(self, prefix: str, limit: int = 21) -> list[AutocompleteSuggestion]:
        """Prefix scan over all term sources.

        Each sub-source is capped at limit candidates independently
        to ensure diversity across categories. All sub-sources are
        always checked regardless of how many results earlier sources
        produced because a user typing 'bos' should see both SNOMED
        terms starting with 'bos' and Boston as a city.
        HIPAA: prefix is not logged.
        """
        prefix_lower = prefix.lower()
        results: list[AutocompleteSuggestion] = []

        count = 0
        for key in self._exact_keys:
            if count >= limit:
                break
            if key.startswith(prefix_lower):
                record = self._exact_index[key]
                results.append(AutocompleteSuggestion(
                    display=record["preferred_term"],
                    snomed_code=record["concept_id"],
                    category="snomed",
                    raw_score=0.99,
                    match_type="prefix",
                ))
                count += 1

        count = 0
        for key in self._synonym_keys:
            if count >= limit:
                break
            if key.startswith(prefix_lower):
                record = self._synonym_index[key]
                results.append(AutocompleteSuggestion(
                    display=record["preferred_term"],
                    snomed_code=record["concept_id"],
                    category="snomed",
                    raw_score=0.95,
                    match_type="prefix",
                ))
                count += 1

        count = 0
        for key in self._alias_keys:
            if count >= limit:
                break
            if key.startswith(prefix_lower):
                target = self._alias_dict[key]
                record = self._exact_index.get(target)
                if record:
                    results.append(AutocompleteSuggestion(
                        display=record["preferred_term"],
                        snomed_code=record["concept_id"],
                        category="snomed",
                        raw_score=0.93,
                        match_type="alias",
                    ))
                    count += 1

        cities = self._geo_data.get("cities", {})
        count = 0
        for key in self._geo_city_keys:
            if count >= limit:
                break
            if key.startswith(prefix_lower):
                entry = cities[key]
                results.append(AutocompleteSuggestion(
                    display=f"{entry['canonical']}, {entry['state']}",
                    snomed_code=None,
                    category="geo_city",
                    raw_score=0.97,
                    match_type="prefix",
                ))
                count += 1

        states = self._geo_data.get("states", {})
        count = 0
        for key in self._geo_state_keys:
            if count >= limit:
                break
            if len(key) < 3:
                continue
            if key.startswith(prefix_lower):
                results.append(AutocompleteSuggestion(
                    display=states[key],
                    snomed_code=None,
                    category="geo_state",
                    raw_score=0.96,
                    match_type="prefix",
                ))
                count += 1

        regions = self._geo_data.get("regions", {})
        count = 0
        for key in self._geo_region_keys:
            if count >= limit:
                break
            if key.startswith(prefix_lower):
                results.append(AutocompleteSuggestion(
                    display=key.title(),
                    snomed_code=None,
                    category="geo_region",
                    raw_score=0.94,
                    match_type="prefix",
                ))
                count += 1

        for phase in _PHASES:
            if phase.lower().startswith(prefix_lower):
                results.append(AutocompleteSuggestion(
                    display=phase,
                    snomed_code=None,
                    category="phase",
                    raw_score=1.0,
                    match_type="prefix",
                ))

        return results

    def fuzzy_search(self, prefix: str, limit: int = 21) -> list[AutocompleteSuggestion]:
        """Fuzzy search using rapidfuzz WRatio on SNOMED preferred terms.

        Pre-filters to terms sharing the first character of the prefix
        before running rapidfuzz. This reduces the 90k candidate pool
        by roughly 96 percent with no recall loss because edit distance
        up to 2 on a prefix of 3 or more characters never changes the
        first character in this domain.

        Only searches SNOMED preferred terms. Geo and phase terms are
        short enough that Tier 1 handles their typos adequately.
        HIPAA: prefix is not logged.
        """
        if not self._exact_keys:
            return []

        prefix_lower = prefix.lower()
        if prefix_lower:
            first_char = prefix_lower[0]
            candidates = [k for k in self._exact_keys if k.startswith(first_char)]
        else:
            candidates = self._exact_keys

        if not candidates:
            return []

        matches = rf_process.extract(
            prefix,
            candidates,
            scorer=fuzz.WRatio,
            score_cutoff=AUTOCOMPLETE_FUZZY_THRESHOLD,
            limit=limit,
        )

        results: list[AutocompleteSuggestion] = []
        for term, score_val, _ in matches:
            record = self._exact_index.get(term)
            if record is None:
                continue
            results.append(AutocompleteSuggestion(
                display=record["preferred_term"],
                snomed_code=record["concept_id"],
                category="snomed",
                raw_score=round(score_val / 100.0, 4),
                match_type="fuzzy",
            ))

        return results

    def semantic_search(self, prefix: str, limit: int = 21) -> list[AutocompleteSuggestion]:
        """Semantic search using the strategy's pre-built embeddings.

        Prefers get_top_neighbors duck-typed call because it is already
        battle-tested in the pipeline's EmbeddingAmbiguityGate. Falls
        back to direct numpy matmul on _np_embeddings if unavailable.

        The threshold here is 0.65, higher than the pipeline's Layer 2
        threshold of 0.42, because autocomplete suggestions must be
        confident. Borderline semantic matches confuse users more than
        they help.
        HIPAA: prefix is not logged.
        """
        if not self._semantic_available:
            return []

        if hasattr(self._strategy, "get_top_neighbors"):
            try:
                neighbors = self._strategy.get_top_neighbors(
                    prefix,
                    n=limit,
                    low_threshold=AUTOCOMPLETE_SEMANTIC_THRESHOLD,
                )
                return [
                    AutocompleteSuggestion(
                        display=m.display,
                        snomed_code=m.code,
                        category="snomed",
                        raw_score=m.confidence,
                        match_type="semantic",
                    )
                    for m in neighbors
                    if m.confidence >= AUTOCOMPLETE_SEMANTIC_THRESHOLD
                ]
            except Exception as exc:
                logger.warning(
                    "AutocompleteIndex.semantic_search: get_top_neighbors raised (%s) falling back to numpy",
                    type(exc).__name__,
                )

        if self._embedder is None or self._np_embeddings is None or not self._np_terms:
            return []

        try:
            query_vec = self._embedder.encode([prefix], show_progress_bar=False)
            norm = float(np.linalg.norm(query_vec))
            if norm == 0.0:
                return []
            query_norm = (query_vec / norm).astype("float32")
            similarities = (self._np_embeddings @ query_norm.T).ravel()
            top_indices = np.argsort(similarities)[::-1][:limit]

            results: list[AutocompleteSuggestion] = []
            for idx in top_indices:
                sim = float(similarities[idx])
                if sim < AUTOCOMPLETE_SEMANTIC_THRESHOLD:
                    break
                term = self._np_terms[int(idx)]
                record = self._exact_index.get(term)
                if record is None:
                    continue
                results.append(AutocompleteSuggestion(
                    display=record["preferred_term"],
                    snomed_code=record["concept_id"],
                    category="snomed",
                    raw_score=round(sim, 4),
                    match_type="semantic",
                ))
            return results

        except Exception as exc:
            logger.warning(
                "AutocompleteIndex.semantic_search: numpy failed (%s)",
                type(exc).__name__,
            )
            return []
