from __future__ import annotations

import hashlib
from typing import Optional

from src.autocomplete.index import AutocompleteSuggestion

_MAX_CACHE_SIZE: int = 10_000
_store: dict[str, list[dict]] = {}


def _make_key(prefix: str, limit: int) -> str:
    """Deterministic 16-char cache key from normalized prefix and limit.

    SHA-256 truncated to 16 hex chars gives negligible collision
    probability for a 10k entry cache. The raw prefix is hashed and
    never stored, satisfying HIPAA requirements.
    """
    normalized = prefix.lower().strip()
    return hashlib.sha256(f"{normalized}:{limit}".encode()).hexdigest()[:16]


def cache_get(prefix: str, limit: int) -> Optional[list[AutocompleteSuggestion]]:
    raw = _store.get(_make_key(prefix, limit))
    if raw is None:
        return None
    return [AutocompleteSuggestion(**item) for item in raw]


def cache_store(prefix: str, limit: int, suggestions: list[AutocompleteSuggestion]) -> None:
    """Store suggestions with FIFO eviction when at capacity.

    Evicts the oldest 10 percent of entries when the store reaches
    maximum size. FIFO is an acceptable approximation of LRU for
    medical term autocomplete where popular terms repeat heavily
    and true LRU adds unnecessary complexity for MVP scale.
    """
    if len(_store) >= _MAX_CACHE_SIZE:
        evict_count = _MAX_CACHE_SIZE // 10
        for key in list(_store.keys())[:evict_count]:
            _store.pop(key, None)
    _store[_make_key(prefix, limit)] = [
        {
            "display":    s.display,
            "snomed_code": s.snomed_code,
            "category":   s.category,
            "raw_score":  s.raw_score,
            "match_type": s.match_type,
            "deprecated": s.deprecated,
        }
        for s in suggestions
    ]


def cache_clear() -> None:
    _store.clear()
