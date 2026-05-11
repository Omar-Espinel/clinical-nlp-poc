from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class SNOMEDMatch:
    """A single SNOMED CT concept match returned by a search strategy.

    Spans are character offsets into the query string passed to search().
    NegationAnnotator requires valid span values; strategies may not omit them.
    """

    code: str
    display: str
    match_type: str          # "exact" | "synonym" | "fuzzy" | "semantic" | "ngram"
    confidence: float
    original_text: str
    span: tuple[int, int]    # (start_char, end_char) — REQUIRED
    negated: bool = False    # set by NegationAnnotator; default False

    def __post_init__(self) -> None:
        if self.span is None:
            raise ValueError("SNOMEDMatch.span is required and cannot be None.")
        if self.span[0] < 0:
            raise ValueError(
                f"SNOMEDMatch.span start must be >= 0; got {self.span[0]}."
            )
        if self.span[1] <= self.span[0]:
            raise ValueError(
                f"SNOMEDMatch.span end must be > start; got span={self.span}."
            )


@runtime_checkable
class SNOMEDSearchStrategy(Protocol):
    """Every SNOMED search algorithm implements this Protocol.
    Implementations are drop-in replaceable; the pipeline depends ONLY on this Protocol.

    Contract (must hold for every implementation):
    - Read-only after __init__. No state mutation in search() or anywhere else.
      Thread-safe by construction; multiple concurrent search() calls are valid.
    - Deterministic given fixed dictionary + same query. No randomness.
    - Must populate SNOMEDMatch.span with valid char offsets into the query.
    - Returns ALL candidates from the underlying algorithm; does NOT pre-filter
      by MIN_CONFIDENCE (caller filters in assembler).
    - Resource initialization happens in __init__; no lazy loading. No shared mutable
      state across instances. External resources (DB connections, network sockets)
      should use connection pools owned by the strategy.
    - search() must be free of catastrophic backtracking. No regex with unbounded
      quantifiers on user input. Strategies that use regex MUST anchor and bound them.
    - Implementations document their candidate-extraction approach in search()'s
      docstring.
    """

    @property
    def name(self) -> str: ...

    def __init__(self, dictionary_path: str, **kwargs) -> None: ...

    def search(self, query: str) -> list[SNOMEDMatch]: ...

    def health_check(self) -> dict: ...
