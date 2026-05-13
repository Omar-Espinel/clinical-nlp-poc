"""Base Protocol for filter normalizers."""

from typing import Any, Optional, Protocol


class NormalizedFilter:
    """Generic placeholder for normalizer output.

    GeoResult is the concrete type for geographic normalization.
    Future normalizers can subclass or implement FilterNormalizer directly.
    """
    pass


class FilterNormalizer(Protocol):
    """Protocol for filter normalizers.

    Each normalizer takes a raw value and produces a NormalizedFilter.
    """

    @property
    def name(self) -> str:
        """Human-readable name of the normalizer."""
        ...

    def normalize(self, raw_value: Any) -> NormalizedFilter:
        """Normalize a raw value to canonical form.

        Args:
            raw_value: The raw input to normalize.

        Returns:
            A NormalizedFilter subclass instance.
        """
        ...


class MetricFilterNormalizerProtocol(Protocol):
    """Protocol for metric filter normalizers."""

    @staticmethod
    def normalize_operator(text: str) -> str: ...

    @staticmethod
    def normalize_value(
        raw: Optional[str], data_type: str, operator: str
    ) -> Optional[float | str]: ...
