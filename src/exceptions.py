"""Centralized exception types for the clinical NLP pipeline."""

from typing import Optional


class LLMProviderError(Exception):
    """Raised when the LLM provider encounters an unrecoverable failure
    (after retry budget exhausted, or for permanent errors like invalid API key)."""

    def __init__(
        self,
        message: str,
        provider_name: str,
        original_error: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.provider_name = provider_name
        self.original_error = original_error

    def __str__(self) -> str:
        # User-safe representation; never includes original_error details.
        return f"LLM provider '{self.provider_name}' unavailable"


class PipelineError(Exception):
    """Raised by the pipeline for orchestration failures
    (extraction timeout, parallel-path coordination failure)."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code  # enum: "extraction_timeout" | "strategy_unavailable" | ...


class StrategyError(Exception):
    """Raised by SNOMED strategies for init failures (missing dictionary, etc.)."""
    pass


class ExtractionError(Exception):
    """Raised when LLM extraction fails."""
    pass
