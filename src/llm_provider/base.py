"""LLMProvider Protocol — the only interface the pipeline depends on."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """Every provider implements this. Pipeline depends only on this Protocol."""

    @property
    def name(self) -> str: ...  # "groq" / "claude" / etc.

    @property
    def model_id(self) -> str: ...

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        timeout: float = 10.0,
        json_mode: bool = True,
    ) -> str: ...
    # Returns raw response text. Provider handles retries internally per the contract.
    # Raises LLMProviderError on unrecoverable failure (retry budget exhausted, or permanent).
