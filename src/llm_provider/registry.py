"""Provider factory and registry."""

import os
from typing import Optional

from src.llm_provider.base import LLMProvider
from src.llm_provider.groq_provider import GroqProvider

PROVIDER_REGISTRY: dict[str, type] = {
    "groq": GroqProvider,
}

DEFAULT_PROVIDER = "groq"


def get_provider(name: Optional[str] = None, **kwargs) -> LLMProvider:
    """Instantiate and return a provider by name.

    Falls back to LLM_PROVIDER env var, then DEFAULT_PROVIDER.
    Raises ValueError for unknown provider names.
    """
    resolved = name or os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER)
    if resolved not in PROVIDER_REGISTRY:
        raise ValueError(
            f"Unknown provider: {resolved!r}. Available: {sorted(PROVIDER_REGISTRY)}"
        )
    return PROVIDER_REGISTRY[resolved](**kwargs)
