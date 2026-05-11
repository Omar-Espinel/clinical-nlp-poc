"""LLM provider abstraction package.

GroqProvider is intentionally NOT imported eagerly: doing so would pull
the groq SDK into any module that imports from this package, defeating
the provider-isolation discipline. Import GroqProvider directly from
src.llm_provider.groq_provider when needed.
"""

from src.llm_provider.base import LLMProvider
from src.llm_provider.registry import get_provider

__all__ = ["LLMProvider", "get_provider"]
