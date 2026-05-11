"""Tests for the LLM provider abstraction (Wave 1).

All tests use mocking — no actual API calls are made.
"""

import os
import pytest
from unittest.mock import MagicMock, patch, call

import groq

from src.exceptions import LLMProviderError
from src.llm_provider import get_provider
from src.llm_provider.base import LLMProvider
from src.llm_provider.groq_provider import GroqProvider


# ---------------------------------------------------------------------------
# MockLLMProvider — in-process canned response implementation
# ---------------------------------------------------------------------------

class MockLLMProvider:
    """Minimal LLMProvider that returns a fixed canned response."""

    name = "mock"
    model_id = "mock-model-v1"

    def __init__(self, response: str = '{"result": "ok"}') -> None:
        self._response = response

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        timeout: float = 10.0,
        json_mode: bool = True,
    ) -> str:
        return self._response


# ---------------------------------------------------------------------------
# Helper to build a fake Groq response object
# ---------------------------------------------------------------------------

def _make_groq_response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices[0].message.content = content
    return response


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMockLLMProvider:
    def test_returns_canned_response(self):
        provider = MockLLMProvider(response='{"answer": 42}')
        result = provider.complete(system_prompt="sys", user_prompt="user")
        assert result == '{"answer": 42}'

    def test_satisfies_protocol(self):
        provider = MockLLMProvider()
        assert isinstance(provider, LLMProvider)


class TestGroqProviderRetryOnTransient:
    """GroqProvider retries 3 times on RateLimitError, then succeeds."""

    @patch("src.llm_provider.groq_provider.time.sleep", return_value=None)
    @patch("src.llm_provider.groq_provider.groq.Groq")
    def test_retries_three_times_then_succeeds(self, mock_groq_cls, mock_sleep):
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client

        # First 3 calls raise RateLimitError; 4th succeeds
        mock_client.chat.completions.create.side_effect = [
            groq.RateLimitError("rate limit", response=MagicMock(), body={}),
            groq.RateLimitError("rate limit", response=MagicMock(), body={}),
            groq.RateLimitError("rate limit", response=MagicMock(), body={}),
            _make_groq_response('{"ok": true}'),
        ]

        provider = GroqProvider(api_key="test-key")
        result = provider.complete(system_prompt="sys", user_prompt="user")

        assert result == '{"ok": true}'
        assert mock_client.chat.completions.create.call_count == 4
        # Backoffs: attempt 0 → 0s (no sleep), 1 → 1s, 2 → 2s, 3 → 4s
        mock_sleep.assert_has_calls([call(1), call(2), call(4)])
        assert mock_sleep.call_count == 3


class TestGroqProviderExhaustedTransients:
    """LLMProviderError with original_error set after all retries exhausted."""

    @patch("src.llm_provider.groq_provider.time.sleep", return_value=None)
    @patch("src.llm_provider.groq_provider.groq.Groq")
    def test_raises_after_four_transient_failures(self, mock_groq_cls, mock_sleep):
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client

        rate_err = groq.RateLimitError("rate limit", response=MagicMock(), body={})
        mock_client.chat.completions.create.side_effect = [
            rate_err, rate_err, rate_err, rate_err
        ]

        provider = GroqProvider(api_key="test-key")
        with pytest.raises(LLMProviderError) as exc_info:
            provider.complete(system_prompt="sys", user_prompt="user")

        err = exc_info.value
        assert err.original_error is rate_err
        assert err.provider_name == "groq"
        assert mock_client.chat.completions.create.call_count == 4


class TestGroqProviderPermanentError:
    """LLMProviderError raised immediately on AuthenticationError (no retry)."""

    @patch("src.llm_provider.groq_provider.time.sleep", return_value=None)
    @patch("src.llm_provider.groq_provider.groq.Groq")
    def test_raises_immediately_on_auth_error(self, mock_groq_cls, mock_sleep):
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client

        auth_err = groq.AuthenticationError("invalid key", response=MagicMock(), body={})
        mock_client.chat.completions.create.side_effect = auth_err

        provider = GroqProvider(api_key="bad-key")
        with pytest.raises(LLMProviderError) as exc_info:
            provider.complete(system_prompt="sys", user_prompt="user")

        err = exc_info.value
        assert err.original_error is auth_err
        assert err.provider_name == "groq"
        # Only one call — no retry
        assert mock_client.chat.completions.create.call_count == 1
        mock_sleep.assert_not_called()


class TestGetProvider:
    """Registry and factory behaviour."""

    def test_raises_value_error_for_unknown_provider(self):
        with pytest.raises(ValueError) as exc_info:
            get_provider("nonexistent")
        msg = str(exc_info.value)
        assert "nonexistent" in msg
        assert "groq" in msg  # available names listed

    def test_reads_env_var_when_name_is_none(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        with patch("src.llm_provider.groq_provider.groq.Groq"):
            provider = get_provider(None, api_key="test-key")
        assert provider.name == "groq"

    def test_falls_back_to_groq_when_no_env_var(self, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        with patch("src.llm_provider.groq_provider.groq.Groq"):
            provider = get_provider(None, api_key="test-key")
        assert provider.name == "groq"
