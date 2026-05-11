"""GroqProvider — the only file in this package that imports groq."""

import logging
import time
from typing import Optional

import groq

from src.exceptions import LLMProviderError

logger = logging.getLogger(__name__)


class GroqProvider:
    """LLMProvider implementation backed by the Groq API."""

    name = "groq"

    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant") -> None:
        self._client = groq.Groq(api_key=api_key)
        self._model = model

    @property
    def model_id(self) -> str:
        return self._model

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        timeout: float = 10.0,
        json_mode: bool = True,
    ) -> str:
        """Call Groq and return raw response text.

        Retries on transient errors with exponential backoff [1s, 2s, 4s].
        Raises LLMProviderError on permanent errors or after retry budget exhausted.
        """
        last_error: Optional[Exception] = None
        # attempt=0 → no backoff; attempts 1-3 → backoffs 1s, 2s, 4s
        for attempt, backoff in enumerate([0, 1, 2, 4]):
            if backoff:
                time.sleep(backoff)
            try:
                t_start = time.monotonic()
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_format={"type": "json_object"} if json_mode else None,
                    timeout=timeout,
                )
                latency = time.monotonic() - t_start
                content = response.choices[0].message.content
                logger.info(
                    "provider=%s model=%s attempt=%d response_len=%d latency=%.3fs",
                    self.name,
                    self._model,
                    attempt,
                    len(content) if content else 0,
                    latency,
                )
                return content
            except (groq.RateLimitError, groq.APITimeoutError, groq.APIConnectionError) as e:
                logger.warning(
                    "provider=%s model=%s attempt=%d transient error type=%s",
                    self.name,
                    self._model,
                    attempt,
                    type(e).__name__,
                )
                last_error = e  # transient — retry
                continue
            except (groq.AuthenticationError, groq.NotFoundError, groq.BadRequestError) as e:
                # Permanent — no retry
                logger.error(
                    "provider=%s model=%s permanent error type=%s",
                    self.name,
                    self._model,
                    type(e).__name__,
                )
                raise LLMProviderError(
                    str(e), provider_name=self.name, original_error=e
                )

        raise LLMProviderError(
            "retry budget exhausted",
            provider_name=self.name,
            original_error=last_error,
        )
