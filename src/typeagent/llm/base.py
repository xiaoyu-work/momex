# Copyright (c) Xiaoyu Zhang.
# Licensed under the MIT License.

"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from typing import Any

import stamina

logger = logging.getLogger(__name__)


def _transient_errors() -> tuple[type[Exception], ...]:
    """Exception types worth retrying, for whichever SDKs are installed.

    Rate limits, connection drops, timeouts and 5xx are all conditions that
    succeed on a second attempt. Everything else -- a bad key, a missing
    deployment, a malformed request -- will fail identically however many
    times it is retried, so it is left to propagate immediately.
    """
    errors: list[type[Exception]] = []
    try:
        import openai

        errors += [
            openai.RateLimitError,
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.InternalServerError,
        ]
    except ImportError:  # pragma: no cover - openai is a declared dependency
        pass
    try:  # pragma: no cover - anthropic is optional
        import anthropic  # type: ignore[import-not-found]

        errors += [
            anthropic.RateLimitError,
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
            anthropic.InternalServerError,
        ]
    except ImportError:
        pass
    return tuple(errors) or (Exception,)


# Matches what typeagent.aitools.model_adapters already applies to its own
# chat path. Without it, Momex's LLM layer had no retry at all: a single 429
# aborted the call, and because contradiction detection treats any exception
# as "nothing contradicted", a rate limit silently switched that feature off.
_RETRIER = stamina.AsyncRetryingCaller(attempts=6, timeout=120).on(_transient_errors())


@dataclass
class LLMConfig:
    """Configuration for LLM provider.

    Attributes:
        provider: Provider name (openai, azure, anthropic, deepseek, qwen).
        model: Model name (e.g., gpt-4o, claude-sonnet-4-20250514).
        api_key: API key for the provider.
        api_base: Base URL for the API (required for Azure).
        temperature: Temperature for responses.
        max_tokens: Maximum tokens for responses.
    """

    provider: str = "openai"
    model: str = ""
    api_key: str = ""
    api_base: str = ""
    temperature: float = 0.0
    max_tokens: int | None = None


@dataclass
class LLMResponse:
    """Response from LLM."""

    content: str
    raw: Any = None  # Original response object


class LLMBase(ABC):
    """Abstract base class for LLM providers.

    Subclasses implement `_chat`, the single provider-specific call. Retrying
    and the prompt-shaped convenience wrapper live here so that every provider
    gets them and none can forget.
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = None

    @abstractmethod
    async def _chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Perform one provider-specific chat call, without retrying."""

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send chat messages and get response, retrying transient failures.

        Args:
            messages: List of messages with 'role' and 'content' keys.
            temperature: Override temperature (uses config default if None).
            max_tokens: Override max_tokens (uses config default if None).

        Returns:
            LLMResponse with content and raw response.

        Raises:
            The provider's own exception, once retries are exhausted or the
            failure is not transient.
        """
        return await _RETRIER(self._chat, messages, temperature, max_tokens)

    async def complete(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send a single prompt and get response.

        Args:
            prompt: The prompt text.
            temperature: Override temperature.
            max_tokens: Override max_tokens.

        Returns:
            LLMResponse with content and raw response.
        """
        return await self.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _get_temperature(self, temperature: float | None) -> float:
        """Get temperature, using override or config default."""
        return temperature if temperature is not None else self.config.temperature

    def _get_max_tokens(self, max_tokens: int | None) -> int | None:
        """Get max_tokens, using override or config default."""
        return max_tokens if max_tokens is not None else self.config.max_tokens
