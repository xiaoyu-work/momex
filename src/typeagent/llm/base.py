# Copyright (c) Xiaoyu Zhang.
# Licensed under the MIT License.

"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


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
    """Abstract base class for LLM providers."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = None

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send chat messages and get response.

        Args:
            messages: List of messages with 'role' and 'content' keys.
            temperature: Override temperature (uses config default if None).
            max_tokens: Override max_tokens (uses config default if None).

        Returns:
            LLMResponse with content and raw response.
        """
        pass

    @abstractmethod
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
        pass

    def _get_temperature(self, temperature: float | None) -> float:
        """Get temperature, using override or config default."""
        return temperature if temperature is not None else self.config.temperature

    def _get_max_tokens(self, max_tokens: int | None) -> int | None:
        """Get max_tokens, using override or config default."""
        return max_tokens if max_tokens is not None else self.config.max_tokens
