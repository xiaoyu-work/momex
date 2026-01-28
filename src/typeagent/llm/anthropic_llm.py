# Copyright (c) Xiaoyu Zhang.
# Licensed under the MIT License.

"""Anthropic LLM implementation."""

from .base import LLMBase, LLMConfig, LLMResponse


class AnthropicLLM(LLMBase):
    """Anthropic LLM provider."""

    DEFAULT_MAX_TOKENS = 4096

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._init_client()

    def _init_client(self):
        from anthropic import AsyncAnthropic
        self._client = AsyncAnthropic(api_key=self.config.api_key)

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        # Anthropic requires max_tokens
        mt = self._get_max_tokens(max_tokens) or self.DEFAULT_MAX_TOKENS

        response = await self._client.messages.create(
            model=self.config.model,
            messages=messages,
            temperature=self._get_temperature(temperature),
            max_tokens=mt,
        )
        return LLMResponse(
            content=response.content[0].text if response.content else "",
            raw=response,
        )

    async def complete(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        return await self.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
