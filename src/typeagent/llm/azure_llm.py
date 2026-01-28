# Copyright (c) Xiaoyu Zhang.
# Licensed under the MIT License.

"""Azure OpenAI LLM implementation."""

from .base import LLMBase, LLMConfig, LLMResponse


class AzureLLM(LLMBase):
    """Azure OpenAI LLM provider."""

    API_VERSION = "2024-02-15-preview"

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._init_client()

    def _init_client(self):
        from openai import AsyncAzureOpenAI
        self._client = AsyncAzureOpenAI(
            api_key=self.config.api_key,
            azure_endpoint=self.config.api_base,
            api_version=self.API_VERSION,
        )

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        kwargs = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self._get_temperature(temperature),
        }
        if (mt := self._get_max_tokens(max_tokens)) is not None:
            kwargs["max_tokens"] = mt

        response = await self._client.chat.completions.create(**kwargs)
        return LLMResponse(
            content=response.choices[0].message.content or "",
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
