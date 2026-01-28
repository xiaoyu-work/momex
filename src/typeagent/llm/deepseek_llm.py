# Copyright (c) Xiaoyu Zhang.
# Licensed under the MIT License.

"""DeepSeek LLM implementation (OpenAI-compatible)."""

from .openai_llm import OpenAILLM
from .base import LLMConfig


class DeepSeekLLM(OpenAILLM):
    """DeepSeek LLM provider (OpenAI-compatible API)."""

    BASE_URL = "https://api.deepseek.com"

    def _init_client(self):
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.api_base or self.BASE_URL,
        )
