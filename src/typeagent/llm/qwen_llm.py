# Copyright (c) Xiaoyu Zhang.
# Licensed under the MIT License.

"""Qwen LLM implementation (Alibaba Cloud DashScope, OpenAI-compatible)."""

from .openai_llm import OpenAILLM
from .base import LLMConfig


class QwenLLM(OpenAILLM):
    """Qwen LLM provider (Alibaba Cloud DashScope, OpenAI-compatible API)."""

    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def _init_client(self):
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.api_base or self.BASE_URL,
        )
