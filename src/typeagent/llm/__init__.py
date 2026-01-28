# Copyright (c) Xiaoyu Zhang.
# Licensed under the MIT License.

"""LLM abstraction layer for TypeAgent.

Supports multiple LLM providers:
- OpenAI
- Azure OpenAI
- Anthropic
- DeepSeek
- Qwen (Alibaba Cloud)

Example:
    >>> from typeagent.llm import create_llm, LLMConfig
    >>> config = LLMConfig(provider="openai", model="gpt-4o", api_key="sk-xxx")
    >>> llm = create_llm(config)
    >>> response = await llm.complete("Hello, world!")
    >>> print(response.content)
"""

from .base import LLMBase, LLMConfig, LLMResponse
from .factory import (
    create_llm,
    create_llm_from_dict,
    register_provider,
    get_supported_providers,
)
from .typechat_adapter import TypeChatLLMAdapter, create_typechat_model_from_config

__all__ = [
    # Base classes
    "LLMBase",
    "LLMConfig",
    "LLMResponse",
    # Factory functions
    "create_llm",
    "create_llm_from_dict",
    "register_provider",
    "get_supported_providers",
    # TypeChat integration
    "TypeChatLLMAdapter",
    "create_typechat_model_from_config",
]
