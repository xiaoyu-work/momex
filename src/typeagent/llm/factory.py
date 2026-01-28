# Copyright (c) Xiaoyu Zhang.
# Licensed under the MIT License.

"""Factory for creating LLM instances."""

from .base import LLMBase, LLMConfig


# Provider name to class mapping
_PROVIDER_CLASSES: dict[str, type[LLMBase]] = {}


def _load_providers():
    """Lazy load provider classes to avoid import errors if dependencies missing."""
    global _PROVIDER_CLASSES
    if _PROVIDER_CLASSES:
        return

    from .openai_llm import OpenAILLM
    from .azure_llm import AzureLLM
    from .anthropic_llm import AnthropicLLM
    from .deepseek_llm import DeepSeekLLM
    from .qwen_llm import QwenLLM

    _PROVIDER_CLASSES = {
        "openai": OpenAILLM,
        "azure": AzureLLM,
        "anthropic": AnthropicLLM,
        "deepseek": DeepSeekLLM,
        "qwen": QwenLLM,
    }


def create_llm(config: LLMConfig) -> LLMBase:
    """Create LLM instance based on config.

    Args:
        config: LLM configuration with provider, model, api_key, etc.

    Returns:
        LLMBase instance for the specified provider.

    Raises:
        ValueError: If provider is not supported.
    """
    _load_providers()

    provider = config.provider.lower()
    if provider not in _PROVIDER_CLASSES:
        supported = ", ".join(_PROVIDER_CLASSES.keys())
        raise ValueError(f"Unsupported provider: {provider}. Supported: {supported}")

    return _PROVIDER_CLASSES[provider](config)


def create_llm_from_dict(config_dict: dict) -> LLMBase:
    """Create LLM instance from a dictionary config.

    Args:
        config_dict: Dictionary with provider, model, api_key, etc.

    Returns:
        LLMBase instance.
    """
    config = LLMConfig(**config_dict)
    return create_llm(config)


def register_provider(name: str, cls: type[LLMBase]):
    """Register a custom LLM provider.

    Args:
        name: Provider name (e.g., "custom").
        cls: LLMBase subclass.
    """
    _load_providers()
    _PROVIDER_CLASSES[name.lower()] = cls


def get_supported_providers() -> list[str]:
    """Get list of supported provider names."""
    _load_providers()
    return list(_PROVIDER_CLASSES.keys())
