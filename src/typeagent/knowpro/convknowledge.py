# Copyright (c) Xiaoyu Zhang.
# Licensed under the MIT License.

import asyncio
from dataclasses import dataclass, field
import os
from typing import TYPE_CHECKING

import typechat

from . import kplib
from ..aitools import auth

if TYPE_CHECKING:
    from ..llm import LLMConfig

# TODO: Move ModelWrapper and create_typechat_model() to aitools package.


# TODO: Make these parameters that can be configured (e.g. from command line).
DEFAULT_MAX_RETRY_ATTEMPTS = 0
DEFAULT_TIMEOUT_SECONDS = 25

# Global LLM config - can be set by MomexConfig
_global_llm_config: "LLMConfig | None" = None


def set_llm_config(config: "LLMConfig") -> None:
    """Set the global LLM config for TypeAgent.

    This should be called before creating any KnowledgeExtractor.
    Typically called by Momex when initializing Memory.

    Args:
        config: LLM configuration from MomexConfig.
    """
    global _global_llm_config
    _global_llm_config = config


def get_llm_config() -> "LLMConfig | None":
    """Get the global LLM config."""
    return _global_llm_config


class ModelWrapper(typechat.TypeChatLanguageModel):
    def __init__(
        self,
        base_model: typechat.TypeChatLanguageModel,
        token_provider: auth.AzureTokenProvider,
    ):
        self.base_model = base_model
        self.token_provider = token_provider

    async def complete(
        self, prompt: str | list[typechat.PromptSection]
    ) -> typechat.Result[str]:
        if self.token_provider.needs_refresh():
            loop = asyncio.get_running_loop()
            api_key = await loop.run_in_executor(
                None, self.token_provider.refresh_token
            )
            env: dict[str, str | None] = dict(os.environ)
            key_name = "AZURE_OPENAI_API_KEY"
            env[key_name] = api_key
            self.base_model = typechat.create_language_model(env)
            self.base_model.timeout_seconds = DEFAULT_TIMEOUT_SECONDS
        return await self.base_model.complete(prompt)


def create_typechat_model(config: "LLMConfig | None" = None) -> typechat.TypeChatLanguageModel:
    """Create a TypeChat language model.

    Args:
        config: Optional LLM config. If provided, uses our LLM abstraction.
                If None, checks global config, then falls back to env vars.

    Returns:
        TypeChatLanguageModel instance.
    """
    # Use provided config, or global config, or fall back to env vars
    config = config or _global_llm_config

    if config is not None:
        # Use our LLM abstraction with TypeChat adapter
        from ..llm import create_typechat_model_from_config
        return create_typechat_model_from_config(config)

    # Legacy: fall back to environment variables
    env: dict[str, str | None] = dict(os.environ)
    key_name = "AZURE_OPENAI_API_KEY"
    key = env.get(key_name)
    shared_token_provider: auth.AzureTokenProvider | None = None
    if key is not None and key.lower() == "identity":
        shared_token_provider = auth.get_shared_token_provider()
        env[key_name] = shared_token_provider.get_token()
    model = typechat.create_language_model(env)
    model.timeout_seconds = DEFAULT_TIMEOUT_SECONDS
    model.max_retry_attempts = DEFAULT_MAX_RETRY_ATTEMPTS
    if shared_token_provider is not None:
        model = ModelWrapper(model, shared_token_provider)
    return model


@dataclass
class KnowledgeExtractor:
    model: typechat.TypeChatLanguageModel = field(default_factory=create_typechat_model)
    max_chars_per_chunk: int = 2048
    merge_action_knowledge: bool = (
        False  # TODO: Implement merge_action_knowledge_into_response
    )
    # Not in the signature:
    translator: typechat.TypeChatJsonTranslator[kplib.KnowledgeResponse] = field(
        init=False
    )

    def __post_init__(self):
        self.translator = self.create_translator(self.model)

    # TODO: Use max_chars_per_chunk and merge_action_knowledge.

    async def extract(self, message: str) -> typechat.Result[kplib.KnowledgeResponse]:
        result = await self.translator.translate(message)
        if isinstance(result, typechat.Success):
            if self.merge_action_knowledge:
                self.merge_action_knowledge_into_response(result.value)
        else:
            result.message += f" -- MESSAGE={message!r}"
        return result

    def create_translator(
        self, model: typechat.TypeChatLanguageModel
    ) -> typechat.TypeChatJsonTranslator[kplib.KnowledgeResponse]:
        schema = kplib.KnowledgeResponse
        type_name = "KnowledgeResponse"
        validator = typechat.TypeChatValidator[kplib.KnowledgeResponse](schema)
        translator = typechat.TypeChatJsonTranslator[kplib.KnowledgeResponse](
            model, validator, kplib.KnowledgeResponse
        )
        schema_text = translator.schema_str.rstrip()

        def create_request_prompt(intent: str) -> str:
            return (
                f"You are a service that translates user messages in a conversation "
                + f'into JSON objects of type "{type_name}" '
                + f"according to the following TypeScript definitions:\n"
                + f"```\n"
                + f"{schema_text}\n"
                + f"```\n"
                + f"The following are messages in a conversation:\n"
                + f'"""\n'
                + f"{intent}\n"
                + f'"""\n'
                + f"The following is the user request translated into a JSON object "
                + f"with 2 spaces of indentation and no properties with the value undefined:\n"
            )

        translator._create_request_prompt = create_request_prompt
        return translator

    def merge_action_knowledge_into_response(
        self, knowledge: kplib.KnowledgeResponse
    ) -> None:
        """Merge action knowledge into a single knowledge object."""
        raise NotImplementedError("TODO")
