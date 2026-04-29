# Copyright (c) Xiaoyu Zhang.
# Licensed under the MIT License.

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import typechat

from . import knowledge_schema as kplib
from ..aitools.model_adapters import create_chat_model

if TYPE_CHECKING:
    from ..llm import LLMConfig


_global_llm_config: "LLMConfig | None" = None


def set_llm_config(config: "LLMConfig") -> None:
    """Set the global LLM config for TypeAgent."""
    global _global_llm_config
    _global_llm_config = config


def get_llm_config() -> "LLMConfig | None":
    """Get the global LLM config."""
    return _global_llm_config


def create_typechat_model(
    config: "LLMConfig | None" = None,
) -> typechat.TypeChatLanguageModel:
    """Create a TypeChat language model.

    Momex can provide an explicit LLM config; otherwise use the upstream
    pydantic-ai model adapter configured from environment variables.
    """
    config = config or _global_llm_config
    if config is not None:
        from ..llm import create_typechat_model_from_config

        return create_typechat_model_from_config(config)

    return create_chat_model()


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
