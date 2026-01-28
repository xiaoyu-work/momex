# Copyright (c) Xiaoyu Zhang.
# Licensed under the MIT License.

"""TypeChat adapter for LLM abstraction."""

import typechat

from .base import LLMBase, LLMConfig
from .factory import create_llm


class TypeChatLLMAdapter(typechat.TypeChatLanguageModel):
    """Adapter to use LLMBase with TypeChat.

    This allows TypeChat translators to use any LLM provider
    supported by our LLM abstraction.
    """

    def __init__(self, llm: LLMBase):
        self.llm = llm
        # TypeChat settings
        self.timeout_seconds = 25
        self.max_retry_attempts = 0

    async def complete(
        self, prompt: str | list[typechat.PromptSection]
    ) -> typechat.Result[str]:
        """Complete a prompt using the underlying LLM.

        Args:
            prompt: Either a string or list of PromptSection objects.

        Returns:
            typechat.Result with success/failure.
        """
        try:
            # Convert PromptSection list to string if needed
            if isinstance(prompt, list):
                prompt_text = "\n".join(
                    f"{section.get('role', 'user')}: {section.get('content', '')}"
                    if isinstance(section, dict)
                    else str(section)
                    for section in prompt
                )
            else:
                prompt_text = prompt

            response = await self.llm.complete(prompt_text)
            return typechat.Success(response.content)

        except Exception as e:
            return typechat.Failure(str(e))


def create_typechat_model_from_config(config: LLMConfig) -> typechat.TypeChatLanguageModel:
    """Create a TypeChat language model from LLMConfig.

    Args:
        config: LLM configuration.

    Returns:
        TypeChatLanguageModel that uses the configured LLM provider.
    """
    llm = create_llm(config)
    return TypeChatLLMAdapter(llm)
