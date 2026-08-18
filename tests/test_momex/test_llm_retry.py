"""Tests that the LLM layer retries transient failures.

Momex's own LLM abstraction had no retry at all, while the TypeAgent chat path
it sits beside has used stamina since it was written. That mattered more than
it looks: contradiction detection treats any exception as "nothing
contradicted", so a single 429 silently switched the feature off rather than
failing loudly or waiting a moment and succeeding.
"""

import httpx
import pytest
import stamina

import openai

from typeagent.llm.base import LLMBase, LLMConfig, LLMResponse


@pytest.fixture
def retrying(request):
    """Let retries actually happen, without waiting for backoff.

    tests/conftest.py sets stamina's testing mode globally, which collapses
    every retry loop to a single attempt so the suite does not sit in
    exponential backoff. Retry behaviour is therefore invisible unless a test
    asks for it; `attempts` keeps the loop instant while letting it run.
    """
    attempts = getattr(request, "param", 3)
    stamina.set_testing(True, attempts=attempts)
    try:
        yield attempts
    finally:
        stamina.set_testing(True)


def _rate_limit() -> openai.RateLimitError:
    """A RateLimitError shaped enough like the real thing for stamina."""
    response = httpx.Response(
        429, request=httpx.Request("POST", "https://example.invalid/chat")
    )
    return openai.RateLimitError("rate limited", response=response, body=None)


class _Recorder(LLMBase):
    """Fails a fixed number of times, then succeeds."""

    def __init__(self, failures: int, error: BaseException | None = None):
        super().__init__(LLMConfig(provider="openai", model="m", api_key="k"))
        self.remaining = failures
        self.error = error or _rate_limit()
        self.calls = 0
        self.seen: list[list[dict[str, str]]] = []

    async def _chat(self, messages, temperature=None, max_tokens=None):
        self.calls += 1
        self.seen.append(messages)
        if self.remaining:
            self.remaining -= 1
            raise self.error
        return LLMResponse(content="ok", raw=None)


@pytest.mark.asyncio
async def test_succeeds_after_a_rate_limit(retrying):
    llm = _Recorder(failures=2)

    result = await llm.complete("hello")

    assert result.content == "ok"
    assert llm.calls == 3


@pytest.mark.asyncio
async def test_gives_up_eventually_rather_than_hanging(retrying):
    llm = _Recorder(failures=99)

    with pytest.raises(openai.RateLimitError):
        await llm.complete("hello")

    assert llm.calls == retrying


@pytest.mark.asyncio
async def test_permanent_errors_are_not_retried(retrying):
    """A bad key fails identically however many times it is sent."""
    llm = _Recorder(failures=99, error=ValueError("invalid api key"))

    with pytest.raises(ValueError):
        await llm.complete("hello")

    assert llm.calls == 1


@pytest.mark.asyncio
async def test_complete_is_shaped_as_a_single_user_message():
    """Every provider used to reimplement this identically."""
    llm = _Recorder(failures=0)

    await llm.complete("hello")

    assert llm.seen == [[{"role": "user", "content": "hello"}]]


@pytest.mark.asyncio
async def test_chat_passes_messages_through():
    llm = _Recorder(failures=0)
    messages = [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hi"},
    ]

    await llm.chat(messages)

    assert llm.seen == [messages]


def test_every_provider_implements_the_retried_hook():
    """A provider that overrode chat() directly would skip retrying."""
    from typeagent.llm.anthropic_llm import AnthropicLLM
    from typeagent.llm.azure_llm import AzureLLM
    from typeagent.llm.deepseek_llm import DeepSeekLLM
    from typeagent.llm.openai_llm import OpenAILLM
    from typeagent.llm.qwen_llm import QwenLLM

    for provider in (AnthropicLLM, AzureLLM, OpenAILLM, DeepSeekLLM, QwenLLM):
        assert "_chat" in dir(provider)
        assert "chat" not in vars(
            provider
        ), f"{provider.__name__} overrides chat(), bypassing the retry wrapper"
        assert "complete" not in vars(
            provider
        ), f"{provider.__name__} reimplements complete()"
