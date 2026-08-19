"""Tests for query-translation caching and the time context it depends on.

Two defects sat together here. The compiled query depends on the conversation
-- its time range is put into the prompt -- but the cache was keyed on the
query text alone, in a process-global dict. For a library whose whole shape is
one collection per tenant, that meant one tenant's compiled query answering
another tenant's question.

And the time-range prompt told the model to use the range "ONLY IF the user
request explicitly asks for time ranges", so a relative expression like "last
year" was resolved against the present instead. On a 2023 conversation asked
in 2026 that produced a 2025 filter, which matched nothing.
"""

from typing import Any

import pytest

import typechat

from typeagent.knowpro import searchlang
from typeagent.knowpro.convutils import get_time_range_prompt_section_for_conversation


class _Messages:
    def __init__(self, timestamps: list[str]):
        self._timestamps = timestamps

    async def size(self) -> int:
        return len(self._timestamps)

    async def get_item(self, ordinal: int):
        class _M:
            timestamp = self._timestamps[ordinal]

        return _M()

    async def get_slice(self, start: int, stop: int):
        return [
            await self.get_item(i)
            for i in range(start, min(stop, len(self._timestamps)))
        ]


class _Conversation:
    def __init__(self, name_tag: str, timestamps: list[str] | None = None):
        self.name_tag = name_tag
        self.tags = [name_tag]
        self.messages = _Messages(timestamps or [])


class _Translator:
    """Records every translation and returns a distinguishable result."""

    def __init__(self):
        self.calls: list[tuple[str, list]] = []

    async def translate(self, query_text: str, prompt_preamble: Any = None):
        self.calls.append((query_text, list(prompt_preamble or [])))
        return typechat.Success(f"compiled:{query_text}:{len(self.calls)}")


# The fakes implement only the slice of IConversation this code path touches,
# so they are passed through helpers typed loosely rather than sprinkling
# ignores over every call.
async def translate(conversation: Any, translator: Any, query: str) -> Any:
    result = await searchlang.search_query_from_language(
        conversation, translator, query
    )
    assert isinstance(result, typechat.Success)
    return result


async def time_section(conversation: Any):
    return await get_time_range_prompt_section_for_conversation(conversation)


@pytest.fixture(autouse=True)
def clear_cache():
    searchlang._query_translation_cache.clear()
    yield
    searchlang._query_translation_cache.clear()


@pytest.mark.asyncio
async def test_same_question_in_two_collections_is_translated_twice():
    """The regression: one tenant used to answer with another's compiled query."""
    translator = _Translator()

    alice = await translate(
        _Conversation("user:alice"), translator, "what did I do last year?"
    )
    bob = await translate(
        _Conversation("user:bob"), translator, "what did I do last year?"
    )

    assert len(translator.calls) == 2
    assert alice.value != bob.value


@pytest.mark.asyncio
async def test_the_same_collection_still_hits_the_cache():
    """The cache has to keep working, or every repeat query costs a call."""
    translator = _Translator()
    conversation = _Conversation("user:alice")

    first = await translate(conversation, translator, "what did I do?")
    second = await translate(conversation, translator, "what did I do?")

    assert len(translator.calls) == 1
    assert first.value == second.value


@pytest.mark.asyncio
async def test_a_changed_time_range_invalidates_the_entry():
    """Ingesting more messages moves the range, so the old query is stale."""
    translator = _Translator()

    await translate(
        _Conversation("user:alice", ["2023-01-01T00:00:00"]),
        translator,
        "what happened recently?",
    )
    await translate(
        _Conversation("user:alice", ["2023-01-01T00:00:00", "2024-06-01T00:00:00"]),
        translator,
        "what happened recently?",
    )

    assert len(translator.calls) == 2


@pytest.mark.asyncio
async def test_the_conversation_time_range_reaches_the_prompt():
    translator = _Translator()
    conversation = _Conversation(
        "user:alice", ["2023-05-01T00:00:00", "2023-11-01T00:00:00"]
    )

    await translate(conversation, translator, "what happened last year?")

    _, preamble = translator.calls[0]
    content = " ".join(section["content"] for section in preamble)
    assert "2023-05-01" in content


class TestTimeRangePrompt:
    @pytest.mark.asyncio
    async def test_reports_the_conversation_range(self):
        """What the range is depends on the timestamps add() stored.

        Momex used to stamp every message with the ingestion time, so this
        section reported the few minutes an import took rather than the period
        the memories cover. add(timestamp=...) is what makes it meaningful.
        """
        section = await time_section(
            _Conversation("c", ["2023-05-01T00:00:00", "2023-11-01T00:00:00"])
        )

        assert section is not None
        content = section["content"]
        assert "2023-05-01" in content and "2023-11-01" in content

    @pytest.mark.asyncio
    async def test_absent_when_the_conversation_has_no_timestamps(self):
        assert await time_section(_Conversation("c", [])) is None
