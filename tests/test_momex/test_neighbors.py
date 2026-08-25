"""Tests for widening message results to the turns spoken around them.

Retrieval scores every turn on its own, which is not how conversation works.
"I finally bought a clarinet" / "Nice, so that's two instruments now" -- asked
which instruments someone plays, the second turn matches nothing by itself, so
ranking cannot reach it at any depth. The adjacency that answers the question
is sitting in the collection and never consulted.

Measured on LOCOMO conv-26, 120 stratified questions, gpt-4.1-mini: widening
each result by two turns either side moved judged accuracy from 79.2% to 85.8%,
with multi-hop going 57.7% -> 76.9% and single-hop 80.0% -> 89.1%.
"""

from typing import Any

import pytest

from momex.results import SearchItem
from momex.search import expand_with_neighbors


class _Messages:
    """Enough of the message collection for expansion to work against."""

    def __init__(self, texts: list[str]):
        self._texts = texts
        self.get_multiple_calls = 0

    async def size(self) -> int:
        return len(self._texts)

    async def get_multiple(self, ordinals: list[int]) -> list[Any]:
        self.get_multiple_calls += 1
        return [_Message(self._texts[o]) for o in ordinals]

    async def get_item(self, ordinal: int) -> Any:
        return _Message(self._texts[ordinal])


class _Message:
    def __init__(self, text: str):
        self.text_chunks = [text]


class _Conversation:
    def __init__(self, texts: list[str]):
        self.messages = _Messages(texts)


TURNS = ["turn-0", "turn-1", "turn-2", "turn-3", "turn-4"]


def _message_item(ordinal: int, score: float = 0.9) -> SearchItem:
    return SearchItem(
        type="message",
        text=TURNS[ordinal],
        score=score,
        raw=object(),
        ordinal=ordinal,
    )


@pytest.mark.asyncio
async def test_radius_zero_is_a_no_op():
    conversation = _Conversation(TURNS)
    items = [_message_item(2)]

    result = await expand_with_neighbors(conversation, items, radius=0)

    assert result == items
    assert conversation.messages.get_multiple_calls == 0


@pytest.mark.asyncio
async def test_a_result_gains_the_turns_on_either_side():
    conversation = _Conversation(TURNS)

    result = await expand_with_neighbors(conversation, [_message_item(2)], radius=1)

    assert result[0].text == "turn-1\nturn-2\nturn-3"


@pytest.mark.asyncio
async def test_expansion_keeps_the_result_in_order_and_intact():
    """Only the text grows: score, ordinal and count must not move."""
    conversation = _Conversation(TURNS)
    items = [_message_item(2, score=0.9), _message_item(4, score=0.5)]

    result = await expand_with_neighbors(conversation, items, radius=1)

    assert len(result) == 2
    assert [item.score for item in result] == [0.9, 0.5]
    assert [item.ordinal for item in result] == [2, 4]


@pytest.mark.asyncio
async def test_edges_do_not_run_off_the_ends():
    conversation = _Conversation(TURNS)

    first, last = await expand_with_neighbors(
        conversation, [_message_item(0), _message_item(4)], radius=2
    )

    assert first.text == "turn-0\nturn-1\nturn-2"
    assert last.text == "turn-2\nturn-3\nturn-4"


@pytest.mark.asyncio
async def test_knowledge_results_are_left_alone():
    """Entities and actions have no position in the transcript to widen to."""
    conversation = _Conversation(TURNS)
    knowledge = SearchItem(type="entity", text="clarinet", score=8.0, raw=object())

    (result,) = await expand_with_neighbors(conversation, [knowledge], radius=2)

    assert result is knowledge


@pytest.mark.asyncio
async def test_a_message_without_a_position_is_left_alone():
    conversation = _Conversation(TURNS)
    orphan = SearchItem(type="message", text="from somewhere", score=0.5, raw=object())

    (result,) = await expand_with_neighbors(conversation, [orphan], radius=2)

    assert result.text == "from somewhere"


@pytest.mark.asyncio
async def test_overlapping_windows_are_fetched_once():
    """Neighbouring hits share turns; reading them twice is wasted work."""
    conversation = _Conversation(TURNS)

    await expand_with_neighbors(
        conversation, [_message_item(1), _message_item(2)], radius=1
    )

    assert conversation.messages.get_multiple_calls == 1


@pytest.mark.asyncio
async def test_expansion_does_not_mutate_the_input():
    """Callers holding the original results should not see them change."""
    conversation = _Conversation(TURNS)
    original = _message_item(2)

    await expand_with_neighbors(conversation, [original], radius=1)

    assert original.text == "turn-2"
