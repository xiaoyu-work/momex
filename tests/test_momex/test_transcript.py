"""Tests for Memory.transcript(), the durable source-message stream."""

import pytest

from momex import LLMConfig, Memory, MomexConfig, StorageConfig


@pytest.fixture
def config(tmp_path):
    return MomexConfig(
        llm=LLMConfig(provider="openai", model="gpt-4o", api_key="sk-dummy"),
        storage=StorageConfig(path=str(tmp_path)),
    )


class _Collection:
    def __init__(self, items):
        self._items = items
        self.requested: tuple[int, int] | None = None

    async def size(self):
        return len(self._items)

    async def get_slice(self, start, stop):
        self.requested = (start, stop)
        return self._items[start:stop]


class _Message:
    def __init__(self, text, ordinal):
        self.text_chunks = [text]
        self.timestamp = f"2026-01-{ordinal + 1:02d}T00:00:00Z"
        self.tags = (
            ["valid_from:2026-01-01", "valid_to:2026-12-31"] if ordinal == 1 else []
        )


class _Conversation:
    def __init__(self, messages):
        self.messages = _Collection(messages)


def _memory(config, count=5):
    memory = Memory(collection="test:history", config=config)
    memory._initialized = True
    memory._conversation = _Conversation(  # type: ignore[assignment]
        [_Message(f"message-{i}", i) for i in range(count)]
    )
    return memory


@pytest.mark.asyncio
async def test_returns_every_message_in_conversation_order(config):
    memory = _memory(config)

    items = await memory.transcript()

    assert [item.text for item in items] == [
        "message-0",
        "message-1",
        "message-2",
        "message-3",
        "message-4",
    ]
    assert [item.ordinal for item in items] == [0, 1, 2, 3, 4]
    assert all(item.type == "message" and item.score == 0.0 for item in items)


@pytest.mark.asyncio
async def test_start_and_limit_slice_without_renumbering(config):
    memory = _memory(config)

    items = await memory.transcript(start=2, limit=2)

    assert [item.text for item in items] == ["message-2", "message-3"]
    assert [item.ordinal for item in items] == [2, 3]
    assert memory._conversation.messages.requested == (2, 4)  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_limit_is_clamped_to_the_collection_end(config):
    memory = _memory(config, count=3)

    items = await memory.transcript(start=2, limit=100)

    assert [item.ordinal for item in items] == [2]
    assert memory._conversation.messages.requested == (2, 3)  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_timestamp_and_validity_window_are_preserved(config):
    memory = _memory(config)

    item = (await memory.transcript(start=1, limit=1))[0]

    assert item.timestamp == "2026-01-02T00:00:00Z"
    assert item.valid_from == "2026-01-01"
    assert item.valid_to == "2026-12-31"


@pytest.mark.asyncio
async def test_empty_and_out_of_range_slices_return_empty(config):
    memory = _memory(config, count=2)

    assert await memory.transcript(limit=0) == []
    assert await memory.transcript(start=2) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("start", "limit", "message"),
    [(-1, None, "start cannot be negative"), (0, -1, "limit cannot be negative")],
)
async def test_rejects_negative_bounds(config, start, limit, message):
    memory = _memory(config)

    with pytest.raises(ValueError, match=message):
        await memory.transcript(start=start, limit=limit)
