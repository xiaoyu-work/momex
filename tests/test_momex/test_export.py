"""Tests for Memory.export().

export() sliced both collections with a hardcoded upper bound of 999999, so a
collection larger than that lost the tail without any indication. The bound is
now taken from the collections themselves.
"""

import json

import pytest

from momex import LLMConfig, Memory, MomexConfig, StorageConfig


@pytest.fixture
def config(tmp_path):
    return MomexConfig(
        llm=LLMConfig(provider="openai", model="gpt-4o", api_key="sk-dummy"),
        storage=StorageConfig(path=str(tmp_path)),
    )


class _Collection:
    """Records the slice it was asked for, like the real collections do."""

    def __init__(self, items):
        self._items = items
        self.requested: tuple[int, int] | None = None

    async def size(self):
        return len(self._items)

    async def get_slice(self, start, stop):
        self.requested = (start, stop)
        return self._items[start:stop]


class _Message:
    def __init__(self, text, timestamp="2026-01-01T00:00:00Z"):
        self.text_chunks = [text]
        self.timestamp = timestamp


class _Conversation:
    def __init__(self, messages, semrefs):
        self.messages = _Collection(messages)
        self.semantic_refs = _Collection(semrefs)


@pytest.mark.asyncio
async def test_exports_every_message(config, tmp_path):
    memory = Memory(collection="test:export", config=config)
    memory._initialized = True
    conversation = _Conversation([_Message(f"m{i}") for i in range(2500)], [])
    memory._conversation = conversation  # type: ignore[assignment]

    out = tmp_path / "dump.json"
    await memory.export(str(out))

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["collection"] == "test:export"
    assert len(data["messages"]) == 2500
    assert data["messages"][0]["text"] == "m0"
    assert data["messages"][-1]["text"] == "m2499"


@pytest.mark.asyncio
async def test_slice_bound_comes_from_the_collection(config, tmp_path):
    """Not from a constant that a large collection would exceed."""
    memory = Memory(collection="test:export-bound", config=config)
    memory._initialized = True
    conversation = _Conversation([_Message("only")], [])
    memory._conversation = conversation  # type: ignore[assignment]

    await memory.export(str(tmp_path / "dump.json"))

    assert conversation.messages.requested == (0, 1)
    assert conversation.semantic_refs.requested == (0, 0)


@pytest.mark.asyncio
async def test_empty_collection_exports_cleanly(config, tmp_path):
    memory = Memory(collection="test:export-empty", config=config)
    memory._initialized = True
    memory._conversation = _Conversation([], [])  # type: ignore[assignment]

    out = tmp_path / "dump.json"
    await memory.export(str(out))

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["messages"] == []
    assert data["knowledge"] == []
