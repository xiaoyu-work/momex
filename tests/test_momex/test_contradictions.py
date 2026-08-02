"""Tests for automatic contradiction detection during add().

Contradiction detection only ever considers extracted knowledge, so it must not
pay for the embedding search, which returns messages exclusively.
"""

import pytest

from momex import LLMConfig, Memory, MomexConfig, StorageConfig
from momex.memory import SearchItem


class _FakeSemanticRef:
    def __init__(self, ordinal: int):
        self.semantic_ref_ordinal = ordinal


class _RecordingPropertyIndex:
    def __init__(self):
        self.removed: list[int] = []

    async def remove_all_for_semref(self, semref_id: int) -> None:
        self.removed.append(semref_id)


class _FakeStorageProvider:
    def __init__(self):
        self.property_index = _RecordingPropertyIndex()


class _FakeConversation:
    def __init__(self):
        self.storage_provider = _FakeStorageProvider()


class _FakeResponse:
    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    def __init__(self, content: str):
        self.content = content
        self.calls = 0

    async def complete(self, prompt: str, **kwargs):
        self.calls += 1
        return _FakeResponse(self.content)


@pytest.fixture
def memory(tmp_path, monkeypatch):
    config = MomexConfig(
        llm=LLMConfig(provider="openai", model="gpt-4o", api_key="k"),
        storage=StorageConfig(path=str(tmp_path)),
    )
    mem = Memory(collection="test:contradictions", config=config)
    mem._conversation = _FakeConversation()  # type: ignore[assignment]
    mem._initialized = True
    mem._deleted_semref_ids = set()

    async def _no_persist(deleted_ids):
        return None

    monkeypatch.setattr(mem, "_store_deleted_semref_ids", _no_persist)
    return mem


def _knowledge(text, ordinal):
    return SearchItem(
        type="entity", text=text, score=10.0, raw=_FakeSemanticRef(ordinal)
    )


def _wire(memory, monkeypatch, llm, structured_results):
    """Stub the structured path and record whether the embedding path runs."""
    calls = {"structured": 0, "embedding": 0}

    async def fake_structured(query_text, limit=10, **kwargs):
        calls["structured"] += 1
        return structured_results

    async def fake_embedding(query_text, limit=10, **kwargs):
        calls["embedding"] += 1
        return []

    monkeypatch.setattr(memory, "_search_structured", fake_structured)
    monkeypatch.setattr(memory, "_search_embedding", fake_embedding)
    monkeypatch.setattr(memory.config, "create_llm", lambda: llm)
    return calls


@pytest.mark.asyncio
async def test_does_not_run_embedding_search(memory, monkeypatch):
    llm = _FakeLLM("none")
    calls = _wire(memory, monkeypatch, llm, [_knowledge("likes sushi", 1)])

    await memory._detect_and_remove_contradictions("I don't like sushi")

    assert calls["structured"] == 1
    assert calls["embedding"] == 0


@pytest.mark.asyncio
async def test_removes_contradicting_knowledge(memory, monkeypatch):
    llm = _FakeLLM("0")
    _wire(memory, monkeypatch, llm, [_knowledge("likes sushi", 1)])

    removed = await memory._detect_and_remove_contradictions("I don't like sushi")

    assert removed == 1
    assert memory._conversation.storage_provider.property_index.removed == [1]


@pytest.mark.asyncio
async def test_no_contradiction_removes_nothing(memory, monkeypatch):
    llm = _FakeLLM("none")
    _wire(memory, monkeypatch, llm, [_knowledge("likes sushi", 1)])

    removed = await memory._detect_and_remove_contradictions("I like ramen")

    assert removed == 0
    assert memory._conversation.storage_provider.property_index.removed == []


@pytest.mark.asyncio
async def test_already_deleted_ids_are_not_recounted(memory, monkeypatch):
    llm = _FakeLLM("0")
    _wire(memory, monkeypatch, llm, [_knowledge("likes sushi", 1)])
    memory._deleted_semref_ids = {1}

    removed = await memory._detect_and_remove_contradictions("I don't like sushi")

    assert removed == 0
    assert memory._conversation.storage_provider.property_index.removed == []


@pytest.mark.asyncio
async def test_no_existing_knowledge_skips_llm_call(memory, monkeypatch):
    llm = _FakeLLM("0")
    _wire(memory, monkeypatch, llm, [])

    removed = await memory._detect_and_remove_contradictions("I like ramen")

    assert removed == 0
    assert llm.calls == 0
