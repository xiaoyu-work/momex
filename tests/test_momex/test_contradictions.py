"""Tests for automatic contradiction detection during add().

Contradiction detection only ever considers extracted knowledge, so it must not
pay for the embedding search, which returns messages exclusively.

Contradicted memories are superseded, not destroyed: the detector appends
ledger entries and returns them, and the underlying semantic refs stay intact
so restore() can put them back.
"""

import logging

import pytest

from momex import LLMConfig, Memory, MomexConfig, StorageConfig
from momex.memory import SearchItem


class _FakeSemanticRef:
    def __init__(self, ordinal: int):
        self.semantic_ref_ordinal = ordinal


class _FakeMetadata:
    def __init__(self):
        self.extra: dict[str, str] = {}


class _FakeStorageProvider:
    """Enough of the storage API for the ledger's metadata round trip.

    The ledger is re-read from storage on every append, so a provider that
    cannot persist would make each call start from an empty ledger.
    """

    def __init__(self):
        self._metadata = _FakeMetadata()

    async def get_conversation_metadata(self):
        return self._metadata

    async def set_conversation_metadata(self, **kwds):
        for key, value in kwds.items():
            if value is None:
                self._metadata.extra.pop(key, None)
            else:
                self._metadata.extra[key] = value


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
    mem._supersession_ledger = []

    async def _no_persist(_):
        return None

    monkeypatch.setattr(mem, "_store_deleted_semref_ids", _no_persist)
    return mem


def _hidden(memory) -> list[int]:
    """Ordinals currently hidden from search, in ledger order."""
    return [r.ordinal for r in memory._supersession_ledger if r.active]


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
async def test_supersedes_contradicting_knowledge(memory, monkeypatch):
    llm = _FakeLLM("0")
    _wire(memory, monkeypatch, llm, [_knowledge("likes sushi", 1)])

    superseded = await memory._detect_and_remove_contradictions(
        "I don't like sushi", superseded_by=[9]
    )

    assert [r.ordinal for r in superseded] == [1]
    assert _hidden(memory) == [1]

    (record,) = superseded
    assert record.reason == "contradiction"
    assert record.text == "likes sushi"
    assert record.query == "I don't like sushi"
    assert record.superseded_by == [9]


@pytest.mark.asyncio
async def test_no_contradiction_removes_nothing(memory, monkeypatch):
    llm = _FakeLLM("none")
    _wire(memory, monkeypatch, llm, [_knowledge("likes sushi", 1)])

    superseded = await memory._detect_and_remove_contradictions("I like ramen")

    assert superseded == []
    assert _hidden(memory) == []


@pytest.mark.asyncio
async def test_already_superseded_ids_are_not_recounted(memory, monkeypatch):
    from momex.memory import SupersededRecord

    llm = _FakeLLM("0")
    _wire(memory, monkeypatch, llm, [_knowledge("likes sushi", 1)])
    await memory._store_ledger(
        [
            SupersededRecord(
                ordinal=1,
                superseded_by=[],
                at="2026-01-01T00:00:00Z",
                reason="contradiction",
            )
        ]
    )

    superseded = await memory._detect_and_remove_contradictions("I don't like sushi")

    assert superseded == []
    assert _hidden(memory) == [1]


@pytest.mark.asyncio
async def test_no_existing_knowledge_skips_llm_call(memory, monkeypatch):
    llm = _FakeLLM("0")
    _wire(memory, monkeypatch, llm, [])

    superseded = await memory._detect_and_remove_contradictions("I like ramen")

    assert superseded == []
    assert llm.calls == 0


@pytest.mark.asyncio
async def test_llm_failure_is_logged_and_does_not_block_add(
    memory, monkeypatch, caplog
):
    """A broken LLM must degrade loudly, not silently."""

    class _BrokenLLM:
        async def complete(self, prompt, **kwargs):
            raise RuntimeError("llm is down")

    _wire(memory, monkeypatch, _BrokenLLM(), [_knowledge("likes sushi", 1)])

    with caplog.at_level(logging.WARNING, logger="momex.memory"):
        superseded = await memory._detect_and_remove_contradictions(
            "I don't like sushi"
        )

    assert superseded == []
    assert "Contradiction detection failed" in caplog.text
    assert "llm is down" in caplog.text


@pytest.mark.asyncio
async def test_lookup_failure_is_logged_and_does_not_block_add(
    memory, monkeypatch, caplog
):
    """The lookup is an LLM round trip too, and it ran outside the guard."""

    async def broken_structured(query_text, limit=10, **kwargs):
        raise RuntimeError("query translation failed")

    monkeypatch.setattr(memory, "_search_structured", broken_structured)
    monkeypatch.setattr(memory.config, "create_llm", lambda: _FakeLLM("0"))

    with caplog.at_level(logging.WARNING, logger="momex.memory"):
        superseded = await memory._detect_and_remove_contradictions(
            "I don't like sushi"
        )

    assert superseded == []
    assert "Contradiction detection lookup failed" in caplog.text
    assert "query translation failed" in caplog.text


@pytest.mark.asyncio
async def test_add_reports_the_write_even_when_detection_cannot_run(
    memory, monkeypatch
):
    """add() commits before detecting, so detection must never fail the call."""

    class _Result:
        messages_added = 1
        semrefs_added = 2

    class _Semrefs:
        async def size(self):
            return 0

    class _Conversation:
        semantic_refs = _Semrefs()
        storage_provider = _FakeStorageProvider()

        async def add_messages_with_indexing(self, messages):
            return _Result()

    async def broken_structured(query_text, limit=10, **kwargs):
        raise RuntimeError("query translation failed")

    memory._conversation = _Conversation()  # type: ignore[assignment]
    monkeypatch.setattr(memory, "_search_structured", broken_structured)

    result = await memory.add("I don't like sushi")

    assert result.messages_added == 1
    assert result.entities_extracted == 2
    assert result.contradictions_removed == 0
    assert result.superseded is None
