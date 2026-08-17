"""Tests for Memory.delete() candidate selection and accounting.

delete() used to report one deletion per candidate id regardless of what
actually happened, and re-deleting the same query inflated the count further.

Deletion is now a supersession: the semantic ref is never destroyed, it is
appended to the ledger with reason="delete" and hidden from search.
"""

import pytest

from momex import LLMConfig, Memory, MomexConfig, StorageConfig
from momex.memory import SearchItem


class _FakeSemanticRef:
    def __init__(self, ordinal: int):
        self.semantic_ref_ordinal = ordinal


class _FakeStorageProvider:
    pass


class _FakeConversation:
    def __init__(self):
        self.storage_provider = _FakeStorageProvider()


@pytest.fixture
def memory(tmp_path, monkeypatch):
    config = MomexConfig(
        llm=LLMConfig(provider="openai", model="gpt-4o", api_key="k"),
        storage=StorageConfig(path=str(tmp_path)),
    )
    mem = Memory(collection="test:delete", config=config)
    mem._conversation = _FakeConversation()  # type: ignore[assignment]
    mem._initialized = True
    mem._deleted_semref_ids = set()
    mem._supersession_ledger = []

    async def _no_persist_ledger(records):
        mem._supersession_ledger = records

    monkeypatch.setattr(mem, "_store_ledger", _no_persist_ledger)
    return mem


def _hidden(memory) -> list[int]:
    """Ordinals currently hidden from search, in ledger order."""
    return [r.ordinal for r in memory._supersession_ledger if r.active]


def _stub_search(memory, items):
    async def fake_search(query_text, limit=10, **kwargs):
        return items

    memory.search = fake_search  # type: ignore[method-assign]


def _knowledge(text, ordinal, score=10.0):
    return SearchItem(
        type="entity", text=text, score=score, raw=_FakeSemanticRef(ordinal)
    )


def _message(text, score=0.9):
    return SearchItem(type="message", text=text, score=score, raw=object())


@pytest.mark.asyncio
async def test_deletes_knowledge_and_reports_accurate_count(memory):
    _stub_search(memory, [_knowledge("likes sushi", 1), _knowledge("sushi", 2)])

    assert await memory.delete("likes sushi") == 2
    assert _hidden(memory) == [1, 2]


@pytest.mark.asyncio
async def test_messages_are_not_deleted(memory):
    _stub_search(memory, [_message("I like sushi"), _knowledge("likes sushi", 1)])

    assert await memory.delete("sushi") == 1
    assert _hidden(memory) == [1]


@pytest.mark.asyncio
async def test_repeat_delete_does_not_inflate_count(memory):
    _stub_search(memory, [_knowledge("likes sushi", 1)])

    assert await memory.delete("likes sushi") == 1
    assert await memory.delete("likes sushi") == 0
    assert _hidden(memory) == [1]


@pytest.mark.asyncio
async def test_duplicate_ids_counted_once(memory):
    _stub_search(memory, [_knowledge("a", 7), _knowledge("b", 7)])

    assert await memory.delete("whatever") == 1
    assert _hidden(memory) == [7]


@pytest.mark.asyncio
async def test_min_score_filters_weak_matches(memory):
    _stub_search(
        memory,
        [_knowledge("strong", 1, score=9.0), _knowledge("weak", 2, score=0.4)],
    )

    assert await memory.delete("sushi", min_score=1.0) == 1
    assert _hidden(memory) == [1]


@pytest.mark.asyncio
async def test_dry_run_changes_nothing(memory):
    _stub_search(memory, [_knowledge("likes sushi", 1), _knowledge("sushi", 2)])

    assert await memory.delete("likes sushi", dry_run=True) == 2
    assert _hidden(memory) == []

    # A real delete afterwards still reports both.
    assert await memory.delete("likes sushi") == 2


@pytest.mark.asyncio
async def test_no_matches_returns_zero(memory):
    _stub_search(memory, [])

    assert await memory.delete("nothing") == 0
    assert _hidden(memory) == []


@pytest.mark.asyncio
async def test_ledger_entry_records_why_and_what(memory):
    """The audit trail must carry enough to review the decision later."""
    _stub_search(memory, [_knowledge("likes sushi", 1)])

    await memory.delete("sushi preferences")

    (record,) = memory._supersession_ledger
    assert record.ordinal == 1
    assert record.reason == "delete"
    assert record.text == "likes sushi"
    assert record.query == "sushi preferences"
    assert record.superseded_by == []
    assert record.at
    assert record.active
