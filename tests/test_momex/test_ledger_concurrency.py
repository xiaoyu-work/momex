"""Tests that concurrent ledger writers do not erase each other.

The supersession ledger is one metadata blob, so appending to it is a
read-modify-write. If the read suspends, every concurrent writer reads the same
list and writes back its own version, and the last one wins -- silently
un-hiding memories that had just been superseded.

Whether the read suspends depends on the backend. The SQLite provider does its
work synchronously inside `async def`, so a whole append happens between two
event-loop ticks and cannot interleave. The PostgreSQL provider talks to a
socket, so it *does* suspend, and that is where the lost updates happen. These
tests reproduce that by making the metadata read yield, which is the one
behavioural difference that matters here.
"""

import asyncio

import pytest

from momex import LLMConfig, Memory, MomexConfig, StorageConfig
from momex.memory import SearchItem, SupersededRecord


class _FakeSemanticRef:
    def __init__(self, ordinal: int):
        self.semantic_ref_ordinal = ordinal


@pytest.fixture
def config(tmp_path):
    return MomexConfig(
        llm=LLMConfig(provider="openai", model="gpt-4o", api_key="sk-dummy"),
        storage=StorageConfig(path=str(tmp_path)),
    )


def _record(ordinal: int) -> SupersededRecord:
    return SupersededRecord(
        ordinal=ordinal, superseded_by=[], at="2026-01-01T00:00:00Z", reason="delete"
    )


def _make_reads_suspend(memory, monkeypatch):
    """Model a networked storage provider, whose metadata read awaits I/O."""
    original = memory._get_conversation_metadata

    async def suspending_read():
        await asyncio.sleep(0)
        return await original()

    monkeypatch.setattr(memory, "_get_conversation_metadata", suspending_read)


@pytest.mark.asyncio
async def test_concurrent_appends_all_survive(config, monkeypatch):
    """Ten racing appends must produce ten ledger entries, not one."""
    memory = Memory(collection="test:ledger-race", config=config)
    await memory._ensure_initialized()
    _make_reads_suspend(memory, monkeypatch)

    await asyncio.gather(
        *(memory._append_supersessions([_record(i)]) for i in range(10))
    )

    assert sorted(r.ordinal for r in await memory.history()) == list(range(10))
    await memory.close()


@pytest.mark.asyncio
async def test_appends_survive_across_batches(config, monkeypatch):
    memory = Memory(collection="test:ledger-batches", config=config)
    await memory._ensure_initialized()
    _make_reads_suspend(memory, monkeypatch)

    await asyncio.gather(
        memory._append_supersessions([_record(1), _record(2)]),
        memory._append_supersessions([_record(3)]),
        memory._append_supersessions([_record(4)]),
    )

    assert sorted(r.ordinal for r in await memory.history()) == [1, 2, 3, 4]
    await memory.close()


@pytest.mark.asyncio
async def test_concurrent_deletes_are_all_recorded(config, monkeypatch):
    """The same race reached through the public API."""
    memory = Memory(collection="test:delete-race", config=config)
    await memory._ensure_initialized()
    _make_reads_suspend(memory, monkeypatch)

    async def fake_structured(query_text, limit=10, **kwargs):
        ordinal = int(query_text)
        return [
            SearchItem(
                type="entity",
                text=f"item {ordinal}",
                score=9.0,
                raw=_FakeSemanticRef(ordinal),
            )
        ]

    monkeypatch.setattr(memory, "_search_structured", fake_structured)

    counts = await asyncio.gather(*(memory.delete(str(i)) for i in range(8)))

    assert counts == [1] * 8
    assert sorted(r.ordinal for r in await memory.history()) == list(range(8))
    await memory.close()


@pytest.mark.asyncio
async def test_restore_does_not_clobber_a_concurrent_append(config, monkeypatch):
    memory = Memory(collection="test:restore-race", config=config)
    await memory._ensure_initialized()
    await memory._append_supersessions([_record(1)])
    _make_reads_suspend(memory, monkeypatch)

    restored, _ = await asyncio.gather(
        memory.restore(1),
        memory._append_supersessions([_record(2)]),
    )

    assert restored == 1
    assert [r.ordinal for r in await memory.history()] == [2]
    assert sorted(r.ordinal for r in await memory.history(include_restored=True)) == [
        1,
        2,
    ]
    await memory.close()


@pytest.mark.asyncio
async def test_a_second_instance_does_not_overwrite_the_first(config):
    """A live instance's cached ledger must not erase another's entries."""
    first = Memory(collection="test:two-instances", config=config)
    second = Memory(collection="test:two-instances", config=config)

    # Warm both caches, so neither will re-read unless the append makes it.
    assert await first.history() == []
    assert await second.history() == []

    await first._append_supersessions([_record(1)])
    await second._append_supersessions([_record(2)])

    assert sorted(r.ordinal for r in await second.history()) == [1, 2]

    await first.close()
    await second.close()

    reader = Memory(collection="test:two-instances", config=config)
    assert sorted(r.ordinal for r in await reader.history()) == [1, 2]
    await reader.close()
