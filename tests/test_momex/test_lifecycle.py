"""Tests for Memory lifecycle: close() and async context manager support.

These use the real SQLite backend with a dummy API key. Initialization never
calls out to the LLM or embedding service, so they run offline.
"""

import pytest

from momex import LLMConfig, Memory, MomexConfig, StorageConfig


@pytest.fixture
def config(tmp_path):
    return MomexConfig(
        llm=LLMConfig(provider="openai", model="gpt-4o", api_key="sk-dummy"),
        storage=StorageConfig(path=str(tmp_path)),
    )


@pytest.mark.asyncio
async def test_close_releases_storage_provider(config):
    memory = Memory(collection="test:close", config=config)
    await memory._ensure_initialized()
    provider = memory._conversation.storage_provider  # type: ignore[union-attr]

    assert memory.is_initialized
    await memory.close()

    assert not memory.is_initialized
    assert memory._conversation is None
    # SqliteStorageProvider.close() drops its connection handle.
    assert not hasattr(provider, "db")


@pytest.mark.asyncio
async def test_close_is_idempotent(config):
    memory = Memory(collection="test:idempotent", config=config)
    await memory._ensure_initialized()

    await memory.close()
    await memory.close()  # must not raise

    assert not memory.is_initialized


@pytest.mark.asyncio
async def test_close_before_init_is_a_noop(config):
    memory = Memory(collection="test:never-opened", config=config)
    await memory.close()
    assert not memory.is_initialized


@pytest.mark.asyncio
async def test_async_context_manager_closes_on_exit(config):
    async with Memory(collection="test:ctx", config=config) as memory:
        assert memory.is_initialized

    assert not memory.is_initialized


@pytest.mark.asyncio
async def test_context_manager_closes_on_exception(config):
    memory = Memory(collection="test:ctx-error", config=config)

    with pytest.raises(RuntimeError):
        async with memory:
            raise RuntimeError("boom")

    assert not memory.is_initialized


@pytest.mark.asyncio
async def test_reinitializes_after_close(config):
    memory = Memory(collection="test:reopen", config=config)

    await memory._ensure_initialized()
    await memory.close()

    stats = await memory.stats()  # transparently re-initializes
    assert memory.is_initialized
    assert stats["collection"] == "test:reopen"
    assert stats["total_messages"] == 0

    await memory.close()


@pytest.mark.asyncio
async def test_close_drops_the_metadata_caches(config):
    """Both ledger caches are backed by metadata that close() releases."""
    from momex.results import SupersededRecord

    memory = Memory(collection="test:caches", config=config)
    await memory._ensure_initialized()

    memory._ledger._legacy_ids = {1, 2}
    memory._ledger._records = [
        SupersededRecord(
            ordinal=7, superseded_by=[], at="2026-01-01T00:00:00Z", reason="delete"
        )
    ]

    await memory.close()

    assert memory._ledger._legacy_ids is None
    assert memory._ledger._records is None


@pytest.mark.asyncio
async def test_ledger_is_re_read_from_storage_after_close(config):
    """A stale ledger cache would keep hiding refs the collection no longer has."""
    from momex.results import SupersededRecord

    memory = Memory(collection="test:stale-ledger", config=config)
    await memory._ensure_initialized()
    assert await memory.history() == []

    # Something that was never persisted must not survive the close.
    memory._ledger._records = [
        SupersededRecord(
            ordinal=7, superseded_by=[], at="2026-01-01T00:00:00Z", reason="delete"
        )
    ]
    await memory.close()

    assert await memory.history() == []
    assert (await memory.stats())["superseded"] == 0

    await memory.close()
