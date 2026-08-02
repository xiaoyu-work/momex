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
