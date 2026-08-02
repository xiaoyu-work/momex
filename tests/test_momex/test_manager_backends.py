"""Tests that MemoryManager is backend-aware.

The filesystem-based operations used to run against ./momex_data even when the
configured backend was PostgreSQL, silently reporting on (or modifying) an
unrelated local directory. They must now fail loudly instead.
"""

import pytest

from momex import LLMConfig, MemoryManager, MomexConfig, StorageConfig
from momex.exceptions import StorageError


@pytest.fixture
def pg_manager(tmp_path):
    config = MomexConfig(
        llm=LLMConfig(provider="openai", model="gpt-4o", api_key="k"),
        storage=StorageConfig(
            backend="postgres",
            path=str(tmp_path),
            postgres_url="postgresql://user:pw@localhost:5432/momex",
        ),
    )
    return MemoryManager(config=config)


@pytest.fixture
def sqlite_manager(tmp_path):
    config = MomexConfig(
        llm=LLMConfig(provider="openai", model="gpt-4o", api_key="k"),
        storage=StorageConfig(path=str(tmp_path)),
    )
    return MemoryManager(config=config)


class TestPostgresGuards:
    def test_rename_rejects_postgres(self, pg_manager):
        with pytest.raises(StorageError, match="not supported on the PostgreSQL"):
            pg_manager.rename("a", "b")

    def test_copy_rejects_postgres(self, pg_manager):
        with pytest.raises(StorageError, match="not supported on the PostgreSQL"):
            pg_manager.copy("a", "b")

    def test_info_rejects_postgres(self, pg_manager):
        with pytest.raises(StorageError, match="not supported on the PostgreSQL"):
            pg_manager.info("a")

    @pytest.mark.asyncio
    async def test_exists_rejects_sync_call_inside_event_loop(self, pg_manager):
        with pytest.raises(StorageError, match="requires async existence checks"):
            pg_manager.exists("a")

    @pytest.mark.asyncio
    async def test_delete_rejects_shared_schema(self, tmp_path):
        config = MomexConfig(
            llm=LLMConfig(provider="openai", model="gpt-4o", api_key="k"),
            storage=StorageConfig(
                backend="postgres",
                postgres_url="postgresql://user:pw@localhost:5432/momex",
                postgres_schema="shared",
            ),
        )
        manager = MemoryManager(config=config)

        with pytest.raises(StorageError, match="all collections share"):
            await manager.delete_async("a")


class TestSqliteUnaffected:
    def test_sqlite_operations_still_work(self, sqlite_manager, tmp_path):
        db = tmp_path / "user" / "alice" / "memory.db"
        db.parent.mkdir(parents=True)
        db.write_text("x")

        assert sqlite_manager.exists("user:alice")
        assert sqlite_manager.list_collections() == ["user:alice"]
        assert sqlite_manager.info("user:alice")["collection"] == "user:alice"

        assert sqlite_manager.rename("user:alice", "user:bob")
        assert sqlite_manager.exists("user:bob")
        assert not sqlite_manager.exists("user:alice")

        assert sqlite_manager.copy("user:bob", "user:carol")
        assert sqlite_manager.exists("user:carol")

        assert sqlite_manager.delete("user:carol")
        assert not sqlite_manager.exists("user:carol")

    @pytest.mark.asyncio
    async def test_exists_async_delegates_to_sqlite(self, sqlite_manager, tmp_path):
        db = tmp_path / "solo" / "memory.db"
        db.parent.mkdir(parents=True)
        db.write_text("x")

        assert await sqlite_manager.exists_async("solo")
        assert not await sqlite_manager.exists_async("missing")
