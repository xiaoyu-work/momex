"""Tests for Memex Memory, MemoryPool, and MemoryManager classes."""

import os
import tempfile

import pytest

from memex import AddResult, Memory, MemexConfig, MemoryItem, MemoryManager, MemoryPool


class TestMemexConfig:
    """Tests for MemexConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MemexConfig()
        assert config.storage_path == "./memex_data"
        assert config.llm_provider == "openai"
        assert config.auto_extract is True
        assert config.db_name == "memory.db"

    def test_custom_config(self):
        """Test custom configuration."""
        config = MemexConfig(
            storage_path="/custom/path",
            llm_provider="azure",
            llm_model="gpt-4",
            auto_extract=False,
        )
        assert config.storage_path == "/custom/path"
        assert config.llm_provider == "azure"
        assert config.llm_model == "gpt-4"
        assert config.auto_extract is False


class TestMemory:
    """Tests for Memory class."""

    def test_memory_init(self):
        """Test Memory initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemexConfig(storage_path=tmpdir)
            memory = Memory(collection="user:alice", config=config)
            assert memory.collection == "user:alice"
            # Collection "user:alice" becomes path "user/alice" for cross-platform compatibility
            assert "user" in memory.db_path and "alice" in memory.db_path
            assert not memory.is_initialized

    def test_memory_db_path(self):
        """Test Memory database path generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemexConfig(storage_path=tmpdir)

            # Simple collection - "user:alice" becomes path "user/alice"
            m1 = Memory(collection="user:alice", config=config)
            assert m1.db_path.endswith("memory.db")
            assert "user" in m1.db_path and "alice" in m1.db_path

            # Another collection - "team:engineering" becomes "team/engineering"
            m2 = Memory(collection="team:engineering", config=config)
            assert "team" in m2.db_path and "engineering" in m2.db_path


class TestMemoryPool:
    """Tests for MemoryPool class."""

    def test_pool_init(self):
        """Test MemoryPool initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemexConfig(storage_path=tmpdir)
            pool = MemoryPool(
                collections=["user:alice", "team:engineering"],
                config=config,
            )
            assert pool.collections == ["user:alice", "team:engineering"]
            assert pool.default_collection is None

    def test_pool_with_default(self):
        """Test MemoryPool with default collection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemexConfig(storage_path=tmpdir)
            pool = MemoryPool(
                collections=["user:alice", "team:engineering"],
                default_collection="user:alice",
                config=config,
            )
            assert pool.default_collection == "user:alice"

    def test_pool_invalid_default(self):
        """Test MemoryPool with invalid default collection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemexConfig(storage_path=tmpdir)
            with pytest.raises(ValueError):
                MemoryPool(
                    collections=["user:alice"],
                    default_collection="user:bob",  # Not in list
                    config=config,
                )

    def test_pool_empty_collections(self):
        """Test MemoryPool with empty collections."""
        with pytest.raises(ValueError):
            MemoryPool(collections=[])

    def test_pool_get_memory(self):
        """Test getting Memory instance from pool."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemexConfig(storage_path=tmpdir)
            pool = MemoryPool(
                collections=["user:alice", "team:engineering"],
                config=config,
            )

            memory = pool.get_memory("user:alice")
            assert memory.collection == "user:alice"

            with pytest.raises(ValueError):
                pool.get_memory("user:bob")  # Not in pool


class TestMemoryManager:
    """Tests for MemoryManager class."""

    def test_manager_list_empty(self):
        """Test listing collections when empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemexConfig(storage_path=tmpdir)
            manager = MemoryManager(config=config)

            collections = manager.list_collections()
            assert collections == []

    def test_manager_exists(self):
        """Test checking if collection exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemexConfig(storage_path=tmpdir)
            manager = MemoryManager(config=config)

            # Manually create collection
            db_dir = os.path.join(tmpdir, "user", "alice")
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, "memory.db")
            with open(db_path, "w") as f:
                f.write("")

            assert manager.exists("user:alice") is True
            assert manager.exists("user:bob") is False

    def test_manager_delete(self):
        """Test deleting a collection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemexConfig(storage_path=tmpdir)
            manager = MemoryManager(config=config)

            # Create collection
            db_dir = os.path.join(tmpdir, "user", "alice")
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, "memory.db")
            with open(db_path, "w") as f:
                f.write("")

            assert manager.exists("user:alice") is True

            # Delete
            result = manager.delete("user:alice")
            assert result is True
            assert manager.exists("user:alice") is False

            # Delete non-existent
            result = manager.delete("user:bob")
            assert result is False

    def test_manager_rename(self):
        """Test renaming a collection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemexConfig(storage_path=tmpdir)
            manager = MemoryManager(config=config)

            # Create collection
            db_dir = os.path.join(tmpdir, "user", "alice")
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, "memory.db")
            with open(db_path, "w") as f:
                f.write("")

            # Rename
            result = manager.rename("user:alice", "user:alice_backup")
            assert result is True
            assert manager.exists("user:alice") is False
            assert manager.exists("user:alice_backup") is True

    def test_manager_info(self):
        """Test getting collection info."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemexConfig(storage_path=tmpdir)
            manager = MemoryManager(config=config)

            # Create collection
            db_dir = os.path.join(tmpdir, "user", "alice")
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, "memory.db")
            with open(db_path, "w") as f:
                f.write("test data")

            info = manager.info("user:alice")
            assert info["collection"] == "user:alice"
            assert "size" in info
            assert "db_path" in info

    def test_manager_list_collections(self):
        """Test listing multiple collections."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemexConfig(storage_path=tmpdir)
            manager = MemoryManager(config=config)

            # Create multiple collections using subdirectory structure
            # "user:alice" -> "user/alice", "team:engineering" -> "team/engineering"
            for parts in [("user", "alice"), ("user", "bob"), ("team", "engineering")]:
                db_dir = os.path.join(tmpdir, *parts)
                os.makedirs(db_dir, exist_ok=True)
                db_path = os.path.join(db_dir, "memory.db")
                with open(db_path, "w") as f:
                    f.write("")

            collections = manager.list_collections()
            assert len(collections) == 3
            assert "user:alice" in collections
            assert "user:bob" in collections
            assert "team:engineering" in collections


@pytest.mark.asyncio
class TestMemoryAsync:
    """Async tests for Memory class (require LLM)."""

    @pytest.mark.skip(reason="Requires LLM API key")
    async def test_add_and_query(self):
        """Test adding and querying memories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemexConfig(storage_path=tmpdir)
            memory = Memory(collection="user:test", config=config)

            result = await memory.add_async(
                "Alice likes cats",
                speaker="narrator",
            )
            assert result.success
            assert result.messages_added == 1

            answer = await memory.query_async("What does Alice like?")
            assert "cat" in answer.lower()

    @pytest.mark.skip(reason="Requires LLM API key")
    async def test_search(self):
        """Test searching memories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemexConfig(storage_path=tmpdir)
            memory = Memory(collection="user:test", config=config)

            await memory.add_async("Alice is a software engineer")
            await memory.add_async("Bob is a data scientist")

            results = await memory.search_async("Alice")
            assert len(results) >= 1
            assert any("Alice" in r.text for r in results)

    @pytest.mark.skip(reason="Requires LLM API key")
    async def test_stats(self):
        """Test memory statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemexConfig(storage_path=tmpdir)
            memory = Memory(collection="user:test", config=config)

            await memory.add_async("Test content")
            stats = await memory.stats_async()

            assert "total_memories" in stats
            assert stats["collection"] == "user:test"


@pytest.mark.asyncio
class TestMemoryPoolAsync:
    """Async tests for MemoryPool class (require LLM)."""

    @pytest.mark.skip(reason="Requires LLM API key")
    async def test_pool_add_to_multiple(self):
        """Test adding to multiple collections."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemexConfig(storage_path=tmpdir)
            pool = MemoryPool(
                collections=["user:alice", "team:engineering"],
                config=config,
            )

            result = await pool.add_async(
                "Shared knowledge",
                collections=["user:alice", "team:engineering"],
            )
            assert result.messages_added == 2  # Added to both
            assert result.collections == ["user:alice", "team:engineering"]

    @pytest.mark.skip(reason="Requires LLM API key")
    async def test_pool_query_all(self):
        """Test querying across all collections."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemexConfig(storage_path=tmpdir)
            pool = MemoryPool(
                collections=["user:alice", "team:engineering"],
                default_collection="user:alice",
                config=config,
            )

            await pool.add_async("Personal note", collections=["user:alice"])
            await pool.add_async("Team uses PostgreSQL", collections=["team:engineering"])

            answer = await pool.query_async("What database?")
            assert "PostgreSQL" in answer
