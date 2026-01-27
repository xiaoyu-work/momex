"""Tests for Memex Memory, MemoryManager, and prefix query functions."""

import os
import tempfile

import pytest

from memex import AddResult, Memory, MemexConfig, MemoryItem, MemoryManager


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
            memory = Memory(collection="company:engineering:alice", config=config)
            assert memory.collection == "company:engineering:alice"
            assert "company" in memory.db_path
            assert "engineering" in memory.db_path
            assert "alice" in memory.db_path
            assert not memory.is_initialized

    def test_memory_db_path_simple(self):
        """Test Memory with simple collection name (no hierarchy)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemexConfig(storage_path=tmpdir)
            memory = Memory(collection="alice", config=config)
            assert memory.db_path.endswith("memory.db")
            assert "alice" in memory.db_path

    def test_memory_db_path_hierarchical(self):
        """Test Memory with hierarchical collection name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemexConfig(storage_path=tmpdir)

            # Two levels
            m1 = Memory(collection="company:alice", config=config)
            assert "company" in m1.db_path and "alice" in m1.db_path

            # Three levels
            m2 = Memory(collection="company:engineering:bob", config=config)
            assert "company" in m2.db_path
            assert "engineering" in m2.db_path
            assert "bob" in m2.db_path

            # Four levels
            m3 = Memory(collection="org:team:project:user", config=config)
            assert "org" in m3.db_path
            assert "team" in m3.db_path
            assert "project" in m3.db_path
            assert "user" in m3.db_path


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
            db_dir = os.path.join(tmpdir, "company", "engineering", "alice")
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, "memory.db")
            with open(db_path, "w") as f:
                f.write("")

            assert manager.exists("company:engineering:alice") is True
            assert manager.exists("company:engineering:bob") is False

    def test_manager_delete(self):
        """Test deleting a collection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemexConfig(storage_path=tmpdir)
            manager = MemoryManager(config=config)

            # Create collection
            db_dir = os.path.join(tmpdir, "company", "alice")
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, "memory.db")
            with open(db_path, "w") as f:
                f.write("")

            assert manager.exists("company:alice") is True

            # Delete
            result = manager.delete("company:alice")
            assert result is True
            assert manager.exists("company:alice") is False

            # Delete non-existent
            result = manager.delete("company:bob")
            assert result is False

    def test_manager_rename(self):
        """Test renaming a collection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemexConfig(storage_path=tmpdir)
            manager = MemoryManager(config=config)

            # Create collection
            db_dir = os.path.join(tmpdir, "company", "alice")
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, "memory.db")
            with open(db_path, "w") as f:
                f.write("")

            # Rename
            result = manager.rename("company:alice", "company:alice_backup")
            assert result is True
            assert manager.exists("company:alice") is False
            assert manager.exists("company:alice_backup") is True

    def test_manager_info(self):
        """Test getting collection info."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemexConfig(storage_path=tmpdir)
            manager = MemoryManager(config=config)

            # Create collection
            db_dir = os.path.join(tmpdir, "company", "alice")
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, "memory.db")
            with open(db_path, "w") as f:
                f.write("test data")

            info = manager.info("company:alice")
            assert info["collection"] == "company:alice"
            assert "size" in info
            assert "db_path" in info

    def test_manager_list_collections(self):
        """Test listing multiple collections."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemexConfig(storage_path=tmpdir)
            manager = MemoryManager(config=config)

            # Create hierarchical collections
            for parts in [
                ("company", "engineering", "alice"),
                ("company", "engineering", "bob"),
                ("company", "marketing", "charlie"),
            ]:
                db_dir = os.path.join(tmpdir, *parts)
                os.makedirs(db_dir, exist_ok=True)
                db_path = os.path.join(db_dir, "memory.db")
                with open(db_path, "w") as f:
                    f.write("")

            collections = manager.list_collections()
            assert len(collections) == 3
            assert "company:engineering:alice" in collections
            assert "company:engineering:bob" in collections
            assert "company:marketing:charlie" in collections

    def test_manager_list_collections_with_prefix(self):
        """Test listing collections with prefix filter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemexConfig(storage_path=tmpdir)
            manager = MemoryManager(config=config)

            # Create hierarchical collections
            for parts in [
                ("company", "engineering", "alice"),
                ("company", "engineering", "bob"),
                ("company", "marketing", "charlie"),
                ("other", "team", "dave"),
            ]:
                db_dir = os.path.join(tmpdir, *parts)
                os.makedirs(db_dir, exist_ok=True)
                db_path = os.path.join(db_dir, "memory.db")
                with open(db_path, "w") as f:
                    f.write("")

            # All collections
            all_collections = manager.list_collections()
            assert len(all_collections) == 4

            # Filter by company
            company_collections = manager.list_collections(prefix="company")
            assert len(company_collections) == 3
            assert "company:engineering:alice" in company_collections
            assert "company:engineering:bob" in company_collections
            assert "company:marketing:charlie" in company_collections
            assert "other:team:dave" not in company_collections

            # Filter by company:engineering
            eng_collections = manager.list_collections(prefix="company:engineering")
            assert len(eng_collections) == 2
            assert "company:engineering:alice" in eng_collections
            assert "company:engineering:bob" in eng_collections
            assert "company:marketing:charlie" not in eng_collections

            # Filter by exact match
            exact_collections = manager.list_collections(prefix="company:engineering:alice")
            assert len(exact_collections) == 1
            assert "company:engineering:alice" in exact_collections

            # Filter with no matches
            no_match = manager.list_collections(prefix="nonexistent")
            assert len(no_match) == 0


@pytest.mark.asyncio
class TestMemoryAsync:
    """Async tests for Memory class (require LLM)."""

    @pytest.mark.skip(reason="Requires LLM API key")
    async def test_add_and_query(self):
        """Test adding and querying memories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemexConfig(storage_path=tmpdir)
            memory = Memory(collection="company:engineering:alice", config=config)

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
            memory = Memory(collection="company:engineering:alice", config=config)

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
            memory = Memory(collection="company:engineering:alice", config=config)

            await memory.add_async("Test content")
            stats = await memory.stats_async()

            assert "total_memories" in stats
            assert stats["collection"] == "company:engineering:alice"


@pytest.mark.asyncio
class TestPrefixQueryAsync:
    """Async tests for prefix query functions (require LLM)."""

    @pytest.mark.skip(reason="Requires LLM API key")
    async def test_query_single_collection(self):
        """Test querying a single collection by exact prefix."""
        from memex import query_async

        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemexConfig(storage_path=tmpdir)

            # Add memory
            memory = Memory(collection="company:engineering:alice", config=config)
            await memory.add_async("Alice likes Python")

            # Query exact collection
            answer = await query_async("company:engineering:alice", "What does Alice like?", config=config)
            assert "Python" in answer

    @pytest.mark.skip(reason="Requires LLM API key")
    async def test_query_prefix_multiple(self):
        """Test querying multiple collections by prefix."""
        from memex import query_async

        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemexConfig(storage_path=tmpdir)

            # Add memories to different collections
            alice = Memory(collection="company:engineering:alice", config=config)
            await alice.add_async("Alice likes Python")

            bob = Memory(collection="company:engineering:bob", config=config)
            await bob.add_async("Bob likes Java")

            # Query by prefix - should find both
            answer = await query_async("company:engineering", "What programming languages?", config=config)
            assert "Python" in answer or "Java" in answer
