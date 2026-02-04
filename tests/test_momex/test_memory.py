"""Tests for Momex Memory, MemoryManager, and prefix query functions."""

import os
import tempfile

import pytest

from momex import AddResult, Memory, MomexConfig, MemoryManager


class TestMomexConfig:
    """Tests for MomexConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MomexConfig()
        assert config.storage_path == "./momex_data"
        assert config.backend == "sqlite"

    def test_custom_config(self):
        """Test custom configuration."""
        config = MomexConfig(
            storage_path="/custom/path",
            provider="azure",
            model="gpt-4",
        )
        assert config.storage_path == "/custom/path"
        assert config.provider == "azure"
        assert config.model == "gpt-4"


class TestMemory:
    """Tests for Memory class."""

    def test_memory_init(self):
        """Test Memory initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MomexConfig(storage_path=tmpdir)
            memory = Memory(collection="momex:engineering:xiaoyuzhang", config=config)
            assert memory.collection == "momex:engineering:xiaoyuzhang"
            assert "momex" in memory.db_path
            assert "engineering" in memory.db_path
            assert "xiaoyuzhang" in memory.db_path
            assert not memory.is_initialized

    def test_memory_db_path_simple(self):
        """Test Memory with simple collection name (no hierarchy)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MomexConfig(storage_path=tmpdir)
            memory = Memory(collection="xiaoyuzhang", config=config)
            assert memory.db_path.endswith("memory.db")
            assert "xiaoyuzhang" in memory.db_path

    def test_memory_db_path_hierarchical(self):
        """Test Memory with hierarchical collection name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MomexConfig(storage_path=tmpdir)

            # Two levels
            m1 = Memory(collection="momex:xiaoyuzhang", config=config)
            assert "momex" in m1.db_path and "xiaoyuzhang" in m1.db_path

            # Three levels
            m2 = Memory(collection="momex:engineering:gvanrossum", config=config)
            assert "momex" in m2.db_path
            assert "engineering" in m2.db_path
            assert "gvanrossum" in m2.db_path

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
            config = MomexConfig(storage_path=tmpdir)
            manager = MemoryManager(config=config)

            collections = manager.list_collections()
            assert collections == []

    def test_manager_exists(self):
        """Test checking if collection exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MomexConfig(storage_path=tmpdir)
            manager = MemoryManager(config=config)

            # Manually create collection
            db_dir = os.path.join(tmpdir, "momex", "engineering", "xiaoyuzhang")
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, "memory.db")
            with open(db_path, "w") as f:
                f.write("")

            assert manager.exists("momex:engineering:xiaoyuzhang") is True
            assert manager.exists("momex:engineering:gvanrossum") is False

    def test_manager_delete(self):
        """Test deleting a collection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MomexConfig(storage_path=tmpdir)
            manager = MemoryManager(config=config)

            # Create collection
            db_dir = os.path.join(tmpdir, "momex", "xiaoyuzhang")
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, "memory.db")
            with open(db_path, "w") as f:
                f.write("")

            assert manager.exists("momex:xiaoyuzhang") is True

            # Delete
            result = manager.delete("momex:xiaoyuzhang")
            assert result is True
            assert manager.exists("momex:xiaoyuzhang") is False

            # Delete non-existent
            result = manager.delete("momex:gvanrossum")
            assert result is False

    def test_manager_rename(self):
        """Test renaming a collection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MomexConfig(storage_path=tmpdir)
            manager = MemoryManager(config=config)

            # Create collection
            db_dir = os.path.join(tmpdir, "momex", "xiaoyuzhang")
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, "memory.db")
            with open(db_path, "w") as f:
                f.write("")

            # Rename
            result = manager.rename("momex:xiaoyuzhang", "momex:xiaoyuzhang_backup")
            assert result is True
            assert manager.exists("momex:xiaoyuzhang") is False
            assert manager.exists("momex:xiaoyuzhang_backup") is True

    def test_manager_info(self):
        """Test getting collection info."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MomexConfig(storage_path=tmpdir)
            manager = MemoryManager(config=config)

            # Create collection
            db_dir = os.path.join(tmpdir, "momex", "xiaoyuzhang")
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, "memory.db")
            with open(db_path, "w") as f:
                f.write("test data")

            info = manager.info("momex:xiaoyuzhang")
            assert info["collection"] == "momex:xiaoyuzhang"
            assert "size" in info
            assert "db_path" in info

    def test_manager_list_collections(self):
        """Test listing multiple collections."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MomexConfig(storage_path=tmpdir)
            manager = MemoryManager(config=config)

            # Create hierarchical collections
            for parts in [
                ("momex", "engineering", "xiaoyuzhang"),
                ("momex", "engineering", "gvanrossum"),
                ("momex", "marketing", "charlie"),
            ]:
                db_dir = os.path.join(tmpdir, *parts)
                os.makedirs(db_dir, exist_ok=True)
                db_path = os.path.join(db_dir, "memory.db")
                with open(db_path, "w") as f:
                    f.write("")

            collections = manager.list_collections()
            assert len(collections) == 3
            assert "momex:engineering:xiaoyuzhang" in collections
            assert "momex:engineering:gvanrossum" in collections
            assert "momex:marketing:charlie" in collections

    def test_manager_list_collections_with_prefix(self):
        """Test listing collections with prefix filter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MomexConfig(storage_path=tmpdir)
            manager = MemoryManager(config=config)

            # Create hierarchical collections
            for parts in [
                ("momex", "engineering", "xiaoyuzhang"),
                ("momex", "engineering", "gvanrossum"),
                ("momex", "marketing", "charlie"),
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

            # Filter by momex
            momex_collections = manager.list_collections(prefix="momex")
            assert len(momex_collections) == 3
            assert "momex:engineering:xiaoyuzhang" in momex_collections
            assert "momex:engineering:gvanrossum" in momex_collections
            assert "momex:marketing:charlie" in momex_collections
            assert "other:team:dave" not in momex_collections

            # Filter by momex:engineering
            eng_collections = manager.list_collections(prefix="momex:engineering")
            assert len(eng_collections) == 2
            assert "momex:engineering:xiaoyuzhang" in eng_collections
            assert "momex:engineering:gvanrossum" in eng_collections
            assert "momex:marketing:charlie" not in eng_collections

            # Filter by exact match
            exact_collections = manager.list_collections(prefix="momex:engineering:xiaoyuzhang")
            assert len(exact_collections) == 1
            assert "momex:engineering:xiaoyuzhang" in exact_collections

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
            config = MomexConfig(storage_path=tmpdir)
            memory = Memory(collection="momex:engineering:xiaoyuzhang", config=config)

            result = await memory.add_async(
                "Xiaoyuzhang likes cats",
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
            config = MomexConfig(storage_path=tmpdir)
            memory = Memory(collection="momex:engineering:xiaoyuzhang", config=config)

            await memory.add_async("Alice is a software engineer")
            await memory.add_async("Bob is a data scientist")

            results = await memory.search_async("Alice")
            assert len(results) >= 1
            assert any("Alice" in r.text for r in results)

    @pytest.mark.skip(reason="Requires LLM API key")
    async def test_stats(self):
        """Test memory statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MomexConfig(storage_path=tmpdir)
            memory = Memory(collection="momex:engineering:xiaoyuzhang", config=config)

            await memory.add_async("Test content")
            stats = await memory.stats_async()

            assert "total_memories" in stats
            assert stats["collection"] == "momex:engineering:xiaoyuzhang"


@pytest.mark.asyncio
class TestPrefixQueryAsync:
    """Async tests for prefix query functions (require LLM)."""

    @pytest.mark.skip(reason="Requires LLM API key")
    async def test_query_single_collection(self):
        """Test querying a single collection by exact prefix."""
        from momex import query_async

        with tempfile.TemporaryDirectory() as tmpdir:
            config = MomexConfig(storage_path=tmpdir)

            # Add memory
            memory = Memory(collection="momex:engineering:xiaoyuzhang", config=config)
            await memory.add_async("Xiaoyuzhang likes Python")

            # Query exact collection
            answer = await query_async("momex:engineering:xiaoyuzhang", "What does Alice like?", config=config)
            assert "Python" in answer

    @pytest.mark.skip(reason="Requires LLM API key")
    async def test_query_prefix_multiple(self):
        """Test querying multiple collections by prefix."""
        from momex import query_async

        with tempfile.TemporaryDirectory() as tmpdir:
            config = MomexConfig(storage_path=tmpdir)

            # Add memories to different collections
            xiaoyuzhang = Memory(collection="momex:engineering:xiaoyuzhang", config=config)
            await xiaoyuzhang.add_async("Xiaoyuzhang likes Python")

            gvanrossum = Memory(collection="momex:engineering:gvanrossum", config=config)
            await gvanrossum.add_async("Bob likes Java")

            # Query by prefix - should find both
            answer = await query_async("momex:engineering", "What programming languages?", config=config)
            assert "Python" in answer or "Java" in answer
