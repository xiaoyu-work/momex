"""Tests for Memex Memory class."""

import os
import tempfile

import pytest

from memex import AddResult, Memory, MemexConfig, MemoryItem


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

    def test_get_db_path_default(self):
        """Test database path generation with no tenant info."""
        config = MemexConfig(storage_path="/data")
        path = config.get_db_path()
        assert path == os.path.join("/data", "default", "memory.db")

    def test_get_db_path_with_user(self):
        """Test database path generation with user_id."""
        config = MemexConfig(storage_path="/data")
        path = config.get_db_path(user_id="user_123")
        assert "user_123" in path
        assert path.endswith("memory.db")

    def test_get_db_path_with_org_and_user(self):
        """Test database path generation with org and user."""
        config = MemexConfig(storage_path="/data")
        path = config.get_db_path(org_id="acme", user_id="user_123")
        assert "acme" in path
        assert "user_123" in path


class TestMemory:
    """Tests for Memory class."""

    def test_memory_init_default(self):
        """Test Memory initialization with defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemexConfig(storage_path=tmpdir)
            memory = Memory(config=config)
            assert memory.user_id is None
            assert memory.config is not None
            assert not memory.is_initialized

    def test_memory_init_with_tenant(self):
        """Test Memory initialization with tenant info."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemexConfig(storage_path=tmpdir)
            memory = Memory(
                user_id="user_123",
                org_id="acme",
                config=config,
            )
            assert memory.user_id == "user_123"
            assert memory.org_id == "acme"
            assert "user_123" in memory.db_path
            assert "acme" in memory.db_path

    def test_memory_init_with_direct_path(self):
        """Test Memory initialization with direct db_path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "custom.db")
            memory = Memory(db_path=db_path)
            assert memory.db_path == db_path


@pytest.mark.asyncio
class TestMemoryAsync:
    """Async tests for Memory class (require LLM)."""

    @pytest.mark.skip(reason="Requires LLM API key")
    async def test_add_and_query(self):
        """Test adding and querying memories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemexConfig(storage_path=tmpdir)
            memory = Memory(user_id="test_user", config=config)

            # Add memory
            result = await memory.add_async(
                "张三说下周五完成API开发",
                speaker="记录者",
            )
            assert result.success
            assert result.messages_added == 1

            # Query
            answer = await memory.query_async("谁负责API?")
            assert "张三" in answer

    @pytest.mark.skip(reason="Requires LLM API key")
    async def test_search(self):
        """Test searching memories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemexConfig(storage_path=tmpdir)
            memory = Memory(user_id="test_user", config=config)

            await memory.add_async("张三是前端工程师")
            await memory.add_async("李四是后端工程师")

            results = await memory.search_async("张三")
            assert len(results) >= 1
            assert any("张三" in r.text for r in results)

    @pytest.mark.skip(reason="Requires LLM API key")
    async def test_stats(self):
        """Test memory statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemexConfig(storage_path=tmpdir)
            memory = Memory(user_id="test_user", config=config)

            await memory.add_async("测试内容")
            stats = await memory.stats_async()

            assert "total_memories" in stats
            assert stats["user_id"] == "test_user"

    @pytest.mark.skip(reason="Requires LLM API key")
    async def test_export(self):
        """Test exporting memories."""
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemexConfig(storage_path=tmpdir)
            memory = Memory(user_id="test_user", config=config)

            await memory.add_async("测试内容1")
            await memory.add_async("测试内容2")

            export_path = os.path.join(tmpdir, "export.json")
            await memory.export_async(export_path)

            with open(export_path) as f:
                data = json.load(f)

            assert data["user_id"] == "test_user"
            assert len(data["memories"]) == 2

    @pytest.mark.skip(reason="Requires LLM API key")
    async def test_clear(self):
        """Test clearing memories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemexConfig(storage_path=tmpdir)
            memory = Memory(user_id="test_user", config=config)

            await memory.add_async("测试内容")
            assert memory.is_initialized

            await memory.clear_async()
            assert not memory.is_initialized
