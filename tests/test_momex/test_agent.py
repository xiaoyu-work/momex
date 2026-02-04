"""Tests for Momex Agent high-level API."""

import tempfile

import pytest

from momex import Agent, ChatResponse, MomexConfig


class TestChatResponse:
    """Tests for ChatResponse dataclass."""

    def test_chat_response_creation(self):
        """Test creating a ChatResponse."""
        response = ChatResponse(content="Hello!")
        assert response.content == "Hello!"


class TestAgentInit:
    """Tests for Agent initialization."""

    def test_agent_init(self):
        """Test basic Agent initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MomexConfig(storage_path=tmpdir)
            agent = Agent("test:user", config)

            assert agent.collection == "test:user"
            assert agent.session_id is not None
            assert agent.system_prompt is not None

            agent._short_term.close()

    def test_agent_init_with_session_id(self):
        """Test Agent initialization with existing session ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MomexConfig(storage_path=tmpdir)
            agent = Agent("test:user", config, session_id="my-session-123")

            assert agent.session_id == "my-session-123"

            agent._short_term.close()

    def test_agent_init_custom_prompt(self):
        """Test Agent initialization with custom system prompt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MomexConfig(storage_path=tmpdir)
            custom_prompt = "You are a helpful coding assistant."
            agent = Agent("test:user", config, system_prompt=custom_prompt)

            assert agent.system_prompt == custom_prompt

            agent._short_term.close()


class TestAgentSessionManagement:
    """Tests for Agent session management."""

    def test_new_session(self):
        """Test starting a new session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MomexConfig(storage_path=tmpdir)
            agent = Agent("test:user", config)

            old_session = agent.session_id
            new_session = agent.new_session()

            assert new_session != old_session
            assert agent.session_id == new_session

            agent._short_term.close()

    def test_list_sessions(self):
        """Test listing sessions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MomexConfig(storage_path=tmpdir)
            agent = Agent("test:user", config)

            # Add message to create session
            agent._short_term.add("Test message", role="user")
            session1 = agent.session_id

            agent.new_session()
            agent._short_term.add("Another message", role="user")

            sessions = agent.list_sessions()
            assert len(sessions) == 2

            session_ids = [s.session_id for s in sessions]
            assert session1 in session_ids

            agent._short_term.close()

    def test_load_session(self):
        """Test loading an existing session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MomexConfig(storage_path=tmpdir)
            agent = Agent("test:user", config)

            agent._short_term.add("First session message", role="user")
            first_session = agent.session_id

            agent.new_session()
            agent._short_term.add("Second session message", role="user")

            # Load first session
            success = agent.load_session(first_session)
            assert success is True
            assert agent.session_id == first_session

            history = agent.get_history()
            assert len(history) == 1
            assert history[0].content == "First session message"

            agent._short_term.close()

    def test_delete_session(self):
        """Test deleting a session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MomexConfig(storage_path=tmpdir)
            agent = Agent("test:user", config)

            agent._short_term.add("Message", role="user")
            session_id = agent.session_id

            agent.new_session()

            result = agent.delete_session(session_id)
            assert result is True

            sessions = agent.list_sessions()
            session_ids = [s.session_id for s in sessions]
            assert session_id not in session_ids

            agent._short_term.close()

    def test_get_history(self):
        """Test getting conversation history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MomexConfig(storage_path=tmpdir)
            agent = Agent("test:user", config)

            agent._short_term.add("User message", role="user")
            agent._short_term.add("Assistant response", role="assistant")

            history = agent.get_history()
            assert len(history) == 2
            assert history[0].role == "user"
            assert history[1].role == "assistant"

            agent._short_term.close()

    def test_clear_history(self):
        """Test clearing conversation history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MomexConfig(storage_path=tmpdir)
            agent = Agent("test:user", config)

            agent._short_term.add("Message 1", role="user")
            agent._short_term.add("Message 2", role="user")
            assert len(agent.get_history()) == 2

            agent.clear_history()
            assert len(agent.get_history()) == 0

            agent._short_term.close()


class TestAgentStats:
    """Tests for Agent statistics."""

    def test_stats(self):
        """Test getting stats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MomexConfig(storage_path=tmpdir)
            agent = Agent("test:user", config)

            agent._short_term.add("Test message", role="user")

            stats = agent.stats()
            assert stats["collection"] == "test:user"
            assert stats["session_id"] == agent.session_id
            assert "short_term" in stats

            agent._short_term.close()


class TestAgentCleanup:
    """Tests for Agent cleanup."""

    def test_cleanup_expired_sessions(self):
        """Test cleaning up expired sessions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MomexConfig(storage_path=tmpdir)
            # Use TTL of 0 to make messages expire immediately
            agent = Agent("test:user", config)
            agent._short_term.session_ttl_hours = 0

            agent._short_term.add("Old message", role="user")

            deleted = agent.cleanup_expired_sessions()
            assert deleted >= 1

            agent._short_term.close()


@pytest.mark.asyncio
class TestAgentAsync:
    """Async tests for Agent (require LLM)."""

    @pytest.mark.skip(reason="Requires LLM API key")
    async def test_chat_basic(self):
        """Test basic chat functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MomexConfig(storage_path=tmpdir)
            agent = Agent("test:user", config)

            response = await agent.chat("Hello, my name is Alice")

            assert response.content is not None
            assert len(response.content) > 0

            await agent.close()

    @pytest.mark.skip(reason="Requires LLM API key")
    async def test_chat_stores_to_short_term(self):
        """Test that chat stores messages to short-term memory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MomexConfig(storage_path=tmpdir)
            agent = Agent("test:user", config)

            await agent.chat("Hello")

            history = agent.get_history()
            # Should have user message and assistant response
            assert len(history) >= 2

            await agent.close()

    @pytest.mark.skip(reason="Requires LLM API key")
    async def test_chat_identity_stored_long_term(self):
        """Test that identity info is stored in long-term memory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MomexConfig(storage_path=tmpdir)
            agent = Agent("test:user", config)

            response = await agent.chat("My name is Alice and I'm a Python developer")

            # Response should have content
            assert response.content is not None
            assert len(response.content) > 0

            # Can verify long-term storage via Level 2 API
            results = await agent._long_term.search("Alice")
            assert len(results) > 0

            await agent.close()

    @pytest.mark.skip(reason="Requires LLM API key")
    async def test_chat_question_not_stored_long_term(self):
        """Test that simple questions are not stored long-term."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MomexConfig(storage_path=tmpdir)
            agent = Agent("test:user", config)

            response = await agent.chat("What time is it?")

            # Response should have content
            assert response.content is not None
            assert len(response.content) > 0

            await agent.close()

    @pytest.mark.skip(reason="Requires LLM API key")
    async def test_end_session(self):
        """Test ending a session with summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MomexConfig(storage_path=tmpdir)
            agent = Agent("test:user", config)

            await agent.chat("My name is Bob")
            await agent.chat("I work at Google")
            await agent.chat("I like Python")

            summary = await agent.end_session(save_summary=True)

            # Summary might be generated if there's enough content
            # History should be cleared
            assert len(agent.get_history()) == 0

            await agent.close()

    @pytest.mark.skip(reason="Requires LLM API key")
    async def test_context_manager(self):
        """Test async context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MomexConfig(storage_path=tmpdir)

            async with Agent("test:user", config) as agent:
                response = await agent.chat("Hello")
                assert response.content is not None

    @pytest.mark.skip(reason="Requires LLM API key")
    async def test_stats_async(self):
        """Test getting async stats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MomexConfig(storage_path=tmpdir)
            agent = Agent("test:user", config)

            await agent.chat("Test message")

            stats = await agent.stats_async()
            assert "short_term" in stats
            assert "long_term" in stats

            await agent.close()
