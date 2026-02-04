"""Tests for Momex ShortTermMemory."""

import os
import tempfile

import pytest

from momex import ShortTermMemory, Message, SessionInfo, MomexConfig


class TestMessage:
    """Tests for Message dataclass."""

    def test_message_creation(self):
        """Test creating a message with defaults."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.timestamp is not None
        assert msg.id is None

    def test_message_with_all_fields(self):
        """Test creating a message with all fields."""
        msg = Message(
            role="assistant",
            content="Hi there",
            timestamp="2024-01-01T00:00:00Z",
            id=42,
        )
        assert msg.role == "assistant"
        assert msg.content == "Hi there"
        assert msg.timestamp == "2024-01-01T00:00:00Z"
        assert msg.id == 42


class TestSessionInfo:
    """Tests for SessionInfo dataclass."""

    def test_session_info_creation(self):
        """Test creating session info."""
        info = SessionInfo(
            session_id="abc-123",
            started_at="2024-01-01T00:00:00Z",
            last_message_at="2024-01-01T01:00:00Z",
            message_count=5,
        )
        assert info.session_id == "abc-123"
        assert info.message_count == 5


class TestShortTermMemory:
    """Tests for ShortTermMemory class."""

    def test_init_creates_db(self):
        """Test initialization creates database file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MomexConfig(storage_path=tmpdir)
            with ShortTermMemory("test:user", config) as stm:
                assert os.path.exists(stm.db_path)
                assert stm.session_id is not None
                assert stm.collection == "test:user"

    def test_init_with_session_id(self):
        """Test initialization with existing session ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MomexConfig(storage_path=tmpdir)
            with ShortTermMemory("test:user", config, session_id="my-session-123") as stm:
                assert stm.session_id == "my-session-123"

    def test_add_message(self):
        """Test adding a message."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MomexConfig(storage_path=tmpdir)
            with ShortTermMemory("test:user", config) as stm:
                msg = stm.add("Hello world", role="user")

                assert msg.content == "Hello world"
                assert msg.role == "user"
                assert msg.id is not None  # DB ID assigned

    def test_add_multiple_messages(self):
        """Test adding multiple messages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MomexConfig(storage_path=tmpdir)
            with ShortTermMemory("test:user", config) as stm:
                stm.add("Message 1", role="user")
                stm.add("Message 2", role="assistant")
                stm.add("Message 3", role="user")

                messages = stm.get_all()
                assert len(messages) == 3
                assert messages[0].content == "Message 1"
                assert messages[1].content == "Message 2"
                assert messages[2].content == "Message 3"

    def test_get_with_limit(self):
        """Test getting messages with limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MomexConfig(storage_path=tmpdir)
            with ShortTermMemory("test:user", config) as stm:
                for i in range(10):
                    stm.add(f"Message {i}", role="user")

                messages = stm.get(limit=3)
                assert len(messages) == 3
                # Should return last 3 messages
                assert messages[0].content == "Message 7"
                assert messages[1].content == "Message 8"
                assert messages[2].content == "Message 9"

    def test_clear(self):
        """Test clearing messages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MomexConfig(storage_path=tmpdir)
            with ShortTermMemory("test:user", config) as stm:
                stm.add("Message 1", role="user")
                stm.add("Message 2", role="user")
                assert len(stm.get_all()) == 2

                stm.clear()
                assert len(stm.get_all()) == 0

    def test_stats(self):
        """Test getting statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MomexConfig(storage_path=tmpdir)
            with ShortTermMemory("test:user", config) as stm:
                stm.add("Message 1", role="user")
                stm.add("Message 2", role="assistant")

                stats = stm.stats()
                assert stats["collection"] == "test:user"
                assert stats["session_id"] == stm.session_id
                assert stats["message_count"] == 2
                assert stats["cache_size"] == 2
                assert stats["total_sessions"] == 1
                assert stats["total_messages"] == 2

    def test_max_messages_limit(self):
        """Test max_messages limit on cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MomexConfig(storage_path=tmpdir)
            with ShortTermMemory("test:user", config, max_messages=5) as stm:
                for i in range(10):
                    stm.add(f"Message {i}", role="user")

                # Cache should only have last 5
                messages = stm.get_all()
                assert len(messages) == 5
                assert messages[0].content == "Message 5"
                assert messages[4].content == "Message 9"


class TestShortTermMemorySessionManagement:
    """Tests for session management in ShortTermMemory."""

    def test_new_session(self):
        """Test starting a new session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MomexConfig(storage_path=tmpdir)
            with ShortTermMemory("test:user", config) as stm:
                stm.add("Message in session 1", role="user")
                old_session_id = stm.session_id

                new_session_id = stm.new_session()

                assert new_session_id != old_session_id
                assert stm.session_id == new_session_id
                assert len(stm.get_all()) == 0  # New session is empty

    def test_load_session(self):
        """Test loading an existing session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MomexConfig(storage_path=tmpdir)
            with ShortTermMemory("test:user", config) as stm:
                # Create first session with messages
                stm.add("Message 1", role="user")
                stm.add("Message 2", role="assistant")
                first_session_id = stm.session_id

                # Start new session
                stm.new_session()
                stm.add("Message in new session", role="user")
                assert len(stm.get_all()) == 1

                # Load first session
                success = stm.load_session(first_session_id)
                assert success is True
                assert stm.session_id == first_session_id

                messages = stm.get_all()
                assert len(messages) == 2
                assert messages[0].content == "Message 1"
                assert messages[1].content == "Message 2"

    def test_load_nonexistent_session(self):
        """Test loading a session that doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MomexConfig(storage_path=tmpdir)
            with ShortTermMemory("test:user", config) as stm:
                success = stm.load_session("nonexistent-session-id")
                assert success is False

    def test_list_sessions(self):
        """Test listing all sessions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MomexConfig(storage_path=tmpdir)
            with ShortTermMemory("test:user", config) as stm:
                # Create multiple sessions
                stm.add("Session 1 message", role="user")
                session1_id = stm.session_id

                stm.new_session()
                stm.add("Session 2 message 1", role="user")
                stm.add("Session 2 message 2", role="assistant")
                session2_id = stm.session_id

                stm.new_session()
                stm.add("Session 3 message", role="user")

                sessions = stm.list_sessions()
                assert len(sessions) == 3

                # Sessions should be ordered by last_message_at DESC
                session_ids = [s.session_id for s in sessions]
                assert session1_id in session_ids
                assert session2_id in session_ids

                # Find session 2 and check message count
                session2 = next(s for s in sessions if s.session_id == session2_id)
                assert session2.message_count == 2

    def test_delete_session(self):
        """Test deleting a session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MomexConfig(storage_path=tmpdir)
            with ShortTermMemory("test:user", config) as stm:
                stm.add("Message", role="user")
                session_id = stm.session_id

                stm.new_session()

                # Delete the first session
                result = stm.delete_session(session_id)
                assert result is True

                # Verify it's gone
                sessions = stm.list_sessions()
                session_ids = [s.session_id for s in sessions]
                assert session_id not in session_ids

    def test_delete_current_session(self):
        """Test deleting the current session starts a new one."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MomexConfig(storage_path=tmpdir)
            with ShortTermMemory("test:user", config) as stm:
                stm.add("Message", role="user")
                old_session_id = stm.session_id

                stm.delete_session(old_session_id)

                # Should have new session ID
                assert stm.session_id != old_session_id
                assert len(stm.get_all()) == 0

    def test_delete_nonexistent_session(self):
        """Test deleting a session that doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MomexConfig(storage_path=tmpdir)
            with ShortTermMemory("test:user", config) as stm:
                result = stm.delete_session("nonexistent-session-id")
                assert result is False


class TestShortTermMemoryPersistence:
    """Tests for persistence and recovery."""

    def test_persistence_across_instances(self):
        """Test that messages persist across instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MomexConfig(storage_path=tmpdir)

            # Create first instance and add messages
            with ShortTermMemory("test:user", config) as stm1:
                stm1.add("Persisted message 1", role="user")
                stm1.add("Persisted message 2", role="assistant")
                session_id = stm1.session_id

            # Create second instance with same session ID
            with ShortTermMemory("test:user", config, session_id=session_id) as stm2:
                messages = stm2.get_all()
                assert len(messages) == 2
                assert messages[0].content == "Persisted message 1"
                assert messages[1].content == "Persisted message 2"

    def test_cleanup_expired(self):
        """Test cleaning up expired sessions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MomexConfig(storage_path=tmpdir)

            # Create with very short TTL
            with ShortTermMemory("test:user", config, session_ttl_hours=0) as stm:
                stm.add("Old message", role="user")

                # Cleanup should delete the message (TTL = 0 hours)
                deleted = stm.cleanup_expired()
                assert deleted >= 1

    def test_db_path_hierarchical(self):
        """Test database path with hierarchical collection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MomexConfig(storage_path=tmpdir)
            with ShortTermMemory("org:team:user", config) as stm:
                assert "org" in stm.db_path
                assert "team" in stm.db_path
                assert "user" in stm.db_path
                assert stm.db_path.endswith("short_term.db")
