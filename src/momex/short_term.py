"""Short-term memory with in-memory cache and database persistence.

Provides session-based conversation history that survives restarts.
Uses the same storage backend (SQLite/PostgreSQL) as long-term memory.

Example:
    >>> from momex import ShortTermMemory, MomexConfig
    >>>
    >>> short_term = ShortTermMemory("user:xiaoyuzhang", config)
    >>>
    >>> # Add messages
    >>> short_term.add("Hello, I'm Alice", role="user")
    >>> short_term.add("Nice to meet you, Alice!", role="assistant")
    >>>
    >>> # Get recent messages
    >>> messages = short_term.get(limit=10)
    >>>
    >>> # Session management
    >>> session_id = short_term.session_id
    >>> short_term.new_session()  # Start fresh
    >>> short_term.load_session(session_id)  # Resume previous
"""

from __future__ import annotations

import re
import sqlite3
import uuid
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, Any, Generator

from .config import MomexConfig


@dataclass
class Message:
    """A single message in short-term memory."""

    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    id: int | None = None  # Database ID, set after persistence


@dataclass
class SessionInfo:
    """Information about a session."""

    session_id: str
    started_at: str
    last_message_at: str
    message_count: int


def _collection_to_db_path(collection: str, base_path: str, db_name: str) -> Path:
    """Convert collection name to database path."""
    parts = collection.split(":")
    sanitized = [re.sub(r'[<>"|?*:\\]', "_", part) for part in parts]
    return Path(base_path) / Path(*sanitized) / db_name


class ShortTermMemory:
    """Short-term memory with cache and persistence.

    Maintains conversation history with:
    - In-memory deque for fast access
    - SQLite persistence for durability (connections opened/closed on demand)
    - Session-based organization
    - Automatic expiration cleanup

    No manual close() needed - database connections are managed automatically.

    Example:
        >>> stm = ShortTermMemory("user:alice", config)
        >>> stm.add("I like coffee", role="user")
        >>> stm.add("Good to know!", role="assistant")
        >>>
        >>> # Get history
        >>> for msg in stm.get(limit=10):
        ...     print(f"{msg.role}: {msg.content}")
        >>>
        >>> # Restart app, resume session
        >>> stm2 = ShortTermMemory("user:alice", config, session_id=stm.session_id)
        >>> stm2.get()  # Previous messages restored
    """

    def __init__(
        self,
        collection: str,
        config: MomexConfig | None = None,
        *,
        session_id: str | None = None,
        max_messages: int = 100,
        session_ttl_hours: int = 24,
    ):
        """Initialize short-term memory.

        Args:
            collection: Collection name (e.g., "user:xiaoyuzhang").
                        Stored in same directory as long-term memory.
            config: Momex configuration. Uses default if None.
            session_id: Resume existing session. Creates new if None.
            max_messages: Maximum messages to keep in memory cache.
            session_ttl_hours: Hours before sessions expire (for cleanup).
        """
        self.collection = collection
        self.config = config or MomexConfig.get_default()
        self.max_messages = max_messages
        self.session_ttl_hours = session_ttl_hours

        # Session ID
        self._session_id = session_id or str(uuid.uuid4())

        # In-memory cache
        self._cache: deque[Message] = deque(maxlen=max_messages)

        # Database path
        self._db_path = _collection_to_db_path(
            collection,
            self.config.storage_path,
            "short_term.db",
        )
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database schema and load session
        self._init_db()
        self._load_from_db()

    @contextmanager
    def _get_db(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection (opened and closed automatically)."""
        conn = sqlite3.connect(self._db_path)
        try:
            yield conn
        finally:
            conn.close()

    @property
    def session_id(self) -> str:
        """Current session ID."""
        return self._session_id

    @property
    def db_path(self) -> str:
        """Database file path."""
        return str(self._db_path)

    # =========================================================================
    # Core API
    # =========================================================================

    def add(
        self,
        content: str,
        role: Literal["user", "assistant", "system"] = "user",
    ) -> Message:
        """Add a message to short-term memory.

        Writes to both in-memory cache and database.

        Args:
            content: Message content.
            role: Message role (user, assistant, or system).

        Returns:
            The created Message with database ID.
        """
        message = Message(role=role, content=content)

        # Persist to database
        self._save_to_db(message)

        # Add to cache
        self._cache.append(message)

        return message

    def get(self, limit: int = 20) -> list[Message]:
        """Get recent messages from current session.

        Args:
            limit: Maximum number of messages to return.

        Returns:
            List of messages, oldest first.
        """
        return list(self._cache)[-limit:]

    def get_all(self) -> list[Message]:
        """Get all messages from current session.

        Returns:
            List of all messages, oldest first.
        """
        return list(self._cache)

    def clear(self) -> None:
        """Clear all messages in current session."""
        with self._get_db() as conn:
            conn.execute(
                "DELETE FROM ShortTermMessages WHERE session_id = ?",
                (self._session_id,),
            )
            conn.commit()
        self._cache.clear()

    def stats(self) -> dict[str, Any]:
        """Get statistics for current session.

        Returns:
            Dict with message counts and session info.
        """
        with self._get_db() as conn:
            # Current session stats
            row = conn.execute(
                """
                SELECT COUNT(*), MIN(timestamp), MAX(timestamp)
                FROM ShortTermMessages
                WHERE session_id = ?
                """,
                (self._session_id,),
            ).fetchone()

            # Total sessions
            total_sessions = conn.execute(
                "SELECT COUNT(DISTINCT session_id) FROM ShortTermMessages"
            ).fetchone()[0]

            # Total messages across all sessions
            total_messages = conn.execute(
                "SELECT COUNT(*) FROM ShortTermMessages"
            ).fetchone()[0]

        return {
            "collection": self.collection,
            "session_id": self._session_id,
            "message_count": row[0] or 0,
            "started_at": row[1],
            "last_message_at": row[2],
            "cache_size": len(self._cache),
            "total_sessions": total_sessions,
            "total_messages": total_messages,
        }

    # =========================================================================
    # Session Management
    # =========================================================================

    def new_session(self) -> str:
        """Start a new session.

        Returns:
            The new session ID.
        """
        self._cache.clear()
        self._session_id = str(uuid.uuid4())
        return self._session_id

    def load_session(self, session_id: str) -> bool:
        """Load an existing session.

        Args:
            session_id: Session ID to load.

        Returns:
            True if session exists and was loaded, False otherwise.
        """
        with self._get_db() as conn:
            exists = conn.execute(
                "SELECT 1 FROM ShortTermMessages WHERE session_id = ? LIMIT 1",
                (session_id,),
            ).fetchone()

        if not exists:
            return False

        self._session_id = session_id
        self._cache.clear()
        self._load_from_db()
        return True

    def list_sessions(self, limit: int = 50) -> list[SessionInfo]:
        """List all sessions, newest first.

        Args:
            limit: Maximum number of sessions to return.

        Returns:
            List of SessionInfo objects.
        """
        with self._get_db() as conn:
            rows = conn.execute(
                """
                SELECT session_id, MIN(timestamp), MAX(timestamp), COUNT(*)
                FROM ShortTermMessages
                GROUP BY session_id
                ORDER BY MAX(timestamp) DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [
            SessionInfo(
                session_id=row[0],
                started_at=row[1],
                last_message_at=row[2],
                message_count=row[3],
            )
            for row in rows
        ]

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages.

        Args:
            session_id: Session ID to delete.

        Returns:
            True if session was deleted, False if not found.
        """
        with self._get_db() as conn:
            cursor = conn.execute(
                "DELETE FROM ShortTermMessages WHERE session_id = ?",
                (session_id,),
            )
            conn.commit()
            deleted = cursor.rowcount > 0

        # Clear cache if deleting current session
        if session_id == self._session_id:
            self._cache.clear()
            self._session_id = str(uuid.uuid4())

        return deleted

    def cleanup_expired(self) -> int:
        """Remove sessions older than session_ttl_hours.

        Returns:
            Number of messages deleted.
        """
        cutoff = (
            datetime.now(timezone.utc) - timedelta(hours=self.session_ttl_hours)
        ).isoformat()

        with self._get_db() as conn:
            cursor = conn.execute(
                "DELETE FROM ShortTermMessages WHERE timestamp < ?",
                (cutoff,),
            )
            conn.commit()
            return cursor.rowcount

    # =========================================================================
    # Database Operations (Internal)
    # =========================================================================

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_db() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ShortTermMessages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_stm_session
                ON ShortTermMessages(session_id)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_stm_timestamp
                ON ShortTermMessages(timestamp)
                """
            )
            conn.commit()

    def _load_from_db(self) -> None:
        """Load current session from database into cache."""
        with self._get_db() as conn:
            rows = conn.execute(
                """
                SELECT id, role, content, timestamp
                FROM ShortTermMessages
                WHERE session_id = ?
                ORDER BY timestamp ASC
                LIMIT ?
                """,
                (self._session_id, self.max_messages),
            ).fetchall()

        self._cache.clear()
        for row in rows:
            self._cache.append(
                Message(id=row[0], role=row[1], content=row[2], timestamp=row[3])
            )

    def _save_to_db(self, message: Message) -> None:
        """Save a message to database."""
        with self._get_db() as conn:
            cursor = conn.execute(
                """
                INSERT INTO ShortTermMessages (session_id, role, content, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                (self._session_id, message.role, message.content, message.timestamp),
            )
            conn.commit()
            message.id = cursor.lastrowid
