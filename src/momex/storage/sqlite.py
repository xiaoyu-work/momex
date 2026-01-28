"""SQLite storage backend for Momex.

A standalone SQLite implementation that stores memories with embeddings.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from .base import SearchResult, StorageBackend, StorageRecord


class SQLiteBackend(StorageBackend):
    """SQLite storage backend with local file storage.

    Stores memories in a SQLite database with embeddings as BLOB.
    Vector similarity search is performed in Python using numpy.

    Args:
        db_path: Path to the SQLite database file.
        embedding_dim: Dimension of embedding vectors (default 1536 for OpenAI).

    Example:
        backend = SQLiteBackend("./data/memory.db")
        await backend.initialize()
        record_id = await backend.add("Hello", embedding=[0.1, 0.2, ...])
    """

    def __init__(
        self,
        db_path: str | Path,
        embedding_dim: int = 1536,
    ) -> None:
        self.db_path = Path(db_path)
        self.embedding_dim = embedding_dim
        self._conn: sqlite3.Connection | None = None

    async def initialize(self) -> None:
        """Initialize the database and create tables."""
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row

        # Enable WAL mode for better concurrent performance
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")

        # Create tables
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                speaker TEXT,
                timestamp TEXT,
                importance REAL DEFAULT 0.5,
                deleted INTEGER DEFAULT 0,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS embeddings (
                memory_id INTEGER PRIMARY KEY,
                embedding BLOB NOT NULL,
                FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_memories_deleted ON memories(deleted);
            CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories(timestamp);
        """)
        self._conn.commit()

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def _ensure_connected(self) -> sqlite3.Connection:
        """Ensure database connection is open."""
        if self._conn is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self._conn

    def _embedding_to_blob(self, embedding: list[float]) -> bytes:
        """Convert embedding list to binary blob."""
        return np.array(embedding, dtype=np.float32).tobytes()

    def _blob_to_embedding(self, blob: bytes) -> np.ndarray:
        """Convert binary blob to numpy array."""
        return np.frombuffer(blob, dtype=np.float32)

    def _row_to_record(self, row: sqlite3.Row) -> StorageRecord:
        """Convert database row to StorageRecord."""
        metadata = None
        if row["metadata"]:
            try:
                metadata = json.loads(row["metadata"])
            except json.JSONDecodeError:
                pass

        return StorageRecord(
            id=row["id"],
            text=row["text"],
            speaker=row["speaker"],
            timestamp=row["timestamp"],
            importance=row["importance"],
            deleted=bool(row["deleted"]),
            metadata=metadata,
        )

    async def add(
        self,
        text: str,
        embedding: list[float],
        speaker: str | None = None,
        timestamp: str | None = None,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Add a new memory record."""
        conn = self._ensure_connected()

        metadata_json = json.dumps(metadata) if metadata else None

        cursor = conn.execute(
            """
            INSERT INTO memories (text, speaker, timestamp, importance, metadata)
            VALUES (?, ?, ?, ?, ?)
            """,
            (text, speaker, timestamp, importance, metadata_json),
        )
        record_id = cursor.lastrowid

        # Store embedding
        embedding_blob = self._embedding_to_blob(embedding)
        conn.execute(
            "INSERT INTO embeddings (memory_id, embedding) VALUES (?, ?)",
            (record_id, embedding_blob),
        )

        conn.commit()
        return record_id

    async def get(self, record_id: int) -> StorageRecord | None:
        """Get a memory record by ID."""
        conn = self._ensure_connected()

        cursor = conn.execute(
            "SELECT * FROM memories WHERE id = ?",
            (record_id,),
        )
        row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_record(row)

    async def search(
        self,
        embedding: list[float],
        limit: int = 10,
        threshold: float = 0.3,
    ) -> list[SearchResult]:
        """Search for similar memories using cosine similarity."""
        conn = self._ensure_connected()

        # Get all non-deleted memories with embeddings
        cursor = conn.execute(
            """
            SELECT m.*, e.embedding
            FROM memories m
            JOIN embeddings e ON m.id = e.memory_id
            WHERE m.deleted = 0
            """
        )
        rows = cursor.fetchall()

        if not rows:
            return []

        # Calculate cosine similarity for each memory
        query_embedding = np.array(embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return []
        query_embedding = query_embedding / query_norm

        results: list[tuple[StorageRecord, float]] = []
        for row in rows:
            mem_embedding = self._blob_to_embedding(row["embedding"])
            mem_norm = np.linalg.norm(mem_embedding)
            if mem_norm == 0:
                continue
            mem_embedding = mem_embedding / mem_norm

            # Cosine similarity
            similarity = float(np.dot(query_embedding, mem_embedding))

            if similarity >= threshold:
                record = self._row_to_record(row)
                results.append((record, similarity))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)

        # Return top results
        return [
            SearchResult(record=record, similarity=similarity)
            for record, similarity in results[:limit]
        ]

    async def delete(self, record_id: int) -> bool:
        """Soft delete a memory."""
        conn = self._ensure_connected()

        cursor = conn.execute(
            "UPDATE memories SET deleted = 1 WHERE id = ? AND deleted = 0",
            (record_id,),
        )
        conn.commit()

        return cursor.rowcount > 0

    async def restore(self, record_id: int) -> bool:
        """Restore a soft-deleted memory."""
        conn = self._ensure_connected()

        cursor = conn.execute(
            "UPDATE memories SET deleted = 0 WHERE id = ? AND deleted = 1",
            (record_id,),
        )
        conn.commit()

        return cursor.rowcount > 0

    async def list_deleted(self) -> list[StorageRecord]:
        """List all soft-deleted memories."""
        conn = self._ensure_connected()

        cursor = conn.execute(
            "SELECT * FROM memories WHERE deleted = 1 ORDER BY id"
        )
        rows = cursor.fetchall()

        return [self._row_to_record(row) for row in rows]

    async def update_importance(self, record_id: int, importance: float) -> bool:
        """Update the importance score of a memory."""
        conn = self._ensure_connected()

        importance = max(0.0, min(1.0, importance))
        cursor = conn.execute(
            "UPDATE memories SET importance = ? WHERE id = ?",
            (importance, record_id),
        )
        conn.commit()

        return cursor.rowcount > 0

    async def clear(self) -> None:
        """Delete all records."""
        conn = self._ensure_connected()

        conn.execute("DELETE FROM embeddings")
        conn.execute("DELETE FROM memories")
        conn.commit()

    async def count(self) -> int:
        """Get count of active memories."""
        conn = self._ensure_connected()

        cursor = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE deleted = 0"
        )
        return cursor.fetchone()[0]

    async def count_deleted(self) -> int:
        """Get count of deleted memories."""
        conn = self._ensure_connected()

        cursor = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE deleted = 1"
        )
        return cursor.fetchone()[0]

    async def get_all_records(self, include_deleted: bool = False) -> list[StorageRecord]:
        """Get all records."""
        conn = self._ensure_connected()

        if include_deleted:
            cursor = conn.execute("SELECT * FROM memories ORDER BY id")
        else:
            cursor = conn.execute(
                "SELECT * FROM memories WHERE deleted = 0 ORDER BY id"
            )
        rows = cursor.fetchall()

        return [self._row_to_record(row) for row in rows]
