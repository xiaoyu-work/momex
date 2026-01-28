"""PostgreSQL storage backend for Memex.

Uses pgvector extension for efficient vector similarity search.
Compatible with AWS RDS, Azure PostgreSQL, Supabase, Neon, etc.
"""

from __future__ import annotations

import json
from typing import Any

from .base import SearchResult, StorageBackend, StorageRecord


class PostgresBackend(StorageBackend):
    """PostgreSQL storage backend with pgvector for vector search.

    Requires:
        - PostgreSQL with pgvector extension
        - asyncpg library: pip install asyncpg

    Args:
        connection_string: PostgreSQL connection string.
        table_prefix: Prefix for table names (default "memex").
        embedding_dim: Dimension of embedding vectors (default 1536).

    Example:
        backend = PostgresBackend(
            connection_string="postgresql://user:pass@localhost/memex"
        )
        await backend.initialize()
        record_id = await backend.add("Hello", embedding=[0.1, 0.2, ...])

    Cloud compatibility:
        - AWS RDS: postgresql://user:pass@xxx.rds.amazonaws.com:5432/memex
        - Azure: postgresql://user:pass@xxx.postgres.database.azure.com:5432/memex
        - Supabase: postgresql://user:pass@db.xxx.supabase.co:5432/postgres
        - Neon: postgresql://user:pass@xxx.neon.tech/memex
    """

    def __init__(
        self,
        connection_string: str,
        table_prefix: str = "memex",
        embedding_dim: int = 1536,
    ) -> None:
        self.connection_string = connection_string
        self.table_prefix = table_prefix
        self.embedding_dim = embedding_dim
        self._pool = None

    @property
    def _memories_table(self) -> str:
        return f"{self.table_prefix}_memories"

    async def initialize(self) -> None:
        """Initialize the database and create tables."""
        try:
            import asyncpg
        except ImportError:
            raise ImportError(
                "asyncpg is required for PostgreSQL backend. "
                "Install it with: pip install asyncpg"
            )

        self._pool = await asyncpg.create_pool(self.connection_string)

        async with self._pool.acquire() as conn:
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # Create memories table with vector column
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._memories_table} (
                    id SERIAL PRIMARY KEY,
                    text TEXT NOT NULL,
                    speaker TEXT,
                    timestamp TIMESTAMPTZ,
                    importance REAL DEFAULT 0.5,
                    deleted BOOLEAN DEFAULT FALSE,
                    metadata JSONB,
                    embedding vector({self.embedding_dim}),
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_prefix}_deleted
                ON {self._memories_table}(deleted)
            """)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_prefix}_timestamp
                ON {self._memories_table}(timestamp)
            """)

            # Create vector index for similarity search (HNSW is faster than IVFFlat)
            # This may take a while for large datasets
            try:
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.table_prefix}_embedding
                    ON {self._memories_table}
                    USING hnsw (embedding vector_cosine_ops)
                """)
            except Exception:
                # HNSW might not be available in older pgvector versions
                # Fall back to IVFFlat
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.table_prefix}_embedding
                    ON {self._memories_table}
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                """)

    async def close(self) -> None:
        """Close the database connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None

    def _ensure_connected(self):
        """Ensure database pool is initialized."""
        if self._pool is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self._pool

    def _row_to_record(self, row) -> StorageRecord:
        """Convert database row to StorageRecord."""
        return StorageRecord(
            id=row["id"],
            text=row["text"],
            speaker=row["speaker"],
            timestamp=row["timestamp"].isoformat() if row["timestamp"] else None,
            importance=row["importance"],
            deleted=row["deleted"],
            metadata=row["metadata"],
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
        pool = self._ensure_connected()

        # Convert embedding list to pgvector format
        embedding_str = f"[{','.join(str(x) for x in embedding)}]"

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                INSERT INTO {self._memories_table}
                (text, speaker, timestamp, importance, metadata, embedding)
                VALUES ($1, $2, $3::timestamptz, $4, $5::jsonb, $6::vector)
                RETURNING id
                """,
                text,
                speaker,
                timestamp,
                importance,
                json.dumps(metadata) if metadata else None,
                embedding_str,
            )
            return row["id"]

    async def get(self, record_id: int) -> StorageRecord | None:
        """Get a memory record by ID."""
        pool = self._ensure_connected()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT * FROM {self._memories_table} WHERE id = $1",
                record_id,
            )

            if row is None:
                return None

            return self._row_to_record(row)

    async def search(
        self,
        embedding: list[float],
        limit: int = 10,
        threshold: float = 0.3,
    ) -> list[SearchResult]:
        """Search for similar memories using pgvector cosine similarity."""
        pool = self._ensure_connected()

        # Convert embedding to pgvector format
        embedding_str = f"[{','.join(str(x) for x in embedding)}]"

        async with pool.acquire() as conn:
            # Use pgvector's cosine distance operator (<=>)
            # Note: <=> returns distance (0 = identical), so similarity = 1 - distance
            rows = await conn.fetch(
                f"""
                SELECT *, 1 - (embedding <=> $1::vector) as similarity
                FROM {self._memories_table}
                WHERE deleted = FALSE
                AND 1 - (embedding <=> $1::vector) >= $2
                ORDER BY embedding <=> $1::vector
                LIMIT $3
                """,
                embedding_str,
                threshold,
                limit,
            )

            return [
                SearchResult(
                    record=self._row_to_record(row),
                    similarity=float(row["similarity"]),
                )
                for row in rows
            ]

    async def delete(self, record_id: int) -> bool:
        """Soft delete a memory."""
        pool = self._ensure_connected()

        async with pool.acquire() as conn:
            result = await conn.execute(
                f"UPDATE {self._memories_table} SET deleted = TRUE WHERE id = $1 AND deleted = FALSE",
                record_id,
            )
            return result == "UPDATE 1"

    async def restore(self, record_id: int) -> bool:
        """Restore a soft-deleted memory."""
        pool = self._ensure_connected()

        async with pool.acquire() as conn:
            result = await conn.execute(
                f"UPDATE {self._memories_table} SET deleted = FALSE WHERE id = $1 AND deleted = TRUE",
                record_id,
            )
            return result == "UPDATE 1"

    async def list_deleted(self) -> list[StorageRecord]:
        """List all soft-deleted memories."""
        pool = self._ensure_connected()

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT * FROM {self._memories_table} WHERE deleted = TRUE ORDER BY id"
            )
            return [self._row_to_record(row) for row in rows]

    async def update_importance(self, record_id: int, importance: float) -> bool:
        """Update the importance score of a memory."""
        pool = self._ensure_connected()

        importance = max(0.0, min(1.0, importance))

        async with pool.acquire() as conn:
            result = await conn.execute(
                f"UPDATE {self._memories_table} SET importance = $1 WHERE id = $2",
                importance,
                record_id,
            )
            return result == "UPDATE 1"

    async def clear(self) -> None:
        """Delete all records."""
        pool = self._ensure_connected()

        async with pool.acquire() as conn:
            await conn.execute(f"TRUNCATE {self._memories_table} RESTART IDENTITY")

    async def count(self) -> int:
        """Get count of active memories."""
        pool = self._ensure_connected()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT COUNT(*) FROM {self._memories_table} WHERE deleted = FALSE"
            )
            return row[0]

    async def count_deleted(self) -> int:
        """Get count of deleted memories."""
        pool = self._ensure_connected()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT COUNT(*) FROM {self._memories_table} WHERE deleted = TRUE"
            )
            return row[0]

    async def get_all_records(self, include_deleted: bool = False) -> list[StorageRecord]:
        """Get all records."""
        pool = self._ensure_connected()

        async with pool.acquire() as conn:
            if include_deleted:
                rows = await conn.fetch(
                    f"SELECT * FROM {self._memories_table} ORDER BY id"
                )
            else:
                rows = await conn.fetch(
                    f"SELECT * FROM {self._memories_table} WHERE deleted = FALSE ORDER BY id"
                )
            return [self._row_to_record(row) for row in rows]
