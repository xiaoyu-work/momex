# Copyright (c) Xiaoyu Zhang.
# Licensed under the MIT License.

"""PostgreSQL storage provider implementation."""

from contextlib import AbstractAsyncContextManager
from contextvars import ContextVar
from datetime import datetime, timezone
import sys
from types import TracebackType
from typing import Any

from ...aitools.vectorbase import TextEmbeddingIndexSettings
from ...knowpro import interfaces
from ...knowpro.convsettings import MessageTextIndexSettings, RelatedTermIndexSettings
from ...knowpro.interfaces import ConversationMetadata, STATUS_INGESTED
from .collections import PostgresMessageCollection, PostgresSemanticRefCollection
from .connection import Pool, TransactionPool
from .messageindex import PostgresMessageTextIndex
from .propindex import PostgresPropertyIndex
from .reltermsindex import PostgresRelatedTermsIndex
from .schema import (
    CONVERSATION_SCHEMA_VERSION,
    get_db_schema_version,
    init_db_schema,
    set_conversation_metadata,
)
from .semrefindex import PostgresTermToSemanticRefIndex
from .timestampindex import PostgresTimestampToTextRangeIndex


class PostgresStorageProvider[TMessage: interfaces.IMessage](
    interfaces.IStorageProvider[TMessage]
):
    """PostgreSQL-backed storage provider implementation with pgvector support.

    This provider uses asyncpg for async PostgreSQL access and pgvector for
    embedding similarity search.
    """

    def __init__(
        self,
        pool: Pool,
        message_type: type[TMessage] = None,  # type: ignore
        semantic_ref_type: type[interfaces.SemanticRef] = None,  # type: ignore
        message_text_index_settings: MessageTextIndexSettings | None = None,
        related_term_index_settings: RelatedTermIndexSettings | None = None,
        metadata: ConversationMetadata | None = None,
        schema: str | None = None,
    ):
        """Initialize PostgreSQL storage provider.

        Args:
            pool: asyncpg connection pool (must be created with create_pool())
            message_type: Type for deserializing messages
            semantic_ref_type: Type for deserializing semantic refs
            message_text_index_settings: Settings for message text embedding index
            related_term_index_settings: Settings for related terms index
            metadata: Initial conversation metadata
        """
        self.pool = TransactionPool(pool, schema)
        self._transaction: ContextVar[AbstractAsyncContextManager[Any] | None] = (
            ContextVar("storage_transaction", default=None)
        )
        self.message_type = message_type
        self.semantic_ref_type = semantic_ref_type
        self._metadata = metadata
        self._initialized = False
        self.schema = schema
        self._pgbouncer = False  # Set by create() if using pgbouncer mode

        # Set up embedding settings
        if message_text_index_settings is None:
            base_embedding_settings = TextEmbeddingIndexSettings()
            self.message_text_index_settings = MessageTextIndexSettings(
                base_embedding_settings
            )
        else:
            self.message_text_index_settings = message_text_index_settings
            base_embedding_settings = (
                message_text_index_settings.embedding_index_settings
            )

        if related_term_index_settings is None:
            self.related_term_index_settings = RelatedTermIndexSettings(
                base_embedding_settings
            )
        else:
            self.related_term_index_settings = related_term_index_settings

        # Initialize collections (lazy - actual DB init happens in initialize())
        self._message_collection: PostgresMessageCollection[TMessage] | None = None
        self._semantic_ref_collection: PostgresSemanticRefCollection | None = None

        # Initialize indexes
        self._term_to_semantic_ref_index: PostgresTermToSemanticRefIndex | None = None
        self._property_index: PostgresPropertyIndex | None = None
        self._timestamp_index: PostgresTimestampToTextRangeIndex | None = None
        self._message_text_index: PostgresMessageTextIndex | None = None
        self._related_terms_index: PostgresRelatedTermsIndex | None = None

    async def _set_search_path(self, conn) -> None:
        """Set search_path for the connection if schema is configured.

        Uses SET LOCAL so it works within pgbouncer transaction-pooling mode.
        """
        if self.schema and self._pgbouncer:
            from .schema import format_search_path

            await conn.execute(f"SET search_path TO {format_search_path(self.schema)}")

    async def initialize(self) -> None:
        """Initialize the database schema and components.

        This must be called after creating the provider before any operations.
        """
        if self._initialized:
            return

        # Get embedding size for schema creation
        embedding_size = (
            self.message_text_index_settings.embedding_index_settings.embedding_size
        )

        # Initialize database schema
        await init_db_schema(self.pool, embedding_size, schema=self.schema)
        await self._check_embedding_consistency()

        # Initialize collections
        self._message_collection = PostgresMessageCollection(
            self.pool, self.message_type
        )
        self._semantic_ref_collection = PostgresSemanticRefCollection(self.pool)

        # Initialize indexes
        self._term_to_semantic_ref_index = PostgresTermToSemanticRefIndex(self.pool)
        self._property_index = PostgresPropertyIndex(self.pool)
        self._timestamp_index = PostgresTimestampToTextRangeIndex(self.pool)
        self._message_text_index = PostgresMessageTextIndex(
            self.pool,
            self.message_text_index_settings,
            self._message_collection,
        )
        self._related_terms_index = PostgresRelatedTermsIndex(
            self.pool,
            self.related_term_index_settings.embedding_index_settings,
        )

        # Connect message collection to message text index
        self._message_collection.set_message_text_index(self._message_text_index)

        self._initialized = True

    @classmethod
    async def create(
        cls,
        connection_string: str,
        message_type: type[TMessage] = None,  # type: ignore
        semantic_ref_type: type[interfaces.SemanticRef] = None,  # type: ignore
        message_text_index_settings: MessageTextIndexSettings | None = None,
        related_term_index_settings: RelatedTermIndexSettings | None = None,
        metadata: ConversationMetadata | None = None,
        min_pool_size: int = 2,
        max_pool_size: int = 10,
        schema: str | None = None,
        pgbouncer: bool = False,
    ) -> "PostgresStorageProvider[TMessage]":
        """Create and initialize a PostgreSQL storage provider.

        Args:
            connection_string: PostgreSQL connection string
                (e.g., "postgresql://user:pass@localhost:5432/dbname")
            message_type: Type for deserializing messages
            semantic_ref_type: Type for deserializing semantic refs
            message_text_index_settings: Settings for message text embedding index
            related_term_index_settings: Settings for related terms index
            metadata: Initial conversation metadata
            min_pool_size: Minimum connections in pool
            max_pool_size: Maximum connections in pool
            pgbouncer: Enable pgbouncer compatibility mode (disables prepared statements).
                Required for Supabase, PgBouncer, and similar connection poolers.

        Returns:
            Initialized PostgresStorageProvider instance
        """
        import asyncpg  # type: ignore[import-not-found]

        # For pgbouncer mode with schema, create the schema first before creating pool
        if pgbouncer and schema:
            from .schema import quote_ident

            temp_conn = await asyncpg.connect(connection_string, statement_cache_size=0)
            try:
                await temp_conn.execute(
                    f"CREATE SCHEMA IF NOT EXISTS {quote_ident(schema)}"
                )
            finally:
                await temp_conn.close()

        # Create connection pool
        pool_kwargs: dict = {
            "min_size": min_pool_size,
            "max_size": max_pool_size,
        }

        # TransactionPool sets search_path inside each pinned transaction.
        if pgbouncer:
            pool_kwargs["statement_cache_size"] = 0
        else:
            # For non-pgbouncer, use server_settings (session-level)
            if schema:
                from .schema import format_search_path

                pool_kwargs["server_settings"] = {
                    "search_path": format_search_path(schema)
                }

        pool = await asyncpg.create_pool(
            connection_string,
            **pool_kwargs,
        )

        # Create provider
        provider = cls(
            pool=pool,
            message_type=message_type,
            semantic_ref_type=semantic_ref_type,
            message_text_index_settings=message_text_index_settings,
            related_term_index_settings=related_term_index_settings,
            metadata=metadata,
            schema=schema,
        )
        provider._pgbouncer = pgbouncer

        # Initialize
        await provider.initialize()

        return provider

    async def __aenter__(self) -> "PostgresStorageProvider[TMessage]":
        """Enter transaction context."""
        if not self._initialized:
            await self.initialize()
        context = self.pool.transaction()
        connection = await context.__aenter__()
        try:
            await connection.execute(
                "LOCK TABLE Messages, SemanticRefs IN SHARE ROW EXCLUSIVE MODE"
            )
            await self._init_conversation_metadata_if_needed()
        except BaseException:
            await context.__aexit__(*sys.exc_info())
            raise
        self._transaction.set(context)
        return self

    async def _check_embedding_consistency(self) -> None:
        """Check that stored embedding metadata matches configured settings."""
        expected_size = (
            self.message_text_index_settings.embedding_index_settings.embedding_size
        )
        expected_name = (
            self.message_text_index_settings.embedding_index_settings.embedding_model.model_name
        )

        async with self.pool.acquire() as conn:
            await self._set_search_path(conn)
            rows = await conn.fetch("SELECT key, value FROM ConversationMetadata")

        if not rows:
            return

        metadata_dict: dict[str, list[str]] = {}
        for row in rows:
            key, value = row[0], row[1]
            if key not in metadata_dict:
                metadata_dict[key] = []
            metadata_dict[key].append(value)

        def get_single(key: str) -> str | None:
            values = metadata_dict.get(key)
            if values is None:
                return None
            if len(values) > 1:
                raise ValueError(
                    f"Expected single value for key '{key}', got {len(values)}"
                )
            return values[0]

        stored_size_str = get_single("embedding_size")
        stored_name = get_single("embedding_name")
        stored_size = int(stored_size_str) if stored_size_str else None

        if stored_size is not None and stored_size != expected_size:
            raise ValueError(
                "Conversation metadata embedding_size does not match provider settings"
            )
        if stored_name is not None and stored_name != expected_name:
            raise ValueError(
                "Conversation metadata embedding_model does not match provider settings"
            )

        updates: dict[str, str] = {}
        if stored_size is None:
            updates["embedding_size"] = str(expected_size)
        if stored_name is None:
            updates["embedding_name"] = expected_name
        if updates:
            await set_conversation_metadata(
                self.pool, schema=self.schema if self._pgbouncer else None, **updates
            )

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit transaction context."""
        context = self._transaction.get()
        if context is None:
            raise RuntimeError("No active storage transaction")
        self._transaction.set(None)
        await context.__aexit__(exc_type, exc_val, exc_tb)

    async def close(self) -> None:
        """Close the database connection pool."""
        if hasattr(self, "pool") and self.pool:
            await self.pool.close()

    async def _init_conversation_metadata_if_needed(self) -> None:
        """Initialize conversation metadata if the database is new."""
        from ...knowpro.universal_message import format_timestamp_utc

        current_time = datetime.now(timezone.utc)

        async with self.pool.acquire() as conn:
            await self._set_search_path(conn)
            row = await conn.fetchrow("SELECT 1 FROM ConversationMetadata LIMIT 1")
            if row is not None:
                return

        # Use provided metadata values, or generate defaults
        if self._metadata:
            name_tag = self._metadata.name_tag or "conversation"
            tags = self._metadata.tags
            extras = self._metadata.extra or {}
        else:
            name_tag = "conversation"
            tags = None
            extras = {}

        actual_embedding_size = (
            self.message_text_index_settings.embedding_index_settings.embedding_size
        )
        actual_embedding_name = (
            self.message_text_index_settings.embedding_index_settings.embedding_model.model_name
        )

        metadata_kwds: dict[str, str | list[str]] = {
            "name_tag": name_tag,
            "schema_version": str(CONVERSATION_SCHEMA_VERSION),
            "created_at": format_timestamp_utc(current_time),
            "updated_at": format_timestamp_utc(current_time),
            "embedding_size": str(actual_embedding_size),
            "embedding_name": actual_embedding_name,
        }

        if tags:
            metadata_kwds["tag"] = tags

        for key, value in extras.items():
            if key not in {"embedding_size", "embedding_name"}:
                metadata_kwds[key] = value

        await set_conversation_metadata(
            self.pool, schema=self.schema if self._pgbouncer else None, **metadata_kwds
        )

    @property
    def messages(self) -> PostgresMessageCollection[TMessage]:
        assert self._message_collection is not None, "Provider not initialized"
        return self._message_collection

    @property
    def semantic_refs(self) -> PostgresSemanticRefCollection:
        assert self._semantic_ref_collection is not None, "Provider not initialized"
        return self._semantic_ref_collection

    @property
    def term_to_semantic_ref_index(self) -> PostgresTermToSemanticRefIndex:
        assert self._term_to_semantic_ref_index is not None, "Provider not initialized"
        return self._term_to_semantic_ref_index

    @property
    def semantic_ref_index(self) -> PostgresTermToSemanticRefIndex:
        return self.term_to_semantic_ref_index

    @property
    def conversation_threads(self) -> interfaces.IConversationThreads:
        from ..memory.convthreads import ConversationThreads

        return ConversationThreads(
            self.message_text_index_settings.embedding_index_settings
        )

    @property
    def property_index(self) -> PostgresPropertyIndex:
        assert self._property_index is not None, "Provider not initialized"
        return self._property_index

    @property
    def timestamp_index(self) -> PostgresTimestampToTextRangeIndex:
        assert self._timestamp_index is not None, "Provider not initialized"
        return self._timestamp_index

    @property
    def message_text_index(self) -> PostgresMessageTextIndex:
        assert self._message_text_index is not None, "Provider not initialized"
        return self._message_text_index

    @property
    def related_terms_index(self) -> PostgresRelatedTermsIndex:
        assert self._related_terms_index is not None, "Provider not initialized"
        return self._related_terms_index

    # Async getters required by base class
    async def get_message_collection(
        self, message_type: type[TMessage] | None = None
    ) -> interfaces.IMessageCollection[TMessage]:
        return self.messages

    async def get_semantic_ref_collection(self) -> interfaces.ISemanticRefCollection:
        return self.semantic_refs

    async def get_semantic_ref_index(self) -> interfaces.ITermToSemanticRefIndex:
        return self.term_to_semantic_ref_index

    async def get_property_index(self) -> interfaces.IPropertyToSemanticRefIndex:
        return self.property_index

    async def get_timestamp_index(self) -> interfaces.ITimestampToTextRangeIndex:
        return self.timestamp_index

    async def get_message_text_index(self) -> interfaces.IMessageTextIndex[TMessage]:
        return self.message_text_index

    async def get_related_terms_index(self) -> interfaces.ITermToRelatedTermsIndex:
        return self.related_terms_index

    async def get_conversation_threads(self) -> interfaces.IConversationThreads:
        """Get the conversation threads."""
        from ...storage.memory.convthreads import ConversationThreads

        return ConversationThreads(
            self.message_text_index_settings.embedding_index_settings
        )

    async def clear(self) -> None:
        """Clear all data from the storage provider."""
        async with self.pool.acquire() as conn:
            await self._set_search_path(conn)
            # Clear in reverse dependency order
            await conn.execute("DELETE FROM RelatedTermsFuzzy")
            await conn.execute("DELETE FROM RelatedTermsAliases")
            await conn.execute("DELETE FROM MessageTextIndex")
            await conn.execute("DELETE FROM PropertyIndex")
            await conn.execute("DELETE FROM SemanticRefIndex")
            await conn.execute("DELETE FROM SemanticRefs")
            await conn.execute("DELETE FROM Messages")
            await conn.execute("DELETE FROM ConversationMetadata")
            await conn.execute("DELETE FROM IngestedSources")
            await conn.execute("DELETE FROM ChunkFailures")

    def serialize(self) -> dict:
        """Serialize all storage provider data."""
        raise NotImplementedError("Use serialize_async for PostgreSQL provider")

    async def serialize_async(self) -> dict:
        """Serialize all storage provider data (async version)."""
        return {
            "termToSemanticRefIndexData": await self.term_to_semantic_ref_index.serialize(),
            "relatedTermsIndexData": await self.related_terms_index.serialize(),
        }

    async def deserialize(self, data: dict) -> None:
        """Deserialize storage provider data."""
        if data.get("termToSemanticRefIndexData"):
            await self.term_to_semantic_ref_index.deserialize(
                data["termToSemanticRefIndexData"]
            )

        if data.get("relatedTermsIndexData"):
            await self.related_terms_index.deserialize(data["relatedTermsIndexData"])

        if data.get("messageIndexData"):
            await self.message_text_index.deserialize(data["messageIndexData"])

    async def get_conversation_metadata(self) -> ConversationMetadata:
        """Get conversation metadata."""
        async with self.pool.acquire() as conn:
            await self._set_search_path(conn)
            rows = await conn.fetch("SELECT key, value FROM ConversationMetadata")

            if not rows:
                return ConversationMetadata()

            metadata_dict: dict[str, list[str]] = {}
            for row in rows:
                key, value = row[0], row[1]
                if key not in metadata_dict:
                    metadata_dict[key] = []
                metadata_dict[key].append(value)

            def get_single(key: str) -> str | None:
                values = metadata_dict.get(key)
                if values is None:
                    return None
                if len(values) > 1:
                    raise ValueError(
                        f"Expected single value for key '{key}', got {len(values)}"
                    )
                return values[0]

            def parse_datetime(value_str: str) -> datetime:
                if value_str.endswith("Z"):
                    value_str = value_str[:-1] + "+00:00"
                try:
                    return datetime.fromisoformat(value_str)
                except ValueError:
                    return datetime.now(timezone.utc)

            name_tag = get_single("name_tag")
            schema_version_str = get_single("schema_version")
            schema_version = int(schema_version_str) if schema_version_str else None
            created_at_str = get_single("created_at")
            created_at = parse_datetime(created_at_str) if created_at_str else None
            updated_at_str = get_single("updated_at")
            updated_at = parse_datetime(updated_at_str) if updated_at_str else None
            embedding_size_str = get_single("embedding_size")
            embedding_size = int(embedding_size_str) if embedding_size_str else None
            embedding_model = get_single("embedding_name")
            tags = metadata_dict.get("tag")

            standard_keys = {
                "name_tag",
                "schema_version",
                "created_at",
                "updated_at",
                "tag",
                "embedding_size",
                "embedding_name",
            }
            extra = {}
            for key, values in metadata_dict.items():
                if key not in standard_keys:
                    extra[key] = ", ".join(values)

            return ConversationMetadata(
                name_tag=name_tag,
                schema_version=schema_version,
                created_at=created_at,
                updated_at=updated_at,
                embedding_size=embedding_size,
                embedding_model=embedding_model,
                tags=tags,
                extra=extra if extra else None,
            )

    async def set_conversation_metadata(self, **kwds: str | list[str] | None) -> None:
        """Set or update conversation metadata."""
        await set_conversation_metadata(
            self.pool, schema=self.schema if self._pgbouncer else None, **kwds
        )

    async def update_conversation_timestamps(
        self,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
    ) -> None:
        """Update conversation timestamps."""
        from ...knowpro.universal_message import format_timestamp_utc

        metadata_kwds: dict[str, str] = {}
        if created_at is not None:
            metadata_kwds["created_at"] = format_timestamp_utc(created_at)
        if updated_at is not None:
            metadata_kwds["updated_at"] = format_timestamp_utc(updated_at)

        if metadata_kwds:
            await set_conversation_metadata(
                self.pool,
                schema=self.schema if self._pgbouncer else None,
                **metadata_kwds,
            )

    async def get_db_version(self) -> int:
        """Get the database schema version."""
        return await get_db_schema_version(self.pool)

    async def is_source_ingested(self, source_id: str) -> bool:
        """Check if a source has already been ingested."""
        async with self.pool.acquire() as conn:
            await self._set_search_path(conn)
            row = await conn.fetchrow(
                "SELECT status FROM IngestedSources WHERE source_id = $1",
                source_id,
            )
            return row is not None and row[0] == STATUS_INGESTED

    async def claim_sources(self, source_ids: list[str]) -> set[str]:
        claimed: set[str] = set()
        async with self.pool.acquire() as conn:
            for source_id in dict.fromkeys(source_ids):
                value = await conn.fetchval(
                    "INSERT INTO IngestedSources (source_id, status) VALUES ($1, $2) "
                    "ON CONFLICT DO NOTHING RETURNING source_id",
                    source_id,
                    STATUS_INGESTED,
                )
                if value is not None:
                    claimed.add(value)
        return claimed

    async def get_source_status(self, source_id: str) -> str | None:
        """Get the ingestion status of a source."""
        async with self.pool.acquire() as conn:
            await self._set_search_path(conn)
            row = await conn.fetchrow(
                "SELECT status FROM IngestedSources WHERE source_id = $1",
                source_id,
            )
            return row[0] if row else None

    async def mark_source_ingested(
        self, source_id: str, status: str = STATUS_INGESTED
    ) -> None:
        """Mark a source as ingested."""
        async with self.pool.acquire() as conn:
            await self._set_search_path(conn)
            await conn.execute(
                """
                INSERT INTO IngestedSources (source_id, status)
                VALUES ($1, $2)
                ON CONFLICT (source_id) DO UPDATE SET status = $2
                """,
                source_id,
                status,
            )

    async def mark_sources_ingested_batch(
        self, source_ids: list[str], status: str = STATUS_INGESTED
    ) -> None:
        async with self.pool.acquire() as conn:
            await conn.executemany(
                "INSERT INTO IngestedSources (source_id, status) VALUES ($1, $2) "
                "ON CONFLICT (source_id) DO UPDATE SET status = EXCLUDED.status",
                [(source_id, status) for source_id in source_ids],
            )

    async def record_chunk_failure(
        self,
        message_ordinal: int,
        chunk_ordinal: int,
        error_class: str,
        error_message: str,
    ) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO ChunkFailures VALUES ($1, $2, $3, $4, $5) "
                "ON CONFLICT (msg_id, chunk_ordinal) DO UPDATE SET "
                "error_class=EXCLUDED.error_class, error_message=EXCLUDED.error_message, "
                "failed_at=EXCLUDED.failed_at",
                message_ordinal,
                chunk_ordinal,
                error_class,
                error_message,
                datetime.now(timezone.utc),
            )

    async def clear_chunk_failure(
        self, message_ordinal: int, chunk_ordinal: int
    ) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM ChunkFailures WHERE msg_id=$1 AND chunk_ordinal=$2",
                message_ordinal,
                chunk_ordinal,
            )

    async def get_chunk_failures(self) -> list[interfaces.ChunkFailure]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT msg_id, chunk_ordinal, error_class, error_message, failed_at "
                "FROM ChunkFailures ORDER BY msg_id, chunk_ordinal"
            )
        return [interfaces.ChunkFailure(*row) for row in rows]
