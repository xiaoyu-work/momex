# Copyright (c) Xiaoyu Zhang.
# Licensed under the MIT License.

"""PostgreSQL storage provider implementation."""

from datetime import datetime, timezone

import asyncpg

from ...aitools.embeddings import AsyncEmbeddingModel
from ...aitools.vectorbase import TextEmbeddingIndexSettings
from ...knowpro import interfaces
from ...knowpro.convsettings import MessageTextIndexSettings, RelatedTermIndexSettings
from ...knowpro.interfaces import ConversationMetadata, STATUS_INGESTED
from .collections import PostgresMessageCollection, PostgresSemanticRefCollection
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
        pool: asyncpg.Pool,
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
        self.pool = pool
        self.message_type = message_type
        self.semantic_ref_type = semantic_ref_type
        self._metadata = metadata
        self._initialized = False
        self.schema = schema

        # Set up embedding settings
        if message_text_index_settings is None:
            base_embedding_settings = TextEmbeddingIndexSettings()
            self.message_text_index_settings = MessageTextIndexSettings(
                base_embedding_settings
            )
        else:
            self.message_text_index_settings = message_text_index_settings
            base_embedding_settings = message_text_index_settings.embedding_index_settings

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

    async def initialize(self) -> None:
        """Initialize the database schema and components.

        This must be called after creating the provider before any operations.
        """
        if self._initialized:
            return

        # Get embedding size for schema creation
        embedding_size = self.message_text_index_settings.embedding_index_settings.embedding_size

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

        Returns:
            Initialized PostgresStorageProvider instance
        """
        # Create connection pool
        server_settings = None
        if schema:
            from .schema import format_search_path

            server_settings = {"search_path": format_search_path(schema)}

        pool = await asyncpg.create_pool(
            connection_string,
            min_size=min_pool_size,
            max_size=max_pool_size,
            server_settings=server_settings,
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

        # Initialize
        await provider.initialize()

        return provider

    async def __aenter__(self) -> "PostgresStorageProvider[TMessage]":
        """Enter transaction context."""
        if not self._initialized:
            await self.initialize()
        # PostgreSQL transactions are handled at the connection level
        # For now, we don't wrap in a transaction here
        await self._init_conversation_metadata_if_needed()
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
            await set_conversation_metadata(self.pool, **updates)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit transaction context."""
        # asyncpg handles connection management automatically
        pass

    async def close(self) -> None:
        """Close the database connection pool."""
        if hasattr(self, "pool") and self.pool:
            await self.pool.close()

    async def _init_conversation_metadata_if_needed(self) -> None:
        """Initialize conversation metadata if the database is new."""
        from ...knowpro.universal_message import format_timestamp_utc

        current_time = datetime.now(timezone.utc)

        async with self.pool.acquire() as conn:
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

        metadata_kwds = {
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

        await set_conversation_metadata(self.pool, **metadata_kwds)

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

    def get_conversation_metadata(self) -> ConversationMetadata:
        """Get conversation metadata (sync version - use get_conversation_metadata_async)."""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self.get_conversation_metadata_async()
        )

    async def get_conversation_metadata_async(self) -> ConversationMetadata:
        """Get conversation metadata."""
        async with self.pool.acquire() as conn:
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
                "name_tag", "schema_version", "created_at", "updated_at",
                "tag", "embedding_size", "embedding_name",
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

    def set_conversation_metadata(self, **kwds: str | list[str] | None) -> None:
        """Set or update conversation metadata (sync version)."""
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            set_conversation_metadata(self.pool, **kwds)
        )

    def update_conversation_timestamps(
        self,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
    ) -> None:
        """Update conversation timestamps."""
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            self._update_conversation_timestamps_async(created_at, updated_at)
        )

    async def _update_conversation_timestamps_async(
        self,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
    ) -> None:
        """Update conversation timestamps (async version)."""
        from ...knowpro.universal_message import format_timestamp_utc

        metadata_kwds: dict[str, str] = {}
        if created_at is not None:
            metadata_kwds["created_at"] = format_timestamp_utc(created_at)
        if updated_at is not None:
            metadata_kwds["updated_at"] = format_timestamp_utc(updated_at)

        if metadata_kwds:
            await set_conversation_metadata(self.pool, **metadata_kwds)

    def get_db_version(self) -> int:
        """Get the database schema version."""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            get_db_schema_version(self.pool)
        )

    def is_source_ingested(self, source_id: str) -> bool:
        """Check if a source has already been ingested."""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self._is_source_ingested_async(source_id)
        )

    async def _is_source_ingested_async(self, source_id: str) -> bool:
        """Check if a source has already been ingested (async)."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT status FROM IngestedSources WHERE source_id = $1",
                source_id,
            )
            return row is not None and row[0] == STATUS_INGESTED

    def get_source_status(self, source_id: str) -> str | None:
        """Get the ingestion status of a source."""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self._get_source_status_async(source_id)
        )

    async def _get_source_status_async(self, source_id: str) -> str | None:
        """Get the ingestion status of a source (async)."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT status FROM IngestedSources WHERE source_id = $1",
                source_id,
            )
            return row[0] if row else None

    def mark_source_ingested(
        self, source_id: str, status: str = STATUS_INGESTED
    ) -> None:
        """Mark a source as ingested."""
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            self._mark_source_ingested_async(source_id, status)
        )

    async def _mark_source_ingested_async(
        self, source_id: str, status: str = STATUS_INGESTED
    ) -> None:
        """Mark a source as ingested (async)."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO IngestedSources (source_id, status)
                VALUES ($1, $2)
                ON CONFLICT (source_id) DO UPDATE SET status = $2
                """,
                source_id, status,
            )
