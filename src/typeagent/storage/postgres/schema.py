# Copyright (c) Xiaoyu Zhang.
# Licensed under the MIT License.

"""PostgreSQL database schema definitions."""

from datetime import datetime, timezone

import numpy as np

from ...aitools.embeddings import NormalizedEmbedding
from ...knowpro.interfaces import ConversationMetadata, STATUS_INGESTED

# Constants
CONVERSATION_SCHEMA_VERSION = 1
VECTOR_INDEX_MIN_ROWS = 1000

# Enable pgvector extension
PGVECTOR_EXTENSION = """
CREATE EXTENSION IF NOT EXISTS vector;
"""

MESSAGES_SCHEMA = """
CREATE TABLE IF NOT EXISTS Messages (
    msg_id SERIAL PRIMARY KEY,
    chunks JSONB NULL,
    chunk_uri TEXT NULL,
    start_timestamp TIMESTAMPTZ NULL,
    tags JSONB NULL,
    metadata JSONB NULL,
    extra JSONB NULL,
    CONSTRAINT chunks_xor_chunkuri CHECK (
        (chunks IS NOT NULL AND chunk_uri IS NULL) OR
        (chunks IS NULL AND chunk_uri IS NOT NULL)
    )
);
"""

TIMESTAMP_INDEX_SCHEMA = """
CREATE INDEX IF NOT EXISTS idx_messages_start_timestamp ON Messages(start_timestamp);
"""

CONVERSATION_METADATA_SCHEMA = """
CREATE TABLE IF NOT EXISTS ConversationMetadata (
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY (key, value)
);
"""

SEMANTIC_REFS_SCHEMA = """
CREATE TABLE IF NOT EXISTS SemanticRefs (
    semref_id INTEGER PRIMARY KEY,
    range_json JSONB NOT NULL,
    knowledge_type TEXT NOT NULL,
    knowledge_json JSONB NOT NULL
);
"""

SEMANTIC_REF_INDEX_SCHEMA = """
CREATE TABLE IF NOT EXISTS SemanticRefIndex (
    term TEXT NOT NULL,
    semref_id INTEGER NOT NULL REFERENCES SemanticRefs(semref_id) ON DELETE CASCADE
);
"""

SEMANTIC_REF_INDEX_TERM_INDEX = """
CREATE INDEX IF NOT EXISTS idx_semantic_ref_index_term ON SemanticRefIndex(term);
"""

# Use pgvector for embeddings - dimension will be set dynamically
MESSAGE_TEXT_INDEX_SCHEMA_TEMPLATE = """
CREATE TABLE IF NOT EXISTS MessageTextIndex (
    id SERIAL PRIMARY KEY,
    msg_id INTEGER NOT NULL REFERENCES Messages(msg_id) ON DELETE CASCADE,
    chunk_ordinal INTEGER NOT NULL,
    embedding vector({embedding_size}) NOT NULL,
    UNIQUE (msg_id, chunk_ordinal)
);
"""

MESSAGE_TEXT_INDEX_EMBEDDING_INDEX_TEMPLATE = """
CREATE INDEX IF NOT EXISTS idx_message_text_embedding
ON MessageTextIndex USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
"""

PROPERTY_INDEX_SCHEMA = """
CREATE TABLE IF NOT EXISTS PropertyIndex (
    prop_name TEXT NOT NULL,
    value_str TEXT NOT NULL,
    score REAL NOT NULL DEFAULT 1.0,
    semref_id INTEGER NOT NULL REFERENCES SemanticRefs(semref_id) ON DELETE CASCADE
);
"""

PROPERTY_INDEX_PROP_NAME_INDEX = """
CREATE INDEX IF NOT EXISTS idx_property_index_prop_name ON PropertyIndex(prop_name);
"""

PROPERTY_INDEX_VALUE_STR_INDEX = """
CREATE INDEX IF NOT EXISTS idx_property_index_value_str ON PropertyIndex(value_str);
"""

PROPERTY_INDEX_COMBINED_INDEX = """
CREATE INDEX IF NOT EXISTS idx_property_index_combined ON PropertyIndex(prop_name, value_str);
"""

RELATED_TERMS_ALIASES_SCHEMA = """
CREATE TABLE IF NOT EXISTS RelatedTermsAliases (
    term TEXT NOT NULL,
    alias TEXT NOT NULL,
    PRIMARY KEY (term, alias)
);
"""

RELATED_TERMS_ALIASES_TERM_INDEX = """
CREATE INDEX IF NOT EXISTS idx_related_aliases_term ON RelatedTermsAliases(term);
"""

RELATED_TERMS_ALIASES_ALIAS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_related_aliases_alias ON RelatedTermsAliases(alias);
"""

# Use pgvector for term embeddings
RELATED_TERMS_FUZZY_SCHEMA_TEMPLATE = """
CREATE TABLE IF NOT EXISTS RelatedTermsFuzzy (
    term TEXT NOT NULL PRIMARY KEY,
    term_embedding vector({embedding_size}) NOT NULL
);
"""

RELATED_TERMS_FUZZY_EMBEDDING_INDEX_TEMPLATE = """
CREATE INDEX IF NOT EXISTS idx_related_terms_embedding
ON RelatedTermsFuzzy USING ivfflat (term_embedding vector_cosine_ops)
WITH (lists = 100);
"""

INGESTED_SOURCES_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS IngestedSources (
    source_id TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT '{STATUS_INGESTED}'
);
"""


def serialize_embedding(embedding: NormalizedEmbedding | None) -> str | None:
    """Serialize a numpy embedding array to pgvector string format."""
    if embedding is None:
        return None
    # pgvector expects format: [0.1, 0.2, 0.3, ...]
    return "[" + ",".join(str(x) for x in embedding) + "]"


def deserialize_embedding(pgvector_str: str | None) -> NormalizedEmbedding | None:
    """Deserialize pgvector string back to numpy embedding array."""
    if pgvector_str is None:
        return None
    # Handle both string and list formats
    if isinstance(pgvector_str, (list, tuple)):
        return np.array(pgvector_str, dtype=np.float32)
    # Remove brackets and parse
    clean = pgvector_str.strip("[]")
    values = [float(x) for x in clean.split(",")]
    return np.array(values, dtype=np.float32)


def _create_default_metadata() -> ConversationMetadata:
    """Create default conversation metadata."""
    current_time = datetime.now(timezone.utc)
    return ConversationMetadata(
        name_tag="",
        schema_version=CONVERSATION_SCHEMA_VERSION,
        tags=[],
        extra={},
        created_at=current_time,
        updated_at=current_time,
    )


def quote_ident(name: str) -> str:
    """Safely quote a PostgreSQL identifier."""
    return '"' + name.replace('"', '""') + '"'


def format_search_path(schema: str) -> str:
    """Format search_path with a preferred schema and public fallback."""
    return f'{quote_ident(schema)}, public'


async def _index_exists(conn, index_name: str) -> bool:
    row = await conn.fetchrow(
        """
        SELECT 1
        FROM pg_indexes
        WHERE schemaname = current_schema()
          AND indexname = $1
        """,
        index_name,
    )
    return row is not None


async def ensure_message_text_embedding_index(
    conn, min_rows: int = VECTOR_INDEX_MIN_ROWS
) -> bool:
    if await _index_exists(conn, "idx_message_text_embedding"):
        return True
    row = await conn.fetchrow("SELECT COUNT(*) FROM MessageTextIndex")
    if row[0] < min_rows:
        return False
    await conn.execute(MESSAGE_TEXT_INDEX_EMBEDDING_INDEX_TEMPLATE)
    return True


async def ensure_related_terms_embedding_index(
    conn, min_rows: int = VECTOR_INDEX_MIN_ROWS
) -> bool:
    if await _index_exists(conn, "idx_related_terms_embedding"):
        return True
    row = await conn.fetchrow("SELECT COUNT(*) FROM RelatedTermsFuzzy")
    if row[0] < min_rows:
        return False
    await conn.execute(RELATED_TERMS_FUZZY_EMBEDDING_INDEX_TEMPLATE)
    return True


async def init_db_schema(
    pool, embedding_size: int = 1536, schema: str | None = None
) -> None:
    """Initialize the database schema with all required tables."""
    async with pool.acquire() as conn:
        # Optional per-collection schema
        if schema:
            await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {quote_ident(schema)}")
            await conn.execute(f"SET search_path TO {format_search_path(schema)}")

        # Enable pgvector
        await conn.execute(PGVECTOR_EXTENSION)

        # Create tables
        await conn.execute(CONVERSATION_METADATA_SCHEMA)
        await conn.execute(MESSAGES_SCHEMA)
        await conn.execute(SEMANTIC_REFS_SCHEMA)
        await conn.execute(SEMANTIC_REF_INDEX_SCHEMA)
        await conn.execute(
            MESSAGE_TEXT_INDEX_SCHEMA_TEMPLATE.format(embedding_size=embedding_size)
        )
        await conn.execute(PROPERTY_INDEX_SCHEMA)
        await conn.execute(RELATED_TERMS_ALIASES_SCHEMA)
        await conn.execute(
            RELATED_TERMS_FUZZY_SCHEMA_TEMPLATE.format(embedding_size=embedding_size)
        )
        await conn.execute(TIMESTAMP_INDEX_SCHEMA)
        await conn.execute(INGESTED_SOURCES_SCHEMA)

        # Create indexes
        await conn.execute(SEMANTIC_REF_INDEX_TERM_INDEX)
        await conn.execute(PROPERTY_INDEX_PROP_NAME_INDEX)
        await conn.execute(PROPERTY_INDEX_VALUE_STR_INDEX)
        await conn.execute(PROPERTY_INDEX_COMBINED_INDEX)
        await conn.execute(RELATED_TERMS_ALIASES_TERM_INDEX)
        await conn.execute(RELATED_TERMS_ALIASES_ALIAS_INDEX)

        # Create vector index only if there's enough data (ivfflat needs data to build)
        # This will be created lazily when enough data is added
        # await conn.execute(MESSAGE_TEXT_INDEX_EMBEDDING_INDEX_TEMPLATE)


async def get_db_schema_version(pool) -> int:
    """Get the database schema version."""
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT value FROM ConversationMetadata WHERE key = 'schema_version' LIMIT 1"
            )
            if row:
                return int(row["value"])
            return CONVERSATION_SCHEMA_VERSION
    except Exception:
        return CONVERSATION_SCHEMA_VERSION


async def set_conversation_metadata(
    pool, schema: str | None = None, **kwds: str | list[str] | None
) -> None:
    """Set or update conversation metadata key-value pairs."""
    async with pool.acquire() as conn:
        if schema:
            await conn.execute(f"SET search_path TO {format_search_path(schema)}")
        for key, value in kwds.items():
            # Delete existing rows for this key
            await conn.execute(
                "DELETE FROM ConversationMetadata WHERE key = $1", key
            )

            if value is None:
                continue
            elif isinstance(value, list):
                for v in value:
                    await conn.execute(
                        "INSERT INTO ConversationMetadata (key, value) VALUES ($1, $2)",
                        key, str(v)
                    )
            else:
                await conn.execute(
                    "INSERT INTO ConversationMetadata (key, value) VALUES ($1, $2)",
                    key, str(value)
                )
