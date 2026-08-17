"""Building the TypeAgent storage provider a collection is backed by.

Momex supports SQLite (one database file per collection, nothing to set up)
and PostgreSQL (one schema per collection, connection pooling). The two are
constructed differently enough that keeping them side by side is the only way
to see what they have in common: both are handed the same index settings and
both derive their location from the collection name.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from .config import MomexConfig
from .paths import collection_to_db_path, collection_to_schema

if TYPE_CHECKING:
    from typeagent.knowpro.convsettings import (
        MessageTextIndexSettings,
        RelatedTermIndexSettings,
    )

DB_FILENAME = "memory.db"


def create_sqlite_provider(
    collection: str,
    config: MomexConfig,
    message_text_index_settings: MessageTextIndexSettings,
    related_term_index_settings: RelatedTermIndexSettings,
) -> Any:
    """Create a SQLite storage provider for one collection."""
    from typeagent.knowpro.universal_message import ConversationMessage
    from typeagent.storage.sqlite import SqliteStorageProvider

    db_path = collection_to_db_path(collection, config.storage_path, DB_FILENAME)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    storage_provider = SqliteStorageProvider(
        db_path=str(db_path),
        message_type=ConversationMessage,
        message_text_index_settings=message_text_index_settings,
        related_term_index_settings=related_term_index_settings,
    )

    # Commit any pending schema initialization transaction
    storage_provider.db.commit()

    return storage_provider


async def create_postgres_provider(
    collection: str,
    config: MomexConfig,
    message_text_index_settings: MessageTextIndexSettings,
    related_term_index_settings: RelatedTermIndexSettings,
) -> Any:
    """Create a PostgreSQL storage provider for one collection.

    Collections are isolated by schema. An explicit `storage.postgres_schema`
    puts every collection in that one schema; otherwise each gets a schema
    derived from its name.
    """
    from typeagent.knowpro.interfaces import ConversationMetadata
    from typeagent.knowpro.universal_message import ConversationMessage
    from typeagent.storage.postgres import PostgresStorageProvider

    schema = config.storage.postgres_schema or collection_to_schema(collection)

    return await PostgresStorageProvider.create(
        connection_string=config.storage.postgres_url,
        message_type=ConversationMessage,
        message_text_index_settings=message_text_index_settings,
        related_term_index_settings=related_term_index_settings,
        min_pool_size=config.storage.postgres_pool_min,
        max_pool_size=config.storage.postgres_pool_max,
        schema=schema,
        pgbouncer=config.storage.postgres_pgbouncer,
        metadata=ConversationMetadata(name_tag=collection, tags=[collection]),
    )
