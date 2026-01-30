"""MemoryManager - Manage collections (list, delete, rename, etc.)."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from .config import MomexConfig
from .exceptions import CollectionNotFoundError, ValidationError


def _collection_to_path(collection: str) -> Path:
    """Convert collection name to path.

    Converts "user:xiaoyuzhang" to Path("user/xiaoyuzhang") for cross-platform compatibility.
    """
    parts = collection.split(":")
    sanitized = [re.sub(r'[<>"|?*:\\]', '_', part) for part in parts]
    return Path(*sanitized)


def _path_to_collection(path: Path) -> str:
    """Convert path back to collection name.

    Converts Path("user/xiaoyuzhang") to "user:xiaoyuzhang".
    """
    return ":".join(path.parts)


class MemoryManager:
    """Manage memory collections (list, delete, rename, info).

    Example:
        >>> from momex import MemoryManager
        >>> manager = MemoryManager()
        >>> collections = manager.list_collections()
        >>> manager.delete("user:old_user")
        >>> manager.rename("user:xiaoyuzhang", "user:xiaoyuzhang_backup")
    """

    def __init__(self, config: MomexConfig | None = None) -> None:
        """Initialize MemoryManager.

        Args:
            config: Configuration object. If None, uses default config.
        """
        self.config = config or MomexConfig()
        self._storage_path = Path(self.config.storage_path)

    def list_collections(self, prefix: str | None = None) -> list[str]:
        """List all collections, optionally filtered by prefix.

        Args:
            prefix: Optional prefix to filter collections.
                    e.g., "momex:engineering" matches "momex:engineering:xiaoyuzhang"

        Returns:
            List of collection names.
        """
        if self.config.is_postgres:
            import asyncio

            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(self.list_collections_async(prefix=prefix))
            raise RuntimeError(
                "PostgreSQL backend requires async collection listing. "
                "Use await list_collections_async()."
            )

        collections = []

        if not self._storage_path.exists():
            return collections

        # Walk through storage directory using pathlib
        for db_file in self._storage_path.rglob("memory.db"):
            # Get relative path from storage root to the directory containing db
            rel_path = db_file.parent.relative_to(self._storage_path)
            # Convert path to collection name
            collection_name = _path_to_collection(rel_path)
            if collection_name:
                # Filter by prefix if specified
                if prefix is None:
                    collections.append(collection_name)
                elif collection_name == prefix or collection_name.startswith(prefix + ":"):
                    collections.append(collection_name)

        return sorted(collections)

    async def list_collections_async(self, prefix: str | None = None) -> list[str]:
        """List all collections asynchronously (required for PostgreSQL)."""
        if not self.config.is_postgres:
            return self.list_collections(prefix=prefix)

        import asyncpg
        from typeagent.storage.postgres.schema import quote_ident

        conn = await asyncpg.connect(self.config.postgres.url)
        try:
            schemas = await conn.fetch(
                """
                SELECT nspname
                FROM pg_namespace
                WHERE nspname NOT LIKE 'pg_%'
                  AND nspname <> 'information_schema'
                """
            )

            collections: list[str] = []
            for row in schemas:
                schema = row[0]
                has_table = await conn.fetchval(
                    """
                    SELECT 1
                    FROM pg_tables
                    WHERE schemaname = $1 AND tablename = 'conversationmetadata'
                    LIMIT 1
                    """,
                    schema,
                )
                if not has_table:
                    continue

                tag_rows = await conn.fetch(
                    f"SELECT value FROM {quote_ident(schema)}.ConversationMetadata WHERE key = $1",
                    "tag",
                )
                names = [r[0] for r in tag_rows]
                if not names:
                    name_tag = await conn.fetchval(
                        f"SELECT value FROM {quote_ident(schema)}.ConversationMetadata WHERE key = $1 LIMIT 1",
                        "name_tag",
                    )
                    if name_tag:
                        names = [name_tag]

                for name in names:
                    if prefix is None or name == prefix or name.startswith(prefix + ":"):
                        collections.append(name)

            return sorted(set(collections))
        finally:
            await conn.close()

    def exists(self, collection: str) -> bool:
        """Check if a collection exists.

        Args:
            collection: Collection name.

        Returns:
            True if the collection exists.
        """
        db_path = self._get_db_path(collection)
        return db_path.exists()

    def delete(self, collection: str) -> bool:
        """Delete a collection and all its data.

        Args:
            collection: Collection name.

        Returns:
            True if deleted, False if not found.
        """
        if self.config.is_postgres:
            import asyncio

            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(self.delete_async(collection))
            raise RuntimeError(
                "PostgreSQL backend requires async delete. "
                "Use await delete_async()."
            )

        collection_dir = self._get_collection_dir(collection)

        if not collection_dir.exists():
            return False

        shutil.rmtree(collection_dir)

        # Clean up empty parent directories
        self._cleanup_empty_dirs(collection_dir.parent)

        return True

    async def delete_async(self, collection: str) -> bool:
        """Delete a collection asynchronously (required for PostgreSQL)."""
        if not self.config.is_postgres:
            return self.delete(collection)

        import asyncpg
        from typeagent.storage.postgres.schema import quote_ident
        from .memory import _collection_to_schema

        schema = (
            self.config.postgres.schema
            if self.config.postgres.schema
            else _collection_to_schema(collection)
        )

        conn = await asyncpg.connect(self.config.postgres.url)
        try:
            exists = await conn.fetchval(
                "SELECT 1 FROM pg_namespace WHERE nspname = $1",
                schema,
            )
            if not exists:
                return False

            await conn.execute(f"DROP SCHEMA {quote_ident(schema)} CASCADE")
            return True
        finally:
            await conn.close()

    def rename(self, old_name: str, new_name: str) -> bool:
        """Rename a collection.

        Args:
            old_name: Current collection name.
            new_name: New collection name.

        Returns:
            True if renamed, False if source not found.
        """
        old_dir = self._get_collection_dir(old_name)
        new_dir = self._get_collection_dir(new_name)

        if not old_dir.exists():
            return False

        if new_dir.exists():
            raise ValidationError(
                message=f"Collection '{new_name}' already exists.",
                field="new_name",
                value=new_name,
                suggestion="Choose a different name or delete the existing collection first.",
            )

        # Ensure parent directory exists
        new_dir.parent.mkdir(parents=True, exist_ok=True)

        shutil.move(old_dir, new_dir)

        # Clean up empty parent directories
        self._cleanup_empty_dirs(old_dir.parent)

        return True

    def info(self, collection: str) -> dict[str, Any]:
        """Get information about a collection.

        Args:
            collection: Collection name.

        Returns:
            Dict with collection info (size, path, etc.).
        """
        db_path = self._get_db_path(collection)

        if not db_path.exists():
            raise CollectionNotFoundError(collection=collection)

        # Get file stats
        stat = db_path.stat()
        size_bytes = stat.st_size
        if size_bytes < 1024:
            size_str = f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            size_str = f"{size_bytes / 1024:.1f} KB"
        else:
            size_str = f"{size_bytes / (1024 * 1024):.1f} MB"

        return {
            "collection": collection,
            "db_path": str(db_path),
            "size_bytes": size_bytes,
            "size": size_str,
            "modified_timestamp": stat.st_mtime,
        }

    def copy(self, source: str, destination: str) -> bool:
        """Copy a collection to a new name.

        Args:
            source: Source collection name.
            destination: Destination collection name.

        Returns:
            True if copied.
        """
        source_dir = self._get_collection_dir(source)
        dest_dir = self._get_collection_dir(destination)

        if not source_dir.exists():
            raise CollectionNotFoundError(collection=source)

        if dest_dir.exists():
            raise ValidationError(
                message=f"Destination collection '{destination}' already exists.",
                field="destination",
                value=destination,
                suggestion="Choose a different name or delete the existing collection first.",
            )

        # Ensure parent directory exists
        dest_dir.parent.mkdir(parents=True, exist_ok=True)

        shutil.copytree(source_dir, dest_dir)

        return True

    def _get_collection_dir(self, collection: str) -> Path:
        """Get the directory path for a collection."""
        return self._storage_path / _collection_to_path(collection)

    def _get_db_path(self, collection: str) -> Path:
        """Get the database file path for a collection."""
        return self._get_collection_dir(collection) / "memory.db"

    def _cleanup_empty_dirs(self, path: Path) -> None:
        """Remove empty parent directories up to storage_path."""
        while path and path != self._storage_path:
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()
                path = path.parent
            else:
                break
