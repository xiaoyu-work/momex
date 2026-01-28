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

    Converts "user:alice" to Path("user/alice") for cross-platform compatibility.
    """
    parts = collection.split(":")
    sanitized = [re.sub(r'[<>"|?*:\\]', '_', part) for part in parts]
    return Path(*sanitized)


def _path_to_collection(path: Path) -> str:
    """Convert path back to collection name.

    Converts Path("user/alice") to "user:alice".
    """
    return ":".join(path.parts)


class MemoryManager:
    """Manage memory collections (list, delete, rename, info).

    Example:
        >>> from momex import MemoryManager
        >>> manager = MemoryManager()
        >>> collections = manager.list_collections()
        >>> manager.delete("user:old_user")
        >>> manager.rename("user:alice", "user:alice_backup")
    """

    def __init__(self, config: MomexConfig | None = None) -> None:
        """Initialize MemoryManager.

        Args:
            config: Configuration object. If None, uses default config.
        """
        self.config = config or MomexConfig()
        self._storage_path = Path(self.config.storage.path)

    def list_collections(self, prefix: str | None = None) -> list[str]:
        """List all collections, optionally filtered by prefix.

        Args:
            prefix: Optional prefix to filter collections.
                    e.g., "company:engineering" matches "company:engineering:alice"

        Returns:
            List of collection names.
        """
        collections = []

        if not self._storage_path.exists():
            return collections

        # Walk through storage directory using pathlib
        for db_file in self._storage_path.rglob(self.config.db_name):
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
        collection_dir = self._get_collection_dir(collection)

        if not collection_dir.exists():
            return False

        shutil.rmtree(collection_dir)

        # Clean up empty parent directories
        self._cleanup_empty_dirs(collection_dir.parent)

        return True

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
        return self._get_collection_dir(collection) / self.config.db_name

    def _cleanup_empty_dirs(self, path: Path) -> None:
        """Remove empty parent directories up to storage_path."""
        while path and path != self._storage_path:
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()
                path = path.parent
            else:
                break
