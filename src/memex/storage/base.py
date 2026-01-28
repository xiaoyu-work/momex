"""Storage backend protocol for Memex.

Defines the interface that all storage backends must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class StorageRecord:
    """A single record in storage."""

    id: int
    text: str
    speaker: str | None = None
    timestamp: str | None = None
    importance: float = 0.5
    deleted: bool = False
    metadata: dict[str, Any] | None = None


@dataclass
class SearchResult:
    """A search result with similarity score."""

    record: StorageRecord
    similarity: float


class StorageBackend(ABC):
    """Abstract base class for storage backends.

    All storage backends must implement these methods to be compatible
    with Memex. This allows swapping between SQLite, PostgreSQL, etc.

    Example implementation:
        class MyBackend(StorageBackend):
            async def add(self, text, embedding, ...) -> int:
                # Store the record
                return record_id

            async def search(self, embedding, limit, threshold) -> list[SearchResult]:
                # Vector similarity search
                return results
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the storage backend (create tables, etc.)."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the storage backend connection."""
        ...

    @abstractmethod
    async def add(
        self,
        text: str,
        embedding: list[float],
        speaker: str | None = None,
        timestamp: str | None = None,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Add a new record to storage.

        Args:
            text: The text content to store.
            embedding: Vector embedding of the text.
            speaker: Who said this (optional).
            timestamp: ISO timestamp (optional).
            importance: Importance score 0.0-1.0.
            metadata: Additional metadata.

        Returns:
            The ID of the newly created record.
        """
        ...

    @abstractmethod
    async def get(self, record_id: int) -> StorageRecord | None:
        """Get a record by ID.

        Args:
            record_id: The record ID.

        Returns:
            The record, or None if not found.
        """
        ...

    @abstractmethod
    async def search(
        self,
        embedding: list[float],
        limit: int = 10,
        threshold: float = 0.3,
    ) -> list[SearchResult]:
        """Search for similar records using vector similarity.

        Args:
            embedding: Query embedding vector.
            limit: Maximum number of results.
            threshold: Minimum similarity score (0.0-1.0).

        Returns:
            List of SearchResult objects sorted by similarity (descending).
        """
        ...

    @abstractmethod
    async def delete(self, record_id: int) -> bool:
        """Soft delete a record.

        Args:
            record_id: The record ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        ...

    @abstractmethod
    async def restore(self, record_id: int) -> bool:
        """Restore a soft-deleted record.

        Args:
            record_id: The record ID to restore.

        Returns:
            True if restored, False if not found or not deleted.
        """
        ...

    @abstractmethod
    async def list_deleted(self) -> list[StorageRecord]:
        """List all soft-deleted records.

        Returns:
            List of deleted records.
        """
        ...

    @abstractmethod
    async def update_importance(self, record_id: int, importance: float) -> bool:
        """Update the importance score of a record.

        Args:
            record_id: The record ID.
            importance: New importance score (0.0-1.0).

        Returns:
            True if updated, False if not found.
        """
        ...

    @abstractmethod
    async def clear(self) -> None:
        """Delete all records (for testing/reset)."""
        ...

    @abstractmethod
    async def count(self) -> int:
        """Get total count of active (non-deleted) records.

        Returns:
            Number of active records.
        """
        ...

    @abstractmethod
    async def count_deleted(self) -> int:
        """Get count of deleted records.

        Returns:
            Number of deleted records.
        """
        ...

    @abstractmethod
    async def get_all_records(self, include_deleted: bool = False) -> list[StorageRecord]:
        """Get all records.

        Args:
            include_deleted: Whether to include deleted records.

        Returns:
            List of all records.
        """
        ...
