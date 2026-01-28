"""Memex storage backends.

Provides pluggable storage backends for Memex memory system.

Available backends:
    - SQLiteBackend: Local SQLite database (default, uses TypeAgent)
    - PostgresBackend: PostgreSQL with pgvector extension

Example:
    >>> from memex import MemexConfig
    >>> from memex.storage import PostgresBackend
    >>>
    >>> # Use PostgreSQL backend
    >>> config = MemexConfig(
    ...     storage=StorageConfig(
    ...         backend="postgres",
    ...         connection_string="postgresql://user:pass@localhost/memex"
    ...     )
    ... )
"""

from .base import StorageBackend, StorageRecord, SearchResult
from .sqlite import SQLiteBackend

__all__ = [
    "StorageBackend",
    "StorageRecord",
    "SearchResult",
    "SQLiteBackend",
]

# Conditionally export PostgresBackend if asyncpg is available
try:
    from .postgres import PostgresBackend
    __all__.append("PostgresBackend")
except ImportError:
    pass
