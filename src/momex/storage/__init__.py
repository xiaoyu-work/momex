"""Momex storage backends.

Provides pluggable storage backends for Momex memory system.

Available backends:
    - SQLiteBackend: Local SQLite database (default, uses TypeAgent)
    - PostgresBackend: PostgreSQL with pgvector extension

Example:
    >>> from momex import MomexConfig
    >>> from momex.storage import PostgresBackend
    >>>
    >>> # Use PostgreSQL backend
    >>> config = MomexConfig(
    ...     storage=StorageConfig(
    ...         backend="postgres",
    ...         connection_string="postgresql://user:pass@localhost/momex"
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
