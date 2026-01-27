"""Memex - Structured RAG Memory for AI Agents.

A high-level async API for structured knowledge memory, built on TypeAgent's
Structured RAG technology.

Example:
    >>> import asyncio
    >>> from memex import Memory, query
    >>>
    >>> async def main():
    ...     # Create memory with hierarchical collection name
    ...     memory = Memory(collection="company:engineering:alice")
    ...
    ...     # Add memories
    ...     await memory.add("Alice likes cats")
    ...     await memory.add("The project deadline is Friday")
    ...
    ...     # Query single collection
    ...     answer = await memory.query("What does Alice like?")
    ...
    ...     # Query with prefix (searches all matching collections)
    ...     answer = await query("company:engineering", "What are the deadlines?")
    ...
    >>> asyncio.run(main())

Configuration:
    LLM is configured via TypeAgent's environment variables:
        export OPENAI_API_KEY=sk-xxx
        export OPENAI_MODEL=gpt-4o
        # or for Azure:
        export AZURE_OPENAI_API_KEY=xxx
        export AZURE_OPENAI_ENDPOINT=https://xxx.openai.azure.com

    Memex-specific config (storage path, fact types) can be set via MemexConfig:
    >>> from memex import MemexConfig
    >>> MemexConfig.set_default(storage_path="./my_data")
"""

from .config import DEFAULT_FACT_TYPES, FactType, MemexConfig
from .exceptions import (
    CollectionNotFoundError,
    ConfigurationError,
    EmbeddingError,
    ExportError,
    LLMError,
    MemexError,
    MemoryNotFoundError,
    StorageError,
    ValidationError,
)
from .manager import MemoryManager
from .memory import (
    AddResult,
    ConversationResult,
    Memory,
    MemoryEvent,
    MemoryItem,
    MemoryOperation,
)
from .query import query, search, stats

__all__ = [
    # Core classes
    "Memory",
    "MemoryManager",
    "MemexConfig",
    "FactType",
    "DEFAULT_FACT_TYPES",
    # Data classes
    "MemoryItem",
    "MemoryEvent",
    "MemoryOperation",
    "AddResult",
    "ConversationResult",
    # Prefix query functions (async)
    "query",
    "search",
    "stats",
    # Exceptions
    "MemexError",
    "CollectionNotFoundError",
    "MemoryNotFoundError",
    "ConfigurationError",
    "ValidationError",
    "StorageError",
    "EmbeddingError",
    "LLMError",
    "ExportError",
]
