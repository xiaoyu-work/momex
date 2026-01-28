"""Momex - Structured RAG Memory for AI Agents.

A high-level async API for structured knowledge memory, built on TypeAgent's
Structured RAG technology.

Example:
    >>> import asyncio
    >>> from momex import Memory, query
    >>>
    >>> async def main():
    ...     # Create memory with hierarchical collection name
    ...     memory = Memory(collection="momex:engineering:xiaoyuzhang")
    ...
    ...     # Add memories - LLM extracts facts and deduplicates automatically
    ...     await memory.add("Xiaoyuzhang likes cats")
    ...     await memory.add("I really love cats")  # Deduplicates with above
    ...
    ...     # Or pass conversation format
    ...     await memory.add([
    ...         {"role": "user", "content": "The deadline is Friday"},
    ...         {"role": "assistant", "content": "Got it!"},
    ...     ])
    ...
    ...     # Direct storage without LLM processing
    ...     await memory.add("Raw log entry", infer=False)
    ...
    ...     # Query single collection
    ...     answer = await memory.query("What does Xiaoyuzhang like?")
    ...
    ...     # Query with prefix (searches all matching collections)
    ...     answer = await query("momex:engineering", "What are the deadlines?")
    ...
    >>> asyncio.run(main())

Configuration:
    LLM is configured via TypeAgent's environment variables:
        export OPENAI_API_KEY=sk-xxx
        export OPENAI_MODEL=gpt-4o
        # or for Azure:
        export AZURE_OPENAI_API_KEY=xxx
        export AZURE_OPENAI_ENDPOINT=https://xxx.openai.azure.com

    Momex-specific config (storage path, fact types) can be set via MomexConfig:
    >>> from momex import MomexConfig
    >>> MomexConfig.set_default(storage_path="./my_data")
"""

from .config import DEFAULT_FACT_TYPES, FactType, MomexConfig, StorageConfig
from .exceptions import (
    CollectionNotFoundError,
    ConfigurationError,
    EmbeddingError,
    ExportError,
    LLMError,
    MomexError,
    MemoryNotFoundError,
    StorageError,
    ValidationError,
)
from .manager import MemoryManager
from .memory import (
    AddResult,
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
    "MomexConfig",
    "StorageConfig",
    "FactType",
    "DEFAULT_FACT_TYPES",
    # Data classes
    "MemoryItem",
    "MemoryEvent",
    "MemoryOperation",
    "AddResult",
    # Prefix query functions (async)
    "query",
    "search",
    "stats",
    # Exceptions
    "MomexError",
    "CollectionNotFoundError",
    "MemoryNotFoundError",
    "ConfigurationError",
    "ValidationError",
    "StorageError",
    "EmbeddingError",
    "LLMError",
    "ExportError",
]
