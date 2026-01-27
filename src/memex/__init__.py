"""Memex - Structured RAG Memory for AI Agents.

A high-level API for structured knowledge memory, built on TypeAgent's
Structured RAG technology.

Example - Add memories:
    >>> from memex import Memory
    >>>
    >>> # Create memory with hierarchical collection name
    >>> memory = Memory(collection="company:engineering:alice")
    >>>
    >>> # Add memories
    >>> memory.add("Alice likes cats")
    >>> memory.add("The project deadline is Friday")

Example - Query with prefix (hierarchical):
    >>> from memex import query
    >>>
    >>> # Query single person
    >>> answer = query("company:engineering:alice", "What does Alice like?")
    >>>
    >>> # Query entire team (matches all under prefix)
    >>> answer = query("company:engineering", "What are the deadlines?")
    >>>
    >>> # Query entire company
    >>> answer = query("company", "Who likes cats?")

Example - Managing Collections:
    >>> from memex import MemoryManager
    >>>
    >>> manager = MemoryManager()
    >>> collections = manager.list_collections()  # all
    >>> collections = manager.list_collections(prefix="company:engineering")  # filtered
    >>> manager.delete("company:engineering:old_user")

Configuration:
    LLM is configured via TypeAgent's environment variables:
        export OPENAI_API_KEY=sk-xxx
        # or for Azure:
        export AZURE_OPENAI_API_KEY=xxx
        export AZURE_OPENAI_ENDPOINT=https://xxx.openai.azure.com

    Memex-specific config (storage path, fact types) can be set via MemexConfig:
    >>> from memex import Memory, MemexConfig
    >>>
    >>> MemexConfig.set_default(storage_path="./my_data")
    >>> memory = Memory(collection="user:alice")
"""

from .config import DEFAULT_FACT_TYPES, FactType, MemexConfig
from .manager import MemoryManager
from .memory import (
    AddResult,
    ConversationResult,
    Memory,
    MemoryEvent,
    MemoryItem,
    MemoryOperation,
)
from .query import query, search, stats, query_async, search_async, stats_async

__all__ = [
    "Memory",
    "MemoryManager",
    "MemexConfig",
    "FactType",
    "DEFAULT_FACT_TYPES",
    "MemoryItem",
    "MemoryEvent",
    "MemoryOperation",
    "AddResult",
    "ConversationResult",
    # Prefix query functions
    "query",
    "search",
    "stats",
    "query_async",
    "search_async",
    "stats_async",
]
