"""Memex - Structured RAG Memory for AI Agents.

A high-level API for structured knowledge memory, built on TypeAgent's
Structured RAG technology.

Example - Single Collection:
    >>> from memex import Memory
    >>>
    >>> # Create memory for a collection
    >>> memory = Memory(collection="user:alice")
    >>>
    >>> # Add memories
    >>> memory.add("Alice likes cats")
    >>> memory.add("The project deadline is Friday")
    >>>
    >>> # Query with natural language
    >>> answer = memory.query("What does Alice like?")
    >>> print(answer)  # "Alice likes cats"

Example - Multiple Collections with MemoryPool:
    >>> from memex import MemoryPool
    >>>
    >>> # Create a pool with multiple collections
    >>> pool = MemoryPool(
    ...     collections=["user:alice", "team:engineering", "project:x"],
    ...     default_collection="user:alice"
    ... )
    >>>
    >>> # Add to specific collections
    >>> pool.add("Personal note", collections=["user:alice"])
    >>> pool.add("Team decision", collections=["team:engineering", "project:x"])
    >>>
    >>> # Query across all collections
    >>> answer = pool.query("What decisions were made?")

Example - Managing Collections:
    >>> from memex import MemoryManager
    >>>
    >>> manager = MemoryManager()
    >>> collections = manager.list_collections()
    >>> manager.delete("user:old_user")

Configuration:
    >>> from memex import Memory, MemexConfig
    >>>
    >>> config = MemexConfig(
    ...     storage_path="./data",
    ...     llm_provider="openai",
    ...     llm_model="gpt-4o",
    ... )
    >>> memory = Memory(collection="user:alice", config=config)
"""

from .config import MemexConfig
from .manager import MemoryManager
from .memory import AddResult, Memory, MemoryItem
from .pool import MemoryPool

__all__ = [
    "Memory",
    "MemoryPool",
    "MemoryManager",
    "MemexConfig",
    "MemoryItem",
    "AddResult",
]

__version__ = "0.2.0"
