"""Momex - Structured RAG Memory for AI Agents.

A high-level async API for structured knowledge memory, built on TypeAgent's
Structured RAG technology. Uses TypeAgent's full indexing system (SemanticRefs,
TermIndex) for entity/action/topic extraction and search.

Logging:
    This library uses Python's standard logging module. To see log messages,
    configure logging in your application:

        import logging
        logging.basicConfig(level=logging.INFO)

    For debug output:
        logging.basicConfig(level=logging.DEBUG)

Example:
    >>> import asyncio
    >>> from momex import Memory, MomexConfig
    >>>
    >>> async def main():
    ...     # Configure LLM (required)
    ...     config = MomexConfig(
    ...         provider="openai",  # openai, azure, anthropic, deepseek, qwen
    ...         model="gpt-4o",
    ...         api_key="sk-xxx",
    ...     )
    ...
    ...     # Create memory
    ...     memory = Memory(collection="user:xiaoyuzhang", config=config)
    ...
    ...     # Add memories - TypeAgent extracts entities, actions, topics
    ...     await memory.add("I like Python programming")
    ...
    ...     # Query
    ...     answer = await memory.query("What does the user like?")
    ...
    >>> asyncio.run(main())

Configuration:
    LLM can be configured via code, YAML, or environment variables.
    The same config is used for both TypeAgent (knowledge extraction) and
    Momex (contradiction detection).

    Supported providers: openai, azure, anthropic, deepseek, qwen

    Code:
        config = MomexConfig(provider="openai", model="gpt-4o", api_key="sk-xxx")

    YAML:
        config = MomexConfig.from_yaml("config.yaml")

    Environment variables:
        export MOMEX_PROVIDER=openai
        export MOMEX_MODEL=gpt-4o
        export MOMEX_API_KEY=sk-xxx
"""

import logging

from .config import MomexConfig, PostgresConfig
from .exceptions import (
    CollectionNotFoundError,
    ConfigurationError,
    ExportError,
    LLMError,
    MomexError,
    MemoryNotFoundError,
    StorageError,
    ValidationError,
)
from .manager import MemoryManager
from .memory import AddResult, Memory, SearchItem
from .query import query, search, stats

__all__ = [
    # Core classes
    "Memory",
    "MemoryManager",
    "MomexConfig",
    "PostgresConfig",
    # Data classes
    "AddResult",
    "SearchItem",
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
    "LLMError",
    "ExportError",
]

# Set up library-level logging with NullHandler to prevent
# "No handler found" warnings when the library is used.
# Users can configure logging in their application code.
logging.getLogger(__name__).addHandler(logging.NullHandler())
