"""Memex - Structured RAG Memory for AI Agents.

A high-level API for structured knowledge memory, built on TypeAgent's
Structured RAG technology.

Example:
    >>> from memex import Memory
    >>>
    >>> # Create memory for a user
    >>> memory = Memory(user_id="user_123")
    >>>
    >>> # Add memories
    >>> memory.add("张三说下周五完成API开发")
    >>> memory.add("李四负责前端，王五负责后端")
    >>>
    >>> # Query with natural language
    >>> answer = memory.query("谁负责API?")
    >>> print(answer)  # "张三负责API开发"
    >>>
    >>> # Search by keyword
    >>> results = memory.search("张三")
    >>> for item in results:
    ...     print(item.text)

Multi-tenant support:
    >>> # Isolated memory per user/org
    >>> memory = Memory(
    ...     user_id="user_123",
    ...     org_id="company_abc",
    ... )

Configuration:
    >>> from memex import Memory, MemexConfig
    >>>
    >>> config = MemexConfig(
    ...     storage_path="./data",
    ...     llm_provider="openai",
    ...     llm_model="gpt-4o",
    ... )
    >>> memory = Memory(user_id="xxx", config=config)
"""

from .config import MemexConfig
from .memory import AddResult, Memory, MemoryItem

__all__ = [
    "Memory",
    "MemexConfig",
    "MemoryItem",
    "AddResult",
]

__version__ = "0.1.0"
