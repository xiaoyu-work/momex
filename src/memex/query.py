"""Memex prefix-based query functions."""

from __future__ import annotations

from typing import Any

from .config import MemexConfig
from .manager import MemoryManager
from .memory import Memory, MemoryItem
from .sync import run_sync


async def query_async(
    prefix: str,
    question: str,
    config: MemexConfig | None = None,
) -> str:
    """Query memories across all collections matching a prefix.

    Args:
        prefix: Collection prefix (e.g., "company:engineering" matches
                "company:engineering:alice", "company:engineering:bob", etc.)
        question: Natural language question.
        config: Configuration object. If None, uses default config.

    Returns:
        Combined answer string based on stored memories.
    """
    config = config or MemexConfig()
    manager = MemoryManager(config=config)

    # Find all collections matching prefix
    collections = manager.list_collections(prefix=prefix)

    if not collections:
        return f"No collections found matching prefix '{prefix}'."

    # Query each collection
    answers = []
    for coll_name in collections:
        memory = Memory(collection=coll_name, config=config)
        try:
            answer = await memory.query_async(question)
            if answer and not answer.startswith("No answer found"):
                answers.append(f"[{coll_name}] {answer}")
        except Exception:
            # Skip collections that fail
            pass

    if not answers:
        return "No answer found in any collection."

    # Combine answers
    if len(answers) == 1:
        return answers[0]
    else:
        return "\n\n".join(answers)


async def search_async(
    prefix: str,
    query_text: str,
    limit: int = 10,
    config: MemexConfig | None = None,
) -> list[MemoryItem]:
    """Search memories across all collections matching a prefix.

    Args:
        prefix: Collection prefix.
        query_text: Search query (keyword, entity name, or topic).
        limit: Maximum total results to return.
        config: Configuration object. If None, uses default config.

    Returns:
        List of MemoryItem objects from all matching collections.
    """
    config = config or MemexConfig()
    manager = MemoryManager(config=config)

    # Find all collections matching prefix
    collections = manager.list_collections(prefix=prefix)

    if not collections:
        return []

    results: list[MemoryItem] = []
    per_collection_limit = max(1, limit // len(collections))

    for coll_name in collections:
        if len(results) >= limit:
            break
        memory = Memory(collection=coll_name, config=config)
        coll_results = await memory.search_async(query_text, limit=per_collection_limit)
        results.extend(coll_results)

    return results[:limit]


async def stats_async(
    prefix: str,
    config: MemexConfig | None = None,
) -> dict[str, Any]:
    """Get statistics for all collections matching a prefix.

    Args:
        prefix: Collection prefix.
        config: Configuration object. If None, uses default config.

    Returns:
        Dict with stats per collection and totals.
    """
    config = config or MemexConfig()
    manager = MemoryManager(config=config)

    # Find all collections matching prefix
    collections = manager.list_collections(prefix=prefix)

    stats_per_collection = {}
    total_memories = 0
    total_entities = 0

    for coll_name in collections:
        memory = Memory(collection=coll_name, config=config)
        coll_stats = await memory.stats_async()
        stats_per_collection[coll_name] = coll_stats
        total_memories += coll_stats.get("total_memories", 0)
        total_entities += coll_stats.get("entities_extracted", 0)

    return {
        "prefix": prefix,
        "collections": stats_per_collection,
        "total_memories": total_memories,
        "total_entities": total_entities,
        "collection_count": len(collections),
    }


# Sync wrappers
def query(prefix: str, question: str, config: MemexConfig | None = None) -> str:
    """Query memories across all collections matching a prefix (sync)."""
    return run_sync(query_async(prefix, question, config))


def search(
    prefix: str,
    query_text: str,
    limit: int = 10,
    config: MemexConfig | None = None,
) -> list[MemoryItem]:
    """Search memories across all collections matching a prefix (sync)."""
    return run_sync(search_async(prefix, query_text, limit, config))


def stats(prefix: str, config: MemexConfig | None = None) -> dict[str, Any]:
    """Get statistics for all collections matching a prefix (sync)."""
    return run_sync(stats_async(prefix, config))
