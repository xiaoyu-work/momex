"""Memex prefix-based query functions."""

from __future__ import annotations

from typing import Any

from .config import MemexConfig
from .manager import MemoryManager
from .memory import Memory, MemoryItem


async def query(
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
    config = config or MemexConfig.get_default()
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
            answer = await memory.query(question)
            if answer and not answer.startswith("No answer found"):
                answers.append(answer)
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


async def search(
    prefix: str,
    query_text: str,
    limit: int = 10,
    threshold: float | None = None,
    config: MemexConfig | None = None,
) -> list[MemoryItem]:
    """Search memories across all collections matching a prefix using vector similarity.

    This function performs semantic search using embeddings. Results are sorted
    by similarity score, so the most relevant memories appear first.

    Use this when you want raw search results as context for a chat agent,
    without LLM summarization (cheaper than query()).

    Args:
        prefix: Collection prefix.
        query_text: Search query (natural language question or topic).
        limit: Maximum total results to return.
        threshold: Minimum similarity score (0.0-1.0). If None, uses config default.
        config: Configuration object. If None, uses default config.

    Returns:
        List of MemoryItem objects from all matching collections, sorted by score.
    """
    config = config or MemexConfig.get_default()
    manager = MemoryManager(config=config)

    # Find all collections matching prefix
    collections = manager.list_collections(prefix=prefix)

    if not collections:
        return []

    all_results: list[MemoryItem] = []
    per_collection_limit = max(1, limit // len(collections)) + 5  # Get extra for re-ranking

    for coll_name in collections:
        memory = Memory(collection=coll_name, config=config)
        coll_results = await memory.search(
            query_text, limit=per_collection_limit, threshold=threshold
        )
        all_results.extend(coll_results)

    # Re-sort all results by score across collections
    all_results.sort(key=lambda x: x.score or 0.0, reverse=True)

    return all_results[:limit]


async def stats(
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
    config = config or MemexConfig.get_default()
    manager = MemoryManager(config=config)

    # Find all collections matching prefix
    collections = manager.list_collections(prefix=prefix)

    stats_per_collection = {}
    total_memories = 0
    total_entities = 0

    for coll_name in collections:
        memory = Memory(collection=coll_name, config=config)
        coll_stats = await memory.stats()
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
