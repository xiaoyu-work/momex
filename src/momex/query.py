"""Momex prefix-based query functions."""

from __future__ import annotations

import asyncio
from typing import Any

from .config import MomexConfig
from .manager import MemoryManager
from .memory import Memory, MemoryItem

# Maximum concurrent queries to avoid rate limiting
MAX_CONCURRENT_QUERIES = 5


async def query(
    prefix: str,
    question: str,
    config: MomexConfig | None = None,
) -> str:
    """Query memories across all collections matching a prefix.

    Args:
        prefix: Collection prefix (e.g., "momex:engineering" matches
                "momex:engineering:xiaoyuzhang", "momex:engineering:gvanrossum", etc.)
        question: Natural language question.
        config: Configuration object. If None, uses default config.

    Returns:
        Combined answer string based on stored memories.
    """
    config = config or MomexConfig.get_default()
    manager = MemoryManager(config=config)

    # Find all collections matching prefix
    collections = manager.list_collections(prefix=prefix)

    if not collections:
        return f"No collections found matching prefix '{prefix}'."

    # Query collections in parallel with concurrency limit
    sem = asyncio.Semaphore(MAX_CONCURRENT_QUERIES)

    async def query_one(coll_name: str) -> str | None:
        async with sem:
            try:
                memory = Memory(collection=coll_name, config=config)
                answer = await memory.query(question)
                if answer and not answer.startswith("No answer found"):
                    return answer
            except Exception:
                # Skip collections that fail
                pass
            return None

    results = await asyncio.gather(*[query_one(c) for c in collections])
    answers = [r for r in results if r is not None]

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
    config: MomexConfig | None = None,
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
    config = config or MomexConfig.get_default()
    manager = MemoryManager(config=config)

    # Find all collections matching prefix
    collections = manager.list_collections(prefix=prefix)

    if not collections:
        return []

    per_collection_limit = max(1, limit // len(collections)) + 5  # Get extra for re-ranking

    # Search collections in parallel with concurrency limit
    sem = asyncio.Semaphore(MAX_CONCURRENT_QUERIES)

    async def search_one(coll_name: str) -> list[MemoryItem]:
        async with sem:
            try:
                memory = Memory(collection=coll_name, config=config)
                return await memory.search(
                    query_text, limit=per_collection_limit, threshold=threshold
                )
            except Exception:
                return []

    results = await asyncio.gather(*[search_one(c) for c in collections])

    # Flatten results
    all_results: list[MemoryItem] = []
    for coll_results in results:
        all_results.extend(coll_results)

    # Re-sort all results by score across collections
    all_results.sort(key=lambda x: x.score or 0.0, reverse=True)

    return all_results[:limit]


async def stats(
    prefix: str,
    config: MomexConfig | None = None,
) -> dict[str, Any]:
    """Get statistics for all collections matching a prefix.

    Args:
        prefix: Collection prefix.
        config: Configuration object. If None, uses default config.

    Returns:
        Dict with stats per collection and totals.
    """
    config = config or MomexConfig.get_default()
    manager = MemoryManager(config=config)

    # Find all collections matching prefix
    collections = manager.list_collections(prefix=prefix)

    if not collections:
        return {
            "prefix": prefix,
            "collections": {},
            "total_memories": 0,
            "total_entities": 0,
            "collection_count": 0,
        }

    # Get stats in parallel with concurrency limit
    sem = asyncio.Semaphore(MAX_CONCURRENT_QUERIES)

    async def stats_one(coll_name: str) -> tuple[str, dict[str, Any]]:
        async with sem:
            try:
                memory = Memory(collection=coll_name, config=config)
                coll_stats = await memory.stats()
                return (coll_name, coll_stats)
            except Exception:
                return (coll_name, {"total_memories": 0, "entities_extracted": 0})

    results = await asyncio.gather(*[stats_one(c) for c in collections])

    stats_per_collection = {}
    total_memories = 0
    total_entities = 0

    for coll_name, coll_stats in results:
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
