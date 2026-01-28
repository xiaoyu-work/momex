"""Momex prefix-based query functions using TypeAgent's full indexing."""

from __future__ import annotations

import asyncio
from typing import Any

from .config import MomexConfig
from .manager import MemoryManager
from .memory import Memory, SearchItem

# Maximum concurrent queries to avoid rate limiting
MAX_CONCURRENT_QUERIES = 5


async def query(
    prefix: str,
    question: str,
    config: MomexConfig | None = None,
) -> str:
    """Query memories across all collections matching a prefix.

    This function searches all matching collections using TypeAgent's
    term-based indexing, then sends results to LLM to generate an answer.

    Args:
        prefix: Collection prefix (e.g., "momex:engineering" matches
                "momex:engineering:xiaoyuzhang", "momex:engineering:gvanrossum", etc.)
        question: Natural language question.
        config: Configuration object. If None, uses default config.

    Returns:
        Answer string based on stored memories from all matching collections.
    """
    config = config or MomexConfig.get_default()

    # Search all collections matching prefix
    results = await search(prefix, question, limit=20, config=config)

    if not results:
        return "No relevant memories found."

    # Build context from search results
    context = "\n".join([
        f"- [{coll}] ({item.type}): {item.text}"
        for coll, items in results
        for item in items
    ])

    if not context:
        return "No relevant memories found."

    # Use LLM to answer the question
    from typeagent.knowpro import convknowledge
    import typechat

    model = convknowledge.create_typechat_model()
    prompt = f"""Based on the following memories from different people/collections, answer the question.

Memories:
{context}

Question: {question}

Answer concisely based only on the memories provided."""

    try:
        result = await model.complete(prompt)
        if isinstance(result, typechat.Success):
            return result.value
        else:
            return f"Failed to generate answer: {result.message}"
    except Exception as e:
        return f"Error querying memories: {e}"


async def search(
    prefix: str,
    query_text: str,
    limit: int = 10,
    config: MomexConfig | None = None,
) -> list[tuple[str, list[SearchItem]]]:
    """Search memories across all collections matching a prefix.

    Args:
        prefix: Collection prefix.
        query_text: Search query (natural language question or topic).
        limit: Maximum results per collection.
        config: Configuration object. If None, uses default config.

    Returns:
        List of (collection_name, list[SearchItem]) tuples.
    """
    config = config or MomexConfig.get_default()
    manager = MemoryManager(config=config)

    # Find all collections matching prefix
    collections = manager.list_collections(prefix=prefix)

    if not collections:
        return []

    # Search collections in parallel with concurrency limit
    sem = asyncio.Semaphore(MAX_CONCURRENT_QUERIES)

    async def search_one(coll_name: str) -> tuple[str, list[SearchItem]]:
        async with sem:
            try:
                memory = Memory(collection=coll_name, config=config)
                results = await memory.search(query_text, limit=limit)
                return (coll_name, results)
            except Exception:
                return (coll_name, [])

    results = await asyncio.gather(*[search_one(c) for c in collections])

    # Filter out empty results
    return [(name, items) for name, items in results if items]


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
            "total_messages": 0,
            "total_semantic_refs": 0,
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
                return (coll_name, {"total_messages": 0, "total_semantic_refs": 0})

    results = await asyncio.gather(*[stats_one(c) for c in collections])

    stats_per_collection = {}
    total_messages = 0
    total_semrefs = 0

    for coll_name, coll_stats in results:
        stats_per_collection[coll_name] = coll_stats
        total_messages += coll_stats.get("total_messages", 0)
        total_semrefs += coll_stats.get("total_semantic_refs", 0)

    return {
        "prefix": prefix,
        "collections": stats_per_collection,
        "total_messages": total_messages,
        "total_semantic_refs": total_semrefs,
        "collection_count": len(collections),
    }
