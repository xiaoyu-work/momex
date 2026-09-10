"""Momex prefix-based query functions using TypeAgent's full indexing."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .config import MomexConfig
from .manager import MemoryManager
from .memory import Memory
from .results import SearchItem
from .search import RRF_K
from .timewindow import normalize_as_of

logger = logging.getLogger(__name__)

# Maximum concurrent queries to avoid rate limiting
MAX_CONCURRENT_QUERIES = 5


async def search(
    prefix: str,
    query_text: str,
    limit: int = 10,
    config: MomexConfig | None = None,
    *,
    as_of: str | None = None,
    include_expired: bool = False,
    include_superseded: bool = False,
    include_unconfirmed: bool = False,
    neighbors: int = 0,
    total_limit: int | None = None,
) -> list[tuple[str, list[SearchItem]]]:
    """Search memories across all collections matching a prefix.

    Args:
        prefix: Collection prefix.
        query_text: Search query (natural language question or topic).
        limit: Maximum results per collection.
        total_limit: Optional maximum across all collections, ranked by fusion score.
        config: Configuration object. If None, uses default config.

    Returns:
        List of (collection_name, list[SearchItem]) tuples.
    """
    as_of = normalize_as_of(as_of)
    if limit < 0 or neighbors < 0 or (total_limit is not None and total_limit < 0):
        raise ValueError("search limits and neighbors cannot be negative")
    if limit == 0 or total_limit == 0:
        return []
    per_collection = min(limit, total_limit) if total_limit is not None else limit
    config = config or MomexConfig.get_default()
    manager = MemoryManager(config=config)

    # Find all collections matching prefix
    if config.is_postgres:
        collections = await manager.list_collections_async(prefix=prefix)
    else:
        collections = manager.list_collections(prefix=prefix)

    if not collections:
        return []

    # Search collections in parallel with concurrency limit
    sem = asyncio.Semaphore(MAX_CONCURRENT_QUERIES)

    async def search_one(coll_name: str) -> tuple[str, list[SearchItem]]:
        async with sem:
            memory = Memory(collection=coll_name, config=config)
            try:
                results = await memory.search(
                    query_text,
                    limit=per_collection,
                    as_of=as_of,
                    include_expired=include_expired,
                    include_superseded=include_superseded,
                    include_unconfirmed=include_unconfirmed,
                    neighbors=neighbors,
                )
                return (coll_name, results)
            except Exception:
                logger.warning(
                    "Search failed for collection %r; skipping it.",
                    coll_name,
                    exc_info=True,
                )
                return (coll_name, [])
            finally:
                await memory.close()

    results = await asyncio.gather(*[search_one(c) for c in collections])

    if total_limit is not None:
        ranked = [
            (
                name,
                item,
                (
                    item.fusion_score
                    if item.fusion_score is not None
                    else 1 / (RRF_K + rank + 1)
                ),
            )
            for name, items in results
            for rank, item in enumerate(items)
        ]
        ranked.sort(key=lambda entry: entry[2], reverse=True)
        grouped: dict[str, list[SearchItem]] = {}
        for name, item, _ in ranked[:total_limit]:
            grouped.setdefault(name, []).append(item)
        return list(grouped.items())

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
    if config.is_postgres:
        collections = await manager.list_collections_async(prefix=prefix)
    else:
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
            memory = Memory(collection=coll_name, config=config)
            try:
                coll_stats = await memory.stats()
                return (coll_name, coll_stats)
            except Exception:
                logger.warning(
                    "Stats failed for collection %r; reporting zeros.",
                    coll_name,
                    exc_info=True,
                )
                return (coll_name, {"total_messages": 0, "total_semantic_refs": 0})
            finally:
                await memory.close()

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
