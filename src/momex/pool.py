"""MemoryPool - Aggregate multiple collections for querying and adding."""

from __future__ import annotations

from typing import Any

from .config import MomexConfig
from .memory import AddResult, Memory, MemoryItem
from .sync import run_sync


class MemoryPool:
    """Aggregate multiple collections for unified querying and adding.

    Example:
        >>> from momex import MemoryPool
        >>> pool = MemoryPool(
        ...     collections=["user:xiaoyuzhang", "team:engineering", "project:x"],
        ...     default_collection="user:xiaoyuzhang"
        ... )
        >>> pool.add("Personal note")  # Goes to default collection
        >>> pool.add("Team decision", collections=["team:engineering"])
        >>> answer = pool.query("What decisions were made?")  # Searches all
    """

    def __init__(
        self,
        collections: list[str],
        default_collection: str | None = None,
        config: MomexConfig | None = None,
    ) -> None:
        """Initialize MemoryPool with multiple collections.

        Args:
            collections: List of collection names to include in the pool.
            default_collection: Default collection for add() without explicit collections.
                Must be one of the collections in the list.
            config: Configuration object. If None, uses default config.
        """
        if not collections:
            raise ValueError("At least one collection is required")

        self.collection_names = collections
        self.config = config or MomexConfig()

        if default_collection and default_collection not in collections:
            raise ValueError(
                f"default_collection '{default_collection}' must be in collections list"
            )
        self.default_collection = default_collection

        # Create Memory instances for each collection
        self._memories: dict[str, Memory] = {}
        for name in collections:
            self._memories[name] = Memory(collection=name, config=self.config)

    def _get_target_collections(
        self, collections: list[str] | None
    ) -> list[str]:
        """Resolve target collections for an operation."""
        if collections:
            # Validate all collections are in the pool
            for c in collections:
                if c not in self._memories:
                    raise ValueError(
                        f"Collection '{c}' is not in this pool. "
                        f"Available: {self.collection_names}"
                    )
            return collections
        elif self.default_collection:
            return [self.default_collection]
        else:
            raise ValueError(
                "No collections specified and no default_collection set. "
                "Either specify collections=[...] or set default_collection."
            )

    # =========================================================================
    # Async API
    # =========================================================================

    async def add_async(
        self,
        text: str,
        collections: list[str] | None = None,
        speaker: str | None = None,
        timestamp: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AddResult:
        """Add a memory to specified collections asynchronously.

        Args:
            text: The text content to remember.
            collections: Target collections. If None, uses default_collection.
            speaker: Who said this (optional).
            timestamp: ISO timestamp (optional, defaults to now).
            tags: Optional tags for indexing.
            metadata: Optional additional metadata.

        Returns:
            AddResult with statistics.
        """
        target_collections = self._get_target_collections(collections)

        total_messages = 0
        total_entities = 0

        for coll_name in target_collections:
            memory = self._memories[coll_name]
            result = await memory.add_async(
                text=text,
                speaker=speaker,
                timestamp=timestamp,
                tags=tags,
                metadata=metadata,
            )
            total_messages += result.messages_added
            total_entities += result.entities_extracted

        return AddResult(
            messages_added=total_messages,
            entities_extracted=total_entities,
            collections=target_collections,
        )

    async def add_batch_async(
        self,
        items: list[dict[str, Any]],
        collections: list[str] | None = None,
    ) -> AddResult:
        """Add multiple memories to specified collections asynchronously.

        Args:
            items: List of dicts with keys: text, speaker, timestamp, tags, metadata.
            collections: Target collections. If None, uses default_collection.

        Returns:
            AddResult with statistics.
        """
        target_collections = self._get_target_collections(collections)

        total_messages = 0
        total_entities = 0

        for coll_name in target_collections:
            memory = self._memories[coll_name]
            result = await memory.add_batch_async(items)
            total_messages += result.messages_added
            total_entities += result.entities_extracted

        return AddResult(
            messages_added=total_messages,
            entities_extracted=total_entities,
            collections=target_collections,
        )

    async def query_async(
        self,
        question: str,
        collections: list[str] | None = None,
    ) -> str:
        """Query memories across collections asynchronously.

        Args:
            question: Natural language question.
            collections: Collections to search. If None, searches all.

        Returns:
            Combined answer string based on stored memories.
        """
        target_collections = collections or self.collection_names

        # Validate collections
        for c in target_collections:
            if c not in self._memories:
                raise ValueError(f"Collection '{c}' is not in this pool.")

        # Query each collection
        answers = []
        for coll_name in target_collections:
            memory = self._memories[coll_name]
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
        self,
        query: str,
        collections: list[str] | None = None,
        limit: int = 10,
    ) -> list[MemoryItem]:
        """Search memories across collections asynchronously.

        Args:
            query: Search query (keyword, entity name, or topic).
            collections: Collections to search. If None, searches all.
            limit: Maximum total results to return.

        Returns:
            List of MemoryItem objects from all searched collections.
        """
        target_collections = collections or self.collection_names

        # Validate collections
        for c in target_collections:
            if c not in self._memories:
                raise ValueError(f"Collection '{c}' is not in this pool.")

        results: list[MemoryItem] = []
        per_collection_limit = max(1, limit // len(target_collections))

        for coll_name in target_collections:
            if len(results) >= limit:
                break
            memory = self._memories[coll_name]
            coll_results = await memory.search_async(query, limit=per_collection_limit)
            results.extend(coll_results)

        return results[:limit]

    async def stats_async(
        self,
        collections: list[str] | None = None,
    ) -> dict[str, Any]:
        """Get statistics for collections asynchronously.

        Args:
            collections: Collections to get stats for. If None, gets all.

        Returns:
            Dict with stats per collection and totals.
        """
        target_collections = collections or self.collection_names

        stats_per_collection = {}
        total_memories = 0
        total_entities = 0

        for coll_name in target_collections:
            if coll_name not in self._memories:
                continue
            memory = self._memories[coll_name]
            coll_stats = await memory.stats_async()
            stats_per_collection[coll_name] = coll_stats
            total_memories += coll_stats.get("total_memories", 0)
            total_entities += coll_stats.get("entities_extracted", 0)

        return {
            "collections": stats_per_collection,
            "total_memories": total_memories,
            "total_entities": total_entities,
            "collection_count": len(target_collections),
        }

    # =========================================================================
    # Sync API (default)
    # =========================================================================

    def add(
        self,
        text: str,
        collections: list[str] | None = None,
        speaker: str | None = None,
        timestamp: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AddResult:
        """Add a memory to specified collections synchronously."""
        return run_sync(
            self.add_async(
                text=text,
                collections=collections,
                speaker=speaker,
                timestamp=timestamp,
                tags=tags,
                metadata=metadata,
            )
        )

    def add_batch(
        self,
        items: list[dict[str, Any]],
        collections: list[str] | None = None,
    ) -> AddResult:
        """Add multiple memories to specified collections synchronously."""
        return run_sync(self.add_batch_async(items, collections))

    def query(
        self,
        question: str,
        collections: list[str] | None = None,
    ) -> str:
        """Query memories across collections synchronously."""
        return run_sync(self.query_async(question, collections))

    def search(
        self,
        query: str,
        collections: list[str] | None = None,
        limit: int = 10,
    ) -> list[MemoryItem]:
        """Search memories across collections synchronously."""
        return run_sync(self.search_async(query, collections, limit))

    def stats(self, collections: list[str] | None = None) -> dict[str, Any]:
        """Get statistics for collections synchronously."""
        return run_sync(self.stats_async(collections))

    # =========================================================================
    # Collection Management
    # =========================================================================

    def get_memory(self, collection: str) -> Memory:
        """Get the Memory instance for a specific collection.

        Args:
            collection: Collection name.

        Returns:
            Memory instance for that collection.
        """
        if collection not in self._memories:
            raise ValueError(f"Collection '{collection}' is not in this pool.")
        return self._memories[collection]

    @property
    def collections(self) -> list[str]:
        """Get list of collection names in this pool."""
        return self.collection_names.copy()
