"""Memex Memory - Simplified high-level API for Structured RAG memory."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import MemexConfig
from .sync import run_sync


@dataclass
class MemoryItem:
    """A single memory item returned from search."""

    id: str
    text: str
    speaker: str | None = None
    timestamp: str | None = None
    metadata: dict[str, Any] | None = None
    score: float | None = None
    collection: str | None = None


@dataclass
class AddResult:
    """Result of adding memories."""

    messages_added: int
    entities_extracted: int
    success: bool = True
    collections: list[str] | None = None


def _collection_to_path(collection: str) -> Path:
    """Convert collection name to path.

    Converts "user:alice" to Path("user/alice") for cross-platform compatibility.
    """
    # Split by : and create path parts
    parts = collection.split(":")
    # Sanitize each part for invalid characters (Windows forbidden chars)
    sanitized = [re.sub(r'[<>"|?*:\\]', '_', part) for part in parts]
    return Path(*sanitized)


class Memory:
    """High-level API for Structured RAG memory with single collection.

    Example:
        >>> from memex import Memory
        >>> memory = Memory(collection="user:alice")
        >>> memory.add("Alice likes cats")
        >>> answer = memory.query("What does Alice like?")
    """

    def __init__(
        self,
        collection: str,
        config: MemexConfig | None = None,
    ) -> None:
        """Initialize Memory instance for a single collection.

        Args:
            collection: Collection name (e.g., "user:alice", "team:engineering")
            config: Configuration object. If None, uses default config.
        """
        self.collection = collection
        self.config = config or MemexConfig()

        # Generate database path from collection name using pathlib
        storage_path = Path(self.config.storage_path)
        collection_path = _collection_to_path(collection)
        self._db_path = storage_path / collection_path / self.config.db_name

        # Ensure directory exists
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conversation = None
        self._initialized = False

        # Auto-load dotenv
        self._load_dotenv()

    def _load_dotenv(self) -> None:
        """Load environment variables from .env file."""
        try:
            from typeagent.aitools.utils import load_dotenv
            load_dotenv()
        except ImportError:
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                pass

    async def _ensure_initialized(self) -> None:
        """Ensure the conversation is initialized."""
        if self._initialized:
            return

        from typeagent import create_conversation
        from typeagent.knowpro.universal_message import ConversationMessage

        self._conversation = await create_conversation(
            str(self._db_path),
            ConversationMessage,
            name=f"memex:{self.collection}",
        )
        self._initialized = True

    # =========================================================================
    # Async API
    # =========================================================================

    async def add_async(
        self,
        text: str,
        speaker: str | None = None,
        timestamp: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AddResult:
        """Add a memory asynchronously.

        Args:
            text: The text content to remember.
            speaker: Who said this (optional).
            timestamp: ISO timestamp (optional, defaults to now).
            tags: Optional tags for indexing.
            metadata: Optional additional metadata.

        Returns:
            AddResult with statistics about what was added.
        """
        await self._ensure_initialized()

        from typeagent.knowpro.universal_message import (
            ConversationMessage,
            ConversationMessageMeta,
        )

        if timestamp is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%Sz")

        msg = ConversationMessage(
            text_chunks=[text],
            tags=tags or [],
            timestamp=timestamp,
            metadata=ConversationMessageMeta(
                speaker=speaker,
            ),
        )

        result = await self._conversation.add_messages_with_indexing([msg])

        return AddResult(
            messages_added=result.messages_added,
            entities_extracted=result.semrefs_added,
            collections=[self.collection],
        )

    async def add_batch_async(
        self,
        items: list[dict[str, Any]],
    ) -> AddResult:
        """Add multiple memories asynchronously.

        Args:
            items: List of dicts with keys: text, speaker, timestamp, tags, metadata.

        Returns:
            AddResult with statistics.
        """
        await self._ensure_initialized()

        from typeagent.knowpro.universal_message import (
            ConversationMessage,
            ConversationMessageMeta,
        )

        messages = []
        for item in items:
            timestamp = item.get("timestamp")
            if timestamp is None:
                timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%Sz")

            msg = ConversationMessage(
                text_chunks=[item["text"]],
                tags=item.get("tags", []),
                timestamp=timestamp,
                metadata=ConversationMessageMeta(
                    speaker=item.get("speaker"),
                ),
            )
            messages.append(msg)

        result = await self._conversation.add_messages_with_indexing(messages)

        return AddResult(
            messages_added=result.messages_added,
            entities_extracted=result.semrefs_added,
            collections=[self.collection],
        )

    async def query_async(self, question: str) -> str:
        """Query memories with natural language asynchronously.

        Args:
            question: Natural language question.

        Returns:
            Answer string based on stored memories.
        """
        await self._ensure_initialized()
        return await self._conversation.query(question)

    async def search_async(
        self,
        query: str,
        limit: int = 10,
    ) -> list[MemoryItem]:
        """Search memories by keyword/entity asynchronously.

        Args:
            query: Search query (keyword, entity name, or topic).
            limit: Maximum number of results to return.

        Returns:
            List of MemoryItem objects.
        """
        await self._ensure_initialized()

        results: list[MemoryItem] = []

        if self._conversation.messages:
            for i, msg in enumerate(self._conversation.messages):
                if len(results) >= limit:
                    break

                text = " ".join(msg.text_chunks) if msg.text_chunks else ""
                if query.lower() in text.lower():
                    results.append(
                        MemoryItem(
                            id=str(i),
                            text=text,
                            speaker=msg.metadata.speaker if msg.metadata else None,
                            timestamp=msg.timestamp,
                            collection=self.collection,
                        )
                    )

        return results

    async def clear_async(self) -> bool:
        """Clear all memories for this collection asynchronously.

        Returns:
            True if successful.
        """
        if self._db_path.exists():
            self._db_path.unlink()

        self._conversation = None
        self._initialized = False

        return True

    async def stats_async(self) -> dict[str, Any]:
        """Get memory statistics asynchronously.

        Returns:
            Dict with total count, entity count, etc.
        """
        await self._ensure_initialized()

        message_count = len(self._conversation.messages) if self._conversation.messages else 0
        semref_count = len(self._conversation.semantic_refs) if self._conversation.semantic_refs else 0

        return {
            "collection": self.collection,
            "total_memories": message_count,
            "entities_extracted": semref_count,
            "db_path": str(self._db_path),
        }

    async def export_async(self, path: str) -> None:
        """Export all memories to a JSON file asynchronously.

        Args:
            path: Path to the output JSON file.
        """
        import json

        await self._ensure_initialized()

        data = {
            "collection": self.collection,
            "memories": [],
        }

        if self._conversation.messages:
            for i, msg in enumerate(self._conversation.messages):
                data["memories"].append(
                    {
                        "id": str(i),
                        "text": " ".join(msg.text_chunks) if msg.text_chunks else "",
                        "speaker": msg.metadata.speaker if msg.metadata else None,
                        "timestamp": msg.timestamp,
                        "tags": msg.tags,
                    }
                )

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # =========================================================================
    # Sync API (default)
    # =========================================================================

    def add(
        self,
        text: str,
        speaker: str | None = None,
        timestamp: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AddResult:
        """Add a memory synchronously."""
        return run_sync(
            self.add_async(
                text=text,
                speaker=speaker,
                timestamp=timestamp,
                tags=tags,
                metadata=metadata,
            )
        )

    def add_batch(self, items: list[dict[str, Any]]) -> AddResult:
        """Add multiple memories synchronously."""
        return run_sync(self.add_batch_async(items))

    def query(self, question: str) -> str:
        """Query memories with natural language synchronously."""
        return run_sync(self.query_async(question))

    def search(self, query: str, limit: int = 10) -> list[MemoryItem]:
        """Search memories by keyword/entity synchronously."""
        return run_sync(self.search_async(query, limit))

    def clear(self) -> bool:
        """Clear all memories for this collection synchronously."""
        return run_sync(self.clear_async())

    def stats(self) -> dict[str, Any]:
        """Get memory statistics synchronously."""
        return run_sync(self.stats_async())

    def export(self, path: str) -> None:
        """Export all memories to a JSON file synchronously."""
        return run_sync(self.export_async(path))

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def db_path(self) -> str:
        """Get the database file path."""
        return str(self._db_path)

    @property
    def is_initialized(self) -> bool:
        """Check if the memory is initialized."""
        return self._initialized
