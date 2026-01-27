"""Memex Memory - Simplified high-level API for Structured RAG memory."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
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


@dataclass
class AddResult:
    """Result of adding memories."""

    messages_added: int
    entities_extracted: int
    success: bool = True


class Memory:
    """High-level API for Structured RAG memory.

    Example:
        >>> from memex import Memory
        >>> memory = Memory(user_id="user_123")
        >>> memory.add("张三说下周五完成API")
        >>> answer = memory.query("谁负责API?")
        >>> results = memory.search("张三")
    """

    def __init__(
        self,
        user_id: str | None = None,
        agent_id: str | None = None,
        org_id: str | None = None,
        config: MemexConfig | None = None,
        db_path: str | None = None,
    ) -> None:
        """Initialize Memory instance.

        Args:
            user_id: User identifier for multi-tenant isolation.
            agent_id: Agent identifier for multi-tenant isolation.
            org_id: Organization identifier for multi-tenant isolation.
            config: Configuration object. If None, uses default config.
            db_path: Direct path to database file. Overrides config-based path.
        """
        self.user_id = user_id
        self.agent_id = agent_id
        self.org_id = org_id
        self.config = config or MemexConfig()

        # Determine database path
        if db_path:
            self._db_path = db_path
            # Ensure directory exists
            os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        else:
            self._db_path = self.config.get_db_path(
                user_id=user_id,
                agent_id=agent_id,
                org_id=org_id,
            )

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
            # Fallback to python-dotenv
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
            self._db_path,
            ConversationMessage,
            name=f"memex:{self.user_id or 'default'}",
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

        # Build message
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

        # Add and index
        result = await self._conversation.add_messages_with_indexing([msg])

        return AddResult(
            messages_added=result.messages_added,
            entities_extracted=result.semrefs_added,
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

        # Use the conversation's search capabilities
        # For now, we'll use query and parse results
        # TODO: Implement proper search when knowpro exposes search API

        results: list[MemoryItem] = []

        # Access messages directly for basic search
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
                        )
                    )

        return results

    async def delete_async(self, memory_id: str) -> bool:
        """Delete a memory by ID asynchronously.

        Args:
            memory_id: The ID of the memory to delete.

        Returns:
            True if deleted, False if not found.
        """
        await self._ensure_initialized()
        # TODO: Implement when knowpro supports deletion
        raise NotImplementedError("Delete not yet supported by underlying storage")

    async def clear_async(self) -> bool:
        """Clear all memories for this tenant asynchronously.

        Returns:
            True if successful.
        """
        # Delete the database file and reinitialize
        import os

        if os.path.exists(self._db_path):
            os.remove(self._db_path)

        self._conversation = None
        self._initialized = False

        return True

    async def stats_async(self) -> dict[str, Any]:
        """Get memory statistics asynchronously.

        Returns:
            Dict with total count, entity count, topic count, etc.
        """
        await self._ensure_initialized()

        message_count = len(self._conversation.messages) if self._conversation.messages else 0
        semref_count = len(self._conversation.semantic_refs) if self._conversation.semantic_refs else 0

        return {
            "total_memories": message_count,
            "entities_extracted": semref_count,
            "db_path": self._db_path,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "org_id": self.org_id,
        }

    async def export_async(self, path: str) -> None:
        """Export all memories to a JSON file asynchronously.

        Args:
            path: Path to the output JSON file.
        """
        import json

        await self._ensure_initialized()

        data = {
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "org_id": self.org_id,
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
        """Add a memory synchronously.

        Args:
            text: The text content to remember.
            speaker: Who said this (optional).
            timestamp: ISO timestamp (optional, defaults to now).
            tags: Optional tags for indexing.
            metadata: Optional additional metadata.

        Returns:
            AddResult with statistics about what was added.
        """
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
        """Add multiple memories synchronously.

        Args:
            items: List of dicts with keys: text, speaker, timestamp, tags, metadata.

        Returns:
            AddResult with statistics.
        """
        return run_sync(self.add_batch_async(items))

    def query(self, question: str) -> str:
        """Query memories with natural language synchronously.

        Args:
            question: Natural language question.

        Returns:
            Answer string based on stored memories.
        """
        return run_sync(self.query_async(question))

    def search(self, query: str, limit: int = 10) -> list[MemoryItem]:
        """Search memories by keyword/entity synchronously.

        Args:
            query: Search query (keyword, entity name, or topic).
            limit: Maximum number of results to return.

        Returns:
            List of MemoryItem objects.
        """
        return run_sync(self.search_async(query, limit))

    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID synchronously.

        Args:
            memory_id: The ID of the memory to delete.

        Returns:
            True if deleted, False if not found.
        """
        return run_sync(self.delete_async(memory_id))

    def clear(self) -> bool:
        """Clear all memories for this tenant synchronously.

        Returns:
            True if successful.
        """
        return run_sync(self.clear_async())

    def stats(self) -> dict[str, Any]:
        """Get memory statistics synchronously.

        Returns:
            Dict with total count, entity count, topic count, etc.
        """
        return run_sync(self.stats_async())

    def export(self, path: str) -> None:
        """Export all memories to a JSON file synchronously.

        Args:
            path: Path to the output JSON file.
        """
        return run_sync(self.export_async(path))

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def db_path(self) -> str:
        """Get the database file path."""
        return self._db_path

    @property
    def is_initialized(self) -> bool:
        """Check if the memory is initialized."""
        return self._initialized
