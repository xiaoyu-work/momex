"""Memex Memory - Simplified high-level API for Structured RAG memory."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

from .config import MemexConfig
from .exceptions import EmbeddingError, ExportError, LLMError, MemoryNotFoundError


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


class MemoryEvent(StrEnum):
    """Memory operation event types."""

    ADD = "ADD"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    NONE = "NONE"


@dataclass
class MemoryOperation:
    """A single memory operation result."""

    id: str
    text: str
    event: MemoryEvent
    old_memory: str | None = None


@dataclass
class ConversationResult:
    """Result of processing a conversation."""

    facts_extracted: list[str] = field(default_factory=list)
    operations: list[MemoryOperation] = field(default_factory=list)
    memories_added: int = 0
    memories_updated: int = 0
    memories_deleted: int = 0
    success: bool = True
    error: str | None = None


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
        self.config = config or MemexConfig.get_default()

        # Generate database path from collection name using pathlib
        storage_path = Path(self.config.storage_path)
        collection_path = _collection_to_path(collection)
        self._db_path = storage_path / collection_path / self.config.db_name

        # Ensure directory exists
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conversation = None
        self._initialized = False

        # Soft delete support
        self._deleted_file = self._db_path.parent / "deleted.json"
        self._deleted_ids: set[int] = set()
        self._load_deleted()

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

    def _load_deleted(self) -> None:
        """Load deleted IDs from file."""
        if self._deleted_file.exists():
            try:
                with open(self._deleted_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._deleted_ids = set(data.get("deleted_ids", []))
            except (json.JSONDecodeError, IOError):
                self._deleted_ids = set()

    def _save_deleted(self) -> None:
        """Save deleted IDs to file."""
        data = {
            "deleted_ids": list(self._deleted_ids),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(self._deleted_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _mark_deleted(self, msg_id: int) -> None:
        """Mark a message as deleted."""
        self._deleted_ids.add(msg_id)
        self._save_deleted()

    def _is_deleted(self, msg_id: int) -> bool:
        """Check if a message is deleted."""
        return msg_id in self._deleted_ids

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

    async def add(
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
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Use collection as speaker if not provided
        if speaker is None:
            speaker = self.collection

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

    async def add_batch(
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
                timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

            # Use collection as speaker if not provided
            speaker = item.get("speaker") or self.collection

            msg = ConversationMessage(
                text_chunks=[item["text"]],
                tags=item.get("tags", []),
                timestamp=timestamp,
                metadata=ConversationMessageMeta(
                    speaker=speaker,
                ),
            )
            messages.append(msg)

        result = await self._conversation.add_messages_with_indexing(messages)

        return AddResult(
            messages_added=result.messages_added,
            entities_extracted=result.semrefs_added,
            collections=[self.collection],
        )

    async def query(self, question: str) -> str:
        """Query memories with natural language asynchronously.

        Args:
            question: Natural language question.

        Returns:
            Answer string based on stored memories.
        """
        await self._ensure_initialized()
        return await self._conversation.query(question)

    async def search(
        self,
        query: str,
        limit: int = 10,
        threshold: float | None = None,
    ) -> list[MemoryItem]:
        """Search memories using vector similarity.

        This method uses embedding-based semantic search to find memories
        that are similar to the query. Results are sorted by similarity score.

        Args:
            query: Search query (natural language question or topic).
            limit: Maximum number of results to return.
            threshold: Minimum similarity score (0.0-1.0). If None, uses config default.

        Returns:
            List of MemoryItem objects sorted by relevance score.
        """
        await self._ensure_initialized()

        from typeagent.aitools.embeddings import AsyncEmbeddingModel
        import numpy as np

        message_count = await self._conversation.messages.size()
        if message_count == 0:
            return []

        # Collect all non-deleted memories and their texts
        memories: list[tuple[int, str, Any]] = []  # (id, text, msg)
        for i in range(message_count):
            if self._is_deleted(i):
                continue
            msg = await self._conversation.messages.get_item(i)
            text = " ".join(msg.text_chunks) if msg.text_chunks else ""
            if text:
                memories.append((i, text, msg))

        if not memories:
            return []

        # Get embeddings
        try:
            embedding_model = AsyncEmbeddingModel()
            query_embedding = await embedding_model.get_embedding(query)
            memory_texts = [m[1] for m in memories]
            memory_embeddings = await embedding_model.get_embeddings(memory_texts)
        except Exception as e:
            raise EmbeddingError(
                message=f"Failed to generate embeddings: {e}",
            ) from e

        # Calculate cosine similarity scores
        # Embeddings are already normalized, so dot product = cosine similarity
        scores = np.dot(memory_embeddings, query_embedding)

        # Use threshold from config if not specified
        min_threshold = threshold if threshold is not None else self.config.similarity_threshold

        # Combine with memory data and filter by threshold
        scored_memories = [
            (memories[i], float(scores[i]))
            for i in range(len(memories))
            if scores[i] >= min_threshold
        ]

        # Sort by score descending
        scored_memories.sort(key=lambda x: x[1], reverse=True)

        # Return top results
        results: list[MemoryItem] = []
        for (msg_id, text, msg), score in scored_memories[:limit]:
            results.append(
                MemoryItem(
                    id=str(msg_id),
                    text=text,
                    speaker=msg.metadata.speaker if msg.metadata else None,
                    timestamp=msg.timestamp,
                    collection=self.collection,
                    score=score,
                )
            )

        return results

    async def clear(self) -> bool:
        """Clear all memories for this collection asynchronously.

        Returns:
            True if successful.
        """
        if self._db_path.exists():
            self._db_path.unlink()

        # Also clear deleted records
        if self._deleted_file.exists():
            self._deleted_file.unlink()
        self._deleted_ids = set()

        self._conversation = None
        self._initialized = False

        return True

    async def stats(self) -> dict[str, Any]:
        """Get memory statistics asynchronously.

        Returns:
            Dict with total count, entity count, etc.
        """
        await self._ensure_initialized()

        total_count = await self._conversation.messages.size()
        deleted_count = len(self._deleted_ids)
        active_count = total_count - deleted_count
        semref_count = await self._conversation.semantic_refs.size()

        return {
            "collection": self.collection,
            "total_memories": active_count,
            "deleted_memories": deleted_count,
            "entities_extracted": semref_count,
            "db_path": str(self._db_path),
        }

    async def export(self, path: str) -> None:
        """Export all memories to a JSON file asynchronously.

        Args:
            path: Path to the output JSON file.
        """
        await self._ensure_initialized()

        data = {
            "collection": self.collection,
            "memories": [],
        }

        message_count = await self._conversation.messages.size()
        for i in range(message_count):
            # Skip deleted messages
            if self._is_deleted(i):
                continue

            msg = await self._conversation.messages.get_item(i)
            data["memories"].append(
                {
                    "id": str(i),
                    "text": " ".join(msg.text_chunks) if msg.text_chunks else "",
                    "speaker": msg.metadata.speaker if msg.metadata else None,
                    "timestamp": msg.timestamp,
                    "tags": msg.tags,
                }
            )

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except (IOError, OSError) as e:
            raise ExportError(
                message=f"Failed to export memories: {e}",
                export_path=path,
            ) from e

    async def add_conversation(
        self,
        messages: list[dict[str, str]],
        similarity_limit: int = 5,
    ) -> ConversationResult:
        """Extract facts from a conversation and add to memory asynchronously.

        This method uses a multi-stage LLM process similar to mem0:
        1. Extract facts from the conversation
        2. For each fact, vector search to find similar existing memories
        3. LLM decides ADD/UPDATE/DELETE/NONE based on similar memories
        4. Execute operations

        Args:
            messages: List of conversation messages, each with "role" and "content" keys.
                     Example: [{"role": "user", "content": "I like Python"},
                              {"role": "assistant", "content": "Great choice!"}]
            similarity_limit: Max number of similar memories to retrieve per fact.

        Returns:
            ConversationResult with extracted facts and operations performed.

        Example:
            >>> result = await memory.add_conversation([
            ...     {"role": "user", "content": "My name is Alice and I love Python"},
            ...     {"role": "assistant", "content": "Nice to meet you, Alice!"},
            ... ])
            >>> print(result.facts_extracted)
            ['Name is Alice', 'Loves Python']
        """
        await self._ensure_initialized()

        from .prompts import get_fact_extraction_prompt, get_memory_update_prompt
        from typeagent.knowpro import convknowledge
        from typeagent.aitools.embeddings import AsyncEmbeddingModel
        import numpy as np

        result = ConversationResult()

        # Format conversation for the prompt
        conversation_text = "\n".join(
            f"{msg.get('role', 'user').capitalize()}: {msg.get('content', '')}"
            for msg in messages
        )

        # Stage 1: Extract facts from conversation (using configured fact_types)
        model = convknowledge.create_typechat_model()
        fact_prompt = get_fact_extraction_prompt(
            conversation_text,
            fact_types=self.config.fact_types,
        )

        try:
            fact_response = await model.complete(fact_prompt)
            # Parse JSON response
            fact_json = self._extract_json(fact_response)
            result.facts_extracted = fact_json.get("facts", [])
        except Exception as e:
            result.success = False
            result.error = f"Failed to extract facts: {e}"
            return result

        if not result.facts_extracted:
            # No facts to process
            return result

        # Build existing memories index with embeddings (excluding deleted)
        existing_memories: list[dict] = []
        memory_embeddings: list = []

        message_count = await self._conversation.messages.size()
        if message_count > 0:
            embedding_model = AsyncEmbeddingModel()
            texts_to_embed = []

            for i in range(message_count):
                # Skip deleted messages
                if self._is_deleted(i):
                    continue

                msg = await self._conversation.messages.get_item(i)
                text = " ".join(msg.text_chunks) if msg.text_chunks else ""
                if text:
                    existing_memories.append({"id": str(i), "text": text})
                    texts_to_embed.append(text)

            if texts_to_embed:
                memory_embeddings = await embedding_model.get_embeddings(texts_to_embed)

        # Stage 2 & 3: For each fact, find similar memories and decide operation
        if existing_memories and len(memory_embeddings) > 0:
            embedding_model = AsyncEmbeddingModel()

            for fact in result.facts_extracted:
                # Get embedding for this fact
                fact_embedding = await embedding_model.get_embedding(fact)

                # Calculate cosine similarity with all existing memories
                similarities = []
                for i, mem_emb in enumerate(memory_embeddings):
                    similarity = float(np.dot(fact_embedding, mem_emb))
                    similarities.append((i, similarity))

                # Sort by similarity and get top-k
                similarities.sort(key=lambda x: x[1], reverse=True)
                top_similar = similarities[:similarity_limit]

                # Get the similar memories for LLM decision
                similar_memories = [
                    existing_memories[idx]
                    for idx, score in top_similar
                    if score > self.config.similarity_threshold
                ]

                # LLM decides what to do with this fact
                update_prompt = get_memory_update_prompt(similar_memories, [fact])

                try:
                    update_response = await model.complete(update_prompt)
                    update_json = self._extract_json(update_response)
                    memory_actions = update_json.get("memory", [])
                except Exception as e:
                    # If decision fails for this fact, skip it
                    continue

                # Execute operations for this fact
                for action in memory_actions:
                    event_str = action.get("event", "NONE").upper()
                    try:
                        event = MemoryEvent(event_str)
                    except ValueError:
                        event = MemoryEvent.NONE

                    text = action.get("text", "")
                    action_id = action.get("id", "")

                    op = MemoryOperation(
                        id=action_id,
                        text=text,
                        event=event,
                        old_memory=action.get("old_memory"),
                    )
                    result.operations.append(op)

                    if event == MemoryEvent.ADD and text:
                        await self.add(text)
                        result.memories_added += 1
                    elif event == MemoryEvent.UPDATE and text:
                        # Mark old memory as deleted, add new one
                        if action_id and action_id.isdigit():
                            self._mark_deleted(int(action_id))
                        await self.add(text)
                        result.memories_updated += 1
                    elif event == MemoryEvent.DELETE:
                        # Mark the memory as deleted
                        if action_id and action_id.isdigit():
                            self._mark_deleted(int(action_id))
                        result.memories_deleted += 1
        else:
            # No existing memories, add all facts directly
            for fact in result.facts_extracted:
                op = MemoryOperation(id="new", text=fact, event=MemoryEvent.ADD)
                result.operations.append(op)
                await self.add(fact)
                result.memories_added += 1

        return result

    def _extract_json(self, text: str) -> dict:
        """Extract JSON from LLM response text."""
        # Try to parse directly
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON in markdown code blocks
        import re
        patterns = [
            r"```json\s*(.*?)\s*```",
            r"```\s*(.*?)\s*```",
            r"\{.*\}",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1) if "```" in pattern else match.group(0))
                except json.JSONDecodeError:
                    continue

        return {"facts": [], "memory": []}

    async def delete(self, memory_id: int) -> bool:
        """Delete a memory by ID (soft delete).

        Args:
            memory_id: The ID of the memory to delete.

        Returns:
            True if deleted, False if already deleted or not found.
        """
        await self._ensure_initialized()

        # Check if memory exists
        message_count = await self._conversation.messages.size()
        if memory_id < 0 or memory_id >= message_count:
            return False

        # Check if already deleted
        if self._is_deleted(memory_id):
            return False

        self._mark_deleted(memory_id)
        return True

    async def restore(self, memory_id: int) -> bool:
        """Restore a deleted memory.

        Args:
            memory_id: The ID of the memory to restore.

        Returns:
            True if restored, False if not deleted.
        """
        if memory_id not in self._deleted_ids:
            return False

        self._deleted_ids.remove(memory_id)
        self._save_deleted()
        return True

    async def list_deleted(self) -> list[MemoryItem]:
        """List all deleted memories.

        Returns:
            List of deleted MemoryItem objects.
        """
        await self._ensure_initialized()

        results: list[MemoryItem] = []
        message_count = await self._conversation.messages.size()

        for i in self._deleted_ids:
            if i < message_count:
                msg = await self._conversation.messages.get_item(i)
                text = " ".join(msg.text_chunks) if msg.text_chunks else ""
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
