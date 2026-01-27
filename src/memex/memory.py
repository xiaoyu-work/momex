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
    score: float | None = None  # Final weighted score
    similarity: float | None = None  # Raw similarity score
    importance: float = 0.5  # 0.0-1.0, higher = more important
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

        # Importance tracking
        self._importance_file = self._db_path.parent / "importance.json"
        self._importance_map: dict[int, float] = {}
        self._load_importance()

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

    def _load_importance(self) -> None:
        """Load importance scores from file."""
        if self._importance_file.exists():
            try:
                with open(self._importance_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Convert string keys back to int
                    self._importance_map = {
                        int(k): v for k, v in data.get("importance", {}).items()
                    }
            except (json.JSONDecodeError, IOError):
                self._importance_map = {}

    def _save_importance(self) -> None:
        """Save importance scores to file."""
        data = {
            "importance": {str(k): v for k, v in self._importance_map.items()},
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(self._importance_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _set_importance(self, msg_id: int, importance: float) -> None:
        """Set importance score for a message."""
        self._importance_map[msg_id] = max(0.0, min(1.0, importance))
        self._save_importance()

    def _get_importance(self, msg_id: int) -> float:
        """Get importance score for a message (default 0.5)."""
        return self._importance_map.get(msg_id, 0.5)

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
        messages: str | list[dict[str, str]],
        *,
        infer: bool = True,
        similarity_limit: int = 5,
    ) -> AddResult:
        """Add memories with optional LLM processing and deduplication.

        This is the main API for adding memories. By default (infer=True), it uses
        LLM to extract facts and intelligently decide whether to ADD, UPDATE, or
        DELETE existing memories based on semantic similarity.

        Args:
            messages: Content to add. Can be:
                - str: A single message (treated as user message)
                - list[dict]: Conversation messages with "role" and "content" keys
            infer: If True (default), use LLM to extract facts and deduplicate.
                   If False, add the content directly without LLM processing.
            similarity_limit: Max similar memories to consider per fact (when infer=True).

        Returns:
            AddResult with statistics about what was added/updated/deleted.

        Examples:
            # String input - LLM extracts facts and deduplicates
            await memory.add("I like Python and FastAPI")

            # Conversation input - LLM processes the full conversation
            await memory.add([
                {"role": "user", "content": "My name is Alice"},
                {"role": "assistant", "content": "Nice to meet you!"},
            ])

            # Direct storage without LLM processing
            await memory.add("Raw log entry: user logged in", infer=False)
        """
        # Normalize input to conversation format
        if isinstance(messages, str):
            conversation = [{"role": "user", "content": messages}]
        else:
            conversation = messages

        if infer:
            # Use LLM to extract facts and deduplicate
            result = await self._add_with_inference(conversation, similarity_limit)
            return AddResult(
                messages_added=result.memories_added,
                entities_extracted=len(result.facts_extracted),
                success=result.success,
                collections=[self.collection],
            )
        else:
            # Direct storage without LLM processing
            total_added = 0
            total_entities = 0
            for msg in conversation:
                content = msg.get("content", "")
                if content:
                    raw_result = await self._add_raw(content)
                    total_added += raw_result.messages_added
                    total_entities += raw_result.entities_extracted
            return AddResult(
                messages_added=total_added,
                entities_extracted=total_entities,
                collections=[self.collection],
            )

    async def _add_raw(
        self,
        text: str,
        speaker: str | None = None,
        timestamp: str | None = None,
        tags: list[str] | None = None,
        importance: float = 0.5,
    ) -> AddResult:
        """Add a memory directly without LLM processing (internal method).

        Args:
            text: The text content to store.
            speaker: Who said this (optional, defaults to collection name).
            timestamp: ISO timestamp (optional, defaults to now).
            tags: Optional tags for indexing.
            importance: Importance score 0.0-1.0 (default 0.5).

        Returns:
            AddResult with statistics.
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

        # Get current message count to determine new message ID
        msg_id = await self._conversation.messages.size()

        msg = ConversationMessage(
            text_chunks=[text],
            tags=tags or [],
            timestamp=timestamp,
            metadata=ConversationMessageMeta(
                speaker=speaker,
            ),
        )

        result = await self._conversation.add_messages_with_indexing([msg])

        # Store importance for the new message
        if result.messages_added > 0:
            self._set_importance(msg_id, importance)

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
            embedding_model = AsyncEmbeddingModel(model_name=self.config.embedding_model)
            query_embedding = await embedding_model.get_embedding(query)
            memory_texts = [m[1] for m in memories]
            memory_embeddings = await embedding_model.get_embeddings(memory_texts)
        except Exception as e:
            raise EmbeddingError(
                message=f"Failed to generate embeddings: {e}",
            ) from e

        # Calculate cosine similarity scores
        # Embeddings are already normalized, so dot product = cosine similarity
        similarity_scores = np.dot(memory_embeddings, query_embedding)

        # Use threshold from config if not specified
        min_threshold = threshold if threshold is not None else self.config.similarity_threshold

        # Two-stage ranking:
        # Stage 1: Filter by similarity threshold (relevance gate)
        # Stage 2: Among filtered results, rank by weighted score
        importance_weight = self.config.importance_weight
        scored_memories: list[tuple[tuple[int, str, Any], float, float, float]] = []
        for i in range(len(memories)):
            similarity = float(similarity_scores[i])
            # Stage 1: Filter out irrelevant memories
            if similarity < min_threshold:
                continue
            # Stage 2: Calculate weighted score for relevant memories only
            msg_id = memories[i][0]
            importance = self._get_importance(msg_id)
            # Weighted final score (only applied to already-relevant items)
            final_score = similarity * (1 - importance_weight) + importance * importance_weight
            scored_memories.append((memories[i], final_score, similarity, importance))

        # Sort by final score descending
        scored_memories.sort(key=lambda x: x[1], reverse=True)

        # Return top results
        results: list[MemoryItem] = []
        for (msg_id, text, msg), final_score, similarity, importance in scored_memories[:limit]:
            results.append(
                MemoryItem(
                    id=str(msg_id),
                    text=text,
                    speaker=msg.metadata.speaker if msg.metadata else None,
                    timestamp=msg.timestamp,
                    collection=self.collection,
                    score=final_score,
                    similarity=similarity,
                    importance=importance,
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

        # Clear importance records
        if self._importance_file.exists():
            self._importance_file.unlink()
        self._importance_map = {}

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

    async def _add_with_inference(
        self,
        messages: list[dict[str, str]],
        similarity_limit: int = 5,
    ) -> ConversationResult:
        """Internal method: Extract facts and add with LLM-based deduplication.

        Uses a multi-stage LLM process:
        1. Extract facts from the conversation
        2. Vector search to find similar existing memories
        3. LLM decides ADD/UPDATE/DELETE/NONE based on similarity
        4. Execute operations

        Args:
            messages: List of conversation messages with "role" and "content" keys.
            similarity_limit: Max similar memories to consider per fact.

        Returns:
            ConversationResult with extracted facts and operations performed.
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
            import typechat
            fact_result = await model.complete(fact_prompt)
            # Handle typechat.Result type
            if isinstance(fact_result, typechat.Success):
                fact_response = fact_result.value
            else:
                # Failure case
                result.success = False
                result.error = f"LLM call failed: {fact_result.message}"
                return result
            # Parse JSON response - now expects [{"text": "...", "importance": 0.x}, ...]
            fact_json = self._extract_json(fact_response)
            raw_facts = fact_json.get("facts", [])

            # Parse facts and build importance mapping
            fact_texts: list[str] = []
            fact_importance: dict[str, float] = {}
            for fact in raw_facts:
                if isinstance(fact, dict):
                    text = fact.get("text", "")
                    importance = fact.get("importance", 0.5)
                else:
                    # Backward compatibility: plain string
                    text = str(fact)
                    importance = 0.5
                if text:
                    fact_texts.append(text)
                    fact_importance[text] = importance

            result.facts_extracted = fact_texts
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
            embedding_model = AsyncEmbeddingModel(model_name=self.config.embedding_model)
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

        # Stage 2 & 3: Batch process all facts - find similar memories and decide operations
        if existing_memories and len(memory_embeddings) > 0:
            embedding_model = AsyncEmbeddingModel(model_name=self.config.embedding_model)

            # Batch get embeddings for all facts at once
            fact_embeddings = await embedding_model.get_embeddings(result.facts_extracted)

            # Collect all similar memories across all facts (deduplicated)
            similar_memory_ids: set[int] = set()
            for fact_emb in fact_embeddings:
                # Calculate cosine similarity with all existing memories
                for i, mem_emb in enumerate(memory_embeddings):
                    similarity = float(np.dot(fact_emb, mem_emb))
                    if similarity > self.config.similarity_threshold:
                        similar_memory_ids.add(i)

            # Get top similar memories (limit to avoid too large context)
            similar_memories = [
                existing_memories[idx]
                for idx in sorted(similar_memory_ids)[:similarity_limit * len(result.facts_extracted)]
            ]

            # Single LLM call to decide operations for ALL facts
            update_prompt = get_memory_update_prompt(similar_memories, result.facts_extracted)

            try:
                update_result = await model.complete(update_prompt)
                # Handle typechat.Result type
                if isinstance(update_result, typechat.Success):
                    update_response = update_result.value
                else:
                    result.success = False
                    result.error = f"LLM call failed: {update_result.message}"
                    return result
                update_json = self._extract_json(update_response)
                memory_actions = update_json.get("memory", [])
            except Exception as e:
                result.success = False
                result.error = f"Failed to decide memory operations: {e}"
                return result

            # Execute all operations
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
                    importance = fact_importance.get(text, 0.5)
                    await self._add_raw(text, importance=importance)
                    result.memories_added += 1
                elif event == MemoryEvent.UPDATE and text:
                    # Mark old memory as deleted, add new one
                    if action_id and action_id.isdigit():
                        old_importance = self._get_importance(int(action_id))
                        self._mark_deleted(int(action_id))
                    else:
                        old_importance = 0.5
                    # Use higher of old importance or new fact importance
                    new_importance = max(old_importance, fact_importance.get(text, 0.5))
                    await self._add_raw(text, importance=new_importance)
                    result.memories_updated += 1
                elif event == MemoryEvent.DELETE:
                    # Mark the memory as deleted
                    if action_id and action_id.isdigit():
                        self._mark_deleted(int(action_id))
                    result.memories_deleted += 1
        else:
            # No existing memories, add all facts directly
            for fact_text in result.facts_extracted:
                importance = fact_importance.get(fact_text, 0.5)
                op = MemoryOperation(id="new", text=fact_text, event=MemoryEvent.ADD)
                result.operations.append(op)
                await self._add_raw(fact_text, importance=importance)
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
