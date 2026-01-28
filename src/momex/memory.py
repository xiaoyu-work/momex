"""Momex Memory - Simplified high-level API for Structured RAG memory."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

from .config import MomexConfig
from .exceptions import EmbeddingError, ExportError, LLMError
from .storage.base import StorageBackend


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

    Converts "user:xiaoyuzhang" to Path("user/xiaoyuzhang") for cross-platform compatibility.
    """
    # Split by : and create path parts
    parts = collection.split(":")
    # Sanitize each part for invalid characters (Windows forbidden chars)
    sanitized = [re.sub(r'[<>"|?*:\\]', '_', part) for part in parts]
    return Path(*sanitized)


def _create_backend(collection: str, config: MomexConfig) -> StorageBackend:
    """Create a storage backend based on configuration."""
    storage_config = config.storage

    if storage_config.backend == "sqlite":
        from .storage.sqlite import SQLiteBackend

        # Generate database path from collection name
        storage_path = Path(storage_config.path)
        collection_path = _collection_to_path(collection)
        db_path = storage_path / collection_path / config.db_name

        return SQLiteBackend(
            db_path=db_path,
            embedding_dim=config.embedding_dim,
        )

    elif storage_config.backend == "postgres":
        from .storage.postgres import PostgresBackend

        return PostgresBackend(
            connection_string=storage_config.connection_string,
            table_prefix=f"{storage_config.table_prefix}_{collection.replace(':', '_')}",
            embedding_dim=config.embedding_dim,
        )

    else:
        raise ValueError(f"Unknown storage backend: {storage_config.backend}")


class Memory:
    """High-level API for Structured RAG memory with single collection.

    Example:
        >>> from momex import Memory
        >>> memory = Memory(collection="user:xiaoyuzhang")
        >>> await memory.add("Alice likes cats")
        >>> answer = await memory.query("What does Alice like?")
    """

    def __init__(
        self,
        collection: str,
        config: MomexConfig | None = None,
    ) -> None:
        """Initialize Memory instance for a single collection.

        Args:
            collection: Collection name (e.g., "user:xiaoyuzhang", "team:engineering")
            config: Configuration object. If None, uses default config.
        """
        self.collection = collection
        self.config = config or MomexConfig.get_default()

        # Create storage backend
        self._backend = _create_backend(collection, self.config)
        self._initialized = False

        # TypeAgent conversation (for query functionality)
        self._conversation = None

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
        """Ensure the storage backend is initialized."""
        if self._initialized:
            return

        await self._backend.initialize()
        self._initialized = True

    async def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for text."""
        from typeagent.aitools.embeddings import AsyncEmbeddingModel

        try:
            embedding_model = AsyncEmbeddingModel(model_name=self.config.embedding_model)
            return await embedding_model.get_embedding(text)
        except Exception as e:
            raise EmbeddingError(
                message=f"Failed to generate embedding: {e}",
                model=self.config.embedding_model,
            ) from e

    async def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for multiple texts."""
        from typeagent.aitools.embeddings import AsyncEmbeddingModel

        try:
            embedding_model = AsyncEmbeddingModel(model_name=self.config.embedding_model)
            return await embedding_model.get_embeddings(texts)
        except Exception as e:
            raise EmbeddingError(
                message=f"Failed to generate embeddings: {e}",
                model=self.config.embedding_model,
            ) from e

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
        importance: float = 0.5,
    ) -> AddResult:
        """Add a memory directly without LLM processing (internal method).

        Args:
            text: The text content to store.
            speaker: Who said this (optional, defaults to collection name).
            timestamp: ISO timestamp (optional, defaults to now).
            importance: Importance score 0.0-1.0 (default 0.5).

        Returns:
            AddResult with statistics.
        """
        await self._ensure_initialized()

        if timestamp is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Use collection as speaker if not provided
        if speaker is None:
            speaker = self.collection

        # Get embedding for the text
        embedding = await self._get_embedding(text)

        # Store in backend
        await self._backend.add(
            text=text,
            embedding=embedding,
            speaker=speaker,
            timestamp=timestamp,
            importance=importance,
        )

        return AddResult(
            messages_added=1,
            entities_extracted=0,
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

        # Search for relevant memories
        results = await self.search(question, limit=10)

        if not results:
            return "No relevant memories found."

        # Build context from search results
        context = "\n".join([
            f"- {r.text}" for r in results
        ])

        # Use LLM to answer the question
        from typeagent.knowpro import convknowledge
        import typechat

        model = convknowledge.create_typechat_model()
        prompt = f"""Based on the following memories, answer the question.

Memories:
{context}

Question: {question}

Answer concisely based only on the memories provided. If the memories don't contain relevant information, say so."""

        try:
            result = await model.complete(prompt)
            if isinstance(result, typechat.Success):
                return result.value
            else:
                return f"Failed to generate answer: {result.message}"
        except Exception as e:
            raise LLMError(
                message=f"Failed to query memories: {e}",
                operation="query",
            ) from e

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

        # Get query embedding
        query_embedding = await self._get_embedding(query)

        # Use threshold from config if not specified
        min_threshold = threshold if threshold is not None else self.config.similarity_threshold

        # Search in backend
        search_results = await self._backend.search(
            embedding=query_embedding,
            limit=limit * 2,  # Get more results for importance weighting
            threshold=min_threshold,
        )

        if not search_results:
            return []

        # Apply two-stage ranking:
        # Stage 1: Already filtered by similarity threshold in backend
        # Stage 2: Rank by weighted score
        importance_weight = self.config.importance_weight

        scored_results: list[tuple[MemoryItem, float]] = []
        for sr in search_results:
            record = sr.record
            similarity = sr.similarity
            importance = record.importance

            # Weighted final score
            final_score = similarity * (1 - importance_weight) + importance * importance_weight

            item = MemoryItem(
                id=str(record.id),
                text=record.text,
                speaker=record.speaker,
                timestamp=record.timestamp,
                collection=self.collection,
                score=final_score,
                similarity=similarity,
                importance=importance,
            )
            scored_results.append((item, final_score))

        # Sort by final score descending
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # Return top results
        return [item for item, _ in scored_results[:limit]]

    async def clear(self) -> bool:
        """Clear all memories for this collection asynchronously.

        Returns:
            True if successful.
        """
        await self._ensure_initialized()
        await self._backend.clear()
        return True

    async def stats(self) -> dict[str, Any]:
        """Get memory statistics asynchronously.

        Returns:
            Dict with total count, entity count, etc.
        """
        await self._ensure_initialized()

        active_count = await self._backend.count()
        deleted_count = await self._backend.count_deleted()

        return {
            "collection": self.collection,
            "total_memories": active_count,
            "deleted_memories": deleted_count,
            "backend": self.config.storage.backend,
        }

    async def export(self, path: str) -> None:
        """Export all memories to a JSON file asynchronously.

        Args:
            path: Path to the output JSON file.
        """
        await self._ensure_initialized()

        records = await self._backend.get_all_records(include_deleted=False)

        data = {
            "collection": self.collection,
            "memories": [
                {
                    "id": str(r.id),
                    "text": r.text,
                    "speaker": r.speaker,
                    "timestamp": r.timestamp,
                    "importance": r.importance,
                }
                for r in records
            ],
        }

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
        import typechat
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

        # Get existing memories from backend
        existing_records = await self._backend.get_all_records(include_deleted=False)

        if existing_records:
            # Build existing memories index with embeddings
            existing_texts = [r.text for r in existing_records]
            existing_embeddings = await self._get_embeddings(existing_texts)

            existing_memories = [
                {"id": str(r.id), "text": r.text}
                for r in existing_records
            ]

            # Batch get embeddings for all facts at once
            fact_embeddings = await self._get_embeddings(result.facts_extracted)

            # Collect all similar memories across all facts (deduplicated)
            similar_memory_ids: set[int] = set()
            for fact_emb in fact_embeddings:
                # Calculate cosine similarity with all existing memories
                for i, mem_emb in enumerate(existing_embeddings):
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
                    old_importance = 0.5
                    if action_id and action_id.isdigit():
                        old_record = await self._backend.get(int(action_id))
                        if old_record:
                            old_importance = old_record.importance
                        await self._backend.delete(int(action_id))
                    # Use higher of old importance or new fact importance
                    new_importance = max(old_importance, fact_importance.get(text, 0.5))
                    await self._add_raw(text, importance=new_importance)
                    result.memories_updated += 1
                elif event == MemoryEvent.DELETE:
                    # Mark the memory as deleted
                    if action_id and action_id.isdigit():
                        await self._backend.delete(int(action_id))
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
        return await self._backend.delete(memory_id)

    async def restore(self, memory_id: int) -> bool:
        """Restore a deleted memory.

        Args:
            memory_id: The ID of the memory to restore.

        Returns:
            True if restored, False if not deleted.
        """
        await self._ensure_initialized()
        return await self._backend.restore(memory_id)

    async def list_deleted(self) -> list[MemoryItem]:
        """List all deleted memories.

        Returns:
            List of deleted MemoryItem objects.
        """
        await self._ensure_initialized()

        records = await self._backend.list_deleted()

        return [
            MemoryItem(
                id=str(r.id),
                text=r.text,
                speaker=r.speaker,
                timestamp=r.timestamp,
                collection=self.collection,
                importance=r.importance,
            )
            for r in records
        ]

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def db_path(self) -> str:
        """Get the database file path (for SQLite backend)."""
        if self.config.storage.backend == "sqlite":
            from .storage.sqlite import SQLiteBackend
            if isinstance(self._backend, SQLiteBackend):
                return str(self._backend.db_path)
        return ""

    @property
    def is_initialized(self) -> bool:
        """Check if the memory is initialized."""
        return self._initialized
