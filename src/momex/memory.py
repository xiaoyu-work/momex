"""Momex Memory - High-level API wrapping TypeAgent's Structured RAG.

This module provides a simplified Memory API that uses TypeAgent's full
indexing system (SemanticRefs, TermIndex) rather than text+embedding search.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

from .config import MomexConfig
from .exceptions import LLMError




@dataclass
class AddResult:
    """Result of adding memories."""

    messages_added: int
    entities_extracted: int
    contradictions_removed: int = 0
    success: bool = True
    collections: list[str] | None = None


@dataclass
class SearchItem:
    """A single search result item."""

    type: str  # Uses TypeAgent's native knowledge_type: "entity", "action", "topic", "message"
    text: str
    score: float
    raw: Any  # Original TypeAgent object (SemanticRef or Message)


def _collection_to_db_path(collection: str, base_path: str, db_name: str) -> Path:
    """Convert collection name to database path.

    Converts "momex:engineering:xiaoyuzhang" to
    Path("base_path/momex/engineering/xiaoyuzhang/db_name")
    """
    parts = collection.split(":")
    # Sanitize each part for invalid characters (Windows forbidden chars)
    sanitized = [re.sub(r'[<>"|?*:\\]', '_', part) for part in parts]
    return Path(base_path) / Path(*sanitized) / db_name


class Memory:
    """High-level API for Structured RAG memory using TypeAgent's full indexing.

    This class wraps TypeAgent's ConversationBase to provide:
    - Hierarchical collections (e.g., "momex:engineering:xiaoyuzhang")
    - Simple add/search/query API
    - Full structured knowledge extraction (entities, actions, topics)
    - Term-based indexing (not just vector similarity)

    Example:
        >>> from momex import Memory
        >>> memory = Memory(collection="momex:engineering:xiaoyuzhang")
        >>> await memory.add("I like Python programming")
        >>> results = await memory.search("What languages?")
    """

    def __init__(
        self,
        collection: str,
        config: MomexConfig | None = None,
    ) -> None:
        """Initialize Memory instance for a single collection.

        Args:
            collection: Collection name (e.g., "momex:engineering:xiaoyuzhang")
            config: Configuration object. If None, uses default config.
        """
        self.collection = collection
        self.config = config or MomexConfig.get_default()

        # TypeAgent conversation (lazy initialized)
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
        """Ensure the TypeAgent conversation is initialized."""
        if self._initialized:
            return

        from typeagent.knowpro.conversation_base import ConversationBase
        from typeagent.knowpro.convsettings import ConversationSettings
        from typeagent.knowpro.universal_message import ConversationMessage

        # Validate config before use
        self.config.validate()

        if self.config.is_postgres:
            storage_provider = await self._create_postgres_provider()
        else:
            storage_provider = self._create_sqlite_provider()

        # Create conversation settings with the storage provider
        settings = ConversationSettings(storage_provider=storage_provider)

        # Create conversation using factory method
        self._conversation = await ConversationBase.create(
            settings=settings,
            name=self.collection,
            tags=[self.collection],
        )

        self._initialized = True

    def _create_sqlite_provider(self):
        """Create SQLite storage provider."""
        from typeagent.knowpro.universal_message import ConversationMessage
        from typeagent.storage.sqlite import SqliteStorageProvider

        # Create storage path from collection name
        db_path = _collection_to_db_path(
            self.collection,
            self.config.storage.path,
            self.config.db_name,
        )
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create SQLite storage provider with the collection-specific database
        storage_provider = SqliteStorageProvider(
            db_path=str(db_path),
            message_type=ConversationMessage,
        )

        # Commit any pending schema initialization transaction
        storage_provider.db.commit()

        return storage_provider

    async def _create_postgres_provider(self):
        """Create PostgreSQL storage provider."""
        from typeagent.knowpro.universal_message import ConversationMessage
        from typeagent.storage.postgres import PostgresStorageProvider

        # Use collection name as part of table prefix or schema
        # For now, we'll use a single database with collection stored in metadata
        storage_provider = await PostgresStorageProvider.create(
            connection_string=self.config.postgres.url,
            message_type=ConversationMessage,
            min_pool_size=self.config.postgres.pool_min,
            max_pool_size=self.config.postgres.pool_max,
        )

        return storage_provider

    async def add(
        self,
        messages: str | list[dict[str, str]],
        *,
        infer: bool = True,
        detect_contradictions: bool = True,
    ) -> AddResult:
        """Add memories with TypeAgent's knowledge extraction.

        Automatically detects and removes contradicting memories before adding.
        For example, if memory contains "I like sushi" and you add "I don't like sushi",
        the old contradicting memory will be removed automatically.

        Args:
            messages: Content to add. Can be:
                - str: A single message (treated as user message)
                - list[dict]: Conversation messages with "role" and "content" keys
            infer: If True (default), use LLM to extract knowledge.
                   If False, add directly without LLM processing.
            detect_contradictions: If True (default), use LLM to detect and remove
                   contradicting memories before adding. Set False to skip this.

        Returns:
            AddResult with statistics about what was added.

        Examples:
            # String input - extracts entities, actions, topics
            await memory.add("I like Python and FastAPI")

            # Automatically handles contradictions
            await memory.add("I don't like Python anymore")  # Removes old "like Python"

            # Conversation input
            await memory.add([
                {"role": "user", "content": "My name is Xiaoyu"},
                {"role": "assistant", "content": "Nice to meet you!"},
            ])
        """
        await self._ensure_initialized()

        # Detect and remove contradictions before adding
        contradictions_removed = 0
        if infer and detect_contradictions:
            content_text = messages if isinstance(messages, str) else " ".join(
                m.get("content", "") for m in messages
            )
            contradictions_removed = await self._detect_and_remove_contradictions(content_text)

        from typeagent.knowpro.universal_message import (
            ConversationMessage,
            ConversationMessageMeta,
        )

        # Normalize input to conversation format
        if isinstance(messages, str):
            conversation = [{"role": "user", "content": messages}]
        else:
            conversation = messages

        # Convert to TypeAgent ConversationMessage format
        ta_messages: list[ConversationMessage] = []
        for msg in conversation:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if not content:
                continue

            # Use collection as speaker context, but keep role info
            speaker = f"{self.collection}:{role}"

            ta_message = ConversationMessage(
                text_chunks=[content],
                metadata=ConversationMessageMeta(speaker=speaker),
                timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
            ta_messages.append(ta_message)

        if not ta_messages:
            return AddResult(
                messages_added=0,
                entities_extracted=0,
                contradictions_removed=contradictions_removed,
                collections=[self.collection],
            )

        # Use TypeAgent's add_messages_with_indexing for full knowledge extraction
        if infer:
            result = await self._conversation.add_messages_with_indexing(ta_messages)
            return AddResult(
                messages_added=result.messages_added,
                entities_extracted=result.semrefs_added,
                contradictions_removed=contradictions_removed,
                success=True,
                collections=[self.collection],
            )
        else:
            # Direct add without LLM processing
            # Temporarily disable auto_extract_knowledge
            old_setting = self._conversation.settings.semantic_ref_index_settings.auto_extract_knowledge
            self._conversation.settings.semantic_ref_index_settings.auto_extract_knowledge = False
            try:
                result = await self._conversation.add_messages_with_indexing(ta_messages)
            finally:
                self._conversation.settings.semantic_ref_index_settings.auto_extract_knowledge = old_setting

            return AddResult(
                messages_added=result.messages_added,
                entities_extracted=0,
                contradictions_removed=0,
                success=True,
                collections=[self.collection],
            )

    async def query(self, question: str) -> str:
        """Query memories with natural language.

        Uses TypeAgent's full search pipeline:
        1. Translate question to structured SearchQuery
        2. Search term-based indexes
        3. Generate answer from matched knowledge

        Args:
            question: Natural language question.

        Returns:
            Answer string based on stored memories.
        """
        await self._ensure_initialized()

        try:
            return await self._conversation.query(question)
        except Exception as e:
            raise LLMError(
                message=f"Failed to query memories: {e}",
                operation="query",
            ) from e

    async def search(
        self,
        query_text: str,
        limit: int = 10,
    ) -> list[SearchItem]:
        """Search memories using TypeAgent's term-based indexing.

        Args:
            query_text: Search query (natural language question or topic).
            limit: Maximum number of results to return.

        Returns:
            List of SearchItem with type, text, score, and raw TypeAgent object.
        """
        await self._ensure_initialized()

        from typeagent.knowpro import searchlang, convknowledge, search_query_schema, kplib
        from typeagent.knowpro.interfaces import Topic
        from typeagent.aitools import utils
        import typechat

        # Initialize query translator if needed
        if self._conversation._query_translator is None:
            model = convknowledge.create_typechat_model()
            self._conversation._query_translator = utils.create_translator(
                model, search_query_schema.SearchQuery
            )

        # Use TypeAgent's language search
        options = searchlang.LanguageSearchOptions(
            compile_options=searchlang.LanguageQueryCompileOptions(
                exact_scope=False,
                verb_scope=True,
                term_filter=None,
                apply_scope=True,
            ),
            exact_match=False,
            max_message_matches=limit,
        )

        result = await searchlang.search_conversation_with_language(
            self._conversation,
            self._conversation._query_translator,
            query_text,
            options,
        )

        if isinstance(result, typechat.Failure):
            return []

        # Wrap TypeAgent results into SearchItem
        items: list[SearchItem] = []

        for search_result in result.value:
            # Process knowledge matches
            for knowledge_type, matches in search_result.knowledge_matches.items():
                for scored in matches.semantic_ref_matches[:limit]:
                    sem_ref = await self._conversation.semantic_refs.get_item(
                        scored.semantic_ref_ordinal
                    )
                    if sem_ref is None:
                        continue

                    knowledge = sem_ref.knowledge
                    k_type = knowledge.knowledge_type  # Use native type

                    # Format text
                    if isinstance(knowledge, kplib.ConcreteEntity):
                        text = knowledge.name
                        if knowledge.type:
                            text += f" (type: {', '.join(knowledge.type)})"
                        if knowledge.facets:
                            facets = [f"{f.name}: {f.value}" for f in knowledge.facets if f.value]
                            if facets:
                                text += f" [{'; '.join(facets)}]"
                    elif isinstance(knowledge, kplib.Action):
                        parts = []
                        if knowledge.subject_entity_name:
                            parts.append(knowledge.subject_entity_name)
                        parts.extend(knowledge.verbs)
                        if knowledge.object_entity_name:
                            parts.append(knowledge.object_entity_name)
                        text = " ".join(parts)
                    elif isinstance(knowledge, Topic):
                        text = knowledge.text
                    else:
                        text = str(knowledge)

                    items.append(SearchItem(
                        type=k_type,
                        text=text,
                        score=scored.score,
                        raw=sem_ref,
                    ))

            # Process message matches
            for msg_match in search_result.message_matches[:limit]:
                msg = await self._conversation.messages.get_item(msg_match.message_ordinal)
                if msg is None:
                    continue

                text = " ".join(msg.text_chunks) if hasattr(msg, 'text_chunks') else str(msg)

                items.append(SearchItem(
                    type="message",
                    text=text,
                    score=msg_match.score,
                    raw=msg,
                ))

        # Sort by score and limit
        items.sort(key=lambda x: x.score, reverse=True)
        return items[:limit]

    async def delete(self, query: str, *, limit: int = 50) -> int:
        """Delete memories matching a query.

        For advanced users who want explicit control over deletion.
        Normal users can rely on add() which automatically handles contradictions.

        Args:
            query: Search query to find memories to delete.
            limit: Maximum number of items to delete (default 50).

        Returns:
            Number of items deleted.

        Example:
            # Delete memories about sushi preference
            deleted = await memory.delete("likes sushi")
            print(f"Deleted {deleted} memories")
        """
        await self._ensure_initialized()

        # Search for matching memories
        results = await self.search(query, limit=limit)

        if not results:
            return 0

        # Collect IDs to delete
        semref_ids = []
        for item in results:
            if item.type != "message" and hasattr(item.raw, 'semantic_ref_ordinal'):
                semref_ids.append(item.raw.semantic_ref_ordinal)

        if not semref_ids:
            return 0

        # Delete from indexes
        deleted_count = await self._delete_by_ids(semref_ids)

        return deleted_count

    async def _delete_by_ids(self, semref_ids: list[int]) -> int:
        """Internal: Delete semantic refs by IDs."""
        storage = self._conversation.storage_provider
        deleted_count = 0

        for semref_id in semref_ids:
            try:
                # Remove from property index
                prop_index = await storage.get_property_index()
                await prop_index.remove_all_for_semref(semref_id)
                deleted_count += 1
            except (IndexError, KeyError):
                continue

        # Commit for SQLite
        if self.config.is_sqlite:
            storage.db.commit()

        return deleted_count

    async def _detect_and_remove_contradictions(self, new_content: str) -> int:
        """Internal: Use LLM to detect and remove contradicting memories.

        Args:
            new_content: The new content being added.

        Returns:
            Number of contradicting memories removed.
        """
        from typeagent.aitools import chat

        # Search for potentially related memories
        results = await self.search(new_content, limit=20)

        if not results:
            return 0

        # Build context of existing memories
        existing_memories = []
        for i, item in enumerate(results):
            if item.type != "message":
                existing_memories.append(f"{i}: [{item.type}] {item.text}")

        if not existing_memories:
            return 0

        # Ask LLM to identify contradictions
        prompt = f"""Given the new information and existing memories, identify which existing memories contradict the new information.

New information: "{new_content}"

Existing memories:
{chr(10).join(existing_memories)}

Return ONLY the indices (numbers) of memories that directly contradict the new information, separated by commas.
If no contradictions, return "none".
Only identify clear contradictions (e.g., "likes X" vs "doesn't like X"), not merely related information.

Response:"""

        try:
            model = chat.ChatModel()
            response = await model.get_completion(prompt)
            response_text = response.strip().lower()

            if response_text == "none" or not response_text:
                return 0

            # Parse indices
            indices = []
            for part in response_text.replace(" ", "").split(","):
                try:
                    idx = int(part)
                    if 0 <= idx < len(results):
                        indices.append(idx)
                except ValueError:
                    continue

            if not indices:
                return 0

            # Delete contradicting memories
            semref_ids = []
            for idx in indices:
                item = results[idx]
                if item.type != "message" and hasattr(item.raw, 'semantic_ref_ordinal'):
                    semref_ids.append(item.raw.semantic_ref_ordinal)

            if semref_ids:
                return await self._delete_by_ids(semref_ids)

        except Exception:
            # If contradiction detection fails, just proceed with add
            pass

        return 0

    async def clear(self) -> bool:
        """Clear all memories for this collection.

        Returns:
            True if successful.
        """
        await self._ensure_initialized()
        await self._conversation.storage_provider.clear()

        # Commit for SQLite (PostgreSQL handles this automatically)
        if self.config.is_sqlite:
            self._conversation.storage_provider.db.commit()

        return True

    async def stats(self) -> dict[str, Any]:
        """Get memory statistics.

        Returns:
            Dict with counts of messages, semantic refs, etc.
        """
        await self._ensure_initialized()

        message_count = await self._conversation.messages.size()
        semref_count = await self._conversation.semantic_refs.size()

        backend_name = "postgres" if self.config.is_postgres else "sqlite"

        return {
            "collection": self.collection,
            "total_messages": message_count,
            "total_semantic_refs": semref_count,
            "backend": backend_name,
        }

    async def export(self, path: str) -> None:
        """Export all memories to a JSON file.

        Args:
            path: Path to the output JSON file.
        """
        await self._ensure_initialized()

        from .exceptions import ExportError

        # Get all messages
        messages = await self._conversation.messages.get_slice(0, 999999)
        # Get all semantic refs
        semrefs = await self._conversation.semantic_refs.get_slice(0, 999999)

        from typeagent.knowpro import kplib

        data = {
            "collection": self.collection,
            "messages": [
                {
                    "text": " ".join(m.text_chunks) if hasattr(m, 'text_chunks') else str(m),
                    "speaker": m.metadata.speaker if hasattr(m, 'metadata') and hasattr(m.metadata, 'speaker') else None,
                    "timestamp": m.timestamp if hasattr(m, 'timestamp') else None,
                }
                for m in messages
            ],
            "knowledge": [],
        }

        for sr in semrefs:
            k = sr.knowledge
            if isinstance(k, kplib.ConcreteEntity):
                data["knowledge"].append({
                    "type": "entity",
                    "name": k.name,
                    "types": k.type,
                    "facets": [{"name": f.name, "value": f.value} for f in (k.facets or [])],
                })
            elif isinstance(k, kplib.Action):
                data["knowledge"].append({
                    "type": "action",
                    "verbs": k.verbs,
                    "subject": k.subject_entity_name,
                    "object": k.object_entity_name,
                })
            else:
                data["knowledge"].append({
                    "type": "other",
                    "text": str(k),
                })

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except (IOError, OSError) as e:
            raise ExportError(
                message=f"Failed to export memories: {e}",
                export_path=path,
            ) from e

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def db_path(self) -> str:
        """Get the database path or connection URL.

        For SQLite: returns the file path.
        For PostgreSQL: returns the connection URL.
        """
        if self.config.is_postgres:
            return self.config.postgres.url
        return str(_collection_to_db_path(
            self.collection,
            self.config.storage.path,
            self.config.db_name,
        ))

    @property
    def is_initialized(self) -> bool:
        """Check if the memory is initialized."""
        return self._initialized

