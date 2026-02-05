"""Momex Memory - High-level API wrapping TypeAgent's Structured RAG.

This module provides a simplified Memory API that uses TypeAgent's full
indexing system (SemanticRefs, TermIndex) rather than text+embedding search.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import MomexConfig
from .exceptions import LLMError

if TYPE_CHECKING:
    from typeagent.knowpro.conversation_base import ConversationBase
    from typeagent.knowpro.convsettings import (
        MessageTextIndexSettings,
        RelatedTermIndexSettings,
    )




DELETED_SEMREFS_METADATA_KEY = "momex_deleted_semrefs"


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


def _collection_to_schema(collection: str) -> str:
    """Convert collection name to a PostgreSQL-safe schema name."""
    base = re.sub(r"[^a-zA-Z0-9_]", "_", collection).lower()
    if not base:
        base = "momex"
    if base[0].isdigit():
        base = f"c_{base}"

    max_len = 63
    if len(base) <= max_len:
        return base

    digest = hashlib.md5(collection.encode("utf-8")).hexdigest()[:8]
    return f"{base[:54]}_{digest}"


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
        self._deleted_semref_ids: set[int] | None = None

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
        from typeagent.knowpro.convknowledge import set_llm_config

        # Validate config before use
        self.config.validate()

        # Set LLM config for TypeAgent (used by KnowledgeExtractor)
        set_llm_config(self.config.get_llm_config())

        embedding_model = self.config.create_embedding_model()
        settings = ConversationSettings(model=embedding_model)

        if self.config.is_postgres:
            storage_provider = await self._create_postgres_provider(
                settings.message_text_index_settings,
                settings.related_term_index_settings,
            )
        else:
            storage_provider = self._create_sqlite_provider(
                settings.message_text_index_settings,
                settings.related_term_index_settings,
            )

        # Attach storage provider to settings
        settings.storage_provider = storage_provider

        # Create conversation using factory method
        self._conversation = await ConversationBase.create(
            settings=settings,
            name=self.collection,
            tags=[self.collection],
        )

        self._initialized = True

    def _conversation_required(self) -> "ConversationBase":
        assert self._conversation is not None, "Conversation not initialized"
        return self._conversation

    async def _get_conversation_metadata(self):
        storage = self._conversation_required().storage_provider
        if hasattr(storage, "get_conversation_metadata_async"):
            return await storage.get_conversation_metadata_async()
        return storage.get_conversation_metadata()

    async def _set_conversation_metadata(self, **kwds: str | list[str] | None) -> None:
        storage = self._conversation_required().storage_provider
        if self.config.is_postgres and hasattr(storage, "pool"):
            from typeagent.storage.postgres.schema import set_conversation_metadata

            await set_conversation_metadata(storage.pool, **kwds)
        else:
            storage.set_conversation_metadata(**kwds)

    async def _load_deleted_semref_ids(self) -> set[int]:
        if self._deleted_semref_ids is not None:
            return self._deleted_semref_ids

        metadata = await self._get_conversation_metadata()
        deleted_raw = ""
        if metadata.extra and DELETED_SEMREFS_METADATA_KEY in metadata.extra:
            deleted_raw = metadata.extra[DELETED_SEMREFS_METADATA_KEY]

        if not deleted_raw:
            self._deleted_semref_ids = set()
            return self._deleted_semref_ids

        try:
            parsed = json.loads(deleted_raw)
        except json.JSONDecodeError:
            parsed = []

        deleted_ids: set[int] = set()
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, int):
                    deleted_ids.add(item)
                elif isinstance(item, str) and item.isdigit():
                    deleted_ids.add(int(item))

        self._deleted_semref_ids = deleted_ids
        return deleted_ids

    async def _store_deleted_semref_ids(self, deleted_ids: set[int]) -> None:
        serialized = json.dumps(sorted(deleted_ids))
        await self._set_conversation_metadata(**{DELETED_SEMREFS_METADATA_KEY: serialized})

    def _filter_search_results(
        self,
        results,
        deleted_ids: set[int],
    ):
        if not deleted_ids:
            return results
        filtered = []
        for search_result in results:
            knowledge_matches = {}
            for ktype, kmatches in search_result.knowledge_matches.items():
                kept = [
                    match
                    for match in kmatches.semantic_ref_matches
                    if match.semantic_ref_ordinal not in deleted_ids
                ]
                if kept:
                    kmatches.semantic_ref_matches = kept
                    knowledge_matches[ktype] = kmatches
            search_result.knowledge_matches = knowledge_matches
            if search_result.knowledge_matches or search_result.message_matches:
                filtered.append(search_result)
        return filtered

    def _create_sqlite_provider(
        self,
        message_text_index_settings: MessageTextIndexSettings,
        related_term_index_settings: RelatedTermIndexSettings,
    ):
        """Create SQLite storage provider."""
        from typeagent.knowpro.universal_message import ConversationMessage
        from typeagent.storage.sqlite import SqliteStorageProvider

        # Create storage path from collection name
        db_path = _collection_to_db_path(
            self.collection,
            self.config.storage_path,
            "memory.db",
        )
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create SQLite storage provider with the collection-specific database
        storage_provider = SqliteStorageProvider(
            db_path=str(db_path),
            message_type=ConversationMessage,
            message_text_index_settings=message_text_index_settings,
            related_term_index_settings=related_term_index_settings,
        )

        # Commit any pending schema initialization transaction
        storage_provider.db.commit()

        return storage_provider

    async def _create_postgres_provider(
        self,
        message_text_index_settings: MessageTextIndexSettings,
        related_term_index_settings: RelatedTermIndexSettings,
    ):
        """Create PostgreSQL storage provider."""
        from typeagent.knowpro.universal_message import ConversationMessage
        from typeagent.knowpro.interfaces import ConversationMetadata
        from typeagent.storage.postgres import PostgresStorageProvider

        # Use collection name as part of table prefix or schema
        # For now, we'll use a single database with collection stored in metadata
        schema = (
            self.config.storage.postgres_schema
            if self.config.storage.postgres_schema
            else _collection_to_schema(self.collection)
        )

        storage_provider = await PostgresStorageProvider.create(
            connection_string=self.config.storage.postgres_url,
            message_type=ConversationMessage,
            message_text_index_settings=message_text_index_settings,
            related_term_index_settings=related_term_index_settings,
            min_pool_size=self.config.storage.postgres_pool_min,
            max_pool_size=self.config.storage.postgres_pool_max,
            schema=schema,
            pgbouncer=self.config.storage.postgres_pgbouncer,
            metadata=ConversationMetadata(
                name_tag=self.collection,
                tags=[self.collection],
            ),
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
        conversation_obj = self._conversation_required()

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
            conversation_messages = [{"role": "user", "content": messages}]
        else:
            conversation_messages = messages

        # Convert to TypeAgent ConversationMessage format
        ta_messages: list[ConversationMessage] = []
        for msg in conversation_messages:
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
            result = await conversation_obj.add_messages_with_indexing(ta_messages)
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
            old_setting = (
                conversation_obj.settings.semantic_ref_index_settings.auto_extract_knowledge
            )
            conversation_obj.settings.semantic_ref_index_settings.auto_extract_knowledge = False
            try:
                result = await conversation_obj.add_messages_with_indexing(ta_messages)
            finally:
                conversation_obj.settings.semantic_ref_index_settings.auto_extract_knowledge = old_setting

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
        conversation = self._conversation_required()

        from typeagent.knowpro import answers, answer_response_schema, convknowledge, searchlang, search_query_schema
        from typeagent.aitools import utils
        import typechat

        try:
            if conversation._query_translator is None:
                model = convknowledge.create_typechat_model()
                conversation._query_translator = utils.create_translator(
                    model, search_query_schema.SearchQuery
                )
            if conversation._answer_translator is None:
                model = convknowledge.create_typechat_model()
                conversation._answer_translator = utils.create_translator(
                    model, answer_response_schema.AnswerResponse
                )

            search_options = searchlang.LanguageSearchOptions(
                compile_options=searchlang.LanguageQueryCompileOptions(
                    exact_scope=False,
                    verb_scope=True,
                    term_filter=None,
                    apply_scope=True,
                ),
                exact_match=False,
                max_message_matches=25,
            )

            result = await searchlang.search_conversation_with_language(
                conversation,
                conversation._query_translator,
                question,
                search_options,
            )

            if isinstance(result, typechat.Failure):
                return f"Search failed: {result.message}"

            search_results = result.value
            deleted_ids = await self._load_deleted_semref_ids()
            if deleted_ids:
                search_results = self._filter_search_results(search_results, deleted_ids)

            answer_options = answers.AnswerContextOptions(
                entities_top_k=50, topics_top_k=50, messages_top_k=20, chunking=None
            )

            _, combined_answer = await answers.generate_answers(
                conversation._answer_translator,
                search_results,
                conversation,
                question,
                options=answer_options,
            )

            match combined_answer.type:
                case "NoAnswer":
                    return f"No answer found: {combined_answer.why_no_answer or 'Unable to find relevant information'}"
                case "Answered":
                    return combined_answer.answer or "No answer provided"
                case _:
                    return f"Unexpected answer type: {combined_answer.type}"
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
        conversation = self._conversation_required()

        from typeagent.knowpro import searchlang, convknowledge, search_query_schema, kplib
        from typeagent.knowpro.interfaces import Topic
        from typeagent.aitools import utils
        import typechat

        # Initialize query translator if needed
        if conversation._query_translator is None:
            model = convknowledge.create_typechat_model()
            conversation._query_translator = utils.create_translator(
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
            conversation,
            conversation._query_translator,
            query_text,
            options,
        )

        if isinstance(result, typechat.Failure):
            return []

        # Wrap TypeAgent results into SearchItem
        items: list[SearchItem] = []

        search_results = result.value
        deleted_ids = await self._load_deleted_semref_ids()
        if deleted_ids:
            search_results = self._filter_search_results(search_results, deleted_ids)

        for search_result in search_results:
            # Process knowledge matches
            for _, matches in search_result.knowledge_matches.items():
                for scored in matches.semantic_ref_matches[:limit]:
                    sem_ref = await conversation.semantic_refs.get_item(
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
                msg = await conversation.messages.get_item(msg_match.message_ordinal)
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
        deleted_ids = await self._load_deleted_semref_ids()
        deleted_ids.update(semref_ids)
        await self._store_deleted_semref_ids(deleted_ids)

        deleted_count = await self._delete_by_ids(semref_ids)

        return deleted_count

    async def _delete_by_ids(self, semref_ids: list[int]) -> int:
        """Internal: Delete semantic refs by IDs."""
        storage = self._conversation_required().storage_provider
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
        if self.config.is_sqlite and hasattr(storage, "db"):
            storage.db.commit()

        return deleted_count

    async def _detect_and_remove_contradictions(self, new_content: str) -> int:
        """Internal: Use LLM to detect and remove contradicting memories.

        Args:
            new_content: The new content being added.

        Returns:
            Number of contradicting memories removed.
        """
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
            # Use TypeAgent's LLM abstraction
            llm = self.config.create_llm()
            response = await llm.complete(prompt, max_tokens=100)
            response_text = response.content.strip().lower()

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
                deleted_ids = await self._load_deleted_semref_ids()
                deleted_ids.update(semref_ids)
                await self._store_deleted_semref_ids(deleted_ids)
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
        conversation = self._conversation_required()
        await conversation.storage_provider.clear()

        # Commit for SQLite (PostgreSQL handles this automatically)
        if self.config.is_sqlite and hasattr(conversation.storage_provider, "db"):
            conversation.storage_provider.db.commit()

        self._deleted_semref_ids = set()
        await self._store_deleted_semref_ids(self._deleted_semref_ids)

        return True

    async def stats(self) -> dict[str, Any]:
        """Get memory statistics.

        Returns:
            Dict with counts of messages, semantic refs, etc.
        """
        await self._ensure_initialized()
        conversation = self._conversation_required()

        message_count = await conversation.messages.size()
        semref_count = await conversation.semantic_refs.size()

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
        conversation = self._conversation_required()

        from .exceptions import ExportError

        # Get all messages
        messages = await conversation.messages.get_slice(0, 999999)
        # Get all semantic refs
        semrefs = await conversation.semantic_refs.get_slice(0, 999999)

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
            return self.config.storage.postgres_url
        return str(_collection_to_db_path(
            self.collection,
            self.config.storage_path,
            "memory.db",
        ))

    @property
    def is_initialized(self) -> bool:
        """Check if the memory is initialized."""
        return self._initialized
