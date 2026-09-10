"""Momex Memory - High-level API wrapping TypeAgent's Structured RAG.

This module provides a simplified Memory API that uses TypeAgent's full
indexing system (SemanticRefs, TermIndex) combined with embedding similarity
search for robust hybrid retrieval.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
import json
import logging
from typing import Any, TYPE_CHECKING

from .attribution import extraction_inputs, is_confirmed, prepare_messages, WritePolicy
from .config import load_dotenv_once, MomexConfig
from .contradictions import detect as detect_contradictions
from .contradictions import find_candidates as find_contradiction_candidates
from .ledger import SupersessionLedger
from .paths import collection_to_db_path, utc_now
from .providers import create_postgres_provider, create_sqlite_provider, DB_FILENAME
from .results import AddResult, SearchItem, SupersededRecord
from .search import (
    expand_with_neighbors,
    fetch_many,
    fuse_results,
    item_for_message,
    items_for_semrefs,
    message_text,
    search_by_embedding,
    search_structured,
)
from .timewindow import (
    normalize_as_of,
    validate_iso_date,
    validate_timestamp,
    window_tags,
)
from .visibility import SearchView

if TYPE_CHECKING:
    from typeagent.knowpro.conversation_base import ConversationBase
    from typeagent.knowpro.convsettings import (
        MessageTextIndexSettings,
        RelatedTermIndexSettings,
    )


logger = logging.getLogger(__name__)

# How much wider than the presentation budget each retrieval path is asked to
# go before the two are fused. Sweeping 1x/3x/10x on LOCOMO, 1x is clearly
# worst and 10x is indistinguishable from 3x, so this buys the whole available
# gain without scanning ten times the rows.
_FUSION_OVERFETCH = 3


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
        self._ledger = SupersessionLedger(
            collection,
            self._get_conversation_metadata,
            self._set_conversation_metadata,
        )
        # Guards the temporary auto_extract_knowledge toggle in add(), which
        # mutates state shared by every concurrent call on this instance.
        self._settings_lock = asyncio.Lock()

        # Auto-load dotenv
        load_dotenv_once()

    async def _ensure_initialized(self) -> None:
        """Ensure the TypeAgent conversation is initialized."""
        if self._initialized:
            return

        from typeagent.knowpro.conversation_base import ConversationBase
        from typeagent.knowpro.convknowledge import set_llm_config
        from typeagent.knowpro.convsettings import ConversationSettings

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
        return await storage.get_conversation_metadata()

    async def _set_conversation_metadata(self, **kwds: str | list[str] | None) -> None:
        storage = self._conversation_required().storage_provider
        await storage.set_conversation_metadata(**kwds)

    def _create_sqlite_provider(
        self,
        message_text_index_settings: MessageTextIndexSettings,
        related_term_index_settings: RelatedTermIndexSettings,
    ):
        """Create SQLite storage provider."""
        return create_sqlite_provider(
            self.collection,
            self.config,
            message_text_index_settings,
            related_term_index_settings,
        )

    async def _create_postgres_provider(
        self,
        message_text_index_settings: MessageTextIndexSettings,
        related_term_index_settings: RelatedTermIndexSettings,
    ):
        """Create PostgreSQL storage provider."""
        return await create_postgres_provider(
            self.collection,
            self.config,
            message_text_index_settings,
            related_term_index_settings,
        )

    async def add(
        self,
        messages: str | Sequence[Mapping[str, object]],
        *,
        infer: bool = True,
        detect_contradictions: bool = True,
        valid_from: str | None = None,
        valid_to: str | None = None,
        timestamp: str | None = None,
        write_policy: WritePolicy = "user",
        context_turns: int = 2,
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
            write_policy: "user" extracts confirmed user facts, keeping other
                   roles as unconfirmed context. "all" opts into extracting all
                   roles. A per-message confirmed boolean overrides the policy.
            context_turns: Previous turns supplied only to resolve references.
            detect_contradictions: If True (default), use LLM to detect and remove
                   contradicting memories before adding. Set False to skip this.
            valid_from: ISO date string (e.g., "2026-04-01"). Memory is only relevant
                   from this date. None means no start constraint.
            valid_to: ISO date string (e.g., "2026-05-01"). Memory expires after this
                   date and will be excluded from search results. None means no expiry.
            timestamp: When the memory happened, as ISO-8601 UTC
                   ("2023-05-08T13:56:00Z"). Defaults to now, which is only
                   right for something being said as it is stored. Anything
                   backfilled -- an imported chat log, a migrated database, a
                   summary of last week -- needs its real time, or the whole
                   collection collapses onto the moment it was ingested and
                   every temporal query becomes meaningless.

        Returns:
            AddResult with statistics about what was added.

        Examples:
            # String input - extracts entities, actions, topics
            await memory.add("I like Python and FastAPI")

            # Automatically handles contradictions
            await memory.add("I don't like Python anymore")  # Removes old "like Python"

            # Time-bound memory
            await memory.add(
                "Netflix subscription renews May 1 at $15.99",
                valid_from="2026-04-01",
                valid_to="2026-05-02",
            )

            # Conversation input
            await memory.add([
                {"role": "user", "content": "My name is Xiaoyu"},
                {"role": "assistant", "content": "Nice to meet you!"},
            ])
        """
        await self._ensure_initialized()
        conversation_obj = self._conversation_required()

        # Reject unusable dates before anything is written. The window checks
        # compare these lexicographically, so a non-padded value would fail
        # silently rather than loudly.
        valid_from = validate_iso_date(valid_from, "valid_from")
        valid_to = validate_iso_date(valid_to, "valid_to")
        occurred_at = validate_timestamp(timestamp) if timestamp else utc_now()
        if context_turns < 0:
            raise ValueError("context_turns cannot be negative")

        # Contradiction handling runs *after* the write (see below): retiring
        # first means a failed insert leaves the old facts hidden and the
        # replacement missing.
        superseded: list[SupersededRecord] = []

        ta_messages = prepare_messages(
            messages,
            collection=self.collection,
            timestamp=occurred_at,
            tags=window_tags(valid_from, valid_to),
            write_policy=write_policy,
        )

        if not ta_messages:
            return AddResult(
                messages_added=0,
                entities_extracted=0,
                contradictions_removed=0,
                collections=[self.collection],
            )

        async with self._settings_lock:
            if infer:
                size = await conversation_obj.messages.size()
                previous = (
                    await conversation_obj.messages.get_slice(
                        max(0, size - context_turns), size
                    )
                    if context_turns
                    else []
                )
                inputs = extraction_inputs(ta_messages, previous, context_turns)
                semref_baseline = await conversation_obj.semantic_refs.size()
                result = await conversation_obj.add_messages_with_indexing(
                    ta_messages, knowledge_inputs=inputs
                )
                confirmed = [m for m in ta_messages if is_confirmed(m)]
                if detect_contradictions and confirmed:
                    superseded = await self._detect_and_remove_contradictions(
                        "\n".join(message_text(m) for m in confirmed),
                        new_ordinals=list(
                            range(
                                semref_baseline, semref_baseline + result.semrefs_added
                            )
                        ),
                    )
            else:
                index_settings = conversation_obj.settings.semantic_ref_index_settings
                old_setting = index_settings.auto_extract_knowledge
                index_settings.auto_extract_knowledge = False
                try:
                    result = await conversation_obj.add_messages_with_indexing(
                        ta_messages
                    )
                finally:
                    index_settings.auto_extract_knowledge = old_setting

        return AddResult(
            messages_added=result.messages_added,
            entities_extracted=result.semrefs_added if infer else 0,
            contradictions_removed=len(superseded),
            collections=[self.collection],
            superseded=superseded or None,
        )

    # =========================================================================
    # Search
    # =========================================================================

    async def _search_view(
        self,
        *,
        include_expired: bool = False,
        include_superseded: bool = False,
        include_unconfirmed: bool = False,
        as_of: str | None = None,
    ) -> SearchView:
        hidden = await self._ledger.hidden_ordinals(as_of)
        source_ordinals: set[int] = set()
        if hidden:
            refs = await fetch_many(
                self._conversation_required().semantic_refs, sorted(hidden)
            )
            source_ordinals = {
                ref.range.start.message_ordinal
                for ref in refs.values()
                if getattr(ref, "range", None)
            }
        return SearchView(
            superseded_knowledge=hidden,
            superseded_messages=source_ordinals,
            include_expired=include_expired,
            include_superseded=include_superseded,
            include_unconfirmed=include_unconfirmed,
            as_of=as_of,
            collection=self.collection,
        )

    async def search(
        self,
        query_text: str,
        limit: int = 10,
        *,
        include_expired: bool = False,
        include_superseded: bool = False,
        include_unconfirmed: bool = False,
        neighbors: int = 0,
        as_of: str | None = None,
    ) -> list[SearchItem]:
        """Hybrid search: structured term matching + embedding similarity in parallel.

        Runs both search paths concurrently and merges them with reciprocal rank
        fusion for better recall without sacrificing precision.

        Args:
            query_text: Search query (natural language question or topic).
            limit: Maximum number of results to return.
            include_expired: If True, include memories past their valid_to date.
            include_superseded: If True, also return memories that have been
                superseded by newer ones. Off by default, which is the normal
                "current view" of the collection.
            neighbors: Widen each message result to include this many turns on
                either side of it. Retrieval scores turns independently, but an
                answer is often in the reply to the turn that matched, and that
                reply may match nothing on its own. Costs one batch read and
                does not change how many results come back.

        Returns:
            List of SearchItem with type, text, score, and raw TypeAgent object,
            ordered by fusion_score.

        Neither path is allowed to take the other down: if one fails it is
        logged and contributes nothing, and the results of the other are
        returned on their own. Both failing yields an empty list.
        """
        as_of = normalize_as_of(as_of)
        if limit < 0 or neighbors < 0:
            raise ValueError("limit and neighbors cannot be negative")
        if limit == 0:
            return []
        await self._ensure_initialized()

        # Retrieval depth is decoupled from presentation depth. Asking each
        # path for exactly `limit` leaves fusion picking `limit` winners out of
        # at most `2 * limit` candidates, so an item ranked just past the cut
        # in both lists can never surface no matter how strongly both paths
        # agree on it -- which is precisely the agreement RRF exists to reward.
        # Measured on LOCOMO, widening the fetch and trimming afterwards raised
        # evidence recall at a fixed budget of 20 from 64.0% to 67.7% on one
        # conversation and 67.0% to 71.7% on another, and the gap grew with the
        # budget. It costs no extra model calls: the structured path compiles
        # one query whatever its depth, and the embedding path is a wider
        # top-k over vectors already in memory.
        fetch_limit = limit * _FUSION_OVERFETCH

        # Run structured search and embedding search in parallel
        structured_items, embedding_items = await asyncio.gather(
            self._search_structured_guarded(
                query_text,
                limit=fetch_limit,
                include_expired=include_expired,
                include_superseded=include_superseded,
                include_unconfirmed=include_unconfirmed,
                as_of=as_of,
            ),
            self._search_embedding(
                query_text,
                limit=fetch_limit,
                include_expired=include_expired,
                include_superseded=include_superseded,
                include_unconfirmed=include_unconfirmed,
                as_of=as_of,
            ),
        )

        fused = fuse_results(structured_items, embedding_items, limit=limit)
        if neighbors:
            fused = await expand_with_neighbors(
                self._conversation_required(),
                fused,
                radius=neighbors,
                view=await self._search_view(
                    include_expired=include_expired,
                    include_superseded=include_superseded,
                    include_unconfirmed=include_unconfirmed,
                    as_of=as_of,
                ),
            )
        return fused

    async def _search_structured(
        self,
        query_text: str,
        limit: int = 10,
        *,
        include_expired: bool = False,
        include_superseded: bool = False,
        include_unconfirmed: bool = False,
        dedupe: bool = True,
        as_of: str | None = None,
    ) -> list[SearchItem]:
        """Structured RAG search using LLM query translation + term matching."""
        view = await self._search_view(
            include_expired=include_expired,
            include_superseded=include_superseded,
            include_unconfirmed=include_unconfirmed,
            as_of=as_of,
        )
        return await search_structured(
            self._conversation_required(),
            query_text,
            limit=limit,
            include_expired=include_expired,
            dedupe=dedupe,
            view=view,
        )

    async def _search_structured_guarded(
        self,
        query_text: str,
        limit: int = 10,
        *,
        include_expired: bool = False,
        include_superseded: bool = False,
        include_unconfirmed: bool = False,
        as_of: str | None = None,
    ) -> list[SearchItem]:
        """Structured search for the hybrid path. Degrades to empty on failure.

        The structured path depends on an LLM to translate the query, so it
        fails for ordinary operational reasons -- rate limits, timeouts, a
        transient 5xx. Left unguarded it took the whole of search() with it,
        including the embedding results that had already come back, which is
        the opposite of what search_by_embedding() documents itself as.
        """
        try:
            return await self._search_structured(
                query_text,
                limit=limit,
                include_expired=include_expired,
                include_superseded=include_superseded,
                include_unconfirmed=include_unconfirmed,
                as_of=as_of,
            )
        except Exception:
            logger.warning(
                "Structured search failed for collection %r; "
                "returning embedding results only.",
                self.collection,
                exc_info=True,
            )
            return []

    async def _search_embedding(
        self,
        query_text: str,
        limit: int = 10,
        *,
        include_expired: bool = False,
        include_superseded: bool = False,
        include_unconfirmed: bool = False,
        as_of: str | None = None,
    ) -> list[SearchItem]:
        """Internal embedding search. Logs and degrades to empty on failure."""
        try:
            return await self.search_by_embedding(
                query_text,
                limit=limit,
                include_expired=include_expired,
                include_superseded=include_superseded,
                include_unconfirmed=include_unconfirmed,
                as_of=as_of,
            )
        except Exception:
            logger.warning(
                "Embedding search failed for collection %r; "
                "returning structured results only.",
                self.collection,
                exc_info=True,
            )
            return []

    async def search_by_embedding(
        self,
        query_text: str,
        limit: int = 10,
        min_score: float = 0.3,
        *,
        include_expired: bool = False,
        include_superseded: bool = False,
        include_unconfirmed: bool = False,
        as_of: str | None = None,
    ) -> list[SearchItem]:
        """Embedding-only search without LLM. Used as fallback when structured search fails.

        Directly queries the MessageTextIndex for embedding similarity,
        bypassing the LLM query translation step entirely.

        Args:
            query_text: Search query text.
            limit: Maximum number of results.
            min_score: Minimum similarity score threshold.
            include_expired: If True, include memories past their valid_to date.

        Returns:
            List of SearchItem with type="message".
        """
        as_of = normalize_as_of(as_of)
        if limit < 0:
            raise ValueError("limit cannot be negative")
        if limit == 0:
            return []
        await self._ensure_initialized()
        return await search_by_embedding(
            self._conversation_required(),
            self.collection,
            query_text,
            limit=limit,
            min_score=min_score,
            include_expired=include_expired,
            view=await self._search_view(
                include_expired=include_expired,
                include_superseded=include_superseded,
                include_unconfirmed=include_unconfirmed,
                as_of=as_of,
            ),
        )

    async def delete(
        self,
        query: str,
        *,
        limit: int = 50,
        min_score: float = 0.0,
        dry_run: bool = False,
    ) -> int:
        """Delete knowledge (entities, actions, topics) matching a query.

        For advanced users who want explicit control over deletion.
        Normal users can rely on add() which automatically handles contradictions.

        This removes extracted *knowledge*, not the source messages. The
        original message text stays in the collection as historical evidence.
        Search excludes that source by default; include_superseded=True returns
        it with status="superseded". Unrelated active knowledge extracted from
        the same message remains searchable. Use clear() to remove everything.

        Because matching is semantic, a bare query can match more loosely than
        intended. Prefer previewing with ``dry_run=True``, and/or raising
        ``min_score``, before deleting.

        Args:
            query: Search query to find memories to delete.
            limit: Maximum number of items to consider (default 50).
            min_score: Drop candidates whose native index score is below this.
                Scores here are structured term-match weights, which are
                unbounded. Defaults to 0.0, which keeps every candidate.
            dry_run: If True, report how many items *would* be deleted without
                changing anything.

        Returns:
            Number of knowledge items deleted (or that would be deleted when
            dry_run is True).

        Example:
            # See what would go first
            await memory.delete("likes sushi", dry_run=True)
            # Then delete for real
            deleted = await memory.delete("likes sushi")
        """
        await self._ensure_initialized()

        # The structured path alone, not search(). Two reasons, and the second
        # one is a correctness bug rather than a cost:
        #
        #   - Only extracted knowledge can be superseded, and the embedding
        #     half of search() returns nothing but messages, so all of its work
        #     was discarded by the type check below.
        #   - search() merges the two paths with reciprocal rank fusion, which
        #     collapses items by rendered text. Two distinct semantic refs that
        #     render identically -- the same topic extracted from two messages,
        #     say -- became one result, so delete() retired one ordinal and
        #     silently left the other visible.
        #
        # dedupe=False for the same reason: here two refs that read the same
        # are still two refs, and the caller asked for both to go.
        results = await self._search_structured(query, limit=limit, dedupe=False)

        candidate_ids: list[int] = []
        texts_by_ordinal: dict[int, str] = {}
        ids_by_ordinal: dict[int, str | None] = {}
        for item in results:
            if item.type == "message" or item.score < min_score:
                continue
            ordinal = getattr(item.raw, "semantic_ref_ordinal", None)
            if ordinal is not None:
                candidate_ids.append(ordinal)
                texts_by_ordinal.setdefault(ordinal, item.text)
                ids_by_ordinal.setdefault(ordinal, item.memory_id)

        if not candidate_ids:
            return 0

        # Superseded items are filtered out of search already, but guard anyway
        # so repeated calls do not inflate the reported count.
        hidden = await self._ledger.hidden_ordinals()
        new_ids = [
            ordinal for ordinal in dict.fromkeys(candidate_ids) if ordinal not in hidden
        ]

        if not new_ids or dry_run:
            return len(new_ids)

        now = utc_now()
        await self._ledger.append(
            [
                SupersededRecord(
                    ordinal=ordinal,
                    superseded_by=[],
                    at=now,
                    reason="delete",
                    text=texts_by_ordinal.get(ordinal),
                    query=query,
                    memory_id=ids_by_ordinal.get(ordinal),
                )
                for ordinal in new_ids
            ]
        )

        return len(new_ids)

    async def _detect_and_remove_contradictions(
        self,
        new_content: str,
        *,
        new_ordinals: list[int] | None = None,
    ) -> list[SupersededRecord]:
        """Internal: use an LLM to retire memories the new content contradicts.

        Nothing is destroyed. Contradicted memories are appended to the
        supersession ledger, which hides them from search and keeps enough
        context to undo the judgment later.

        Args:
            new_content: The new content being added.
            new_ordinals: Semantic-ref ordinals the caller's own write just
                produced. They drive the search for what those assertions might
                contradict, are excluded from the results (a memory must not
                contradict itself), and are recorded as what replaced whatever
                is retired.

        Returns:
            The ledger entries appended, empty when nothing was retired.
        """
        ordinals = list(new_ordinals or [])
        if not ordinals:
            return []

        conversation = self._conversation_required()

        async def candidates() -> list[SearchItem]:
            return await find_contradiction_candidates(
                conversation,
                ordinals,
                include_expired=True,
                include_new_peers=True,
            )

        async def new_items() -> list[SearchItem]:
            return await items_for_semrefs(conversation, ordinals, include_expired=True)

        return await detect_contradictions(
            new_content,
            collection=self.collection,
            find_candidates=candidates,
            create_llm=self.config.create_llm,
            append=self._ledger.append,
            superseded_by=ordinals,
            find_new_items=new_items,
        )

    async def history(
        self,
        *,
        include_restored: bool = False,
    ) -> list[SupersededRecord]:
        """Return the supersession ledger, oldest first.

        This is the audit trail: what was retired, when, why, what replaced it,
        and the text it had at the time. Nothing here was destroyed -- every
        entry can be undone with restore().

        Args:
            include_restored: If True, also return entries already reversed by
                restore(). Off by default.

        Returns:
            List of SupersededRecord in insertion order.

        Example:
            for record in await memory.history():
                print(record.at, record.text, "->", record.superseded_by)
        """
        await self._ensure_initialized()
        ledger = await self._ledger.load()
        if include_restored:
            return list(ledger)
        return [r for r in ledger if r.active]

    async def restore(self, ordinals: int | list[int]) -> int:
        """Undo a supersession, making the memory visible to search again.

        The point of an append-only ledger: a wrong contradiction judgment is
        recoverable, because the underlying record was never deleted.

        Args:
            ordinals: Semantic-ref ordinal(s) to restore. Ordinals that are not
                currently superseded are ignored.

        Returns:
            Number of memories restored.

        Example:
            # The LLM decided "works in Portland" contradicted "works in
            # Seattle", but they were two offices.
            (record,) = await memory.history()
            await memory.restore(record.ordinal)
        """
        await self._ensure_initialized()

        wanted = {ordinals} if isinstance(ordinals, int) else set(ordinals)
        return await self._ledger.restore(wanted)

    async def clear(self) -> bool:
        """Clear all memories for this collection.

        Returns:
            True if successful.
        """
        await self._ensure_initialized()
        conversation = self._conversation_required()
        await conversation.storage_provider.clear()  # type: ignore[attr-defined]

        # Commit for SQLite (PostgreSQL handles this automatically)
        if self.config.is_sqlite and hasattr(conversation.storage_provider, "db"):
            conversation.storage_provider.db.commit()  # type: ignore[attr-defined]

        await self._ledger.reset()

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

        ledger = await self._ledger.load()
        active_supersessions = len(await self._ledger.hidden_ordinals())

        backend_name = "postgres" if self.config.is_postgres else "sqlite"

        return {
            "collection": self.collection,
            "total_messages": message_count,
            "total_semantic_refs": semref_count,
            # Superseded refs are still counted above: they exist, they are
            # just not part of the current view.
            "visible_semantic_refs": max(semref_count - active_supersessions, 0),
            "superseded": active_supersessions,
            "ledger_entries": len(ledger),
            "backend": backend_name,
        }

    async def transcript(
        self,
        *,
        start: int = 0,
        limit: int | None = None,
        as_of: str | None = None,
    ) -> list[SearchItem]:
        """Return source messages in conversation order.

        Search is the normal way to build a small context. Callers optimizing
        only for answer quality may instead want the entire transcript, or may
        append it after retrieved evidence so no turn can be missed. This reads
        that transcript from Momex's durable store rather than requiring the
        caller to retain a second copy of the input.

        Args:
            start: Zero-based message ordinal to start from.
            limit: Maximum messages to return. ``None`` means through the end.

        Returns:
            Message SearchItems ordered by ordinal. Scores are zero because
            history is chronological, not a ranked retrieval.
        """
        as_of = normalize_as_of(as_of)
        if start < 0:
            raise ValueError("start cannot be negative")
        if limit is not None and limit < 0:
            raise ValueError("limit cannot be negative")
        if limit == 0:
            return []

        await self._ensure_initialized()
        messages = self._conversation_required().messages
        size = await messages.size()
        if start >= size:
            return []
        end = size if limit is None else min(start + limit, size)
        stored = await messages.get_slice(start, end)
        view = await self._search_view(
            include_expired=True,
            include_superseded=True,
            include_unconfirmed=True,
            as_of=as_of,
        )

        items: list[SearchItem] = []
        for ordinal, message in enumerate(stored, start):
            if not view.allows(message):
                continue
            items.append(item_for_message(message, ordinal, 0.0, view))
        return items

    async def export(self, path: str) -> None:
        """Export all memories to a JSON file.

        Args:
            path: Path to the output JSON file.
        """
        await self._ensure_initialized()
        conversation = self._conversation_required()

        from .exceptions import ExportError

        # Sized from the collections themselves. A fixed upper bound silently
        # truncated anything larger, and an export that drops records without
        # saying so is worse than one that fails.
        messages = await conversation.messages.get_slice(
            0, await conversation.messages.size()
        )
        semrefs = await conversation.semantic_refs.get_slice(
            0, await conversation.semantic_refs.size()
        )

        from typeagent.knowpro import knowledge_schema as kplib

        data = {
            "collection": self.collection,
            "messages": [
                {
                    "text": (
                        " ".join(m.text_chunks) if hasattr(m, "text_chunks") else str(m)
                    ),
                    "speaker": (
                        m.metadata.speaker
                        if hasattr(m, "metadata") and hasattr(m.metadata, "speaker")
                        else None
                    ),
                    "timestamp": m.timestamp if hasattr(m, "timestamp") else None,
                }
                for m in messages
            ],
            "knowledge": [],
        }

        for sr in semrefs:
            k = sr.knowledge
            if isinstance(k, kplib.ConcreteEntity):
                data["knowledge"].append(
                    {
                        "type": "entity",
                        "name": k.name,
                        "types": k.type,
                        "facets": [
                            {"name": f.name, "value": f.value} for f in (k.facets or [])
                        ],
                    }
                )
            elif isinstance(k, kplib.Action):
                data["knowledge"].append(
                    {
                        "type": "action",
                        "verbs": k.verbs,
                        "subject": k.subject_entity_name,
                        "object": k.object_entity_name,
                    }
                )
            else:
                data["knowledge"].append(
                    {
                        "type": "other",
                        "text": str(k),
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

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def close(self) -> None:
        """Release the underlying storage resources.

        Closes the SQLite connection or the PostgreSQL connection pool. Safe to
        call more than once; a later operation transparently re-initializes the
        conversation.
        """
        conversation = self._conversation
        self._conversation = None
        self._initialized = False
        # The ledger caches are backed by the collection's metadata, which is
        # about to be closed. Dropping them means the next operation reads the
        # ledger back from storage rather than trusting a copy that may have
        # been superseded -- by another process, or by the collection having
        # been deleted and recreated under the same name.
        self._ledger.invalidate()

        if conversation is not None:
            await conversation.storage_provider.close()

    async def __aenter__(self) -> "Memory":
        await self._ensure_initialized()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.close()

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
        return str(
            collection_to_db_path(
                self.collection,
                self.config.storage_path,
                DB_FILENAME,
            )
        )

    @property
    def is_initialized(self) -> bool:
        """Check if the memory is initialized."""
        return self._initialized
