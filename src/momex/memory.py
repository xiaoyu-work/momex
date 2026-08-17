"""Momex Memory - High-level API wrapping TypeAgent's Structured RAG.

This module provides a simplified Memory API that uses TypeAgent's full
indexing system (SemanticRefs, TermIndex) combined with embedding similarity
search for robust hybrid retrieval.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import re
from typing import Any, TYPE_CHECKING

from .config import MomexConfig
from .exceptions import ValidationError

if TYPE_CHECKING:
    from typeagent.knowpro.conversation_base import ConversationBase
    from typeagent.knowpro.convsettings import (
        MessageTextIndexSettings,
        RelatedTermIndexSettings,
    )


logger = logging.getLogger(__name__)

DELETED_SEMREFS_METADATA_KEY = "momex_deleted_semrefs"

# Append-only supersession ledger. Replaces the tombstone set above, which is
# still read so collections written by older versions keep their deletions.
SUPERSESSION_METADATA_KEY = "momex_supersession_ledger"
SUPERSESSION_LEDGER_VERSION = 1

# Upper bound on how many ordinals a single stored [start, end] pair may expand
# to, so corrupt metadata cannot exhaust memory on load.
_MAX_TOMBSTONE_RANGE = 10_000_000

# Reciprocal rank fusion constant, from the original RRF paper. Damps the
# influence of the top ranks so one list cannot dominate the merged order.
RRF_K = 60


@dataclass
class SupersededRecord:
    """One entry in the append-only supersession ledger.

    A memory is never destroyed. It is marked as superseded by whatever
    replaced it, hidden from search, and can be restored -- so a bad
    contradiction judgment is recoverable and the change itself is preserved.
    """

    ordinal: int
    """Semantic-ref ordinal of the memory that was retired."""

    superseded_by: list[int]
    """Ordinals of the memories that replaced it. Empty for explicit delete()."""

    at: str
    """ISO-8601 UTC timestamp of the supersession."""

    reason: str
    """One of "contradiction", "delete", or "legacy"."""

    text: str | None = None
    """Rendered text at the time of supersession, for auditing."""

    query: str | None = None
    """The content or query that triggered it."""

    restored_at: str | None = None
    """Set when restore() reversed this record. Non-None means inactive."""

    @property
    def active(self) -> bool:
        return self.restored_at is None


@dataclass
class AddResult:
    """Result of adding memories."""

    messages_added: int
    entities_extracted: int
    contradictions_removed: int = 0
    collections: list[str] | None = None
    superseded: list[SupersededRecord] | None = None
    """What add() retired, not just how many. None when nothing was retired."""


@dataclass
class SearchItem:
    """A single search result item."""

    type: str  # Uses TypeAgent's native knowledge_type: "entity", "action", "topic", "message"
    text: str
    score: float  # Native score of the index that produced this item
    raw: Any  # Original TypeAgent object (SemanticRef or Message)
    timestamp: str | None = None  # When the memory was recorded (ISO format)
    valid_from: str | None = None
    valid_to: str | None = None
    # Rank-fusion score used to order hybrid search() results. None when the
    # item comes from a single-path search such as search_by_embedding().
    fusion_score: float | None = None


def _utc_now() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# Characters that must never reach the filesystem as part of a path segment.
# Both separators are included: a collection segment is always exactly one
# directory, so "a/b" must not silently become two of them.
_UNSAFE_PATH_CHARS = re.compile(r'[<>"|?*:\\/]')


def _sanitize_collection_part(part: str, collection: str) -> str:
    """Map one ':'-delimited collection segment to a safe path component.

    Segments come from the caller, and in a multi-tenant deployment that
    usually means from user input. A segment of "." or ".." would resolve
    *outside* the storage directory, so those are rejected rather than
    sanitized: silently rewriting them would let two different tenants land on
    the same directory.
    """
    if not part.strip(". \t\r\n"):
        raise ValidationError(
            message=(
                f"Invalid collection name {collection!r}: segment {part!r} is "
                "empty or consists only of dots/whitespace."
            ),
            field="collection",
            value=collection,
            suggestion=("Use non-empty segments separated by ':', e.g. 'user:alice'."),
        )
    return _UNSAFE_PATH_CHARS.sub("_", part)


def _collection_to_path(collection: str) -> Path:
    """Convert a collection name to a relative path, one segment per ':'.

    Converts "user:xiaoyuzhang" to Path("user/xiaoyuzhang").
    """
    parts = [
        _sanitize_collection_part(part, collection) for part in collection.split(":")
    ]
    return Path(*parts)


def _collection_to_db_path(collection: str, base_path: str, db_name: str) -> Path:
    """Convert collection name to database path.

    Converts "momex:engineering:xiaoyuzhang" to
    Path("base_path/momex/engineering/xiaoyuzhang/db_name")
    """
    return Path(base_path) / _collection_to_path(collection) / db_name


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


def _encode_deleted_ids(ids: set[int]) -> list[int | list[int]]:
    """Range-encode tombstoned ordinals for compact storage.

    All knowledge extracted from one message gets consecutive ordinals, so
    deletions arrive in runs. Storing runs as [start, end] keeps the metadata
    payload proportional to the number of deleted *regions* rather than the
    number of deleted items.
    """
    encoded: list[int | list[int]] = []
    start: int | None = None
    prev = 0

    for ordinal in sorted(ids):
        if start is None:
            start = prev = ordinal
        elif ordinal == prev + 1:
            prev = ordinal
        else:
            encoded.append(start if start == prev else [start, prev])
            start = prev = ordinal

    if start is not None:
        encoded.append(start if start == prev else [start, prev])
    return encoded


def _decode_deleted_ids(parsed: Any) -> set[int]:
    """Decode tombstones, accepting range pairs and the older flat int list."""
    ids: set[int] = set()
    if not isinstance(parsed, list):
        return ids

    for item in parsed:
        if isinstance(item, bool):
            continue
        if isinstance(item, int):
            ids.add(item)
        elif isinstance(item, str) and item.isdigit():
            ids.add(int(item))
        elif isinstance(item, list) and len(item) == 2:
            try:
                start, end = int(item[0]), int(item[1])
            except (TypeError, ValueError):
                continue
            # Guard against a corrupt range expanding into a huge set.
            if 0 <= start <= end <= start + _MAX_TOMBSTONE_RANGE:
                ids.update(range(start, end + 1))
    return ids


def _encode_ledger(records: list[SupersededRecord]) -> dict[str, Any]:
    """Serialize the supersession ledger.

    Unlike the tombstone set this replaces, the ledger is append-only: entries
    are added and marked restored, never removed. It is versioned so the shape
    can change without silently misreading old data.
    """
    return {
        "version": SUPERSESSION_LEDGER_VERSION,
        "records": [
            {
                "ordinal": r.ordinal,
                "superseded_by": list(r.superseded_by),
                "at": r.at,
                "reason": r.reason,
                "text": r.text,
                "query": r.query,
                "restored_at": r.restored_at,
            }
            for r in records
        ],
    }


def _decode_ledger(parsed: Any) -> list[SupersededRecord]:
    """Decode the ledger, skipping entries that are not usable.

    A corrupt entry must not take the whole ledger with it: dropping one record
    hides one fewer memory, while raising would make the collection unreadable.
    """
    if not isinstance(parsed, dict):
        return []

    version = parsed.get("version")
    if version != SUPERSESSION_LEDGER_VERSION:
        logger.warning(
            "Ignoring supersession ledger with unsupported version %r.", version
        )
        return []

    raw_records = parsed.get("records")
    if not isinstance(raw_records, list):
        return []

    records: list[SupersededRecord] = []
    for raw in raw_records:
        if not isinstance(raw, dict):
            continue
        ordinal = raw.get("ordinal")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int):
            continue

        superseded_by = [
            o
            for o in (raw.get("superseded_by") or [])
            if isinstance(o, int) and not isinstance(o, bool)
        ]
        at = raw.get("at")
        reason = raw.get("reason")
        records.append(
            SupersededRecord(
                ordinal=ordinal,
                superseded_by=superseded_by,
                at=at if isinstance(at, str) else "",
                reason=reason if isinstance(reason, str) else "unknown",
                text=raw.get("text") if isinstance(raw.get("text"), str) else None,
                query=raw.get("query") if isinstance(raw.get("query"), str) else None,
                restored_at=(
                    raw.get("restored_at")
                    if isinstance(raw.get("restored_at"), str)
                    else None
                ),
            )
        )
    return records


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
        self._supersession_ledger: list[SupersededRecord] | None = None
        # Guards the temporary auto_extract_knowledge toggle in add(), which
        # mutates state shared by every concurrent call on this instance.
        self._settings_lock = asyncio.Lock()
        # Serializes the ledger's read-modify-write. It lives in one metadata
        # blob, so two concurrent writers would otherwise clobber each other.
        self._ledger_lock = asyncio.Lock()

        # Auto-load dotenv
        self._load_dotenv()

    def _load_dotenv(self) -> None:
        """Load environment variables from .env file."""
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

    async def _load_deleted_semref_ids(self) -> set[int]:
        """Read the legacy tombstone set.

        Superseded by the supersession ledger. Kept because collections written
        by older versions still carry this key; `_load_ledger` migrates it once
        and nothing writes it again except `clear()`, which resets it so an
        older reader does not resurrect deletions.
        """
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
            logger.warning(
                "Ignoring corrupt %s metadata for collection %r; "
                "previously deleted memories may reappear.",
                DELETED_SEMREFS_METADATA_KEY,
                self.collection,
                exc_info=True,
            )
            parsed = []

        deleted_ids = _decode_deleted_ids(parsed)

        self._deleted_semref_ids = deleted_ids
        return deleted_ids

    async def _store_deleted_semref_ids(self, deleted_ids: set[int]) -> None:
        serialized = json.dumps(_encode_deleted_ids(deleted_ids))
        await self._set_conversation_metadata(
            **{DELETED_SEMREFS_METADATA_KEY: serialized}
        )

    async def _load_ledger(self) -> list[SupersededRecord]:
        """Load the supersession ledger, migrating legacy tombstones once.

        Collections written before the ledger existed only have the tombstone
        set. Those ordinals are folded in as reason="legacy" records with no
        `superseded_by` and no text, since that information was never kept.
        """
        if self._supersession_ledger is not None:
            return self._supersession_ledger

        metadata = await self._get_conversation_metadata()
        raw = ""
        if metadata.extra and SUPERSESSION_METADATA_KEY in metadata.extra:
            raw = metadata.extra[SUPERSESSION_METADATA_KEY]

        records: list[SupersededRecord] = []
        if raw:
            try:
                records = _decode_ledger(json.loads(raw))
            except json.JSONDecodeError:
                logger.warning(
                    "Ignoring corrupt %s metadata for collection %r; "
                    "superseded memories may reappear.",
                    SUPERSESSION_METADATA_KEY,
                    self.collection,
                    exc_info=True,
                )
                records = []
        else:
            legacy = await self._load_deleted_semref_ids()
            if legacy:
                now = _utc_now()
                records = [
                    SupersededRecord(
                        ordinal=ordinal,
                        superseded_by=[],
                        at=now,
                        reason="legacy",
                    )
                    for ordinal in sorted(legacy)
                ]
                logger.info(
                    "Migrated %d legacy tombstone(s) to the supersession ledger "
                    "for collection %r.",
                    len(records),
                    self.collection,
                )

        self._supersession_ledger = records
        return records

    async def _store_ledger(self, records: list[SupersededRecord]) -> None:
        self._supersession_ledger = records
        await self._set_conversation_metadata(
            **{SUPERSESSION_METADATA_KEY: json.dumps(_encode_ledger(records))}
        )

    async def _hidden_ordinals(self) -> set[int]:
        """Ordinals currently hidden from search: superseded and not restored."""
        return {r.ordinal for r in await self._load_ledger() if r.active}

    async def _append_supersessions(
        self, records: list[SupersededRecord]
    ) -> list[SupersededRecord]:
        """Append entries to the ledger, skipping already-hidden ordinals.

        The ledger is stored as a single metadata blob, so appending to it is a
        read-modify-write. Concurrent add() and delete() calls therefore raced:
        each read the same list and wrote back its own version, and whichever
        finished last erased the other's entries -- leaving memories that had
        been judged contradictory visible again.

        The lock serializes those writers, and the cache is dropped inside it
        so the merge starts from what is actually stored rather than from a
        copy this instance read earlier. Concurrency *between processes* is
        still not covered: the underlying blob has no compare-and-swap.
        """
        async with self._ledger_lock:
            self._supersession_ledger = None
            ledger = await self._load_ledger()
            hidden = {r.ordinal for r in ledger if r.active}

            added: list[SupersededRecord] = []
            for record in records:
                if record.ordinal in hidden:
                    continue
                hidden.add(record.ordinal)
                added.append(record)

            if added:
                await self._store_ledger(ledger + added)
            return added

    def _filter_search_results(
        self,
        results,
        hidden_ordinals: set[int],
    ):
        """Drop superseded knowledge from raw search results.

        The underlying records still exist; the ledger decides what is visible.
        """
        if not hidden_ordinals:
            return results
        filtered = []
        for search_result in results:
            knowledge_matches = {}
            for ktype, kmatches in search_result.knowledge_matches.items():
                kept = [
                    match
                    for match in kmatches.semantic_ref_matches
                    if match.semantic_ref_ordinal not in hidden_ordinals
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
        from typeagent.knowpro.interfaces import ConversationMetadata
        from typeagent.knowpro.universal_message import ConversationMessage
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
        valid_from: str | None = None,
        valid_to: str | None = None,
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
            valid_from: ISO date string (e.g., "2026-04-01"). Memory is only relevant
                   from this date. None means no start constraint.
            valid_to: ISO date string (e.g., "2026-05-01"). Memory expires after this
                   date and will be excluded from search results. None means no expiry.

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
        valid_from = self._validate_iso_date(valid_from, "valid_from")
        valid_to = self._validate_iso_date(valid_to, "valid_to")

        # Contradiction handling runs *after* the write (see below): retiring
        # first means a failed insert leaves the old facts hidden and the
        # replacement missing.
        superseded: list[SupersededRecord] = []

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
        # Store time windows as tags so they survive serialization
        time_tags: list[str] = []
        if valid_from:
            time_tags.append(f"valid_from:{valid_from}")
        if valid_to:
            time_tags.append(f"valid_to:{valid_to}")

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
                tags=list(time_tags),
                timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
            ta_messages.append(ta_message)

        if not ta_messages:
            return AddResult(
                messages_added=0,
                entities_extracted=0,
                contradictions_removed=0,
                collections=[self.collection],
            )

        # Use TypeAgent's add_messages_with_indexing for full knowledge extraction
        if infer:
            # Ordinals at or above this baseline belong to the write below, and
            # must never be retired as contradictions of themselves.
            try:
                semref_baseline = await conversation_obj.semantic_refs.size()
            except Exception:  # pragma: no cover - defensive
                semref_baseline = None

            result = await conversation_obj.add_messages_with_indexing(ta_messages)

            # Only now that the new content is durable do we retire whatever it
            # contradicts. Detection matches the new facts too, so exclude the
            # semrefs this call just produced -- and point the ledger's
            # superseded_by at them, so the replacement is recorded, not just
            # the removal.
            if detect_contradictions:
                content_text = (
                    messages
                    if isinstance(messages, str)
                    else " ".join(m.get("content", "") for m in messages)
                )
                new_ordinals: list[int] = []
                if semref_baseline is not None:
                    new_ordinals = list(
                        range(semref_baseline, semref_baseline + result.semrefs_added)
                    )
                superseded = await self._detect_and_remove_contradictions(
                    content_text,
                    protect_semrefs_from=semref_baseline,
                    superseded_by=new_ordinals,
                )

            return AddResult(
                messages_added=result.messages_added,
                entities_extracted=result.semrefs_added,
                contradictions_removed=len(superseded),
                collections=[self.collection],
                superseded=superseded or None,
            )
        else:
            # Direct add without LLM processing. The toggle below mutates state
            # shared by every concurrent add() on this instance, so serialize
            # the whole window -- an interleaved call would otherwise have its
            # extraction silently disabled and could restore a stale value.
            async with self._settings_lock:
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
                entities_extracted=0,
                contradictions_removed=0,
                collections=[self.collection],
            )

    @staticmethod
    def _extract_time_window(msg: Any) -> tuple[str | None, str | None]:
        """Extract valid_from/valid_to from a message's tags."""
        tags = getattr(msg, "tags", None) or []
        valid_from = None
        valid_to = None
        for tag in tags:
            if isinstance(tag, str):
                if tag.startswith("valid_from:"):
                    valid_from = tag[len("valid_from:") :]
                elif tag.startswith("valid_to:"):
                    valid_to = tag[len("valid_to:") :]
        return valid_from, valid_to

    @staticmethod
    def _validate_iso_date(value: str | None, field: str) -> str | None:
        """Normalize an ISO date, or raise if it cannot be compared safely.

        The window checks compare these values lexicographically against
        `YYYY-MM-DD`, which is only correct for zero-padded ISO dates. An
        unpadded string like "2026-4-1" would sort *after* "2026-08-17" and
        silently never expire, so reject it at write time instead.
        """
        if value is None:
            return None
        try:
            parsed = date.fromisoformat(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{field} must be an ISO date string (YYYY-MM-DD); got {value!r}"
            ) from exc
        return parsed.isoformat()

    @staticmethod
    def _is_expired(valid_to: str | None) -> bool:
        """Check if a time window has expired (valid_to < today UTC)."""
        if not valid_to:
            return False
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return valid_to < today

    @staticmethod
    def _is_not_yet_active(valid_from: str | None) -> bool:
        """Check if a time window has not opened yet (valid_from > today UTC)."""
        if not valid_from:
            return False
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return valid_from > today

    @classmethod
    def _is_outside_window(cls, valid_from: str | None, valid_to: str | None) -> bool:
        """True when today falls outside [valid_from, valid_to]."""
        return cls._is_expired(valid_to) or cls._is_not_yet_active(valid_from)

    async def _get_source_messages(self, sem_refs) -> dict[int, Any]:
        """Batch-fetch the source message of each semantic ref, keyed by ordinal."""
        conversation = self._conversation_required()
        ordinals = sorted(
            {
                sem_ref.range.start.message_ordinal
                for sem_ref in sem_refs
                if getattr(sem_ref, "range", None)
            }
        )
        if not ordinals:
            return {}

        try:
            msgs = await conversation.messages.get_multiple(ordinals)
            return dict(zip(ordinals, msgs))
        except (IndexError, KeyError):
            msg_map: dict[int, Any] = {}
            for ordinal in ordinals:
                try:
                    msg_map[ordinal] = await conversation.messages.get_item(ordinal)
                except (IndexError, KeyError):
                    continue
            return msg_map

    async def search(
        self,
        query_text: str,
        limit: int = 10,
        *,
        include_expired: bool = False,
        include_superseded: bool = False,
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

        Returns:
            List of SearchItem with type, text, score, and raw TypeAgent object,
            ordered by fusion_score.

        Neither path is allowed to take the other down: if one fails it is
        logged and contributes nothing, and the results of the other are
        returned on their own. Both failing yields an empty list.
        """
        await self._ensure_initialized()

        # Run structured search and embedding search in parallel
        structured_items, embedding_items = await asyncio.gather(
            self._search_structured_guarded(
                query_text,
                limit=limit,
                include_expired=include_expired,
                include_superseded=include_superseded,
            ),
            self._search_embedding(
                query_text, limit=limit, include_expired=include_expired
            ),
        )

        return self._fuse_results(structured_items, embedding_items, limit=limit)

    async def _search_structured_guarded(
        self,
        query_text: str,
        limit: int = 10,
        *,
        include_expired: bool = False,
        include_superseded: bool = False,
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
            )
        except Exception:
            logger.warning(
                "Structured search failed for collection %r; "
                "returning embedding results only.",
                self.collection,
                exc_info=True,
            )
            return []

    @staticmethod
    def _fuse_results(
        *result_lists: list[SearchItem],
        limit: int,
    ) -> list[SearchItem]:
        """Merge ranked result lists using reciprocal rank fusion.

        Structured search returns term-match weights, which are unbounded and
        routinely exceed 1, while embedding search returns cosine similarities
        in [0, 1]. Sorting the two together by raw score is meaningless, so they
        are combined by rank instead of by magnitude.
        """
        best: dict[str, SearchItem] = {}
        fused_scores: dict[str, float] = {}

        for items in result_lists:
            seen: set[str] = set()
            for rank, item in enumerate(items):
                # Same text from two indexes is one memory, and should be
                # rewarded for appearing in both -- but only once per list.
                if item.text in seen:
                    continue
                seen.add(item.text)
                fused_scores[item.text] = fused_scores.get(item.text, 0.0) + 1.0 / (
                    RRF_K + rank + 1
                )
                best.setdefault(item.text, item)

        for text, item in best.items():
            item.fusion_score = fused_scores[text]

        ordered = sorted(
            best.values(),
            key=lambda item: (item.fusion_score or 0.0, item.score),
            reverse=True,
        )
        return ordered[:limit]

    async def _search_structured(
        self,
        query_text: str,
        limit: int = 10,
        *,
        include_expired: bool = False,
        include_superseded: bool = False,
    ) -> list[SearchItem]:
        """Structured RAG search using LLM query translation + term matching."""
        conversation = self._conversation_required()

        import typechat

        from typeagent.aitools import utils
        from typeagent.knowpro import (
            convknowledge,
        )
        from typeagent.knowpro import (
            search_query_schema,
            searchlang,
        )
        from typeagent.knowpro import knowledge_schema as kplib
        from typeagent.knowpro.interfaces import Topic

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
                apply_scope=False,
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
        if not include_superseded:
            hidden = await self._hidden_ordinals()
            if hidden:
                search_results = self._filter_search_results(search_results, hidden)

        # Collect all ordinals first for batch fetching
        semref_requests: list[tuple[int, float]] = []  # (ordinal, score)
        msg_requests: list[tuple[int, float]] = []  # (ordinal, score)

        for search_result in search_results:
            for _, matches in search_result.knowledge_matches.items():
                for scored in matches.semantic_ref_matches[:limit]:
                    semref_requests.append((scored.semantic_ref_ordinal, scored.score))
            for msg_match in search_result.message_matches[:limit]:
                msg_requests.append((msg_match.message_ordinal, msg_match.score))

        # Batch fetch SemanticRefs
        if semref_requests:
            ordinals = [o for o, _ in semref_requests]
            try:
                sem_refs = await conversation.semantic_refs.get_multiple(ordinals)
                sem_ref_map = dict(zip(ordinals, sem_refs))
            except (IndexError, KeyError):
                sem_ref_map = {}
                for o in ordinals:
                    try:
                        sem_ref_map[o] = await conversation.semantic_refs.get_item(o)
                    except (IndexError, KeyError):
                        pass

            # Knowledge inherits the timestamp and validity window of the
            # message it was extracted from, so fetch those up front.
            src_msg_map = await self._get_source_messages(sem_ref_map.values())

            for ordinal, score in semref_requests:
                sem_ref = sem_ref_map.get(ordinal)
                if sem_ref is None:
                    continue

                knowledge = sem_ref.knowledge
                k_type = knowledge.knowledge_type

                src_timestamp: str | None = None
                valid_from: str | None = None
                valid_to: str | None = None
                if getattr(sem_ref, "range", None):
                    src_msg = src_msg_map.get(sem_ref.range.start.message_ordinal)
                    if src_msg is not None:
                        src_timestamp = getattr(src_msg, "timestamp", None)
                        valid_from, valid_to = self._extract_time_window(src_msg)

                if not include_expired and self._is_outside_window(
                    valid_from, valid_to
                ):
                    continue

                if isinstance(knowledge, kplib.ConcreteEntity):
                    text = knowledge.name
                    if knowledge.type:
                        text += f" (type: {', '.join(knowledge.type)})"
                    if knowledge.facets:
                        facets = [
                            f"{f.name}: {f.value}" for f in knowledge.facets if f.value
                        ]
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

                items.append(
                    SearchItem(
                        type=k_type,
                        text=text,
                        score=score,
                        raw=sem_ref,
                        timestamp=src_timestamp,
                        valid_from=valid_from,
                        valid_to=valid_to,
                    )
                )

        # Batch fetch Messages
        if msg_requests:
            msg_ordinals = [o for o, _ in msg_requests]
            try:
                msgs = await conversation.messages.get_multiple(msg_ordinals)
                msg_map = dict(zip(msg_ordinals, msgs))
            except (IndexError, KeyError):
                msg_map = {}
                for o in msg_ordinals:
                    try:
                        msg_map[o] = await conversation.messages.get_item(o)
                    except (IndexError, KeyError):
                        pass

            for ordinal, score in msg_requests:
                msg = msg_map.get(ordinal)
                if msg is None:
                    continue

                vf, vt = self._extract_time_window(msg)
                if not include_expired and self._is_outside_window(vf, vt):
                    continue

                text = (
                    " ".join(msg.text_chunks)
                    if hasattr(msg, "text_chunks")
                    else str(msg)
                )

                items.append(
                    SearchItem(
                        type="message",
                        text=text,
                        score=score,
                        raw=msg,
                        timestamp=getattr(msg, "timestamp", None),
                        valid_from=vf,
                        valid_to=vt,
                    )
                )

        # Sort by score and limit
        items.sort(key=lambda x: x.score, reverse=True)
        return items[:limit]

    async def _search_embedding(
        self,
        query_text: str,
        limit: int = 10,
        *,
        include_expired: bool = False,
    ) -> list[SearchItem]:
        """Internal embedding search. Logs and degrades to empty on failure."""
        try:
            return await self.search_by_embedding(
                query_text, limit=limit, include_expired=include_expired
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
        await self._ensure_initialized()
        conversation = self._conversation_required()

        # Get the message text index from secondary indexes
        if (
            conversation.secondary_indexes is None
            or conversation.secondary_indexes.message_index is None
        ):
            return []

        msg_index = conversation.secondary_indexes.message_index

        try:
            scored_ordinals = await msg_index.lookup_messages(
                query_text,
                max_matches=limit,
                threshold_score=min_score,
            )
        except Exception:
            logger.warning(
                "Message index lookup failed for collection %r.",
                self.collection,
                exc_info=True,
            )
            return []

        if not scored_ordinals:
            return []

        # Fetch messages and build SearchItems
        items: list[SearchItem] = []
        for scored in scored_ordinals:
            try:
                msg = await conversation.messages.get_item(scored.message_ordinal)

                vf, vt = self._extract_time_window(msg)
                if not include_expired and self._is_outside_window(vf, vt):
                    continue

                text = (
                    " ".join(msg.text_chunks)
                    if hasattr(msg, "text_chunks")
                    else str(msg)
                )
                items.append(
                    SearchItem(
                        type="message",
                        text=text,
                        score=scored.score,
                        raw=msg,
                        timestamp=getattr(msg, "timestamp", None),
                        valid_from=vf,
                        valid_to=vt,
                    )
                )
            except (IndexError, KeyError):
                continue

        items.sort(key=lambda x: x.score, reverse=True)
        return items[:limit]

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
        original message text stays in the collection and can still surface via
        search_by_embedding() (and therefore via the embedding half of
        search()). Use clear() to remove everything in the collection.

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
        # This mirrors what _detect_and_remove_contradictions already does.
        results = await self._search_structured(query, limit=limit)

        candidate_ids: list[int] = []
        texts_by_ordinal: dict[int, str] = {}
        for item in results:
            if item.type == "message" or item.score < min_score:
                continue
            ordinal = getattr(item.raw, "semantic_ref_ordinal", None)
            if ordinal is not None:
                candidate_ids.append(ordinal)
                texts_by_ordinal.setdefault(ordinal, item.text)

        if not candidate_ids:
            return 0

        # Superseded items are filtered out of search already, but guard anyway
        # so repeated calls do not inflate the reported count.
        hidden = await self._hidden_ordinals()
        new_ids = [
            ordinal for ordinal in dict.fromkeys(candidate_ids) if ordinal not in hidden
        ]

        if not new_ids or dry_run:
            return len(new_ids)

        now = _utc_now()
        await self._append_supersessions(
            [
                SupersededRecord(
                    ordinal=ordinal,
                    superseded_by=[],
                    at=now,
                    reason="delete",
                    text=texts_by_ordinal.get(ordinal),
                    query=query,
                )
                for ordinal in new_ids
            ]
        )

        return len(new_ids)

    async def _detect_and_remove_contradictions(
        self,
        new_content: str,
        *,
        protect_semrefs_from: int | None = None,
        superseded_by: list[int] | None = None,
    ) -> list[SupersededRecord]:
        """Internal: use an LLM to retire memories the new content contradicts.

        Nothing is destroyed. Contradicted memories are appended to the
        supersession ledger, which hides them from search and keeps enough
        context to undo the judgment later.

        Args:
            new_content: The new content being added.
            protect_semrefs_from: Semantic-ref ordinal marking the start of the
                knowledge extracted by the caller's own write. Those refs match
                the query by construction and must be excluded, or the new
                memory would be retired as a contradiction of itself.
            superseded_by: Ordinals of the knowledge that replaced them,
                recorded on each ledger entry.

        Returns:
            The ledger entries appended, empty when nothing was retired.
        """
        # Only extracted knowledge can contradict; message hits are discarded
        # below. The embedding half of search() returns nothing but messages,
        # so run the structured path alone and skip that wasted round trip.
        #
        # This lookup is itself an LLM round trip, so it fails for the same
        # ordinary reasons the write does. It must not escape: add() has
        # already committed the new messages by this point, and raising here
        # would report a failure for a write that actually landed.
        try:
            results = await self._search_structured(new_content, limit=20)
        except Exception:
            logger.warning(
                "Contradiction detection lookup failed for collection %r; "
                "the new memory was added without it.",
                self.collection,
                exc_info=True,
            )
            return []

        if not results:
            return []

        # Build context of existing memories. Only extracted knowledge can be
        # retired, and the caller's own just-written refs are excluded. Prompt
        # indices are dense so a small model has no gaps to misread; candidates
        # maps them back to the original results.
        candidates: list[SearchItem] = []
        for item in results:
            if item.type == "message":
                continue
            if protect_semrefs_from is not None:
                ordinal = getattr(item.raw, "semantic_ref_ordinal", None)
                if ordinal is not None and ordinal >= protect_semrefs_from:
                    continue
            candidates.append(item)

        existing_memories = [
            f"{i}: [{item.type}] {item.text}" for i, item in enumerate(candidates)
        ]

        if not existing_memories:
            return []

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
                return []

            # Parse indices
            indices = []
            for part in response_text.replace(" ", "").split(","):
                try:
                    idx = int(part)
                    if 0 <= idx < len(candidates):
                        indices.append(idx)
                except ValueError:
                    continue

            if not indices:
                return []

            # Retire the contradicted memories
            semref_ids = []
            texts_by_ordinal: dict[int, str] = {}
            for idx in indices:
                candidate = candidates[idx]
                ordinal = getattr(candidate.raw, "semantic_ref_ordinal", None)
                if ordinal is not None:
                    semref_ids.append(ordinal)
                    texts_by_ordinal.setdefault(ordinal, candidate.text)

            if semref_ids:
                now = _utc_now()
                added = await self._append_supersessions(
                    [
                        SupersededRecord(
                            ordinal=ordinal,
                            superseded_by=list(superseded_by or []),
                            at=now,
                            reason="contradiction",
                            text=texts_by_ordinal.get(ordinal),
                            query=new_content,
                        )
                        for ordinal in dict.fromkeys(semref_ids)
                    ]
                )
                return added

        except Exception:
            # Contradiction detection is best-effort and must never block add().
            logger.warning(
                "Contradiction detection failed for collection %r; "
                "adding the new memory without it.",
                self.collection,
                exc_info=True,
            )

        return []

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
        ledger = await self._load_ledger()
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
        if not wanted:
            return 0

        # Same read-modify-write as _append_supersessions, and the same lock:
        # restoring against a stale copy would resurrect entries that a
        # concurrent add() had just appended.
        async with self._ledger_lock:
            self._supersession_ledger = None
            ledger = await self._load_ledger()
            now = _utc_now()
            restored = 0
            for record in ledger:
                if record.active and record.ordinal in wanted:
                    record.restored_at = now
                    restored += 1

            if restored:
                await self._store_ledger(ledger)
            return restored

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

        # clear() is the one genuinely destructive operation, and it drops the
        # underlying records too -- so the ledger has nothing left to describe.
        # Under the same lock as the appenders, so a concurrent supersession
        # cannot land entries pointing at refs that no longer exist.
        async with self._ledger_lock:
            self._deleted_semref_ids = set()
            await self._store_deleted_semref_ids(self._deleted_semref_ids)
            await self._store_ledger([])

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

        ledger = await self._load_ledger()
        active_supersessions = sum(1 for r in ledger if r.active)

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
        # Both caches are backed by the collection's metadata, which is about
        # to be closed. Dropping them together means the next operation reads
        # the ledger back from storage rather than trusting a copy that may
        # have been superseded -- by another process, or by the collection
        # having been deleted and recreated under the same name.
        self._deleted_semref_ids = None
        self._supersession_ledger = None

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
            _collection_to_db_path(
                self.collection,
                self.config.storage_path,
                "memory.db",
            )
        )

    @property
    def is_initialized(self) -> bool:
        """Check if the memory is initialized."""
        return self._initialized
