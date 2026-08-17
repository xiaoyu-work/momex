"""The append-only supersession ledger.

Retiring a memory never destroys it. The semantic ref stays in the collection
and a ledger entry hides it from search, recording what replaced it, when and
why -- so a wrong contradiction judgment can be reversed and the change itself
is preserved.

The ledger lives in one conversation-metadata blob, which is what makes the
concurrency here delicate: appending is a read-modify-write with no
compare-and-swap underneath, so the writers have to be serialized by hand.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable

from .paths import utc_now
from .results import SupersededRecord

logger = logging.getLogger(__name__)

DELETED_SEMREFS_METADATA_KEY = "momex_deleted_semrefs"

# Append-only supersession ledger. Replaces the tombstone set above, which is
# still read so collections written by older versions keep their deletions.
SUPERSESSION_METADATA_KEY = "momex_supersession_ledger"
SUPERSESSION_LEDGER_VERSION = 1

# Upper bound on how many ordinals a single stored [start, end] pair may expand
# to, so corrupt metadata cannot exhaust memory on load.
_MAX_TOMBSTONE_RANGE = 10_000_000

MetadataReader = Callable[[], Awaitable[Any]]
MetadataWriter = Callable[..., Awaitable[None]]


def encode_deleted_ids(ids: set[int]) -> list[int | list[int]]:
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


def decode_deleted_ids(parsed: Any) -> set[int]:
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


def encode_ledger(records: list[SupersededRecord]) -> dict[str, Any]:
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
                "memory_id": r.memory_id,
            }
            for r in records
        ],
    }


def decode_ledger(parsed: Any) -> list[SupersededRecord]:
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
                memory_id=(
                    raw.get("memory_id")
                    if isinstance(raw.get("memory_id"), str)
                    else None
                ),
            )
        )
    return records


class SupersessionLedger:
    """Reads and writes one collection's ledger through its metadata.

    Deliberately knows nothing about conversations or storage providers: it is
    handed a reader and a writer for the metadata blob, which is the entire
    surface it needs. That keeps the concurrency rules below in one place
    rather than spread across the Memory methods that trigger them.
    """

    def __init__(
        self,
        collection: str,
        read_metadata: MetadataReader,
        write_metadata: MetadataWriter,
    ) -> None:
        self.collection = collection
        self._read_metadata = read_metadata
        self._write_metadata = write_metadata
        self._records: list[SupersededRecord] | None = None
        self._legacy_ids: set[int] | None = None
        # Serializes the read-modify-write below. The ledger is one blob, so
        # two concurrent writers would otherwise clobber each other.
        self._lock = asyncio.Lock()

    def invalidate(self) -> None:
        """Drop the caches, so the next read goes back to storage.

        Called when the connection behind the metadata is released: a cached
        ledger would otherwise outlive the collection it describes, and keep
        hiding ordinals in a collection recreated under the same name.
        """
        self._records = None
        self._legacy_ids = None

    async def load_legacy_ids(self) -> set[int]:
        """Read the legacy tombstone set.

        Superseded by the ledger. Kept because collections written by older
        versions still carry this key; load() migrates it once and nothing
        writes it again except reset(), which clears it so an older reader does
        not resurrect deletions.
        """
        if self._legacy_ids is not None:
            return self._legacy_ids

        metadata = await self._read_metadata()
        deleted_raw = ""
        if metadata.extra and DELETED_SEMREFS_METADATA_KEY in metadata.extra:
            deleted_raw = metadata.extra[DELETED_SEMREFS_METADATA_KEY]

        if not deleted_raw:
            self._legacy_ids = set()
            return self._legacy_ids

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

        self._legacy_ids = decode_deleted_ids(parsed)
        return self._legacy_ids

    async def _store_legacy_ids(self, ids: set[int]) -> None:
        self._legacy_ids = ids
        await self._write_metadata(
            **{DELETED_SEMREFS_METADATA_KEY: json.dumps(encode_deleted_ids(ids))}
        )

    async def load(self) -> list[SupersededRecord]:
        """Load the ledger, migrating legacy tombstones once.

        Collections written before the ledger existed only have the tombstone
        set. Those ordinals are folded in as reason="legacy" records with no
        `superseded_by` and no text, since that information was never kept.
        """
        if self._records is not None:
            return self._records

        metadata = await self._read_metadata()
        raw = ""
        if metadata.extra and SUPERSESSION_METADATA_KEY in metadata.extra:
            raw = metadata.extra[SUPERSESSION_METADATA_KEY]

        records: list[SupersededRecord] = []
        if raw:
            try:
                records = decode_ledger(json.loads(raw))
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
            legacy = await self.load_legacy_ids()
            if legacy:
                now = utc_now()
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

        self._records = records
        return records

    async def store(self, records: list[SupersededRecord]) -> None:
        self._records = records
        await self._write_metadata(
            **{SUPERSESSION_METADATA_KEY: json.dumps(encode_ledger(records))}
        )

    async def hidden_ordinals(self) -> set[int]:
        """Ordinals currently hidden from search: superseded and not restored."""
        return {r.ordinal for r in await self.load() if r.active}

    async def append(self, records: list[SupersededRecord]) -> list[SupersededRecord]:
        """Append entries, skipping ordinals that are already hidden.

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
        async with self._lock:
            self._records = None
            ledger = await self.load()
            hidden = {r.ordinal for r in ledger if r.active}

            added: list[SupersededRecord] = []
            for record in records:
                if record.ordinal in hidden:
                    continue
                hidden.add(record.ordinal)
                added.append(record)

            if added:
                await self.store(ledger + added)
            return added

    async def restore(self, ordinals: set[int]) -> int:
        """Mark superseded entries as restored. Unknown ordinals are ignored."""
        if not ordinals:
            return 0

        # Same read-modify-write as append(), and the same lock: restoring
        # against a stale copy would resurrect entries a concurrent add() had
        # just appended.
        async with self._lock:
            self._records = None
            ledger = await self.load()
            now = utc_now()
            restored = 0
            for record in ledger:
                if record.active and record.ordinal in ordinals:
                    record.restored_at = now
                    restored += 1

            if restored:
                await self.store(ledger)
            return restored

    async def reset(self) -> None:
        """Empty the ledger, for when the underlying records are gone too.

        clear() is the one genuinely destructive operation, and it drops the
        semantic refs as well -- so the ledger has nothing left to describe.
        The legacy key is reset alongside, so an older reader does not
        resurrect deletions the collection no longer contains.
        """
        async with self._lock:
            await self._store_legacy_ids(set())
            await self.store([])
