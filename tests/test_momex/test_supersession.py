"""Tests for the append-only supersession ledger.

Retiring a memory no longer destroys it. The semantic ref stays in the
collection; a ledger entry hides it from search and records what replaced it,
when, and why -- so a wrong judgment is recoverable and the change itself is
preserved.
"""

import json

import pytest

from momex import LLMConfig, Memory, MomexConfig, StorageConfig, SupersededRecord
from momex.memory import (
    _decode_ledger,
    _encode_ledger,
    SUPERSESSION_LEDGER_VERSION,
    SUPERSESSION_METADATA_KEY,
)

# --- serialization ---------------------------------------------------------


def _record(ordinal, **kw):
    kw.setdefault("superseded_by", [])
    kw.setdefault("at", "2026-08-17T00:00:00Z")
    kw.setdefault("reason", "contradiction")
    return SupersededRecord(ordinal=ordinal, **kw)


class TestSerialization:
    def test_round_trip(self):
        records = [
            _record(1, superseded_by=[9, 10], text="likes sushi", query="no sushi"),
            _record(4, reason="delete", restored_at="2026-08-18T00:00:00Z"),
        ]
        decoded = _decode_ledger(json.loads(json.dumps(_encode_ledger(records))))
        assert decoded == records

    def test_empty_round_trip(self):
        assert _decode_ledger(json.loads(json.dumps(_encode_ledger([])))) == []

    def test_payload_is_versioned(self):
        assert _encode_ledger([])["version"] == SUPERSESSION_LEDGER_VERSION

    def test_unknown_version_is_ignored(self):
        """Better to show a memory that should be hidden than to misread the
        ledger and hide the wrong ones."""
        assert _decode_ledger({"version": 999, "records": [{"ordinal": 1}]}) == []

    def test_non_dict_yields_empty(self):
        assert _decode_ledger([1, 2, 3]) == []
        assert _decode_ledger(None) == []

    def test_bad_records_are_skipped_not_fatal(self):
        """One corrupt entry must not take the whole ledger with it."""
        decoded = _decode_ledger(
            {
                "version": SUPERSESSION_LEDGER_VERSION,
                "records": [
                    {"ordinal": "nope"},
                    None,
                    {"ordinal": True},
                    {"ordinal": 5, "at": "t", "reason": "delete"},
                ],
            }
        )
        assert [r.ordinal for r in decoded] == [5]

    def test_missing_fields_get_safe_defaults(self):
        (record,) = _decode_ledger(
            {"version": SUPERSESSION_LEDGER_VERSION, "records": [{"ordinal": 3}]}
        )
        assert record.ordinal == 3
        assert record.superseded_by == []
        assert record.reason == "unknown"
        assert record.text is None
        assert record.active


class TestActiveFlag:
    def test_active_until_restored(self):
        record = _record(1)
        assert record.active
        record.restored_at = "2026-08-18T00:00:00Z"
        assert not record.active


# --- ledger behaviour on a Memory -----------------------------------------


class _FakeMetadata:
    def __init__(self, extra=None):
        self.extra = extra or {}


class _FakeStorageProvider:
    def __init__(self, extra=None):
        self.metadata = _FakeMetadata(extra)

    async def get_conversation_metadata(self):
        return self.metadata

    async def set_conversation_metadata(self, **kwds):
        self.metadata.extra.update({k: v for k, v in kwds.items() if v is not None})


class _FakeConversation:
    def __init__(self, extra=None):
        self.storage_provider = _FakeStorageProvider(extra)


def _make_memory(tmp_path, extra=None) -> Memory:
    config = MomexConfig(
        llm=LLMConfig(provider="openai", model="gpt-4o", api_key="k"),
        storage=StorageConfig(path=str(tmp_path)),
    )
    memory = Memory(collection="test:supersession", config=config)
    memory._conversation = _FakeConversation(extra)  # type: ignore[assignment]
    memory._initialized = True
    return memory


@pytest.mark.asyncio
async def test_ledger_persists_through_metadata(tmp_path):
    memory = _make_memory(tmp_path)
    await memory._append_supersessions([_record(1, text="likes sushi")])

    # Drop the cache and reload from the fake metadata store.
    memory._supersession_ledger = None
    (record,) = await memory.history()

    assert record.ordinal == 1
    assert record.text == "likes sushi"


@pytest.mark.asyncio
async def test_append_is_idempotent_for_hidden_ordinals(tmp_path):
    memory = _make_memory(tmp_path)

    assert len(await memory._append_supersessions([_record(1)])) == 1
    assert await memory._append_supersessions([_record(1)]) == []
    assert await memory._hidden_ordinals() == {1}


@pytest.mark.asyncio
async def test_restore_makes_a_memory_visible_again(tmp_path):
    """The whole point: a bad contradiction judgment is recoverable."""
    memory = _make_memory(tmp_path)
    await memory._append_supersessions([_record(1), _record(2)])

    assert await memory.restore(1) == 1
    assert await memory._hidden_ordinals() == {2}


@pytest.mark.asyncio
async def test_restore_accepts_a_list_and_ignores_unknown_ordinals(tmp_path):
    memory = _make_memory(tmp_path)
    await memory._append_supersessions([_record(1)])

    assert await memory.restore([1, 42]) == 1
    assert await memory.restore([1]) == 0  # already restored
    assert await memory._hidden_ordinals() == set()


@pytest.mark.asyncio
async def test_restored_entries_stay_in_the_ledger(tmp_path):
    """Append-only: restoring adds a timestamp, it does not erase history."""
    memory = _make_memory(tmp_path)
    await memory._append_supersessions([_record(1)])
    await memory.restore(1)

    assert await memory.history() == []
    (record,) = await memory.history(include_restored=True)
    assert record.ordinal == 1
    assert record.restored_at


@pytest.mark.asyncio
async def test_restored_ordinal_can_be_superseded_again(tmp_path):
    memory = _make_memory(tmp_path)
    await memory._append_supersessions([_record(1)])
    await memory.restore(1)

    assert len(await memory._append_supersessions([_record(1)])) == 1
    assert await memory._hidden_ordinals() == {1}
    assert len(await memory.history(include_restored=True)) == 2


# --- migration from the legacy tombstone set -------------------------------


@pytest.mark.asyncio
async def test_legacy_tombstones_are_migrated(tmp_path):
    """Collections written before the ledger keep their deletions."""
    memory = _make_memory(
        tmp_path, extra={"momex_deleted_semrefs": json.dumps([1, [4, 6]])}
    )

    records = await memory.history()

    assert [r.ordinal for r in records] == [1, 4, 5, 6]
    assert all(r.reason == "legacy" for r in records)
    assert all(r.superseded_by == [] for r in records)
    assert await memory._hidden_ordinals() == {1, 4, 5, 6}


@pytest.mark.asyncio
async def test_migrated_tombstones_are_restorable(tmp_path):
    memory = _make_memory(tmp_path, extra={"momex_deleted_semrefs": json.dumps([7])})

    assert await memory.restore(7) == 1
    assert await memory._hidden_ordinals() == set()


@pytest.mark.asyncio
async def test_ledger_wins_over_legacy_tombstones(tmp_path):
    """Once a ledger exists it is authoritative; no re-migration."""
    memory = _make_memory(
        tmp_path,
        extra={
            "momex_deleted_semrefs": json.dumps([99]),
            SUPERSESSION_METADATA_KEY: json.dumps(_encode_ledger([_record(1)])),
        },
    )

    assert [r.ordinal for r in await memory.history()] == [1]


@pytest.mark.asyncio
async def test_corrupt_ledger_does_not_raise(tmp_path):
    memory = _make_memory(tmp_path, extra={SUPERSESSION_METADATA_KEY: "{not json"})

    assert await memory.history() == []
