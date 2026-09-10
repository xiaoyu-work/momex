"""Exact IDs, idempotency, lossless snapshots and permanent source removal."""

import asyncio
import json

import pytest
import pytest_asyncio

import typechat

from momex import LLMConfig, Memory, MomexConfig, StorageConfig, SupersededRecord
from momex.search import items_for_semrefs
from momex.snapshot import Backup
from typeagent.knowpro import convknowledge
from typeagent.knowpro.knowledge_schema import KnowledgeResponse

from .test_search_after_reopen import _FakeEmbeddingModel


@pytest_asyncio.fixture
async def memory(tmp_path, monkeypatch):
    config = MomexConfig(
        llm=LLMConfig(model="offline", api_key="dummy"),
        storage=StorageConfig(path=str(tmp_path / "data")),
    )
    monkeypatch.setattr(config, "create_embedding_model", _FakeEmbeddingModel)

    class Extractor:
        async def extract(self, prompt):
            target = json.loads(prompt.split("\n", 1)[1])["TARGET"]
            return typechat.Success(
                KnowledgeResponse(
                    entities=[],
                    actions=[],
                    inverse_actions=[],
                    topics=[target["text"]],
                )
            )

    monkeypatch.setattr(convknowledge, "KnowledgeExtractor", Extractor)
    memory = Memory("test:backup", config)
    yield memory
    await memory.close()


@pytest.mark.asyncio
async def test_source_retries_and_batch_duplicates_do_not_add_or_reextract(
    memory, monkeypatch
):
    first = await memory.add("same content", source_id="external-1")
    assert first.source_ids == ["external-1"] and first.memory_ids
    await memory.close()

    def no_extractor():
        raise AssertionError("a retry must not re-extract")

    monkeypatch.setattr(convknowledge, "KnowledgeExtractor", no_extractor)
    duplicate = await memory.add("same content", source_id="external-1")
    assert duplicate.messages_added == 0
    assert duplicate.source_ids == []
    assert duplicate.skipped_source_ids == ["external-1"]
    batch = await memory.add(
        [{"content": "batch", "source_id": "external-2"}] * 2, infer=False
    )
    assert batch.messages_added == 1 and batch.skipped_source_ids == ["external-2"]
    assert len(await memory.transcript()) == 2


@pytest.mark.asyncio
async def test_conflicting_id_rolls_back_the_entire_batch(memory):
    await memory.add("original", source_id="existing", infer=False)
    with pytest.raises(ValueError, match="different content"):
        await memory.add(
            [
                {"content": "must roll back", "source_id": "new"},
                {"content": "changed", "source_id": "existing"},
            ],
            infer=False,
        )
    assert await memory.get_source("new") is None
    assert (await memory.get_source("existing")).text == "original"
    result = await memory.add("must roll back", source_id="new", infer=False)
    assert result.messages_added == 1


@pytest.mark.asyncio
async def test_exact_knowledge_ids_are_reversible_without_semantic_search(memory):
    result = await memory.add("I prefer tea", source_id="preference")
    memory_id = result.memory_ids[-1]
    item = await memory.get(memory_id)
    assert item is not None and item.source_id == "preference"
    assert await memory.delete_by_id(memory_id) == 1
    assert await memory.delete_by_id(memory_id) == 0
    assert (await memory.get(memory_id)).status == "superseded"
    assert await memory.restore_by_id(memory_id) == 1
    assert (await memory.get(memory_id)).status == "current"
    assert await memory.get("unknown") is None


@pytest.mark.asyncio
async def test_backup_round_trip_keeps_ids_history_time_attribution_and_vectors(
    memory, tmp_path
):
    result = await memory.add(
        [{"content": "a preference", "speaker": "Alice", "session_id": "chat"}],
        source_id="source",
        timestamp="2024-01-01",
        valid_to="2024-12-31",
    )
    await memory.delete_by_id(result.memory_ids[-1])
    await memory.add(
        [{"role": "assistant", "content": "unconfirmed"}],
        source_id="assistant",
        infer=False,
        timestamp="2024-01-02",
    )
    before = await memory.transcript()
    history = await memory.history(include_restored=True)
    archive = tmp_path / "full.json"
    await memory.backup(str(archive))
    data = Backup.model_validate_json(archive.read_text(encoding="utf-8"))
    assert data.tables["MessageTextIndex"]
    await memory.clear()
    await memory.import_backup(str(archive))
    after = await memory.transcript()
    assert [
        (item.source_id, item.text, item.timestamp, item.status) for item in after
    ] == [(item.source_id, item.text, item.timestamp, item.status) for item in before]
    assert await memory.history(include_restored=True) == history
    assert await memory.get(result.memory_ids[-1]) is not None
    assert await memory.restore_by_id(result.memory_ids[-1]) == 1
    found = await memory.search_by_embedding(
        "a preference", include_expired=True, min_score=-1
    )
    assert found[0].source_id == "source"
    assert (
        await memory.add("unconfirmed", source_id="new", infer=False)
    ).messages_added == 1


@pytest.mark.asyncio
async def test_restore_refuses_overwrite_and_corrupt_archives_leave_data_intact(
    memory, tmp_path
):
    await memory.add("keep", infer=False, source_id="keep")
    archive = tmp_path / "archive.json"
    await memory.backup(str(archive))
    with pytest.raises(ValueError, match="empty collection"):
        await memory.import_backup(str(archive))
    data = json.loads(archive.read_text(encoding="utf-8"))
    data["tables"]["MessageTextIndex"][0]["embedding"] = [1.0]
    archive.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid embedding"):
        await memory.import_backup(str(archive), replace=True)
    assert (await memory.get_source("keep")).text == "keep"


@pytest.mark.asyncio
async def test_restore_failure_rolls_back_already_written_sql(
    memory, tmp_path, monkeypatch
):
    from momex import snapshot

    await memory.add("keep", infer=False, source_id="keep")
    archive = tmp_path / "archive.json"
    await memory.backup(str(archive))

    async def broken_write(conn, sqlite, data, collection):
        conn.execute("UPDATE Messages SET chunks = '[\"changed\"]'")
        raise RuntimeError("injected restore failure")

    monkeypatch.setattr(snapshot, "write_tables", broken_write)
    with pytest.raises(RuntimeError, match="injected restore failure"):
        await memory.import_backup(str(archive), replace=True)
    assert (await memory.get_source("keep")).text == "keep"


@pytest.mark.asyncio
async def test_forget_removes_all_derived_data_but_keeps_other_ids_and_indexes(
    memory, tmp_path
):
    deleted = await memory.add("PRIVATE_MARKER", source_id="private")
    kept = await memory.add("keep Python", source_id="keep")
    old = await memory.get(kept.memory_ids[-1])
    replacement = await memory.get(deleted.memory_ids[-1])
    assert old is not None and replacement is not None
    await memory._ledger.append(
        [
            SupersededRecord(
                ordinal=old.raw.semantic_ref_ordinal,
                superseded_by=[replacement.raw.semantic_ref_ordinal],
                at="2024-01-01T00:00:00Z",
                reason="contradiction",
                text=old.text,
                query="PRIVATE_MARKER",
                memory_id=old.memory_id,
            )
        ]
    )
    storage = memory._conversation_required().storage_provider
    async with storage:
        await storage.record_chunk_failure(0, 0, "error", "PRIVATE_MARKER")
    assert await memory.forget("private") == 1
    assert await memory.forget("private") == 0
    assert await memory.get_source("private") is None
    assert await memory.get(deleted.memory_ids[-1]) is None
    assert (await memory.get_source("keep")).ordinal == 0
    assert await memory.get(kept.memory_ids[-1]) is not None
    assert await memory.restore_by_id(kept.memory_ids[-1]) == 1
    assert (await memory.search_by_embedding("keep Python"))[0].source_id == "keep"
    archive = tmp_path / "after.json"
    await memory.backup(str(archive))
    assert "private_marker" not in archive.read_text(encoding="utf-8").lower()
    await memory.add("after compaction", source_id="after", infer=False)
    assert [item.ordinal for item in await memory.transcript()] == [0, 1]


@pytest.mark.asyncio
async def test_clear_releases_source_ids_for_reingestion(memory, tmp_path):
    await memory.add("same", source_id="id", infer=False)
    await memory.clear()
    assert (await memory.add("same", source_id="id", infer=False)).messages_added == 1
    await memory.backup(str(tmp_path / "after-clear.json"))


@pytest.mark.asyncio
async def test_backup_can_restore_into_another_collection(memory, tmp_path):
    result = await memory.add("portable", source_id="portable")
    archive = tmp_path / "portable.json"
    await memory.backup(str(archive))
    async with Memory("test:copy", memory.config) as copy:
        await copy.import_backup(str(archive))
        item = await copy.get(result.memory_ids[-1])
        source = await copy.get_source("portable")
        assert item is not None and item.collection == "test:copy"
        assert source is not None and source.source_id == "portable"


@pytest.mark.asyncio
async def test_same_instance_concurrent_retries_create_one_source(memory):
    results = await asyncio.gather(
        *[memory.add("one", source_id="one", infer=False) for _ in range(3)]
    )
    assert sum(result.messages_added for result in results) == 1
    assert len(await memory.transcript()) == 1


@pytest.mark.asyncio
async def test_separate_writers_claim_a_source_atomically(memory):
    # Initial WAL setup is separate from concurrent writes to an existing store.
    async with Memory("test:parallel", memory.config):
        pass

    def write():
        async def run():
            async with Memory("test:parallel", memory.config) as writer:
                return await writer.add("one source", source_id="shared", infer=False)

        return asyncio.run(run())

    results = await asyncio.gather(asyncio.to_thread(write), asyncio.to_thread(write))
    assert sorted(result.messages_added for result in results) == [0, 1]


@pytest.mark.asyncio
async def test_forget_preserves_refs_ending_at_the_deleted_turn_boundary(memory):
    await memory.add("keep", source_id="keep", infer=False)
    await memory.add("remove", source_id="remove", infer=False)
    conversation = memory._conversation_required()
    (item,) = await items_for_semrefs(conversation, [0])
    db = conversation.storage_provider.db
    span = json.loads(
        db.execute("SELECT range_json FROM SemanticRefs WHERE semref_id=0").fetchone()[
            0
        ]
    )
    span["end"] = {"messageOrdinal": 1, "chunkOrdinal": 0}
    db.execute(
        "UPDATE SemanticRefs SET range_json=? WHERE semref_id=0", (json.dumps(span),)
    )
    assert await memory.forget("remove") == 1
    remaining = await memory.get(item.memory_id)
    assert remaining is not None
    assert remaining.raw.range.end.message_ordinal == 1
