"""Event-time replacements, backfills and snapshots use the same durable timeline."""

import json
from types import SimpleNamespace

import pytest
import pytest_asyncio

import typechat

from momex import LLMConfig, Memory, MomexConfig, search, StorageConfig
from momex.contradictions import temporal_records
from momex.results import SearchItem, SupersededRecord
from momex.timewindow import normalize_as_of
from typeagent.knowpro import convknowledge
from typeagent.knowpro.knowledge_schema import Action, KnowledgeResponse

from .test_search_after_reopen import _FakeEmbeddingModel


@pytest_asyncio.fixture
async def memory(tmp_path, monkeypatch):
    config = MomexConfig(
        llm=LLMConfig(model="offline", api_key="dummy"),
        storage=StorageConfig(path=str(tmp_path)),
    )
    monkeypatch.setattr(config, "create_embedding_model", _FakeEmbeddingModel)

    class Extractor:
        async def extract(self, prompt):
            target = json.loads(prompt.split("\n", 1)[1])["TARGET"]
            return typechat.Success(
                KnowledgeResponse(
                    entities=[],
                    inverse_actions=[],
                    topics=[],
                    actions=[
                        Action(
                            verbs=["live", "in"],
                            verb_tense="present",
                            subject_entity_name=target["speaker"],
                            object_entity_name=target["text"].split()[-1],
                        )
                    ],
                )
            )

    class Judge:
        async def complete(self, prompt, **kwargs):
            data = json.loads(prompt.split("\n", 1)[1])
            pairs = [
                [i, j]
                for i, left in enumerate(data["NEW"])
                for j, right in enumerate(data["EXISTING"])
                if left["text"] != right["text"]
            ]
            return SimpleNamespace(content=json.dumps(pairs))

    monkeypatch.setattr(convknowledge, "KnowledgeExtractor", Extractor)
    monkeypatch.setattr(config, "create_llm", Judge)
    memory = Memory("test:timeline", config)
    yield memory
    await memory.close()


async def sources_at(memory, when):
    items = await memory.search_by_embedding("home", min_score=-1, as_of=when)
    return {item.text for item in items}


@pytest.mark.asyncio
@pytest.mark.parametrize("backfill", [False, True])
async def test_event_time_not_ingestion_order_decides_the_current_fact(
    memory, backfill
):
    entries = [
        ("I live in Seattle", "2024-01-01"),
        ("I live in Portland", "2024-06-01"),
    ]
    for text, timestamp in reversed(entries) if backfill else entries:
        await memory.add(text, timestamp=timestamp)
    await memory.close()
    assert await sources_at(memory, "2024-05-01") == {"I live in Seattle"}
    assert await sources_at(memory, "2024-07-01") == {"I live in Portland"}
    (record,) = await memory.history()
    assert record.effective_at == "2024-06-01T00:00:00Z"
    assert record.text.endswith("Seattle")


@pytest.mark.asyncio
async def test_per_message_timestamps_reconcile_changes_within_one_batch(memory):
    await memory.add(
        [
            {"content": "I live in Portland", "timestamp": "2024-06-01"},
            {"content": "I live in Seattle", "timestamp": "2024-01-01"},
        ]
    )
    assert await sources_at(memory, "2024-05-01") == {"I live in Seattle"}
    assert await sources_at(memory, "2024-07-01") == {"I live in Portland"}


@pytest.mark.asyncio
async def test_temporary_future_replacement_only_hides_during_its_window(memory):
    await memory.add("I live in Seattle", timestamp="2024-01-01")
    await memory.add(
        "I live in Portland",
        timestamp="2024-01-02",
        valid_from="2024-03-01",
        valid_to="2024-04-30",
    )
    assert await sources_at(memory, "2024-02-01") == {"I live in Seattle"}
    assert await sources_at(memory, "2024-03-01") == {"I live in Portland"}
    assert await sources_at(memory, "2024-04-30") == {"I live in Portland"}
    assert await sources_at(memory, "2024-05-01") == {"I live in Seattle"}


@pytest.mark.asyncio
async def test_snapshot_never_leaks_later_events_even_when_history_is_requested(memory):
    await memory.add("I live in Seattle", timestamp="2024-01-01")
    await memory.add("I live in Portland", timestamp="2024-06-01")
    items = await memory.search_by_embedding(
        "home",
        min_score=-1,
        as_of="2024-05-01",
        include_expired=True,
        include_superseded=True,
    )
    assert [item.text for item in items] == ["I live in Seattle"]
    assert len(await memory.transcript(as_of="2024-05-01")) == 1


@pytest.mark.asyncio
async def test_filtered_top_matches_do_not_hide_older_relevant_sources(memory):
    await memory.add("I live in Seattle", infer=False, timestamp="2024-01-01")
    await memory.add(
        [{"content": "I live in Portland"} for _ in range(25)],
        infer=False,
        timestamp="2024-06-01",
    )
    items = await memory.search_by_embedding(
        "I live in Portland", limit=1, min_score=-1, as_of="2024-05-01"
    )
    assert [item.text for item in items] == ["I live in Seattle"]


def test_date_cutoff_includes_the_whole_day_and_offsets_are_normalized():
    assert normalize_as_of("2024-06-01") == "2024-06-01T23:59:59Z"
    assert normalize_as_of("2024-06-01T01:00:00+01:00") == "2024-06-01T00:00:00Z"


@pytest.mark.asyncio
async def test_bad_window_is_rejected_before_a_write(memory):
    with pytest.raises(ValueError, match="valid_from cannot be after"):
        await memory.add("invalid", valid_from="2024-06-01", valid_to="2024-01-01")
    assert await memory.transcript() == []
    with pytest.raises(ValueError):
        await memory.search("home", as_of="yesterday")


def test_restore_preserves_the_historical_supersession_interval():
    record = SupersededRecord(
        ordinal=0,
        superseded_by=[1],
        reason="contradiction",
        at="2024-09-01T00:00:00Z",
        effective_at="2024-06-01T00:00:00Z",
        restored_at="2024-10-01T00:00:00Z",
    )
    assert not record.applies_at("2024-05-01T00:00:00Z")
    assert record.applies_at("2024-07-01T00:00:00Z")
    assert not record.applies_at("2024-10-01T00:00:00Z")


def test_nonoverlapping_facts_and_self_pairs_cannot_be_retired():
    old = SearchItem(
        type="action",
        text="old",
        score=1,
        raw=SimpleNamespace(semantic_ref_ordinal=0),
        timestamp="2024-01-01T00:00:00Z",
        valid_to="2024-02-01",
    )
    new = SearchItem(
        type="action",
        text="new",
        score=1,
        raw=SimpleNamespace(semantic_ref_ordinal=1),
        timestamp="2024-06-01T00:00:00Z",
    )
    assert temporal_records("[[0,0]]", [new], [old]) == []
    assert temporal_records("[[0,0]]", [new], [new]) == []


@pytest.mark.asyncio
async def test_prefix_search_forwards_the_same_snapshot_options(memory, monkeypatch):
    captured = []

    async def search_one(self, text, **options):
        captured.append(options)
        return []

    await memory.add("I live in Seattle", infer=False, timestamp="2024-01-01")
    monkeypatch.setattr(Memory, "search", search_one)
    await search("test", "home", config=memory.config, as_of="2024-02-01")
    assert captured[0]["as_of"] == "2024-02-01T23:59:59Z"
