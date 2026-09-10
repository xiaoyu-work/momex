"""Current-view rules must survive every retrieval path and a storage reopen."""

from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from momex import LLMConfig, Memory, MomexConfig, StorageConfig, SupersededRecord
from momex.search import expand_with_neighbors, items_for_semrefs

from .test_search_after_reopen import _FakeEmbeddingModel


@pytest_asyncio.fixture
async def memory(tmp_path, monkeypatch):
    config = MomexConfig(
        llm=LLMConfig(model="offline", api_key="dummy"),
        storage=StorageConfig(path=str(tmp_path)),
    )
    monkeypatch.setattr(config, "create_embedding_model", _FakeEmbeddingModel)
    memory = Memory("test:visibility", config)
    await memory.add("I like tea and read novels", infer=False)
    await memory.add("I no longer like tea", infer=False)
    await memory.add("temporary address", infer=False, valid_to="2000-01-01")
    await memory._ledger.append(
        [
            SupersededRecord(
                ordinal=0,
                superseded_by=[2],
                at="2026-01-01T00:00:00Z",
                reason="contradiction",
            )
        ]
    )
    yield memory
    await memory.close()


@pytest.mark.asyncio
async def test_vector_search_hides_superseded_sources_after_reopen(memory):
    await memory.close()
    items = await memory.search_by_embedding("I like tea and read novels", min_score=-1)
    assert {item.ordinal for item in items} == {1}
    history = await memory.search_by_embedding(
        "I like tea and read novels", min_score=-1, include_superseded=True
    )
    assert next(item for item in history if item.ordinal == 0).status == "superseded"


@pytest.mark.asyncio
async def test_hybrid_search_uses_the_same_source_policy(memory, monkeypatch):
    monkeypatch.setattr(
        memory, "_search_structured_guarded", AsyncMock(return_value=[])
    )
    items = await memory.search("I like tea and read novels")
    assert all(item.ordinal != 0 for item in items)
    history = await memory.search("I like tea and read novels", include_superseded=True)
    assert any(item.ordinal == 0 and item.status == "superseded" for item in history)


@pytest.mark.asyncio
async def test_other_knowledge_in_the_same_source_remains_current(memory):
    conversation = memory._conversation_required()
    view = await memory._search_view()
    items = await items_for_semrefs(conversation, [0, 1], view=view)
    assert len(items) == 1
    assert items[0].status == "current"


@pytest.mark.asyncio
async def test_neighbors_do_not_resurrect_hidden_or_expired_text(memory):
    center = (await memory.transcript(start=1, limit=1))[0]
    items = await expand_with_neighbors(
        memory._conversation_required(),
        [center],
        radius=1,
        view=await memory._search_view(),
    )
    assert items[0].text == "I no longer like tea"
    historical = await expand_with_neighbors(
        memory._conversation_required(),
        [center],
        radius=1,
        view=await memory._search_view(include_expired=True, include_superseded=True),
    )
    assert "[superseded] I like tea and read novels" in historical[0].text
    assert "[expired] temporary address" in historical[0].text


@pytest.mark.asyncio
async def test_transcript_labels_history_without_discarding_it(memory):
    assert [item.status for item in await memory.transcript()] == [
        "superseded",
        "current",
        "expired",
    ]


@pytest.mark.asyncio
async def test_restore_reopens_the_source_for_current_search(memory):
    assert await memory.restore(0) == 1
    items = await memory.search_by_embedding("I like tea and read novels", min_score=-1)
    assert next(item for item in items if item.ordinal == 0).status == "current"
