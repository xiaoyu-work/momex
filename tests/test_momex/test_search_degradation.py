"""Tests that neither half of the hybrid search can take the other down.

search() runs a structured (LLM-translated) query and an embedding query
concurrently. The embedding half was guarded; the structured half was not, so
a rate limit or timeout in query translation raised out of search() and
discarded embedding results that had already succeeded -- even though
search_by_embedding() documents itself as the fallback for exactly that.
"""

import pytest

from momex import LLMConfig, Memory, MomexConfig, StorageConfig
from momex.results import SearchItem


@pytest.fixture
def memory(tmp_path):
    config = MomexConfig(
        llm=LLMConfig(provider="openai", model="gpt-4o", api_key="k"),
        storage=StorageConfig(path=str(tmp_path)),
    )
    mem = Memory(collection="test:degrade", config=config)
    mem._initialized = True
    mem._conversation = object()  # type: ignore[assignment]
    mem._supersession_ledger = []
    return mem


def _structured(text, score=8.0):
    return SearchItem(type="entity", text=text, score=score, raw=object())


def _message(text, score=0.9):
    return SearchItem(type="message", text=text, score=score, raw=object())


def _stub(memory, *, structured=None, embedding=None):
    async def fake_structured(query_text, limit=10, **kwargs):
        if isinstance(structured, Exception):
            raise structured
        return list(structured or [])

    async def fake_embedding(query_text, limit=10, **kwargs):
        if isinstance(embedding, Exception):
            raise embedding
        return list(embedding or [])

    memory._search_structured = fake_structured  # type: ignore[method-assign]
    memory.search_by_embedding = fake_embedding  # type: ignore[method-assign]


@pytest.mark.asyncio
async def test_structured_failure_degrades_to_embedding_results(memory, caplog):
    _stub(
        memory,
        structured=RuntimeError("rate limited"),
        embedding=[_message("I like sushi")],
    )

    results = await memory.search("sushi")

    assert [item.text for item in results] == ["I like sushi"]
    assert "Structured search failed" in caplog.text


@pytest.mark.asyncio
async def test_embedding_failure_degrades_to_structured_results(memory):
    _stub(
        memory,
        structured=[_structured("sushi (type: food)")],
        embedding=RuntimeError("embedding endpoint down"),
    )

    results = await memory.search("sushi")

    assert [item.text for item in results] == ["sushi (type: food)"]


@pytest.mark.asyncio
async def test_both_failing_returns_empty_rather_than_raising(memory):
    _stub(memory, structured=RuntimeError("a"), embedding=RuntimeError("b"))

    assert await memory.search("sushi") == []


@pytest.mark.asyncio
async def test_both_succeeding_still_fuses(memory):
    _stub(
        memory,
        structured=[_structured("sushi")],
        embedding=[_message("I like sushi")],
    )

    results = await memory.search("sushi")

    assert {item.text for item in results} == {"sushi", "I like sushi"}
    assert all(item.fusion_score is not None for item in results)
