"""Tests that retrieval depth is not pinned to presentation depth.

search() used to ask each path for exactly `limit` results and then fuse them
down to `limit`. That leaves reciprocal rank fusion choosing `limit` winners
from at most `2 * limit` candidates, and it means an item sitting just past the
cut in *both* lists can never appear -- however strongly the two paths agree on
it. Agreement between independent retrievers is the entire signal RRF exists to
reward, so pinning the fetch to the budget throws away the reason to fuse.

Measured on LOCOMO at a fixed presentation budget of 20, widening the fetch and
trimming afterwards moved evidence recall from 64.0% to 67.7% on one
conversation and 67.0% to 71.7% on another, with the gap widening as the budget
grew.

These tests pin the behaviour rather than the constant: what matters is that
both paths are asked for more than the caller will see, and that the extra
depth can actually change the answer.
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
    mem = Memory(collection="test:depth", config=config)
    mem._initialized = True
    mem._conversation = object()  # type: ignore[assignment]
    mem._ledger._records = []
    return mem


def _item(kind: str, text: str, score: float) -> SearchItem:
    return SearchItem(type=kind, text=text, score=score, raw=object())


def _stub(memory, structured: list[SearchItem], embedding: list[SearchItem]):
    """Stub both paths so they honour `limit` the way a real retriever does.

    Truncating inside the stub is the whole point: a path asked for `limit`
    genuinely cannot return the item ranked `limit + 1`.
    """
    asked: dict[str, int] = {}

    async def fake_structured(query_text, limit=10, **kwargs):
        asked["structured"] = limit
        return structured[:limit]

    async def fake_embedding(query_text, limit=10, **kwargs):
        asked["embedding"] = limit
        return embedding[:limit]

    memory._search_structured = fake_structured  # type: ignore[method-assign]
    memory.search_by_embedding = fake_embedding  # type: ignore[method-assign]
    return asked


@pytest.mark.asyncio
async def test_both_paths_are_asked_for_more_than_the_caller_sees(memory):
    asked = _stub(memory, [], [])

    await memory.search("anything", limit=10)

    assert asked["structured"] > 10
    assert asked["embedding"] > 10


@pytest.mark.asyncio
async def test_fetch_depth_scales_with_the_requested_limit(memory):
    """A fixed extra margin would vanish at large budgets."""
    asked_small = _stub(memory, [], [])
    await memory.search("anything", limit=5)
    small = asked_small["embedding"]

    asked_large = _stub(memory, [], [])
    await memory.search("anything", limit=50)

    assert asked_large["embedding"] > small


@pytest.mark.asyncio
async def test_an_item_both_paths_agree_on_survives_past_the_budget(memory):
    """The regression, stated as behaviour.

    "shared" is ranked below the budget in both lists, so the old code never
    fetched it. Both paths do agree on it, which under RRF outranks anything
    only one path found.
    """
    limit = 3
    filler_structured = [_item("entity", f"s{i}", 9.0 - i) for i in range(limit)]
    filler_embedding = [_item("message", f"e{i}", 0.9 - i / 100) for i in range(limit)]
    shared_structured = _item("message", "shared", 1.0)
    shared_embedding = _item("message", "shared", 0.5)

    _stub(
        memory,
        structured=filler_structured + [shared_structured],
        embedding=filler_embedding + [shared_embedding],
    )

    results = await memory.search("q", limit=limit)

    assert "shared" in [item.text for item in results]
    assert results[0].text == "shared", "an item found twice should lead"


@pytest.mark.asyncio
async def test_the_caller_still_gets_no_more_than_it_asked_for(memory):
    """Fetching wide must not widen what is returned."""
    _stub(
        memory,
        structured=[_item("entity", f"s{i}", 9.0 - i) for i in range(30)],
        embedding=[_item("message", f"e{i}", 0.9 - i / 100) for i in range(30)],
    )

    results = await memory.search("q", limit=4)

    assert len(results) == 4


@pytest.mark.asyncio
async def test_a_short_path_does_not_hold_back_the_other(memory):
    """One path running dry should not cap the budget at what it returned."""
    _stub(
        memory,
        structured=[_item("entity", "only-one", 9.0)],
        embedding=[_item("message", f"e{i}", 0.9 - i / 100) for i in range(30)],
    )

    results = await memory.search("q", limit=10)

    assert len(results) == 10
