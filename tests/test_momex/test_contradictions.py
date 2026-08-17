"""Tests for the contradiction judgment itself.

detect() is handed its collaborators -- a candidate finder, an LLM factory, a
ledger appender -- so the judgment can be tested without a conversation behind
it. Candidate *finding* is tested in test_relation_candidates.py.

The central claim here is that nothing escapes: add() has already committed the
new messages by the time this runs, so every failure path has to degrade to
"nothing contradicted" rather than raise.
"""

import logging

import pytest

from momex.contradictions import detect
from momex.results import SearchItem, SupersededRecord


class _FakeSemanticRef:
    def __init__(self, ordinal: int):
        self.semantic_ref_ordinal = ordinal


class _FakeResponse:
    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    def __init__(self, content: str):
        self.content = content
        self.calls = 0
        self.prompt: str | None = None

    async def complete(self, prompt: str, **kwargs):
        self.calls += 1
        self.prompt = prompt
        return _FakeResponse(self.content)


class _BrokenLLM:
    async def complete(self, prompt, **kwargs):
        raise RuntimeError("llm is down")


class _Ledger:
    """Stands in for SupersessionLedger.append, including its idempotence."""

    def __init__(self, hidden=()):
        self.records: list[SupersededRecord] = []
        self._hidden = set(hidden)

    async def append(self, records):
        added = [r for r in records if r.ordinal not in self._hidden]
        for record in added:
            self._hidden.add(record.ordinal)
        self.records.extend(added)
        return added


def _knowledge(text, ordinal, memory_id=None):
    """An action: the one knowledge type that is always propositional."""
    return SearchItem(
        type="action",
        text=text,
        score=10.0,
        raw=_FakeSemanticRef(ordinal),
        memory_id=memory_id,
    )


def _candidates(*items):
    async def find():
        return list(items)

    return find


async def _run(*, candidates, llm, ledger=None, superseded_by=None):
    ledger = ledger or _Ledger()
    superseded = await detect(
        "I don't like sushi",
        collection="test:contradictions",
        find_candidates=candidates,
        create_llm=lambda: llm,
        append=ledger.append,
        superseded_by=superseded_by,
    )
    return superseded, ledger


@pytest.mark.asyncio
async def test_supersedes_contradicting_knowledge():
    llm = _FakeLLM("0")
    superseded, ledger = await _run(
        candidates=_candidates(_knowledge("likes sushi", 1, memory_id="deadbeef")),
        llm=llm,
        superseded_by=[9],
    )

    assert [r.ordinal for r in superseded] == [1]
    (record,) = ledger.records
    assert record.reason == "contradiction"
    assert record.text == "likes sushi"
    assert record.query == "I don't like sushi"
    assert record.superseded_by == [9]
    assert record.memory_id == "deadbeef"


@pytest.mark.asyncio
async def test_no_contradiction_removes_nothing():
    superseded, ledger = await _run(
        candidates=_candidates(_knowledge("likes sushi", 1)), llm=_FakeLLM("none")
    )

    assert superseded == []
    assert ledger.records == []


@pytest.mark.asyncio
async def test_no_candidates_skips_the_llm_call():
    """Nothing to compare against means nothing to pay a model for."""
    llm = _FakeLLM("0")
    superseded, _ = await _run(candidates=_candidates(), llm=llm)

    assert superseded == []
    assert llm.calls == 0


@pytest.mark.asyncio
async def test_already_superseded_ids_are_not_recounted():
    ledger = _Ledger(hidden={1})
    superseded, _ = await _run(
        candidates=_candidates(_knowledge("likes sushi", 1)),
        llm=_FakeLLM("0"),
        ledger=ledger,
    )

    assert superseded == []


@pytest.mark.asyncio
async def test_unparseable_reply_retires_nothing():
    """A model that ignores the format must not cause arbitrary deletions."""
    superseded, ledger = await _run(
        candidates=_candidates(_knowledge("likes sushi", 1)),
        llm=_FakeLLM("I think memory 1 might be related?"),
    )

    assert superseded == []
    assert ledger.records == []


@pytest.mark.asyncio
async def test_out_of_range_indices_are_ignored():
    superseded, _ = await _run(
        candidates=_candidates(_knowledge("a", 1), _knowledge("b", 2)),
        llm=_FakeLLM("1, 47"),
    )

    assert [r.ordinal for r in superseded] == [2]


@pytest.mark.asyncio
async def test_prompt_lists_candidates_by_dense_index():
    llm = _FakeLLM("none")
    await _run(candidates=_candidates(_knowledge("a", 7), _knowledge("b", 42)), llm=llm)

    assert "0: [action] a" in (llm.prompt or "")
    assert "1: [action] b" in (llm.prompt or "")


@pytest.mark.asyncio
async def test_llm_failure_is_logged_and_does_not_block_add(caplog):
    """A broken LLM must degrade loudly, not silently."""
    with caplog.at_level(logging.WARNING, logger="momex.contradictions"):
        superseded, ledger = await _run(
            candidates=_candidates(_knowledge("likes sushi", 1)), llm=_BrokenLLM()
        )

    assert superseded == []
    assert ledger.records == []
    assert "Contradiction detection failed" in caplog.text
    assert "llm is down" in caplog.text


@pytest.mark.asyncio
async def test_lookup_failure_is_logged_and_does_not_block_add(caplog):
    """The candidate lookup runs after the write, so it must not escape."""

    async def broken_candidates():
        raise RuntimeError("property index unavailable")

    with caplog.at_level(logging.WARNING, logger="momex.contradictions"):
        superseded, ledger = await _run(candidates=broken_candidates, llm=_FakeLLM("0"))

    assert superseded == []
    assert ledger.records == []
    assert "Contradiction detection lookup failed" in caplog.text
    assert "property index unavailable" in caplog.text
