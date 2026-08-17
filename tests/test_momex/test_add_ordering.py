"""Regression tests for add() write ordering, date validation, and the
settings toggle used by infer=False.

Contradicted memories are superseded rather than deleted, so add() reports
what it retired (AddResult.superseded), not just how many.

All offline: the conversation object is faked, no LLM or embedding key needed.
"""

import asyncio

import pytest

from momex import LLMConfig, Memory, MomexConfig, StorageConfig
from momex.memory import SupersededRecord


class _FakeIndexSettings:
    def __init__(self):
        self.auto_extract_knowledge = True


class _FakeSettings:
    def __init__(self):
        self.semantic_ref_index_settings = _FakeIndexSettings()


class _FakeAddResult:
    def __init__(self, messages_added, semrefs_added):
        self.messages_added = messages_added
        self.semrefs_added = semrefs_added


class _FakeSemanticRefs:
    def __init__(self, count=0):
        self._count = count

    async def size(self):
        return self._count


class _FakeConversation:
    """Records the order of writes and deletions."""

    def __init__(self, *, fail_write=False, semref_count=0):
        self.settings = _FakeSettings()
        self.semantic_refs = _FakeSemanticRefs(semref_count)
        self.fail_write = fail_write
        self.events: list[str] = []
        self.seen_extract_flag: list[bool] = []

    async def add_messages_with_indexing(self, messages):
        self.seen_extract_flag.append(
            self.settings.semantic_ref_index_settings.auto_extract_knowledge
        )
        await asyncio.sleep(0)  # force a real suspension point
        if self.fail_write:
            self.events.append("write-failed")
            raise RuntimeError("indexing blew up")
        self.events.append("write")
        return _FakeAddResult(len(messages), 2)


def _make_memory(conversation) -> Memory:
    config = MomexConfig(
        llm=LLMConfig(provider="openai", model="gpt-4o", api_key="k"),
        storage=StorageConfig(path="/tmp"),
    )
    memory = Memory(collection="test:ordering", config=config)
    memory._conversation = conversation  # type: ignore[assignment]
    memory._initialized = True
    memory._deleted_semref_ids = set()
    memory._supersession_ledger = []

    async def _no_persist_ledger(records):
        memory._supersession_ledger = records

    memory._store_ledger = _no_persist_ledger  # type: ignore[method-assign]
    return memory


def _record_contradictions(memory, conversation, monkeypatch, *, returns=1):
    seen: dict = {}

    async def fake_detect(
        new_content, *, protect_semrefs_from=None, superseded_by=None
    ):
        conversation.events.append("supersede")
        seen["protect_semrefs_from"] = protect_semrefs_from
        seen["superseded_by"] = superseded_by
        return [
            SupersededRecord(
                ordinal=100 + i,
                superseded_by=list(superseded_by or []),
                at="2026-01-01T00:00:00Z",
                reason="contradiction",
            )
            for i in range(returns)
        ]

    monkeypatch.setattr(memory, "_detect_and_remove_contradictions", fake_detect)
    return seen


# --- 1. write-before-delete ------------------------------------------------


@pytest.mark.asyncio
async def test_contradictions_are_removed_after_the_write(monkeypatch):
    """The new content must be durable before anything is retired."""
    conversation = _FakeConversation()
    memory = _make_memory(conversation)
    _record_contradictions(memory, conversation, monkeypatch)

    result = await memory.add("I don't like sushi")

    assert conversation.events == ["write", "supersede"]
    assert result.contradictions_removed == 1
    assert result.superseded is not None and len(result.superseded) == 1


@pytest.mark.asyncio
async def test_failed_write_deletes_nothing(monkeypatch):
    """A failed insert must not leave the old facts tombstoned."""
    conversation = _FakeConversation(fail_write=True)
    memory = _make_memory(conversation)
    _record_contradictions(memory, conversation, monkeypatch)

    with pytest.raises(RuntimeError):
        await memory.add("I don't like sushi")

    assert "supersede" not in conversation.events


@pytest.mark.asyncio
async def test_own_semrefs_are_protected_from_self_contradiction(monkeypatch):
    """Detection must skip the refs this same call just wrote."""
    conversation = _FakeConversation(semref_count=7)
    memory = _make_memory(conversation)
    seen = _record_contradictions(memory, conversation, monkeypatch)

    await memory.add("I don't like sushi")

    assert seen["protect_semrefs_from"] == 7
    # The two refs this write produced are what the old ones were superseded by.
    assert seen["superseded_by"] == [7, 8]


# --- 2. valid_from enforcement + date validation ---------------------------


def test_not_yet_active_window_is_outside():
    assert Memory._is_not_yet_active("2999-01-01") is True
    assert Memory._is_not_yet_active("2000-01-01") is False
    assert Memory._is_not_yet_active(None) is False
    assert Memory._is_outside_window("2999-01-01", None) is True
    assert Memory._is_outside_window(None, "2000-01-01") is True
    assert Memory._is_outside_window("2000-01-01", "2999-01-01") is False


@pytest.mark.parametrize("bad", ["2026-4-1", "04/01/2026", "next tuesday", ""])
@pytest.mark.asyncio
async def test_malformed_dates_are_rejected_at_write_time(bad, monkeypatch):
    """Lexicographic comparison only works for padded ISO dates, so refuse
    anything else instead of silently never expiring."""
    conversation = _FakeConversation()
    memory = _make_memory(conversation)
    _record_contradictions(memory, conversation, monkeypatch)

    with pytest.raises(ValueError):
        await memory.add("something", valid_to=bad)

    assert conversation.events == []


@pytest.mark.asyncio
async def test_valid_dates_are_normalized(monkeypatch):
    conversation = _FakeConversation()
    memory = _make_memory(conversation)
    _record_contradictions(memory, conversation, monkeypatch)

    result = await memory.add("netflix", valid_from="2026-04-01", valid_to="2026-05-02")

    assert result.success


# --- 3. infer=False settings race ------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_infer_false_does_not_disable_other_extraction():
    """Two concurrent add() calls must not clobber each other's flag."""
    conversation = _FakeConversation()
    memory = _make_memory(conversation)

    await asyncio.gather(
        memory.add("a", infer=False, detect_contradictions=False),
        memory.add("b", infer=False, detect_contradictions=False),
    )

    # Both ran with extraction off, and the original value is restored.
    assert conversation.seen_extract_flag == [False, False]
    assert conversation.settings.semantic_ref_index_settings.auto_extract_knowledge


@pytest.mark.asyncio
async def test_infer_false_restores_flag_after_failure():
    conversation = _FakeConversation(fail_write=True)
    memory = _make_memory(conversation)

    with pytest.raises(RuntimeError):
        await memory.add("a", infer=False)

    assert conversation.settings.semantic_ref_index_settings.auto_extract_knowledge
