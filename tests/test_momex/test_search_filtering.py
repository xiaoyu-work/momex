"""Tests for Momex search result filtering (expiry windows, tombstones).

These exercise Memory._search_structured / Memory.search with fake collections
so they run offline, without an LLM or embedding API key.
"""

from datetime import datetime, timedelta, timezone
import tempfile

import pytest

import typechat

from momex import Memory, MomexConfig, StorageConfig
from typeagent.knowpro.interfaces_core import (
    ScoredMessageOrdinal,
    ScoredSemanticRefOrdinal,
    Topic,
)
from typeagent.knowpro.interfaces_search import SemanticRefSearchResult
from typeagent.knowpro.search import ConversationSearchResult


def _day_offset(days: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=days)).strftime("%Y-%m-%d")


PAST = _day_offset(-10)
FUTURE = _day_offset(+10)


class _FakeCollection:
    def __init__(self, items):
        self._items = list(items)

    async def get_multiple(self, ordinals):
        return [self._items[o] for o in ordinals]

    async def get_item(self, ordinal):
        return self._items[ordinal]

    async def size(self):
        return len(self._items)


class _FakeMessage:
    def __init__(self, text, tags=None):
        self.text_chunks = [text]
        self.tags = list(tags or [])
        self.timestamp = "2026-01-01T00:00:00Z"


class _FakeTextLocation:
    def __init__(self, message_ordinal):
        self.message_ordinal = message_ordinal


class _FakeTextRange:
    def __init__(self, message_ordinal):
        self.start = _FakeTextLocation(message_ordinal)


class _FakeSemanticRef:
    def __init__(self, knowledge, message_ordinal):
        self.knowledge = knowledge
        self.range = _FakeTextRange(message_ordinal)


class _FakeConversation:
    """Minimal stand-in for ConversationBase for the structured search path."""

    def __init__(self, messages, semantic_refs):
        self.messages = _FakeCollection(messages)
        self.semantic_refs = _FakeCollection(semantic_refs)
        self._query_translator = object()  # non-None: skips model creation
        self.secondary_indexes = None  # embedding path yields no results


def _make_memory(conversation) -> Memory:
    config = MomexConfig(storage=StorageConfig(path=tempfile.gettempdir()))
    memory = Memory(collection="test:expiry", config=config)
    memory._conversation = conversation  # type: ignore[assignment]
    memory._initialized = True
    memory._deleted_semref_ids = set()  # avoid touching real metadata storage
    return memory


def _stub_language_search(monkeypatch, knowledge_matches, message_matches=()):
    from typeagent.knowpro import searchlang

    result = ConversationSearchResult(
        message_matches=list(message_matches),
        knowledge_matches=knowledge_matches,
    )

    async def fake_search(*args, **kwargs):
        return typechat.Success([result])

    monkeypatch.setattr(searchlang, "search_conversation_with_language", fake_search)


@pytest.mark.asyncio
async def test_expired_knowledge_is_filtered_out(monkeypatch):
    """Entities/topics extracted from an expired message must not be returned."""
    messages = [
        _FakeMessage("Netflix renews May 1", tags=[f"valid_to:{PAST}"]),
        _FakeMessage("I like Python", tags=[f"valid_to:{FUTURE}"]),
    ]
    semantic_refs = [
        _FakeSemanticRef(Topic(text="netflix subscription"), 0),  # expired source
        _FakeSemanticRef(Topic(text="python"), 1),  # still valid
    ]
    memory = _make_memory(_FakeConversation(messages, semantic_refs))

    _stub_language_search(
        monkeypatch,
        {
            "topic": SemanticRefSearchResult(
                term_matches=set(),
                semantic_ref_matches=[
                    ScoredSemanticRefOrdinal(semantic_ref_ordinal=0, score=9.0),
                    ScoredSemanticRefOrdinal(semantic_ref_ordinal=1, score=8.0),
                ],
            )
        },
    )

    results = await memory.search("what do I like", limit=10)

    texts = [item.text for item in results]
    assert "netflix subscription" not in texts
    assert "python" in texts


@pytest.mark.asyncio
async def test_expired_knowledge_included_when_requested(monkeypatch):
    """include_expired=True brings expired knowledge back."""
    messages = [_FakeMessage("Netflix renews May 1", tags=[f"valid_to:{PAST}"])]
    semantic_refs = [_FakeSemanticRef(Topic(text="netflix subscription"), 0)]
    memory = _make_memory(_FakeConversation(messages, semantic_refs))

    _stub_language_search(
        monkeypatch,
        {
            "topic": SemanticRefSearchResult(
                term_matches=set(),
                semantic_ref_matches=[
                    ScoredSemanticRefOrdinal(semantic_ref_ordinal=0, score=9.0)
                ],
            )
        },
    )

    results = await memory.search("netflix", limit=10, include_expired=True)

    assert [item.text for item in results] == ["netflix subscription"]


@pytest.mark.asyncio
async def test_knowledge_inherits_time_window_from_source_message(monkeypatch):
    """SearchItem.valid_from/valid_to are populated for knowledge results."""
    messages = [
        _FakeMessage(
            "Netflix renews May 1",
            tags=[f"valid_from:{PAST}", f"valid_to:{FUTURE}"],
        )
    ]
    semantic_refs = [_FakeSemanticRef(Topic(text="netflix subscription"), 0)]
    memory = _make_memory(_FakeConversation(messages, semantic_refs))

    _stub_language_search(
        monkeypatch,
        {
            "topic": SemanticRefSearchResult(
                term_matches=set(),
                semantic_ref_matches=[
                    ScoredSemanticRefOrdinal(semantic_ref_ordinal=0, score=9.0)
                ],
            )
        },
    )

    (item,) = await memory.search("netflix", limit=10)

    assert item.type == "topic"
    assert item.valid_from == PAST
    assert item.valid_to == FUTURE
    assert item.timestamp == "2026-01-01T00:00:00Z"


@pytest.mark.asyncio
async def test_expired_messages_are_filtered_out(monkeypatch):
    """The message branch keeps filtering expired results (regression guard)."""
    messages = [
        _FakeMessage("Netflix renews May 1", tags=[f"valid_to:{PAST}"]),
        _FakeMessage("I like Python", tags=[]),
    ]
    memory = _make_memory(_FakeConversation(messages, []))

    _stub_language_search(
        monkeypatch,
        {},
        message_matches=[
            ScoredMessageOrdinal(message_ordinal=0, score=9.0),
            ScoredMessageOrdinal(message_ordinal=1, score=8.0),
        ],
    )

    results = await memory.search("anything", limit=10)

    texts = [item.text for item in results]
    assert "Netflix renews May 1" not in texts
    assert "I like Python" in texts
