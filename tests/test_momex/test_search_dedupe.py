"""Tests that structured search does not spend its budget on duplicates.

Extraction produces one semantic ref per message, so an entity mentioned in
hundreds of messages becomes hundreds of refs that all render identically and
all score the same for a term query naming it. Measured on LOCOMO, the top 20
structured results were 20 copies of "Caroline (type: person)" and answered 0%
of questions; search() only escaped that because rank fusion happens to
deduplicate by text afterwards.
"""

import pytest

from momex.search import search_structured
from typeagent.knowpro.interfaces import Topic
from typeagent.knowpro.knowledge_schema import ConcreteEntity


def _topic(text: str) -> Topic:
    return Topic(text=text)


def _entity(name: str) -> ConcreteEntity:
    return ConcreteEntity(name=name, type=[], facets=None)


class _SemanticRef:
    def __init__(self, ordinal, knowledge):
        self.semantic_ref_ordinal = ordinal
        self.knowledge = knowledge
        self.range = None


class _Collection:
    def __init__(self, items):
        self._items = items

    async def get_multiple(self, ordinals):
        return [self._items[o] for o in ordinals]

    async def get_item(self, ordinal):
        return self._items[ordinal]


class _Scored:
    def __init__(self, ordinal, score):
        self.semantic_ref_ordinal = ordinal
        self.score = score


class _Matches:
    def __init__(self, scored):
        self.semantic_ref_matches = scored


class _SearchResult:
    def __init__(self, scored):
        self.knowledge_matches = {"entity": _Matches(scored)}
        self.message_matches = []


class _Conversation:
    def __init__(self, semrefs):
        self.semantic_refs = _Collection(semrefs)
        self.messages = _Collection({})
        self._query_translator = object()
        self.secondary_indexes = None


@pytest.fixture
def duplicated(monkeypatch):
    """A collection where one entity was extracted from many messages."""
    import typechat

    from typeagent.knowpro import searchlang

    semrefs = {i: _SemanticRef(i, _topic("Caroline")) for i in range(25)}
    semrefs[25] = _SemanticRef(25, _topic("poetry reading"))
    semrefs[26] = _SemanticRef(26, _topic("conference"))

    # The duplicates all score the same, as they do in practice; the distinct
    # ones score lower, so only a budget wasted on copies can hide them.
    scored = [_Scored(i, 1.0) for i in range(25)] + [
        _Scored(25, 0.9),
        _Scored(26, 0.9),
    ]

    async def fake_search(*args, **kwargs):
        return typechat.Success([_SearchResult(scored)])

    monkeypatch.setattr(searchlang, "search_conversation_with_language", fake_search)
    return _Conversation(semrefs)


@pytest.mark.asyncio
async def test_duplicates_do_not_crowd_out_distinct_results(duplicated):
    items = await search_structured(duplicated, "Caroline", limit=5)

    texts = [i.text for i in items]
    assert texts.count("Caroline") == 1
    assert "poetry reading" in texts
    assert "conference" in texts


@pytest.mark.asyncio
async def test_without_dedupe_the_budget_is_all_copies(duplicated):
    """The behaviour delete() needs, and the reason the default is the other way."""
    items = await search_structured(duplicated, "Caroline", limit=5, dedupe=False)

    assert [i.text for i in items] == ["Caroline"] * 5


@pytest.mark.asyncio
async def test_dedupe_off_keeps_every_matching_ordinal(duplicated):
    """delete() retires ordinals, so it must see refs that read alike."""
    items = await search_structured(duplicated, "Caroline", limit=50, dedupe=False)

    ordinals = {i.raw.semantic_ref_ordinal for i in items}
    assert len(ordinals) == 27


@pytest.mark.asyncio
async def test_dedupe_keeps_the_highest_scoring_occurrence(duplicated):
    items = await search_structured(duplicated, "Caroline", limit=50)

    kept = next(i for i in items if i.text == "Caroline")
    assert kept.score == 1.0


@pytest.mark.asyncio
async def test_same_text_different_type_is_not_a_duplicate(monkeypatch):
    """An entity and a topic that read alike are different memories."""
    import typechat

    from typeagent.knowpro import searchlang

    semrefs = {
        0: _SemanticRef(0, _topic("pottery")),
        1: _SemanticRef(1, _entity("pottery")),
    }

    class _Result:
        def __init__(self):
            self.knowledge_matches = {"k": _Matches([_Scored(0, 1.0), _Scored(1, 0.9)])}
            self.message_matches = []

    async def fake_search(*args, **kwargs):
        return typechat.Success([_Result()])

    monkeypatch.setattr(searchlang, "search_conversation_with_language", fake_search)

    items = await search_structured(_Conversation(semrefs), "pottery", limit=10)

    assert sorted((i.type, i.text) for i in items) == [
        ("entity", "pottery"),
        ("topic", "pottery"),
    ]
