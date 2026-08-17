"""Tests for stable memory identity.

The ledger used to name memories by `semantic_ref_ordinal` alone -- a position.
A memory_id is derived from the source message plus what the knowledge asserts,
so an entry written today still names the same memory after a reindex.
"""

import pytest

from momex import LLMConfig, Memory, MomexConfig, StorageConfig
from momex.identity import canonical_knowledge, memory_id, new_source_id
from typeagent.knowpro.interfaces import Topic
from typeagent.knowpro.knowledge_schema import Action, ConcreteEntity, Facet, VerbTense
from typeagent.knowpro.universal_message import ConversationMessage


def _entity(name="Xiaoyu", type_=None, facets=None, aliases=None):
    return ConcreteEntity(
        name=name,
        type=list(type_ or ["person"]),
        facets=facets,
        aliases=aliases,
    )


def _action(subject="user", verbs=None, obj="sushi", tense: VerbTense = "present"):
    return Action(
        verbs=list(verbs or ["like"]),
        verb_tense=tense,
        subject_entity_name=subject,
        object_entity_name=obj,
    )


class TestNewSourceId:
    def test_ids_are_unique(self):
        assert len({new_source_id() for _ in range(100)}) == 100


class TestMemoryId:
    def test_same_knowledge_same_message_is_one_memory(self):
        src = "abc"
        assert memory_id(src, _action()) == memory_id(src, _action())

    def test_same_knowledge_different_message_is_a_different_memory(self):
        """The bug delete() hit: identical text from two messages is two refs."""
        assert memory_id("abc", _action()) != memory_id("def", _action())

    def test_object_change_changes_the_id(self):
        assert memory_id("abc", _action(obj="sushi")) != memory_id(
            "abc", _action(obj="ramen")
        )

    def test_tense_change_changes_the_id(self):
        """ "liked" and "likes" are different claims."""
        assert memory_id("abc", _action(tense="past")) != memory_id(
            "abc", _action(tense="present")
        )

    def test_entity_and_action_do_not_collide(self):
        assert memory_id("abc", _entity()) != memory_id("abc", _action())

    def test_facet_change_changes_the_id(self):
        with_ms = _entity(facets=[Facet(name="employer", value="Microsoft")])
        with_goog = _entity(facets=[Facet(name="employer", value="Google")])
        assert memory_id("abc", with_ms) != memory_id("abc", with_goog)

    def test_aliases_do_not_change_the_id(self):
        """Learning a nickname does not make it a different memory."""
        plain = _entity()
        aliased = _entity(aliases=["XZ", "Xiao"])
        assert memory_id("abc", plain) == memory_id("abc", aliased)

    def test_type_order_does_not_change_the_id(self):
        a = _entity(type_=["person", "engineer"])
        b = _entity(type_=["engineer", "person"])
        assert memory_id("abc", a) == memory_id("abc", b)

    def test_missing_source_id_yields_none(self):
        """Collections written before source_id existed still work."""
        assert memory_id(None, _action()) is None
        assert memory_id("", _action()) is None

    def test_unknown_knowledge_type_still_gets_an_id(self):
        assert memory_id("abc", object()) is not None

    def test_topics_are_identified_by_text(self):
        assert memory_id("abc", Topic(text="food")) == memory_id(
            "abc", Topic(text="food")
        )
        assert memory_id("abc", Topic(text="food")) != memory_id(
            "abc", Topic(text="travel")
        )

    def test_id_is_short_and_hex(self):
        value = memory_id("abc", _action())
        assert value is not None and len(value) == 16
        assert all(c in "0123456789abcdef" for c in value)


class TestCanonicalKnowledge:
    def test_excludes_rendered_text(self):
        """Reformatting how a memory displays must not rename it."""
        canonical = canonical_knowledge(_entity())
        assert "type: person" not in canonical  # the rendered form


class _FakeAddResult:
    def __init__(self, messages_added, semrefs_added=0):
        self.messages_added = messages_added
        self.semrefs_added = semrefs_added


class _FakeIndexSettings:
    auto_extract_knowledge = True


class _FakeSettings:
    def __init__(self):
        self.semantic_ref_index_settings = _FakeIndexSettings()


class _CapturingConversation:
    """Captures what add() hands to the indexer, without touching a backend."""

    def __init__(self):
        self.settings = _FakeSettings()
        self.added_messages: list = []

    async def add_messages_with_indexing(self, messages):
        self.added_messages.extend(messages)
        return _FakeAddResult(len(messages))


def _memory(tmp_path, conversation):
    config = MomexConfig(
        llm=LLMConfig(provider="openai", model="gpt-4o", api_key="sk-dummy"),
        storage=StorageConfig(path=str(tmp_path)),
    )
    memory = Memory(collection="test:source-ids", config=config)
    memory._conversation = conversation  # type: ignore[assignment]
    memory._initialized = True
    memory._ledger._records = []
    return memory


@pytest.mark.asyncio
async def test_add_stamps_a_source_id_on_every_message(tmp_path):
    """Without this, nothing downstream can have a stable identity."""
    conversation = _CapturingConversation()
    memory = _memory(tmp_path, conversation)

    await memory.add(
        [
            {"role": "user", "content": "I like sushi"},
            {"role": "assistant", "content": "Noted"},
        ],
        infer=False,
    )

    source_ids = [m.source_id for m in conversation.added_messages]

    assert len(source_ids) == 2
    assert all(source_ids)
    assert len(set(source_ids)) == 2


@pytest.mark.asyncio
async def test_source_id_is_persisted(tmp_path):
    """The id is only stable if it survives serialization."""
    conversation = _CapturingConversation()
    memory = _memory(tmp_path, conversation)

    await memory.add("I like sushi", infer=False)

    (message,) = conversation.added_messages
    restored = ConversationMessage.deserialize(message.serialize())

    assert restored.source_id == message.source_id
    assert restored.source_id
