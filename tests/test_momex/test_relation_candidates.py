"""Tests for how contradiction candidates are found.

Candidates used to come from recompiling the raw new text into a
natural-language query -- an LLM round trip that returned whatever was
topically similar. They now come from the knowledge the write already
produced, probed against the property index: same subject, same relation.
"""

import pytest

from momex.contradictions import find_candidates, related_ordinals, relation_probes
from typeagent.knowpro.interfaces import Topic
from typeagent.knowpro.knowledge_schema import Action, ConcreteEntity, Facet
from typeagent.storage.memory.propindex import PropertyNames


def _action(subject="user", verbs=("like",), obj="sushi"):
    return Action(
        verbs=list(verbs),
        verb_tense="present",
        subject_entity_name=subject,
        object_entity_name=obj,
    )


def _entity(name="Xiaoyu", facets=None):
    return ConcreteEntity(name=name, type=["person"], facets=facets)


class _ScoredOrdinal:
    def __init__(self, ordinal):
        self.semantic_ref_ordinal = ordinal


class _PropertyIndex:
    """Maps (property, value) to ordinals, like the real index does."""

    def __init__(self, mapping):
        self._mapping = mapping

    async def lookup_property(self, name, value):
        found = self._mapping.get((name, value))
        return [_ScoredOrdinal(o) for o in found] if found else None


class _SemanticRef:
    def __init__(self, ordinal, knowledge):
        self.semantic_ref_ordinal = ordinal
        self.knowledge = knowledge
        self.range = None


class _Collection:
    def __init__(self, items_by_ordinal):
        self._items = items_by_ordinal

    async def get_multiple(self, ordinals):
        return [self._items[o] for o in ordinals]

    async def get_item(self, ordinal):
        return self._items[ordinal]


class _Indexes:
    def __init__(self, property_index):
        self.property_to_semantic_ref_index = property_index


class _Conversation:
    def __init__(self, semrefs, property_mapping):
        self.semantic_refs = _Collection(semrefs)
        self.messages = _Collection({})
        self.secondary_indexes: _Indexes | None = _Indexes(
            _PropertyIndex(property_mapping)
        )


class TestRelationProbes:
    def test_action_anchors_on_verb_and_object(self):
        subject, anchors = relation_probes(_action())

        assert subject == (PropertyNames.Subject.value, "user")
        assert set(anchors) == {
            (PropertyNames.Verb.value, "like"),
            (PropertyNames.Object.value, "sushi"),
        }

    def test_multi_word_verbs_are_joined(self):
        _, anchors = relation_probes(_action(verbs=("work", "at"), obj="Microsoft"))
        assert (PropertyNames.Verb.value, "work at") in anchors

    def test_literal_none_is_not_a_subject(self):
        """Action fields default to the string "none" rather than being absent."""
        assert relation_probes(_action(subject="none")) == (None, [])

    def test_action_without_object_still_anchors_on_verb(self):
        subject, anchors = relation_probes(_action(obj="none"))
        assert subject == (PropertyNames.Subject.value, "user")
        assert anchors == [(PropertyNames.Verb.value, "like")]

    def test_entity_anchors_on_facet_names(self):
        entity = _entity(facets=[Facet(name="employer", value="Microsoft")])
        subject, anchors = relation_probes(entity)

        assert subject == (PropertyNames.EntityName.value, "Xiaoyu")
        assert anchors == [(PropertyNames.FacetName.value, "employer")]

    def test_entity_without_facets_asserts_nothing(self):
        assert relation_probes(_entity()) == (None, [])

    def test_topics_are_not_probed(self):
        assert relation_probes(Topic(text="food")) == (None, [])


class TestRelatedOrdinals:
    @pytest.mark.asyncio
    async def test_object_anchor_catches_a_polarity_flip(self):
        """ "dislike sushi" must still find "like sushi" -- the verb changed."""
        conversation = _Conversation(
            {},
            {
                (PropertyNames.Subject.value, "user"): [1, 2],
                (PropertyNames.Object.value, "sushi"): [1],
                (PropertyNames.Verb.value, "dislike"): [],
            },
        )

        found = await related_ordinals(
            conversation, _action(verbs=("dislike",), obj="sushi")
        )
        assert found == {1}

    @pytest.mark.asyncio
    async def test_verb_anchor_catches_a_value_replacement(self):
        """ "work at Google" must still find "work at Microsoft"."""
        conversation = _Conversation(
            {},
            {
                (PropertyNames.Subject.value, "user"): [1, 2],
                (PropertyNames.Verb.value, "work at"): [2],
                (PropertyNames.Object.value, "Google"): [],
            },
        )

        found = await related_ordinals(
            conversation, _action(verbs=("work", "at"), obj="Google")
        )
        assert found == {2}

    @pytest.mark.asyncio
    async def test_a_different_subject_is_not_a_candidate(self):
        """Someone else liking sushi says nothing about what I like."""
        conversation = _Conversation(
            {},
            {
                (PropertyNames.Subject.value, "user"): [1],
                (PropertyNames.Object.value, "sushi"): [1, 5],
                (PropertyNames.Verb.value, "like"): [1, 5],
            },
        )

        found = await related_ordinals(conversation, _action())
        assert found == {1}

    @pytest.mark.asyncio
    async def test_same_subject_but_unrelated_relation_is_not_a_candidate(self):
        """ "I like ramen" is not about sushi and not about liking Google."""
        conversation = _Conversation(
            {},
            {
                (PropertyNames.Subject.value, "user"): [1, 2, 3],
                (PropertyNames.Object.value, "sushi"): [1],
                (PropertyNames.Verb.value, "like"): [1, 2],
            },
        )

        found = await related_ordinals(conversation, _action())
        assert found == {1, 2}
        assert 3 not in found

    @pytest.mark.asyncio
    async def test_missing_property_index_yields_nothing(self):
        conversation = _Conversation({}, {})
        conversation.secondary_indexes = None

        assert await related_ordinals(conversation, _action()) == set()


class TestFindCandidates:
    @pytest.mark.asyncio
    async def test_excludes_the_writes_own_refs(self):
        """A memory must not be retired as a contradiction of itself."""
        new = _SemanticRef(10, _action(verbs=("dislike",)))
        old = _SemanticRef(1, _action())
        conversation = _Conversation(
            {10: new, 1: old},
            {
                (PropertyNames.Subject.value, "user"): [1, 10],
                (PropertyNames.Object.value, "sushi"): [1, 10],
                (PropertyNames.Verb.value, "dislike"): [10],
            },
        )

        found = await find_candidates(conversation, [10])
        assert [c.raw.semantic_ref_ordinal for c in found] == [1]

    @pytest.mark.asyncio
    async def test_excludes_already_hidden_refs(self):
        new = _SemanticRef(10, _action(verbs=("dislike",)))
        conversation = _Conversation(
            {10: new, 1: _SemanticRef(1, _action())},
            {
                (PropertyNames.Subject.value, "user"): [1, 10],
                (PropertyNames.Object.value, "sushi"): [1, 10],
            },
        )

        found = await find_candidates(conversation, [10], hidden_ordinals={1})
        assert found == []

    @pytest.mark.asyncio
    async def test_non_propositional_new_knowledge_finds_nothing(self):
        """A topic asserts nothing, so it can contradict nothing."""
        conversation = _Conversation(
            {10: _SemanticRef(10, Topic(text="food"))},
            {(PropertyNames.Subject.value, "user"): [1]},
        )

        assert await find_candidates(conversation, [10]) == []

    @pytest.mark.asyncio
    async def test_non_propositional_candidates_are_dropped(self):
        """The index can return topics; they are not eligible either."""
        conversation = _Conversation(
            {
                10: _SemanticRef(10, _action(verbs=("dislike",))),
                1: _SemanticRef(1, Topic(text="sushi")),
                2: _SemanticRef(2, _action()),
            },
            {
                (PropertyNames.Subject.value, "user"): [1, 2],
                (PropertyNames.Object.value, "sushi"): [1, 2],
            },
        )

        found = await find_candidates(conversation, [10])
        assert [c.raw.semantic_ref_ordinal for c in found] == [2]

    @pytest.mark.asyncio
    async def test_no_new_ordinals_finds_nothing(self):
        conversation = _Conversation({}, {})
        assert await find_candidates(conversation, []) == []
