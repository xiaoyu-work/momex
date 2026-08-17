"""Tests for which memories are eligible to be contradicted.

Contradiction is a relation between propositions. Most extracted knowledge is
not one: an entity names a thing, a topic labels a subject, and neither can be
made false by a later statement. Feeding them to the judge asks a question with
no correct answer, and the only thing it can produce is a memory retired for
contradicting something it could not have contradicted.
"""

import pytest

from momex.contradictions import is_propositional
from momex.results import SearchItem
from typeagent.knowpro.knowledge_schema import Action, ConcreteEntity, Facet


class _SemanticRef:
    def __init__(self, ordinal, knowledge=None):
        self.semantic_ref_ordinal = ordinal
        self.knowledge = knowledge


def _item(type_, text, ordinal=0, knowledge=None):
    return SearchItem(
        type=type_,
        text=text,
        score=9.0,
        raw=_SemanticRef(ordinal, knowledge),
    )


def _entity(name, facets=None):
    return ConcreteEntity(name=name, type=["person"], facets=facets)


def _action(subject, verbs, obj):
    return Action(
        verbs=verbs,
        verb_tense="present",
        subject_entity_name=subject,
        object_entity_name=obj,
    )


class TestIsPropositional:
    def test_action_is_propositional(self):
        """ "user like sushi" has a truth value; "not-like" denies it."""
        assert is_propositional(
            _item(
                "action",
                "user like sushi",
                knowledge=_action("user", ["like"], "sushi"),
            )
        )

    def test_bare_entity_is_not(self):
        """ "sushi (type: food)" cannot be made false by a preference."""
        assert not is_propositional(
            _item("entity", "sushi (type: food)", knowledge=_entity("sushi"))
        )

    def test_entity_with_facets_is(self):
        """The facet carries the assertion: employer, city, and so on."""
        entity = _entity("Xiaoyu", facets=[Facet(name="employer", value="Microsoft")])
        assert is_propositional(
            _item("entity", "Xiaoyu [employer: Microsoft]", knowledge=entity)
        )

    def test_entity_with_empty_facet_list_is_not(self):
        assert not is_propositional(
            _item("entity", "Xiaoyu", knowledge=_entity("Xiaoyu", facets=[]))
        )

    def test_topic_is_not(self):
        """A topic is a label, not a claim."""
        assert not is_propositional(_item("topic", "dietary preferences"))

    def test_message_is_not(self):
        assert not is_propositional(_item("message", "I like sushi"))

    def test_missing_knowledge_is_not(self):
        """Absent the knowledge object there is nothing to judge."""
        assert not is_propositional(_item("entity", "sushi"))

    def test_a_mixed_result_set_reduces_to_its_assertions(self):
        """What one sentence typically extracts, and what survives of it."""
        results = [
            _item("message", "I like sushi", 0),
            _item("entity", "sushi (type: food)", 1, knowledge=_entity("sushi")),
            _item("topic", "dietary preferences", 2),
            _item(
                "entity",
                "Xiaoyu [employer: Microsoft]",
                3,
                knowledge=_entity(
                    "Xiaoyu", facets=[Facet(name="employer", value="Microsoft")]
                ),
            ),
            _item(
                "action",
                "user like sushi",
                4,
                knowledge=_action("user", ["like"], "sushi"),
            ),
        ]

        kept = [item.text for item in results if is_propositional(item)]
        assert kept == ["Xiaoyu [employer: Microsoft]", "user like sushi"]
