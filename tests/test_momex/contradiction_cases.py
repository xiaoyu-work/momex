"""Shared evaluation set for contradiction handling.

Momex retires a memory when a new one contradicts it. That judgment decides
what a collection still remembers, and until now the arguments for how it works
were structural -- "a topic is not a proposition, so it cannot be contradicted"
-- with nothing measuring whether the result is right on concrete cases.

This is the concrete cases. It is deliberately small and hand-written: every
entry is one situation somebody can read and disagree with, which is the point.
Two harnesses consume it:

  tests/test_momex/test_contradiction_recall.py
      Offline and deterministic. Measures what the candidate lookup surfaces,
      which is a ceiling on the whole feature -- a memory that is never a
      candidate can never be retired, however good the model is.

  tools/eval_contradictions.py
      Online. Measures what a real model does with those candidates.

The four kinds are the distinctions that matter, and the last two are where a
memory system does damage:

  polarity     "I like sushi" then "I don't like sushi". Retire.
  replacement  "I work at Microsoft" then "I work at Google". Retire, but only
               because employment is single-valued.
  multivalue   "I like sushi" then "I like ramen". Keep both. Nothing about
               the surface form distinguishes this from a replacement; only
               knowing that preferences accumulate does.
  unrelated    Different subject, or nothing to do with each other. Keep.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typeagent.knowpro.knowledge_schema import Action, ConcreteEntity, Facet


def action(subject: str, verbs: list[str], obj: str = "none") -> Action:
    return Action(
        verbs=list(verbs),
        verb_tense="present",
        subject_entity_name=subject,
        object_entity_name=obj,
    )


def entity(name: str, types: list[str], **facets: str) -> ConcreteEntity:
    return ConcreteEntity(
        name=name,
        type=list(types),
        facets=[Facet(name=k, value=v) for k, v in facets.items()] or None,
    )


def topic(text: str):
    from typeagent.knowpro.interfaces import Topic

    return Topic(text=text)


@dataclass
class Memory:
    """One stored memory, named so cases can refer to it."""

    id: str
    knowledge: object


@dataclass
class Case:
    id: str
    kind: str
    """One of "polarity", "replacement", "multivalue", "unrelated"."""

    new_text: str
    """What the user said, as the judge sees it."""

    new_knowledge: list[object]
    """What extraction produces from new_text."""

    existing: list[Memory]
    expect: set[str] = field(default_factory=set)
    """Ids of `existing` that should end up superseded. Empty means keep all."""

    note: str = ""
    """Why this case is interesting, when it is not obvious."""

    @property
    def should_retire(self) -> bool:
        return bool(self.expect)


# Memories that appear in many cases as noise: true, about the same subject,
# and in a relation none of the cases touch. A candidate lookup that returns
# these is one that has not narrowed anything.
#
# Note the constraint on picking these: the noise action has to use a verb no
# case uses, or it stops being noise. "user own bicycle" was wrong here --
# against "I got a second cat" it is the same subject and the same `own`
# relation, so surfacing it is correct behaviour, not a miss.
def _noise() -> list[Memory]:
    return [
        Memory("noise-topic", topic("food")),
        Memory("noise-entity", entity("sushi", ["food"])),
        Memory("noise-action", action("user", ["read"], "newspaper")),
        Memory("noise-other-subject", action("alice", ["like"], "sushi")),
    ]


CASES: list[Case] = [
    # ---------------------------------------------------------------- polarity
    Case(
        id="polarity-like-sushi",
        kind="polarity",
        new_text="I don't like sushi anymore",
        new_knowledge=[action("user", ["dislike"], "sushi")],
        existing=[Memory("old", action("user", ["like"], "sushi"))] + _noise(),
        expect={"old"},
        note="The verb changed, so only the object still matches.",
    ),
    Case(
        id="polarity-negated-verb",
        kind="polarity",
        new_text="I no longer drink coffee",
        new_knowledge=[action("user", ["stop", "drinking"], "coffee")],
        existing=[Memory("old", action("user", ["drink"], "coffee"))] + _noise(),
        expect={"old"},
        note="Negation phrased as a different verb entirely.",
    ),
    Case(
        id="polarity-enjoy-vs-hate",
        kind="polarity",
        new_text="I hate running",
        new_knowledge=[action("user", ["hate"], "running")],
        existing=[Memory("old", action("user", ["enjoy"], "running"))] + _noise(),
        expect={"old"},
    ),
    Case(
        id="polarity-among-siblings",
        kind="polarity",
        new_text="I don't like sushi anymore",
        new_knowledge=[action("user", ["dislike"], "sushi")],
        existing=[
            Memory("old", action("user", ["like"], "sushi")),
            Memory("keep-ramen", action("user", ["like"], "ramen")),
            Memory("keep-udon", action("user", ["like"], "udon")),
        ]
        + _noise(),
        expect={"old"},
        note="The siblings share the subject, and a subject-only lookup would "
        "put all three in front of the judge. The object anchor separates "
        "them without a model: only 'sushi' matches, and the new verb "
        "'dislike' matches nothing.",
    ),
    # ------------------------------------------------------------- replacement
    Case(
        id="replacement-employer-action",
        kind="replacement",
        new_text="I work at Google now",
        new_knowledge=[action("user", ["work", "at"], "Google")],
        existing=[Memory("old", action("user", ["work", "at"], "Microsoft"))]
        + _noise(),
        expect={"old"},
        note="The object changed, so only the verb still matches.",
    ),
    Case(
        id="replacement-employer-facet",
        kind="replacement",
        new_text="I joined Google",
        new_knowledge=[entity("Xiaoyu", ["person"], employer="Google")],
        existing=[
            Memory("old", entity("Xiaoyu", ["person"], employer="Microsoft")),
            Memory("keep-role", entity("Xiaoyu", ["person"], role="engineer")),
        ]
        + _noise(),
        expect={"old"},
        note="Entities assert through facets; a different facet is untouched.",
    ),
    Case(
        id="replacement-city",
        kind="replacement",
        new_text="I moved to Portland",
        new_knowledge=[action("user", ["live", "in"], "Portland")],
        existing=[Memory("old", action("user", ["live", "in"], "Seattle"))] + _noise(),
        expect={"old"},
    ),
    Case(
        id="replacement-with-history",
        kind="replacement",
        new_text="I work at Google now",
        new_knowledge=[action("user", ["work", "at"], "Google")],
        existing=[
            Memory("old", action("user", ["work", "at"], "Microsoft")),
            Memory("older", action("user", ["work", "at"], "Amazon")),
        ]
        + _noise(),
        expect={"old", "older"},
        note="Both past employers share the verb and neither shares the new "
        "object, so the lookup cannot rank them and hands over both. This "
        "is where the judge earns its place -- and where it can do the most "
        "damage, since employment history is a thing people want kept.",
    ),
    # -------------------------------------------------------------- multivalue
    Case(
        id="multivalue-second-food",
        kind="multivalue",
        new_text="I like ramen too",
        new_knowledge=[action("user", ["like"], "ramen")],
        existing=[Memory("keep", action("user", ["like"], "sushi"))] + _noise(),
        note="Same subject and verb as a replacement. Only knowing that "
        "preferences accumulate tells the two apart.",
    ),
    Case(
        id="multivalue-second-language",
        kind="multivalue",
        new_text="I also speak Chinese",
        new_knowledge=[action("user", ["speak"], "Chinese")],
        existing=[Memory("keep", action("user", ["speak"], "English"))] + _noise(),
    ),
    Case(
        id="multivalue-second-pet",
        kind="multivalue",
        new_text="I got a second cat",
        new_knowledge=[action("user", ["own"], "cat")],
        existing=[Memory("keep", action("user", ["own"], "dog"))] + _noise(),
    ),
    # --------------------------------------------------------------- unrelated
    Case(
        id="unrelated-other-subject",
        kind="unrelated",
        new_text="I don't like sushi",
        new_knowledge=[action("user", ["dislike"], "sushi")],
        existing=[Memory("keep", action("alice", ["like"], "sushi"))],
        note="Someone else's preference says nothing about mine.",
    ),
    Case(
        id="unrelated-different-relation",
        kind="unrelated",
        new_text="I work at Google",
        new_knowledge=[action("user", ["work", "at"], "Google")],
        existing=[Memory("keep", action("user", ["like"], "sushi"))] + _noise(),
    ),
    Case(
        id="unrelated-topic-only",
        kind="unrelated",
        new_text="Let's talk about food",
        new_knowledge=[topic("food")],
        existing=[Memory("keep", action("user", ["like"], "sushi"))] + _noise(),
        note="A topic asserts nothing, so it can contradict nothing.",
    ),
]


def cases_by_kind(kind: str) -> list[Case]:
    return [case for case in CASES if case.kind == kind]
