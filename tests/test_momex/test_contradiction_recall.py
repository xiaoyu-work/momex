"""Measures what the contradiction candidate lookup surfaces.

Offline and deterministic: the property index is a plain data structure, so
this runs in CI with no API key and no variance.

Two things are worth measuring here, and only one of them is a pass/fail bar:

  Recall is a ceiling on the whole feature. A memory that never becomes a
  candidate can never be retired, no matter how good the model is, and no
  amount of prompt work recovers it. Every case that expects a supersession
  must surface its target, so that is asserted.

  Candidate load is the cost side. Every surfaced candidate is prompt tokens
  and one more chance for the judge to be wrong. It is reported rather than
  asserted, because the right number is a judgment call, not a threshold.

The index is populated through the same collect_*_properties functions
production uses, so this exercises the real lookup rather than a model of it.
"""

from __future__ import annotations

import pytest

from momex.contradictions import find_candidates
from typeagent.storage.memory.propindex import (
    collect_action_properties,
    collect_entity_properties,
    PropertyIndex,
)

from .contradiction_cases import Case, CASES


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


class _Indexes:
    def __init__(self, property_index):
        self.property_to_semantic_ref_index = property_index


class _Conversation:
    def __init__(self, semrefs, property_index):
        self.semantic_refs = _Collection(semrefs)
        self.messages = _Collection({})
        self.secondary_indexes = _Indexes(property_index)


async def _index_properties(property_index, knowledge, ordinal):
    """Index one piece of knowledge exactly as the ingest path would."""
    knowledge_type = getattr(knowledge, "knowledge_type", None)
    if knowledge_type == "action":
        props = collect_action_properties(knowledge, ordinal)
    elif knowledge_type == "entity":
        props = collect_entity_properties(knowledge, ordinal)
    else:
        return  # Topics are not indexed by these properties.
    await property_index.add_properties_batch(props)


async def _build(case: Case):
    """Lay out a case as a collection: existing memories, then the new write."""
    property_index = PropertyIndex()
    semrefs: dict[int, _SemanticRef] = {}
    ordinal_of: dict[str, int] = {}

    for memory in case.existing:
        ordinal = len(semrefs)
        semrefs[ordinal] = _SemanticRef(ordinal, memory.knowledge)
        ordinal_of[memory.id] = ordinal
        await _index_properties(property_index, memory.knowledge, ordinal)

    new_ordinals: list[int] = []
    for knowledge in case.new_knowledge:
        ordinal = len(semrefs)
        semrefs[ordinal] = _SemanticRef(ordinal, knowledge)
        new_ordinals.append(ordinal)
        await _index_properties(property_index, knowledge, ordinal)

    return _Conversation(semrefs, property_index), ordinal_of, new_ordinals


async def _candidate_ids(case: Case) -> set[str]:
    conversation, ordinal_of, new_ordinals = await _build(case)
    candidates = await find_candidates(conversation, new_ordinals)

    found = {c.raw.semantic_ref_ordinal for c in candidates}
    return {mem_id for mem_id, o in ordinal_of.items() if o in found}


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
@pytest.mark.asyncio
async def test_expected_targets_are_reachable(case: Case):
    """Whatever should be retired must at least become a candidate."""
    if not case.should_retire:
        pytest.skip("nothing expected to be retired")

    surfaced = await _candidate_ids(case)
    missing = case.expect - surfaced

    assert not missing, (
        f"{case.id}: {sorted(missing)} can never be retired -- the lookup does "
        f"not surface them. Surfaced: {sorted(surfaced)}"
    )


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
@pytest.mark.asyncio
async def test_the_write_never_contradicts_itself(case: Case):
    """The refs a write just produced must never be among its own candidates."""
    conversation, _, new_ordinals = await _build(case)
    candidates = await find_candidates(conversation, new_ordinals)

    assert not ({c.raw.semantic_ref_ordinal for c in candidates} & set(new_ordinals))


@pytest.mark.parametrize(
    "case", [c for c in CASES if c.kind == "unrelated"], ids=lambda c: c.id
)
@pytest.mark.asyncio
async def test_unrelated_memories_cost_nothing(case: Case):
    """Surfacing nothing means the judge is never called at all.

    This is the case the lookup should get right on its own: a different
    subject, or a different relation, is not something a model should have to
    rule on.
    """
    assert await _candidate_ids(case) == set()


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
@pytest.mark.asyncio
async def test_noise_is_not_surfaced(case: Case):
    """Topics, bare entities and other subjects must not reach the prompt."""
    surfaced = await _candidate_ids(case)

    assert not {s for s in surfaced if s.startswith("noise-")}, (
        f"{case.id}: noise reached the judge: "
        f"{sorted(s for s in surfaced if s.startswith('noise-'))}"
    )


@pytest.mark.asyncio
async def test_report_candidate_load(capsys):
    """Print what each case surfaces. Reported, not asserted -- the right
    number is a judgment call rather than a threshold."""
    rows: list[tuple[str, str, int, str]] = []
    for case in CASES:
        surfaced = await _candidate_ids(case)
        verdict = "retire" if case.should_retire else "keep"
        rows.append(
            (case.id, verdict, len(surfaced), ",".join(sorted(surfaced)) or "-")
        )

    with capsys.disabled():
        print("\n  candidate load per case (lower is cheaper and safer)")
        print(f"  {'case':<32} {'expects':<8} {'n':>3}  surfaced")
        for case_id, verdict, count, surfaced in rows:
            print(f"  {case_id:<32} {verdict:<8} {count:>3}  {surfaced}")

        judged = sum(1 for _, _, count, _ in rows if count)
        total = len(rows)
        print(f"\n  cases reaching the judge: {judged}/{total}")
        print(f"  total candidates across all cases: {sum(r[2] for r in rows)}")
