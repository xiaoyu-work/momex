"""Deciding which existing memories a new one contradicts.

When "I don't like sushi" arrives, the earlier "likes sushi" should stop being
current. Momex asks an LLM to make that judgment, because the two statements
are only contradictory in meaning -- no index can see it.

The judgment is unreliable by nature, so nothing here deletes: it produces
ledger entries, and the ledger is reversible. Everything in this module is
best-effort, and every failure path returns "no contradictions found" rather
than raising, because by the time it runs the new memory is already committed.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from .paths import utc_now
from .results import SearchItem, SupersededRecord

logger = logging.getLogger(__name__)

# How many existing memories to put in front of the model. Enough to cover what
# a single statement plausibly contradicts, small enough to keep the prompt
# cheap and the indices easy to read back.
CANDIDATE_LIMIT = 20

PROMPT = """Given the new information and existing memories, identify which existing memories contradict the new information.

New information: "{new_content}"

Existing memories:
{existing_memories}

Return ONLY the indices (numbers) of memories that directly contradict the new information, separated by commas.
If no contradictions, return "none".
Only identify clear contradictions (e.g., "likes X" vs "doesn't like X"), not merely related information.

Response:"""


def is_propositional(item: SearchItem) -> bool:
    """True when this piece of knowledge asserts something that can be false.

    Contradiction is a relation between propositions, and most extracted
    knowledge is not one:

      - An action is. "user like sushi" has a truth value, and "user not-like
        sushi" denies it.
      - A bare entity is not. "sushi (type: food)" names a thing and
        categorises it; no statement about preferences can make it false.
      - An entity *with facets* is, through them. "Xiaoyu (type: person)
        [employer: Microsoft]" asserts an employer, which a later "I work at
        Google" replaces.
      - A topic is not. "dietary preferences" is a label, not a claim.

    Passing the others to the judge asks it a question with no correct answer,
    and the only outcome it can produce is a false positive -- a memory retired
    for contradicting something it cannot contradict. Missing a real
    contradiction merely leaves a stale memory, which the next add() can still
    correct; retiring a good one is silent data loss. So this errs toward
    keeping memories.
    """
    if item.type == "action":
        return True
    if item.type == "entity":
        knowledge = getattr(item.raw, "knowledge", None)
        return bool(getattr(knowledge, "facets", None))
    return False


def select_candidates(
    results: list[SearchItem],
    protect_semrefs_from: int | None,
) -> list[SearchItem]:
    """Narrow search results to what may legitimately be retired.

    Two exclusions. Anything that is not a proposition cannot be contradicted
    (see is_propositional). And the semantic refs at or above
    `protect_semrefs_from` are where the caller's own write starts -- those
    match the query by construction, and retiring them would make the new
    memory contradict itself.
    """
    candidates: list[SearchItem] = []
    for item in results:
        if not is_propositional(item):
            continue
        if protect_semrefs_from is not None:
            ordinal = getattr(item.raw, "semantic_ref_ordinal", None)
            if ordinal is not None and ordinal >= protect_semrefs_from:
                continue
        candidates.append(item)
    return candidates


def build_prompt(new_content: str, candidates: list[SearchItem]) -> str:
    """Render the prompt. Indices are dense so a small model cannot misread
    gaps as meaningful; the caller maps them back to the original results."""
    existing = "\n".join(
        f"{i}: [{item.type}] {item.text}" for i, item in enumerate(candidates)
    )
    return PROMPT.format(new_content=new_content, existing_memories=existing)


def parse_indices(response_text: str, candidate_count: int) -> list[int]:
    """Read the model's comma-separated indices, ignoring anything unusable.

    Deliberately lenient: a stray word or an out-of-range number costs one
    missed contradiction, whereas failing the parse would abandon all of them.
    """
    indices: list[int] = []
    for part in response_text.replace(" ", "").split(","):
        try:
            idx = int(part)
        except ValueError:
            continue
        if 0 <= idx < candidate_count:
            indices.append(idx)
    return indices


def records_for(
    indices: list[int],
    candidates: list[SearchItem],
    *,
    new_content: str,
    superseded_by: list[int] | None,
) -> list[SupersededRecord]:
    """Turn chosen indices into ledger entries, one per distinct ordinal."""
    texts_by_ordinal: dict[int, str] = {}
    ordinals: list[int] = []
    for idx in indices:
        candidate = candidates[idx]
        ordinal = getattr(candidate.raw, "semantic_ref_ordinal", None)
        if ordinal is not None:
            ordinals.append(ordinal)
            texts_by_ordinal.setdefault(ordinal, candidate.text)

    now = utc_now()
    return [
        SupersededRecord(
            ordinal=ordinal,
            superseded_by=list(superseded_by or []),
            at=now,
            reason="contradiction",
            text=texts_by_ordinal.get(ordinal),
            query=new_content,
        )
        for ordinal in dict.fromkeys(ordinals)
    ]


async def detect(
    new_content: str,
    *,
    collection: str,
    search_structured: Callable[..., Awaitable[list[SearchItem]]],
    create_llm: Callable[[], Any],
    append: Callable[[list[SupersededRecord]], Awaitable[list[SupersededRecord]]],
    protect_semrefs_from: int | None = None,
    superseded_by: list[int] | None = None,
) -> list[SupersededRecord]:
    """Retire the memories `new_content` contradicts, and report what was retired.

    Args:
        new_content: The new content being added.
        collection: Collection name, for log messages.
        search_structured: Finds existing knowledge. The structured path only:
            the embedding half returns messages exclusively, and messages are
            discarded here, so running it would be a wasted round trip.
        create_llm: Builds the LLM used to make the judgment.
        append: Appends to the supersession ledger, returning what it accepted.
        protect_semrefs_from: Semantic-ref ordinal marking the start of the
            knowledge extracted by the caller's own write.
        superseded_by: Ordinals of the knowledge that replaced them, recorded
            on each ledger entry.

    Returns:
        The ledger entries appended, empty when nothing was retired.
    """
    # This lookup is itself an LLM round trip, so it fails for the same
    # ordinary reasons the write does. It must not escape: add() has already
    # committed the new messages by this point, and raising here would report
    # a failure for a write that actually landed.
    try:
        results = await search_structured(new_content, limit=CANDIDATE_LIMIT)
    except Exception:
        logger.warning(
            "Contradiction detection lookup failed for collection %r; "
            "the new memory was added without it.",
            collection,
            exc_info=True,
        )
        return []

    if not results:
        return []

    candidates = select_candidates(results, protect_semrefs_from)
    if not candidates:
        return []

    try:
        llm = create_llm()
        response = await llm.complete(
            build_prompt(new_content, candidates), max_tokens=100
        )
        response_text = response.content.strip().lower()

        if response_text == "none" or not response_text:
            return []

        indices = parse_indices(response_text, len(candidates))
        if not indices:
            return []

        records = records_for(
            indices,
            candidates,
            new_content=new_content,
            superseded_by=superseded_by,
        )
        if records:
            return await append(records)

    except Exception:
        # Contradiction detection is best-effort and must never block add().
        logger.warning(
            "Contradiction detection failed for collection %r; "
            "adding the new memory without it.",
            collection,
            exc_info=True,
        )

    return []
