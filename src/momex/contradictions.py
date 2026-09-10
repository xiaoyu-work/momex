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

import json
import logging
from typing import Any, Awaitable, Callable

from .paths import utc_now
from .results import SearchItem, SupersededRecord
from .search import items_for_semrefs, lookup_property_ordinals

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


def relation_probes(
    knowledge: Any,
) -> tuple[tuple[str, str] | None, list[tuple[str, str]]]:
    """The property lookups that find propositions about the same relation.

    Returns a subject probe and a list of anchor probes. A candidate has to
    match the subject *and* at least one anchor, which is what keeps this from
    degenerating into "everything ever said about the user".

    Two anchors, because contradiction arrives in two shapes:

      - Polarity flip: "user like sushi" vs "user dislike sushi". The verb
        changes, so the *object* is what still matches.
      - Value replacement: "user work-at Microsoft" vs "user work-at Google".
        The object changes, so the *verb* is what still matches.

    Matching on the verb alone would miss every polarity flip, which is the
    case this feature exists for.

    Entities work the same way with different property names: the entity is the
    subject and each facet name is an anchor, so "[employer: Microsoft]" is
    reachable from "[employer: Google]".
    """
    from typeagent.storage.memory.propindex import PropertyNames

    knowledge_type = getattr(knowledge, "knowledge_type", None)

    if knowledge_type == "action":
        subject = _usable(getattr(knowledge, "subject_entity_name", None))
        if not subject:
            return None, []
        anchors: list[tuple[str, str]] = []
        verbs = getattr(knowledge, "verbs", None) or []
        if verbs:
            anchors.append((PropertyNames.Verb.value, " ".join(verbs)))
        obj = _usable(getattr(knowledge, "object_entity_name", None))
        if obj:
            anchors.append((PropertyNames.Object.value, obj))
        return (PropertyNames.Subject.value, subject), anchors

    if knowledge_type == "entity":
        facets = getattr(knowledge, "facets", None) or []
        name = _usable(getattr(knowledge, "name", None))
        if not name or not facets:
            return None, []
        return (
            (PropertyNames.EntityName.value, name),
            [(PropertyNames.FacetName.value, f.name) for f in facets if f.name],
        )

    return None, []


def _usable(value: Any) -> str | None:
    """Action fields default to the literal "none" rather than being absent."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or text.lower() == "none":
        return None
    return text


async def related_ordinals(conversation: Any, knowledge: Any) -> set[int]:
    """Semantic refs asserting something about the same subject and relation."""
    subject_probe, anchor_probes = relation_probes(knowledge)
    if subject_probe is None or not anchor_probes:
        return set()

    subject_hits = await lookup_property_ordinals(conversation, *subject_probe)
    if not subject_hits:
        return set()

    anchor_hits: set[int] = set()
    for probe in anchor_probes:
        anchor_hits |= await lookup_property_ordinals(conversation, *probe)

    return subject_hits & anchor_hits


async def find_candidates(
    conversation: Any,
    new_ordinals: list[int],
    *,
    hidden_ordinals: set[int] | None = None,
    limit: int = CANDIDATE_LIMIT,
    include_expired: bool = False,
    include_new_peers: bool = False,
) -> list[SearchItem]:
    """Find what the memories just written might contradict.

    Driven by the knowledge the write already produced, rather than by
    recompiling the raw text into a natural-language query. That avoids an LLM
    round trip, and it asks a better question: "what else is asserted about
    this subject and this relation" instead of "what is topically similar".

    The caller's own semantic refs are excluded -- they match by construction,
    and a memory must not be retired as a contradiction of itself.
    """
    new_refs = await items_for_semrefs(
        conversation, list(new_ordinals), include_expired=include_expired
    )

    exclude = (set() if include_new_peers else set(new_ordinals)) | (
        hidden_ordinals or set()
    )
    found: set[int] = set()
    for item in new_refs:
        if not is_propositional(item):
            continue
        found |= await related_ordinals(conversation, item.raw.knowledge)

    ordinals = sorted(found - exclude)
    if not ordinals:
        return []

    candidates = await items_for_semrefs(
        conversation, ordinals, include_expired=include_expired
    )
    candidates = [item for item in candidates if is_propositional(item)]
    if include_new_peers:
        candidates = [
            candidate
            for candidate in candidates
            if any(
                is_propositional(item)
                and item.raw.semantic_ref_ordinal != candidate.raw.semantic_ref_ordinal
                and item.text != candidate.text
                for item in new_refs
            )
        ]
    if include_expired:
        candidates.sort(key=lambda item: item.timestamp or "", reverse=True)
    return candidates[:limit]


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
        knowledge = getattr(item.raw, "knowledge", None)
        if getattr(knowledge, "verbs", None) == ["say", "speak"]:
            return False
        return True
    if item.type == "entity":
        knowledge = getattr(item.raw, "knowledge", None)
        return bool(getattr(knowledge, "facets", None))
    return False


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
    ids_by_ordinal: dict[int, str | None] = {}
    ordinals: list[int] = []
    for idx in indices:
        candidate = candidates[idx]
        ordinal = getattr(candidate.raw, "semantic_ref_ordinal", None)
        if ordinal is not None:
            ordinals.append(ordinal)
            texts_by_ordinal.setdefault(ordinal, candidate.text)
            ids_by_ordinal.setdefault(ordinal, candidate.memory_id)

    now = utc_now()
    return [
        SupersededRecord(
            ordinal=ordinal,
            superseded_by=list(superseded_by or []),
            at=now,
            reason="contradiction",
            text=texts_by_ordinal.get(ordinal),
            query=new_content,
            memory_id=ids_by_ordinal.get(ordinal),
        )
        for ordinal in dict.fromkeys(ordinals)
    ]


def temporal_prompt(new_items: list[SearchItem], candidates: list[SearchItem]) -> str:
    def describe(item: SearchItem) -> dict[str, str | None]:
        return {
            "text": item.text,
            "timestamp": item.timestamp,
            "valid_from": item.valid_from,
            "valid_to": item.valid_to,
        }

    return (
        "Identify direct contradictions between NEW and EXISTING propositions. "
        "Do not treat multi-valued additions (another food, language or office) "
        "as replacements without clear evidence. Dates are event times, not "
        "ingestion order: a NEW entry can be older than an EXISTING entry. "
        "Report incompatible pairs only, not which side should win. Return "
        "only a JSON array of [new_index, existing_index] pairs, or [] if none.\n"
        + json.dumps(
            {
                "NEW": [describe(item) for item in new_items],
                "EXISTING": [describe(item) for item in candidates],
            },
            ensure_ascii=False,
        )
    )


def temporal_records(
    response: str, new_items: list[SearchItem], candidates: list[SearchItem]
) -> list[SupersededRecord]:
    """A later event replaces an earlier one, only during their overlap."""
    pairs = json.loads(response)
    if not isinstance(pairs, list):
        raise ValueError("contradiction response must be an array of index pairs")

    def start(item: SearchItem) -> str:
        return max(
            item.timestamp or "",
            f"{item.valid_from}T00:00:00Z" if item.valid_from else "",
        )

    records: list[SupersededRecord] = []
    for pair in pairs:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or any(type(index) is not int for index in pair)
            or not 0 <= pair[0] < len(new_items)
            or not 0 <= pair[1] < len(candidates)
        ):
            raise ValueError("invalid contradiction index pair")
        new, existing = new_items[pair[0]], candidates[pair[1]]
        if new.raw.semantic_ref_ordinal == existing.raw.semantic_ref_ordinal:
            continue
        older, newer = (
            (new, existing) if start(new) < start(existing) else (existing, new)
        )
        effective_at = start(newer) or utc_now()
        ends = [
            f"{item.valid_to}T23:59:59Z" for item in (older, newer) if item.valid_to
        ]
        effective_to = min(ends) if ends else None
        if effective_to and effective_to < effective_at:
            continue
        records.append(
            SupersededRecord(
                ordinal=older.raw.semantic_ref_ordinal,
                superseded_by=[newer.raw.semantic_ref_ordinal],
                at=utc_now(),
                reason="contradiction",
                text=older.text,
                query=newer.text,
                memory_id=older.memory_id,
                effective_at=effective_at,
                effective_to=effective_to,
            )
        )
    return records


async def detect(
    new_content: str,
    *,
    collection: str,
    find_candidates: Callable[[], Awaitable[list[SearchItem]]],
    create_llm: Callable[[], Any],
    append: Callable[[list[SupersededRecord]], Awaitable[list[SupersededRecord]]],
    superseded_by: list[int] | None = None,
    find_new_items: Callable[[], Awaitable[list[SearchItem]]] | None = None,
) -> list[SupersededRecord]:
    """Retire the memories `new_content` contradicts, and report what was retired.

    Args:
        new_content: The new content being added.
        collection: Collection name, for log messages.
        find_candidates: Produces the propositions that might be contradicted.
        create_llm: Builds the LLM used to make the judgment.
        append: Appends to the supersession ledger, returning what it accepted.
        superseded_by: Ordinals of the knowledge that replaced them, recorded
            on each ledger entry.

    Returns:
        The ledger entries appended, empty when nothing was retired.
    """
    # Candidate lookup must not escape: add() has already committed the new
    # messages by this point, and raising here would report a failure for a
    # write that actually landed.
    try:
        new_items = (
            [item for item in await find_new_items() if is_propositional(item)]
            if find_new_items is not None
            else None
        )
        candidates = await find_candidates()
    except Exception:
        logger.warning(
            "Contradiction detection lookup failed for collection %r; "
            "the new memory was added without it.",
            collection,
            exc_info=True,
        )
        return []

    if not candidates or new_items == []:
        return []

    try:
        llm = create_llm()
        response = await llm.complete(
            (
                temporal_prompt(new_items, candidates)
                if new_items is not None
                else build_prompt(new_content, candidates)
            ),
            max_tokens=512 if new_items is not None else 100,
        )
        response_text = response.content.strip().lower()

        if new_items is not None:
            records = temporal_records(response_text, new_items, candidates)
            return await append(records) if records else []

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
