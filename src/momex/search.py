"""Retrieval: the two search paths and the fusion that combines them.

Momex searches twice for every query. The structured path asks an LLM to
compile the question into a TypeAgent term query and matches it against the
knowledge index; the embedding path skips the LLM and does similarity search
over message text. They find different things, and neither subsumes the other,
so both run and the results are merged.

The scores they produce are not comparable -- term-match weights are unbounded,
cosine similarities live in [0, 1] -- which is why the merge is by rank.
"""

from __future__ import annotations

import logging
from typing import Any

from .identity import memory_id
from .results import SearchItem
from .timewindow import extract_time_window, is_outside_window

logger = logging.getLogger(__name__)

# Reciprocal rank fusion constant, from the original RRF paper. Damps the
# influence of the top ranks so one list cannot dominate the merged order.
RRF_K = 60

# How far past `limit` to look for candidates when deduplicating. Extraction
# writes one semantic ref per message, so a frequently mentioned entity
# produces long runs of identical renderings that all score alike; the window
# has to be wide enough to reach past such a run and still find the results
# that differ. Bounded, so a broad query cannot turn into an unbounded fetch.
_DEDUPE_CANDIDATE_FACTOR = 10


async def fetch_many(collection: Any, ordinals: list[int]) -> dict[int, Any]:
    """Fetch ordinals in one call, falling back to one at a time.

    A batch fetch fails as a unit if any ordinal in it is missing, which the
    per-item path then works around. Gaps are expected: supersession leaves
    ordinals that no longer resolve.
    """
    if not ordinals:
        return {}
    try:
        items = await collection.get_multiple(ordinals)
        return dict(zip(ordinals, items))
    except (IndexError, KeyError):
        found: dict[int, Any] = {}
        for ordinal in ordinals:
            try:
                found[ordinal] = await collection.get_item(ordinal)
            except (IndexError, KeyError):
                continue
        return found


async def get_source_messages(conversation: Any, sem_refs) -> dict[int, Any]:
    """Batch-fetch the source message of each semantic ref, keyed by ordinal."""
    ordinals = sorted(
        {
            sem_ref.range.start.message_ordinal
            for sem_ref in sem_refs
            if getattr(sem_ref, "range", None)
        }
    )
    return await fetch_many(conversation.messages, ordinals)


def render_knowledge(knowledge: Any) -> str:
    """Render an extracted knowledge object as the text a caller sees.

    Entities carry their type and facets, actions read as subject-verb-object,
    topics are already text. Anything unrecognised falls back to repr rather
    than being dropped, so a new knowledge type degrades to something readable
    instead of vanishing from results.
    """
    from typeagent.knowpro import knowledge_schema as kplib
    from typeagent.knowpro.interfaces import Topic

    if isinstance(knowledge, kplib.ConcreteEntity):
        text = knowledge.name
        if knowledge.type:
            text += f" (type: {', '.join(knowledge.type)})"
        if knowledge.facets:
            facets = [f"{f.name}: {f.value}" for f in knowledge.facets if f.value]
            if facets:
                text += f" [{'; '.join(facets)}]"
        return text

    if isinstance(knowledge, kplib.Action):
        parts = []
        if knowledge.subject_entity_name:
            parts.append(knowledge.subject_entity_name)
        parts.extend(knowledge.verbs)
        if knowledge.object_entity_name:
            parts.append(knowledge.object_entity_name)
        return " ".join(parts)

    if isinstance(knowledge, Topic):
        return knowledge.text

    return str(knowledge)


def message_text(msg: Any) -> str:
    return " ".join(msg.text_chunks) if hasattr(msg, "text_chunks") else str(msg)


def filter_search_results(results, hidden_ordinals: set[int]):
    """Drop superseded knowledge from raw search results.

    The underlying records still exist; the ledger decides what is visible.
    """
    if not hidden_ordinals:
        return results
    filtered = []
    for search_result in results:
        knowledge_matches = {}
        for ktype, kmatches in search_result.knowledge_matches.items():
            kept = [
                match
                for match in kmatches.semantic_ref_matches
                if match.semantic_ref_ordinal not in hidden_ordinals
            ]
            if kept:
                kmatches.semantic_ref_matches = kept
                knowledge_matches[ktype] = kmatches
        search_result.knowledge_matches = knowledge_matches
        if search_result.knowledge_matches or search_result.message_matches:
            filtered.append(search_result)
    return filtered


def fuse_results(*result_lists: list[SearchItem], limit: int) -> list[SearchItem]:
    """Merge ranked result lists using reciprocal rank fusion.

    Structured search returns term-match weights, which are unbounded and
    routinely exceed 1, while embedding search returns cosine similarities
    in [0, 1]. Sorting the two together by raw score is meaningless, so they
    are combined by rank instead of by magnitude.
    """
    best: dict[str, SearchItem] = {}
    fused_scores: dict[str, float] = {}

    for items in result_lists:
        seen: set[str] = set()
        for rank, item in enumerate(items):
            # Same text from two indexes is one memory, and should be
            # rewarded for appearing in both -- but only once per list.
            if item.text in seen:
                continue
            seen.add(item.text)
            fused_scores[item.text] = fused_scores.get(item.text, 0.0) + 1.0 / (
                RRF_K + rank + 1
            )
            best.setdefault(item.text, item)

    for text, item in best.items():
        item.fusion_score = fused_scores[text]

    ordered = sorted(
        best.values(),
        key=lambda item: (item.fusion_score or 0.0, item.score),
        reverse=True,
    )
    return ordered[:limit]


async def items_for_semrefs(
    conversation: Any,
    ordinals: list[int],
    *,
    scores: dict[int, float] | None = None,
    include_expired: bool = False,
) -> list[SearchItem]:
    """Build SearchItems for the given semantic refs.

    Knowledge inherits the timestamp, validity window and source identity of
    the message it was extracted from, so those are fetched up front and
    attached here. Refs that no longer resolve, or whose window has closed, are
    dropped.
    """
    sem_ref_map = await fetch_many(conversation.semantic_refs, ordinals)
    src_msg_map = await get_source_messages(conversation, sem_ref_map.values())

    items: list[SearchItem] = []
    for ordinal in ordinals:
        sem_ref = sem_ref_map.get(ordinal)
        if sem_ref is None:
            continue

        src_timestamp: str | None = None
        valid_from: str | None = None
        valid_to: str | None = None
        source_id: str | None = None
        if getattr(sem_ref, "range", None):
            src_msg = src_msg_map.get(sem_ref.range.start.message_ordinal)
            if src_msg is not None:
                src_timestamp = getattr(src_msg, "timestamp", None)
                source_id = getattr(src_msg, "source_id", None)
                valid_from, valid_to = extract_time_window(src_msg)

        if not include_expired and is_outside_window(valid_from, valid_to):
            continue

        items.append(
            SearchItem(
                type=sem_ref.knowledge.knowledge_type,
                text=render_knowledge(sem_ref.knowledge),
                score=scores.get(ordinal, 0.0) if scores is not None else 1.0,
                raw=sem_ref,
                timestamp=src_timestamp,
                valid_from=valid_from,
                valid_to=valid_to,
                memory_id=memory_id(source_id, sem_ref.knowledge),
            )
        )
    return items


async def lookup_property_ordinals(
    conversation: Any, property_name: str, value: str
) -> set[int]:
    """Semantic refs carrying one property value, via the property index.

    A direct index probe: no LLM, no query compilation. Returns an empty set
    when the index is absent or nothing matches.
    """
    if not value:
        return set()

    indexes = getattr(conversation, "secondary_indexes", None)
    property_index = getattr(indexes, "property_to_semantic_ref_index", None)
    if property_index is None:
        return set()

    scored = await property_index.lookup_property(property_name, value)
    if not scored:
        return set()
    return {s.semantic_ref_ordinal for s in scored}


async def search_structured(
    conversation: Any,
    query_text: str,
    limit: int = 10,
    *,
    hidden_ordinals: set[int] | None = None,
    include_expired: bool = False,
    dedupe: bool = True,
) -> list[SearchItem]:
    """Structured RAG search using LLM query translation + term matching.

    `hidden_ordinals` are dropped from the raw results before anything is
    fetched, so superseded knowledge never reaches the caller.

    `dedupe` collapses knowledge that renders identically, keeping the
    highest-scoring occurrence. This matters more than it sounds. Extraction
    produces one semantic ref per message, so an entity mentioned in two
    hundred messages becomes two hundred refs that all render to
    "Caroline (type: person)" and all score the same for a term query naming
    her. Without collapsing them, the entire top-k budget goes to copies of
    one entity: measured on LOCOMO, the top 20 structured results were 20
    duplicates and answered 0% of questions, while the same search deduped
    leaves room for the refs that actually differ.

    Turn it off when you need every matching ordinal rather than every
    distinct memory -- delete() does, because two refs that read the same are
    still two refs and retiring one would leave the other visible.
    """
    import typechat

    from typeagent.aitools import utils
    from typeagent.knowpro import convknowledge, search_query_schema, searchlang

    # Initialize query translator if needed
    if conversation._query_translator is None:
        model = convknowledge.create_typechat_model()
        conversation._query_translator = utils.create_translator(
            model, search_query_schema.SearchQuery
        )

    options = searchlang.LanguageSearchOptions(
        compile_options=searchlang.LanguageQueryCompileOptions(
            exact_scope=False,
            verb_scope=True,
            term_filter=None,
            apply_scope=False,
        ),
        exact_match=False,
        max_message_matches=limit,
    )

    result = await searchlang.search_conversation_with_language(
        conversation,
        conversation._query_translator,
        query_text,
        options,
    )

    if isinstance(result, typechat.Failure):
        return []

    search_results = result.value
    if hidden_ordinals:
        search_results = filter_search_results(search_results, hidden_ordinals)

    # Collect all ordinals first for batch fetching. When deduping, take a
    # wider window than the caller asked for: the duplicates are only visible
    # once the knowledge is fetched and rendered, so slicing at `limit` here
    # would discard the distinct results before anything could tell them
    # apart. A run of one entity's copies is routinely longer than `limit`.
    window = limit * _DEDUPE_CANDIDATE_FACTOR if dedupe else limit

    semref_requests: list[tuple[int, float]] = []  # (ordinal, score)
    msg_requests: list[tuple[int, float]] = []  # (ordinal, score)

    for search_result in search_results:
        for _, matches in search_result.knowledge_matches.items():
            for scored in matches.semantic_ref_matches[:window]:
                semref_requests.append((scored.semantic_ref_ordinal, scored.score))
        for msg_match in search_result.message_matches[:window]:
            msg_requests.append((msg_match.message_ordinal, msg_match.score))

    items: list[SearchItem] = []

    if semref_requests:
        # Later duplicates of an ordinal keep the first (highest) score, since
        # the requests arrive in the order the index ranked them.
        scores: dict[int, float] = {}
        for ordinal, score in semref_requests:
            scores.setdefault(ordinal, score)

        items.extend(
            await items_for_semrefs(
                conversation,
                list(scores),
                scores=scores,
                include_expired=include_expired,
            )
        )

    if msg_requests:
        msg_map = await fetch_many(conversation.messages, [o for o, _ in msg_requests])

        for ordinal, score in msg_requests:
            msg = msg_map.get(ordinal)
            if msg is None:
                continue

            vf, vt = extract_time_window(msg)
            if not include_expired and is_outside_window(vf, vt):
                continue

            items.append(
                SearchItem(
                    type="message",
                    text=message_text(msg),
                    score=score,
                    raw=msg,
                    timestamp=getattr(msg, "timestamp", None),
                    valid_from=vf,
                    valid_to=vt,
                )
            )

    items.sort(key=lambda x: x.score, reverse=True)

    if dedupe:
        # Already sorted, so the first occurrence of a rendering is also its
        # highest-scoring one.
        seen: set[tuple[str, str]] = set()
        distinct: list[SearchItem] = []
        for item in items:
            key = (item.type, item.text)
            if key in seen:
                continue
            seen.add(key)
            distinct.append(item)
        items = distinct

    return items[:limit]


async def search_by_embedding(
    conversation: Any,
    collection: str,
    query_text: str,
    limit: int = 10,
    min_score: float = 0.3,
    *,
    include_expired: bool = False,
) -> list[SearchItem]:
    """Embedding similarity search over message text, without an LLM.

    Queries the MessageTextIndex directly, bypassing query translation. Returns
    messages only -- extracted knowledge is not embedded.
    """
    if (
        conversation.secondary_indexes is None
        or conversation.secondary_indexes.message_index is None
    ):
        return []

    msg_index = conversation.secondary_indexes.message_index

    try:
        scored_ordinals = await msg_index.lookup_messages(
            query_text,
            max_matches=limit,
            threshold_score=min_score,
        )
    except Exception:
        logger.warning(
            "Message index lookup failed for collection %r.",
            collection,
            exc_info=True,
        )
        return []

    if not scored_ordinals:
        return []

    items: list[SearchItem] = []
    for scored in scored_ordinals:
        try:
            msg = await conversation.messages.get_item(scored.message_ordinal)
        except (IndexError, KeyError):
            continue

        vf, vt = extract_time_window(msg)
        if not include_expired and is_outside_window(vf, vt):
            continue

        items.append(
            SearchItem(
                type="message",
                text=message_text(msg),
                score=scored.score,
                raw=msg,
                timestamp=getattr(msg, "timestamp", None),
                valid_from=vf,
                valid_to=vt,
            )
        )

    items.sort(key=lambda x: x.score, reverse=True)
    return items[:limit]
