# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import black

import typechat

logger = logging.getLogger(__name__)

from .answer_context_schema import AnswerContext, RelevantAction, RelevantKnowledge, RelevantMessage
from .answer_response_schema import AnswerResponse
from .collections import get_top_k, Scored
from .interfaces import (
    DateRange,
    Datetime,
    IConversation,
    IMessage,
    IMessageCollection,
    IMessageMetadata,
    ISemanticRefCollection,
    ITermToSemanticRefIndex,
    Knowledge,
    KnowledgeType,
    MessageOrdinal,
    ScoredMessageOrdinal,
    ScoredSemanticRefOrdinal,
    SemanticRef,
    SemanticRefSearchResult,
    TextLocation,
    TextRange,
    Topic,
)
from .kplib import Action, ConcreteEntity, Facet
from .search import ConversationSearchResult


@dataclass
class AnswerContextOptions:
    entities_top_k: int | None = None
    topics_top_k: int | None = None
    messages_top_k: int | None = None
    chunking: bool | None = None
    debug: bool = False


async def generate_answers(
    translator: typechat.TypeChatJsonTranslator[AnswerResponse],
    search_results: list[ConversationSearchResult],
    conversation: IConversation,
    orig_query_text: str,
    options: AnswerContextOptions | None = None,
) -> tuple[list[AnswerResponse], AnswerResponse]:  # (all answers, combined answer)
    all_answers: list[AnswerResponse] = []
    good_answers: list[str] = []
    for result in search_results:
        answer = await generate_answer(translator, result, conversation, options)
        all_answers.append(answer)
        match answer.type:
            case "Answered":
                assert answer.answer is not None, "Answered answer must not be None"
                good = answer.answer.strip()
                if good:
                    good_answers.append(good)
            case "NoAnswer":
                pass
            case _:
                assert False, f"Unexpected answer type: {answer.type}"
    if len(all_answers) == 1:
        return all_answers, all_answers[0]
    combined_answer: AnswerResponse | None = None
    if len(good_answers) >= 2:
        combined_answer = await combine_answers(
            translator, good_answers, orig_query_text
        )
    elif len(good_answers) == 1:
        combined_answer = AnswerResponse(type="Answered", answer=good_answers[0])
    else:
        combined_answer = AnswerResponse(
            type="NoAnswer", why_no_answer="No good answers found."
        )
    return all_answers, combined_answer


async def generate_answer[TMessage: IMessage, TIndex: ITermToSemanticRefIndex](
    translator: typechat.TypeChatJsonTranslator[AnswerResponse],
    search_result: ConversationSearchResult,
    conversation: IConversation[TMessage, TIndex],
    options: AnswerContextOptions | None = None,
) -> AnswerResponse:
    assert search_result.raw_query_text is not None, "Raw query text must not be None"
    context = await make_context(search_result, conversation, options)
    request = f"{create_question_prompt(search_result.raw_query_text)}\n\n{create_context_prompt(context)}"
    if options and options.debug:
        logger.debug("Stage 4 input:\n%s\n%s", request, "-" * 50)
    result = await translator.translate(request)
    if isinstance(result, typechat.Failure):
        return AnswerResponse(
            type="NoAnswer",
            answer=None,
            why_no_answer=f"TypeChat failure: {result.message}",
        )
    else:
        return result.value


def create_question_prompt(question: str) -> str:
    prompt = [
        "The following is a user question:",
        "===",
        question,
        "===",
        "- The included [ANSWER CONTEXT] contains information that MAY be relevant to answering the question.",
        "- Answer the user question PRECISELY using ONLY relevant topics, entities, actions, messages and time ranges/timestamps found in [ANSWER CONTEXT].",
        "- Return 'NoAnswer' if unsure or if the topics and entity names/types in the question are not in [ANSWER CONTEXT].",
        "- Use the 'name', 'type' and 'facets' properties of the provided JSON entities to identify those highly relevant to answering the question.",
        "- When asked for lists, ensure the the list contents answer the question and nothing else.",
        "E.g. for the question 'List all books': List only the books in [ANSWER CONTEXT].",
        "- Use direct quotes only when needed or asked. Otherwise answer in your own words.",
        "- Your answer is readable and complete, with appropriate formatting: line breaks, numbered lists, bullet points etc.",
    ]
    return "\n".join(prompt)


def create_context_prompt(context: AnswerContext) -> str:
    # TODO: Use a more compact representation of the context than JSON.
    prompt = [
        "[ANSWER CONTEXT]",
        "===",
        black.format_str(str(dictify(context)), mode=black.Mode(line_length=200)),
        "===",
    ]
    return "\n".join(prompt)


def dictify(object: object) -> Any:
    """Convert an object to a dictionary, recursively."""
    # NOTE: Can't use dataclasses.asdict() because not every object is a dataclass.
    if ann := getattr(object.__class__, "__annotations__", None):
        return {
            k: dictify(v) for k in ann if (v := getattr(object, k, None)) is not None
        }
    elif isinstance(object, dict):
        return {k: dictify(v) for k, v in object.items() if v is not None}
    elif isinstance(object, list):
        return [dictify(item) for item in object]
    elif hasattr(object, "__dict__"):
        return {
            k: dictify(v) for k, v in object.__dict__.items() if v is not None
        }  #  if not k.startswith("_")
    else:
        if isinstance(object, float) and object.is_integer():
            return int(object)
        else:
            return object


async def make_context[TMessage: IMessage, TIndex: ITermToSemanticRefIndex](
    search_result: ConversationSearchResult,
    conversation: IConversation[TMessage, TIndex],
    options: AnswerContextOptions | None = None,
) -> AnswerContext:
    context = AnswerContext([], [], [], [])

    if search_result.message_matches:
        context.messages = await get_relevant_messages_for_answer(
            conversation,
            search_result.message_matches,
            options and options.messages_top_k,
        )

    # First pass: collect all knowledge by type
    all_actions: list[RelevantAction] = []
    entity_map: dict[str, ConcreteEntity] = {}  # For bidirectional linking

    for knowledge_type, knowledge in search_result.knowledge_matches.items():
        match knowledge_type:
            case "entity":
                context.entities = await get_relevant_entities_for_answer(
                    conversation,
                    knowledge,
                    options and options.entities_top_k,
                )
                # Build entity map for linking
                for rel_entity in context.entities:
                    if isinstance(rel_entity.knowledge, ConcreteEntity):
                        entity_map[rel_entity.knowledge.name.lower()] = rel_entity.knowledge
            case "action":
                all_actions = await get_relevant_actions_for_answer(
                    conversation,
                    knowledge,
                    entity_map,
                )
            case "topic":
                context.topics = await get_relevant_topics_for_answer(
                    conversation,
                    knowledge,
                    options and options.topics_top_k,
                )
            case _:
                pass  # Tags not supported yet

    # Second pass: bidirectional linking - always find actions related to entities
    if context.entities:
        related_actions = await find_actions_for_entities(
            conversation,
            [e.knowledge for e in context.entities if isinstance(e.knowledge, ConcreteEntity)],
            entity_map,
        )
        # Merge and deduplicate actions
        all_actions = merge_relevant_actions(all_actions, related_actions)

    context.actions = all_actions

    return context


def merge_relevant_actions(
    actions1: list[RelevantAction],
    actions2: list[RelevantAction],
) -> list[RelevantAction]:
    """Merge two lists of RelevantAction, removing duplicates."""
    seen: set[str] = set()
    merged: list[RelevantAction] = []

    for action in actions1 + actions2:
        # Create a unique key for deduplication
        key = f"{action.subject}|{','.join(action.verbs)}|{action.object}"
        if key not in seen:
            seen.add(key)
            merged.append(action)

    return merged


type MergedFacets = dict[str, list[str]]


# NOT a dataclass -- an optional merge-in attribute for MergedEntity etc.
class MergedKnowledge:
    source_message_ordinals: set[MessageOrdinal] | None = None


@dataclass
class MergedTopic(MergedKnowledge):
    topic: Topic


@dataclass
class MergedEntity(MergedKnowledge):
    name: str
    type: list[str]
    facets: MergedFacets | None = None


@dataclass
class MergedAction(MergedKnowledge):
    subject: str | None
    verbs: list[str]
    object: str | None
    indirect_object: str | None = None


async def get_relevant_messages_for_answer[
    TMessage: IMessage, TIndex: ITermToSemanticRefIndex
](
    conversation: IConversation[TMessage, TIndex],
    message_matches: list[ScoredMessageOrdinal],
    top_k: int | None = None,
) -> list[RelevantMessage]:
    relevant_messages = []

    for scored_msg_ord in message_matches:
        msg = await conversation.messages.get_item(scored_msg_ord.message_ordinal)
        if not msg.text_chunks:
            continue
        metadata: IMessageMetadata | None = msg.metadata
        assert metadata is not None  # For type checkers
        relevant_messages.append(
            RelevantMessage(
                from_=metadata.source,
                to=metadata.dest,
                timestamp=msg.timestamp,
                message_text=(
                    msg.text_chunks[0] if len(msg.text_chunks) == 1 else msg.text_chunks
                ),
            )
        )
        if top_k and len(relevant_messages) >= top_k:
            break

    return relevant_messages


async def get_relevant_topics_for_answer(
    conversation: IConversation,
    search_result: SemanticRefSearchResult,
    top_k: int | None = None,
) -> list[RelevantKnowledge]:
    assert conversation.semantic_refs is not None, "Semantic refs must not be None"
    scored_topics: Iterable[Scored[SemanticRef]] = (
        await get_scored_semantic_refs_from_ordinals_iter(
            conversation.semantic_refs,
            search_result.semantic_ref_matches,
            "topic",
        )
    )
    merged_topics = merge_scored_topics(scored_topics, True)
    candidate_topics: Iterable[Scored[MergedTopic]] = merged_topics.values()
    if top_k and len(merged_topics) > top_k:
        candidate_topics = get_top_k(candidate_topics, top_k)

    relevant_topics: list[RelevantKnowledge] = []

    for scored_value in candidate_topics:
        merged_topic = scored_value.item
        relevant_topics.append(
            await create_relevant_knowledge(
                conversation,
                merged_topic.topic,
                merged_topic.source_message_ordinals,
            )
        )

    return relevant_topics


def merge_scored_topics(
    scored_topics: Iterable[Scored[SemanticRef]],
    merge_ordinals: bool,
) -> dict[str, Scored[MergedTopic]]:
    merged_topics: dict[str, Scored[MergedTopic]] = {}

    for scored_topic in scored_topics:
        assert isinstance(scored_topic.item.knowledge, Topic)
        topic = scored_topic.item.knowledge
        existing = merged_topics.get(topic.text)
        if existing is not None:
            assert existing.item.topic.text == topic.text
            # Merge scores.
            if existing.score < scored_topic.score:
                existing.score = scored_topic.score
        else:
            existing = Scored(
                item=MergedTopic(topic=topic),
                score=scored_topic.score,
            )
            merged_topics[topic.text] = existing
        if merge_ordinals:
            merge_message_ordinals(existing.item, scored_topic.item)

    return merged_topics


async def get_relevant_entities_for_answer(
    conversation: IConversation,
    search_result: SemanticRefSearchResult,
    top_k: int | None = None,
) -> list[RelevantKnowledge]:
    assert conversation.semantic_refs is not None, "Semantic refs must not be None"
    merged_entities = merge_scored_concrete_entities(
        await get_scored_semantic_refs_from_ordinals_iter(
            conversation.semantic_refs,
            search_result.semantic_ref_matches,
            "entity",
        ),
        merge_ordinals=True,
    )
    candidate_entities = merged_entities.values()
    if top_k and len(merged_entities) > top_k:
        candidate_entities = get_top_k(candidate_entities, top_k)

    relevant_entities: list[RelevantKnowledge] = []

    for scored_value in candidate_entities:
        merged_entity = scored_value.item
        relevane_entity = await create_relevant_knowledge(
            conversation,
            merged_to_concrete_entity(merged_entity),
            merged_entity.source_message_ordinals,
        )
        relevant_entities.append(relevane_entity)

    return relevant_entities


async def get_relevant_actions_for_answer(
    conversation: IConversation,
    search_result: SemanticRefSearchResult,
    entity_map: dict[str, ConcreteEntity],
    top_k: int | None = None,
) -> list[RelevantAction]:
    """Extract relevant actions from search results with entity linking."""
    assert conversation.semantic_refs is not None, "Semantic refs must not be None"

    scored_actions = await get_scored_semantic_refs_from_ordinals_iter(
        conversation.semantic_refs,
        search_result.semantic_ref_matches,
        "action",
    )

    # Merge actions with same subject-verb-object
    merged_actions = merge_scored_actions(scored_actions, merge_ordinals=True)
    candidate_actions = list(merged_actions.values())

    if top_k and len(candidate_actions) > top_k:
        candidate_actions = list(get_top_k(candidate_actions, top_k))

    relevant_actions: list[RelevantAction] = []

    for scored_action in candidate_actions:
        merged = scored_action.item
        # Create RelevantAction with entity linking
        relevant_action = RelevantAction(
            subject=merged.subject,
            verbs=merged.verbs,
            object=merged.object,
            subject_entity=entity_map.get(merged.subject.lower()) if merged.subject else None,
            object_entity=entity_map.get(merged.object.lower()) if merged.object else None,
        )

        # Add time range if available
        if merged.source_message_ordinals:
            relevant_action.time_range = await get_enclosing_data_range_for_messages(
                conversation.messages, merged.source_message_ordinals
            )

        relevant_actions.append(relevant_action)

    return relevant_actions


def merge_scored_actions(
    scored_actions: Iterable[Scored[SemanticRef]],
    merge_ordinals: bool,
) -> dict[str, Scored[MergedAction]]:
    """Merge actions with the same subject-verb-object."""
    merged_actions: dict[str, Scored[MergedAction]] = {}

    for scored_action in scored_actions:
        assert isinstance(scored_action.item.knowledge, Action)
        action = scored_action.item.knowledge

        subject = action.subject_entity_name if action.subject_entity_name != "none" else None
        obj = action.object_entity_name if action.object_entity_name != "none" else None
        indirect_obj = action.indirect_object_entity_name if action.indirect_object_entity_name != "none" else None

        # Create a key for merging
        key = f"{subject}|{','.join(action.verbs)}|{obj}"

        existing = merged_actions.get(key)
        if existing is not None:
            # Merge scores
            if existing.score < scored_action.score:
                existing.score = scored_action.score
        else:
            existing = Scored(
                item=MergedAction(
                    subject=subject,
                    verbs=list(action.verbs),
                    object=obj,
                    indirect_object=indirect_obj,
                ),
                score=scored_action.score,
            )
            merged_actions[key] = existing

        if merge_ordinals:
            merge_message_ordinals(existing.item, scored_action.item)

    return merged_actions


async def find_actions_for_entities(
    conversation: IConversation,
    entities: list[ConcreteEntity],
    entity_map: dict[str, ConcreteEntity],
) -> list[RelevantAction]:
    """Find all actions where the given entities appear as subject or object.

    This implements bidirectional linking: when we find an entity,
    we also find all actions related to that entity.
    """
    assert conversation.semantic_refs is not None, "Semantic refs must not be None"
    assert conversation.secondary_indexes is not None, "Secondary indexes must not be None"

    property_index = conversation.secondary_indexes.property_to_semantic_ref_index
    entity_names = {e.name.lower() for e in entities}

    # Find all semantic refs where entity appears as subject or object
    related_action_ordinals: set[int] = set()

    for entity_name in entity_names:
        # Search for actions where this entity is the subject
        subject_matches = await property_index.lookup_property("subject", entity_name)
        if subject_matches:
            related_action_ordinals.update(subject_matches)

        # Search for actions where this entity is the object
        object_matches = await property_index.lookup_property("object", entity_name)
        if object_matches:
            related_action_ordinals.update(object_matches)

    if not related_action_ordinals:
        return []

    # Get the actual actions
    relevant_actions: list[RelevantAction] = []

    for ordinal in related_action_ordinals:
        semantic_ref = await conversation.semantic_refs.get_item(ordinal)
        if not isinstance(semantic_ref.knowledge, Action):
            continue

        action = semantic_ref.knowledge
        subject = action.subject_entity_name if action.subject_entity_name != "none" else None
        obj = action.object_entity_name if action.object_entity_name != "none" else None

        relevant_action = RelevantAction(
            subject=subject,
            verbs=list(action.verbs),
            object=obj,
            subject_entity=entity_map.get(subject.lower()) if subject else None,
            object_entity=entity_map.get(obj.lower()) if obj else None,
        )

        # Add time range
        if semantic_ref.range:
            relevant_action.time_range = await get_enclosing_data_range_for_messages(
                conversation.messages, {semantic_ref.range.start.message_ordinal}
            )

        relevant_actions.append(relevant_action)

    return relevant_actions


async def create_relevant_knowledge(
    conversation: IConversation,
    knowledge: Knowledge,
    source_message_ordinals: set[MessageOrdinal] | None = None,
) -> RelevantKnowledge:
    relevant_knowledge = RelevantKnowledge(knowledge)

    if source_message_ordinals:
        relevant_knowledge.time_range = await get_enclosing_data_range_for_messages(
            conversation.messages, source_message_ordinals
        )
        meta = await get_enclosing_metadata_for_messages(
            conversation.messages, source_message_ordinals
        )
        if meta.source:
            relevant_knowledge.origin = meta.source
        if meta.dest:
            relevant_knowledge.audience = meta.dest

    return relevant_knowledge


async def get_enclosing_data_range_for_messages(
    messages: IMessageCollection,
    message_ordinals: Iterable[MessageOrdinal],
) -> DateRange | None:
    text_range = get_enclosing_text_range(message_ordinals)
    if not text_range:
        return None
    return await get_enclosing_date_range_for_text_range(messages, text_range)


def get_enclosing_text_range(
    message_ordinals: Iterable[MessageOrdinal],
) -> TextRange | None:
    start: MessageOrdinal | None = None
    end: MessageOrdinal | None = start
    for ordinal in message_ordinals:
        if start is None or ordinal < start:
            start = ordinal
        if end is None or ordinal > end:
            end = ordinal
    if start is None or end is None:
        return None
    return text_range_from_message_range(start, end)


def text_range_from_message_range(
    start: MessageOrdinal, end: MessageOrdinal
) -> TextRange | None:
    if start == end:
        # Point location
        return TextRange(start=TextLocation(start))
    elif start < end:
        return TextRange(
            start=TextLocation(start),
            end=TextLocation(end),
        )
    else:
        raise ValueError(f"Expect message ordinal range: {start} <= {end}")


async def get_enclosing_date_range_for_text_range(
    messages: IMessageCollection,
    range: TextRange,
) -> DateRange | None:
    start_timestamp = (await messages.get_item(range.start.message_ordinal)).timestamp
    if not start_timestamp:
        return None
    end_timestamp = (
        (await messages.get_item(range.end.message_ordinal)).timestamp
        if range.end
        else None
    )
    return DateRange(
        start=Datetime.fromisoformat(start_timestamp),
        end=Datetime.fromisoformat(end_timestamp) if end_timestamp else None,
    )


@dataclass
class MessageMetadata(IMessageMetadata):
    source: str | list[str] | None = None
    dest: str | list[str] | None = None


async def get_enclosing_metadata_for_messages(
    messages: IMessageCollection,
    message_ordinals: Iterable[MessageOrdinal],
) -> IMessageMetadata:
    source: set[str] = set()
    dest: set[str] = set()

    def collect(s: set[str], value: str | list[str] | None) -> None:
        if isinstance(value, str):
            s.add(value)
        elif isinstance(value, list):
            s.update(value)

    for ordinal in message_ordinals:
        metadata = (await messages.get_item(ordinal)).metadata
        if not metadata:
            continue
        collect(source, metadata.source)
        collect(dest, metadata.dest)

    return MessageMetadata(
        source=list(source) if source else None, dest=list(dest) if dest else None
    )


async def get_scored_semantic_refs_from_ordinals_iter(
    semantic_refs: ISemanticRefCollection,
    semantic_ref_matches: list[ScoredSemanticRefOrdinal],
    knowledge_type: KnowledgeType,
) -> list[Scored[SemanticRef]]:
    result = []
    for semantic_ref_match in semantic_ref_matches:
        semantic_ref = await semantic_refs.get_item(
            semantic_ref_match.semantic_ref_ordinal
        )
        if semantic_ref.knowledge.knowledge_type == knowledge_type:
            result.append(
                Scored(
                    item=semantic_ref,
                    score=semantic_ref_match.score,
                )
            )
    return result


def merge_scored_concrete_entities(
    scored_entities: Iterable[Scored[SemanticRef]],
    merge_ordinals: bool,
) -> dict[str, Scored[MergedEntity]]:
    merged_entities: dict[str, Scored[MergedEntity]] = {}

    for scored_entity in scored_entities:
        assert isinstance(scored_entity.item.knowledge, ConcreteEntity)
        merged_entity = concrete_to_merged_entity(
            scored_entity.item.knowledge,
        )
        existing = merged_entities.get(merged_entity.name)
        if existing is not None:
            assert existing.item.name == merged_entity.name
            # Merge type list.
            if not existing.item.type:
                existing.item.type = merged_entity.type
            elif merged_entity.type:
                existing.item.type = sorted(
                    set(existing.item.type) | set(merged_entity.type)
                )
            # Merge facet dicts.
            if not existing.item.facets:
                existing.item.facets = merged_entity.facets
            elif merged_entity.facets:
                for name, value in merged_entity.facets.items():
                    existing.item.facets.setdefault(name, []).extend(value)
            # Merge scores.
            if existing.score < scored_entity.score:
                existing.score = scored_entity.score
        else:
            existing = Scored(
                item=merged_entity,
                score=scored_entity.score,
            )
            merged_entities[merged_entity.name] = existing
        if existing and merge_ordinals:
            merge_message_ordinals(existing.item, scored_entity.item)

    return merged_entities


def merge_message_ordinals(merged_entity: MergedKnowledge, sr: SemanticRef) -> None:
    if merged_entity.source_message_ordinals is None:
        merged_entity.source_message_ordinals = set()
    merged_entity.source_message_ordinals.add(sr.range.start.message_ordinal)


def concrete_to_merged_entity(
    entity: ConcreteEntity,
) -> MergedEntity:
    return MergedEntity(
        name=entity.name.lower(),
        type=sorted(tp.lower() for tp in entity.type),
        facets=facets_to_merged_facets(entity.facets) if entity.facets else None,
    )


def merged_to_concrete_entity(merged_entity: MergedEntity) -> ConcreteEntity:
    entity = ConcreteEntity(name=merged_entity.name, type=merged_entity.type)
    if merged_entity.facets:
        entity.facets = merged_facets_to_facets(merged_entity.facets)
    return entity


def facets_to_merged_facets(facets: list[Facet]) -> MergedFacets:
    merged_facets: MergedFacets = {}
    for facet in facets:
        name = facet.name.lower()
        value = str(facet).lower()
        merged_facets.setdefault(name, []).append(value)
    return merged_facets


def merged_facets_to_facets(merged_facets: MergedFacets) -> list[Facet]:
    facets: list[Facet] = []
    for facet_name, facet_values in merged_facets.items():
        if facet_values:
            facets.append(Facet(name=facet_name, value="; ".join(facet_values)))
    return facets


async def combine_answers(
    translator: typechat.TypeChatJsonTranslator[AnswerResponse],
    answers: list[str],
    original_query_text: str,
) -> AnswerResponse:
    """Combine multiple answers into a single answer."""
    if not answers:
        return AnswerResponse(type="NoAnswer", why_no_answer="No answers provided.")
    if len(answers) == 1:
        return AnswerResponse(type="Answered", answer=answers[0])
    request_parts = [
        "The following are multiple partial answers to the same question.",
        "Combine the partial answers into a single answer to the original question.",
        "Don't just concatenate the answers, but blend them into a single accurate and precise answer.",
        "",
        "*** Original Question ***",
        original_query_text,
        "*** Partial answers ***",
        "===",
    ]
    for answer in answers:
        request_parts.append(answer.strip())
        request_parts.append("===")
    request = "\n".join(request_parts)
    result = await translator.translate(request)
    if isinstance(result, typechat.Failure):
        return AnswerResponse(type="NoAnswer", why_no_answer=result.message)
    else:
        return result.value
