# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# TODO: Are we sure this isn't used as a translator schema class?

from dataclasses import dataclass
from typing import Annotated, Any, Union

from typing_extensions import Doc

from ..knowpro.interfaces import DateRange

EntityNames = Union[str, list[str]]


@dataclass
class RelevantKnowledge:
    knowledge: Annotated[Any, Doc("The actual knowledge")]
    origin: Annotated[
        EntityNames | None, Doc("Entity or entities who mentioned the knowledge")
    ] = None
    audience: Annotated[
        EntityNames | None,
        Doc("Entity or entities who received or consumed this knowledge"),
    ] = None
    time_range: Annotated[
        DateRange | None, Doc("Time period during which this knowledge was gathered")
    ] = None


@dataclass
class RelevantMessage:
    from_: Annotated[EntityNames | None, Doc("Sender(s) of the message")]
    to: Annotated[EntityNames | None, Doc("Recipient(s) of the message")]
    timestamp: Annotated[str | None, Doc("Timestamp of the message in ISO format")]
    message_text: Annotated[str | list[str] | None, Doc("Text chunks in this message")]


@dataclass
class RelevantAction:
    """An action representing a relationship between entities."""

    subject: Annotated[
        str | None,
        Doc("The entity performing the action (e.g., 'xiaoyuzhang' in 'xiaoyuzhang likes Python')"),
    ]
    verbs: Annotated[
        list[str],
        Doc("The action verbs (e.g., ['likes'] in 'xiaoyuzhang likes Python')"),
    ]
    object: Annotated[
        str | None,
        Doc("The entity receiving the action (e.g., 'Python' in 'xiaoyuzhang likes Python')"),
    ]
    subject_entity: Annotated[
        Any | None,
        Doc("Full entity details for the subject, if available"),
    ] = None
    object_entity: Annotated[
        Any | None,
        Doc("Full entity details for the object, if available"),
    ] = None
    time_range: Annotated[
        DateRange | None,
        Doc("Time period during which this action occurred"),
    ] = None


@dataclass
class AnswerContext:
    """Use empty lists for unneeded properties."""

    entities: Annotated[
        list[RelevantKnowledge],
        Doc(
            "Relevant entities. Use the 'name' and 'type' properties of entities to PRECISELY identify those that answer the user question."
        ),
    ]
    actions: Annotated[
        list[RelevantAction],
        Doc(
            "Relevant actions representing relationships between entities. "
            "Use 'subject', 'verbs', and 'object' to understand WHO did WHAT to WHOM."
        ),
    ]
    topics: Annotated[list[RelevantKnowledge], Doc("Relevant topics")]
    messages: Annotated[list[RelevantMessage], Doc("Relevant messages")]
