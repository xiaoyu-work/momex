"""Prepare attributed writes without replacing the durable original text."""

from collections.abc import Mapping, Sequence
import json
from typing import Any, Literal

from typeagent.knowpro.universal_message import (
    ConversationMessage,
    ConversationMessageMeta,
)

from .identity import new_source_id
from .timewindow import (
    VALID_FROM_TAG,
    VALID_TO_TAG,
    validate_iso_date,
    validate_timestamp,
    window_tags,
)

WritePolicy = Literal["user", "all"]
ROLE_TAG = "momex:role:"
SESSION_TAG = "momex:session:"
UNCONFIRMED_TAG = "momex:unconfirmed"


def tag_value(message: Any, prefix: str) -> str | None:
    return next(
        (
            tag[len(prefix) :]
            for tag in (getattr(message, "tags", None) or [])
            if tag.startswith(prefix)
        ),
        None,
    )


def is_confirmed(message: Any) -> bool:
    return UNCONFIRMED_TAG not in (getattr(message, "tags", None) or [])


def prepare_messages(
    messages: str | Sequence[Mapping[str, object]],
    *,
    collection: str,
    timestamp: str,
    tags: list[str],
    write_policy: WritePolicy,
    source_id: str | None = None,
) -> list[ConversationMessage]:
    if write_policy not in ("user", "all"):
        raise ValueError("write_policy must be 'user' or 'all'")
    inputs = [{"content": messages}] if isinstance(messages, str) else messages
    if source_id is not None and len(inputs) != 1:
        raise ValueError(
            "source_id requires one message; use per-message IDs for batches"
        )
    prepared: list[ConversationMessage] = []
    for message in inputs:
        message_source_id = message.get("source_id", source_id)
        if source_id is not None and message_source_id != source_id:
            raise ValueError("conflicting source_id argument and message field")
        if message_source_id is not None and (
            not isinstance(message_source_id, str) or not message_source_id.strip()
        ):
            raise ValueError("source_id must be a non-empty string")
        content = message.get("content", "")
        role = message.get("role", "user")
        speaker = message.get("speaker", f"{collection}:{role}")
        session = message.get("session_id")
        confirmed = message.get("confirmed", role == "user" or write_policy == "all")
        occurred_at = message.get("timestamp", timestamp)
        valid_from = message.get(
            "valid_from",
            next(
                (
                    tag[len(VALID_FROM_TAG) :]
                    for tag in tags
                    if tag.startswith(VALID_FROM_TAG)
                ),
                None,
            ),
        )
        valid_to = message.get(
            "valid_to",
            next(
                (
                    tag[len(VALID_TO_TAG) :]
                    for tag in tags
                    if tag.startswith(VALID_TO_TAG)
                ),
                None,
            ),
        )
        if not isinstance(content, str) or not isinstance(role, str):
            raise ValueError("message content and role must be strings")
        if not isinstance(speaker, str) or not speaker:
            raise ValueError("message speaker must be a non-empty string")
        if session is not None and not isinstance(session, str):
            raise ValueError("message session_id must be a string")
        if not isinstance(confirmed, bool):
            raise ValueError("message confirmed must be a boolean")
        if not isinstance(occurred_at, str):
            raise ValueError("message timestamp must be a string")
        if valid_from is not None and not isinstance(valid_from, str):
            raise ValueError("message valid_from must be a string or None")
        if valid_to is not None and not isinstance(valid_to, str):
            raise ValueError("message valid_to must be a string or None")
        valid_from = validate_iso_date(valid_from, "valid_from")
        valid_to = validate_iso_date(valid_to, "valid_to")
        if valid_from and valid_to and valid_from > valid_to:
            raise ValueError("valid_from cannot be after valid_to")
        if not content:
            continue
        message_tags = [*window_tags(valid_from, valid_to), f"{ROLE_TAG}{role}"]
        if session is not None:
            message_tags.append(f"{SESSION_TAG}{session}")
        if not confirmed:
            message_tags.append(UNCONFIRMED_TAG)
        prepared.append(
            ConversationMessage(
                text_chunks=[content],
                metadata=ConversationMessageMeta(speaker=speaker),
                tags=message_tags,
                timestamp=validate_timestamp(occurred_at),
                source_id=message_source_id or new_source_id(),
            )
        )
    return prepared


def extraction_inputs(
    messages: list[ConversationMessage],
    previous: Sequence[Any],
    context_turns: int,
) -> list[str | None]:
    def describe(message: Any) -> dict[str, Any]:
        return {
            "speaker": message.metadata.speaker,
            "role": tag_value(message, ROLE_TAG),
            "text": " ".join(message.text_chunks),
        }

    history = list(previous)
    inputs: list[str | None] = []
    for message in messages:
        context = [
            item
            for item in history
            if tag_value(item, SESSION_TAG) == tag_value(message, SESSION_TAG)
        ]
        inputs.append(
            "Extract only facts asserted or explicitly confirmed by TARGET. "
            "Resolve first-person references using TARGET's speaker. CONTEXT is "
            "only for resolving references and short replies; do not extract "
            "its guesses, questions, quoted claims or hypothetical statements "
            "as facts asserted by TARGET.\n"
            + json.dumps(
                {
                    "CONTEXT": (
                        [describe(item) for item in context[-context_turns:]]
                        if context_turns
                        else []
                    ),
                    "TARGET": describe(message),
                },
                ensure_ascii=False,
            )
            if is_confirmed(message)
            else None
        )
        history.append(message)
    return inputs
