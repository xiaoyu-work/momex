"""Stable identity for an individual memory.

The supersession ledger has to name the memories it hides, and until now it
named them by `semantic_ref_ordinal` -- a position in a collection. Positions
are only meaningful while nothing shifts them, and upstream's own TODO lists
"fix all bugs related to ordinals/ids relying on starting at 0 and no gaps
(prepare for deletions)" as outstanding work. If ordinals are ever compacted or
rebuilt, every ledger entry silently starts hiding a different memory, and
because the entries carry only an integer there is no way to detect it or
repair it.

SemanticRef itself has no identity to borrow -- it is (ordinal, range,
knowledge) and nothing else. Messages do: `IMessage.source_id` is an existing,
serialized upstream field that Momex simply never set. So a memory is named by
the message it came from plus what it says:

    memory_id = H(source_id, knowledge_type, canonical(knowledge))

Both halves survive reindexing. The message is reached through
`sem_ref.range.start.message_ordinal`, which is resolved when the memory is
read rather than stored, so it is correct whatever the ordinals currently are.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any
import uuid

# Length of the hex digest kept. 16 hex characters is 64 bits: collision
# becomes likely around 2**32 memories in one collection, which is far past the
# point where the single-blob ledger would have failed for other reasons.
_ID_LENGTH = 16

_FIELD_SEPARATOR = "\x1f"


def new_source_id() -> str:
    """Mint an identifier for a message Momex is about to write."""
    return uuid.uuid4().hex


def canonical_knowledge(knowledge: Any) -> str:
    """Render the *meaning* of a knowledge object as a stable string.

    Only the fields that make the assertion what it is are included, because
    everything included here is something that changes the id when it changes:

      - Rendered display text is excluded. It is a presentation concern, and
        editing how an entity is formatted would otherwise rename every memory
        in every collection.
      - Entity aliases are excluded. They accumulate as more is learned about
        an entity, and learning a nickname does not make it a different memory.
      - Verb tense is included. "liked" and "likes" are different claims.
    """
    payload: dict[str, Any]

    knowledge_type = getattr(knowledge, "knowledge_type", None)

    if knowledge_type == "entity":
        payload = {
            "name": getattr(knowledge, "name", None),
            "type": sorted(getattr(knowledge, "type", None) or []),
            "facets": sorted(
                f"{f.name}={f.value}"
                for f in (getattr(knowledge, "facets", None) or [])
            ),
        }
    elif knowledge_type == "action":
        facet = getattr(knowledge, "subject_entity_facet", None)
        payload = {
            "verbs": list(getattr(knowledge, "verbs", None) or []),
            "tense": getattr(knowledge, "verb_tense", None),
            "subject": getattr(knowledge, "subject_entity_name", None),
            "object": getattr(knowledge, "object_entity_name", None),
            "indirect_object": getattr(knowledge, "indirect_object_entity_name", None),
            "subject_facet": f"{facet.name}={facet.value}" if facet else None,
        }
    elif knowledge_type == "topic":
        payload = {"text": getattr(knowledge, "text", None)}
    else:
        # An unrecognised knowledge type still gets an id rather than none, so
        # a new upstream type degrades to "identified by its repr" instead of
        # dropping out of the ledger.
        payload = {"repr": repr(knowledge)}

    return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)


def memory_id(source_id: str | None, knowledge: Any) -> str | None:
    """Derive the stable id of one memory, or None if it cannot have one.

    Returns None when the source message carries no `source_id` -- collections
    written before this existed, or messages ingested by the upstream tools.
    Those memories keep working; they just cannot be reconciled if ordinals
    move, which is exactly the situation that already applied to all of them.

    Two extractions of the same claim from the same message collapse to one id.
    That is intended: they are the same memory recorded twice, and the ledger
    hiding both together is the behaviour a reader would expect.
    """
    if not source_id:
        return None

    knowledge_type = getattr(knowledge, "knowledge_type", None) or "unknown"
    payload = _FIELD_SEPARATOR.join(
        (source_id, str(knowledge_type), canonical_knowledge(knowledge))
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:_ID_LENGTH]
