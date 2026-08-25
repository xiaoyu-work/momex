"""The values Momex hands back to callers.

These are the whole public result surface: what add() reports, what search()
returns, and what the supersession ledger records. They are kept apart from the
code that produces them so that reading the API does not mean reading the
engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SupersededRecord:
    """One entry in the append-only supersession ledger.

    A memory is never destroyed. It is marked as superseded by whatever
    replaced it, hidden from search, and can be restored -- so a bad
    contradiction judgment is recoverable and the change itself is preserved.
    """

    ordinal: int
    """Semantic-ref ordinal of the memory that was retired."""

    superseded_by: list[int]
    """Ordinals of the memories that replaced it. Empty for explicit delete()."""

    at: str
    """ISO-8601 UTC timestamp of the supersession."""

    reason: str
    """One of "contradiction", "delete", or "legacy"."""

    text: str | None = None
    """Rendered text at the time of supersession, for auditing."""

    query: str | None = None
    """The content or query that triggered it."""

    restored_at: str | None = None
    """Set when restore() reversed this record. Non-None means inactive."""

    memory_id: str | None = None
    """Stable identity of the retired memory; see momex.identity.

    `ordinal` is a position, and positions move. This does not, so an entry
    written today can still be matched to its memory after a reindex. None for
    entries written before this existed, and for memories whose source message
    has no source_id.
    """

    @property
    def active(self) -> bool:
        return self.restored_at is None


@dataclass
class AddResult:
    """Result of adding memories."""

    messages_added: int
    entities_extracted: int
    contradictions_removed: int = 0
    collections: list[str] | None = None
    superseded: list[SupersededRecord] | None = None
    """What add() retired, not just how many. None when nothing was retired."""


@dataclass
class SearchItem:
    """A single search result item."""

    type: str  # Uses TypeAgent's native knowledge_type: "entity", "action", "topic", "message"
    text: str
    score: float  # Native score of the index that produced this item
    raw: Any  # Original TypeAgent object (SemanticRef or Message)
    timestamp: str | None = None  # When the memory was recorded (ISO format)
    valid_from: str | None = None
    valid_to: str | None = None
    # Rank-fusion score used to order hybrid search() results. None when the
    # item comes from a single-path search such as search_by_embedding().
    fusion_score: float | None = None
    # Stable identity of this memory (see momex.identity). None for messages,
    # and for knowledge whose source message predates source_id being set.
    memory_id: str | None = None
    # Position of the source message in the collection, counting from zero.
    # Lets a caller locate a result in the conversation it came from -- to show
    # what was said around it, or to order results by when they were said
    # rather than by score. None when the item did not come from a message.
    ordinal: int | None = None
