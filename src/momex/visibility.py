"""One visibility policy for knowledge, source messages and their neighbors."""

from dataclasses import dataclass, field
from typing import Any, Literal

from .attribution import is_confirmed
from .paths import utc_now
from .timewindow import (
    extract_time_window,
    is_expired,
    is_not_yet_active,
    normalize_as_of,
)

MemoryStatus = Literal["current", "superseded", "expired", "future", "unconfirmed"]


@dataclass
class SearchView:
    superseded_knowledge: set[int] = field(default_factory=set)
    superseded_messages: set[int] = field(default_factory=set)
    include_superseded: bool = False
    include_expired: bool = False
    include_unconfirmed: bool = False
    as_of: str | None = None
    collection: str | None = None
    _now: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        self.as_of = normalize_as_of(self.as_of)

    @property
    def reference_time(self) -> str:
        return self.as_of or self._now

    def occurred_later(self, message: Any) -> bool:
        timestamp = getattr(message, "timestamp", None)
        return bool(timestamp and timestamp > self.reference_time)

    def status(self, message: Any, *, superseded: bool = False) -> MemoryStatus:
        if superseded:
            return "superseded"
        if not is_confirmed(message):
            return "unconfirmed"
        if self.occurred_later(message):
            return "future"
        valid_from, valid_to = extract_time_window(message)
        if is_expired(valid_to, self.reference_time):
            return "expired"
        if is_not_yet_active(valid_from, self.reference_time):
            return "future"
        return "current"

    def allows(self, message: Any, *, superseded: bool = False) -> bool:
        if self.as_of is not None and self.occurred_later(message):
            return False
        if superseded and not self.include_superseded:
            return False
        if not is_confirmed(message) and not self.include_unconfirmed:
            return False
        valid_from, valid_to = extract_time_window(message)
        return self.include_expired or not (
            self.occurred_later(message)
            or is_expired(valid_to, self.reference_time)
            or is_not_yet_active(valid_from, self.reference_time)
        )
