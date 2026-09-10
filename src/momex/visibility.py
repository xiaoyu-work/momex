"""One visibility policy for knowledge, source messages and their neighbors."""

from dataclasses import dataclass, field
from typing import Any, Literal

from .attribution import is_confirmed
from .timewindow import extract_time_window, is_expired, is_not_yet_active

MemoryStatus = Literal["current", "superseded", "expired", "future", "unconfirmed"]


@dataclass
class SearchView:
    superseded_knowledge: set[int] = field(default_factory=set)
    superseded_messages: set[int] = field(default_factory=set)
    include_superseded: bool = False
    include_expired: bool = False
    include_unconfirmed: bool = False

    def status(self, message: Any, *, superseded: bool = False) -> MemoryStatus:
        if superseded:
            return "superseded"
        if not is_confirmed(message):
            return "unconfirmed"
        valid_from, valid_to = extract_time_window(message)
        if is_expired(valid_to):
            return "expired"
        if is_not_yet_active(valid_from):
            return "future"
        return "current"

    def allows(self, message: Any, *, superseded: bool = False) -> bool:
        if superseded and not self.include_superseded:
            return False
        if not is_confirmed(message) and not self.include_unconfirmed:
            return False
        valid_from, valid_to = extract_time_window(message)
        return self.include_expired or not (
            is_expired(valid_to) or is_not_yet_active(valid_from)
        )
