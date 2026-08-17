"""Validity windows attached to a memory.

A memory can carry a `valid_from`/`valid_to` pair, so that "Netflix renews on
the 1st" stops being current once the date passes. The window is stored on the
source message as tags, and every comparison here is lexicographic against a
`YYYY-MM-DD` string, which is only correct for zero-padded ISO dates -- hence
the validation on the way in.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

VALID_FROM_TAG = "valid_from:"
VALID_TO_TAG = "valid_to:"


def validate_iso_date(value: str | None, field: str) -> str | None:
    """Normalize an ISO date, or raise if it cannot be compared safely.

    The window checks compare these values lexicographically against
    `YYYY-MM-DD`, which is only correct for zero-padded ISO dates. An
    unpadded string like "2026-4-1" would sort *after* "2026-08-17" and
    silently never expire, so reject it at write time instead.
    """
    if value is None:
        return None
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field} must be an ISO date string (YYYY-MM-DD); got {value!r}"
        ) from exc
    return parsed.isoformat()


def window_tags(valid_from: str | None, valid_to: str | None) -> list[str]:
    """Render a window as message tags, so it survives serialization."""
    tags: list[str] = []
    if valid_from:
        tags.append(f"{VALID_FROM_TAG}{valid_from}")
    if valid_to:
        tags.append(f"{VALID_TO_TAG}{valid_to}")
    return tags


def extract_time_window(msg: Any) -> tuple[str | None, str | None]:
    """Extract valid_from/valid_to from a message's tags."""
    tags = getattr(msg, "tags", None) or []
    valid_from = None
    valid_to = None
    for tag in tags:
        if isinstance(tag, str):
            if tag.startswith(VALID_FROM_TAG):
                valid_from = tag[len(VALID_FROM_TAG) :]
            elif tag.startswith(VALID_TO_TAG):
                valid_to = tag[len(VALID_TO_TAG) :]
    return valid_from, valid_to


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def is_expired(valid_to: str | None) -> bool:
    """Check if a time window has expired (valid_to < today UTC)."""
    if not valid_to:
        return False
    return valid_to < _today()


def is_not_yet_active(valid_from: str | None) -> bool:
    """Check if a time window has not opened yet (valid_from > today UTC)."""
    if not valid_from:
        return False
    return valid_from > _today()


def is_outside_window(valid_from: str | None, valid_to: str | None) -> bool:
    """True when today falls outside [valid_from, valid_to]."""
    return is_expired(valid_to) or is_not_yet_active(valid_from)
