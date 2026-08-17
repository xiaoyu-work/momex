"""Mapping collection names onto storage locations.

A collection name like "momex:engineering:xiaoyuzhang" is hierarchical, and
each backend renders that hierarchy differently: SQLite as nested directories,
PostgreSQL as a single flattened schema name. Both derivations live here,
because both are constrained by the same thing -- the name is caller-supplied,
and in a multi-tenant deployment that means user-supplied.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re

from .exceptions import ValidationError

# Characters that must never reach the filesystem as part of a path segment.
# Both separators are included: a collection segment is always exactly one
# directory, so "a/b" must not silently become two of them.
_UNSAFE_PATH_CHARS = re.compile(r'[<>"|?*:\\/]')


def utc_now() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sanitize_collection_part(part: str, collection: str) -> str:
    """Map one ':'-delimited collection segment to a safe path component.

    Segments come from the caller, and in a multi-tenant deployment that
    usually means from user input. A segment of "." or ".." would resolve
    *outside* the storage directory, so those are rejected rather than
    sanitized: silently rewriting them would let two different tenants land on
    the same directory.
    """
    if not part.strip(". \t\r\n"):
        raise ValidationError(
            message=(
                f"Invalid collection name {collection!r}: segment {part!r} is "
                "empty or consists only of dots/whitespace."
            ),
            field="collection",
            value=collection,
            suggestion="Use non-empty segments separated by ':', e.g. 'user:alice'.",
        )
    return _UNSAFE_PATH_CHARS.sub("_", part)


def collection_to_path(collection: str) -> Path:
    """Convert a collection name to a relative path, one segment per ':'.

    Converts "user:xiaoyuzhang" to Path("user/xiaoyuzhang").
    """
    parts = [
        sanitize_collection_part(part, collection) for part in collection.split(":")
    ]
    return Path(*parts)


def path_to_collection(path: Path) -> str:
    """Convert a relative path back to a collection name.

    Converts Path("user/xiaoyuzhang") to "user:xiaoyuzhang".
    """
    return ":".join(path.parts)


def collection_to_db_path(collection: str, base_path: str, db_name: str) -> Path:
    """Convert collection name to database path.

    Converts "momex:engineering:xiaoyuzhang" to
    Path("base_path/momex/engineering/xiaoyuzhang/db_name")
    """
    return Path(base_path) / collection_to_path(collection) / db_name


def collection_to_schema(collection: str) -> str:
    """Convert collection name to a PostgreSQL-safe schema name."""
    base = re.sub(r"[^a-zA-Z0-9_]", "_", collection).lower()
    if not base:
        base = "momex"
    if base[0].isdigit():
        base = f"c_{base}"

    max_len = 63
    if len(base) <= max_len:
        return base

    digest = hashlib.md5(collection.encode("utf-8")).hexdigest()[:8]
    return f"{base[:54]}_{digest}"
