# Copyright (c) Xiaoyu Zhang.
# Licensed under the MIT License.

"""PostgreSQL-based timestamp index implementation."""

from datetime import datetime, timezone

import asyncpg

from ...knowpro import interfaces


def _to_datetime(value) -> datetime | None:
    """Convert a timestamp string or datetime to a datetime object for asyncpg."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


class PostgresTimestampToTextRangeIndex(interfaces.ITimestampToTextRangeIndex):
    """PostgreSQL-based timestamp index that queries Messages table directly."""

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def size(self) -> int:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT COUNT(*) FROM Messages WHERE start_timestamp IS NOT NULL"
            )
            return row[0]

    async def add_timestamp(
        self, message_ordinal: interfaces.MessageOrdinal, timestamp: str
    ) -> bool:
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE Messages SET start_timestamp = $1 WHERE msg_id = $2",
                _to_datetime(timestamp), message_ordinal,
            )
            return "UPDATE 1" in result

    async def get_timestamp_ranges(
        self, start_timestamp: str, end_timestamp: str | None = None
    ) -> list[interfaces.TimestampedTextRange]:
        """Get timestamp ranges from Messages table."""
        async with self.pool.acquire() as conn:
            if end_timestamp is None:
                rows = await conn.fetch(
                    """
                    SELECT msg_id, start_timestamp
                    FROM Messages
                    WHERE start_timestamp = $1
                    ORDER BY msg_id
                    """,
                    _to_datetime(start_timestamp),
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT msg_id, start_timestamp
                    FROM Messages
                    WHERE start_timestamp >= $1 AND start_timestamp <= $2
                    ORDER BY msg_id
                    """,
                    _to_datetime(start_timestamp), _to_datetime(end_timestamp),
                )

            results = []
            for row in rows:
                msg_id, timestamp = row[0], row[1]
                text_range = interfaces.TextRange(
                    start=interfaces.TextLocation(message_ordinal=msg_id, chunk_ordinal=0)
                )
                results.append(
                    interfaces.TimestampedTextRange(
                        range=text_range,
                        timestamp=timestamp.isoformat() if timestamp else None
                    )
                )

            return results

    async def add_timestamps(
        self, message_timestamps: list[tuple[interfaces.MessageOrdinal, str]]
    ) -> None:
        """Add multiple timestamps."""
        async with self.pool.acquire() as conn:
            for message_ordinal, timestamp_str in message_timestamps:
                await conn.execute(
                    "UPDATE Messages SET start_timestamp = $1 WHERE msg_id = $2",
                    _to_datetime(timestamp_str), message_ordinal,
                )

    async def lookup_range(
        self, date_range: interfaces.DateRange
    ) -> list[interfaces.TimestampedTextRange]:
        """Lookup messages in a date range."""
        start_dt = date_range.start if isinstance(date_range.start, datetime) else _to_datetime(str(date_range.start))
        end_dt = date_range.end if isinstance(date_range.end, datetime) else _to_datetime(str(date_range.end)) if date_range.end else None

        async with self.pool.acquire() as conn:
            if end_dt is None:
                rows = await conn.fetch(
                    """
                    SELECT msg_id, start_timestamp, chunks
                    FROM Messages
                    WHERE start_timestamp = $1
                    ORDER BY msg_id
                    """,
                    start_dt,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT msg_id, start_timestamp, chunks
                    FROM Messages
                    WHERE start_timestamp >= $1 AND start_timestamp < $2
                    ORDER BY msg_id
                    """,
                    start_dt, end_dt,
                )

            results = []
            for row in rows:
                msg_id, timestamp = row[0], row[1]
                text_location = interfaces.TextLocation(
                    message_ordinal=msg_id, chunk_ordinal=0
                )
                text_range = interfaces.TextRange(start=text_location, end=None)
                results.append(
                    interfaces.TimestampedTextRange(
                        timestamp=timestamp.isoformat() if timestamp else None,
                        range=text_range
                    )
                )

            return results
