# Copyright (c) Xiaoyu Zhang.
# Licensed under the MIT License.

"""PostgreSQL-based timestamp index implementation."""

import asyncpg

from ...knowpro import interfaces
from ...knowpro.universal_message import format_timestamp_utc


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
                timestamp, message_ordinal,
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
                    start_timestamp,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT msg_id, start_timestamp
                    FROM Messages
                    WHERE start_timestamp >= $1 AND start_timestamp <= $2
                    ORDER BY msg_id
                    """,
                    start_timestamp, end_timestamp,
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
        from datetime import datetime

        async with self.pool.acquire() as conn:
            for message_ordinal, timestamp_str in message_timestamps:
                # Convert string to datetime for asyncpg
                timestamp = None
                if timestamp_str:
                    if isinstance(timestamp_str, str):
                        if timestamp_str.endswith("Z"):
                            timestamp_str = timestamp_str[:-1] + "+00:00"
                        try:
                            timestamp = datetime.fromisoformat(timestamp_str)
                        except ValueError:
                            pass
                    elif isinstance(timestamp_str, datetime):
                        timestamp = timestamp_str

                await conn.execute(
                    "UPDATE Messages SET start_timestamp = $1 WHERE msg_id = $2",
                    timestamp, message_ordinal,
                )

    async def lookup_range(
        self, date_range: interfaces.DateRange
    ) -> list[interfaces.TimestampedTextRange]:
        """Lookup messages in a date range."""
        start_timestamp = format_timestamp_utc(date_range.start)
        end_timestamp = format_timestamp_utc(date_range.end) if date_range.end else None

        async with self.pool.acquire() as conn:
            if date_range.end is None:
                rows = await conn.fetch(
                    """
                    SELECT msg_id, start_timestamp, chunks
                    FROM Messages
                    WHERE start_timestamp = $1
                    ORDER BY msg_id
                    """,
                    start_timestamp,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT msg_id, start_timestamp, chunks
                    FROM Messages
                    WHERE start_timestamp >= $1 AND start_timestamp < $2
                    ORDER BY msg_id
                    """,
                    start_timestamp, end_timestamp,
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
