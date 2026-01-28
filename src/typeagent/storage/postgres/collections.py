# Copyright (c) Xiaoyu Zhang.
# Licensed under the MIT License.

"""PostgreSQL-based collection implementations."""

import json
import typing

import asyncpg

from ...knowpro import interfaces, serialization


class PostgresMessageCollection[TMessage: interfaces.IMessage](
    interfaces.IMessageCollection[TMessage]
):
    """PostgreSQL-backed message collection."""

    def __init__(
        self,
        pool: asyncpg.Pool,
        message_type: type[TMessage] | None = None,
        message_text_index: "interfaces.IMessageTextIndex[TMessage] | None" = None,
    ):
        self.pool = pool
        self.message_type = message_type
        self.message_text_index = message_text_index

    def set_message_text_index(
        self, message_text_index: "interfaces.IMessageTextIndex[TMessage]"
    ) -> None:
        """Set the message text index for automatic indexing of new messages."""
        self.message_text_index = message_text_index

    @property
    def is_persistent(self) -> bool:
        return True

    async def size(self) -> int:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT COUNT(*) FROM Messages")
            return row[0]

    def __aiter__(self) -> typing.AsyncGenerator[TMessage, None]:
        return self._async_iterator()

    async def _async_iterator(self) -> typing.AsyncGenerator[TMessage, None]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT chunks, chunk_uri, start_timestamp, tags, metadata, extra
                FROM Messages ORDER BY msg_id
                """
            )
            for row in rows:
                message = self._deserialize_message_from_row(row)
                yield message

    def _deserialize_message_from_row(self, row) -> TMessage:
        """Rehydrate a message from database row columns."""
        chunks_json = row["chunks"]
        start_timestamp = row["start_timestamp"]
        tags_json = row["tags"]
        metadata_json = row["metadata"]
        extra_json = row["extra"]

        # Parse JSON fields and build a JSON object using camelCase.
        message_data = extra_json if extra_json else {}
        message_data["textChunks"] = chunks_json if chunks_json else []
        message_data["timestamp"] = start_timestamp.isoformat() if start_timestamp else None
        message_data["tags"] = tags_json if tags_json else []
        message_data["metadata"] = metadata_json if metadata_json else {}

        if self.message_type is None:
            raise ValueError(
                "Deserialization requires message_type passed to PostgresMessageCollection"
            )
        return serialization.deserialize_object(self.message_type, message_data)

    def _serialize_message_to_row(self, message: TMessage) -> tuple:
        """Shred a message object into database columns."""
        message_data = serialization.serialize_object(message)

        chunks = message_data.pop("textChunks", [])
        chunk_uri = None
        start_timestamp = message_data.pop("timestamp", None)
        tags = message_data.pop("tags", [])
        metadata = message_data.pop("metadata", {})
        extra = message_data if message_data else None

        return (chunks, chunk_uri, start_timestamp, tags, metadata, extra)

    async def get_item(self, arg: int) -> TMessage:
        if not isinstance(arg, int):
            raise TypeError(f"Index must be an int, not {type(arg).__name__}")
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT chunks, chunk_uri, start_timestamp, tags, metadata, extra
                FROM Messages WHERE msg_id = $1
                """,
                arg,
            )
            if row:
                return self._deserialize_message_from_row(row)
            raise IndexError("Message not found")

    async def get_slice(self, start: int, stop: int) -> list[TMessage]:
        if stop <= start:
            return []
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT chunks, chunk_uri, start_timestamp, tags, metadata, extra
                FROM Messages WHERE msg_id >= $1 AND msg_id < $2 ORDER BY msg_id
                """,
                start, stop,
            )
            return [self._deserialize_message_from_row(row) for row in rows]

    async def get_multiple(self, arg: list[int]) -> list[TMessage]:
        if not arg:
            return []
        size = await self.size()
        if not all((0 <= i < size) for i in arg):
            raise IndexError("One or more Message indices are out of bounds")
        if len(arg) < 2:
            return [await self.get_item(i) for i in arg]

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT msg_id, chunks, chunk_uri, start_timestamp, tags, metadata, extra
                FROM Messages WHERE msg_id = ANY($1)
                """,
                arg,
            )
            rowdict = {}
            for row in rows:
                msg_id = row["msg_id"]
                rowdict[msg_id] = row
            return [self._deserialize_message_from_row(rowdict[i]) for i in arg]

    async def append(self, item: TMessage) -> None:
        chunks, chunk_uri, start_timestamp, tags, metadata, extra = (
            self._serialize_message_to_row(item)
        )
        msg_id = await self.size()

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO Messages (msg_id, chunks, chunk_uri, start_timestamp, tags, metadata, extra)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                msg_id, json.dumps(chunks) if chunks else None, chunk_uri,
                start_timestamp, json.dumps(tags) if tags else None,
                json.dumps(metadata) if metadata else None,
                json.dumps(extra) if extra else None,
            )

        if self.message_text_index is not None:
            await self.message_text_index.add_messages_starting_at(msg_id, [item])

    async def extend(self, items: typing.Iterable[TMessage]) -> None:
        items_list = list(items)
        if not items_list:
            return

        current_size = await self.size()

        async with self.pool.acquire() as conn:
            for msg_id, item in enumerate(items_list, current_size):
                chunks, chunk_uri, start_timestamp, tags, metadata, extra = (
                    self._serialize_message_to_row(item)
                )
                await conn.execute(
                    """
                    INSERT INTO Messages (msg_id, chunks, chunk_uri, start_timestamp, tags, metadata, extra)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    msg_id, json.dumps(chunks) if chunks else None, chunk_uri,
                    start_timestamp, json.dumps(tags) if tags else None,
                    json.dumps(metadata) if metadata else None,
                    json.dumps(extra) if extra else None,
                )

        if self.message_text_index is not None:
            await self.message_text_index.add_messages_starting_at(
                current_size, items_list
            )


class PostgresSemanticRefCollection(interfaces.ISemanticRefCollection):
    """PostgreSQL-backed semantic reference collection."""

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    def _deserialize_semantic_ref_from_row(self, row) -> interfaces.SemanticRef:
        """Deserialize a semantic ref from database row columns."""
        semref_id = row["semref_id"]
        range_json = row["range_json"]
        knowledge_type = row["knowledge_type"]
        knowledge_json = row["knowledge_json"]

        semantic_ref_data = interfaces.SemanticRefData(
            semanticRefOrdinal=semref_id,
            range=range_json,
            knowledgeType=knowledge_type,
            knowledge=knowledge_json,
        )

        return interfaces.SemanticRef.deserialize(semantic_ref_data)

    def _serialize_semantic_ref_to_row(
        self, semantic_ref: interfaces.SemanticRef
    ) -> tuple:
        """Serialize a semantic ref object into database columns."""
        semantic_ref_data = semantic_ref.serialize()

        semref_id = semantic_ref_data["semanticRefOrdinal"]
        range_json = semantic_ref_data["range"]
        knowledge_type = semantic_ref_data["knowledgeType"]
        knowledge_json = semantic_ref_data["knowledge"]

        return (semref_id, range_json, knowledge_type, knowledge_json)

    @property
    def is_persistent(self) -> bool:
        return True

    async def size(self) -> int:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT COUNT(*) FROM SemanticRefs")
            return row[0]

    async def __aiter__(self) -> typing.AsyncGenerator[interfaces.SemanticRef, None]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT semref_id, range_json, knowledge_type, knowledge_json
                FROM SemanticRefs ORDER BY semref_id
                """
            )
            for row in rows:
                yield self._deserialize_semantic_ref_from_row(row)

    async def get_item(self, arg: int) -> interfaces.SemanticRef:
        if not isinstance(arg, int):
            raise TypeError(f"Index must be an int, not {type(arg).__name__}")
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT semref_id, range_json, knowledge_type, knowledge_json
                FROM SemanticRefs WHERE semref_id = $1
                """,
                arg,
            )
            if row:
                return self._deserialize_semantic_ref_from_row(row)
            raise IndexError("SemanticRef not found")

    async def get_slice(self, start: int, stop: int) -> list[interfaces.SemanticRef]:
        if stop <= start:
            return []
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT semref_id, range_json, knowledge_type, knowledge_json
                FROM SemanticRefs WHERE semref_id >= $1 AND semref_id < $2
                ORDER BY semref_id
                """,
                start, stop,
            )
            return [self._deserialize_semantic_ref_from_row(row) for row in rows]

    async def get_multiple(self, arg: list[int]) -> list[interfaces.SemanticRef]:
        if not arg:
            return []
        size = await self.size()
        if not all((0 <= i < size) for i in arg):
            raise IndexError("One or more SemanticRef indices are out of bounds")
        if len(arg) < 2:
            return [await self.get_item(ordinal) for ordinal in arg]

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT semref_id, range_json, knowledge_type, knowledge_json
                FROM SemanticRefs WHERE semref_id = ANY($1)
                """,
                arg,
            )
            rowdict = {row["semref_id"]: row for row in rows}
            return [self._deserialize_semantic_ref_from_row(rowdict[ordl]) for ordl in arg]

    async def append(self, item: interfaces.SemanticRef) -> None:
        semref_id, range_json, knowledge_type, knowledge_json = (
            self._serialize_semantic_ref_to_row(item)
        )
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO SemanticRefs (semref_id, range_json, knowledge_type, knowledge_json)
                VALUES ($1, $2, $3, $4)
                """,
                semref_id, json.dumps(range_json), knowledge_type, json.dumps(knowledge_json),
            )

    async def extend(self, items: typing.Iterable[interfaces.SemanticRef]) -> None:
        items_list = list(items)
        if not items_list:
            return

        async with self.pool.acquire() as conn:
            for item in items_list:
                semref_id, range_json, knowledge_type, knowledge_json = (
                    self._serialize_semantic_ref_to_row(item)
                )
                await conn.execute(
                    """
                    INSERT INTO SemanticRefs (semref_id, range_json, knowledge_type, knowledge_json)
                    VALUES ($1, $2, $3, $4)
                    """,
                    semref_id, json.dumps(range_json), knowledge_type, json.dumps(knowledge_json),
                )
