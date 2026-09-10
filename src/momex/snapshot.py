"""Portable snapshots of all persistent Momex data, without re-extraction."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Literal

import numpy as np

from pydantic import BaseModel, ConfigDict, JsonValue, model_validator, TypeAdapter

from typeagent.knowpro.interfaces import (
    KnowledgeType,
    SemanticRef,
    TextRange,
    TextRangeData,
)
from typeagent.knowpro.serialization import deserialize_knowledge, deserialize_object
from typeagent.knowpro.universal_message import ConversationMessage
from typeagent.storage.sqlite.provider import SqliteStorageProvider

from .ledger import (
    decode_deleted_ids,
    DELETED_SEMREFS_METADATA_KEY,
    encode_deleted_ids,
    encode_ledger,
    SUPERSESSION_LEDGER_VERSION,
    SUPERSESSION_METADATA_KEY,
)
from .results import SupersededRecord

TABLES: dict[str, tuple[str, ...]] = {
    "Messages": (
        "msg_id",
        "chunks",
        "chunk_uri",
        "start_timestamp",
        "tags",
        "metadata",
        "extra",
    ),
    "SemanticRefs": ("semref_id", "range_json", "knowledge_type", "knowledge_json"),
    "SemanticRefIndex": ("term", "semref_id"),
    "MessageTextIndex": ("msg_id", "chunk_ordinal", "embedding"),
    "PropertyIndex": ("prop_name", "value_str", "score", "semref_id"),
    "RelatedTermsAliases": ("term", "alias"),
    "RelatedTermsFuzzy": ("term", "term_embedding"),
    "ConversationMetadata": ("key", "value"),
    "IngestedSources": ("source_id", "status"),
    "ChunkFailures": (
        "msg_id",
        "chunk_ordinal",
        "error_class",
        "error_message",
        "failed_at",
    ),
}
JSON_COLUMNS = {"chunks", "tags", "metadata", "extra", "range_json", "knowledge_json"}
VECTOR_COLUMNS = {"embedding", "term_embedding"}
TIME_COLUMNS = {"start_timestamp", "failed_at"}
LEDGER_ADAPTER = TypeAdapter(list[SupersededRecord])
RANGE_ADAPTER = TypeAdapter(TextRangeData)
KNOWLEDGE_TYPE_ADAPTER = TypeAdapter(KnowledgeType)


def ordinal(value: JsonValue) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("snapshot ordinals must be non-negative integers")
    return value


def object_value(value: JsonValue) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise ValueError("snapshot field must be an object")
    return value


def text_value(value: JsonValue) -> str:
    if not isinstance(value, str):
        raise ValueError("snapshot field must be text")
    return value


def finite_number(value: JsonValue) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def ledger_records(value: JsonValue) -> list[SupersededRecord]:
    if not isinstance(value, str):
        raise ValueError("ledger metadata must be JSON text")
    payload = json.loads(value)
    if (
        not isinstance(payload, dict)
        or payload.get("version") != SUPERSESSION_LEDGER_VERSION
    ):
        raise ValueError("unsupported ledger version in snapshot")
    return LEDGER_ADAPTER.validate_json(json.dumps(payload["records"]), strict=True)


class Backup(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    format: Literal["momex-backup"] = "momex-backup"
    version: Literal[1] = 1
    collection: str
    embedding_model: str | None = None
    embedding_dimensions: int | None = None
    tables: dict[str, list[dict[str, JsonValue]]]

    @model_validator(mode="after")
    def validate_data(self) -> "Backup":
        if set(self.tables) != set(TABLES):
            raise ValueError("snapshot must contain exactly the supported tables")
        for table, rows in self.tables.items():
            if any(set(row) != set(TABLES[table]) for row in rows):
                raise ValueError(f"invalid columns for snapshot table {table}")
            for row in rows:
                for column, value in row.items():
                    if column in ("msg_id", "semref_id", "chunk_ordinal"):
                        ordinal(value)
                    elif column == "score":
                        if not finite_number(value):
                            raise ValueError("snapshot score must be finite")
                    elif column not in JSON_COLUMNS | VECTOR_COLUMNS:
                        if value is None and column in ("chunk_uri", "start_timestamp"):
                            continue
                        if not isinstance(value, str):
                            raise ValueError(f"{table}.{column} must be text")
        messages = self.tables["Messages"]
        refs = self.tables["SemanticRefs"]
        for rows, key in ((messages, "msg_id"), (refs, "semref_id")):
            if [ordinal(row[key]) for row in rows] != list(range(len(rows))):
                raise ValueError(f"{key} must be dense and ordered")

        for row in messages:
            data = dict(object_value(row["extra"])) if row["extra"] is not None else {}
            data.update(
                {
                    "textChunks": row["chunks"],
                    "timestamp": row["start_timestamp"],
                    "tags": row["tags"] or [],
                    "metadata": row["metadata"] or {},
                }
            )
            deserialize_object(ConversationMessage, data)

        for row in refs:
            ref = SemanticRef(
                semantic_ref_ordinal=ordinal(row["semref_id"]),
                range=TextRange.deserialize(
                    RANGE_ADAPTER.validate_python(row["range_json"])
                ),
                knowledge=deserialize_knowledge(
                    KNOWLEDGE_TYPE_ADAPTER.validate_python(row["knowledge_type"]),
                    object_value(row["knowledge_json"]),
                ),
            )
            if not 0 <= ref.range.start.message_ordinal < len(messages):
                raise ValueError("semantic reference points outside snapshot messages")
            if ref.range.end and not 0 <= ref.range.end.message_ordinal <= len(
                messages
            ):
                raise ValueError(
                    "semantic reference end points outside snapshot messages"
                )

        metadata = {
            row["key"]: row["value"] for row in self.tables["ConversationMetadata"]
        }
        dimension = self.embedding_dimensions
        if (
            "embedding_size" in metadata
            and int(str(metadata["embedding_size"])) != dimension
        ):
            raise ValueError("snapshot embedding metadata disagrees with its header")
        if metadata.get("embedding_name") not in (None, self.embedding_model):
            raise ValueError("snapshot embedding model disagrees with its header")
        for table, rows in self.tables.items():
            for row in rows:
                if "semref_id" in row and ordinal(row["semref_id"]) >= len(refs):
                    raise ValueError(f"dangling knowledge index in {table}")
                if "msg_id" in row and ordinal(row["msg_id"]) >= len(messages):
                    raise ValueError(f"dangling message index in {table}")
                if "chunk_ordinal" in row:
                    chunks = messages[ordinal(row["msg_id"])]["chunks"]
                    if not isinstance(chunks, list) or ordinal(
                        row["chunk_ordinal"]
                    ) >= len(chunks):
                        raise ValueError(
                            "snapshot chunk points outside its source message"
                        )
                for column in VECTOR_COLUMNS & row.keys():
                    vector = row[column]
                    if (
                        not isinstance(vector, list)
                        or dimension is None
                        or not self.embedding_model
                        or len(vector) != dimension
                        or any(not finite_number(v) for v in vector)
                    ):
                        raise ValueError("invalid embedding in snapshot")
        for row in self.tables["ConversationMetadata"]:
            if row["key"] == SUPERSESSION_METADATA_KEY:
                for record in ledger_records(row["value"]):
                    if record.ordinal not in range(len(refs)) or any(
                        value not in range(len(refs)) for value in record.superseded_by
                    ):
                        raise ValueError("ledger refers to missing snapshot knowledge")
        return self


@asynccontextmanager
async def connection(storage: Any, *, write: bool) -> AsyncIterator[tuple[Any, bool]]:
    if isinstance(storage, SqliteStorageProvider):
        db = storage.db
        if db.in_transaction:
            raise RuntimeError("snapshot requires its own storage transaction")
        db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
        try:
            yield db, True
        except BaseException:
            db.rollback()
            raise
        else:
            db.commit()
    else:
        from typeagent.storage.postgres.provider import PostgresStorageProvider

        if not isinstance(storage, PostgresStorageProvider):
            raise TypeError("snapshots require SQLite or PostgreSQL storage")
        async with storage.pool.transaction(isolation="repeatable_read") as conn:
            if write:
                await conn.execute(
                    "LOCK TABLE Messages, SemanticRefs IN SHARE ROW EXCLUSIVE MODE"
                )
            yield conn, False


async def read_tables(conn: Any, sqlite: bool, collection: str) -> Backup:
    tables: dict[str, list[dict[str, JsonValue]]] = {}
    for table, columns in TABLES.items():
        query = f"SELECT {', '.join(columns)} FROM {table} ORDER BY 1"
        rows = conn.execute(query).fetchall() if sqlite else await conn.fetch(query)
        values: list[dict[str, JsonValue]] = []
        for row in rows:
            converted: dict[str, JsonValue] = {}
            for column, value in zip(columns, row, strict=True):
                if (
                    value is not None
                    and column in JSON_COLUMNS
                    and isinstance(value, str)
                ):
                    value = json.loads(value)
                elif value is not None and column in VECTOR_COLUMNS:
                    value = (
                        np.frombuffer(value, dtype=np.float32).tolist()
                        if sqlite
                        else (
                            json.loads(value) if isinstance(value, str) else list(value)
                        )
                    )
                elif isinstance(value, datetime):
                    value = (
                        value.astimezone(timezone.utc)
                        .isoformat()
                        .replace("+00:00", "Z")
                    )
                converted[column] = value
            values.append(converted)
        tables[table] = values
    metadata = {
        text_value(row["key"]): row["value"] for row in tables["ConversationMetadata"]
    }
    vectors = [
        value
        for table in ("MessageTextIndex", "RelatedTermsFuzzy")
        for row in tables[table]
        for column, value in row.items()
        if column in VECTOR_COLUMNS and isinstance(value, list)
    ]
    dimensions = (
        int(str(metadata["embedding_size"]))
        if "embedding_size" in metadata
        else len(vectors[0]) if vectors else None
    )
    model = metadata.get("embedding_name")
    return Backup(
        collection=collection,
        tables=tables,
        embedding_dimensions=dimensions,
        embedding_model=text_value(model) if model is not None else None,
    )


async def write_tables(
    conn: Any, sqlite: bool, backup: Backup, collection: str
) -> None:
    for table in reversed(TABLES):
        statement = f"DELETE FROM {table}"
        if sqlite:
            conn.execute(statement)
        else:
            await conn.execute(statement)

    for table, columns in TABLES.items():
        insert_columns = list(columns)
        if sqlite and table == "MessageTextIndex":
            insert_columns.append("index_position")
        placeholders = ", ".join(
            "?" if sqlite else f"${index + 1}" for index in range(len(insert_columns))
        )
        statement = (
            f"INSERT INTO {table} ({', '.join(insert_columns)}) VALUES ({placeholders})"
        )
        records: list[list[Any]] = []
        for position, row in enumerate(backup.tables[table]):
            values: list[Any] = []
            for column in columns:
                value = row[column]
                if (
                    table == "ConversationMetadata"
                    and column == "value"
                    and (
                        row["key"] == "name_tag"
                        or row["key"] == "tag"
                        and value == backup.collection
                    )
                ):
                    value = collection
                if value is not None and column in JSON_COLUMNS:
                    value = json.dumps(value, ensure_ascii=False)
                elif value is not None and column in VECTOR_COLUMNS:
                    value = (
                        np.array(value, dtype=np.float32).tobytes()
                        if sqlite
                        else json.dumps(value)
                    )
                elif not sqlite and value is not None and column in TIME_COLUMNS:
                    value = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
                values.append(value)
            if sqlite and table == "MessageTextIndex":
                values.append(position)
            records.append(values)
        if records:
            if sqlite:
                conn.executemany(statement, records)
            else:
                await conn.executemany(statement, records)

    if not sqlite:
        from typeagent.storage.postgres.schema import (
            ensure_message_text_embedding_index,
            ensure_related_terms_embedding_index,
        )

        await ensure_message_text_embedding_index(conn)
        await ensure_related_terms_embedding_index(conn)


def without_source(backup: Backup, source_id: str) -> tuple[Backup, int]:
    data = backup.model_copy(deep=True)
    tables = data.tables
    removed_messages = {
        ordinal(row["msg_id"])
        for row in tables["Messages"]
        if row["extra"] is not None
        and object_value(row["extra"]).get("source_id") == source_id
    }
    if not removed_messages:
        return backup, 0
    kept_messages = [
        row
        for row in tables["Messages"]
        if ordinal(row["msg_id"]) not in removed_messages
    ]
    message_map: dict[int, int] = {}
    kept_count = 0
    for index in range(len(tables["Messages"]) + 1):
        message_map[index] = kept_count
        if index not in removed_messages:
            kept_count += 1
    removed_refs: set[int] = set()
    for row in tables["SemanticRefs"]:
        span = object_value(row["range_json"])
        start = ordinal(object_value(span["start"])["messageOrdinal"])
        end = (
            ordinal(object_value(span["end"])["messageOrdinal"])
            if span.get("end")
            else start
        )
        if span.get("end"):
            endpoint = object_value(span["end"])
            if not endpoint.get("chunkOrdinal") and not endpoint.get("charOrdinal"):
                end -= 1
        if start in removed_messages or any(
            start <= value <= end for value in removed_messages
        ):
            removed_refs.add(ordinal(row["semref_id"]))
    kept_refs = [
        row
        for row in tables["SemanticRefs"]
        if ordinal(row["semref_id"]) not in removed_refs
    ]
    ref_map = {ordinal(row["semref_id"]): index for index, row in enumerate(kept_refs)}
    tables["Messages"] = kept_messages
    tables["SemanticRefs"] = kept_refs
    for table, rows in list(tables.items()):
        if table not in ("Messages", "SemanticRefs"):
            rows = [
                row
                for row in rows
                if (
                    "msg_id" not in row
                    or ordinal(row["msg_id"]) not in removed_messages
                )
                and (
                    "semref_id" not in row
                    or ordinal(row["semref_id"]) not in removed_refs
                )
            ]
            tables[table] = rows
        for row in rows:
            if "msg_id" in row:
                row["msg_id"] = message_map[ordinal(row["msg_id"])]
            if "semref_id" in row:
                row["semref_id"] = ref_map[ordinal(row["semref_id"])]
    for row in kept_refs:
        span = object_value(row["range_json"])
        for key in ("start", "end"):
            if span.get(key) is not None:
                point = object_value(span[key])
                point["messageOrdinal"] = message_map[ordinal(point["messageOrdinal"])]

    live_terms = {text_value(row["term"]) for row in tables["SemanticRefIndex"]}
    tables["RelatedTermsFuzzy"] = [
        row for row in tables["RelatedTermsFuzzy"] if row["term"] in live_terms
    ]
    tables["RelatedTermsAliases"] = [
        row
        for row in tables["RelatedTermsAliases"]
        if row["term"] in live_terms and row["alias"] in live_terms
    ]
    tables["IngestedSources"] = [
        row for row in tables["IngestedSources"] if row["source_id"] != source_id
    ]
    for row in tables["ConversationMetadata"]:
        if row["key"] == SUPERSESSION_METADATA_KEY:
            records: list[SupersededRecord] = []
            for record in ledger_records(row["value"]):
                if record.ordinal in removed_refs:
                    continue
                record.ordinal = ref_map[record.ordinal]
                if any(value in removed_refs for value in record.superseded_by):
                    record.query = None
                record.superseded_by = [
                    ref_map[value]
                    for value in record.superseded_by
                    if value not in removed_refs
                ]
                records.append(record)
            row["value"] = json.dumps(encode_ledger(records))
        elif row["key"] == DELETED_SEMREFS_METADATA_KEY:
            legacy = decode_deleted_ids(json.loads(str(row["value"])))
            row["value"] = json.dumps(
                encode_deleted_ids(
                    {ref_map[value] for value in legacy if value in ref_map}
                )
            )
    return Backup.model_validate(data.model_dump()), len(removed_messages)


def write_file(path: str, backup: Backup) -> None:
    destination = Path(path)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = stream.name
            stream.write(backup.model_dump_json(indent=2))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
