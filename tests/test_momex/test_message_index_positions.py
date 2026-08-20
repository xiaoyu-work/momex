"""Tests that embedding search survives a reopen and a clear().

The SQLite message index stores each embedding with an `index_position`, and
resolves vector-search hits back to messages through that column. Two things
made those positions disagree with the vectors they name:

  clear() deleted the rows but left this instance's VectorBase populated, so
  the next ingest numbered rows from the old count.

  The constructor loaded embeddings with an unordered SELECT, so their
  in-memory positions were 0..N-1 whatever the stored positions said.

Either way the two spaces stop overlapping and every lookup resolves to
nothing -- while size() still reports the right count, stats() looks healthy,
and no error is raised. Measured on a re-ingested LOCOMO collection, embedding
search returned zero results for every query.

PostgreSQL is unaffected: pgvector does the comparison in the database and
never maps through a position.
"""

from typing import Any

import numpy as np
import pytest

from typeagent.aitools.embeddings import NormalizedEmbedding
from typeagent.knowpro.convsettings import MessageTextIndexSettings
from typeagent.knowpro.universal_message import (
    ConversationMessage,
    ConversationMessageMeta,
)
from typeagent.storage.sqlite.messageindex import SqliteMessageTextIndex
from typeagent.storage.sqlite.schema import init_db_schema


class _FakeEmbeddingModel:
    """Deterministic embeddings, so similarity is decided by the text."""

    embedding_size = 8

    def __init__(self):
        self.encoding_name = "fake"

    async def get_embedding(self, text: str) -> NormalizedEmbedding:
        return self._encode(text)

    async def get_embeddings(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.embedding_size)
        return np.stack([self._encode(t) for t in texts], axis=0)

    async def get_embedding_nocache(self, text: str) -> NormalizedEmbedding:
        return self._encode(text)

    async def get_embeddings_nocache(self, texts: list[str]) -> np.ndarray:
        return await self.get_embeddings(texts)

    def _encode(self, text: str) -> NormalizedEmbedding:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        vec = rng.standard_normal(self.embedding_size).astype(np.float32)
        return (vec / np.linalg.norm(vec)).astype(np.float32)


def _message(text: str) -> ConversationMessage:
    return ConversationMessage(
        text_chunks=[text],
        metadata=ConversationMessageMeta(speaker="u"),
        tags=[],
        timestamp="2023-05-08T00:00:00Z",
    )


@pytest.fixture
def settings():
    from typeagent.aitools.vectorbase import TextEmbeddingIndexSettings

    # The fake implements the slice of IEmbeddingModel this index uses.
    model: Any = _FakeEmbeddingModel()
    return MessageTextIndexSettings(TextEmbeddingIndexSettings(model, min_score=0.0))


@pytest.fixture
def db(tmp_path):
    import sqlite3

    connection = sqlite3.connect(str(tmp_path / "test.db"))
    init_db_schema(connection)
    yield connection
    connection.close()


TEXTS = ["apples and pears", "a cat sat down", "the rain in spain"]


async def _ingest(index: SqliteMessageTextIndex, texts: list[str], start: int = 0):
    await index.add_messages_starting_at(start, [_message(t) for t in texts])


def _positions(db) -> list[int]:
    return [
        row[0]
        for row in db.execute(
            "SELECT index_position FROM MessageTextIndex ORDER BY index_position"
        )
    ]


@pytest.mark.asyncio
async def test_lookup_works_after_reopening_the_database(db, settings):
    """A fresh process must resolve hits the previous one stored."""
    first = SqliteMessageTextIndex(db, settings)
    await _ingest(first, TEXTS)

    reopened = SqliteMessageTextIndex(db, settings)
    hits = await reopened.lookup_messages(TEXTS[1], max_matches=5, threshold_score=0.0)

    assert hits, "the reopened index resolved nothing"
    assert hits[0].message_ordinal == 1


@pytest.mark.asyncio
async def test_positions_stay_dense_across_a_clear(db, settings):
    """The regression: re-ingesting used to number rows from the old count."""
    index = SqliteMessageTextIndex(db, settings)
    await _ingest(index, TEXTS)
    await index.clear()
    await _ingest(index, TEXTS)

    assert _positions(db) == [0, 1, 2]


@pytest.mark.asyncio
async def test_lookup_works_after_clear_and_re_ingest(db, settings):
    index = SqliteMessageTextIndex(db, settings)
    await _ingest(index, TEXTS)
    await index.clear()
    await _ingest(index, TEXTS)

    hits = await index.lookup_messages(TEXTS[2], max_matches=5, threshold_score=0.0)

    assert hits
    assert hits[0].message_ordinal == 2


@pytest.mark.asyncio
async def test_lookup_works_after_clear_re_ingest_and_reopen(db, settings):
    """The exact sequence the benchmark hit: clear, ingest, new process."""
    index = SqliteMessageTextIndex(db, settings)
    await _ingest(index, TEXTS)
    await index.clear()
    await _ingest(index, TEXTS)

    reopened = SqliteMessageTextIndex(db, settings)
    hits = await reopened.lookup_messages(TEXTS[0], max_matches=5, threshold_score=0.0)

    assert hits
    assert hits[0].message_ordinal == 0


@pytest.mark.asyncio
async def test_clear_empties_the_in_memory_vectors(db, settings):
    index = SqliteMessageTextIndex(db, settings)
    await _ingest(index, TEXTS)
    await index.clear()

    assert len(index._vectorbase) == 0
    assert await index.size() == 0


@pytest.mark.asyncio
async def test_incremental_ingest_keeps_positions_contiguous(db, settings):
    index = SqliteMessageTextIndex(db, settings)
    await _ingest(index, TEXTS[:2])
    await _ingest(index, TEXTS[2:], start=2)

    assert _positions(db) == [0, 1, 2]

    reopened = SqliteMessageTextIndex(db, settings)
    hits = await reopened.lookup_messages(TEXTS[2], max_matches=5, threshold_score=0.0)
    assert hits and hits[0].message_ordinal == 2


@pytest.mark.asyncio
async def test_a_database_written_before_the_fix_repairs_itself(db, settings):
    """Fixing the write path does not help collections already on disk.

    A store written by the buggy version holds rows numbered from a stale
    offset. The vectors are intact and in the right relative order, so the
    numbering is repairable, and it has to be -- otherwise those collections
    return nothing forever and nothing says why.
    """
    index = SqliteMessageTextIndex(db, settings)
    await _ingest(index, TEXTS)
    # Exactly the shape the old code produced: dense, ordered, wrong origin.
    db.execute("UPDATE MessageTextIndex SET index_position = index_position + 419")
    assert _positions(db) == [419, 420, 421]

    reopened = SqliteMessageTextIndex(db, settings)

    assert _positions(db) == [0, 1, 2]
    hits = await reopened.lookup_messages(TEXTS[1], max_matches=5, threshold_score=0.0)
    assert hits, "a repaired index still resolved nothing"
    assert hits[0].message_ordinal == 1


@pytest.mark.asyncio
async def test_repair_preserves_which_message_each_vector_names(db, settings):
    """Renumbering must not shuffle the vector-to-message pairing.

    Every text must still find itself, not merely find something.
    """
    index = SqliteMessageTextIndex(db, settings)
    await _ingest(index, TEXTS)
    db.execute("UPDATE MessageTextIndex SET index_position = index_position + 7")

    reopened = SqliteMessageTextIndex(db, settings)

    for expected_ordinal, text in enumerate(TEXTS):
        hits = await reopened.lookup_messages(text, max_matches=1, threshold_score=0.0)
        assert hits and hits[0].message_ordinal == expected_ordinal, text


@pytest.mark.asyncio
async def test_repair_leaves_a_healthy_index_untouched(db, settings):
    """The common path must not pay for the broken one."""
    index = SqliteMessageTextIndex(db, settings)
    await _ingest(index, TEXTS)

    reopened = SqliteMessageTextIndex(db, settings)

    assert reopened._align_stored_positions() == 0
    assert _positions(db) == [0, 1, 2]


@pytest.mark.asyncio
async def test_repair_handles_gaps_left_by_deleted_messages(db, settings):
    """Deleting a message cascades to its embedding row and leaves a hole.

    The rows that remain then name VectorBase slots that no longer exist,
    which is the same failure by a different route.
    """
    index = SqliteMessageTextIndex(db, settings)
    await _ingest(index, TEXTS)
    db.execute("DELETE FROM MessageTextIndex WHERE index_position = 0")
    assert _positions(db) == [1, 2]

    reopened = SqliteMessageTextIndex(db, settings)

    assert _positions(db) == [0, 1]
    for text, expected_ordinal in [(TEXTS[1], 1), (TEXTS[2], 2)]:
        hits = await reopened.lookup_messages(text, max_matches=1, threshold_score=0.0)
        assert hits and hits[0].message_ordinal == expected_ordinal, text
