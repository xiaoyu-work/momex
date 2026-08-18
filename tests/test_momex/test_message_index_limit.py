"""Tests that a message-index lookup returns as many results as asked for.

The SQLite and PostgreSQL implementations dropped their `max_matches` argument
when calling `lookup_text`, passing None instead. None does not mean "no limit"
further down -- VectorBase.fuzzy_lookup_embedding substitutes 10 for a missing
max_hits -- so the embedding half of Momex's hybrid search returned at most ten
messages however many the caller asked for, and trimming afterwards could only
make that smaller.

Measured on LOCOMO, that capped evidence recall at 60.5% no matter the cut-off:
identical at k=20, k=100 and k=419, in a collection of 419 messages.

The in-memory implementation always passed it through, which is why the tests
did not catch this: they use the memory provider.
"""

import pytest

from typeagent.knowpro.convsettings import MessageTextIndexSettings
from typeagent.storage.memory.messageindex import MessageTextIndex
from typeagent.storage.sqlite.messageindex import SqliteMessageTextIndex


class _Recorder:
    """Stands in for the vector search, recording the limit it is handed."""

    def __init__(self):
        self.max_hits: list[int | None] = []

    async def fuzzy_lookup(self, key, max_hits=None, min_score=None, predicate=None):
        self.max_hits.append(max_hits)
        return []


@pytest.mark.asyncio
async def test_sqlite_forwards_the_limit_to_the_vector_search(monkeypatch):
    index = SqliteMessageTextIndex.__new__(SqliteMessageTextIndex)
    recorder = _Recorder()
    index._vectorbase = recorder  # type: ignore[attr-defined]

    await index.lookup_messages("anything", max_matches=50, threshold_score=0.0)

    assert recorder.max_hits == [
        50
    ], "max_matches was dropped; VectorBase will substitute its default of 10"


@pytest.mark.asyncio
async def test_postgres_forwards_the_limit_to_the_vector_search(monkeypatch):
    pytest.importorskip("asyncpg")
    from typeagent.storage.postgres.messageindex import PostgresMessageTextIndex

    index = PostgresMessageTextIndex.__new__(PostgresMessageTextIndex)
    recorder = _Recorder()
    index._vectorbase = recorder  # type: ignore[attr-defined]

    await index.lookup_messages("anything", max_matches=50, threshold_score=0.0)

    assert recorder.max_hits == [50]


@pytest.mark.asyncio
async def test_memory_backend_already_forwarded_it(monkeypatch):
    """The behaviour the other two now match."""
    settings = MessageTextIndexSettings.__new__(MessageTextIndexSettings)
    index = MessageTextIndex.__new__(MessageTextIndex)
    index.settings = settings  # type: ignore[attr-defined]

    seen: list[int | None] = []

    class _Locations:
        async def lookup_text(self, text, max_matches=None, threshold_score=None):
            seen.append(max_matches)
            return []

    index.text_location_index = _Locations()  # type: ignore[attr-defined]

    class _Embedding:
        max_matches = None
        min_score = 0.0

    settings.embedding_index_settings = _Embedding()  # type: ignore[attr-defined]

    await index.lookup_messages("anything", max_matches=50, threshold_score=0.1)

    assert seen == [50]
