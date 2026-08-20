"""Tests that a collection still searches after the process that wrote it exits.

One process ingests and another serves: that is the ordinary shape for a
memory library, and it is where a whole retrieval path went missing without
raising anything. This is the same guarantee as
test_message_index_positions.py, asserted through the public API rather than
against the index internals, so a future storage change cannot quietly reopen
the hole.

Embeddings are faked. The point is the position bookkeeping across a reopen,
not embedding quality, and the test has to run offline.
"""

from typing import Any

import numpy as np
import pytest

from momex import LLMConfig, Memory, MomexConfig, StorageConfig


class _FakeEmbeddingModel:
    """Deterministic per-text vectors, so a query matches its own message.

    Carries model_name and encoding_name because the storage provider records
    which model wrote a collection and refuses to mix vectors from two.
    """

    embedding_size = 16
    model_name = "fake-embedding-model"

    def __init__(self):
        self.encoding_name = "fake"

    async def get_embedding(self, text: str):
        return self._encode(text)

    async def get_embeddings(self, texts: list[str]):
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.embedding_size)
        return np.stack([self._encode(t) for t in texts], axis=0)

    async def get_embedding_nocache(self, text: str):
        return self._encode(text)

    async def get_embeddings_nocache(self, texts: list[str]):
        return await self.get_embeddings(texts)

    def _encode(self, text: str):
        rng = np.random.default_rng(abs(hash(text.strip().lower())) % (2**32))
        vec = rng.standard_normal(self.embedding_size).astype(np.float32)
        return (vec / np.linalg.norm(vec)).astype(np.float32)

    def add_embedding(self, key: str, embedding) -> None:
        """Caching hook. Encoding is deterministic, so there is nothing to do."""


@pytest.fixture
def config(tmp_path, monkeypatch):
    config = MomexConfig(
        llm=LLMConfig(provider="openai", model="gpt-4o", api_key="sk-dummy"),
        storage=StorageConfig(path=str(tmp_path)),
    )
    model: Any = _FakeEmbeddingModel()
    monkeypatch.setattr(config, "create_embedding_model", lambda: model)
    return config


MESSAGES = [
    "Caroline researched adoption agencies",
    "Melanie signed up for a pottery class",
    "the rain in spain stays mainly in the plain",
]


async def _fill(memory: Memory) -> None:
    for text in MESSAGES:
        await memory.add(text, infer=False)


@pytest.mark.asyncio
async def test_a_second_instance_can_search_what_the_first_wrote(config):
    writer = Memory(collection="test:reopen", config=config)
    await _fill(writer)
    await writer.close()

    reader = Memory(collection="test:reopen", config=config)
    results = await reader.search_by_embedding(MESSAGES[0], limit=5, min_score=0.0)

    assert results, "reopening the collection lost the embedding index"
    assert results[0].text == MESSAGES[0]
    await reader.close()


@pytest.mark.asyncio
async def test_search_survives_clear_and_re_ingest(config):
    """The sequence that broke it: clear, write again, reopen."""
    writer = Memory(collection="test:recycled", config=config)
    await _fill(writer)
    await writer.clear()
    await _fill(writer)
    await writer.close()

    reader = Memory(collection="test:recycled", config=config)
    results = await reader.search_by_embedding(MESSAGES[1], limit=5, min_score=0.0)

    assert results
    assert results[0].text == MESSAGES[1]
    await reader.close()


@pytest.mark.asyncio
async def test_incremental_writes_stay_searchable(config):
    """Adding after a reopen must not strand either batch."""
    first = Memory(collection="test:incremental", config=config)
    await first.add(MESSAGES[0], infer=False)
    await first.close()

    second = Memory(collection="test:incremental", config=config)
    await second.add(MESSAGES[1], infer=False)
    await second.close()

    reader = Memory(collection="test:incremental", config=config)
    for text in MESSAGES[:2]:
        results = await reader.search_by_embedding(text, limit=5, min_score=0.0)
        assert results and results[0].text == text
    await reader.close()
