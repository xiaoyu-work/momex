# Copyright (c) Xiaoyu Zhang.
# Licensed under the MIT License.

"""PostgreSQL-based message text index implementation with pgvector."""

import typing

import asyncpg
import numpy as np

from ...aitools.embeddings import NormalizedEmbedding
from ...aitools.vectorbase import VectorBase
from ...knowpro import interfaces
from ...knowpro.convsettings import MessageTextIndexSettings
from ...knowpro.interfaces import TextLocationData, TextToTextLocationIndexData
from ...knowpro.textlocindex import ScoredTextLocation
from ...storage.memory.messageindex import IMessageTextEmbeddingIndex
from .schema import serialize_embedding, deserialize_embedding


class PostgresMessageTextIndex(IMessageTextEmbeddingIndex):
    """PostgreSQL-backed message text index with pgvector embedding support."""

    def __init__(
        self,
        pool: asyncpg.Pool,
        settings: MessageTextIndexSettings,
        message_collection: interfaces.IMessageCollection | None = None,
    ):
        self.pool = pool
        self.settings = settings
        self._message_collection = message_collection
        # Keep VectorBase for embedding generation (uses OpenAI API)
        self._vectorbase = VectorBase(settings=settings.embedding_index_settings)

    async def size(self) -> int:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT COUNT(*) FROM MessageTextIndex")
            return row[0]

    async def add_messages_starting_at(
        self,
        start_message_ordinal: int,
        messages: list[interfaces.IMessage],
    ) -> None:
        """Add messages to the text index starting at the given ordinal."""
        chunks_to_embed: list[tuple[int, int, str]] = []
        for msg_ord, message in enumerate(messages, start_message_ordinal):
            for chunk_ord, chunk in enumerate(message.text_chunks):
                chunks_to_embed.append((msg_ord, chunk_ord, chunk))

        if not chunks_to_embed:
            return

        # Generate embeddings using VectorBase (calls OpenAI API)
        embeddings = await self._vectorbase.get_embeddings(
            [chunk for _, _, chunk in chunks_to_embed], cache=False
        )

        async with self.pool.acquire() as conn:
            for (msg_ord, chunk_ord, _), embedding in zip(chunks_to_embed, embeddings):
                embedding_str = serialize_embedding(embedding)
                await conn.execute(
                    """
                    INSERT INTO MessageTextIndex (msg_id, chunk_ordinal, embedding)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (msg_id, chunk_ordinal) DO UPDATE SET embedding = $3
                    """,
                    msg_ord, chunk_ord, embedding_str,
                )

    async def add_messages(
        self,
        messages: typing.Iterable[interfaces.IMessage],
    ) -> None:
        """Add messages to the text index (backward compatibility method)."""
        message_list = list(messages)
        if not message_list:
            return

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT MAX(msg_id) FROM MessageTextIndex")
            result = row[0]

        if result is None:
            start_ordinal = 0
        else:
            start_ordinal = result + 1

        if start_ordinal < len(message_list):
            messages_to_add = message_list[start_ordinal:]
            await self.add_messages_starting_at(start_ordinal, messages_to_add)

    async def rebuild_from_all_messages(self) -> None:
        """Rebuild the entire message text index from all messages in the collection."""
        if self._message_collection is None:
            return

        await self.clear()

        message_list = await self._message_collection.get_slice(
            0, await self._message_collection.size()
        )

        if message_list:
            await self.add_messages_starting_at(0, message_list)

    async def lookup_text(
        self, text: str, max_matches: int | None = None, min_score: float | None = None
    ) -> list[ScoredTextLocation]:
        """Look up text using pgvector similarity search."""
        # Generate embedding for query text
        query_embedding = await self._vectorbase.get_embedding(text)
        return await self._lookup_by_embedding(query_embedding, max_matches, min_score)

    async def _lookup_by_embedding(
        self,
        embedding: NormalizedEmbedding,
        max_matches: int | None = None,
        min_score: float | None = None,
        predicate: typing.Callable[[interfaces.MessageOrdinal], bool] | None = None,
    ) -> list[ScoredTextLocation]:
        """Look up similar embeddings using pgvector."""
        embedding_str = serialize_embedding(embedding)
        max_matches = max_matches or 100
        min_score = min_score or 0.0

        async with self.pool.acquire() as conn:
            # Use cosine similarity: 1 - cosine_distance
            # pgvector's <=> operator returns cosine distance (0 = identical, 2 = opposite)
            rows = await conn.fetch(
                """
                SELECT msg_id, chunk_ordinal, 1 - (embedding <=> $1) as score
                FROM MessageTextIndex
                WHERE 1 - (embedding <=> $1) >= $2
                ORDER BY embedding <=> $1
                LIMIT $3
                """,
                embedding_str, min_score, max_matches,
            )

            scored_locations = []
            for row in rows:
                msg_id, chunk_ordinal, score = row[0], row[1], row[2]

                if predicate is None or predicate(msg_id):
                    text_location = interfaces.TextLocation(
                        message_ordinal=msg_id,
                        chunk_ordinal=chunk_ordinal,
                    )
                    scored_locations.append(ScoredTextLocation(text_location, score))

            return scored_locations

    def _scored_locations_to_message_ordinals(
        self,
        scored_locations: list[ScoredTextLocation],
        max_matches: int | None = None,
    ) -> list[interfaces.ScoredMessageOrdinal]:
        """Convert scored text locations to scored message ordinals by grouping chunks."""
        message_scores: dict[int, float] = {}
        for scored_loc in scored_locations:
            msg_ord = scored_loc.text_location.message_ordinal
            if msg_ord not in message_scores:
                message_scores[msg_ord] = scored_loc.score
            else:
                message_scores[msg_ord] = max(message_scores[msg_ord], scored_loc.score)

        result = [
            interfaces.ScoredMessageOrdinal(msg_ordinal, score)
            for msg_ordinal, score in message_scores.items()
        ]
        result.sort(key=lambda x: x.score, reverse=True)

        if max_matches is not None:
            result = result[:max_matches]

        return result

    async def lookup_messages(
        self,
        message_text: str,
        max_matches: int | None = None,
        threshold_score: float | None = None,
    ) -> list[interfaces.ScoredMessageOrdinal]:
        """Look up messages by text content."""
        scored_locations = await self.lookup_text(message_text, None, threshold_score)
        return self._scored_locations_to_message_ordinals(scored_locations, max_matches)

    async def lookup_messages_in_subset(
        self,
        message_text: str,
        ordinals_to_search: list[interfaces.MessageOrdinal],
        max_matches: int | None = None,
        threshold_score: float | None = None,
    ) -> list[interfaces.ScoredMessageOrdinal]:
        """Look up messages in a subset of ordinals."""
        all_matches = await self.lookup_messages(message_text, None, threshold_score)

        ordinals_set = set(ordinals_to_search)
        filtered_matches = [
            match for match in all_matches if match.message_ordinal in ordinals_set
        ]

        if max_matches is not None:
            filtered_matches = filtered_matches[:max_matches]

        return filtered_matches

    async def generate_embedding(self, text: str) -> NormalizedEmbedding:
        """Generate an embedding for the given text."""
        return await self._vectorbase.get_embedding(text)

    def lookup_by_embedding(
        self,
        text_embedding: NormalizedEmbedding,
        max_matches: int | None = None,
        threshold_score: float | None = None,
        predicate: typing.Callable[[interfaces.MessageOrdinal], bool] | None = None,
    ) -> list[interfaces.ScoredMessageOrdinal]:
        """Look up messages by embedding (sync version - uses asyncio.run)."""
        import asyncio
        scored_locations = asyncio.get_event_loop().run_until_complete(
            self._lookup_by_embedding(text_embedding, max_matches, threshold_score, predicate)
        )
        return self._scored_locations_to_message_ordinals(scored_locations, max_matches)

    def lookup_in_subset_by_embedding(
        self,
        text_embedding: NormalizedEmbedding,
        ordinals_to_search: list[interfaces.MessageOrdinal],
        max_matches: int | None = None,
        threshold_score: float | None = None,
    ) -> list[interfaces.ScoredMessageOrdinal]:
        """Look up messages in a subset by embedding (synchronous version)."""
        ordinals_set = set(ordinals_to_search)
        return self.lookup_by_embedding(
            text_embedding,
            max_matches,
            threshold_score,
            predicate=lambda ordinal: ordinal in ordinals_set,
        )

    async def is_empty(self) -> bool:
        """Check if the index is empty."""
        size = await self.size()
        return size == 0

    async def serialize(self) -> interfaces.MessageTextIndexData:
        """Serialize the message text index."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT msg_id, chunk_ordinal, embedding
                FROM MessageTextIndex
                ORDER BY msg_id, chunk_ordinal
                """
            )

            text_locations = []
            embeddings_list = []

            for row in rows:
                msg_id, chunk_ordinal, embedding_str = row[0], row[1], row[2]
                text_location = TextLocationData(
                    messageOrdinal=msg_id, chunkOrdinal=chunk_ordinal
                )
                text_locations.append(text_location)

                if embedding_str:
                    embedding = deserialize_embedding(embedding_str)
                    embeddings_list.append(embedding)
                else:
                    embeddings_list.append(None)

            if text_locations:
                valid_embeddings = [e for e in embeddings_list if e is not None]
                if valid_embeddings:
                    embeddings_array = np.array(valid_embeddings, dtype=np.float32)
                else:
                    embeddings_array = None

                index_data = TextToTextLocationIndexData(
                    textLocations=text_locations, embeddings=embeddings_array
                )
                return interfaces.MessageTextIndexData(indexData=index_data)

            return {}

    async def deserialize(self, data: interfaces.MessageTextIndexData) -> None:
        """Deserialize message text index data."""
        async with self.pool.acquire() as conn:
            await conn.execute("DELETE FROM MessageTextIndex")

            index_data = data.get("indexData")
            if not index_data:
                return

            text_locations = index_data.get("textLocations", [])
            if not text_locations:
                return

            embeddings = index_data.get("embeddings")
            if embeddings is None:
                return

            for text_location, embedding in zip(text_locations, embeddings, strict=True):
                msg_id = text_location["messageOrdinal"]
                chunk_ordinal = text_location["chunkOrdinal"]
                embedding_str = serialize_embedding(embedding)

                await conn.execute(
                    """
                    INSERT INTO MessageTextIndex (msg_id, chunk_ordinal, embedding)
                    VALUES ($1, $2, $3)
                    """,
                    msg_id, chunk_ordinal, embedding_str,
                )

    async def clear(self) -> None:
        """Clear the message text index."""
        async with self.pool.acquire() as conn:
            await conn.execute("DELETE FROM MessageTextIndex")
