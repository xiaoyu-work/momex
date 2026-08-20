# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SQLite-based message text index implementation."""

import logging
import sqlite3
import typing

import numpy as np

logger = logging.getLogger(__name__)

from ...aitools.embeddings import NormalizedEmbedding
from ...aitools.vectorbase import ScoredInt, VectorBase
from ...knowpro import interfaces
from ...knowpro.convsettings import MessageTextIndexSettings
from ...knowpro.interfaces import TextLocationData, TextToTextLocationIndexData
from ...knowpro.textlocindex import ScoredTextLocation
from ...storage.memory.messageindex import IMessageTextEmbeddingIndex
from .schema import deserialize_embedding, serialize_embedding


class SqliteMessageTextIndex(IMessageTextEmbeddingIndex):
    """SQLite-backed message text index with embedding support."""

    def __init__(
        self,
        db: sqlite3.Connection,
        settings: MessageTextIndexSettings,
        message_collection: interfaces.IMessageCollection | None = None,
    ):
        self.db = db
        self.settings = settings
        self._message_collection = message_collection
        self._vectorbase = VectorBase(settings=settings.embedding_index_settings)
        self._align_stored_positions()
        if self._size():
            cursor = self.db.cursor()
            # Ordered by the position each embedding was stored under, because
            # that is the key fuzzy_lookup results are resolved through. An
            # unordered SELECT only happens to line up while the stored
            # positions are 0..N-1; after a clear() and re-ingest they are not,
            # and every lookup silently resolves to nothing.
            cursor.execute(
                "SELECT embedding FROM MessageTextIndex ORDER BY index_position"
            )
            rows = cursor.fetchall()
            if rows:
                embeddings: list[NormalizedEmbedding] = [
                    deserialize_embedding(row[0]) for row in rows
                ]
                embeddings_array = np.stack(embeddings, axis=0).astype(
                    np.float32, copy=False
                )
                self._vectorbase.add_embeddings(None, embeddings_array)

    def _align_stored_positions(self) -> int:
        """Renumber index_position to 0..N-1, returning how many rows moved.

        Rows written before the position bug was fixed can start from a stale
        offset -- 419..837 on a collection that had been cleared and re-ingested
        once. Deleting a message leaves a hole the same way. Loading orders by
        this column, so the vectors keep the right relative order, but the
        stored values no longer name VectorBase slots and every lookup resolves
        to nothing. The data is intact and only the numbering is wrong, so
        renumbering in place repairs it without needing a re-ingest.

        Healthy collections are already 0..N-1 and nothing is written.
        """
        cursor = self.db.cursor()
        rows = cursor.execute(
            "SELECT msg_id, chunk_ordinal, index_position FROM MessageTextIndex"
            " ORDER BY index_position"
        ).fetchall()
        updates = [
            (expected, msg_id, chunk_ordinal)
            for expected, (msg_id, chunk_ordinal, stored) in enumerate(rows)
            if stored != expected
        ]
        if updates:
            cursor.executemany(
                "UPDATE MessageTextIndex SET index_position = ?"
                " WHERE msg_id = ? AND chunk_ordinal = ?",
                updates,
            )
        return len(updates)

    async def size(self) -> int:
        return self._size()

    def _size(self) -> int:
        cursor = self.db.cursor()
        cursor.execute("SELECT COUNT(*) FROM MessageTextIndex")
        return cursor.fetchone()[0]

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

        embeddings = await self._vectorbase.get_embeddings(
            [chunk for _, _, chunk in chunks_to_embed], cache=False
        )

        # Positions are derived from what is stored, not from the in-memory
        # VectorBase. The two agree only until something empties the table:
        # clear() deletes the rows but leaves this instance's VectorBase
        # populated, so the next ingest wrote positions starting at the old
        # count while a later process loaded the same rows at 0..N-1. Nothing
        # matched after that, and nothing said so.
        current_size = self._size()
        insertion_data: list[tuple[int, int, bytes, int]] = []
        for idx, ((msg_ord, chunk_ord, _), embedding) in enumerate(
            zip(chunks_to_embed, embeddings)
        ):
            insertion_data.append(
                (
                    msg_ord,
                    chunk_ord,
                    serialize_embedding(embedding),
                    current_size + idx,
                )
            )

        # Bulk insert into DB
        cursor = self.db.cursor()
        if insertion_data:
            cursor.executemany(
                """
                INSERT INTO MessageTextIndex
                (msg_id, chunk_ordinal, embedding, index_position)
                VALUES (?, ?, ?, ?)
                """,
                insertion_data,
            )

        # Keep in-memory VectorBase in sync with DB
        self._vectorbase.add_embeddings(None, embeddings)

    async def add_messages(
        self,
        messages: typing.Iterable[interfaces.IMessage],
    ) -> None:
        """Add messages to the text index (backward compatibility method)."""
        message_list = list(messages)
        if not message_list:
            return

        # Check which messages are already indexed
        # Get the highest msg_id that's already in the index
        cursor = self.db.cursor()
        cursor.execute("SELECT MAX(msg_id) FROM MessageTextIndex")
        result = cursor.fetchone()[0]

        if result is None:
            # Index is empty, add all messages starting at 0
            start_ordinal = 0
        else:
            # Index has some entries, only add messages after the highest indexed msg_id
            start_ordinal = result + 1

        # Only add messages that aren't already indexed
        if start_ordinal < len(message_list):
            messages_to_add = message_list[start_ordinal:]
            await self.add_messages_starting_at(start_ordinal, messages_to_add)

    async def rebuild_from_all_messages(self) -> None:
        """Rebuild the entire message text index from all messages in the collection."""
        if self._message_collection is None:
            return

        # Clear existing index
        await self.clear()

        # Add all messages with their ordinals
        message_list = await self._message_collection.get_slice(
            0, await self._message_collection.size()
        )

        if message_list:
            await self.add_messages_starting_at(0, message_list)

        logger.debug("Rebuilt message text index with %d entries", await self.size())

    async def lookup_text(
        self, text: str, max_matches: int | None = None, min_score: float | None = None
    ) -> list[ScoredTextLocation]:
        """Look up text using VectorBase."""
        fuzzy_results = await self._vectorbase.fuzzy_lookup(
            text, max_hits=max_matches, min_score=min_score
        )
        return self._vectorbase_lookup_to_scored_locations(fuzzy_results)

    def _vectorbase_lookup_to_scored_locations(
        self,
        fuzzy_results: list[ScoredInt],
        predicate: typing.Callable[[interfaces.MessageOrdinal], bool] | None = None,
    ) -> list[ScoredTextLocation]:
        """Convert VectorBase fuzzy results to scored text locations using optimized DB query."""
        if not fuzzy_results:
            return []

        # Fetch the rows corresponding to fuzzy_results
        cursor = self.db.cursor()
        index_positions = [scored_int.item for scored_int in fuzzy_results]
        placeholders = ",".join("?" * len(index_positions))
        cursor.execute(
            f"""
            SELECT msg_id, chunk_ordinal, index_position
            FROM MessageTextIndex
            WHERE index_position IN ({placeholders})
            ORDER BY index_position
            """,
            index_positions,
        )
        rows = cursor.fetchall()

        # Create a mapping from index_position to (msg_id, chunk_ordinal)
        position_to_location = {
            index_position: (msg_id, chunk_ordinal)
            for msg_id, chunk_ordinal, index_position in rows
        }

        # Build scored locations, applying predicate filter if provided
        scored_locations = []
        for scored_int in fuzzy_results:
            if scored_int.item in position_to_location:
                msg_id, chunk_ordinal = position_to_location[scored_int.item]

                # Apply predicate filter if provided
                if predicate is None or predicate(msg_id):
                    text_location = interfaces.TextLocation(
                        message_ordinal=msg_id,
                        chunk_ordinal=chunk_ordinal,
                    )
                    scored_locations.append(
                        ScoredTextLocation(text_location, scored_int.score)
                    )

        return scored_locations

    def _scored_locations_to_message_ordinals(
        self,
        scored_locations: list[ScoredTextLocation],
        max_matches: int | None = None,
    ) -> list[interfaces.ScoredMessageOrdinal]:
        """Convert scored text locations to scored message ordinals by grouping chunks."""
        # Group by message and take the best score per message
        message_scores: dict[int, float] = {}
        for scored_loc in scored_locations:
            msg_ord = scored_loc.text_location.message_ordinal
            if msg_ord not in message_scores:
                message_scores[msg_ord] = scored_loc.score
            else:
                # Take the best score for this message
                message_scores[msg_ord] = max(message_scores[msg_ord], scored_loc.score)

        # Convert to list and sort by score
        result = [
            interfaces.ScoredMessageOrdinal(msg_ordinal, score)
            for msg_ordinal, score in message_scores.items()
        ]
        result.sort(key=lambda x: x.score, reverse=True)

        # Apply max_matches limit to final results
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
        # max_matches has to reach the vector search. Passing None here does
        # not mean "no limit" further down: VectorBase.fuzzy_lookup_embedding
        # substitutes 10 for a missing max_hits, so every caller silently got
        # ten messages however many it asked for, and trimming afterwards
        # could only make that smaller. The in-memory implementation has
        # always passed it through; this makes the backends agree.
        scored_locations = await self.lookup_text(
            message_text, max_matches, threshold_score
        )
        return self._scored_locations_to_message_ordinals(scored_locations, max_matches)

    async def lookup_messages_in_subset(
        self,
        message_text: str,
        ordinals_to_search: list[interfaces.MessageOrdinal],
        max_matches: int | None = None,
        threshold_score: float | None = None,
    ) -> list[interfaces.ScoredMessageOrdinal]:
        """Look up messages in a subset of ordinals."""
        # Scoring happens across every message, so the candidates have to be
        # fetched before the subset filter can be applied. `None` here still
        # means VectorBase's default of ten candidates, which is rarely enough
        # to leave anything once filtered -- pass an explicit max_matches if
        # you start using this.
        all_matches = await self.lookup_messages(message_text, None, threshold_score)

        # Filter to only include the specified ordinals
        ordinals_set = set(ordinals_to_search)
        filtered_matches = [
            match for match in all_matches if match.message_ordinal in ordinals_set
        ]

        # Apply max_matches limit
        if max_matches is not None:
            filtered_matches = filtered_matches[:max_matches]

        return filtered_matches

    async def generate_embedding(self, text: str) -> NormalizedEmbedding:
        """Generate an embedding for the given text."""
        return await self._vectorbase.get_embedding(text)

    async def lookup_by_embedding(
        self,
        text_embedding: NormalizedEmbedding,
        max_matches: int | None = None,
        threshold_score: float | None = None,
        predicate: typing.Callable[[interfaces.MessageOrdinal], bool] | None = None,
    ) -> list[interfaces.ScoredMessageOrdinal]:
        """Look up messages by embedding using optimized VectorBase similarity search."""
        fuzzy_results = self._vectorbase.fuzzy_lookup_embedding(
            text_embedding, max_hits=max_matches, min_score=threshold_score
        )
        scored_locations = self._vectorbase_lookup_to_scored_locations(
            fuzzy_results, predicate
        )
        return self._scored_locations_to_message_ordinals(scored_locations, max_matches)

    async def lookup_in_subset_by_embedding(
        self,
        text_embedding: NormalizedEmbedding,
        ordinals_to_search: list[interfaces.MessageOrdinal],
        max_matches: int | None = None,
        threshold_score: float | None = None,
    ) -> list[interfaces.ScoredMessageOrdinal]:
        """Look up messages in a subset by embedding."""
        ordinals_set = set(ordinals_to_search)
        return await self.lookup_by_embedding(
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
        # Get all data from the MessageTextIndex table
        cursor = self.db.cursor()
        cursor.execute("""
            SELECT msg_id, chunk_ordinal, embedding
            FROM MessageTextIndex
            ORDER BY msg_id, chunk_ordinal
        """)

        # Build the text locations and embeddings
        text_locations = []
        embeddings_list = []

        from ..sqlite.schema import deserialize_embedding

        for msg_id, chunk_ordinal, embedding_blob in cursor.fetchall():
            # Create text location data
            text_location = TextLocationData(
                messageOrdinal=msg_id, chunkOrdinal=chunk_ordinal
            )
            text_locations.append(text_location)

            if embedding_blob:
                embedding = deserialize_embedding(embedding_blob)
                embeddings_list.append(embedding)
            else:
                # Handle case where embedding is None
                embeddings_list.append(None)

        if text_locations:
            # Convert embeddings to numpy array if we have any
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
        cursor = self.db.cursor()

        # Clear existing data
        cursor.execute("DELETE FROM MessageTextIndex")

        # Get the index data
        index_data = data.get("indexData")
        if not index_data:
            return

        text_locations = index_data.get("textLocations", [])
        if not text_locations:
            return

        embeddings = index_data.get("embeddings")
        if embeddings is None:
            return

        # Prepare all insertion data for bulk operation
        insertion_data: list[tuple[int, int, bytes, int]] = []
        for idx, (text_location, embedding) in enumerate(
            zip(text_locations, embeddings, strict=True)
        ):
            msg_id = text_location["messageOrdinal"]
            chunk_ordinal = text_location["chunkOrdinal"]
            assert embedding is not None
            embedding_blob = serialize_embedding(embedding)
            # Get the current VectorBase size to determine the index position
            current_size = len(self._vectorbase)
            index_position = current_size + idx
            insertion_data.append(
                (msg_id, chunk_ordinal, embedding_blob, index_position)
            )

        # Bulk insert all the data
        if insertion_data:
            cursor.executemany(
                """
                INSERT INTO MessageTextIndex
                (msg_id, chunk_ordinal, embedding, index_position)
                VALUES (?, ?, ?, ?)
                """,
                insertion_data,
            )

        # Update VectorBase
        self._vectorbase.add_embeddings(None, embeddings)

    async def clear(self) -> None:
        """Clear the message text index."""
        cursor = self.db.cursor()
        cursor.execute("DELETE FROM MessageTextIndex")
        # And the vectors held alongside them. Leaving those behind left this
        # instance believing it had N embeddings for an empty table, so the
        # next ingest numbered its rows from N while any later process loaded
        # them from 0 -- two disjoint position spaces, and lookups that
        # resolved to nothing without failing.
        self._vectorbase.clear()
