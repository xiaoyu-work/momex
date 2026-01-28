# Copyright (c) Xiaoyu Zhang.
# Licensed under the MIT License.

"""PostgreSQL-based related terms index implementations with pgvector."""

import asyncpg
import numpy as np

from ...aitools.embeddings import NormalizedEmbedding
from ...aitools.vectorbase import TextEmbeddingIndexSettings, VectorBase
from ...knowpro import interfaces
from .schema import deserialize_embedding, serialize_embedding


class PostgresRelatedTermsAliases(interfaces.ITermToRelatedTerms):
    """PostgreSQL-backed implementation of term to related terms aliases."""

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def lookup_term(self, text: str) -> list[interfaces.Term] | None:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT alias FROM RelatedTermsAliases WHERE term = $1",
                text,
            )
            results = [interfaces.Term(row[0]) for row in rows]
            return results if results else None

    async def add_related_term(
        self, text: str, related_terms: interfaces.Term | list[interfaces.Term]
    ) -> None:
        if isinstance(related_terms, interfaces.Term):
            related_terms = [related_terms]

        async with self.pool.acquire() as conn:
            for related_term in related_terms:
                await conn.execute(
                    """
                    INSERT INTO RelatedTermsAliases (term, alias)
                    VALUES ($1, $2)
                    ON CONFLICT DO NOTHING
                    """,
                    text, related_term.text,
                )

    async def remove_term(self, text: str) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM RelatedTermsAliases WHERE term = $1",
                text,
            )

    async def clear(self) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute("DELETE FROM RelatedTermsAliases")

    async def set_related_terms(self, term: str, related_terms: list[str]) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM RelatedTermsAliases WHERE term = $1",
                term,
            )
            for alias in related_terms:
                await conn.execute(
                    "INSERT INTO RelatedTermsAliases (term, alias) VALUES ($1, $2)",
                    term, alias,
                )

    async def size(self) -> int:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT COUNT(DISTINCT term) FROM RelatedTermsAliases"
            )
            return row[0]

    async def get_terms(self) -> list[str]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT DISTINCT term FROM RelatedTermsAliases ORDER BY term"
            )
            return [row[0] for row in rows]

    async def is_empty(self) -> bool:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT COUNT(*) FROM RelatedTermsAliases")
            return row[0] == 0

    async def serialize(self) -> interfaces.TermToRelatedTermsData:
        """Serialize the aliases data."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT term, alias FROM RelatedTermsAliases ORDER BY term, alias"
            )

            term_to_aliases: dict[str, list[str]] = {}
            for row in rows:
                term, alias = row[0], row[1]
                if term not in term_to_aliases:
                    term_to_aliases[term] = []
                term_to_aliases[term].append(alias)

            items = []
            for term, aliases in term_to_aliases.items():
                term_data_list = [interfaces.TermData(text=alias) for alias in aliases]
                items.append(
                    interfaces.TermsToRelatedTermsDataItem(
                        termText=term, relatedTerms=term_data_list
                    )
                )

            return interfaces.TermToRelatedTermsData(relatedTerms=items)

    async def deserialize(self, data: interfaces.TermToRelatedTermsData | None) -> None:
        """Deserialize alias data."""
        async with self.pool.acquire() as conn:
            await conn.execute("DELETE FROM RelatedTermsAliases")

            if data is None:
                return

            related_terms = data.get("relatedTerms", [])

            for item in related_terms:
                if item and item.get("termText") and item.get("relatedTerms"):
                    term = item["termText"]
                    for term_data in item["relatedTerms"]:
                        alias = term_data["text"]
                        await conn.execute(
                            "INSERT INTO RelatedTermsAliases (term, alias) VALUES ($1, $2)",
                            term, alias,
                        )


class PostgresRelatedTermsFuzzy(interfaces.ITermToRelatedTermsFuzzy):
    """PostgreSQL-backed implementation of fuzzy term relationships with pgvector."""

    def __init__(self, pool: asyncpg.Pool, settings: TextEmbeddingIndexSettings):
        self.pool = pool
        self._embedding_settings = settings
        # Keep VectorBase for embedding generation
        self._vector_base = VectorBase(self._embedding_settings)
        self._added_terms: set[str] = set()

    async def _init_from_db(self) -> None:
        """Initialize added_terms set from database."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT term FROM RelatedTermsFuzzy")
            self._added_terms = {row[0] for row in rows}

    async def lookup_term(
        self,
        text: str,
        max_hits: int | None = None,
        min_score: float | None = None,
    ) -> list[interfaces.Term]:
        """Look up similar terms using pgvector fuzzy matching."""
        # Generate embedding for query
        query_embedding = await self._vector_base.get_embedding(text)
        embedding_str = serialize_embedding(query_embedding)

        max_hits = max_hits or 10
        min_score = min_score or 0.0

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT term, 1 - (term_embedding <=> $1) as score
                FROM RelatedTermsFuzzy
                WHERE 1 - (term_embedding <=> $1) >= $2
                ORDER BY term_embedding <=> $1
                LIMIT $3
                """,
                embedding_str, min_score, max_hits,
            )

            results = [
                interfaces.Term(row[0], row[1])
                for row in rows
            ]
            return results

    async def remove_term(self, term: str) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM RelatedTermsFuzzy WHERE term = $1",
                term,
            )
        self._added_terms.discard(term)

    async def clear(self) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute("DELETE FROM RelatedTermsFuzzy")
        self._added_terms.clear()

    async def size(self) -> int:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT COUNT(term) FROM RelatedTermsFuzzy")
            return row[0]

    async def get_terms(self) -> list[str]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT term FROM RelatedTermsFuzzy ORDER BY term"
            )
            return [row[0] for row in rows]

    async def add_terms(self, texts: list[str]) -> None:
        """Add terms with their embeddings."""
        new_terms = [t for t in texts if t not in self._added_terms]
        if not new_terms:
            return

        # Generate embeddings
        embeddings = await self._vector_base.get_embeddings(new_terms, cache=False)

        async with self.pool.acquire() as conn:
            for term, embedding in zip(new_terms, embeddings):
                embedding_str = serialize_embedding(embedding)
                await conn.execute(
                    """
                    INSERT INTO RelatedTermsFuzzy (term, term_embedding)
                    VALUES ($1, $2)
                    ON CONFLICT (term) DO UPDATE SET term_embedding = $2
                    """,
                    term, embedding_str,
                )
                self._added_terms.add(term)

    async def lookup_terms(
        self,
        texts: list[str],
        max_hits: int | None = None,
        min_score: float | None = None,
    ) -> list[list[interfaces.Term]]:
        """Look up multiple terms at once."""
        results = []
        for text in texts:
            term_results = await self.lookup_term(text, max_hits, min_score)
            results.append(term_results)
        return results

    def serialize(self) -> interfaces.TextEmbeddingIndexData:
        """Serialize the fuzzy index data."""
        # This is sync - need to handle differently
        raise NotImplementedError("Use async serialize_async instead")

    async def serialize_async(self) -> interfaces.TextEmbeddingIndexData:
        """Serialize the fuzzy index data (async version)."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT term, term_embedding FROM RelatedTermsFuzzy ORDER BY term"
            )

            text_items = []
            embeddings = []
            for row in rows:
                term, embedding_str = row[0], row[1]
                text_items.append(term)
                embedding = deserialize_embedding(embedding_str)
                embeddings.append(embedding)

            embeddings_array = np.array(embeddings, dtype=np.float32) if embeddings else None

            return interfaces.TextEmbeddingIndexData(
                textItems=text_items,
                embeddings=embeddings_array,
            )

    async def deserialize(self, data: interfaces.TextEmbeddingIndexData) -> None:
        """Deserialize fuzzy index data."""
        async with self.pool.acquire() as conn:
            await conn.execute("DELETE FROM RelatedTermsFuzzy")
        self._added_terms.clear()

        text_items = data.get("textItems")
        embeddings_data = data.get("embeddings")

        if not text_items or embeddings_data is None:
            return

        async with self.pool.acquire() as conn:
            for i, text in enumerate(text_items):
                if i < len(embeddings_data):
                    embedding = embeddings_data[i]
                    embedding_str = serialize_embedding(embedding)
                    await conn.execute(
                        """
                        INSERT INTO RelatedTermsFuzzy (term, term_embedding)
                        VALUES ($1, $2)
                        ON CONFLICT (term) DO UPDATE SET term_embedding = $2
                        """,
                        text, embedding_str,
                    )
                    self._added_terms.add(text)


class PostgresRelatedTermsIndex(interfaces.ITermToRelatedTermsIndex):
    """PostgreSQL-backed implementation of ITermToRelatedTermsIndex."""

    def __init__(self, pool: asyncpg.Pool, settings: TextEmbeddingIndexSettings):
        self.pool = pool
        self._aliases = PostgresRelatedTermsAliases(pool)
        self._fuzzy_index = PostgresRelatedTermsFuzzy(pool, settings)

    @property
    def aliases(self) -> interfaces.ITermToRelatedTerms:
        return self._aliases

    @property
    def fuzzy_index(self) -> interfaces.ITermToRelatedTermsFuzzy | None:
        return self._fuzzy_index

    async def serialize(self) -> interfaces.TermsToRelatedTermsIndexData:
        """Serialize the related terms index."""
        return interfaces.TermsToRelatedTermsIndexData(
            aliasData=await self._aliases.serialize(),
            textEmbeddingData=await self._fuzzy_index.serialize_async(),
        )

    async def deserialize(self, data: interfaces.TermsToRelatedTermsIndexData) -> None:
        """Deserialize related terms index data."""
        alias_data = data.get("aliasData")
        if alias_data is not None:
            await self._aliases.deserialize(alias_data)

        text_embedding_data = data.get("textEmbeddingData")
        if text_embedding_data is not None:
            await self._fuzzy_index.deserialize(text_embedding_data)
