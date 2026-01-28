# Copyright (c) Xiaoyu Zhang.
# Licensed under the MIT License.

"""PostgreSQL-based semantic reference index implementation."""

import re
import unicodedata

import asyncpg

from ...knowpro import interfaces
from ...knowpro.interfaces import ScoredSemanticRefOrdinal


class PostgresTermToSemanticRefIndex(interfaces.ITermToSemanticRefIndex):
    """PostgreSQL-backed implementation of term to semantic ref index."""

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def size(self) -> int:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT COUNT(DISTINCT term) FROM SemanticRefIndex"
            )
            return row[0]

    async def get_terms(self) -> list[str]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT DISTINCT term FROM SemanticRefIndex ORDER BY term"
            )
            return [row[0] for row in rows]

    async def add_term(
        self,
        term: str,
        semantic_ref_ordinal: (
            interfaces.SemanticRefOrdinal | interfaces.ScoredSemanticRefOrdinal
        ),
    ) -> str:
        if not term:
            return term

        term = self._prepare_term(term)

        if isinstance(semantic_ref_ordinal, interfaces.ScoredSemanticRefOrdinal):
            semref_id = semantic_ref_ordinal.semantic_ref_ordinal
        else:
            semref_id = semantic_ref_ordinal

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO SemanticRefIndex (term, semref_id)
                VALUES ($1, $2)
                ON CONFLICT DO NOTHING
                """,
                term, semref_id,
            )

        return term

    async def remove_term(
        self, term: str, semantic_ref_ordinal: interfaces.SemanticRefOrdinal
    ) -> None:
        term = self._prepare_term(term)
        async with self.pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM SemanticRefIndex WHERE term = $1 AND semref_id = $2",
                term, semantic_ref_ordinal,
            )

    async def lookup_term(
        self, term: str
    ) -> list[interfaces.ScoredSemanticRefOrdinal] | None:
        term = self._prepare_term(term)
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT semref_id FROM SemanticRefIndex WHERE term = $1",
                term,
            )

            results = []
            for row in rows:
                semref_id = row[0]
                results.append(ScoredSemanticRefOrdinal(semref_id, 1.0))
            return results

    async def clear(self) -> None:
        """Clear all terms from the semantic ref index."""
        async with self.pool.acquire() as conn:
            await conn.execute("DELETE FROM SemanticRefIndex")

    async def serialize(self) -> interfaces.TermToSemanticRefIndexData:
        """Serialize the index data for compatibility with in-memory version."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT term, semref_id FROM SemanticRefIndex ORDER BY term, semref_id"
            )

            term_to_semrefs: dict[str, list[interfaces.ScoredSemanticRefOrdinalData]] = {}
            for row in rows:
                term, semref_id = row[0], row[1]
                if term not in term_to_semrefs:
                    term_to_semrefs[term] = []
                scored_ref = ScoredSemanticRefOrdinal(semref_id, 1.0)
                term_to_semrefs[term].append(scored_ref.serialize())

            items = []
            for term, semref_ordinals in term_to_semrefs.items():
                items.append(
                    interfaces.TermToSemanticRefIndexItemData(
                        term=term, semanticRefOrdinals=semref_ordinals
                    )
                )

            return interfaces.TermToSemanticRefIndexData(items=items)

    async def deserialize(self, data: interfaces.TermToSemanticRefIndexData) -> None:
        """Deserialize index data by populating the PostgreSQL table."""
        async with self.pool.acquire() as conn:
            await conn.execute("DELETE FROM SemanticRefIndex")

            for item in data["items"]:
                if item and item["term"]:
                    term = self._prepare_term(item["term"])
                    for semref_ordinal_data in item["semanticRefOrdinals"]:
                        if isinstance(semref_ordinal_data, dict):
                            semref_id = semref_ordinal_data["semanticRefOrdinal"]
                        else:
                            semref_id = semref_ordinal_data
                        await conn.execute(
                            """
                            INSERT INTO SemanticRefIndex (term, semref_id)
                            VALUES ($1, $2)
                            ON CONFLICT DO NOTHING
                            """,
                            term, semref_id,
                        )

    def _prepare_term(self, term: str) -> str:
        """Normalize term by converting to lowercase, stripping whitespace, and normalizing Unicode."""
        term = term.strip()
        term = unicodedata.normalize("NFC", term)
        term = re.sub(r"\s+", " ", term)
        return term.lower()
