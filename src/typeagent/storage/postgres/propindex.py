# Copyright (c) Xiaoyu Zhang.
# Licensed under the MIT License.

"""PostgreSQL-based property index implementation."""

import asyncpg

from ...knowpro import interfaces
from ...knowpro.interfaces import ScoredSemanticRefOrdinal


class PostgresPropertyIndex(interfaces.IPropertyToSemanticRefIndex):
    """PostgreSQL-backed implementation of property to semantic ref index."""

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def size(self) -> int:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT COUNT(*) FROM (SELECT DISTINCT prop_name, value_str FROM PropertyIndex) AS t"
            )
            return row[0]

    async def get_values(self) -> list[str]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT DISTINCT value_str FROM PropertyIndex ORDER BY value_str"
            )
            return [row[0] for row in rows]

    async def add_property(
        self,
        property_name: str,
        value: str,
        semantic_ref_ordinal: (
            interfaces.SemanticRefOrdinal | interfaces.ScoredSemanticRefOrdinal
        ),
    ) -> None:
        if isinstance(semantic_ref_ordinal, interfaces.ScoredSemanticRefOrdinal):
            semref_id = semantic_ref_ordinal.semantic_ref_ordinal
            score = semantic_ref_ordinal.score
        else:
            semref_id = semantic_ref_ordinal
            score = 1.0

        from ...storage.memory.propindex import (
            make_property_term_text,
            split_property_term_text,
        )

        term_text = make_property_term_text(property_name, value)
        term_text = term_text.lower()
        property_name, value = split_property_term_text(term_text)
        if property_name.startswith("prop."):
            property_name = property_name[5:]

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO PropertyIndex (prop_name, value_str, score, semref_id)
                VALUES ($1, $2, $3, $4)
                """,
                property_name, value, score, semref_id,
            )

    async def clear(self) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute("DELETE FROM PropertyIndex")

    async def lookup_property(
        self,
        property_name: str,
        value: str,
    ) -> list[interfaces.ScoredSemanticRefOrdinal] | None:
        from ...storage.memory.propindex import (
            make_property_term_text,
            split_property_term_text,
        )

        term_text = make_property_term_text(property_name, value)
        term_text = term_text.lower()
        property_name, value = split_property_term_text(term_text)
        if property_name.startswith("prop."):
            property_name = property_name[5:]

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT semref_id, score FROM PropertyIndex WHERE prop_name = $1 AND value_str = $2",
                property_name, value,
            )

            results = [
                ScoredSemanticRefOrdinal(row[0], row[1])
                for row in rows
            ]

            return results if results else None

    async def remove_property(self, prop_name: str, semref_id: int) -> None:
        """Remove all properties for a specific property name and semantic ref."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM PropertyIndex WHERE prop_name = $1 AND semref_id = $2",
                prop_name, semref_id,
            )

    async def remove_all_for_semref(self, semref_id: int) -> None:
        """Remove all properties for a specific semantic ref."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM PropertyIndex WHERE semref_id = $1",
                semref_id,
            )
