"""Pin all index operations to the transaction's connection."""

from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from contextvars import ContextVar
from typing import Any, Protocol


class Pool(Protocol):
    def acquire(self) -> AbstractAsyncContextManager[Any]: ...
    async def close(self) -> None: ...


class TransactionPool:
    def __init__(self, pool: Pool, schema: str | None = None):
        self._pool = pool
        self._schema = schema
        self._connection: ContextVar[Any] = ContextVar(
            "postgres_transaction", default=None
        )

    async def _set_path(self, connection: Any) -> None:
        if self._schema:
            from .schema import format_search_path

            await connection.execute(
                f"SET LOCAL search_path TO {format_search_path(self._schema)}"
            )

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[Any]:
        pinned = self._connection.get()
        if pinned is not None:
            yield pinned
            return
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                await self._set_path(connection)
                yield connection

    @asynccontextmanager
    async def transaction(self, **options: Any) -> AsyncIterator[Any]:
        if self._connection.get() is not None:
            raise RuntimeError("Cannot nest storage transactions")
        async with self._pool.acquire() as connection:
            async with connection.transaction(**options):
                await self._set_path(connection)
                token = self._connection.set(connection)
                try:
                    yield connection
                finally:
                    self._connection.reset(token)

    async def close(self) -> None:
        await self._pool.close()
