"""Connection-level contracts, independent of an installed PostgreSQL service."""

import asyncio
from contextlib import asynccontextmanager

import pytest

from typeagent.storage.postgres.connection import TransactionPool


class Connection:
    def __init__(self):
        self.events = []
        self.queries = []

    @asynccontextmanager
    async def transaction(self, **options):
        self.events.append(("begin", options))
        try:
            yield self
        except BaseException:
            self.events.append(("rollback", None))
            raise
        else:
            self.events.append(("commit", None))

    async def execute(self, query, *args):
        self.queries.append((query, args))


class Pool:
    def __init__(self):
        self.connections = []
        self.closed = False

    @asynccontextmanager
    async def acquire(self):
        conn = Connection()
        self.connections.append(conn)
        yield conn

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_every_index_acquires_the_transaction_connection():
    raw = Pool()
    pool = TransactionPool(raw, "tenant")
    async with pool.transaction() as owner:
        async with pool.acquire() as first:
            async with pool.acquire() as second:
                assert owner is first is second
    assert len(raw.connections) == 1
    assert owner.events[-1][0] == "commit"
    assert owner.queries[0][0].startswith('SET LOCAL search_path TO "tenant"')


@pytest.mark.asyncio
async def test_rollback_is_propagated_and_releases_the_pinned_connection():
    raw = Pool()
    pool = TransactionPool(raw)
    with pytest.raises(RuntimeError, match="failed"):
        async with pool.transaction():
            raise RuntimeError("failed")
    assert raw.connections[0].events[-1][0] == "rollback"
    async with pool.acquire() as after:
        assert after is not raw.connections[0]


@pytest.mark.asyncio
async def test_sibling_tasks_do_not_share_transaction_state():
    raw = Pool()
    pool = TransactionPool(raw)
    entered = asyncio.Event()
    owners = []

    async def write():
        async with pool.transaction() as owner:
            owners.append(owner)
            if len(owners) == 2:
                entered.set()
            await entered.wait()
            async with pool.acquire() as child:
                assert child is owner

    await asyncio.gather(write(), write())
    assert len(raw.connections) == 2
    assert owners[0] is not owners[1]


@pytest.mark.asyncio
async def test_nested_transactions_fail_explicitly():
    pool = TransactionPool(Pool())
    async with pool.transaction():
        with pytest.raises(RuntimeError, match="nest"):
            async with pool.transaction():
                pass


@pytest.mark.asyncio
async def test_standalone_pgbouncer_acquire_pins_path_and_query_in_one_transaction():
    raw = Pool()
    pool = TransactionPool(raw, 'tenant"quoted')
    async with pool.acquire() as conn:
        await conn.execute("SELECT 1")
    assert conn.queries[0][0].startswith('SET LOCAL search_path TO "tenant""quoted"')
    assert conn.events == [("begin", {}), ("commit", None)]
    await pool.close()
    assert raw.closed
