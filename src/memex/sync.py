"""Synchronous wrapper utilities for async methods."""

from __future__ import annotations

import asyncio
import functools
from typing import Any, Callable, Coroutine, TypeVar

T = TypeVar("T")


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine synchronously.

    Handles the case where an event loop may or may not be running.

    Args:
        coro: The coroutine to run.

    Returns:
        The result of the coroutine.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, create a new one
        return asyncio.run(coro)
    else:
        # Loop is running, use nest_asyncio pattern or run in thread
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()


def sync_wrapper(async_func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., T]:
    """Decorator to create a synchronous version of an async function.

    Args:
        async_func: The async function to wrap.

    Returns:
        A synchronous function that runs the async function.
    """

    @functools.wraps(async_func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        return run_sync(async_func(*args, **kwargs))

    return wrapper
