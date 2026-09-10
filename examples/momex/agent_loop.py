"""Use the Momex SDK from an application that owns its reply model.

This example is a Python helper, not a service or installed command. The caller
supplies an async reply function and owns all model and Memory resources.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from momex import AddResult, ContextResult, format_context, Memory

ReplyFunction = Callable[[str, str], Awaitable[str]]


@dataclass
class TurnResult:
    reply: str
    context: ContextResult
    writeback: AddResult | None


async def run_turn(
    memory: Memory,
    generate_reply: ReplyFunction,
    question: str,
    *,
    token_budget: int = 2048,
    limit: int = 10,
    neighbors: int = 1,
    persist: bool = True,
) -> TurnResult:
    """Retrieve evidence, call generate_reply(question, evidence), and write back.

    The application decides how to prompt its model and manages its clients.
    Assistant output remains unconfirmed context under Momex's user policy.
    """
    if not question.strip():
        raise ValueError("question must not be empty")
    if token_budget < 0 or limit < 0 or neighbors < 0:
        raise ValueError("budgets, limit and neighbors cannot be negative")

    items = await memory.search(question, limit=limit, neighbors=neighbors)
    context = format_context(items, token_budget=token_budget)
    reply = await generate_reply(question, context.text)
    if not reply.strip():
        raise ValueError("The reply function returned empty content")

    writeback = None
    if persist:
        writeback = await memory.add(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": reply},
            ],
            write_policy="user",
        )
    return TurnResult(reply=reply, context=context, writeback=writeback)
