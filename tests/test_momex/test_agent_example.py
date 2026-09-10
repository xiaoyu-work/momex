"""The SDK example uses caller-owned callbacks and real offline memory writes."""

from collections.abc import Mapping, Sequence
import json
from unittest.mock import AsyncMock

import httpx
import pytest

import tiktoken
import typechat

from examples.momex.agent_loop import run_turn
from momex import LLMConfig, Memory, MomexConfig, StorageConfig
from typeagent.knowpro import convknowledge
from typeagent.knowpro.knowledge_schema import KnowledgeResponse

from .test_search_after_reopen import _FakeEmbeddingModel


@pytest.fixture
def config(tmp_path, monkeypatch):
    config = MomexConfig(
        llm=LLMConfig(model="offline", api_key="dummy"),
        storage=StorageConfig(path=str(tmp_path / "data")),
    )
    monkeypatch.setattr(config, "create_embedding_model", _FakeEmbeddingModel)
    monkeypatch.setattr(Memory, "_search_structured", AsyncMock(return_value=[]))

    async def no_http(*args, **kwargs):
        raise AssertionError("SDK example tests must not make HTTP requests")

    monkeypatch.setattr(httpx.AsyncClient, "send", no_http)

    class Extractor:
        async def extract(self, prompt):
            target = json.loads(prompt.split("\n", 1)[1])["TARGET"]
            return typechat.Success(
                KnowledgeResponse(
                    entities=[],
                    actions=[],
                    inverse_actions=[],
                    topics=[target["text"]],
                )
            )

    monkeypatch.setattr(convknowledge, "KnowledgeExtractor", Extractor)
    return config


@pytest.mark.asyncio
async def test_retrieve_reply_writeback_order_budget_and_confirmation(
    config, monkeypatch
):
    events: list[str] = []
    async with Memory("user:demo", config) as memory:
        await memory.add("I prefer Python. " * 500, source_id="seed", infer=False)
        source = await memory.get_source("seed")
        assert source is not None
        monkeypatch.setattr(
            memory, "_search_structured", AsyncMock(return_value=[source])
        )
        original_search = memory.search
        original_add = memory.add

        async def search(question: str, **kwargs):
            events.append("search")
            return await original_search(question, **kwargs)

        async def add(messages: str | Sequence[Mapping[str, object]], **kwargs):
            events.append("writeback")
            assert kwargs["write_policy"] == "user"
            return await original_add(messages, **kwargs)

        async def reply(question: str, evidence: str) -> str:
            events.append("reply")
            assert question == "Which language do I prefer?"
            assert "[m1]" in evidence
            assert len(tiktoken.get_encoding("cl100k_base").encode(evidence)) <= 80
            return "You prefer Python [m1]."

        monkeypatch.setattr(memory, "search", search)
        monkeypatch.setattr(memory, "add", add)
        result = await run_turn(
            memory, reply, "Which language do I prefer?", token_budget=80
        )
        assert events == ["search", "reply", "writeback"]
        assert result.context.truncated
        assert result.context.citations[0].sources[0].source_id == "seed"
        assert result.writeback is not None and result.writeback.messages_added == 2
        assert memory.is_initialized
        transcript = await memory.transcript()
        assert [item.status for item in transcript[-2:]] == ["current", "unconfirmed"]
        assert [item.role for item in transcript[-2:]] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_read_only_turn_does_not_write_back(config):
    async def reply(question: str, evidence: str) -> str:
        return "No stored evidence."

    async with Memory("user:demo", config) as memory:
        result = await run_turn(memory, reply, "A question", persist=False)
        assert result.writeback is None
        assert await memory.transcript() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["exception", "empty"])
async def test_failed_reply_is_not_written_and_caller_keeps_memory(config, failure):
    async def reply(question: str, evidence: str) -> str:
        if failure == "exception":
            raise RuntimeError("reply failed")
        return " "

    async with Memory("user:demo", config) as memory:
        with pytest.raises((RuntimeError, ValueError)):
            await run_turn(memory, reply, "A question")
        assert memory.is_initialized
        assert await memory.transcript() == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options",
    [{"question": ""}, {"token_budget": -1}, {"limit": -1}, {"neighbors": -1}],
)
async def test_invalid_turns_fail_before_retrieval_or_reply(
    config, monkeypatch, options
):
    memory = Memory("user:demo", config)
    search = AsyncMock(return_value=[])
    reply = AsyncMock(return_value="reply")
    monkeypatch.setattr(memory, "search", search)
    with pytest.raises(ValueError):
        await run_turn(memory, reply, **{"question": "A question", **options})
    search.assert_not_awaited()
    reply.assert_not_awaited()
    assert not memory.is_initialized
