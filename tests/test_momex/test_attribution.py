"""Attribution and confirmation are preserved from ingestion through retrieval."""

import asyncio
import json

import pytest
import pytest_asyncio

import typechat

from momex import LLMConfig, Memory, MomexConfig, StorageConfig
from momex.attribution import extraction_inputs, prepare_messages
from typeagent.knowpro import convknowledge
from typeagent.knowpro.knowledge_schema import KnowledgeResponse

from .test_search_after_reopen import _FakeEmbeddingModel


@pytest_asyncio.fixture
async def memory(tmp_path, monkeypatch):
    config = MomexConfig(
        llm=LLMConfig(model="offline", api_key="dummy"),
        storage=StorageConfig(path=str(tmp_path)),
    )
    monkeypatch.setattr(config, "create_embedding_model", _FakeEmbeddingModel)
    memory = Memory("test:attribution", config)
    yield memory
    await memory.close()


@pytest.fixture
def prompts(monkeypatch):
    prompts: list[dict] = []

    class Extractor:
        async def extract(self, text):
            payload = json.loads(text.split("\n", 1)[1])
            prompts.append(payload)
            return typechat.Success(
                KnowledgeResponse(
                    entities=[],
                    actions=[],
                    inverse_actions=[],
                    topics=[payload["TARGET"]["text"]],
                )
            )

    monkeypatch.setattr(convknowledge, "KnowledgeExtractor", Extractor)
    return prompts


@pytest.mark.asyncio
async def test_only_confirmed_targets_are_extracted_without_changing_sources(
    memory, prompts
):
    await memory.add(
        [
            {"role": "assistant", "content": "Do you live in Seattle?"},
            {"role": "user", "speaker": "Alice", "content": "Yes."},
            {"role": "user", "content": "A guess", "confirmed": False},
        ],
        detect_contradictions=False,
    )
    assert len(prompts) == 1
    assert prompts[0]["TARGET"] == {"speaker": "Alice", "role": "user", "text": "Yes."}
    assert prompts[0]["CONTEXT"][0]["text"] == "Do you live in Seattle?"
    sources = await memory.transcript()
    assert [item.text for item in sources] == [
        "Do you live in Seattle?",
        "Yes.",
        "A guess",
    ]
    assert [item.status for item in sources] == [
        "unconfirmed",
        "current",
        "unconfirmed",
    ]
    current = await memory.search_by_embedding("A guess", min_score=-1)
    assert [item.ordinal for item in current] == [1]
    all_items = await memory.search_by_embedding(
        "A guess", min_score=-1, include_unconfirmed=True
    )
    assert len(all_items) == 3
    conversation = memory._conversation_required()
    refs = await conversation.semantic_refs.get_slice(
        0, await conversation.semantic_refs.size()
    )
    topics = [ref for ref in refs if ref.knowledge.knowledge_type == "topic"]
    assert [ref.range.start.message_ordinal for ref in topics] == [1]


@pytest.mark.asyncio
async def test_context_from_a_previous_write_is_available_after_reopen(memory, prompts):
    await memory.add([{"role": "assistant", "content": "Do you prefer tea?"}])
    await memory.close()
    await memory.add("Yes, please.", detect_contradictions=False)
    assert prompts[0]["CONTEXT"][0]["text"] == "Do you prefer tea?"


@pytest.mark.asyncio
async def test_all_policy_and_explicit_confirmation_are_opt_in(memory, prompts):
    await memory.add(
        [{"role": "assistant", "content": "A confirmed summary", "confirmed": True}],
        detect_contradictions=False,
    )
    await memory.add(
        [{"role": "assistant", "content": "Imported speaker"}],
        write_policy="all",
        detect_contradictions=False,
    )
    assert len(prompts) == 2
    assert all(item.status == "current" for item in await memory.transcript())


def test_context_does_not_cross_explicit_sessions():
    messages = prepare_messages(
        [
            {"content": "Other session", "session_id": "a"},
            {"content": "This session", "session_id": "b"},
        ],
        collection="test",
        timestamp="2026-01-01T00:00:00Z",
        tags=[],
        write_policy="user",
    )
    payload = extraction_inputs(messages, [], 2)[1]
    assert payload is not None
    assert json.loads(payload.split("\n", 1)[1])["CONTEXT"] == []


@pytest.mark.asyncio
async def test_mixed_concurrent_writes_do_not_disable_extraction(memory, prompts):
    await asyncio.gather(
        memory.add("unprocessed", infer=False),
        memory.add("processed", detect_contradictions=False),
    )
    assert [prompt["TARGET"]["text"] for prompt in prompts] == ["processed"]


@pytest.mark.asyncio
async def test_invalid_confirmation_is_rejected_before_writing(memory):
    with pytest.raises(ValueError, match="confirmed must be a boolean"):
        await memory.add([{"content": "invalid", "confirmed": "false"}])
    assert await memory.transcript() == []
