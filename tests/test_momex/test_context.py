"""Context budgets include metadata and citations, with exact source deduplication."""

from dataclasses import replace

import pytest

import tiktoken

from momex import (
    format_context,
    LLMConfig,
    Memory,
    MemoryManager,
    MomexConfig,
    search,
    SearchItem,
    SourceReference,
    StorageConfig,
)
from momex.search import fuse_results
from momex.visibility import MemoryStatus

from .test_search_after_reopen import _FakeEmbeddingModel


def message(ordinal, text, *, collection="user:a", status: MemoryStatus = "current"):
    source = SourceReference(
        text=text,
        ordinal=ordinal,
        collection=collection,
        source_id=f"source-{ordinal}",
        timestamp="2024-01-01T12:00:00Z",
        speaker="Alice",
        status=status,
    )
    return SearchItem(
        type="message",
        text=text,
        score=1,
        raw=None,
        ordinal=ordinal,
        collection=collection,
        status=status,
        sources=(source,),
    )


@pytest.mark.parametrize("budget", [0, 1, 8, 31, 100, 500])
def test_exact_token_budget_includes_headers_and_unicode(budget):
    body = "\u4f60\u597d Python <|endoftext|> " * 100
    context = format_context([message(0, body)], token_budget=budget)
    encoding = tiktoken.get_encoding(context.encoding_name)
    assert context.token_count == len(
        encoding.encode(context.text, disallowed_special=())
    )
    assert context.token_count <= budget
    assert "\ufffd" not in context.text
    assert context.truncated
    assert len(context.citations) == context.text.count("[m1]")


def test_overlapping_windows_are_merged_by_source_not_rendered_text():
    turns = [message(i, f"turn-{i}") for i in range(3)]
    left = replace(
        turns[0], text="turn-0\nturn-1", sources=turns[0].sources + turns[1].sources
    )
    right = replace(
        turns[1], text="turn-1\nturn-2", sources=turns[1].sources + turns[2].sources
    )
    context = format_context([left, right])
    assert all(context.text.count(f"turn-{i}") == 1 for i in range(3))
    assert [citation.label for citation in context.citations] == ["m1", "m2", "m3"]
    assert context.citations[1].sources[0].source_id == "source-1"
    assert not context.truncated


def test_same_text_in_different_collections_or_states_stays_distinct():
    items = [
        message(0, "same"),
        message(0, "same", collection="user:b"),
        message(0, "same", status="superseded"),
    ]
    fused = fuse_results(items, limit=10)
    assert len(fused) == 3
    context = format_context(fused)
    assert context.text.count("same") == 3
    assert "[superseded]" in context.text
    assert "user:a" in context.text and "user:b" in context.text


def test_fusion_keeps_all_provenance_for_identical_current_knowledge():
    first = replace(message(0, "fact"), type="action")
    second = replace(message(1, "fact"), type="action")
    (fused,) = fuse_results([first, second], limit=10)
    assert len(fused.sources) == 2
    context = format_context([fused])
    assert len(context.citations) == 1
    assert len(context.citations[0].sources) == 2


def test_unconfirmed_context_is_never_labeled_as_current():
    context = format_context([message(0, "a guess", status="unconfirmed")])
    assert "[unconfirmed]" in context.text
    assert "[current]" not in context.text


def test_negative_budget_is_an_error():
    with pytest.raises(ValueError, match="token_budget"):
        format_context([], token_budget=-1)


@pytest.mark.asyncio
async def test_real_neighbor_results_and_transcripts_have_citable_sources(
    tmp_path, monkeypatch
):
    config = MomexConfig(
        llm=LLMConfig(model="offline", api_key="dummy"),
        storage=StorageConfig(path=str(tmp_path)),
    )
    monkeypatch.setattr(config, "create_embedding_model", _FakeEmbeddingModel)
    async with Memory("user:source", config) as memory:
        await memory.add("I prefer tea", infer=False)
        source = (await memory.transcript())[0]
        result = (await memory.search_by_embedding("I prefer tea"))[0]
        assert result.collection == "user:source"
        assert result.source_id == source.source_id
        assert result.sources[0].ordinal == 0
        assert result.sources[0].role == "user"
        assert format_context([result]).citations[0].sources == result.sources


@pytest.mark.asyncio
async def test_global_result_limit_preserves_default_per_collection_behavior(
    monkeypatch,
):
    monkeypatch.setattr(
        MemoryManager, "list_collections", lambda self, prefix: ["user:a", "user:b"]
    )

    async def query(self, text, **options):
        score = 0.02 if self.collection == "user:a" else 0.03
        return [
            replace(
                message(i, f"{self.collection}-{i}", collection=self.collection),
                fusion_score=score - i * 0.001,
            )
            for i in range(options["limit"])
        ]

    monkeypatch.setattr(Memory, "search", query)
    default = await search("user", "anything", limit=2)
    assert sum(len(items) for _, items in default) == 4
    limited = await search("user", "anything", limit=2, total_limit=1)
    assert len(limited) == 1 and limited[0][0] == "user:b"
    assert sum(len(items) for _, items in limited) == 1
    assert await search("user", "anything", total_limit=0) == []
