"""Offline tests for LOCOMO context construction."""

from benchmarks.locomo import render_hybrid_context
from momex import SearchItem


def _item(text: str, ordinal: int) -> SearchItem:
    return SearchItem(
        type="message",
        text=text,
        score=0.5,
        raw=object(),
        ordinal=ordinal,
    )


def test_hybrid_context_prioritizes_retrieval_then_keeps_every_turn():
    focused = [_item("important", 1)]
    transcript = [_item("turn zero", 0), _item("turn one", 1)]

    context = render_hybrid_context(focused, 1, transcript)

    assert context.index("important") < context.index("turn zero")
    assert "turn zero" in context
    assert "turn one" in context
