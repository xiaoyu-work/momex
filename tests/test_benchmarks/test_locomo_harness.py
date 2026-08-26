"""Offline tests for LOCOMO context construction."""

import argparse
import json

from benchmarks.locomo import (
    Question,
    render_hybrid_context,
    Result,
    save_results,
)
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


def test_question_level_results_are_persisted(tmp_path):
    path = tmp_path / "results.json"
    args = argparse.Namespace(
        output=path,
        answer_prompt="direct",
        categories=[1],
    )
    question = Question("What happened?", "The answer", 1, ["D1:1"])
    result = Result(
        question,
        "The answer",
        1.0,
        True,
        1,
        [_item("evidence", 3)],
    )

    save_results(path, args, [result])

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["results"][0]["judged"] is True
    assert data["results"][0]["context"][0]["ordinal"] == 3
