"""Diagnose why structured knowledge barely reaches the reader.

Runs the same questions through several context-selection strategies against an
already-ingested LOCOMO collection, so the only thing changing is what Momex
hands over. Answers three things the benchmark raised:

  1. Does the fusion step systematically favour messages? Only messages can be
     found by both retrievers, so only messages can collect two RRF votes.
  2. Is the knowledge worth surfacing at all, or is message text sufficient?
  3. Does search()'s embedding threshold silently drop the whole embedding
     half for question-shaped queries?

    uv run python benchmarks/diagnose.py --questions 40
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from benchmarks.locomo import (  # noqa: E402
    ANSWER_PROMPT,
    DATA,
    DEFAULT_CATEGORIES,
    judge_answer,
    load_conversations,
    render_context,
    select_questions,
    token_f1,
)
from momex import Memory, MomexConfig, SearchItem, StorageConfig  # noqa: E402
from momex.search import fuse_results  # noqa: E402


def knowledge_only(items: list[SearchItem]) -> list[SearchItem]:
    return [i for i in items if i.type != "message"]


def messages_only(items: list[SearchItem]) -> list[SearchItem]:
    return [i for i in items if i.type == "message"]


async def gather_context(
    memory: Memory, question: str, limit: int
) -> dict[str, list[SearchItem]]:
    """Build every candidate context for one question, from one retrieval pass."""
    structured = await memory._search_structured(question, limit=limit)

    # The embedding half at search()'s own threshold, and without one.
    strict = await memory.search_by_embedding(question, limit=limit, min_score=0.3)
    lax = await memory.search_by_embedding(question, limit=limit, min_score=0.0)

    current = fuse_results(structured, strict, limit=limit)
    unthresholded = fuse_results(structured, lax, limit=limit)

    # Balanced: fuse each type against its own competition, then give knowledge
    # a guaranteed share of the budget instead of letting it lose a vote it
    # could never have won.
    half = max(1, limit // 2)
    knowledge = knowledge_only(structured)[:half]
    messages = fuse_results(
        messages_only(structured), lax, limit=limit - len(knowledge)
    )
    balanced = knowledge + messages

    return {
        "current (search)": current,
        "no-threshold": unthresholded,
        "balanced": balanced,
        "messages-only": messages_only(unthresholded)[:limit],
        "knowledge-only": knowledge_only(structured)[:limit],
        "structured-only": structured[:limit],
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--conversation", type=int, default=0)
    args = parser.parse_args()

    config = MomexConfig.from_env()
    config.validate()
    config.storage = StorageConfig(path=str(ROOT / "benchmarks" / "store"))

    conversation = load_conversations(DATA)[args.conversation]
    memory = Memory(collection=f"locomo:{conversation.sample_id}", config=config)
    stats = await memory.stats()
    if not stats["total_messages"]:
        print(
            "Collection not ingested. Run benchmarks/locomo.py first.", file=sys.stderr
        )
        return 2

    questions = select_questions(
        conversation.questions, set(DEFAULT_CATEGORIES), args.questions, args.seed
    )
    llm = config.create_llm()

    print(
        f"{conversation.sample_id}: {stats['total_messages']} messages, "
        f"{stats['total_semantic_refs']} refs"
    )
    print(f"{len(questions)} questions, top_k={args.top_k}\n")

    strategies: dict[str, dict] = {}
    empty_embedding = 0

    for index, question in enumerate(questions, 1):
        contexts = await gather_context(memory, question.question, args.top_k)

        strict = await memory.search_by_embedding(
            question.question, limit=args.top_k, min_score=0.3
        )
        if not strict:
            empty_embedding += 1

        for name, items in contexts.items():
            bucket = strategies.setdefault(
                name, {"correct": 0, "f1": 0.0, "types": Counter(), "n": 0}
            )
            if not items:
                predicted = "NO_ANSWER"
            else:
                prompt = ANSWER_PROMPT.format(
                    context=render_context(items, args.top_k),
                    question=question.question,
                )
                response = await llm.complete(prompt, max_tokens=80)
                predicted = response.content.strip()

            correct = await judge_answer(llm, question, predicted)
            bucket["correct"] += int(correct)
            bucket["f1"] += token_f1(predicted, question.answer)
            bucket["types"].update(i.type for i in items)
            bucket["n"] += 1

        if index % 5 == 0:
            print(f"  {index}/{len(questions)}", flush=True)

    print(f"\n  {'strategy':<18} {'judge':>7} {'F1':>7} {'items':>6}  composition")
    for name, bucket in strategies.items():
        total = sum(bucket["types"].values())
        share = (
            ", ".join(f"{t} {c / total:.0%}" for t, c in bucket["types"].most_common(4))
            if total
            else "-"
        )
        print(
            f"  {name:<18} {bucket['correct'] / bucket['n']:>6.1%} "
            f"{bucket['f1'] / bucket['n']:>7.3f} {total / bucket['n']:>6.1f}  {share}"
        )

    print(
        f"\n  embedding half returned nothing at min_score=0.3 for "
        f"{empty_embedding}/{len(questions)} questions "
        f"({empty_embedding / len(questions):.0%})"
    )

    await memory.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
