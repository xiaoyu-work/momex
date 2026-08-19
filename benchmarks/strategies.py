#!/usr/bin/env python
"""Compare retrieval strategies on the same retrieved candidates.

    uv run python benchmarks/strategies.py --questions 60
    uv run python benchmarks/strategies.py --questions 60 --budget 20 --accuracy

Every strategy is scored on evidence recall at one fixed presentation budget,
so they differ only in how they spend it, not in how much they get. LOCOMO
labels the dialogue turns containing each answer, which makes recall
deterministic: no judge, no sampling noise, and cheap enough to compare a dozen
designs before spending anything on generation.

To keep the comparison fair the raw candidates are fetched once per question --
one structured search and one embedding search, both wide -- and each strategy
is then a pure function over those two lists. A strategy that looks better here
is better at spending the budget, not luckier with retrieval.

Needs a collection already ingested by benchmarks/locomo.py.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from benchmarks.locomo import (  # noqa: E402
    ANSWER_PROMPT,
    CATEGORY_NAMES,
    DATA,
    DEFAULT_CATEGORIES,
    judge_answer,
    load_conversations,
    render_context,
    select_questions,
)
from benchmarks.recall import covers, evidence_texts  # noqa: E402
from momex import Memory, MomexConfig, SearchItem, StorageConfig  # noqa: E402
from momex.search import fuse_results, RRF_K  # noqa: E402

# How wide to fetch before any strategy runs. Well past every budget tested,
# so over-fetching strategies are not limited by the harness.
FETCH = 200


@dataclass
class Candidates:
    """One question's raw retrieval, shared by every strategy."""

    structured: list[SearchItem]
    embedding: list[SearchItem]

    def embedding_above(self, min_score: float) -> list[SearchItem]:
        return [i for i in self.embedding if i.score >= min_score]


def _messages(items: list[SearchItem]) -> list[SearchItem]:
    return [i for i in items if i.type == "message"]


def _knowledge(items: list[SearchItem]) -> list[SearchItem]:
    return [i for i in items if i.type != "message"]


def weighted_fuse(
    lists: list[tuple[list[SearchItem], float]], limit: int
) -> list[SearchItem]:
    """Reciprocal rank fusion with a weight per source list.

    Plain RRF treats its inputs as equally trustworthy. Measured on LOCOMO the
    embedding path finds 64% of evidence turns and the structured path 9%, so
    equal treatment spends half the budget on the weaker one.
    """
    best: dict[str, SearchItem] = {}
    scores: dict[str, float] = {}
    for items, weight in lists:
        seen: set[str] = set()
        for rank, item in enumerate(items):
            if item.text in seen:
                continue
            seen.add(item.text)
            scores[item.text] = scores.get(item.text, 0.0) + weight / (RRF_K + rank + 1)
            best.setdefault(item.text, item)
    for text, item in best.items():
        item.fusion_score = scores[text]
    return sorted(
        best.values(),
        key=lambda i: (i.fusion_score or 0.0, i.score),
        reverse=True,
    )[:limit]


def quota(
    messages: list[SearchItem], knowledge: list[SearchItem], limit: int, share: float
) -> list[SearchItem]:
    """Fill the budget with a fixed share of knowledge, rest messages.

    Allocating explicitly rather than letting the two compete: only messages
    can be found by both retrievers, so pooled RRF hands them an advantage
    that has nothing to do with relevance.
    """
    k_slots = int(limit * share)
    taken_k = knowledge[:k_slots]
    return _messages(messages)[: limit - len(taken_k)] + taken_k


# Each strategy maps candidates and a budget to what the reader would see.
STRATEGIES: dict[str, object] = {
    # What search() does today: both lists cut to the budget, pooled RRF.
    "current": lambda c, n: fuse_results(
        c.structured[:n], c.embedding_above(0.3)[:n], limit=n
    ),
    # Same, without the embedding score threshold.
    "no-threshold": lambda c, n: fuse_results(
        c.structured[:n], c.embedding[:n], limit=n
    ),
    # Decouple recall from presentation: fetch wide, fuse, then trim.
    "overfetch-3x": lambda c, n: fuse_results(
        c.structured[: n * 3], c.embedding[: n * 3], limit=n
    ),
    "overfetch-10x": lambda c, n: fuse_results(
        c.structured[: n * 10], c.embedding[: n * 10], limit=n
    ),
    # Drop the structured path from retrieval entirely.
    "messages-only": lambda c, n: c.embedding[:n],
    # Weight the two paths by roughly their measured recall.
    "weighted-rrf": lambda c, n: weighted_fuse(
        [(c.embedding[: n * 3], 0.85), (c.structured[: n * 3], 0.15)], limit=n
    ),
    # Explicit budget split instead of a contest the knowledge cannot win.
    "quota-25%-knowledge": lambda c, n: quota(
        c.embedding, _knowledge(c.structured), n, 0.25
    ),
    "quota-50%-knowledge": lambda c, n: quota(
        c.embedding, _knowledge(c.structured), n, 0.50
    ),
    # Keep both retrievers, but let them compete only over messages, and add
    # knowledge on top of a fixed message budget.
    "msg-fusion+knowledge": lambda c, n: (
        fuse_results(
            _messages(c.structured[: n * 3]), c.embedding[: n * 3], limit=int(n * 0.75)
        )
        + _knowledge(c.structured)[: n - int(n * 0.75)]
    ),
}


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=int, default=60)
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[10, 20, 50],
        help="Presentation budgets to sweep. A strategy that wins at one "
        "budget can lose at another, so one number is not an answer.",
    )
    parser.add_argument("--conversation", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--accuracy",
        action="store_true",
        help="Also score the best strategies end to end (costly).",
    )
    parser.add_argument("--accuracy-budget", type=int, default=20)
    parser.add_argument("--accuracy-top", type=int, default=3)
    args = parser.parse_args()

    config = MomexConfig.from_env()
    config.validate()
    config.storage = StorageConfig(path=str(ROOT / "benchmarks" / "store"))

    raw = json.loads(DATA.read_text(encoding="utf-8"))[args.conversation]
    texts = evidence_texts(raw)
    conversation = load_conversations(DATA)[args.conversation]

    memory = Memory(collection=f"locomo:{conversation.sample_id}", config=config)
    stats = await memory.stats()
    if not stats["total_messages"]:
        print(
            "Collection not ingested. Run benchmarks/locomo.py first.", file=sys.stderr
        )
        return 2

    questions = [
        q
        for q in select_questions(
            conversation.questions, set(DEFAULT_CATEGORIES), args.questions, args.seed
        )
        if q.evidence
    ]

    print(
        f"{conversation.sample_id}: {stats['total_messages']} messages, "
        f"{stats['total_semantic_refs']} refs"
    )
    print(f"{len(questions)} questions, budgets {args.budgets}\n")

    # recall[budget][strategy] = [turns found, turns wanted]
    recall: dict[int, dict[str, list[int]]] = {
        b: {name: [0, 0] for name in STRATEGIES} for b in args.budgets
    }
    per_category: dict[str, dict[int, list[int]]] = {
        name: defaultdict(lambda: [0, 0]) for name in STRATEGIES
    }
    contexts: dict[str, list[tuple]] = {name: [] for name in STRATEGIES}

    for index, question in enumerate(questions, 1):
        wanted = [texts[e] for e in question.evidence if e in texts]
        if not wanted:
            continue

        # Fetched once and shared, so strategies are compared on how they
        # spend a budget rather than on what they happened to retrieve.
        candidates = Candidates(
            structured=await memory._search_structured(question.question, limit=FETCH),
            embedding=await memory.search_by_embedding(
                question.question, limit=FETCH, min_score=0.0
            ),
        )

        for name, strategy in STRATEGIES.items():
            for budget in args.budgets:
                items = strategy(candidates, budget)  # type: ignore[operator]
                found, total = covers(items, wanted)
                recall[budget][name][0] += found
                recall[budget][name][1] += total
                if budget == args.accuracy_budget:
                    per_category[name][question.category][0] += found
                    per_category[name][question.category][1] += total
                    contexts[name].append((question, items))

        if index % 10 == 0:
            print(f"  {index}/{len(questions)}", flush=True)

    def rate(bucket: list[int]) -> float:
        return bucket[0] / bucket[1] if bucket[1] else 0.0

    order = sorted(STRATEGIES, key=lambda n: -rate(recall[max(args.budgets)][n]))

    print("\n  evidence recall by presentation budget")
    print(
        "  "
        + f"{'strategy':<22}"
        + "".join(f"{'k=' + str(b):>9}" for b in args.budgets)
    )
    for name in order:
        row = f"  {name:<22}"
        for budget in args.budgets:
            row += f"{rate(recall[budget][name]):>8.1%} "
        print(row)

    categories = sorted({q.category for q in questions})
    print(f"\n  by category at k={args.accuracy_budget}")
    print(
        "  "
        + f"{'strategy':<22}"
        + "".join(f"{CATEGORY_NAMES.get(c, str(c))[:9]:>11}" for c in categories)
    )
    for name in order:
        row = f"  {name:<22}"
        for category in categories:
            row += f"{rate(per_category[name][category]):>10.1%} "
        print(row)

    if not args.accuracy:
        print("\n  (--accuracy also scores the top strategies end to end)")
        await memory.close()
        return 0

    ranked_at_budget = sorted(
        STRATEGIES, key=lambda n: -rate(recall[args.accuracy_budget][n])
    )
    winners = ranked_at_budget[: args.accuracy_top]
    if "current" not in winners:
        winners.append("current")

    print(f"\n  scoring end to end at k={args.accuracy_budget}: {', '.join(winners)}")
    llm = config.create_llm()
    accuracy: dict[str, list[int]] = {name: [0, 0] for name in winners}

    for name in winners:
        for question, items in contexts[name]:
            prompt = ANSWER_PROMPT.format(
                context=render_context(items, args.accuracy_budget),
                question=question.question,
            )
            response = await llm.complete(prompt, max_tokens=80)
            correct = await judge_answer(llm, question, response.content.strip())
            accuracy[name][0] += int(correct)
            accuracy[name][1] += 1
        done, seen = accuracy[name]
        print(f"    {name:<22} {done}/{seen} = {done / seen:.1%}", flush=True)

    print(f"\n  {'strategy':<22}{'recall':>8}{'accuracy':>10}")
    for name in sorted(winners, key=lambda n: -rate(accuracy[n])):
        print(
            f"  {name:<22}{rate(recall[args.accuracy_budget][name]):>7.1%} "
            f"{rate(accuracy[name]):>9.1%}"
        )

    await memory.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
