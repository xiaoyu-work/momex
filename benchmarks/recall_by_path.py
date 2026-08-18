"""Locate where evidence recall is lost, path by path.

benchmarks/recall.py shows that Momex answers 82% of questions whose evidence
it retrieves and that it retrieves all the evidence for only two thirds of
them, so the loss is in retrieval. This narrows it further: recall is measured
for each path on its own, at a cut-off far past what search() uses.

If a path reaches high recall at k=100, the evidence is findable and the loss
is in ranking or fusion. If no path does, the evidence is not reachable at all
and no amount of budget or reordering will help.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from benchmarks.locomo import (  # noqa: E402
    DATA,
    DEFAULT_CATEGORIES,
    load_conversations,
    select_questions,
)
from benchmarks.recall import covers, evidence_texts  # noqa: E402
from momex import Memory, MomexConfig, StorageConfig  # noqa: E402


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=int, default=60)
    parser.add_argument("--conversation", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = MomexConfig.from_env()
    config.validate()
    config.storage = StorageConfig(path=str(ROOT / "benchmarks" / "store"))

    raw = json.loads(DATA.read_text(encoding="utf-8"))[args.conversation]
    texts = evidence_texts(raw)
    conversation = load_conversations(DATA)[args.conversation]

    memory = Memory(collection=f"locomo:{conversation.sample_id}", config=config)
    await memory._ensure_initialized()
    conv = memory._conversation_required()

    index = conv.secondary_indexes.message_index
    print(f"messages stored:   {await conv.messages.size()}")
    print(f"message index size:{await index.size()}")
    print(f"semantic refs:     {await conv.semantic_refs.size()}\n")

    questions = [
        q
        for q in select_questions(
            conversation.questions, set(DEFAULT_CATEGORIES), args.questions, args.seed
        )
        if q.evidence
    ]

    paths = {
        "structured k=20": lambda q: memory._search_structured(q, limit=20),
        "structured k=100": lambda q: memory._search_structured(q, limit=100),
        "embedding k=20": lambda q: memory.search_by_embedding(
            q, limit=20, min_score=0.0
        ),
        "embedding k=100": lambda q: memory.search_by_embedding(
            q, limit=100, min_score=0.0
        ),
        "embedding k=419": lambda q: memory.search_by_embedding(
            q, limit=419, min_score=0.0
        ),
        "search() k=20": lambda q: memory.search(q, limit=20),
    }
    totals = {name: [0, 0] for name in paths}

    for i, question in enumerate(questions, 1):
        wanted = [texts[e] for e in question.evidence if e in texts]
        if not wanted:
            continue
        for name, run in paths.items():
            items = await run(question.question)
            found, total = covers(items, wanted)
            totals[name][0] += found
            totals[name][1] += total
        if i % 10 == 0:
            print(f"  {i}/{len(questions)}", flush=True)

    print(f"\n  {'path':<18} {'evidence turns found':>22}")
    for name, (found, total) in totals.items():
        print(f"  {name:<18} {found / total:>21.1%}  ({found}/{total})")

    await memory.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
