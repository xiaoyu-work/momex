"""Ask whether multi-hop failures are retrieval, reading, or measurement.

Multi-hop sits near 42% under every retrieval strategy tried, which is either
a real limit of top-k retrieval over independently scored items or an artefact
of how this harness scores. Three things separate those:

  Evidence coverage. Multi-hop questions cite several turns. If most of them
  never arrive, the ceiling is retrieval and nothing downstream matters.

  Accuracy given full coverage. If every cited turn is in the context and the
  answer is still wrong, the loss is in reading, not retrieval.

  Judge disagreement. "clarinet" against a gold answer of "clarinet and
  violin" is a partial answer; scoring it wrong is correct, but scoring a
  complete answer wrong would be a harness fault. Printed so it can be read.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
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
from momex import Memory, MomexConfig, StorageConfig  # noqa: E402


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=int, default=120)
    parser.add_argument("--category", type=int, default=1, help="1 = multi-hop")
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--conversation", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show", type=int, default=6)
    args = parser.parse_args()

    config = MomexConfig.from_env()
    config.validate()
    config.storage = StorageConfig(path=str(ROOT / "benchmarks" / "store"))

    raw = json.loads(DATA.read_text(encoding="utf-8"))[args.conversation]
    texts = evidence_texts(raw)
    data = load_conversations(DATA)[args.conversation]

    memory = Memory(collection=f"locomo:{data.sample_id}", config=config)
    stats = await memory.stats()
    if not stats["total_messages"]:
        print("Collection not ingested.", file=sys.stderr)
        return 2

    questions = [
        q
        for q in select_questions(
            data.questions, set(DEFAULT_CATEGORIES), args.questions, args.seed
        )
        if q.evidence and q.category == args.category
    ]
    llm = config.create_llm()

    name = CATEGORY_NAMES.get(args.category, str(args.category))
    print(
        f"{data.sample_id}: {len(questions)} {name} questions, budget {args.budget}\n"
    )

    # Accuracy split by how much of the cited evidence made it into context.
    buckets: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    evidence_counts: list[int] = []
    found_counts: list[int] = []
    failures: list[tuple] = []

    for i, question in enumerate(questions, 1):
        wanted = [texts[e] for e in question.evidence if e in texts]
        if not wanted:
            continue

        items = await memory.search(question.question, limit=args.budget)
        found, total = covers(items, wanted)
        evidence_counts.append(total)
        found_counts.append(found)

        prompt = ANSWER_PROMPT.format(
            context=render_context(items, args.budget), question=question.question
        )
        response = await llm.complete(prompt, max_tokens=80)
        predicted = response.content.strip()
        correct = await judge_answer(llm, question, predicted)

        bucket = (
            "all evidence"
            if found == total
            else ("some evidence" if found else "no evidence")
        )
        buckets[bucket][0] += int(correct)
        buckets[bucket][1] += 1

        if not correct and len(failures) < args.show:
            failures.append((question, predicted, found, total))

        if i % 10 == 0:
            print(f"  {i}/{len(questions)}", flush=True)

    n = len(evidence_counts)
    print(f"\n  cited evidence turns per question: {sum(evidence_counts) / n:.1f}")
    print(
        f"  of those retrieved:                {sum(found_counts) / sum(evidence_counts):.1%}"
    )

    print(f"\n  accuracy by how much evidence arrived")
    for label in ("all evidence", "some evidence", "no evidence"):
        correct, seen = buckets[label]
        if seen:
            print(f"    {label:<16} {correct}/{seen} = {correct / seen:>6.1%}")

    total_correct = sum(b[0] for b in buckets.values())
    print(f"\n  overall {total_correct}/{n} = {total_correct / n:.1%}")

    print(f"\n  failures (read these: partial answers are correctly wrong)")
    for question, predicted, found, total in failures:
        print(f"\n  Q: {question.question}")
        print(f"     gold: {question.answer}")
        print(f"     got:  {predicted}")
        print(f"     evidence retrieved: {found}/{total}")

    await memory.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
