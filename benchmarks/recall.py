#!/usr/bin/env python
"""Decompose Momex's LOCOMO score into retrieval and reading.

    uv run python benchmarks/recall.py --questions 60

A wrong answer has exactly two causes, and they call for opposite fixes:

  The evidence never reached the reader. That is a retrieval failure, and it
  is a hard ceiling -- no prompt, no model, no reader can recover a turn that
  was not returned.

  The evidence reached the reader and the answer was still wrong. That is a
  reading failure, and it is cheap to fix by comparison.

LOCOMO labels every question with the dialogue turns containing its answer, so
this is directly measurable rather than a matter of opinion. Reports recall at
several cut-offs, and accuracy conditioned on whether the evidence was there.

Needs a collection already ingested by benchmarks/locomo.py.
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
from momex import Memory, MomexConfig, SearchItem, StorageConfig  # noqa: E402


def evidence_texts(sample: dict) -> dict[str, str]:
    """Map each dialogue id (D1:3) to the turn's text."""
    conv = sample["conversation"]
    keys = [k for k in conv if k.startswith("session_") and not k.endswith("date_time")]
    return {
        turn["dia_id"]: turn["text"]
        for key in keys
        for turn in conv[key]
        if turn.get("dia_id") and turn.get("text")
    }


def covers(items: list[SearchItem], wanted: list[str]) -> tuple[int, int]:
    """How many of the wanted evidence texts appear in the retrieved items.

    Compared on a prefix rather than exactly: ingestion prefixes each turn
    with its timestamp and speaker, and a long turn may be chunked, so an
    equality test would report misses that are really formatting.
    """
    blob = "\n".join(i.text for i in items)
    found = sum(1 for text in wanted if text[:60] in blob)
    return found, len(wanted)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=int, default=60)
    parser.add_argument("--conversation", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cutoffs", type=int, nargs="*", default=[5, 10, 20, 50])
    parser.add_argument("--judge", action="store_true", default=True)
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

    questions = select_questions(
        conversation.questions, set(DEFAULT_CATEGORIES), args.questions, args.seed
    )
    questions = [q for q in questions if q.evidence]
    llm = config.create_llm()
    biggest = max(args.cutoffs)

    print(
        f"{conversation.sample_id}: {stats['total_messages']} messages, "
        f"{stats['total_semantic_refs']} refs"
    )
    print(f"{len(questions)} questions with labelled evidence\n")

    # recall[cutoff] = (turns found, turns wanted, questions fully covered)
    recall: dict[int, list[int]] = {c: [0, 0, 0] for c in args.cutoffs}
    by_category: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    conditional = {"covered": [0, 0], "uncovered": [0, 0]}

    for index, question in enumerate(questions, 1):
        wanted = [texts[e] for e in question.evidence if e in texts]
        if not wanted:
            continue

        items = await memory.search(question.question, limit=biggest)

        for cutoff in args.cutoffs:
            found, total = covers(items[:cutoff], wanted)
            recall[cutoff][0] += found
            recall[cutoff][1] += total
            recall[cutoff][2] += int(found == total)

        # Accuracy conditioned on whether the evidence was there, at the
        # cut-off the benchmark actually uses.
        default_items = items[:20]
        found, total = covers(default_items, wanted)
        bucket = "covered" if found == total else "uncovered"

        prompt = ANSWER_PROMPT.format(
            context=render_context(default_items, 20), question=question.question
        )
        response = await llm.complete(prompt, max_tokens=80)
        correct = await judge_answer(llm, question, response.content.strip())

        conditional[bucket][0] += int(correct)
        conditional[bucket][1] += 1
        by_category[question.category][0] += found
        by_category[question.category][1] += total

        if index % 10 == 0:
            print(f"  {index}/{len(questions)}", flush=True)

    print(f"\n  evidence recall")
    print(f"  {'cut-off':>8} {'turns':>9} {'questions fully covered':>26}")
    for cutoff in args.cutoffs:
        found, total, complete = recall[cutoff]
        print(
            f"  {cutoff:>8} {found / total:>8.1%} "
            f"{complete / len(questions):>25.1%}"
        )

    print(f"\n  evidence recall by category (turn level, top 20)")
    for category in sorted(by_category):
        found, total = by_category[category]
        print(
            f"  {CATEGORY_NAMES.get(category, category):<14} {found / total:>6.1%}"
            f"   ({found}/{total} turns)"
        )

    print(f"\n  accuracy given the evidence was retrieved")
    for name, (correct, seen) in conditional.items():
        if seen:
            print(f"  {name:<12} {correct}/{seen} = {correct / seen:.1%}")

    covered_n = conditional["covered"][1]
    total_n = covered_n + conditional["uncovered"][1]
    if total_n:
        print(
            f"\n  of {total_n} questions, evidence fully retrieved for "
            f"{covered_n} ({covered_n / total_n:.1%})"
        )

    await memory.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
