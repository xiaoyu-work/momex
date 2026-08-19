"""Check where the structured path's messages go.

It returns messages for 95% of questions, yet covers 8.6% of evidence turns at
k=20. Those are consistent only if the messages are being returned and then
dropped before the caller sees them.

search_structured puts knowledge and messages in one list, sorts by raw score
and truncates. The scores are not on one scale -- knowledge carries term-match
weights, messages carry their own match score -- which is exactly the mismatch
that made fusion use ranks instead of magnitudes. Inside this function the
comparison is still a raw sort.
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
    parser.add_argument("--questions", type=int, default=40)
    parser.add_argument("--conversation", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = MomexConfig.from_env()
    config.validate()
    config.storage = StorageConfig(path=str(ROOT / "benchmarks" / "store"))

    raw = json.loads(DATA.read_text(encoding="utf-8"))[args.conversation]
    texts = evidence_texts(raw)
    data = load_conversations(DATA)[args.conversation]

    memory = Memory(collection=f"locomo:{data.sample_id}", config=config)
    await memory._ensure_initialized()

    questions = [
        q
        for q in select_questions(
            data.questions, set(DEFAULT_CATEGORIES), args.questions, args.seed
        )
        if q.evidence
    ]

    # [found, wanted] for each way of reading the structured path.
    top20 = [0, 0]
    top20_messages = [0, 0]
    wide_messages = [0, 0]
    counts = {"msg_in_top20": 0, "msg_in_wide": 0, "n": 0}

    for i, question in enumerate(questions, 1):
        wanted = [texts[e] for e in question.evidence if e in texts]
        if not wanted:
            continue

        narrow = await memory._search_structured(question.question, limit=20)
        wide = await memory._search_structured(question.question, limit=200)

        narrow_msgs = [x for x in narrow if x.type == "message"]
        wide_msgs = [x for x in wide if x.type == "message"]

        for bucket, items in (
            (top20, narrow),
            (top20_messages, narrow_msgs),
            (wide_messages, wide_msgs[:20]),
        ):
            found, total = covers(items, wanted)
            bucket[0] += found
            bucket[1] += total

        counts["msg_in_top20"] += len(narrow_msgs)
        counts["msg_in_wide"] += len(wide_msgs)
        counts["n"] += 1

        if i % 10 == 0:
            print(f"  {i}/{len(questions)}", flush=True)

    n = counts["n"]
    print(f"\n  messages surviving the structured path, per question")
    print(f"    inside its top 20:        {counts['msg_in_top20'] / n:.1f}")
    print(f"    when 200 are kept:        {counts['msg_in_wide'] / n:.1f}")

    print(f"\n  evidence recall")
    print(f"    structured top 20 as-is:  {top20[0] / top20[1]:.1%}")
    print(f"    only its messages:        {top20_messages[0] / top20_messages[1]:.1%}")
    print(
        f"    its messages, 20 kept:    " f"{wide_messages[0] / wide_messages[1]:.1%}"
    )

    await memory.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
