#!/usr/bin/env python
"""Measure how well a real model adjudicates contradictions.

    uv run python tools/eval_contradictions.py
    uv run python tools/eval_contradictions.py --repeat 5
    uv run python tools/eval_contradictions.py --case polarity-like-sushi -v

The offline harness (tests/test_momex/test_contradiction_recall.py) measures
what the candidate lookup surfaces, which is deterministic. This measures the
step after it: given those candidates, what does the model decide.

That step is not deterministic, which is the reason this is a tool rather than
a test. Use --repeat to see how stable it is; a judgment that changes between
runs on the same input is a finding in itself, because it means the same
sequence of add() calls can leave different collections behind.

The two error kinds are not equally bad and are reported separately:

  A false positive retires a memory that was true and current. Nobody reads
  the ledger, so this is silent data loss.

  A false negative leaves a stale memory. Search may return something out of
  date, and the next add() on the subject can still correct it.

Reads credentials the same way the library does (.env or MOMEX_*/OPENAI_*).
Costs one model call per case per repeat.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field
import os
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from momex.config import LLMConfig, MomexConfig  # noqa: E402
from momex.contradictions import detect, is_propositional  # noqa: E402
from momex.results import SearchItem  # noqa: E402
from momex.search import render_knowledge  # noqa: E402
from tests.test_momex.contradiction_cases import Case, CASES  # noqa: E402


class _Ref:
    def __init__(self, ordinal: int, knowledge: Any):
        self.semantic_ref_ordinal = ordinal
        self.knowledge = knowledge


@dataclass
class Outcome:
    case_id: str
    kind: str
    expected: set[str]
    actual: set[str]

    @property
    def correct(self) -> bool:
        return self.expected == self.actual

    @property
    def false_positives(self) -> set[str]:
        """Retired something that should have been kept. Silent data loss."""
        return self.actual - self.expected

    @property
    def false_negatives(self) -> set[str]:
        """Left something stale. Recoverable by a later add()."""
        return self.expected - self.actual


@dataclass
class Report:
    outcomes: list[Outcome] = field(default_factory=list)

    def add(self, outcome: Outcome) -> None:
        self.outcomes.append(outcome)

    @property
    def exact(self) -> int:
        return sum(1 for o in self.outcomes if o.correct)

    def counts(self) -> tuple[int, int, int]:
        retired = sum(len(o.actual) for o in self.outcomes)
        wrong = sum(len(o.false_positives) for o in self.outcomes)
        missed = sum(len(o.false_negatives) for o in self.outcomes)
        return retired, wrong, missed


def build_candidates(case: Case) -> tuple[list[SearchItem], dict[int, str]]:
    """The candidates the offline lookup would surface, as the judge sees them.

    Eligibility is decided by production's own is_propositional, not a copy of
    the rule, so this cannot quietly drift from what actually happens.

    Which memories are *reachable* is taken from the case rather than by
    running the lookup: this tool measures the judgment in isolation, so a
    lookup regression shows up in the offline harness rather than being blamed
    on the model.
    """
    items: list[SearchItem] = []
    names: dict[int, str] = {}
    for ordinal, memory in enumerate(case.existing):
        item = SearchItem(
            type=str(getattr(memory.knowledge, "knowledge_type", "unknown")),
            text=render_knowledge(memory.knowledge),
            score=1.0,
            raw=_Ref(ordinal, memory.knowledge),
        )
        if not is_propositional(item):
            continue
        names[ordinal] = memory.id
        items.append(item)
    return items, names


async def run_case(case: Case, config: MomexConfig, verbose: bool) -> Outcome:
    candidates, names = build_candidates(case)

    async def find():
        return candidates

    async def append(records):
        return records

    superseded = await detect(
        case.new_text,
        collection="eval",
        find_candidates=find,
        create_llm=config.create_llm,
        append=append,
    )

    actual = {names[r.ordinal] for r in superseded if r.ordinal in names}

    if verbose:
        print(f"\n  {case.id} ({case.kind})")
        print(f"    new: {case.new_text}")
        for item in candidates:
            mark = "<-" if names[item.raw.semantic_ref_ordinal] in actual else "  "
            print(f"    {mark} {names[item.raw.semantic_ref_ordinal]}: {item.text}")

    return Outcome(case.id, case.kind, set(case.expect), actual)


def load_config() -> MomexConfig:
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.getenv("MOMEX_LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    if not api_key:
        print(
            "No API key. Set MOMEX_LLM_API_KEY or OPENAI_API_KEY "
            "(a .env file is read automatically).",
            file=sys.stderr,
        )
        raise SystemExit(2)

    return MomexConfig(
        llm=LLMConfig(
            provider=os.getenv("MOMEX_LLM_PROVIDER", "openai"),
            model=os.getenv("MOMEX_LLM_MODEL", "gpt-4o-mini"),
            api_key=api_key,
            api_base=os.getenv("MOMEX_LLM_API_BASE", ""),
        )
    )


def print_report(report: Report, repeat: int) -> None:
    by_kind: dict[str, list[Outcome]] = {}
    for outcome in report.outcomes:
        by_kind.setdefault(outcome.kind, []).append(outcome)

    print("\n  by kind")
    print(f"  {'kind':<14} {'exact':>9}")
    for kind, outcomes in sorted(by_kind.items()):
        ok = sum(1 for o in outcomes if o.correct)
        print(f"  {kind:<14} {ok:>4}/{len(outcomes):<4}")

    retired, wrong, missed = report.counts()
    total = len(report.outcomes)
    print(f"\n  exact match:     {report.exact}/{total}")
    print(f"  memories retired: {retired}")
    print(f"  wrongly retired:  {wrong}   (silent data loss)")
    print(f"  missed:           {missed}   (stale, recoverable)")

    failures = [o for o in report.outcomes if not o.correct]
    if failures:
        print("\n  disagreements")
        for o in failures:
            parts = []
            if o.false_positives:
                parts.append(f"wrongly retired {sorted(o.false_positives)}")
            if o.false_negatives:
                parts.append(f"missed {sorted(o.false_negatives)}")
            print(f"    {o.case_id}: {'; '.join(parts)}")

    if repeat > 1:
        unstable = _unstable_cases(report, repeat)
        if unstable:
            print("\n  unstable across runs (same input, different verdict)")
            for case_id, verdicts in unstable:
                print(f"    {case_id}: {verdicts}")
        else:
            print(f"\n  stable across all {repeat} runs")


def _unstable_cases(report: Report, repeat: int) -> list[tuple[str, list[str]]]:
    seen: dict[str, list[frozenset[str]]] = {}
    for outcome in report.outcomes:
        seen.setdefault(outcome.case_id, []).append(frozenset(outcome.actual))
    unstable = []
    for case_id, verdicts in seen.items():
        if len(set(verdicts)) > 1:
            unstable.append(
                (case_id, [",".join(sorted(v)) or "none" for v in verdicts])
            )
    return unstable


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", help="Run only this case id.")
    parser.add_argument("--kind", help="Run only cases of this kind.")
    parser.add_argument(
        "--repeat", type=int, default=1, help="Runs per case, to expose instability."
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    cases = CASES
    if args.case:
        cases = [c for c in cases if c.id == args.case]
    if args.kind:
        cases = [c for c in cases if c.kind == args.kind]
    if not cases:
        print("No cases matched.", file=sys.stderr)
        return 2

    config = load_config()
    print(f"model: {config.llm.provider}/{config.llm.model}")
    print(f"cases: {len(cases)}  repeat: {args.repeat}")

    report = Report()
    for run in range(args.repeat):
        if args.repeat > 1:
            print(f"\n--- run {run + 1}/{args.repeat} ---")
        for case in cases:
            outcome = await run_case(case, config, args.verbose)
            report.add(outcome)
            if not args.verbose:
                print("." if outcome.correct else "F", end="", flush=True)
        if not args.verbose:
            print()

    print_report(report, args.repeat)
    return 0 if report.exact == len(report.outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
