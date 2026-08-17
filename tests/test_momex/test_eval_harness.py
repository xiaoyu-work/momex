"""Tests for the contradiction eval tool's own scoring.

An eval harness that miscounts is worse than none: it produces confident
numbers that are wrong. This checks the scoring offline, with a stub model, so
the tool's arithmetic is trustworthy before any of its output is believed.

The judgment quality it measures is not tested here -- that needs a real model,
which is what the tool is for.
"""

from __future__ import annotations

import pytest

from momex.config import LLMConfig, MomexConfig
from tools.eval_contradictions import build_candidates, Outcome, Report, run_case

from .contradiction_cases import action, Case, Memory, topic


class _Response:
    def __init__(self, content):
        self.content = content


class _StubLLM:
    """Answers with fixed indices, as a real judge would."""

    def __init__(self, reply: str):
        self.reply = reply
        self.prompts: list[str] = []

    async def complete(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return _Response(self.reply)


def _config(llm) -> MomexConfig:
    config = MomexConfig(llm=LLMConfig(provider="openai", model="m", api_key="k"))
    config.create_llm = lambda: llm  # type: ignore[method-assign]
    return config


def _case(**kw) -> Case:
    defaults = dict(
        id="c",
        kind="polarity",
        new_text="I don't like sushi",
        new_knowledge=[action("user", ["dislike"], "sushi")],
        existing=[
            Memory("old", action("user", ["like"], "sushi")),
            Memory("keep", action("user", ["like"], "ramen")),
        ],
        expect={"old"},
    )
    defaults.update(kw)
    return Case(**defaults)  # type: ignore[arg-type]


class TestBuildCandidates:
    def test_only_propositions_reach_the_judge(self):
        """Same rule as production: topics and bare entities assert nothing."""
        from .contradiction_cases import entity

        case = _case(
            existing=[
                Memory("topic", topic("food")),
                Memory("bare", entity("sushi", ["food"])),
                Memory("faceted", entity("Xiaoyu", ["person"], employer="Microsoft")),
                Memory("act", action("user", ["like"], "sushi")),
            ]
        )

        _, names = build_candidates(case)
        assert sorted(names.values()) == ["act", "faceted"]

    def test_ordinals_map_back_to_case_ids(self):
        case = _case()
        candidates, names = build_candidates(case)

        assert [names[c.raw.semantic_ref_ordinal] for c in candidates] == [
            "old",
            "keep",
        ]


class TestRunCase:
    @pytest.mark.asyncio
    async def test_correct_verdict_scores_exact(self):
        outcome = await run_case(_case(), _config(_StubLLM("0")), verbose=False)

        assert outcome.actual == {"old"}
        assert outcome.correct
        assert outcome.false_positives == set()
        assert outcome.false_negatives == set()

    @pytest.mark.asyncio
    async def test_retiring_too_much_is_a_false_positive(self):
        outcome = await run_case(_case(), _config(_StubLLM("0, 1")), verbose=False)

        assert not outcome.correct
        assert outcome.false_positives == {"keep"}
        assert outcome.false_negatives == set()

    @pytest.mark.asyncio
    async def test_retiring_nothing_is_a_false_negative(self):
        outcome = await run_case(_case(), _config(_StubLLM("none")), verbose=False)

        assert not outcome.correct
        assert outcome.false_positives == set()
        assert outcome.false_negatives == {"old"}

    @pytest.mark.asyncio
    async def test_keep_case_scored_correctly_when_model_declines(self):
        case = _case(kind="multivalue", expect=set())
        outcome = await run_case(case, _config(_StubLLM("none")), verbose=False)

        assert outcome.correct
        assert outcome.actual == set()

    @pytest.mark.asyncio
    async def test_keep_case_scored_as_loss_when_model_retires(self):
        case = _case(kind="multivalue", expect=set())
        outcome = await run_case(case, _config(_StubLLM("0")), verbose=False)

        assert not outcome.correct
        assert outcome.false_positives == {"old"}

    @pytest.mark.asyncio
    async def test_a_case_with_no_candidates_costs_no_model_call(self):
        llm = _StubLLM("0")
        case = _case(
            kind="unrelated", existing=[Memory("t", topic("food"))], expect=set()
        )

        outcome = await run_case(case, _config(llm), verbose=False)

        assert outcome.actual == set()
        assert llm.prompts == []


class TestReport:
    def _report(self, *outcomes) -> Report:
        report = Report()
        for outcome in outcomes:
            report.add(outcome)
        return report

    def test_counts_separate_the_two_error_kinds(self):
        report = self._report(
            Outcome("a", "polarity", {"x"}, {"x"}),
            Outcome("b", "multivalue", set(), {"y"}),
            Outcome("c", "replacement", {"z"}, set()),
        )

        retired, wrong, missed = report.counts()
        assert (retired, wrong, missed) == (2, 1, 1)
        assert report.exact == 1

    def test_all_correct_reports_no_errors(self):
        report = self._report(
            Outcome("a", "polarity", {"x"}, {"x"}),
            Outcome("b", "multivalue", set(), set()),
        )

        assert report.exact == 2
        assert report.counts() == (1, 0, 0)
