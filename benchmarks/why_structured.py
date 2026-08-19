#!/usr/bin/env python
"""Find out why the structured path retrieves so little.

    uv run python benchmarks/why_structured.py --questions 40

The structured path finds 8.6% of LOCOMO's labelled evidence turns at k=20,
against the embedding path's 64.2%. That is either an architectural limit --
a knowledge index cannot be expected to return the verbatim turn an answer
appears in -- or a defect in how questions become queries.

The two look identical from outside and call for opposite responses, so this
captures what each question actually compiles into and what that query
matches. Three things separate the explanations:

  A query that compiles to nothing, or to terms absent from the index, is a
  compiler problem. The index holds what it holds; the question never reached
  it.

  A query that matches knowledge but returns no messages is architectural in
  the narrow sense: the path works, and what it returns is not the kind of
  thing the benchmark asks for.

  A query that matches nothing while the terms are present in the index is a
  matching problem, one layer below the compiler.

Needs a collection already ingested by benchmarks/locomo.py.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path
import sys

import typechat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from benchmarks.locomo import (  # noqa: E402
    DATA,
    DEFAULT_CATEGORIES,
    load_conversations,
    select_questions,
)
from benchmarks.recall import evidence_texts  # noqa: E402
from momex import Memory, MomexConfig, StorageConfig  # noqa: E402
from typeagent.aitools import utils  # noqa: E402
from typeagent.knowpro import (  # noqa: E402
    convknowledge,
    search_query_schema,
    searchlang,
)


def describe_query(query) -> tuple[list[str], list[str]]:
    """Pull the search terms and any scoping filters out of a compiled query."""
    terms: list[str] = []
    filters: list[str] = []
    for expr in getattr(query, "search_expressions", None) or []:
        for term in getattr(expr, "search_terms", None) or []:
            terms.append(str(term))
        for filt in getattr(expr, "filters", None) or []:
            for attr in ("entity_search_terms", "action_search_term", "search_terms"):
                value = getattr(filt, attr, None)
                if value:
                    filters.append(f"{attr}={value}")
            for attr in ("time_range", "scope_defining_terms"):
                value = getattr(filt, attr, None)
                if value:
                    filters.append(f"{attr}={value}")
    return terms, filters


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=int, default=40)
    parser.add_argument("--conversation", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show", type=int, default=6)
    args = parser.parse_args()

    config = MomexConfig.from_env()
    config.validate()
    config.storage = StorageConfig(path=str(ROOT / "benchmarks" / "store"))

    raw = json.loads(DATA.read_text(encoding="utf-8"))[args.conversation]
    texts = evidence_texts(raw)
    conversation_data = load_conversations(DATA)[args.conversation]

    memory = Memory(collection=f"locomo:{conversation_data.sample_id}", config=config)
    await memory._ensure_initialized()
    conv = memory._conversation_required()

    index = conv.semantic_ref_index
    indexed_terms = {t.lower() for t in await index.get_terms()}
    print(f"terms in the semantic-ref index: {len(indexed_terms)}")

    if conv._query_translator is None:
        conv._query_translator = utils.create_translator(
            convknowledge.create_typechat_model(), search_query_schema.SearchQuery
        )

    options = searchlang.LanguageSearchOptions(
        compile_options=searchlang.LanguageQueryCompileOptions(
            exact_scope=False, verb_scope=True, term_filter=None, apply_scope=False
        ),
        exact_match=False,
        max_message_matches=100,
    )

    questions = [
        q
        for q in select_questions(
            conversation_data.questions,
            set(DEFAULT_CATEGORIES),
            args.questions,
            args.seed,
        )
        if q.evidence
    ]

    outcomes: Counter[str] = Counter()
    term_stats = [0, 0]  # terms present in index, terms total
    examples: list[tuple] = []

    for i, question in enumerate(questions, 1):
        debug = searchlang.LanguageSearchDebugContext()
        result = await searchlang.search_conversation_with_language(
            conv, conv._query_translator, question.question, options, None, debug
        )

        terms, filters = describe_query(debug.search_query)
        present = [t for t in terms if t.lower() in indexed_terms]
        term_stats[0] += len(present)
        term_stats[1] += len(terms)

        if isinstance(result, typechat.Failure):
            outcome = "compile failed"
            n_knowledge = n_messages = 0
        else:
            n_knowledge = sum(
                len(m.semantic_ref_matches)
                for r in result.value
                for m in r.knowledge_matches.values()
            )
            n_messages = sum(len(r.message_matches) for r in result.value)
            if not n_knowledge and not n_messages:
                outcome = "matched nothing"
            elif not n_messages:
                outcome = "knowledge only, no messages"
            else:
                outcome = "returned messages"

        outcomes[outcome] += 1
        if len(examples) < args.show and outcome != "returned messages":
            examples.append((question, terms, filters, outcome, present))

        if i % 10 == 0:
            print(f"  {i}/{len(questions)}", flush=True)

    total = sum(outcomes.values())
    print(f"\n  what the structured path did, {total} questions")
    for outcome, count in outcomes.most_common():
        print(f"  {outcome:<30} {count:>4}  {count / total:>6.1%}")

    if term_stats[1]:
        print(
            f"\n  compiled search terms present in the index: "
            f"{term_stats[0]}/{term_stats[1]} = {term_stats[0] / term_stats[1]:.1%}"
        )

    print(f"\n  examples that returned no messages")
    for question, terms, filters, outcome, present in examples:
        print(f"\n  Q: {question.question}")
        print(f"     outcome:  {outcome}")
        print(f"     terms:    {terms}")
        print(f"     in index: {present}")
        if filters:
            print(f"     filters:  {filters[:4]}")
        print(
            f"     evidence: {[texts.get(e, '?')[:60] for e in question.evidence][:2]}"
        )

    await memory.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
