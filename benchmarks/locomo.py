#!/usr/bin/env python
"""Run Momex against the LOCOMO long-term conversational memory benchmark.

    uv run python benchmarks/locomo.py --conversations 1
    uv run python benchmarks/locomo.py --conversations 1 --questions 50 --judge
    uv run python benchmarks/locomo.py --reuse --questions 100

LOCOMO (https://github.com/snap-research/locomo) is what mem0 and Zep publish
against: ten multi-session conversations, 5882 turns, 1986 questions in five
categories. Download the data first:

    curl -sL -o benchmarks/data/locomo10.json \\
      https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json

Momex has no answer generation -- search() returns SearchItem objects, and the
README's position is that you feed those to your own agent. So this harness
supplies the missing step: retrieve with Momex, then ask a model to answer from
what came back. That means a score here is a joint measurement of Momex's
retrieval and a thin reader on top, and a low score has to be attributed
between them. `--dump` prints what was retrieved, which is how you tell the two
apart.

Ingestion note: contradiction detection is off by default here. LOCOMO asks
about things that changed over time ("what job did she have before?"), and
supersession deliberately hides exactly that. Use --contradictions to measure
the cost of leaving it on -- it is a real effect worth knowing, not a bug.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
import string
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from momex import Memory, MomexConfig, SearchItem, StorageConfig  # noqa: E402

DATA = ROOT / "benchmarks" / "data" / "locomo10.json"

# LOCOMO's category codes. 5 is the adversarial set, whose questions are
# unanswerable from the conversation; the published comparisons generally
# exclude it, so it is opt-in here.
CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}
DEFAULT_CATEGORIES = (1, 2, 3, 4)

TERSE_PROMPT = """Answer the question using only the memories below.

Memories:
{context}

Question: {question}

Rules:
- Answer as briefly as the question allows.
- Yes/no questions get "Yes" or "No", with a short reason if one is asked for.
- "When" questions get an absolute date. Memories are timestamped, so resolve
  relative words against the timestamp of the memory they appear in: "yesterday"
  in a memory dated 3 July 2023 means 2 July 2023.
- List every item the question asks for, not just the first one you find.
- Do not explain your reasoning or restate the question.
- If the memories do not contain the answer, reply exactly: NO_ANSWER

Answer:"""

# Reasoning first, no word limit, and no way to decline.
#
# Both differences are worth points rather than accuracy, and it is worth being
# explicit about which is which. Reasoning aloud genuinely helps on questions
# that need two facts combined or a date computed. The length and the missing
# refusal do not: LOCOMO's judge is instructed to be generous about wording, so
# a longer answer simply gives it more surface to match against, and a refusal
# is scored identically to a wrong guess -- which makes guessing free. An
# independent audit measured a 10.7 point swing on identical context from this
# prompt difference alone, which is larger than the spread between the systems
# the benchmark is used to rank.
COT_PROMPT = """You are answering a question about a long conversation.

Memories:
{context}

Question: {question}

Work through these steps, then give the answer.

1. RELEVANT MEMORIES: quote the memories that bear on the question.
2. WHAT THEY SAY: state the facts they establish, with who and when.
3. TIME: if the question is about when something happened, or about order or
   duration, resolve every relative expression against the timestamp of the
   memory it appears in. "Yesterday" in a memory dated 3 July 2023 is
   2 July 2023. Compute the actual date or interval.
4. COMPLETENESS: if the question asks what things, which ones, or how many,
   collect every one that appears anywhere in the memories, not just the first.
5. GAPS: note anything the question asks for that the memories do not say.
6. BEST ANSWER: decide, using inference from the memories where they do not
   state it outright. Never decline. If the memories are incomplete, give the
   most likely answer they support and say what it rests on.
7. FINAL ANSWER: state it directly, on one line, beginning with
   "FINAL ANSWER:".

Begin."""

# Short answers, like TERSE_PROMPT, but without two habits that cost real
# points on questions the memories do support.
#
# Brevity was being read as vagueness: asked which console someone owns, the
# reader answered "Nintendo" where the answer was "Nintendo Switch". And the
# refusal clause was actively harmful -- a refusal scores the same as a wrong
# guess, so declining can only lose. Open-domain questions are the ones that
# suffer, because their answers are meant to be inferred ("what underlying
# condition might she have", answer: asthma) rather than quoted, and a prompt
# that says "using only the memories" reads as a ban on inferring at all.
#
# Unlike cot-verbose this does not lengthen the answer, so token F1 stays
# meaningful and can be used to check that answers really did improve rather
# than merely giving the judge more to match against.
DIRECT_PROMPT = """Answer the question from the memories below.

Memories:
{context}

Question: {question}

Rules:
- Answer in one short phrase or sentence. No reasoning, no restating.
- Be specific. Give the full name of a thing, not a shortened form:
  "Nintendo Switch", not "Nintendo".
- Yes/no questions get "Yes" or "No" first, then the reason if one is asked for.
- "When" questions get an absolute date. Memories are timestamped, so resolve
  relative words against the timestamp of the memory they appear in:
  "yesterday" in a memory dated 3 July 2023 means 2 July 2023.
- List every item the question asks for, not just the first one you find.
- The answer may not be stated outright. Infer it from what the memories imply,
  and say what the evidence best supports.
- Never refuse and never say you do not know. Give your best answer.

Answer:"""

ANSWER_PROMPTS = {
    "terse": TERSE_PROMPT,
    "direct": DIRECT_PROMPT,
    "cot": COT_PROMPT,
    "cot-verbose": COT_PROMPT,
}
# Existing diagnostic scripts import the historical default by name. Keep the
# alias so adding selectable prompts does not make every tool fail at import.
ANSWER_PROMPT = TERSE_PROMPT

JUDGE_PROMPT = """You are grading a question-answering system against a gold answer.

Question: {question}
Gold answer: {gold}
System answer: {predicted}

Does the system answer convey the same information as the gold answer?
Minor differences in wording, format, or extra detail are acceptable.
A different fact, a wrong date, or a refusal is not.

Reply with exactly one word: CORRECT or WRONG."""


# ---------------------------------------------------------------- scoring


def normalize(text: str) -> str:
    """SQuAD-style normalisation: lowercase, strip articles and punctuation."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(predicted: str, gold: str) -> float:
    """Token overlap F1, the metric LOCOMO reports for QA."""
    pred_tokens = normalize(predicted).split()
    gold_tokens = normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------- dataset


@dataclass
class Question:
    question: str
    answer: str
    category: int
    evidence: list[str] = field(default_factory=list)


@dataclass
class Conversation:
    sample_id: str
    sessions: list[tuple[str, str, list[dict]]]  # (key, date, turns)
    questions: list[Question]

    @property
    def turn_count(self) -> int:
        return sum(len(turns) for _, _, turns in self.sessions)


def load_conversations(path: Path) -> list[Conversation]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    conversations = []
    for sample in raw:
        conv = sample["conversation"]
        keys = sorted(
            (
                k
                for k in conv
                if k.startswith("session_") and not k.endswith("date_time")
            ),
            key=lambda k: int(k.split("_")[1]),
        )
        sessions = [(k, conv.get(f"{k}_date_time", ""), conv[k]) for k in keys]

        questions = []
        for qa in sample["qa"]:
            answer = qa.get("answer", qa.get("adversarial_answer", ""))
            evidence = qa.get("evidence") or []
            if isinstance(evidence, str):
                try:
                    evidence = json.loads(evidence.replace("'", '"'))
                except json.JSONDecodeError:
                    evidence = [evidence]
            questions.append(
                Question(
                    question=qa["question"],
                    answer=str(answer),
                    category=int(qa.get("category", 0)),
                    evidence=list(evidence),
                )
            )
        conversations.append(Conversation(sample["sample_id"], sessions, questions))
    return conversations


def select_questions(
    questions: list[Question],
    categories: set[int],
    cap: int,
    seed: int,
) -> list[Question]:
    """Take a stratified sample, so every category is represented.

    Taking the first N in file order is not a sample: LOCOMO groups its
    questions, so a prefix can miss whole categories. The first run of this
    harness drew 60 questions and got no single-hop ones at all -- the largest
    and easiest category -- which made the result look worse than it was.
    """
    eligible = [q for q in questions if q.category in categories]
    if not cap or cap >= len(eligible):
        return eligible

    by_category: dict[int, list[Question]] = defaultdict(list)
    for question in eligible:
        by_category[question.category].append(question)

    rng = random.Random(seed)
    for group in by_category.values():
        rng.shuffle(group)

    # Proportional allocation, then round-robin for whatever rounding left over.
    picked: list[Question] = []
    order = sorted(by_category)
    for category in order:
        group = by_category[category]
        share = round(cap * len(group) / len(eligible))
        picked.extend(group[:share])

    index = 0
    while len(picked) < cap:
        group = by_category[order[index % len(order)]]
        remaining = [q for q in group if q not in picked]
        if remaining:
            picked.append(remaining[0])
        index += 1
        if index > len(order) * cap:  # pragma: no cover - safety valve
            break

    return picked[:cap]


# ---------------------------------------------------------------- ingest


def session_timestamp(label: str) -> str | None:
    """Turn LOCOMO's session date into something add() can store.

    They read like "1:56 pm on 8 May, 2023". Without this the whole
    conversation lands on the moment it was imported, and every temporal
    question is asked of the wrong timeline.
    """
    if not label:
        return None
    cleaned = label.replace(",", "").strip()
    for fmt in ("%I:%M %p on %d %B %Y", "%d %B %Y", "%I:%M %p on %d %b %Y"):
        try:
            return (
                datetime.strptime(cleaned, fmt)
                .replace(tzinfo=timezone.utc)
                .strftime("%Y-%m-%dT%H:%M:%SZ")
            )
        except ValueError:
            continue
    return None


async def ingest(
    memory: Memory,
    conversation: Conversation,
    *,
    contradictions: bool,
    verbose: bool,
) -> float:
    """Load a conversation into Momex, one add() per session.

    Batching by session is what a caller would naturally do, and it keeps the
    number of add() round trips proportional to sessions rather than turns.
    Knowledge extraction still runs per message inside TypeAgent.

    Each session is stored with the date it happened, so the collection covers
    the months the conversation does rather than the minutes the import took.
    """
    started = time.monotonic()
    unparsed = 0
    for index, (_key, date, turns) in enumerate(conversation.sessions, 1):
        messages = [
            {
                "role": turn["speaker"],
                "content": f'[{date}] {turn["speaker"]}: {turn["text"]}',
            }
            for turn in turns
            if turn.get("text")
        ]
        if not messages:
            continue
        when = session_timestamp(date)
        unparsed += when is None
        await memory.add(
            messages,
            infer=True,
            detect_contradictions=contradictions,
            timestamp=when,
        )
        if verbose:
            print(
                f"    session {index}/{len(conversation.sessions)} "
                f"({len(messages)} turns) at {when}",
                flush=True,
            )
    if unparsed:
        print(f"    warning: {unparsed} session dates were not parsed", flush=True)
    return time.monotonic() - started


# ---------------------------------------------------------------- answering


def render_context(items: list[SearchItem], limit: int) -> str:
    lines = []
    for item in items[:limit]:
        stamp = f" ({item.timestamp})" if item.timestamp else ""
        lines.append(f"- [{item.type}]{stamp} {item.text}")
    return "\n".join(lines) if lines else "(no memories retrieved)"


def render_full_context(conversation: Conversation) -> str:
    """The whole conversation, chronologically, grouped by session.

    This is the baseline that says what the reader could do if retrieval were
    perfect and free. Published LOCOMO numbers are usually quoted against
    retrieval systems without one, which makes it impossible to tell how much
    of a score belongs to the memory and how much to the reader. Comparing
    against this, with the same model, prompt, judge and questions, is the only
    comparison here that isolates retrieval.
    """
    blocks = []
    for _, date, turns in conversation.sessions:
        lines = [f"=== Session ({date}) ==="]
        for turn in turns:
            text = turn.get("text")
            if not text:
                continue  # ingest() skips these too, so the baseline must
            lines.append(f"{turn.get('speaker', '?')}: {text}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def render_hybrid_context(
    items: list[SearchItem], limit: int, transcript: list[SearchItem]
) -> str:
    """Put retrieved evidence first, followed by the complete transcript.

    Full context cannot miss a turn, while retrieval makes the likely evidence
    prominent enough that it is not lost in 15,000 words of unrelated chat.
    With cost unconstrained, these are complementary rather than competing
    strategies.
    """
    return (
        "=== RETRIEVED EVIDENCE (inspect this first) ===\n"
        f"{render_context(items, limit)}\n\n"
        "=== COMPLETE CONVERSATION (use it to fill gaps and verify) ===\n"
        f"{render_context(transcript, len(transcript))}"
    )


async def answer_from_context(
    llm, question: Question, context: str, style: str = "terse"
) -> str:
    """Answer with a pre-rendered context, skipping retrieval entirely."""
    return await _complete_answer(llm, question, context, style)


def extract_final_answer(text: str) -> str:
    """Take what follows FINAL ANSWER:, or the last non-empty line.

    The chain-of-thought prompt asks for a marker, and the reader usually
    obliges. When it does not, the conclusion is still the last thing it wrote,
    and handing the judge several hundred words of reasoning would score the
    reasoning rather than the answer.
    """
    marker = "FINAL ANSWER:"
    upper = text.upper()
    if marker in upper:
        tail = text[upper.rindex(marker) + len(marker) :]
        for line in tail.splitlines():
            if line.strip():
                return line.strip().lstrip("*# ").strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text.strip()


async def _complete_answer(llm, question: Question, context: str, style: str) -> str:
    prompt = ANSWER_PROMPTS[style].format(context=context, question=question.question)
    # Chain-of-thought needs room for the reasoning it is being asked to show;
    # 80 tokens truncates it mid-thought and loses the answer entirely.
    max_tokens = 1200 if style.startswith("cot") else 80
    try:
        response = await llm.complete(prompt, max_tokens=max_tokens)
    except Exception as exc:  # pragma: no cover
        return f"LLM_ERROR: {exc}"
    content = response.content.strip()
    # cot-verbose hands the judge the reasoning as well as the conclusion. It
    # exists to measure one thing: how much of the reported benefit of
    # chain-of-thought prompting on this benchmark is the reasoning, and how
    # much is simply giving a judge told to "be generous" several hundred more
    # words to find a match in. It is not a way to answer questions better.
    if style == "cot":
        return extract_final_answer(content)
    return content


async def answer_question(
    memory: Memory,
    llm,
    question: Question,
    *,
    top_k: int,
    min_score: float,
    style: str = "terse",
    neighbors: int = 0,
    full_context: list[SearchItem] | None = None,
) -> tuple[str, list[SearchItem]]:
    try:
        items = await memory.search(question.question, limit=top_k, neighbors=neighbors)
    except Exception as exc:  # pragma: no cover - reported, not raised
        return f"RETRIEVAL_ERROR: {exc}", []

    if not items and min_score is not None:
        # Fall back to a laxer embedding threshold. search()'s default of 0.3
        # is tuned for statement-shaped queries; LOCOMO asks questions, which
        # score lower against the statements that answer them.
        try:
            items = await memory.search_by_embedding(
                question.question, limit=top_k, min_score=min_score
            )
        except Exception:
            items = []

    if not items:
        return "NO_ANSWER", []

    context = (
        render_hybrid_context(items, top_k, full_context)
        if full_context is not None
        else render_context(items, top_k)
    )
    answer = await _complete_answer(llm, question, context, style)
    return answer, items


async def judge_answer(llm, question: Question, predicted: str) -> bool:
    prompt = JUDGE_PROMPT.format(
        question=question.question, gold=question.answer, predicted=predicted
    )
    response = await llm.complete(prompt, max_tokens=5)
    return response.content.strip().upper().startswith("CORRECT")


# ---------------------------------------------------------------- reporting


@dataclass
class Result:
    question: Question
    predicted: str
    f1: float
    judged: bool | None
    retrieved: int
    context: list[SearchItem] = field(default_factory=list)

    @property
    def correct(self) -> bool:
        return self.judged if self.judged is not None else self.f1 > 0.5


def report(results: list[Result], ingest_seconds: float, query_seconds: float) -> None:
    by_category: dict[int, list[Result]] = defaultdict(list)
    for result in results:
        by_category[result.question.category].append(result)

    print("\n  by category")
    header = f"  {'category':<14} {'n':>4} {'F1':>7} {'judge':>7} {'no-ctx':>7}"
    print(header)
    for category in sorted(by_category):
        rows = by_category[category]
        f1 = sum(r.f1 for r in rows) / len(rows)
        judged = [r for r in rows if r.judged is not None]
        acc = (
            f"{sum(bool(r.judged) for r in judged) / len(judged):6.1%}"
            if judged
            else "     -"
        )
        empty = sum(1 for r in rows if r.retrieved == 0) / len(rows)
        name = CATEGORY_NAMES.get(category, str(category))
        print(f"  {name:<14} {len(rows):>4} {f1:>7.3f} {acc:>7} {empty:>6.1%}")

    overall_f1 = sum(r.f1 for r in results) / len(results)
    judged = [r for r in results if r.judged is not None]
    empty = sum(1 for r in results if r.retrieved == 0)

    print(f"\n  questions:        {len(results)}")
    print(f"  mean token F1:    {overall_f1:.3f}")
    if judged:
        acc = sum(bool(r.judged) for r in judged) / len(judged)
        print(f"  judge accuracy:   {acc:.1%}")
    print(f"  no memories:      {empty} ({empty / len(results):.1%})")

    # What the retrieved context is actually made of. Momex's thesis is
    # structured RAG -- extracted entities, actions and topics -- so it is
    # worth knowing how much of that reaches the reader, versus plain message
    # text, which a much cheaper index would also have returned.
    composition: Counter[str] = Counter()
    for result in results:
        composition.update(item.type for item in result.context)
    total_items = sum(composition.values())
    if total_items:
        share = ", ".join(
            f"{name} {count / total_items:.0%}"
            for name, count in composition.most_common()
        )
        print(
            f"  context:          {total_items / len(results):.1f} items/question "
            f"({share})"
        )

    print(f"  ingest time:      {ingest_seconds:.0f}s")
    print(
        f"  query time:       {query_seconds:.0f}s "
        f"({query_seconds / len(results):.1f}s per question)"
    )


def dump(results: list[Result], count: int) -> None:
    """Print failures first: they are what needs attributing.

    A wrong answer is either a retrieval failure (Momex never surfaced the
    evidence) or a reader failure (it did, and the model still got it wrong).
    The retrieved context is printed so the two can be told apart -- which
    matters, because this harness supplies the reader itself and a bad reader
    would otherwise be scored as a bad memory system.
    """
    misses = [r for r in results if not r.correct]
    shown = misses[:count] or results[:count]

    print(f"\n  failures ({len(misses)}/{len(results)}), showing {len(shown)}")
    for result in shown:
        print(
            f"\n  ({CATEGORY_NAMES.get(result.question.category)}) "
            f"{result.question.question}"
        )
        print(f"        gold: {result.question.answer}")
        print(f"        got:  {result.predicted}")
        print(f"        retrieved {result.retrieved} items:")
        for item in result.context[:6]:
            print(f"          [{item.type}] {item.text[:105]}")


def save_results(path: Path, args: argparse.Namespace, results: list[Result]) -> None:
    """Persist question-level outcomes so strategies can be compared directly."""
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    rows = []
    for result in results:
        rows.append(
            {
                "question": result.question.question,
                "gold": result.question.answer,
                "category": result.question.category,
                "evidence": result.question.evidence,
                "predicted": result.predicted,
                "f1": result.f1,
                "judged": result.judged,
                "retrieved": result.retrieved,
                "context": [
                    {
                        "type": item.type,
                        "text": item.text,
                        "score": item.score,
                        "fusion_score": item.fusion_score,
                        "timestamp": item.timestamp,
                        "ordinal": item.ordinal,
                    }
                    for item in result.context
                ],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"config": config, "results": rows}, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------- main


async def run(args: argparse.Namespace) -> int:
    if not DATA.exists():
        print(
            f"Missing {DATA}. See the module docstring for the download.",
            file=sys.stderr,
        )
        return 2

    config = MomexConfig.from_env()
    try:
        config.validate()
    except Exception as exc:
        print(f"Config: {exc}", file=sys.stderr)
        return 2
    config.storage = StorageConfig(path=str(ROOT / "benchmarks" / "store"))
    retrieval_model = config.llm.model
    reader_model = args.reader_model or retrieval_model
    judge_model = args.judge_model or config.llm.model

    categories: set[int] = set(args.categories or DEFAULT_CATEGORIES)
    conversations = load_conversations(DATA)[: args.conversations]

    print(f"retriever: {config.llm.provider}/{retrieval_model}")
    print(f"reader:    {config.llm.provider}/{reader_model}")
    print(f"judge:     {config.llm.provider}/{judge_model}")
    print(
        f"answer prompt: {args.answer_prompt}  top-k: {args.top_k}  "
        f"neighbors: {args.neighbors}"
    )
    print(
        f"conversations: {len(conversations)}  "
        f"turns: {sum(c.turn_count for c in conversations)}"
    )
    print(
        f"categories: {sorted(categories)}  "
        f"contradiction detection: {args.contradictions}"
    )

    if reader_model == retrieval_model:
        llm = config.create_llm()
    else:
        reader_config = replace(config.llm, model=reader_model)
        llm = replace(config, llm=reader_config).create_llm()

    # Retrieval, reading and grading are three separate experimental
    # variables. Sharing one mutable config made changing the "reader" also
    # change the structured-query compiler, while sharing one client made a
    # stricter grader look exactly like a worse system.
    if judge_model == reader_model:
        judge_llm = llm
    elif judge_model == retrieval_model:
        judge_llm = config.create_llm()
    else:
        judge_config = replace(config.llm, model=judge_model)
        judge_llm = replace(config, llm=judge_config).create_llm()
    results: list[Result] = []
    ingest_seconds = query_seconds = 0.0

    for conversation in conversations:
        collection = f"locomo:{conversation.sample_id}"
        memory = Memory(collection=collection, config=config)

        questions = select_questions(
            conversation.questions, categories, args.questions, args.seed
        )

        print(
            f"\n{conversation.sample_id}: {conversation.turn_count} turns, "
            f"{len(questions)} questions"
        )

        if args.full_context:
            full_context = render_full_context(conversation)
            print(f"  full context: {len(full_context.split())} words, no retrieval")
            started = time.monotonic()
            for index, question in enumerate(questions, 1):
                predicted = await answer_from_context(
                    llm, question, full_context, args.answer_prompt
                )
                judged = None
                if args.judge:
                    try:
                        judged = await judge_answer(judge_llm, question, predicted)
                    except Exception:
                        judged = None
                results.append(
                    Result(
                        question,
                        predicted,
                        token_f1(predicted, question.answer),
                        judged,
                        conversation.turn_count,
                        [],
                    )
                )
                if index % 10 == 0:
                    print(f"    {index}/{len(questions)}", flush=True)
            query_seconds += time.monotonic() - started
            continue

        stats = await memory.stats()
        if stats["total_messages"] and args.reuse:
            print(f"  reusing {stats['total_messages']} ingested messages")
        else:
            if stats["total_messages"]:
                await memory.clear()
            print("  ingesting...", flush=True)
            ingest_seconds += await ingest(
                memory,
                conversation,
                contradictions=args.contradictions,
                verbose=args.verbose,
            )
            stats = await memory.stats()
        print(
            f"  {stats['total_messages']} messages, "
            f"{stats['total_semantic_refs']} semantic refs"
        )
        transcript = await memory.transcript() if args.hybrid_context else None

        started = time.monotonic()
        for index, question in enumerate(questions, 1):
            predicted, items = await answer_question(
                memory,
                llm,
                question,
                top_k=args.top_k,
                min_score=args.fallback_score,
                style=args.answer_prompt,
                neighbors=args.neighbors,
                full_context=transcript,
            )
            judged = None
            if args.judge:
                try:
                    judged = await judge_answer(judge_llm, question, predicted)
                except Exception:
                    judged = None
            results.append(
                Result(
                    question,
                    predicted,
                    token_f1(predicted, question.answer),
                    judged,
                    len(items),
                    items,
                )
            )
            if index % 10 == 0:
                print(f"    {index}/{len(questions)}", flush=True)
        query_seconds += time.monotonic() - started

        await memory.close()

    if not results:
        print("No questions ran.", file=sys.stderr)
        return 2

    report(results, ingest_seconds, query_seconds)
    if args.dump:
        dump(results, args.dump)
    if args.output:
        save_results(args.output, args, results)
        print(f"  results saved:    {args.output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversations", type=int, default=1)
    parser.add_argument(
        "--questions",
        type=int,
        default=0,
        help="Cap questions per conversation (0 = all). "
        "Sampled proportionally across categories.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Sampling seed, so a run can be repeated.",
    )
    parser.add_argument(
        "--categories",
        type=int,
        nargs="*",
        help=f"LOCOMO categories. Default {list(DEFAULT_CATEGORIES)}.",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--answer-prompt",
        choices=sorted(ANSWER_PROMPTS),
        default="terse",
        help="terse asks for the shortest answer and allows a refusal. "
        "cot reasons first, has no length limit, and cannot decline. "
        "The choice is worth about as much as the retrieval system.",
    )
    parser.add_argument(
        "--reader-model",
        help="Override the model that answers. The judge is unaffected, so "
        "runs stay comparable.",
    )
    parser.add_argument(
        "--judge-model",
        help="Override the model that grades. Changing this changes every "
        "number in the report and none of the system's behaviour, so it "
        "should be held fixed across any comparison.",
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        default=0,
        help="Turns of conversation to include either side of each retrieved "
        "message. Ranking scores turns independently; answers often sit in "
        "the reply to the turn that matched.",
    )
    parser.add_argument(
        "--full-context",
        action="store_true",
        help="Skip retrieval and hand the reader the entire conversation. "
        "This is the only comparison here that isolates retrieval: same "
        "model, prompt, judge and questions, unlimited context.",
    )
    parser.add_argument(
        "--hybrid-context",
        action="store_true",
        help="Put retrieved evidence first, then append the complete "
        "conversation. This spends context to combine retrieval's focus with "
        "full context's perfect coverage.",
    )
    parser.add_argument(
        "--fallback-score",
        type=float,
        default=0.1,
        help="Embedding threshold used when search() returns nothing.",
    )
    parser.add_argument(
        "--contradictions",
        action="store_true",
        help="Leave contradiction detection on during ingest.",
    )
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Skip ingestion if the collection already has messages.",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Also grade with an LLM, as the published numbers do.",
    )
    parser.add_argument(
        "--dump",
        type=int,
        default=0,
        help="Print this many answers with their retrieved context.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write question-level answers, judgments and retrieved context "
        "to a JSON file for paired comparisons.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    if args.neighbors < 0:
        parser.error("--neighbors cannot be negative")
    if args.full_context and args.hybrid_context:
        parser.error("--full-context and --hybrid-context are mutually exclusive")
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
