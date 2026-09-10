# Momex Usage

## Setup

```bash
pip install -e .
```

## Memory API

Momex is a memory layer. It stores and retrieves structured knowledge; your
own agent framework owns the chat loop, conversation history and answer
generation.

## Core Concepts

### Async API

Momex is fully async. All operations use `async/await`:

```python
import asyncio
from momex import Memory, MomexConfig, LLMConfig

async def main():
    # Configure LLM once (required)
    MomexConfig.set_default(
        llm=LLMConfig(
            provider="openai",
            model="gpt-4o",
            api_key="sk-xxx",  # or use MOMEX_LLM_API_KEY env var
        ),
    )

    memory = Memory(collection="user:xiaoyuzhang")
    await memory.add("I like Python")
    for item in await memory.search("What language?"):
        print(f"[{item.type}] {item.text}")

asyncio.run(main())
```

### Collection

A **collection** is a named storage space for memories. Each collection has its own database file.

- Use collections to separate memories by user, team, or purpose
- Collection names support hierarchical structure with `:` separator
- Examples: `"xiaoyuzhang"`, `"user:xiaoyuzhang"`, `"momex:engineering:xiaoyuzhang"`

### Hierarchical Collections

The `:` separator creates a hierarchy that enables prefix queries:

```
momex:engineering:xiaoyuzhang  →  ./momex_data/momex/engineering/xiaoyuzhang/memory.db
momex:engineering:gvanrossum   →  ./momex_data/momex/engineering/gvanrossum/memory.db
momex:marketing:charlie        →  ./momex_data/momex/marketing/charlie/memory.db
```

Search behavior:
- `await search("momex:engineering:xiaoyuzhang", ...)` → searches only xiaoyuzhang
- `await search("momex:engineering", ...)` → searches xiaoyuzhang + gvanrossum
- `await search("momex", ...)` → searches xiaoyuzhang + gvanrossum + charlie

## Basic Usage

### Add and Search

```python
import asyncio
from momex import Memory

async def main():
    memory = Memory(collection="user:xiaoyuzhang")

    # Add memories - TypeAgent extracts entities, actions, topics
    await memory.add("I love Python programming")
    await memory.add("Project deadline is Friday")

    # Search - returns structured results to feed to your own agent
    for item in await memory.search("What programming language does the user like?"):
        print(f"[{item.type}] {item.text}")

asyncio.run(main())
```

### Add with Conversation Format

You can also pass conversation messages:

```python
async def main():
    memory = Memory(collection="user:xiaoyuzhang")

    # Conversation format - TypeAgent extracts knowledge from the dialogue
    await memory.add([
        {"role": "user", "content": "My name is Xiaoyu, I'm a Python developer"},
        {"role": "assistant", "content": "Nice to meet you!"},
        {"role": "user", "content": "I'm working on a FastAPI project"},
    ])

    # Search the memories
    for item in await memory.search("What is the user's name?"):
        print(f"[{item.type}] {item.text}")
```

### Attribution and write policy

By default, `add()` extracts confirmed user facts. Other roles and messages
with `confirmed=False` are preserved as unconfirmed context, not current facts.
They are available through `transcript()` or `include_unconfirmed=True` on
either search method, labeled `status="unconfirmed"`. Neighbor expansion uses
the same rule. Collections written before this policy retain their old behavior.

```python
await memory.add([
    {"role": "assistant", "content": "Do you live in Seattle?", "session_id": "chat-1"},
    {"role": "user", "speaker": "Alice", "content": "Yes.", "session_id": "chat-1"},
    {"role": "assistant", "content": "Reviewed summary", "confirmed": True},
])
```

The extractor receives the target's speaker and role plus up to two preceding
turns (`context_turns=2`), including stored turns from earlier writes. Context
only resolves references and confirmations; it is not itself a target for
fact extraction. Explicit sessions do not borrow context from other sessions.
The original text is stored unchanged, not replaced with an extraction prompt.

Use `write_policy="all"` for trusted transcripts where every speaker is a
source of facts, or `confirmed=True` on an individually reviewed message.
`confirmed=False` always wins over that default. This is an attribution policy,
not a guarantee that an LLM will interpret every statement correctly.

### Direct storage without extraction

Use `infer=False` to skip LLM knowledge extraction:

```python
async def main():
    memory = Memory(collection="user:xiaoyuzhang")

    # Direct storage - no knowledge extraction
    await memory.add("Raw log: user logged in at 2024-01-01", infer=False)
```

### Smart Updates (Automatic)

When facts change, `add()` automatically detects and removes contradicting memories:

```python
async def main():
    memory = Memory(collection="user:xiaoyuzhang")

    # Initial preference
    await memory.add("I like sushi")

    # Later, preference changed - add() automatically supersedes the old memory
    result = await memory.add("I don't like sushi anymore")
    print(f"Added {result.messages_added}, superseded {result.contradictions_removed}")

    # Nothing was destroyed. What was retired, and what replaced it:
    for record in result.superseded or []:
        print(record.text, "->", record.superseded_by, record.reason)
```

The old memory and its source message are hidden from current search, not
deleted. Unrelated active knowledge from that message remains searchable. See
[Supersession and history](#supersession-and-history) for how to review and
undo it.

**What can be superseded.** Contradiction is a relation between propositions,
so only knowledge that asserts something is eligible:

| Knowledge | Eligible | Why |
|-----------|----------|-----|
| `action` — "user like sushi" | yes | Has a truth value; "dislike" denies it |
| `entity` with facets — "Xiaoyu [employer: Microsoft]" | yes | The facet is the assertion |
| `entity` bare — "sushi (type: food)" | no | Names a thing; no preference can make it false |
| `topic` — "dietary preferences" | no | A label, not a claim |

**How candidates are found.** Momex uses the knowledge it just extracted from
your new text, and looks for existing memories with the *same subject* and
either the same verb or the same object:

- "I don't like sushi" → subject `user`, object `sushi` → finds "user like
  sushi" even though the verb changed.
- "I work at Google" → subject `user`, verb `work at` → finds "user work at
  Microsoft" even though the object changed.
- "I like ramen" → subject `user`, verb `like`, object `ramen` → finds "user
  like sushi" as a candidate, and the model is asked whether it is actually
  contradicted. Preferences are multi-valued, so it should say no.

This is a deliberate bias toward keeping memories. A missed contradiction
leaves a stale memory that a later `add()` can still correct; a wrong
supersession is silent data loss.

**Measuring it.** Contradiction handling is scored against a hand-written set
of cases in `tests/test_momex/contradiction_cases.py`, covering polarity flips,
value replacements, multi-valued additions that must *not* be retired, and
unrelated memories:

```bash
# Deterministic, offline, runs in CI: what the candidate lookup surfaces.
# Recall here is a ceiling -- a memory that is never a candidate can never
# be retired, however good the model is.
uv run pytest tests/test_momex/test_contradiction_recall.py -s

# Needs an API key: what a real model decides about those candidates.
# --repeat exposes cases where the same input gets different verdicts.
make eval-contradictions
uv run python tools/eval_contradictions.py --repeat 5 -v
```

The online tool reports the two error kinds separately, because they are not
equally bad: wrongly retiring a memory is silent data loss, while missing one
leaves something stale that is still recoverable.

You can disable automatic contradiction detection:

```python
# Skip contradiction detection (one fewer LLM call per add)
await memory.add("I don't like sushi", detect_contradictions=False)
```

### Time-Bound Memories

Add memories that automatically expire. Expired memories are excluded from search results by default:

```python
async def main():
    memory = Memory(collection="user:xiaoyuzhang")

    # Memory that expires after a date
    await memory.add(
        "Netflix subscription renews May 1 at $15.99",
        valid_to="2026-05-02",
    )

    # Memory with a validity window
    await memory.add(
        "User has a trip to Tokyo next week",
        valid_from="2026-04-10",
        valid_to="2026-04-18",
    )

    # Permanent memory (default, no expiry)
    await memory.add("User prefers aisle seats on flights")

    # Search — expired memories are automatically filtered out
    results = await memory.search("subscription")

    # Include expired memories when needed (e.g., for history)
    results = await memory.search("subscription", include_expired=True)
    for item in results:
        print(f"{item.text} (expires: {item.valid_to})")
```

### Historical snapshots and backfills

Both search methods and prefix search accept `as_of`, an ISO timestamp or a
date (through the end of that UTC day). Validity and supersession are evaluated
at that time, and later events are excluded even when history flags are enabled.
`transcript(as_of=...)` provides the corresponding source cutoff.

```python
await memory.add("I live in Portland", timestamp="2024-06-01")
await memory.add("I live in Seattle", timestamp="2024-01-01")  # backfill
then = await memory.search("Where do I live?", as_of="2024-05-01")
now = await memory.search("Where do I live?", as_of="2024-07-01")
```

When the model identifies a contradiction, event time determines its direction:
the later fact replaces the earlier one, regardless of ingestion order.
The ledger retains audit time `at` separately from `effective_at` and optional
`effective_to`. Future replacements wait until their validity window opens;
temporary replacements hide the earlier fact only during their overlapping
window. Historical supersessions remain inspectable after a manual restore.

Each input message can override `timestamp`, `valid_from`, and `valid_to`, so a
single batch can preserve an imported conversation's timeline. Invalid or
reversed windows are rejected before writing. Legacy ledger entries without
event-time fields retain their original audit-time interpretation.

### Explicit deletion

For manual control over deletion:

```python
async def main():
    memory = Memory(collection="user:xiaoyuzhang")

    # Preview first — matching is semantic, so a broad query matches broadly
    would_delete = await memory.delete("likes sushi", dry_run=True)
    print(f"Would delete {would_delete} items")

    # Then delete for real
    deleted = await memory.delete("likes sushi")
    print(f"Deleted {deleted} memories")

    # Optionally require a minimum relevance score
    deleted = await memory.delete("likes sushi", min_score=1.0)
```

**delete() parameters:**
- `limit`: int (default 50) - Maximum number of candidates to consider
- `min_score`: float (default 0.0) - Drop candidates below this native index
  score. Scales differ per index: structured term weights are unbounded,
  embedding similarities are in `[0, 1]`.
- `dry_run`: bool (default False) - Report the count without deleting

**What delete() removes:** extracted knowledge is retired, not destroyed.
Its source message is retained as historical evidence, but is excluded from
both search paths and neighbor expansion by default. Other active knowledge
from the same source remains searchable. Pass `include_superseded=True` to
either search method to read historical sources, labeled with
`SearchItem.status == "superseded"`. Use `clear()` to remove the collection.

The return value is the number of knowledge items newly hidden. Deleting the
same query twice returns `0` the second time.

**delete() is reversible.** It records a supersession rather than destroying
anything, so a query that matched more broadly than you intended can be undone
with `restore()`. `clear()` is the one genuinely destructive operation.

### Supersession and history

Memories are never destroyed by `add()` or `delete()`. They are appended to a
supersession ledger, which hides them from search and records what replaced
them. Three reasons this beats deleting:

- A wrong contradiction judgment is recoverable. Contradiction detection is an
  LLM call, and "works in Seattle" vs "works in Portland" may be a job change,
  two offices, or a headquarters — the model cannot always tell.
- The change itself is information. "Liked sushi until August" often says more
  than either fact alone.
- Deleting leaves ordinal gaps in the indexes; hiding does not.

```python
async def main():
    memory = Memory(collection="user:xiaoyuzhang")

    await memory.add("I work in Seattle")
    await memory.add("I work in Portland")   # may supersede the Seattle fact

    # What has been retired, and why
    for record in await memory.history():
        print(record.ordinal, record.text, record.reason, record.at)
        print("  replaced by:", record.superseded_by)

    # It was two offices, not a move — put it back
    await memory.restore(record.ordinal)

    # Or read the collection including everything superseded
    everything = await memory.search("where I work", include_superseded=True)
```

**history(include_restored=False)** returns `list[SupersededRecord]`, oldest
first. Each record carries `ordinal`, `superseded_by`, `at`, `reason`
(`"contradiction"`, `"delete"`, or `"legacy"`), the `text` at the time, the
`query` that triggered it, and `restored_at`.

**restore(ordinals)** takes one ordinal or a list, returns how many were
restored. Restoring does not erase the ledger entry — it timestamps it, so the
history of the history survives too. Pass `include_restored=True` to see those.

**Current versus historical evidence.** Search results carry `status`:
`"current"`, `"superseded"`, `"expired"`, or `"future"`. Both search paths and
`neighbors` follow the same policy. `include_expired=True` admits out-of-window
memories; `include_superseded=True` admits retired knowledge and its sources.
Both flags are needed for a source that is both retired and out of its window.
Historical neighbors are explicitly labeled in the expanded text.
`transcript()` remains a complete historical read and labels each source.

Collections written before the ledger existed are migrated automatically on
first read; their entries have `reason="legacy"` and no `superseded_by`, since
that was never recorded. They are restorable like any other.

### Search Across Collections

`search()` returns structured results for you to feed to your own LLM or agent.
Momex does not generate answers itself.

```python
from momex import Memory, search

async def main():
    # Create memories for different users
    xiaoyuzhang = Memory(collection="momex:engineering:xiaoyuzhang")
    await xiaoyuzhang.add("I like Python and FastAPI")

    gvanrossum = Memory(collection="momex:engineering:gvanrossum")
    await gvanrossum.add("I prefer Java")

    # Search a single collection - returns SearchItem objects
    results = await xiaoyuzhang.search("programming")
    for item in results:
        print(f"[{item.type}] {item.text} (score={item.score:.2f})")
        # item.raw contains the original TypeAgent object

    # Search across collections by prefix
    results = await search("momex:engineering", "what programming languages", limit=5)
    results = await search("momex", "who works here", limit=5)

    # Use as context for your own LLM
    context = "\n".join([f"- [{coll}] {item.text}" for coll, items in results for item in items])
```

### Bounded, cited agent context

`format_context()` is a framework-neutral formatter; it does not generate
answers or call an LLM. Its token budget includes citation labels, timestamps,
collection names and speakers. It preserves retrieval order, merges overlapping
source turns, and labels non-current evidence. `truncated` reports omitted or
shortened content, and `token_count` is measured with the selected tokenizer.

```python
from momex import format_context, search

items = await memory.search("What does the user prefer?", neighbors=2)
context = format_context(items, token_budget=2048, encoding_name="cl100k_base")
# Add context.text to your agent's prompt as quoted evidence.
for citation in context.citations:
    print(citation.label, citation.sources)

# limit remains per collection; total_limit adds a global rank-fusion budget.
groups = await search("user", "preferences", limit=10, total_limit=20, neighbors=2)
context = format_context(
    [item for _, items in groups for item in items], token_budget=2048
)
```

`SearchItem.sources` retains every cited turn's collection, source ID, ordinal,
timestamp, speaker, role, text and status, including expanded neighbors and
duplicate facts merged during ranking. Matching text from different collections
or different visibility states is not collapsed.

The default tokenizer is `cl100k_base`; select the encoding used by your answer
model when enforcing its budget. This bounds the memory context, not your whole
system prompt, tool schemas or generated response.

### Embedding-Only Search (Fallback)

`search_by_embedding()` does pure vector similarity search without any LLM call. Useful as a fallback when the LLM is unavailable, or for low-latency scenarios:

```python
async def main():
    memory = Memory(collection="user:xiaoyuzhang")

    # No LLM needed — direct embedding similarity
    results = await memory.search_by_embedding("programming languages")
    for item in results:
        print(f"{item.text} (score={item.score:.2f})")

    # Also supports include_expired
    results = await memory.search_by_embedding("subscription", include_expired=True)
```

### Manage Collections

```python
from momex import MemoryManager

manager = MemoryManager()

# List collections
all_collections = manager.list_collections()  # SQLite
eng_only = manager.list_collections(prefix="momex:engineering")

# For PostgreSQL:
# all_collections = await manager.list_collections_async()
# eng_only = await manager.list_collections_async(prefix="momex:engineering")

# Other operations
manager.exists("momex:engineering:xiaoyuzhang")
manager.delete("momex:engineering:xiaoyuzhang")
manager.rename("user:old", "user:new")
```

## Exact identities and data lifecycle

Supply a `source_id` for a single message, or one per message in a batch, to
make imports retryable. Source IDs identify immutable input: repeating the same
ID and content skips extraction and storage, retaining the original timestamp.
Different content, attribution or validity tags under the same ID raise an
error and roll back the batch. Use a new source ID for revised information.

```python
result = await memory.add("I prefer tea", source_id="chat-42:turn-7")
print(result.source_ids)          # Sources actually added
print(result.memory_ids)          # Stable extracted-knowledge IDs
print(result.skipped_source_ids)  # Existing or repeated sources

source = await memory.get_source("chat-42:turn-7")
fact = await memory.get(result.memory_ids[-1])
assert fact is not None and fact.memory_id is not None
await memory.delete_by_id(fact.memory_id)   # Reversible, exact match
await memory.restore_by_id(fact.memory_id)
```

`get()` and `get_source()` return `None` for an unknown ID and include historical
or unconfirmed records when found. Exact knowledge lookup scans stored refs in
bounded batches; it does not call an LLM. `infer=False` returns source IDs but
does not report extracted-knowledge IDs. Legacy sources without IDs retain their
existing ordinal-based interfaces.

### Full backup and restore

`export()` retains its existing human-readable content-export format. Use
`backup()` for recovery: it includes source text and IDs, attribution, time
windows, extracted knowledge, vectors, every persistent index, ingestion state,
chunk failures and the supersession ledger. Restore does not re-extract or
re-embed, and requires matching embedding model/dimensions.

```python
await memory.backup("memory-backup.json")
copy = Memory("user:backup-copy", config=memory.config)
await copy.import_backup("memory-backup.json")  # Empty destination by default
await copy.close()

# Explicitly replace an existing collection only when intended.
await memory.import_backup("memory-backup.json", replace=True)
```

The versioned JSON format is portable between SQLite and PostgreSQL. Archives
are validated before replacement; database changes are transactional, and the
backup file is replaced atomically. Backups contain memory content, so store
them with the same access controls as the database.

### Permanent forgetting

```python
removed_sources = await memory.forget("chat-42:turn-7")
```

`forget(source_id)` removes that source, its derived knowledge and vectors,
orphaned term indexes, and linked audit text from the live store. It cannot be
undone with `restore()`. Independently recorded copies under other source IDs
are not removed. External backups, copies held by callers/services and database
recovery logs are outside this operation's scope.

Remaining ordinals are compacted to preserve the storage layer's dense-index
invariant. Stable source and memory IDs are unchanged; prefer them over cached
ordinals. Import and forgetting release this `Memory`'s caches and connection.
Run these maintenance operations with other users of the collection quiescent,
then reopen other `Memory` instances so they reload their indexes.

## Configuration

Configuration has three parts:
- **LLM**: Required for knowledge extraction and search query translation
- **Embedding**: Optional, auto-inferred from LLM for OpenAI/Azure
- **Storage**: SQLite (default) or PostgreSQL

### Basic Configuration

```python
from momex import Memory, MomexConfig, LLMConfig

# Configure with code
config = MomexConfig(
    llm=LLMConfig(
        provider="openai",
        model="gpt-4o",
        api_key="sk-xxx",
    ),
)

memory = Memory(collection="user:xiaoyuzhang", config=config)
```

### Global Default

```python
from momex import Memory, MomexConfig, LLMConfig

# Set global default once
MomexConfig.set_default(
    llm=LLMConfig(
        provider="openai",
        model="gpt-4o",
        api_key="sk-xxx",
    ),
)

# Then use Memory without passing config
memory = Memory(collection="user:xiaoyuzhang")
```

### From Environment Variables

```python
from momex import Memory, MomexConfig

# Load from MOMEX_* environment variables
config = MomexConfig.from_env()
memory = Memory(collection="user:xiaoyuzhang", config=config)
```

### Separate LLM and Embedding

For non-OpenAI LLMs, you need to configure embedding separately:

```python
from momex import MomexConfig, LLMConfig, EmbeddingConfig

# Anthropic LLM + OpenAI Embedding
config = MomexConfig(
    llm=LLMConfig(
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        api_key="sk-ant-xxx",
    ),
    embedding=EmbeddingConfig(
        provider="openai",
        api_key="sk-xxx",
    ),
)
```

### PostgreSQL Configuration

For production deployment with multi-instance support:

```bash
# Install PostgreSQL dependencies
pip install momex[postgres]

# PostgreSQL must have pgvector extension
# In PostgreSQL: CREATE EXTENSION IF NOT EXISTS vector;
```

```python
from momex import MomexConfig, LLMConfig, StorageConfig

config = MomexConfig(
    llm=LLMConfig(
        provider="openai",
        model="gpt-4o",
        api_key="sk-xxx",
    ),
    storage=StorageConfig(
        backend="postgres",
        postgres_url="postgresql://user:password@localhost:5432/momex",
        postgres_pool_min=2,
        postgres_pool_max=10,
    ),
)
```

#### When to use `postgres_pgbouncer=True`

Some PostgreSQL services use connection poolers (like PgBouncer) that require special handling:

| Service | Default Connection | Need `postgres_pgbouncer`? |
|---------|-------------------|---------------------------|
| **Azure Database for PostgreSQL** | Direct | ❌ No |
| **AWS RDS PostgreSQL** | Direct | ❌ No |
| **Google Cloud SQL** | Direct | ❌ No |
| **Supabase** (port 6543) | PgBouncer | ✅ **Yes** |
| **Supabase** (port 5432) | Direct | ❌ No |
| **Neon** | Pooled | ✅ **Yes** |
| **Self-hosted PostgreSQL** | Direct | ❌ No |
| **Self-hosted with PgBouncer** | PgBouncer | ✅ **Yes** |

**Why?** PgBouncer in transaction mode doesn't support prepared statements and doesn't preserve session-level settings. Setting `postgres_pgbouncer=True` disables prepared statements and ensures proper schema handling.

```python
# Azure / AWS / GCP - Direct connection, no special config needed
config = MomexConfig(
    llm=LLMConfig(...),
    storage=StorageConfig(
        backend="postgres",
        postgres_url="postgresql://user:pass@your-db.postgres.database.azure.com:5432/mydb",
    ),
)

# Supabase (pooler port 6543) - Requires pgbouncer mode
config = MomexConfig(
    llm=LLMConfig(...),
    storage=StorageConfig(
        backend="postgres",
        postgres_url="postgresql://user:pass@db.xxx.supabase.co:6543/postgres",
        postgres_pgbouncer=True,  # Required!
    ),
)

# Neon - Requires pgbouncer mode
config = MomexConfig(
    llm=LLMConfig(...),
    storage=StorageConfig(
        backend="postgres",
        postgres_url="postgresql://user:pass@ep-xxx.us-east-2.aws.neon.tech/mydb",
        postgres_pgbouncer=True,  # Required!
    ),
)
```

### Azure OpenAI Configuration

```python
from momex import MomexConfig, LLMConfig, EmbeddingConfig, StorageConfig

config = MomexConfig(
    llm=LLMConfig(
        provider="azure",
        model="gpt-4o",  # Your deployment name
        api_key="your-azure-api-key",
        api_base="https://your-resource.openai.azure.com",
    ),
    embedding=EmbeddingConfig(
        provider="azure",
        model="text-embedding-3-small",  # Your embedding deployment name
        api_key="your-azure-api-key",
        api_base="https://your-resource.openai.azure.com",
        # api_version="2024-12-01-preview",  # Optional, has default
    ),
    storage=StorageConfig(
        backend="postgres",
        postgres_url="postgresql://...",
        postgres_pgbouncer=True,  # If using Supabase
    ),
)
```

**Note:** For Azure OpenAI, you must configure `EmbeddingConfig` separately since LLM and embedding may use different deployments.

### YAML Configuration

**config.yaml:**
```yaml
llm:
  provider: openai
  model: gpt-4o
  api_key: sk-xxx  # optional here — MOMEX_LLM_API_KEY is used when omitted
  temperature: 0.0

# embedding: (optional, auto-inferred for OpenAI/Azure)
#   provider: openai
#   model: text-embedding-3-small
#   api_key: sk-xxx          # falls back to MOMEX_EMBEDDING_API_KEY
#   api_version: 2024-02-01  # for Azure

storage:
  backend: sqlite  # or postgres
  path: ./momex_data
  # postgres_url: postgresql://user:password@localhost:5432/momex
  # postgres_schema: optional schema for collection isolation
```

**Load from YAML:**
```python
config = MomexConfig.from_yaml("config.yaml")
memory = Memory(collection="user:xiaoyuzhang", config=config)
```

**Save to YAML:**
```python
# Safe to commit: API keys and the PostgreSQL URL are omitted
config.to_yaml("my_config.yaml")

# Self-contained, including credentials — keep this file out of version control
config.to_yaml("my_config.yaml", include_secrets=True)
```

Any secret missing from the file is read from the environment when loading:
`MOMEX_LLM_API_KEY`, `MOMEX_EMBEDDING_API_KEY` and
`MOMEX_STORAGE_POSTGRES_URL`. Values present in the file take precedence.

### Environment Variables

See [Environment Variables](env-vars.md) for full documentation.

**LLM (required):**

| Variable | Description |
|----------|-------------|
| `MOMEX_LLM_PROVIDER` | LLM provider: `openai`, `azure`, `anthropic`, `deepseek`, `qwen` |
| `MOMEX_LLM_MODEL` | Model name |
| `MOMEX_LLM_API_KEY` | API key |
| `MOMEX_LLM_API_BASE` | Base URL (required for Azure) |

**Embedding (optional):**

| Variable | Description |
|----------|-------------|
| `MOMEX_EMBEDDING_PROVIDER` | Embedding provider: `openai`, `azure` |
| `MOMEX_EMBEDDING_MODEL` | Model name (default: `text-embedding-3-small`) |
| `MOMEX_EMBEDDING_API_KEY` | API key (defaults to LLM key if same provider) |

**Storage:**

| Variable | Description |
|----------|-------------|
| `MOMEX_STORAGE_BACKEND` | `sqlite` or `postgres` |
| `MOMEX_STORAGE_PATH` | SQLite storage directory |
| `MOMEX_STORAGE_POSTGRES_URL` | PostgreSQL connection URL |
| `MOMEX_STORAGE_POSTGRES_SCHEMA` | Schema for collection isolation |
| `MOMEX_STORAGE_POSTGRES_PGBOUNCER` | Set to `true` for Supabase/PgBouncer |

```bash
# OpenAI (simplest)
export MOMEX_LLM_PROVIDER=openai
export MOMEX_LLM_MODEL=gpt-4o
export MOMEX_LLM_API_KEY=sk-xxx

# Anthropic + OpenAI Embedding
export MOMEX_LLM_PROVIDER=anthropic
export MOMEX_LLM_MODEL=claude-sonnet-4-20250514
export MOMEX_LLM_API_KEY=sk-ant-xxx
export MOMEX_EMBEDDING_PROVIDER=openai
export MOMEX_EMBEDDING_API_KEY=sk-xxx

# PostgreSQL
export MOMEX_STORAGE_BACKEND=postgres
export MOMEX_STORAGE_POSTGRES_URL=postgresql://user:pass@localhost:5432/momex
```

## API Reference

### Memory

All methods are async:

| Method | Description |
|--------|-------------|
| `await add(messages)` | Add memories (auto-detects contradictions) |
| `await search(query, limit=10)` | Search, returns `list[SearchItem]` |
| `await search_by_embedding(query, limit=10)` | Embedding-only search, no LLM needed |
| `await delete(query)` | Supersede memories matching query (advanced, reversible) |
| `await history()` | Audit trail of superseded memories |
| `await restore(ordinals)` | Undo a supersession |
| `await get(memory_id)` | Read exact knowledge, including historical records |
| `await get_source(source_id)` | Read an exact source message |
| `await delete_by_id(memory_id)` | Reversibly retire exact knowledge |
| `await restore_by_id(memory_id)` | Restore by stable knowledge ID |
| `await forget(source_id)` | Permanently remove a source and derived data |
| `await transcript(as_of=None)` | Read labeled source history |
| `await stats()` | Get memory statistics |
| `await export(path)` | Human-readable JSON content export |
| `await backup(path)` | Complete portable recovery archive |
| `await import_backup(path, replace=False)` | Restore without model calls |
| `await clear()` | Delete all memories in this collection |
| `await close()` | Release the SQLite connection / PostgreSQL pool |

**add() parameters:**
- `messages`: str or list[dict] - Content to add
- `infer`: bool (default True) - Use LLM to extract knowledge
- `detect_contradictions`: bool (default True) - Auto-supersede contradicting memories
- `valid_from`: str or None - ISO date, memory relevant from this date
- `valid_to`: str or None - ISO date, memory expires after this date
- `timestamp`: str or None - Event time; per-message overrides are supported
- `source_id`: str or None - Stable source ID for a single message
- `write_policy`: `"user"` (default) or `"all"` - Which roles supply facts
- `context_turns`: int (default 2) - Preceding turns used to resolve references

**search() / search_by_embedding() parameters:**
- `include_expired`: bool (default False) - Include memories past their valid_to date
- `include_superseded`: bool (default False) - Include memories that have been
  superseded by newer ones
- `include_unconfirmed`: bool (default False) - Include unconfirmed context
- `as_of`: ISO timestamp or date - Evaluate the historical snapshot

**search() only:**
- `neighbors`: int (default 0) - Visible neighboring turns to include

**Releasing resources:**

A `Memory` holds a SQLite connection or a PostgreSQL connection pool until it is
closed. Use it as an async context manager, or call `close()` explicitly:

```python
async with Memory(collection="user:xiaoyuzhang") as memory:
    await memory.add("I like Python")
# connection/pool released here

# or, equivalently
memory = Memory(collection="user:xiaoyuzhang")
try:
    await memory.add("I like Python")
finally:
    await memory.close()
```

This matters most on the PostgreSQL backend, where each un-closed `Memory`
leaves a connection pool open. `close()` is idempotent, and the next operation
transparently re-opens the collection.

### Prefix Search Functions

All functions are async:

| Function | Description |
|----------|-------------|
| `await search(prefix, query, limit=10)` | Search (returns list of tuples) |
| `await stats(prefix)` | Get combined stats for matching collections |

Prefix search also accepts the same visibility/time options and `neighbors`.
`total_limit` optionally bounds results across all matching collections.

### SearchItem

Returned by `search()` and `search_by_embedding()`:

```python
results = await memory.search("programming")

for item in results:
    print(item.type)        # "entity", "action", "topic", or "message"
    print(item.text)        # Formatted text
    print(item.score)       # Native score of the index that produced the item
    print(item.fusion_score)  # Rank-fusion score used to order search() results
    print(item.raw)         # Original TypeAgent object (SemanticRef or Message)
    print(item.timestamp)   # ISO timestamp of the source message, or None
    print(item.valid_from)  # ISO date or None — when memory becomes relevant
    print(item.valid_to)    # ISO date or None — when memory expires
    print(item.status)      # current / superseded / expired / future / unconfirmed
    print(item.source_id)   # Stable source identity
    print(item.memory_id)   # Stable extracted-knowledge identity, if applicable
    print(item.sources)     # Full provenance, including merged neighboring turns
```

**SearchItem.type values** (from TypeAgent's knowledge_type):
- `"entity"` - Concrete entities (people, places, things)
- `"action"` - Actions with verbs, subjects, objects
- `"topic"` - Topic keywords
- `"message"` - Original message text

**score vs fusion_score:**

`score` is the raw score reported by whichever index produced the item:
unbounded term-match weights for structured results, cosine similarity in
`[0, 1]` for embedding results. The two are not comparable, so `search()`
orders its hybrid results by `fusion_score`, computed with reciprocal rank
fusion over both result lists. Items found by *both* paths rank highest.

`fusion_score` is `None` for single-path calls such as
`search_by_embedding()`, where `score` alone already defines the order.

### AddResult

Returned by `add()`:

```python
result = await memory.add("I don't like Python anymore")

print(f"Messages added: {result.messages_added}")
print(f"Knowledge extracted: {result.entities_extracted}")
print(f"Contradictions superseded: {result.contradictions_removed}")
print(result.source_ids, result.memory_ids, result.skipped_source_ids)

# The records themselves, not just the count (None when nothing was retired)
for record in result.superseded or []:
    print(record.ordinal, record.text, "->", record.superseded_by)
```
