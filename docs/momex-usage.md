# Momex Usage

## Setup

```bash
pip install -e .
```

## Core Concepts

### Async API

Momex is fully async. All operations use `async/await`:

```python
import asyncio
from momex import Memory, MomexConfig

async def main():
    # Configure LLM once (required)
    # Use MOMEX_API_KEY env var for the key
    MomexConfig.set_default(
        provider="openai",
        model="gpt-4o",
    )

    memory = Memory(collection="user:xiaoyuzhang")
    await memory.add("I like Python")
    answer = await memory.query("What language?")
    print(answer)

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

Query behavior:
- `await query("momex:engineering:xiaoyuzhang", ...)` → searches only xiaoyuzhang
- `await query("momex:engineering", ...)` → searches xiaoyuzhang + gvanrossum
- `await query("momex", ...)` → searches xiaoyuzhang + gvanrossum + charlie

## Basic Usage

### Add and Query

```python
import asyncio
from momex import Memory

async def main():
    memory = Memory(collection="user:xiaoyuzhang")

    # Add memories - TypeAgent extracts entities, actions, topics
    await memory.add("I love Python programming")
    await memory.add("Project deadline is Friday")

    # Query - returns LLM-generated answer
    answer = await memory.query("What programming language does the user like?")
    print(answer)

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

    # Query the memories
    answer = await memory.query("What is the user's name?")
```

### Direct Storage (No LLM Processing)

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

    # Later, preference changed - add() automatically removes contradicting memory
    result = await memory.add("I don't like sushi anymore")
    print(f"Added {result.messages_added}, removed {result.contradictions_removed} contradictions")
```

You can disable automatic contradiction detection:

```python
# Skip contradiction detection (faster, but may create inconsistent memories)
await memory.add("I don't like sushi", detect_contradictions=False)
```

### Explicit Delete (Advanced Users)

For manual control over deletion:

```python
async def main():
    memory = Memory(collection="user:xiaoyuzhang")

    # Delete memories matching a query
    deleted = await memory.delete("likes sushi")
    print(f"Deleted {deleted} memories")
```

### Query Across Collections

```python
from momex import Memory, query

async def main():
    # Create memories for different users
    xiaoyuzhang = Memory(collection="momex:engineering:xiaoyuzhang")
    await xiaoyuzhang.add("I like Python")

    gvanrossum = Memory(collection="momex:engineering:gvanrossum")
    await gvanrossum.add("I prefer Java")

    # Query single collection
    answer = await xiaoyuzhang.query("What language?")

    # Query by prefix - searches multiple collections
    answer = await query("momex:engineering", "What languages do people use?")
    answer = await query("momex", "Who works here?")
```

### Search for Raw Results

Use `search()` to get structured results without LLM answer generation. Useful when you want to provide context to your own LLM:

```python
from momex import Memory, search

async def main():
    memory = Memory(collection="momex:engineering:xiaoyuzhang")
    await memory.add("I like Python and FastAPI")

    # Search single collection - returns SearchItem objects
    results = await memory.search("programming")
    for item in results:
        print(f"[{item.type}] {item.text} (score={item.score:.2f})")
        # item.raw contains the original TypeAgent object

    # Search across collections with prefix
    results = await search("momex", "what languages", limit=5)

    # Use as context for your own LLM
    context = "\n".join([f"- [{coll}] {item.text}" for coll, items in results for item in items])
```

**query() vs search():**
- `query()`: Uses LLM to generate a natural language answer
- `search()`: Returns structured `SearchItem` results for you to process

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

## Configuration

Momex supports two storage backends:
- **SQLite** (default): Local file-based storage, one database per collection
- **PostgreSQL**: Shared database for multi-instance deployment

### SQLite Configuration (Default)

```python
from momex import Memory, MomexConfig

# Set global default once
# Use MOMEX_API_KEY env var for the key
MomexConfig.set_default(
    provider="openai",
    model="gpt-4o",
    storage_path="./my_data",  # Optional, default: "./momex_data"
)

# Then use Memory without passing config
memory = Memory(collection="user:xiaoyuzhang")
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
from momex import Memory, MomexConfig, PostgresConfig

MomexConfig.set_default(
    provider="openai",
    model="gpt-4o",
    backend="postgres",
    postgres=PostgresConfig(
        url="postgresql://user:password@localhost:5432/momex",
        pool_min=2,
        pool_max=10,
    )
)

memory = Memory(collection="user:xiaoyuzhang")
```

### YAML Configuration

**SQLite (config_sqlite.yaml):**
```yaml
backend: sqlite
storage_path: ./momex_data

provider: openai
model: gpt-4o
```

**PostgreSQL (config_postgres.yaml):**
```yaml
backend: postgres

postgres:
  url: postgresql://user:password@localhost:5432/momex
  # schema: optional schema for collection isolation
  pool_min: 2
  pool_max: 10
```

If `schema` is not provided, Momex derives one from the collection name by
lowercasing and replacing non-alphanumeric characters with `_`. Different
collection names can map to the same schema (e.g., `a-b` and `a_b`), so set an
explicit schema if you need strict isolation.

**Load from YAML:**
```python
config = MomexConfig.from_yaml("config_postgres.yaml")
MomexConfig._default = config  # Set as global default

memory = Memory(collection="user:xiaoyuzhang")
```

**Save to YAML:**
```python
config.to_yaml("my_config.yaml")
```

### LLM Configuration

LLM is required. Supports **OpenAI**, **Azure**, **Anthropic**, **DeepSeek**, **Qwen**.

```python
from momex import MomexConfig

# OpenAI
MomexConfig.set_default(provider="openai", model="gpt-4o")

# Azure OpenAI
MomexConfig.set_default(provider="azure", model="gpt-4o", api_base="https://xxx.openai.azure.com")

# Anthropic
MomexConfig.set_default(provider="anthropic", model="claude-sonnet-4-20250514")

# DeepSeek
MomexConfig.set_default(provider="deepseek", model="deepseek-chat")

# Qwen (Alibaba Cloud)
MomexConfig.set_default(provider="qwen", model="qwen-plus")
```

**YAML Configuration:**
```yaml
provider: openai  # openai, azure, anthropic, deepseek, qwen
model: gpt-4o
# api_base: https://xxx.openai.azure.com  # Required for Azure
temperature: 0.0
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `MOMEX_BACKEND` | `sqlite` or `postgres` |
| `MOMEX_STORAGE_PATH` | SQLite storage directory |
| `MOMEX_POSTGRES_URL` | PostgreSQL connection URL |
| `MOMEX_POSTGRES_SCHEMA` | Optional schema for collection isolation |
| `MOMEX_PROVIDER` | LLM provider: `openai`, `azure`, `anthropic`, `deepseek`, `qwen` |
| `MOMEX_MODEL` | LLM model name |
| `MOMEX_API_KEY` | LLM API key |
| `MOMEX_API_BASE` | LLM API base URL (required for Azure) |

```bash
# SQLite
export MOMEX_STORAGE_PATH=./my_data

# PostgreSQL
export MOMEX_BACKEND=postgres
export MOMEX_POSTGRES_URL=postgresql://user:pass@localhost:5432/momex

# LLM
export MOMEX_PROVIDER=openai
export MOMEX_MODEL=gpt-4o
export MOMEX_API_KEY=sk-xxx
# export MOMEX_API_BASE=https://xxx.openai.azure.com  # Required for Azure
```

## API Reference

### Memory

All methods are async:

| Method | Description |
|--------|-------------|
| `await add(messages)` | Add memories (auto-detects contradictions) |
| `await query(question)` | Query with natural language (LLM answer) |
| `await search(query, limit=10)` | Search, returns `list[SearchItem]` |
| `await delete(query)` | Delete memories matching query (advanced) |
| `await stats()` | Get memory statistics |
| `await export(path)` | Export to JSON file |
| `await clear()` | Delete all memories in this collection |

**add() parameters:**
- `messages`: str or list[dict] - Content to add
- `infer`: bool (default True) - Use LLM to extract knowledge
- `detect_contradictions`: bool (default True) - Auto-remove contradicting memories

### Prefix Query Functions

All functions are async:

| Function | Description |
|----------|-------------|
| `await query(prefix, question)` | Query with LLM answer (returns str) |
| `await search(prefix, query, limit=10)` | Search (returns list of tuples) |
| `await stats(prefix)` | Get combined stats for matching collections |

### SearchItem

Returned by `search()`:

```python
results = await memory.search("programming")

for item in results:
    print(item.type)   # "entity", "action", "topic", or "message"
    print(item.text)   # Formatted text
    print(item.score)  # Relevance score
    print(item.raw)    # Original TypeAgent object (SemanticRef or Message)
```

**SearchItem.type values** (from TypeAgent's knowledge_type):
- `"entity"` - Concrete entities (people, places, things)
- `"action"` - Actions with verbs, subjects, objects
- `"topic"` - Topic keywords
- `"message"` - Original message text

### AddResult

Returned by `add()`:

```python
result = await memory.add("I don't like Python anymore")

print(f"Messages added: {result.messages_added}")
print(f"Knowledge extracted: {result.entities_extracted}")
print(f"Contradictions removed: {result.contradictions_removed}")
print(f"Success: {result.success}")
```
