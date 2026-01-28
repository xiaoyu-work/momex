# Momex Usage

## Setup

```bash
pip install -e .

# LLM configuration (via TypeAgent)
export OPENAI_API_KEY=your-key
export OPENAI_MODEL=gpt-4o

# Or for Azure:
export AZURE_OPENAI_API_KEY=your-key
export AZURE_OPENAI_ENDPOINT=https://xxx.openai.azure.com
```

## Core Concepts

### Async API

Momex is fully async. All operations use `async/await`:

```python
import asyncio
from momex import Memory

async def main():
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

    # Add memories - LLM extracts facts and deduplicates automatically
    await memory.add("I love Python programming")
    await memory.add("Project deadline is Friday")

    # Adding similar content will UPDATE existing memory, not create duplicates
    await memory.add("I really enjoy Python coding")  # Updates existing

    # Query
    answer = await memory.query("What programming language does the user like?")
    print(answer)

asyncio.run(main())
```

### Add with Conversation Format

You can also pass conversation messages:

```python
async def main():
    memory = Memory(collection="user:xiaoyuzhang")

    # Conversation format - LLM extracts facts from the dialogue
    await memory.add([
        {"role": "user", "content": "My name is Alice, I'm a Python developer"},
        {"role": "assistant", "content": "Nice to meet you!"},
        {"role": "user", "content": "I'm working on a FastAPI project"},
    ])

    # Query the memories
    answer = await memory.query("What is the user's name?")  # "Alice"
```

### Direct Storage (No LLM Processing)

Use `infer=False` to skip LLM processing and store content directly:

```python
async def main():
    memory = Memory(collection="user:xiaoyuzhang")

    # Direct storage - no fact extraction or deduplication
    await memory.add("Raw log: user logged in at 2024-01-01", infer=False)

    # Useful for structured data that doesn't need LLM processing
    await memory.add("Session ID: abc123", infer=False)
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

### Search for Raw Results (for Chat Agent Context)

Use `search()` to get raw vector search results without LLM summarization. This is useful when you want to provide context to a chat agent:

```python
from momex import Memory, search

async def main():
    xiaoyuzhang = Memory(collection="momex:engineering:xiaoyuzhang")
    await xiaoyuzhang.add("I like Python and FastAPI")
    await xiaoyuzhang.add("Working on ML project")

    # Search single collection - returns MemoryItem objects
    # Results are filtered by similarity threshold, then ranked by weighted score
    results = await xiaoyuzhang.search("programming")
    for r in results:
        print(f"[sim={r.similarity:.3f}, imp={r.importance:.2f}, score={r.score:.3f}] {r.text}")

    # Search across collections with prefix
    results = await search("momex", "what are they working on", limit=5)

    # Use as context for a chat agent (no LLM call, cheaper than query())
    context = "\n".join([f"- {r.speaker}: {r.text}" for r in results])
    # Pass context to your chat agent...
```

**query() vs search():**
- `query()`: Uses LLM to summarize results into a natural language answer.
- `search()`: Returns raw vector search results with similarity scores, good for providing context to chat agents.

### Manage Collections

```python
from momex import MemoryManager

manager = MemoryManager()

# List collections
all_collections = manager.list_collections()
eng_only = manager.list_collections(prefix="momex:engineering")

# Other operations
manager.exists("momex:engineering:xiaoyuzhang")
manager.delete("momex:engineering:xiaoyuzhang")
manager.rename("user:old", "user:new")
```

## Configuration

### Storage Backends

Momex supports two storage backends:

| Backend | Use Case | Features |
|---------|----------|----------|
| **SQLite** (default) | Local development, single user | Zero config, file-based |
| **PostgreSQL** | Production, multi-user, cloud | pgvector for fast search, scales |

#### SQLite (Default)

```python
from momex import Memory, MomexConfig, StorageConfig

# Default - uses SQLite
memory = Memory(collection="user:xiaoyuzhang")

# Explicit SQLite config
config = MomexConfig(
    storage=StorageConfig(
        backend="sqlite",
        path="./my_data",
    )
)
memory = Memory(collection="user:xiaoyuzhang", config=config)
```

#### PostgreSQL

Requires `asyncpg` and PostgreSQL with `pgvector` extension:

```bash
pip install asyncpg
```

```python
from momex import Memory, MomexConfig, StorageConfig

config = MomexConfig(
    storage=StorageConfig(
        backend="postgres",
        connection_string="postgresql://user:pass@localhost/momex",
        table_prefix="momex",  # optional, default "momex"
    )
)
memory = Memory(collection="user:xiaoyuzhang", config=config)
```

#### Cloud Database Support

PostgreSQL backend works with all major cloud providers:

| Provider | Connection String |
|----------|------------------|
| **AWS RDS** | `postgresql://user:pass@xxx.rds.amazonaws.com:5432/momex` |
| **Azure** | `postgresql://user:pass@xxx.postgres.database.azure.com:5432/momex` |
| **Supabase** | `postgresql://user:pass@db.xxx.supabase.co:5432/postgres` |
| **Neon** | `postgresql://user:pass@xxx.neon.tech/momex` |
| **Vercel Postgres** | `postgresql://user:pass@xxx.vercel-storage.com/momex` |

### MomexConfig

```python
from momex import Memory, MomexConfig, StorageConfig

# Simple SQLite config (legacy style)
config = MomexConfig(storage_path="./my_data")
memory = Memory(collection="user:xiaoyuzhang", config=config)

# Or set global default
MomexConfig.set_default(storage_path="./my_data")
xiaoyuzhang = Memory(collection="user:xiaoyuzhang")  # uses default
gvanrossum = Memory(collection="user:gvanrossum")    # uses default
```

### YAML Configuration

```yaml
# momex_config.yaml

# Storage backend configuration
storage:
  backend: sqlite  # or "postgres"
  path: ./momex_data  # for sqlite
  # connection_string: postgresql://...  # for postgres
  # table_prefix: momex  # for postgres

# Embedding model - affects search quality and similarity scores
# Options: text-embedding-ada-002, text-embedding-3-small, text-embedding-3-large
embedding_model: text-embedding-3-small

# Similarity threshold - depends on embedding model
# Recommended:
#   text-embedding-ada-002:  0.5-0.7 (scores tend to cluster in 0.6-0.9)
#   text-embedding-3-small:  0.3-0.5 (better score distribution)
#   text-embedding-3-large:  0.3-0.5 (best discrimination)
similarity_threshold: 0.3

# Importance weight for search ranking (0.0-1.0)
# Higher = importance matters more in ranking
importance_weight: 0.3

# Custom fact types to extract
fact_types:
  - name: Personal Preferences
    description: Likes, dislikes, preferences for food, products, activities
  - name: Professional Details
    description: Job titles, projects, skills, career goals
```

```python
config = MomexConfig.from_yaml("momex_config.yaml")
memory = Memory(collection="user:xiaoyuzhang", config=config)
```

### Embedding Model Comparison

| Model | Dimensions | Score Distribution | Recommended Threshold |
|-------|------------|-------------------|----------------------|
| `text-embedding-ada-002` | 1536 | 0.6-0.9 (clustered) | 0.5-0.7 |
| `text-embedding-3-small` | 1536 | 0.1-0.6 (spread out) | 0.3-0.5 |
| `text-embedding-3-large` | 3072 | 0.1-0.6 (best) | 0.3-0.5 |

### Custom Fact Types

Control what types of information to extract from conversations:

```python
from momex import FactType, MomexConfig

config = MomexConfig(
    fact_types=[
        FactType(
            name="Technical Skills",
            description="Programming languages, frameworks, and tools the user knows"
        ),
        FactType(
            name="Project Information",
            description="Current projects, deadlines, and team members"
        ),
    ],
)
```

## API Reference

### Memory

All methods are async:

| Method | Description |
|--------|-------------|
| `await add(messages, infer=True)` | Add memories with LLM deduplication (default) |
| `await add(text, infer=False)` | Add directly without LLM processing |
| `await query(question)` | Query with natural language (LLM summarized) |
| `await search(query, limit=10, threshold=None)` | Vector similarity search |
| `await delete(memory_id)` | Soft delete a memory |
| `await restore(memory_id)` | Restore a deleted memory |
| `await list_deleted()` | List all deleted memories |
| `await stats()` | Get memory statistics |
| `await export(path)` | Export to JSON file |
| `await clear()` | Delete all memories in this collection |

**add() parameters:**
- `messages`: str or list[dict] - Content to add
- `infer`: bool (default True) - Use LLM to extract facts and deduplicate
- `similarity_limit`: int (default 5) - Max similar memories to consider

### Prefix Query Functions

All functions are async:

| Function | Description |
|----------|-------------|
| `await query(prefix, question)` | Query with LLM summarization (returns answer string) |
| `await search(prefix, query, limit=10, threshold=None)` | Vector similarity search (returns MemoryItem list) |
| `await stats(prefix)` | Get combined stats for matching collections |

**MemoryItem fields:**
- `id`: Memory identifier
- `text`: The memory content
- `speaker`: Who said this (collection name by default)
- `timestamp`: When it was added
- `score`: Weighted score combining similarity and importance (0.0-1.0)
- `similarity`: Raw similarity score before importance weighting (0.0-1.0)
- `importance`: Importance score (0.0-1.0, auto-assigned by LLM)
- `collection`: Which collection this came from

### Importance Scoring

Momex automatically assigns importance scores to memories based on content type:

| Category | Importance | Examples |
|----------|------------|----------|
| Health/Safety | 0.9-1.0 | Allergies, medical conditions |
| Identity | 0.7-0.8 | Name, family, job title |
| Preferences | 0.5-0.6 | Likes, dislikes, hobbies |
| Casual | 0.3-0.4 | Recent events, plans |

**Two-stage search ranking:**

1. **Filter**: Only memories with `similarity > threshold` pass (default 0.3)
2. **Rank**: Among filtered results, sort by weighted score:
   ```
   final_score = similarity * (1 - importance_weight) + importance * importance_weight
   ```
   With default `importance_weight=0.3`:
   ```
   final_score = similarity * 0.7 + importance * 0.3
   ```

This ensures:
- Irrelevant content is filtered out regardless of importance
- Among relevant results, important information ranks higher

### AddResult

Returned by `add()`:

```python
result = await memory.add("I like Python")

print(f"Added: {result.messages_added}")
print(f"Entities extracted: {result.entities_extracted}")
print(f"Success: {result.success}")
```
