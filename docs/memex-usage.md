# Memex Usage

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

Memex is fully async. All operations use `async/await`:

```python
import asyncio
from memex import Memory

async def main():
    memory = Memory(collection="user:alice")
    await memory.add("I like Python")
    answer = await memory.query("What language?")
    print(answer)

asyncio.run(main())
```

### Collection

A **collection** is a named storage space for memories. Each collection has its own database file.

- Use collections to separate memories by user, team, or purpose
- Collection names support hierarchical structure with `:` separator
- Examples: `"alice"`, `"user:alice"`, `"company:engineering:alice"`

### Hierarchical Collections

The `:` separator creates a hierarchy that enables prefix queries:

```
company:engineering:alice  →  ./memex_data/company/engineering/alice/memory.db
company:engineering:bob    →  ./memex_data/company/engineering/bob/memory.db
company:marketing:charlie  →  ./memex_data/company/marketing/charlie/memory.db
```

Query behavior:
- `await query("company:engineering:alice", ...)` → searches only alice
- `await query("company:engineering", ...)` → searches alice + bob
- `await query("company", ...)` → searches alice + bob + charlie

## Basic Usage

### Add and Query

```python
import asyncio
from memex import Memory

async def main():
    memory = Memory(collection="user:alice")

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
    memory = Memory(collection="user:alice")

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
    memory = Memory(collection="user:alice")

    # Direct storage - no fact extraction or deduplication
    await memory.add("Raw log: user logged in at 2024-01-01", infer=False)

    # Useful for structured data that doesn't need LLM processing
    await memory.add("Session ID: abc123", infer=False)
```

### Query Across Collections

```python
from memex import Memory, query

async def main():
    # Create memories for different users
    alice = Memory(collection="company:engineering:alice")
    await alice.add("I like Python")

    bob = Memory(collection="company:engineering:bob")
    await bob.add("I prefer Java")

    # Query single collection
    answer = await alice.query("What language?")

    # Query by prefix - searches multiple collections
    answer = await query("company:engineering", "What languages do people use?")
    answer = await query("company", "Who works here?")
```

### Search for Raw Results (for Chat Agent Context)

Use `search()` to get raw vector search results without LLM summarization. This is useful when you want to provide context to a chat agent:

```python
from memex import Memory, search

async def main():
    alice = Memory(collection="company:engineering:alice")
    await alice.add("I like Python and FastAPI")
    await alice.add("Working on ML project")

    # Search single collection - returns MemoryItem objects with similarity scores
    results = await alice.search("programming")
    for r in results:
        print(f"[{r.score:.3f}] {r.text}")

    # Search across collections with prefix
    results = await search("company", "what are they working on", limit=5)

    # Use as context for a chat agent (no LLM call, cheaper than query())
    context = "\n".join([f"- {r.speaker}: {r.text}" for r in results])
    # Pass context to your chat agent...
```

**query() vs search():**
- `query()`: Uses LLM to summarize results into a natural language answer.
- `search()`: Returns raw vector search results with similarity scores, good for providing context to chat agents.

### Manage Collections

```python
from memex import MemoryManager

manager = MemoryManager()

# List collections
all_collections = manager.list_collections()
eng_only = manager.list_collections(prefix="company:engineering")

# Other operations
manager.exists("company:engineering:alice")
manager.delete("company:engineering:alice")
manager.rename("user:old", "user:new")
```

## Configuration

### MemexConfig

```python
from memex import Memory, MemexConfig

# Custom storage path
config = MemexConfig(storage_path="./my_data")
memory = Memory(collection="user:alice", config=config)

# Or set global default
MemexConfig.set_default(storage_path="./my_data")
alice = Memory(collection="user:alice")  # uses default
bob = Memory(collection="user:bob")      # uses default
```

### YAML Configuration

```yaml
# memex_config.yaml
storage_path: ./memex_data
similarity_threshold: 0.5

fact_types:
  - name: Personal Preferences
    description: Likes, dislikes, preferences for food, products, activities
  - name: Professional Details
    description: Job titles, projects, skills, career goals
```

```python
config = MemexConfig.from_yaml("memex_config.yaml")
memory = Memory(collection="user:alice", config=config)
```

### Custom Fact Types

Control what types of information to extract from conversations:

```python
from memex import FactType, MemexConfig

config = MemexConfig(
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
- `score`: Similarity score (0.0-1.0, higher is more relevant)
- `collection`: Which collection this came from

### AddResult

Returned by `add()`:

```python
result = await memory.add("I like Python")

print(f"Added: {result.messages_added}")
print(f"Entities extracted: {result.entities_extracted}")
print(f"Success: {result.success}")
```
