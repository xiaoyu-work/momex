# Momex Usage

## Setup

```bash
pip install -e .
```

## Two API Levels

Momex provides two levels of abstraction:

| Level | API | Use Case |
|-------|-----|----------|
| **Level 1** | `Agent` | Chat applications - automatic memory management |
| **Level 2** | `Memory` | Custom agents - full control over memory |

## Level 1: Agent API (Recommended)

The `Agent` class provides a simple chat interface with automatic memory management.
LLM decides what goes to short-term vs long-term memory.

### Basic Usage

```python
import asyncio
from momex import Agent, MomexConfig

async def main():
    config = MomexConfig(provider="openai", model="gpt-4o")
    agent = Agent("user:xiaoyuzhang", config)

    # Just chat - memory is automatic
    response = await agent.chat("My name is Xiaoyu, I'm a Python developer")
    print(response.content)
    # Stored to long-term memory (identity info)

    response = await agent.chat("What's my name?")
    print(response.content)  # "Your name is Xiaoyu"

    response = await agent.chat("What's the weather today?")
    # NOT stored to long-term memory (temporary query)

asyncio.run(main())
```

### How It Works

When you call `agent.chat(message)`:

1. **Short-term**: Message is always recorded to conversation history
2. **Classification**: LLM decides if message should also be stored long-term
3. **Retrieval**: Relevant memories are retrieved from long-term storage
4. **Response**: LLM generates response using context from both memory types

### Session Management

```python
agent = Agent("user:xiaoyuzhang", config)

# Current session
print(agent.session_id)

# Chat in session
await agent.chat("Hello")
await agent.chat("I like coffee")

# Start new session
new_id = agent.new_session()

# List all sessions
sessions = agent.list_sessions()
for s in sessions:
    print(f"{s.session_id}: {s.message_count} messages")

# Load previous session
agent.load_session(sessions[0].session_id)

# Get conversation history
history = agent.get_history()
for msg in history:
    print(f"{msg.role}: {msg.content}")

# Clear history
agent.clear_history()
```

### End Session with Summary

```python
# Chat throughout a session
await agent.chat("My name is Bob")
await agent.chat("I work at Google")
await agent.chat("I prefer Python over Java")

# End session - optionally save summary to long-term memory
summary = await agent.end_session(save_summary=True)
print(summary)  # "Bob works at Google and prefers Python"
```

### Custom System Prompt

```python
agent = Agent(
    "user:xiaoyuzhang",
    config,
    system_prompt="You are a helpful coding assistant. Be concise.",
    max_context_messages=20,      # Recent messages to include
    max_retrieved_memories=10,    # Long-term memories to retrieve
)
```

### Agent API Reference

| Method | Description |
|--------|-------------|
| `await chat(message)` | Send message, get response (auto memory) |
| `new_session()` | Start new session (returns session_id) |
| `load_session(id)` | Load existing session (returns bool) |
| `list_sessions()` | List all sessions |
| `delete_session(id)` | Delete a session |
| `get_history()` | Get conversation history |
| `clear_history()` | Clear current session |
| `await end_session(save_summary)` | End session, optionally save summary |
| `cleanup_expired_sessions()` | Remove old sessions |
| `stats()` | Get short-term stats |
| `await stats_async()` | Get full stats (short + long term) |

### ChatResponse

```python
response = await agent.chat("I'm Alice")

print(response.content)  # Assistant's reply
```

The Agent automatically handles memory decisions internally. If you need to inspect or control memory storage, use the Level 2 API (`Memory`). For cross-collection queries, use the module-level `query()` and `search()` functions.

---

## Level 2: Memory APIs

For custom agents that need full control over memory.

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
    answer = await xiaoyuzhang.query("What programming language?")

    # Query by prefix - searches multiple collections
    answer = await query("momex:engineering", "What programming languages do people use?")
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
    results = await search("momex", "what programming languages", limit=5)

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

Configuration has three parts:
- **LLM**: Required for knowledge extraction and query answering
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

### YAML Configuration

**config.yaml:**
```yaml
llm:
  provider: openai
  model: gpt-4o
  api_key: sk-xxx  # or use MOMEX_LLM_API_KEY env var
  temperature: 0.0

# embedding: (optional, auto-inferred for OpenAI/Azure)
#   provider: openai
#   model: text-embedding-3-small
#   api_key: sk-xxx

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
config.to_yaml("my_config.yaml")
```

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

## Short-Term Memory

`ShortTermMemory` provides session-based conversation history with persistence.
Unlike `Memory`, it stores raw messages without LLM knowledge extraction.

### Basic Usage

```python
from momex import ShortTermMemory, MomexConfig

config = MomexConfig()

# Use context manager for clean connection handling
with ShortTermMemory("user:xiaoyuzhang", config) as stm:
    # Add messages
    stm.add("Hello, I'm Alice", role="user")
    stm.add("Nice to meet you, Alice!", role="assistant")
    stm.add("I work at Google", role="user")

    # Get recent messages
    messages = stm.get(limit=10)
    for msg in messages:
        print(f"{msg.role}: {msg.content}")

    # Get all messages
    all_messages = stm.get_all()

    # Get statistics
    stats = stm.stats()
    print(f"Messages: {stats['message_count']}")
```

### Session Management

```python
from momex import ShortTermMemory

with ShortTermMemory("user:xiaoyuzhang", config) as stm:
    # Current session ID
    print(f"Session: {stm.session_id}")

    stm.add("First session message", role="user")
    first_session = stm.session_id

    # Start new session
    new_session = stm.new_session()
    stm.add("Second session message", role="user")

    # List all sessions
    sessions = stm.list_sessions()
    for s in sessions:
        print(f"{s.session_id}: {s.message_count} messages")

    # Load previous session
    stm.load_session(first_session)
    print(stm.get_all())  # Shows first session messages

    # Delete a session
    stm.delete_session(new_session)
```

### Persistence Across Restarts

```python
# First run
with ShortTermMemory("user:xiaoyuzhang", config) as stm:
    stm.add("Remember this", role="user")
    saved_session_id = stm.session_id

# After restart - resume session
with ShortTermMemory("user:xiaoyuzhang", config, session_id=saved_session_id) as stm:
    messages = stm.get_all()  # Previous messages restored
```

### Cleanup

```python
with ShortTermMemory("user:xiaoyuzhang", config, session_ttl_hours=24) as stm:
    # Remove sessions older than 24 hours
    deleted = stm.cleanup_expired()
    print(f"Deleted {deleted} old messages")

    # Clear current session
    stm.clear()
```

### ShortTermMemory API

| Method | Description |
|--------|-------------|
| `add(content, role="user")` | Add a message (returns `Message`) |
| `get(limit=20)` | Get recent messages |
| `get_all()` | Get all messages in current session |
| `clear()` | Clear current session |
| `stats()` | Get statistics |
| `new_session()` | Start new session (returns new session_id) |
| `load_session(session_id)` | Load existing session (returns bool) |
| `list_sessions(limit=50)` | List all sessions (returns `list[SessionInfo]`) |
| `delete_session(session_id)` | Delete a session (returns bool) |
| `cleanup_expired()` | Remove old sessions (returns count) |
| `close()` | Close database connection |

### Message and SessionInfo

```python
# Message dataclass
msg = stm.add("Hello", role="user")
print(msg.role)       # "user"
print(msg.content)    # "Hello"
print(msg.timestamp)  # ISO format
print(msg.id)         # Database ID

# SessionInfo dataclass
sessions = stm.list_sessions()
for s in sessions:
    print(s.session_id)
    print(s.started_at)
    print(s.last_message_at)
    print(s.message_count)
```

### Storage Location

Short-term memory is stored alongside long-term memory:

```
momex_data/
└── user/xiaoyuzhang/
    ├── memory.db        # Long-term memory (Memory class)
    └── short_term.db    # Short-term memory (ShortTermMemory class)
```
