# Momex

Momex is a high-level memory API for AI agents, built on TypeAgent's Structured RAG.

## Features

- **Agent API**: Simple chat interface with automatic memory management
- **Collections**: Organize memories by user, team, or purpose
- **Hierarchical structure**: Use `:` separator for nested organization (e.g., `momex:engineering:xiaoyuzhang`)
- **Prefix queries**: Query `momex` to search all collections under `momex:*`
- **Structured knowledge**: Extracts entities, actions, and topics using TypeAgent's KnowledgeExtractor
- **Term-based indexing**: Fast search using TypeAgent's SemanticRef index
- **Short-term memory**: Session-based conversation history with persistence

## Documentation

- [Usage Guide](momex-usage.md) - How to use Momex
- [Design Document](momex-design.md) - Architecture and internals

## Two API Levels

Momex provides two levels of API:

| Level | Class | Description |
|-------|-------|-------------|
| **Level 1** | `Agent` | High-level chat API - automatic memory management |
| **Level 2** | `Memory`, `ShortTermMemory` | Low-level APIs - manual control |

## Agent API (Level 1)

The simplest way to use Momex. Just call `chat()` and memory is handled automatically:

```python
import asyncio
from momex import Agent, MomexConfig

async def main():
    config = MomexConfig(provider="openai", model="gpt-4o")

    xiaoyuzhang = Agent("user:xiaoyuzhang", config)
    gvanrossum = Agent("user:gvanrossum", config)

    # Chat - LLM decides what to remember
    await xiaoyuzhang.chat("My name is Xiaoyu, I love Python")
    await gvanrossum.chat("I'm Guido, I prefer Java")

    # Memory persists
    r = await xiaoyuzhang.chat("What's my name?")
    print(r.content)  # "Your name is Xiaoyu"

    await xiaoyuzhang.close()
    await gvanrossum.close()

asyncio.run(main())
```

## Memory API (Level 2) - Quick Example

```python
import asyncio
from momex import Memory, query

async def main():
    # Create memory for a user
    memory = Memory(collection="momex:engineering:xiaoyuzhang")

    # Add memories - TypeAgent extracts entities, actions, topics
    await memory.add("I like Python programming")

    # Search - returns structured results
    results = await memory.search("What languages?")
    for item in results:
        print(f"[{item.type}] {item.text} (score={item.score:.2f})")
    # Output: [action] none like Python (score=10.00)
    #         [message] I like Python programming (score=5.49)

    # Query single collection - returns LLM-generated answer
    answer = await memory.query("What does the user like?")
    print(f"A: {answer}")

    # Query across collections (prefix query)
    answer = await query("momex:engineering", "What languages do people use?")
    print(f"A: {answer}")

asyncio.run(main())
```

## search() vs query()

- `search()` → Returns `list[SearchItem]` for you to process
- `query()` → Returns LLM-generated answer string

Use `search()` when you want raw results (e.g., as context for your own LLM).
Use `query()` for a ready-to-use answer.

## Short-Term Memory

`ShortTermMemory` provides session-based conversation history with persistence.
Unlike long-term memory (`Memory`), it stores raw messages without knowledge extraction.

```python
from momex import ShortTermMemory, MomexConfig

config = MomexConfig()

# Create short-term memory (uses context manager for clean connection handling)
with ShortTermMemory("user:xiaoyuzhang", config) as stm:
    # Add messages
    stm.add("Hello, I'm Alice", role="user")
    stm.add("Nice to meet you!", role="assistant")

    # Get recent messages
    for msg in stm.get(limit=10):
        print(f"{msg.role}: {msg.content}")

    # Save session_id for later
    session_id = stm.session_id

# Resume session after restart
with ShortTermMemory("user:xiaoyuzhang", config, session_id=session_id) as stm:
    messages = stm.get_all()  # Previous messages restored

    # List all sessions
    sessions = stm.list_sessions()
    for s in sessions:
        print(f"{s.session_id}: {s.message_count} messages")

    # Start fresh session
    new_id = stm.new_session()
```

### Short-Term Memory API

| Method | Description |
|--------|-------------|
| `add(content, role)` | Add a message |
| `get(limit)` | Get recent messages |
| `get_all()` | Get all messages in session |
| `clear()` | Clear current session |
| `stats()` | Get statistics |
| `new_session()` | Start new session |
| `load_session(id)` | Load existing session |
| `list_sessions()` | List all sessions |
| `delete_session(id)` | Delete a session |
| `cleanup_expired()` | Remove old sessions |
| `close()` | Close database connection |

### Storage

Short-term memory is stored in a separate database file alongside long-term memory:

```
momex_data/
└── user/xiaoyuzhang/
    ├── memory.db        # Long-term memory
    └── short_term.db    # Short-term memory
```
