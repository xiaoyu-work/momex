# Momex

Momex is a memory layer for AI agents, built on TypeAgent's Structured RAG. It's not a chatbot — it's the memory system you plug into your own agent.

## Features

- **Collections**: Organize memories by user, team, or purpose
- **Hierarchical structure**: Use `:` separator for nested organization (e.g., `momex:engineering:xiaoyuzhang`)
- **Prefix queries**: Query `momex` to search all collections under `momex:*`
- **Structured knowledge**: Extracts entities, actions, and topics using TypeAgent's KnowledgeExtractor
- **Term-based indexing**: Fast search using TypeAgent's SemanticRef index
- **Embedding fallback**: `search_by_embedding()` for similarity search without LLM
- **Short-term memory**: Session-based conversation history with persistence

## Documentation

- [Usage Guide](momex-usage.md) - How to use Momex
- [Design Document](momex-design.md) - Architecture and internals

## Memory API - Quick Example

```python
import asyncio
from momex import Memory, query

async def main():
    # Create memory for a user
    memory = Memory(collection="momex:engineering:xiaoyuzhang")

    # Add memories - TypeAgent extracts entities, actions, topics
    await memory.add("I like Python programming")

    # Search - returns structured results
    results = await memory.search("What programming languages?")
    for item in results:
        print(f"[{item.type}] {item.text} (score={item.score:.2f})")
    # Output: [action] none like Python (score=10.00)
    #         [message] I like Python programming (score=5.49)

    # Query single collection - returns LLM-generated answer
    answer = await memory.query("What does the user like?")
    print(f"A: {answer}")

    # Query across collections (prefix query)
    answer = await query("momex:engineering", "What does the user like?")
    print(f"A: {answer}")

asyncio.run(main())
```

## search() vs query()

- `search()` → Returns `list[SearchItem]` for you to process (requires LLM for query translation)
- `search_by_embedding()` → Same as `search()` but uses embedding similarity only (no LLM needed)
- `query()` → Returns LLM-generated answer string

Use `search()` when you want raw results (e.g., as context for your own LLM).
Use `search_by_embedding()` as a fast fallback when the LLM is unavailable.
Use `query()` for a ready-to-use answer.

Both `search()` and `search_by_embedding()` automatically filter out expired memories (those past their `valid_to` date). Pass `include_expired=True` to include them.

## Short-Term Memory

`ShortTermMemory` provides session-based conversation history with persistence.
Unlike long-term memory (`Memory`), it stores raw messages without knowledge extraction.

```python
from momex import ShortTermMemory, MomexConfig

config = MomexConfig()

# Create short-term memory
stm = ShortTermMemory("user:xiaoyuzhang", config)

# Add messages
stm.add("Hello, I'm Alice", role="user")
stm.add("Nice to meet you!", role="assistant")

# Get recent messages
for msg in stm.get(limit=10):
    print(f"{msg.role}: {msg.content}")

# Save session_id for later
session_id = stm.session_id

# Resume session after restart
stm = ShortTermMemory("user:xiaoyuzhang", config, session_id=session_id)
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
