# Momex

Momex is a memory layer for AI agents, built on TypeAgent's Structured RAG. It's not a chatbot — it's the memory system you plug into your own agent.

## Features

- **Hybrid search**: `search()` runs structured RAG + embedding similarity in parallel
- **Collections**: Organize memories by user, team, or purpose
- **Hierarchical structure**: Use `:` separator for nested organization (e.g., `momex:engineering:xiaoyuzhang`)
- **Prefix search**: Search across all collections under a prefix (e.g., `momex:*`)
- **Structured knowledge**: Extracts entities, actions, and topics using TypeAgent's KnowledgeExtractor
- **Embedding-only mode**: `search_by_embedding()` for fast similarity search without LLM

## Documentation

- [Usage Guide](momex-usage.md) - How to use Momex
- [Design Document](momex-design.md) - Architecture and internals

## Memory API - Quick Example

```python
import asyncio
from momex import Memory

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

asyncio.run(main())
```

## search() and search_by_embedding()

- `search()` → Hybrid search (structured RAG + embedding), returns `list[SearchItem]`
- `search_by_embedding()` → Embedding similarity only (no LLM needed), returns `list[SearchItem]`

Both return the same `SearchItem` structure:

```python
@dataclass
class SearchItem:
    type: str              # "entity", "action", "topic", or "message"
    text: str              # Human-readable description
    score: float           # Relevance score
    raw: Any               # Original TypeAgent object (SemanticRef or Message)
    timestamp: str | None  # When the memory was recorded (UTC, ISO format)
    valid_from: str | None # Memory active start date (if set)
    valid_to: str | None   # Memory expiration date (if set)
```

Example output:

```python
results = await memory.search("Python")
# [
#   SearchItem(type="entity", text="Python (type: programming_language)", score=100.0, timestamp="2026-04-08T16:47:28Z", ...),
#   SearchItem(type="action", text="user like Python", score=10.0, timestamp="2026-04-08T16:47:28Z", ...),
#   SearchItem(type="message", text="I like Python programming", score=10.0, timestamp="2026-04-08T16:47:28Z", ...),
# ]
```

Use `search()` when you want structured results as context for your own agent/LLM.
Use `search_by_embedding()` as a fast fallback when the LLM is unavailable.

Both automatically filter out expired memories (those past their `valid_to` date). Pass `include_expired=True` to include them.
