# Momex

Momex is a high-level memory API for AI agents, built on TypeAgent's Structured RAG.

## Features

- **Collections**: Organize memories by user, team, or purpose
- **Hierarchical structure**: Use `:` separator for nested organization (e.g., `momex:engineering:xiaoyuzhang`)
- **Prefix queries**: Query `momex` to search all collections under `momex:*`
- **Structured knowledge**: Extracts entities, actions, and topics using TypeAgent's KnowledgeExtractor
- **Term-based indexing**: Fast search using TypeAgent's SemanticRef index

## Documentation

- [Usage Guide](momex-usage.md) - How to use Momex
- [Design Document](momex-design.md) - Architecture and internals

## Quick Example

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
