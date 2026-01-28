# Momex

Momex is a high-level memory API for AI agents, built on TypeAgent's Structured RAG.

## Features

- **Collections**: Organize memories by user, team, or purpose
- **Hierarchical structure**: Use `:` separator for nested organization (e.g., `momex:engineering:xiaoyuzhang`)
- **Prefix queries**: Query `momex` to search all collections under `momex:*`
- **Conversation support**: Add memories directly from chat history
- **Configurable**: Define what information to remember via YAML or code

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

    # Add memories from conversation
    await memory.add([
        {"role": "user", "content": "I'm Xiaoyuzhang, a Python developer"},
        {"role": "assistant", "content": "Nice to meet you!"},
    ])

    # Query single collection
    answer = await memory.query("What does the user do?")

    # Query across collections (prefix query)
    answer = await query("momex:engineering", "What languages do people use?")

asyncio.run(main())
```
