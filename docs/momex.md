# Momex

Momex is a high-level memory API for AI agents, built on TypeAgent's Structured RAG.

## Features

- **Collections**: Organize memories by user, team, or purpose
- **Hierarchical structure**: Use `:` separator for nested organization (e.g., `company:engineering:alice`)
- **Prefix queries**: Query `company` to search all collections under `company:*`
- **Conversation support**: Add memories directly from chat history
- **Configurable**: Define what information to remember via YAML or code

## Documentation

- [Usage Guide](momex-usage.md) - How to use Momex
- [Design Document](momex-design.md) - Architecture and internals

## Quick Example

```python
from momex import Memory, query

# Create memory for a user
memory = Memory(collection="company:engineering:alice")

# Add memories from conversation
memory.add_conversation([
    {"role": "user", "content": "I'm Alice, a Python developer"},
    {"role": "assistant", "content": "Nice to meet you!"},
])

# Query single collection
answer = memory.query("What does the user do?")

# Query across collections (prefix query)
answer = query("company:engineering", "What languages do people use?")
```
