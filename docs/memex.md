# Memex

Memex is a high-level memory API for AI agents, built on TypeAgent's Structured RAG.

## Features

- **Collections**: Organize memories by user, team, or purpose
- **Hierarchical structure**: Use `:` separator for nested organization (e.g., `company:engineering:alice`)
- **Prefix queries**: Query `company` to search all collections under `company:*`
- **Conversation extraction**: Automatically extract facts from chat history
- **Configurable fact types**: Define what information to extract via YAML or code

## Documentation

- [Usage Guide](memex-usage.md) - How to use Memex
- [Design Document](memex-design.md) - Architecture and internals

## Quick Example

```python
from memex import Memory, query

# Create memory for a user
memory = Memory(collection="company:engineering:alice")

# Extract facts from conversation
result = memory.add_conversation([
    {"role": "user", "content": "I'm Alice, a Python developer"},
    {"role": "assistant", "content": "Nice to meet you!"},
])
# Extracts: ["Name is Alice", "Is a Python developer"]

# Query single collection
answer = memory.query("What does the user do?")

# Query across collections (prefix query)
answer = query("company:engineering", "What languages do people use?")
```
