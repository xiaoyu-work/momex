# Memex

Memory API for AI agents, built on TypeAgent's Structured RAG.

## Docs

- [Usage Guide](memex-usage.md)
- [Design Document](memex-design.md)

## Quick Example

```python
from memex import Memory

memory = Memory(collection="user:alice")

# Extract facts from conversation
result = memory.add_conversation([
    {"role": "user", "content": "I'm Alice, a Python developer"},
])

# Query
answer = memory.query("What does the user do?")
```
