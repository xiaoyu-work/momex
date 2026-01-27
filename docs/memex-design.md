# Memex Design

Memex is a memory API built on TypeAgent's Structured RAG.

## Architecture

```
Memex API  →  TypeAgent (Structured RAG)  →  SQLite
```

## Hierarchical Collections

Collection names use `:` separator, stored as directories:

```
company:engineering:alice  →  ./memex_data/company/engineering/alice/memory.db
```

Prefix query behavior:
- `query("company:engineering:alice", ...)` → only alice
- `query("company:engineering", ...)` → alice + bob
- `query("company", ...)` → all under company

## Conversation Fact Extraction

`add_conversation()` flow:

```
Conversation
    ↓
[Stage 1] LLM extracts facts (configurable fact types)
    ↓
[Stage 2] Vector search finds similar existing memories
    ↓
[Stage 3] LLM decides: ADD / UPDATE / DELETE / NONE
    ↓
[Stage 4] Execute operations
```

### Decision Logic

| Scenario | Decision |
|----------|----------|
| New info | ADD |
| More detail than existing | UPDATE |
| Contradicts existing | DELETE |
| Already exists | NONE |

## Configuration

Fact types can be customized via YAML or code. Default types (from mem0):
- Personal Preferences
- Important Personal Details
- Plans and Intentions
- Professional Details
- etc.

## Limitations

- DELETE not fully implemented (TypeAgent limitation)
- Operations are not atomic
