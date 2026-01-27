# Memex Design

## Overview

Memex is a high-level memory API for AI agents, built on TypeAgent's Structured RAG. It provides:

- **Collections**: Named storage spaces for organizing memories by user, team, or purpose
- **Hierarchical organization**: Use `:` separator to create nested collections
- **Prefix queries**: Query parent prefix to search all child collections
- **Conversation extraction**: Automatically extract and deduplicate facts from conversations

## Architecture

```
┌─────────────────┐
│   Memex API     │  Memory, MemoryManager, query()
├─────────────────┤
│   TypeAgent     │  Structured RAG, Embeddings, Knowledge Extraction
├─────────────────┤
│   SQLite        │  Per-collection database files
└─────────────────┘
```

## Collection Storage

Collection names map to directory structure. The `:` separator creates subdirectories:

| Collection Name | Storage Path |
|-----------------|--------------|
| `alice` | `./memex_data/alice/memory.db` |
| `user:alice` | `./memex_data/user/alice/memory.db` |
| `company:engineering:alice` | `./memex_data/company/engineering/alice/memory.db` |

This enables prefix-based queries:
- `query("company:engineering:alice", ...)` → searches only alice
- `query("company:engineering", ...)` → searches all under engineering
- `query("company", ...)` → searches entire company

## Conversation Fact Extraction

The `add_conversation()` method processes conversations in 4 stages:

### Stage 1: Fact Extraction

LLM extracts facts from conversation using configured `FactType` definitions.

Input:
```
User: My name is Alice and I love Python
Assistant: Nice to meet you!
User: I'm working on a FastAPI project
```

Output:
```json
{"facts": ["Name is Alice", "Loves Python", "Working on FastAPI project"]}
```

### Stage 2: Vector Search

For each extracted fact, find similar existing memories using cosine similarity:

1. Generate embedding for the new fact
2. Compare against embeddings of existing memories
3. Return top-k most similar memories (above similarity threshold)

This ensures efficient comparison - only relevant memories are considered, not the entire database.

### Stage 3: Decision Making

LLM compares each new fact against similar existing memories and decides:

| Event | When |
|-------|------|
| `ADD` | New information not in memory |
| `UPDATE` | Similar memory exists but new fact has more detail |
| `DELETE` | New fact contradicts existing memory |
| `NONE` | Same information already exists |

### Stage 4: Execute Operations

Execute the decided operations (add new memories, update existing ones).

## Configuration

### Fact Types

Fact types define what information to extract from conversations. Default types (based on mem0):

| Type | Description |
|------|-------------|
| Personal Preferences | Likes, dislikes, preferences for food, products, activities, entertainment |
| Important Personal Details | Names, relationships, important dates |
| Plans and Intentions | Upcoming events, trips, goals, plans |
| Activity and Service Preferences | Dining, travel, hobbies, services |
| Health and Wellness Preferences | Dietary restrictions, fitness routines |
| Professional Details | Job titles, work habits, career goals |
| Miscellaneous Information | Favorite books, movies, brands |

Custom fact types can be defined via YAML config or code.

### Similarity Threshold

Controls how similar an existing memory must be to consider it for UPDATE/DELETE decisions. Default is 0.5 (range 0.0-1.0).

- Higher threshold: More strict matching, more ADD operations
- Lower threshold: More lenient matching, more UPDATE operations

## Soft Delete

TypeAgent uses append-only storage by design. Memex implements soft delete:

- Deleted message IDs stored in `deleted.json` alongside `memory.db`
- Deleted memories filtered out in queries, searches, exports
- Can restore deleted memories via `restore(memory_id)`

```
./memex_data/user/alice/
├── memory.db      # TypeAgent data (append-only)
└── deleted.json   # Soft delete records
```

## Limitations

- **No hard delete**: Messages remain in database, only filtered out
- **No transactions**: Operations are not atomic
