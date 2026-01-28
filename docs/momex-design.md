# Momex Design

## Overview

Momex is a high-level memory API for AI agents, built on TypeAgent's Structured RAG. It provides:

- **Collections**: Named storage spaces for organizing memories by user, team, or purpose
- **Hierarchical organization**: Use `:` separator to create nested collections
- **Prefix queries**: Query parent prefix to search all child collections
- **Structured knowledge**: Uses TypeAgent's KnowledgeExtractor for entities, actions, topics

## Architecture

```
┌─────────────────┐
│   Momex API     │  Memory, MemoryManager, query(), search()
├─────────────────┤
│   TypeAgent     │  ConversationBase, KnowledgeExtractor, SemanticRefIndex
├─────────────────┤
│   SQLite        │  Per-collection database files (via SqliteStorageProvider)
└─────────────────┘
```

## Collection Storage

Collection names map to directory structure. The `:` separator creates subdirectories:

| Collection Name | Storage Path |
|-----------------|--------------|
| `xiaoyuzhang` | `./momex_data/xiaoyuzhang/memory.db` |
| `user:xiaoyuzhang` | `./momex_data/user/xiaoyuzhang/memory.db` |
| `momex:engineering:xiaoyuzhang` | `./momex_data/momex/engineering/xiaoyuzhang/memory.db` |

This enables prefix-based queries:
- `query("momex:engineering:xiaoyuzhang", ...)` → searches only xiaoyuzhang
- `query("momex:engineering", ...)` → searches all under engineering
- `query("momex", ...)` → searches entire momex

## Knowledge Extraction

When you call `add()`, TypeAgent's KnowledgeExtractor processes the text:

### Input
```
"I like Python programming"
```

### Output (KnowledgeResponse)
```python
KnowledgeResponse(
    entities=[
        ConcreteEntity(name="Python", type=["programming language"])
    ],
    actions=[
        Action(verbs=["like"], subject_entity_name="speaker", object_entity_name="Python")
    ],
    topics=[
        Topic(text="programming"),
        Topic(text="languages")
    ]
)
```

## Data Flow

### add()

```
Input text
    ↓
ConversationMessage (with speaker metadata)
    ↓
add_messages_with_indexing()
    ↓
KnowledgeExtractor.extract()  [LLM call]
    ↓
SemanticRefs created for entities, actions, topics
    ↓
Terms indexed in SemanticRefIndex
```

### search()

```
Query text
    ↓
search_conversation_with_language()
    ↓
[LLM] Translate to structured SearchQuery
    ↓
Lookup in SemanticRefIndex
    ↓
Return ConversationSearchResult
    ↓
Wrap as list[SearchItem]
```

### query()

```
search() results
    ↓
Format as context
    ↓
[LLM] Generate answer
    ↓
Return string
```

## SearchItem

Momex wraps TypeAgent's search results in a simple `SearchItem` dataclass:

```python
@dataclass
class SearchItem:
    type: str   # "entity", "action", "topic", "message"
    text: str   # Formatted text
    score: float
    raw: Any    # Original TypeAgent object
```

This provides:
- Friendly text formatting
- Access to raw TypeAgent objects via `.raw`
- Type information from TypeAgent's native `knowledge_type`

