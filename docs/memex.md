# Memex - Hierarchical Memory API

Memex is a high-level wrapper around TypeAgent's Structured RAG, providing:

- **Simplified API** - No need to manage `TranscriptMessage` or async/await
- **Hierarchical collections** - Use `:` to create nested organization (unlimited levels)
- **Prefix queries** - Query `company` to search all under `company:*`
- **Sync & Async APIs** - Use whichever fits your application

## Installation

This is a fork. Clone and install in editable mode:

```bash
git clone https://github.com/xiaoyu-work/typeagent-py.git
cd typeagent-py
pip install -e .
```

## Quick Start

### Add Memories

```python
from memex import Memory

# Create memory with hierarchical identity
alice_bot = Memory(collection="company:engineering:alice")

# Add memories - they belong to this collection
alice_bot.add("I like Python")
alice_bot.add("The deadline is Friday")
```

### Query with Prefix

```python
from memex import query

# Query single person
answer = query("company:engineering:alice", "What does Alice like?")

# Query entire engineering team
answer = query("company:engineering", "What are the deadlines?")

# Query entire company
answer = query("company", "Who likes Python?")
```

## Storage Structure

The `:` separator creates subdirectories. Any number of levels supported:

```
./memex_data/
└── company/
    ├── engineering/
    │   ├── alice/
    │   │   └── memory.db    # company:engineering:alice
    │   └── bob/
    │       └── memory.db    # company:engineering:bob
    └── marketing/
        └── charlie/
            └── memory.db    # company:marketing:charlie
```

Query behavior:
- `query("company:engineering:alice", ...)` → searches only Alice
- `query("company:engineering", ...)` → searches Alice + Bob
- `query("company", ...)` → searches Alice + Bob + Charlie

## API Reference

### Memory Class

Read/write for a single collection.

```python
Memory(
    collection: str,                    # Hierarchical name: "company:team:user"
    config: MemexConfig | None = None,
)
```

#### Methods

| Method | Description |
|--------|-------------|
| `add(text, speaker?, timestamp?, tags?)` | Add a memory |
| `add_batch(items)` | Add multiple memories |
| `query(question)` | Query this collection only |
| `search(query, limit=10)` | Search this collection only |
| `stats()` | Get statistics |
| `export(path)` | Export to JSON |
| `clear()` | Delete all memories |

### Prefix Query Functions

Query across multiple collections by prefix.

```python
from memex import query, search, stats

# Query all collections matching prefix
answer = query("company:engineering", "question")

# Search all collections matching prefix
results = search("company:engineering", "keyword", limit=10)

# Get stats for all collections matching prefix
info = stats("company:engineering")
```

### MemoryManager Class

Manage collections (list, delete, rename).

```python
from memex import MemoryManager

manager = MemoryManager()

# List all collections
collections = manager.list_collections()

# List collections by prefix
eng_collections = manager.list_collections(prefix="company:engineering")

# Other operations
manager.exists("company:engineering:alice")
manager.delete("company:engineering:alice")
manager.rename("company:alice", "company:alice_backup")
manager.copy("company:alice", "company:alice_copy")
manager.info("company:alice")
```

### MemexConfig Class

```python
MemexConfig(
    storage_path: str = "./memex_data",
    llm_provider: str = "openai",       # "openai", "azure", "ollama", "anthropic"
    llm_model: str | None = None,
    llm_api_key: str | None = None,
    llm_endpoint: str | None = None,
    auto_extract: bool = True,
    db_name: str = "memory.db",
)
```

## Use Case: Personal & Team Assistants

```python
from memex import Memory, query

class Assistant:
    def __init__(self, identity: str):
        """Identity is the hierarchical collection path."""
        self.memory = Memory(collection=identity)
        self.identity = identity

    def remember(self, text: str):
        """Add memory to this assistant's collection."""
        self.memory.add(text)

    def ask(self, question: str, scope: str | None = None):
        """Query memories. Scope determines search range."""
        search_prefix = scope or self.identity
        return query(search_prefix, question)

# Create assistants
alice = Assistant(identity="company:engineering:alice")
bob = Assistant(identity="company:engineering:bob")

# Add memories - each goes to their own collection
alice.remember("I like Python")
bob.remember("I prefer Java")

# Query with different scopes
alice.ask("What do I like?")                           # Only Alice's memories
alice.ask("What languages?", scope="company:engineering")  # Alice + Bob
alice.ask("What languages?", scope="company")          # Entire company
```

## Configuration

### Environment Variables

```bash
# OpenAI (default)
OPENAI_API_KEY=your-api-key
OPENAI_MODEL=gpt-4o

# Or Azure OpenAI
AZURE_OPENAI_API_KEY=your-azure-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/

# Custom storage path
MEMEX_STORAGE_PATH=/path/to/data
```

### Custom Config

```python
from memex import Memory, MemexConfig

config = MemexConfig(
    storage_path="/var/data/memories",
    llm_provider="azure",
    llm_model="gpt-4",
)

memory = Memory(collection="company:alice", config=config)
```
