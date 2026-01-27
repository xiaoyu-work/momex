# Memex - Simplified Collection-Based Memory API

Memex is a high-level wrapper around TypeAgent's Structured RAG, providing:

- **Simplified API** - No need to manage `TranscriptMessage` or async/await
- **Collection-based isolation** - Flexible grouping with any naming scheme
- **Sync & Async APIs** - Use whichever fits your application
- **Auto-configuration** - Automatic `.env` loading and path management

## Installation

This is a fork. Clone and install in editable mode:

```bash
git clone https://github.com/xiaoyu-work/typeagent-py.git
cd typeagent-py
pip install -e .
```

## Quick Start

### Single Collection

```python
from memex import Memory

# Create memory for a collection
memory = Memory(collection="user:alice")

# Add memories
memory.add("Alice likes cats")
memory.add("The project deadline is Friday")

# Query with natural language
answer = memory.query("What does Alice like?")
print(answer)  # "Alice likes cats"
```

### Multiple Collections with MemoryPool

```python
from memex import MemoryPool

# Create a pool with multiple collections
pool = MemoryPool(
    collections=["user:alice", "team:engineering", "project:x"],
    default_collection="user:alice"
)

# Add to specific collections
pool.add("Personal note")  # Goes to default collection
pool.add("Team decision", collections=["team:engineering", "project:x"])

# Query across all collections
answer = pool.query("What decisions were made?")
```

### Managing Collections

```python
from memex import MemoryManager

manager = MemoryManager()

# List all collections
collections = manager.list_collections()

# Check if collection exists
if manager.exists("user:alice"):
    info = manager.info("user:alice")
    print(f"Size: {info['size']}")

# Delete a collection
manager.delete("user:old_user")

# Rename a collection
manager.rename("user:alice", "user:alice_backup")

# Copy a collection
manager.copy("user:alice", "user:alice_copy")
```

## Configuration

### Environment Variables

Create a `.env` file:

```bash
# OpenAI (default)
OPENAI_API_KEY=your-api-key
OPENAI_MODEL=gpt-4o

# Or Azure OpenAI
AZURE_OPENAI_API_KEY=your-azure-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/...

# Optional: Custom storage path
MEMEX_STORAGE_PATH=/path/to/your/data
```

### MemexConfig

```python
from memex import Memory, MemexConfig

config = MemexConfig(
    storage_path="./data",           # Base directory for databases
    llm_provider="openai",           # "openai", "azure", "ollama", "anthropic"
    llm_model="gpt-4o",              # Model name
    auto_extract=True,               # Auto-extract knowledge (default: True)
)

memory = Memory(collection="user:alice", config=config)
```

## Storage Structure

Collections are stored as subdirectories. The `:` separator in collection names is converted to directory separators for cross-platform compatibility:

```
./memex_data/
├── user/
│   ├── alice/
│   │   └── memory.db          # collection="user:alice"
│   └── bob/
│       └── memory.db          # collection="user:bob"
├── team/
│   └── engineering/
│       └── memory.db          # collection="team:engineering"
└── project/
    └── x/
        └── memory.db          # collection="project:x"
```

### Custom Storage Path

```python
from memex import Memory, MemexConfig

config = MemexConfig(storage_path="/var/data/memories")

# Stored at: /var/data/memories/user/alice/memory.db
memory = Memory(collection="user:alice", config=config)
print(memory.db_path)  # Shows actual path
```

## API Reference

### Memory Class

Single collection memory interface.

#### Constructor

```python
Memory(
    collection: str,                    # Collection name (e.g., "user:alice")
    config: MemexConfig | None = None,  # Configuration
)
```

#### Methods (Sync)

| Method | Description |
|--------|-------------|
| `add(text, speaker?, timestamp?, tags?, metadata?)` | Add a memory |
| `add_batch(items)` | Add multiple memories |
| `query(question)` | Query with natural language |
| `search(query, limit=10)` | Search by keyword |
| `stats()` | Get memory statistics |
| `export(path)` | Export to JSON file |
| `clear()` | Delete all memories |

#### Methods (Async)

All sync methods have async versions with `_async` suffix:

```python
await memory.add_async("content")
answer = await memory.query_async("question")
results = await memory.search_async("keyword")
```

#### Properties

| Property | Description |
|----------|-------------|
| `collection` | Collection name |
| `db_path` | Database file path |
| `is_initialized` | Whether memory is initialized |

### MemoryPool Class

Aggregate multiple collections for unified querying.

#### Constructor

```python
MemoryPool(
    collections: list[str],                    # List of collection names
    default_collection: str | None = None,     # Default for add()
    config: MemexConfig | None = None,         # Configuration
)
```

#### Methods (Sync)

| Method | Description |
|--------|-------------|
| `add(text, collections?, ...)` | Add to specified collections |
| `add_batch(items, collections?)` | Add multiple memories |
| `query(question, collections?)` | Query across collections |
| `search(query, collections?, limit=10)` | Search across collections |
| `stats(collections?)` | Get statistics |
| `get_memory(collection)` | Get Memory instance |

#### Properties

| Property | Description |
|----------|-------------|
| `collections` | List of collection names |
| `default_collection` | Default collection name |

### MemoryManager Class

Manage collections (list, delete, rename, etc.).

#### Constructor

```python
MemoryManager(
    config: MemexConfig | None = None,  # Configuration
)
```

#### Methods

| Method | Description |
|--------|-------------|
| `list_collections()` | List all collection names |
| `exists(collection)` | Check if collection exists |
| `delete(collection)` | Delete collection and data |
| `rename(old, new)` | Rename a collection |
| `copy(source, dest)` | Copy a collection |
| `info(collection)` | Get collection info (size, path, etc.) |

### MemexConfig Class

```python
MemexConfig(
    storage_path: str = "./memex_data",  # Base directory
    llm_provider: str = "openai",        # LLM provider
    llm_model: str | None = None,        # Model name (default from env)
    llm_api_key: str | None = None,      # API key (default from env)
    llm_endpoint: str | None = None,     # Custom endpoint
    auto_extract: bool = True,           # Auto-extract knowledge
    db_name: str = "memory.db",          # Database filename
)
```

## Examples

### Adding Memories

```python
# Simple add
memory.add("The meeting is at 3pm")

# With metadata
memory.add(
    "Alice presented the Q4 results",
    speaker="Meeting Notes",
    timestamp="2025-01-26T15:00:00z",
    tags=["meeting", "quarterly"],
)

# Batch add
memory.add_batch([
    {"text": "Item 1", "speaker": "User"},
    {"text": "Item 2", "speaker": "Assistant"},
])
```

### Querying

```python
# Natural language query
answer = memory.query("What time is the meeting?")
# Returns a natural language answer based on stored memories
```

### Searching

```python
# Search by keyword
results = memory.search("meeting", limit=5)

for item in results:
    print(f"ID: {item.id}")
    print(f"Text: {item.text}")
    print(f"Speaker: {item.speaker}")
    print(f"Timestamp: {item.timestamp}")
```

### Statistics

```python
stats = memory.stats()
# {
#     "collection": "user:alice",
#     "total_memories": 42,
#     "entities_extracted": 128,
#     "db_path": "/var/data/memories/user/alice/memory.db",
# }
```

### Chat Application

```python
from memex import Memory

class ChatBot:
    def __init__(self, user_id: str):
        self.memory = Memory(collection=f"user:{user_id}")

    def chat(self, user_message: str) -> str:
        # Store user message
        self.memory.add(user_message, speaker="user")

        # Check memory for context
        context = self.memory.query(f"What do I know about: {user_message}")

        # Generate response (using your LLM)
        response = generate_response(user_message, context)

        # Store assistant response
        self.memory.add(response, speaker="assistant")

        return response
```

### FastAPI Integration

```python
from fastapi import FastAPI
from memex import Memory, MemexConfig

app = FastAPI()
config = MemexConfig(storage_path="/var/data/app_memories")

@app.post("/memory/{collection}")
async def add_memory(collection: str, text: str):
    memory = Memory(collection=collection, config=config)
    result = await memory.add_async(text)
    return {"added": result.messages_added, "db_path": memory.db_path}

@app.get("/memory/{collection}/query")
async def query_memory(collection: str, question: str):
    memory = Memory(collection=collection, config=config)
    answer = await memory.query_async(question)
    return {"answer": answer}
```

### Multi-Collection Query

```python
from memex import MemoryPool, MemexConfig

config = MemexConfig(storage_path="./data")

# User belongs to personal and team collections
pool = MemoryPool(
    collections=["user:alice", "team:engineering"],
    default_collection="user:alice",
    config=config,
)

# Add personal note
pool.add("My TODO: review PR #123")

# Add team knowledge
pool.add("Team uses PostgreSQL for production", collections=["team:engineering"])

# Query across both personal and team knowledge
answer = pool.query("What database does the team use?")
```
