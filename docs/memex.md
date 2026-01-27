# Memex - Simplified Multi-Tenant API

Memex is a high-level wrapper around TypeAgent's Structured RAG, providing:

- **Simplified API** - No need to manage `TranscriptMessage` or async/await
- **Multi-tenant support** - Automatic data isolation per user/org
- **Sync & Async APIs** - Use whichever fits your application
- **Auto-configuration** - Automatic `.env` loading and path management

## Installation

```bash
pip install typeagent
```

## Quick Start

```python
from memex import Memory

# Create memory instance
memory = Memory(user_id="user_123")

# Add memories
memory.add("Alice is the project manager")
memory.add("The deadline is January 30th")

# Query with natural language
answer = memory.query("Who is the project manager?")
print(answer)  # "Alice is the project manager"
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

memory = Memory(user_id="user_123", config=config)
```

## Multi-Tenant Support

Memex automatically isolates data by tenant:

```python
# Data stored at: ./memex_data/acme/alice/memory.db
alice = Memory(user_id="alice", org_id="acme")

# Data stored at: ./memex_data/acme/bob/memory.db
bob = Memory(user_id="bob", org_id="acme")

# Data stored at: ./memex_data/globex/charlie/memory.db
charlie = Memory(user_id="charlie", org_id="globex")
```

### Tenant Identifiers

| Parameter | Description |
|-----------|-------------|
| `user_id` | User identifier |
| `agent_id` | Agent/bot identifier |
| `org_id` | Organization identifier |

Storage path: `{storage_path}/{org_id}/{user_id}/{agent_id}/memory.db`

### Custom Database Path

```python
# Override automatic path generation
memory = Memory(db_path="/custom/path/to/memory.db")
```

## API Reference

### Memory Class

#### Constructor

```python
Memory(
    user_id: str | None = None,
    agent_id: str | None = None,
    org_id: str | None = None,
    config: MemexConfig | None = None,
    db_path: str | None = None,
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
#     "total_memories": 42,
#     "entities_extracted": 128,
#     "db_path": "./memex_data/user_123/memory.db",
#     "user_id": "user_123",
#     "agent_id": None,
#     "org_id": None,
# }
```

### Export & Import

```python
# Export to JSON
memory.export("backup.json")

# Output format:
# {
#     "user_id": "user_123",
#     "agent_id": null,
#     "org_id": null,
#     "memories": [
#         {"id": "0", "text": "...", "speaker": "...", "timestamp": "..."},
#         ...
#     ]
# }
```

### Clear

```python
# Delete all memories for this tenant
memory.clear()
```

## Examples

### Chat Application

```python
from memex import Memory

class ChatBot:
    def __init__(self, user_id: str):
        self.memory = Memory(user_id=user_id)

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
from memex import Memory

app = FastAPI()

@app.post("/memory/{user_id}")
async def add_memory(user_id: str, text: str):
    memory = Memory(user_id=user_id)
    result = await memory.add_async(text)
    return {"added": result.messages_added}

@app.get("/memory/{user_id}/query")
async def query_memory(user_id: str, question: str):
    memory = Memory(user_id=user_id)
    answer = await memory.query_async(question)
    return {"answer": answer}
```

### Multi-Agent System

```python
from memex import Memory, MemexConfig

config = MemexConfig(storage_path="./agent_memories")

# Each agent has isolated memory
research_agent = Memory(user_id="user_1", agent_id="researcher", config=config)
writer_agent = Memory(user_id="user_1", agent_id="writer", config=config)

# Research agent stores findings
research_agent.add("Found 3 relevant papers on RAG systems")

# Writer agent has separate memory
writer_agent.add("Draft outline completed")

# Query specific agent's memory
research_agent.query("What papers did I find?")
```

## Comparison: Memex vs Core TypeAgent API

| Feature | Memex | Core TypeAgent |
|---------|-------|----------------|
| Async required | No (sync by default) | Yes |
| Multi-tenant | Built-in | Manual |
| Message creation | Simplified | Manual TranscriptMessage |
| Auto dotenv | Yes | No |
| Auto path management | Yes | No |
| Full control | Limited | Full |

**Use Memex when:** You want simplicity and multi-tenant support.

**Use Core API when:** You need full control over message types and settings.

## Structured RAG vs Traditional RAG

Memex uses TypeAgent's Structured RAG under the hood:

| Aspect | Traditional RAG | Structured RAG (Memex) |
|--------|-----------------|------------------------|
| Storage | Vector embeddings | Entities, relations, topics |
| Query | Similarity search | Semantic parsing |
| Result | Similar text chunks | Precise answers |
| "Who did X?" | Returns text containing "who" | Returns the actual person |
| Time queries | Weak | Strong (time indexing) |

## Troubleshooting

### "No module named 'memex'"

Make sure you installed with `pip install typeagent` and the package is in your Python path.

### API Key errors

Ensure your `.env` file is in the current directory or parent, or set environment variables:

```bash
export OPENAI_API_KEY=your-key
export OPENAI_MODEL=gpt-4o
```

### Database locked

Each `Memory` instance should be used by one process. For multi-process scenarios, use separate database paths or implement connection pooling.
