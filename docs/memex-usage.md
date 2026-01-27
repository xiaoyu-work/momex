# Memex Usage

## Setup

```bash
pip install -e .
export OPENAI_API_KEY=your-key
```

## Core Concepts

### Collection

A **collection** is a named storage space for memories. Each collection has its own database file.

- Use collections to separate memories by user, team, or purpose
- Collection names support hierarchical structure with `:` separator
- Examples: `"alice"`, `"user:alice"`, `"company:engineering:alice"`

### Hierarchical Collections

The `:` separator creates a hierarchy that enables prefix queries:

```
company:engineering:alice  →  ./memex_data/company/engineering/alice/memory.db
company:engineering:bob    →  ./memex_data/company/engineering/bob/memory.db
company:marketing:charlie  →  ./memex_data/company/marketing/charlie/memory.db
```

Query behavior:
- `query("company:engineering:alice", ...)` → searches only alice
- `query("company:engineering", ...)` → searches alice + bob
- `query("company", ...)` → searches alice + bob + charlie

## Basic Usage

### Create Memory

```python
from memex import Memory, MemexConfig

# Simple - uses default config
memory = Memory(collection="user:alice")

# With custom config
config = MemexConfig(storage_path="./my_data")
memory = Memory(collection="user:alice", config=config)
```

### Add and Query

```python
memory = Memory(collection="user:alice")

# Add memories manually
memory.add("I love Python programming")
memory.add("Project deadline is Friday", speaker="Manager")

# Query
answer = memory.query("What programming language does the user like?")
```

### Add Conversation

Automatically store important information from conversation history:

```python
memory.add_conversation([
    {"role": "user", "content": "My name is Alice, I'm a Python developer"},
    {"role": "assistant", "content": "Nice to meet you!"},
    {"role": "user", "content": "I'm working on a FastAPI project"},
])

# Later, query the memories
answer = memory.query("What is the user's name?")  # "Alice"
answer = memory.query("What project is the user working on?")  # "FastAPI project"
```

### Query Across Collections

```python
from memex import Memory, query

# Create memories for different users
alice = Memory(collection="company:engineering:alice")
alice.add("I like Python")

bob = Memory(collection="company:engineering:bob")
bob.add("I prefer Java")

# Query single collection
answer = alice.query("What language?")

# Query by prefix - searches multiple collections
answer = query("company:engineering", "What languages do people use?")
answer = query("company", "Who works here?")
```

### Manage Collections

```python
from memex import MemoryManager

manager = MemoryManager()

# List collections
all_collections = manager.list_collections()
eng_only = manager.list_collections(prefix="company:engineering")

# Other operations
manager.exists("company:engineering:alice")
manager.delete("company:engineering:alice")
manager.rename("user:old", "user:new")
```

## Configuration

### MemexConfig

```python
from memex import MemexConfig

config = MemexConfig(
    storage_path="./memex_data",      # Where to store databases
    llm_provider="openai",            # "openai", "azure", "anthropic"
    llm_model="gpt-4o",               # Model for fact extraction
    similarity_threshold=0.5,         # For memory deduplication (0.0-1.0)
    fact_types=[...],                 # Custom fact types (see below)
)
```

### YAML Configuration

```yaml
# memex_config.yaml
storage_path: ./memex_data
llm_provider: openai
similarity_threshold: 0.5

fact_types:
  - name: Personal Preferences
    description: Likes, dislikes, preferences for food, products, activities
  - name: Professional Details
    description: Job titles, projects, skills, career goals
  - name: Technical Stack
    description: Programming languages, frameworks, tools
```

```python
config = MemexConfig.from_yaml("memex_config.yaml")
memory = Memory(collection="user:alice", config=config)
```

### Custom Fact Types

Control what types of information to extract from conversations:

```python
from memex import FactType, MemexConfig

config = MemexConfig(
    fact_types=[
        FactType(
            name="Technical Skills",
            description="Programming languages, frameworks, and tools the user knows"
        ),
        FactType(
            name="Project Information",
            description="Current projects, deadlines, and team members"
        ),
    ],
)
```

## API Reference

### Memory

| Method | Description |
|--------|-------------|
| `add(text, speaker?, timestamp?)` | Add a single memory |
| `add_batch(items)` | Add multiple memories |
| `add_conversation(messages)` | Store important info from conversation |
| `query(question)` | Query this collection with natural language |
| `search(keyword, limit=10)` | Search by keyword |
| `delete(memory_id)` | Soft delete a memory |
| `restore(memory_id)` | Restore a deleted memory |
| `list_deleted()` | List all deleted memories |
| `stats()` | Get memory statistics |
| `export(path)` | Export to JSON file |
| `clear()` | Delete all memories in this collection |

### Prefix Query Functions

| Function | Description |
|----------|-------------|
| `query(prefix, question)` | Query all collections matching prefix |
| `search(prefix, keyword)` | Search all collections matching prefix |
| `stats(prefix)` | Get combined stats for matching collections |

### ConversationResult

Returned by `add_conversation()`:

```python
result = memory.add_conversation(messages)

if result.success:
    print("Memories stored")
else:
    print(f"Error: {result.error}")
```
