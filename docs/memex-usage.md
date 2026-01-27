# Memex Usage

## Setup

```bash
pip install -e .
export OPENAI_API_KEY=your-key
```

## Basic Usage

```python
from memex import Memory

memory = Memory(collection="user:alice")

# Add memory
memory.add("I love Python")

# Query
answer = memory.query("What does the user like?")
```

## Extract Facts from Conversation

```python
result = memory.add_conversation([
    {"role": "user", "content": "My name is Alice, I'm a Python developer"},
    {"role": "assistant", "content": "Nice to meet you!"},
])

print(result.facts_extracted)  # ['Name is Alice', 'Is a Python developer']
```

## Hierarchical Query

```python
from memex import Memory, query

# Different collections
alice = Memory(collection="company:engineering:alice")
bob = Memory(collection="company:engineering:bob")

alice.add("I like Python")
bob.add("I like Java")

# Query by prefix
query("company:engineering", "What languages?")  # searches alice + bob
query("company", "What languages?")              # searches all
```

## Configuration

### YAML

```yaml
# memex_config.yaml
storage_path: ./memex_data
similarity_threshold: 0.5

fact_types:
  - name: Technical Skills
    description: Programming languages, frameworks
  - name: Project Info
    description: Current projects, deadlines
```

```python
config = MemexConfig.from_yaml("memex_config.yaml")
memory = Memory(collection="user:alice", config=config)
```

### Code

```python
from memex import FactType, MemexConfig

config = MemexConfig(
    storage_path="./data",
    fact_types=[
        FactType(name="Skills", description="Technical skills"),
    ],
)
```

## API

### Memory

| Method | Description |
|--------|-------------|
| `add(text)` | Add memory |
| `add_conversation(messages)` | Extract facts and add |
| `query(question)` | Query memories |
| `search(keyword)` | Search by keyword |
| `clear()` | Delete all |

### MemoryManager

```python
from memex import MemoryManager

manager = MemoryManager()
manager.list_collections()
manager.list_collections(prefix="company:engineering")
manager.delete("user:old")
```

### Prefix Query

```python
from memex import query, search

query("prefix", "question")
search("prefix", "keyword")
```
