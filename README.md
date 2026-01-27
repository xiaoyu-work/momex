# Python package 'typeagent'

### This is an experimental prototype

Working toward a shared understanding of the MVP for structured RAG.

### This is sample code

This is an in-progress project aiming at a Pythonic translation of
[TypeAgent KnowPro](https://github.com/microsoft/TypeAgent/tree/main/ts/packages/knowPro)
and a few related packages from TypeScript to Python.

### Warning

This library will send its input to an LLM hosted by a third party.
Don't use it to index confidential information.

## Quick Start with Memex

Memex is a simplified, multi-tenant API wrapper for TypeAgent's Structured RAG.

```python
from memex import Memory

# Create memory for a user (auto multi-tenant isolation)
memory = Memory(user_id="user_123")

# Add memories (sync API, no async/await needed)
memory.add("Alice said the project deadline is next Friday")
memory.add("Bob is responsible for the backend API")

# Query with natural language
answer = memory.query("Who is responsible for the API?")
print(answer)  # "Bob is responsible for the backend API"

# Search by keyword
results = memory.search("Alice")
```

See [Memex Documentation](docs/memex.md) for full details.

## Core TypeAgent API

For more control, use the core TypeAgent API directly:

```python
from typeagent import create_conversation
from typeagent.transcripts.transcript import TranscriptMessage, TranscriptMessageMeta

conv = await create_conversation("memory.db", TranscriptMessage)
msg = TranscriptMessage(
    text_chunks=["Content here"],
    metadata=TranscriptMessageMeta(speaker="user"),
)
await conv.add_messages_with_indexing([msg])
answer = await conv.query("Your question?")
```

### Documentation

- Found in the [docs directory](docs/README.md)
- [Memex API (simplified)](docs/memex.md)
- Quick install: `pip install typeagent`
- Download the [PyBay '25 PowerPoint slides](https://github.com/microsoft/typeagent-py/raw/refs/heads/main/docs/StructuredRagPyBay25.pptx)
- Download the [PyBay '25 slides as PDF](https://github.com/microsoft/typeagent-py/raw/refs/heads/main/docs/StructuredRagPyBay25.pdf)
- Watch the [PyBay '25 video](https://youtu.be/-klESD7iB-s)

## Trademarks

This project may contain trademarks or logos for projects, products, or services.
Authorized use of Microsoft trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project
must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
