# Fork Changes

I've been keeping an eye on this repo for a while. I'm also personally interested in personal assistant agents, and I've been trying to find best practices for memory. Structured RAG is a great design, but since this is an experimental project, the feature set isn't complete yet. So I forked the original repo and added more features, aiming to make it work for more general use cases and projects.

This fork adds **Momex** - a simplified, collection-based API wrapper for TypeAgent's Structured RAG.

## What's New

- `src/momex/` - High-level memory API package
- Hierarchical collections with `:` separator (`company:engineering:alice`)
- Prefix queries across multiple collections
- Auto fact extraction and deduplication via LLM
- Importance scoring (health info ranks higher than casual info)
- PostgreSQL backend support (via pgvector)
- Soft delete and restore
- YAML configuration support
- Export to JSON


## Quick Start

```python
from momex import Memory, query

# Add memories with hierarchical identity
alice = Memory(collection="company:engineering:alice")
alice.add("I like Python")

bob = Memory(collection="company:engineering:bob")
bob.add("I prefer Java")

# Query with prefix - searches all matching collections
answer = query("company:engineering", "What languages do people like?")
# Searches both alice and bob

answer = query("company", "Who likes Python?")
# Searches entire company
```

See [docs/momex.md](docs/momex.md) for full documentation.

---

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

### Documentation

- Found in the [docs directory](docs/README.md)
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
