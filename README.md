# Fork Changes

This fork adds **Memex** - a simplified, multi-tenant API wrapper for TypeAgent's Structured RAG.

## What's New

- `src/memex/` - New high-level API package
- `docs/memex.md` - Memex documentation
- `tests/test_memex/` - Unit tests
- `examples/memex/` - Usage examples

## Memex Features

| Feature | Description |
|---------|-------------|
| **Simplified API** | No `TranscriptMessage` or `async/await` needed |
| **Multi-tenant** | Auto data isolation via `user_id`, `org_id`, `agent_id` |
| **Sync API** | Synchronous methods by default |
| **Auto-config** | Auto `.env` loading and path management |

## Quick Start

```python
from memex import Memory

memory = Memory(user_id="user_123")
memory.add("Alice is the project manager")
answer = memory.query("Who is the project manager?")
```

## Multi-Tenant Example

```python
# Data isolated per user/org
alice = Memory(user_id="alice", org_id="acme")
bob = Memory(user_id="bob", org_id="acme")
# Stored at: ./memex_data/acme/alice/memory.db
#            ./memex_data/acme/bob/memory.db
```

See [docs/memex.md](docs/memex.md) for full documentation.

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
