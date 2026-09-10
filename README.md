# Fork Changes

I've been keeping an eye on this repo for a while. I'm also personally interested in personal assistant agents, and I've been trying to find best practices for memory. Structured RAG is a great design, but since this is an experimental project, the feature set isn't complete yet. So I forked the original repo and added more features, aiming to make it work for more general use cases and projects.

This fork adds **Momex** - a Python memory package and API for AI agents, built
on TypeAgent's Structured RAG. It stores and retrieves evidence; your application
owns the conversation and generates answers.

## What's New

- **Hybrid search** - Structured RAG + embedding similarity in parallel for best recall
- **Structured memory** - Entity/action/topic extraction via LLM, term-based indexing
- **Embedding-only mode** - `search_by_embedding()` for fast similarity search without LLM
- Multi-tenant support with hierarchical collections (`user:demo`)
- Attributed, cited context with token budgets and explicit historical/unconfirmed states
- Stable source/memory IDs, retry-safe ingestion, and reversible deletion/history
- PostgreSQL backend with pgvector for production deployment

## Installation

```bash
python -m pip install momex
```

## Quick Start

Set `MOMEX_LLM_API_KEY` securely in your environment, then select a model.
SQLite is the default; PostgreSQL is optional.

```powershell
$env:MOMEX_LLM_PROVIDER = "openai"
$env:MOMEX_LLM_MODEL = "gpt-4o"
$env:MOMEX_STORAGE_PATH = ".\momex_data"
```

```python
import asyncio
from momex import Memory, MomexConfig, format_context

async def main():
    config = MomexConfig.from_env()
    async with Memory(collection="user:demo", config=config) as memory:
        await memory.add("I prefer Python", source_id="demo-preference-1")
        results = await memory.search("Which language do I prefer?", neighbors=1)
        context = format_context(results, token_budget=512)
        print(context.text)

asyncio.run(main())
```

By default, user assertions can become facts; assistant/tool outputs remain
**unconfirmed context** unless explicitly trusted. Retrieval preserves source
IDs, citations and status labels rather than silently presenting history as fact.

**Agent example:** a small Python function demonstrates retrieval, bounded
context and policy-controlled writeback with your own reply function. It does
not add a server, protocol or agent framework.

See [Getting Started](docs/getting-started.md),
[SDK agent example](docs/agent-example.md), and the
[Momex API guide](docs/momex.md). Indexing, hybrid search and the agent example
may send input to your configured LLM/embedding provider; use only data you are
authorized to share.

---

# TypeAgent (upstream, low-level APIs)

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

- [TypeAgent getting started](docs/typeagent-getting-started.md) and
  [low-level API guide](docs/high-level-api.md)
- Momex already packages the `typeagent` namespace. To use the separate upstream
  distribution instead, install `typeagent`; it does not provide the `momex` API.
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
