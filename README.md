# Fork Changes

I've been keeping an eye on this repo for a while. I'm also personally interested in personal assistant agents, and I've been trying to find best practices for memory. Structured RAG is a great design, but since this is an experimental project, the feature set isn't complete yet. So I forked the original repo and added more features, aiming to make it work for more general use cases and projects.

This fork adds **Momex** - a high-level API wrapper for TypeAgent's Structured RAG.

## What's New

- **Agent API** - High-level chat interface with automatic memory management
- **Short-term memory** - Session-based conversation history with persistence
- **Long-term memory** - Structured RAG with entity/action/topic extraction
- Multi-tenant support with hierarchical collections (`user:xiaoyuzhang`)
- PostgreSQL backend with pgvector for production deployment

## Installation

```bash
pip install momex
```

## Quick Start

Just call `chat()` - memory is handled automatically:

```python
import asyncio
from momex import Agent, MomexConfig

async def main():
    # Configure once
    config = MomexConfig(
        provider="openai",
        model="gpt-4o",
        # api_key via MOMEX_API_KEY env var
    )

    # Create agents for different users
    xiaoyuzhang = Agent("user:xiaoyuzhang", config)
    gvanrossum = Agent("user:gvanrossum", config)

    # Chat - LLM automatically decides what to remember
    r = await xiaoyuzhang.chat("My name is Xiaoyu, I love Python")
    print(r.content)
    # Stored to long-term memory (identity + preference)

    r = await gvanrossum.chat("I'm Guido, I prefer Java")
    print(r.content)
    # Stored to long-term memory (identity info)

    # Memory persists - agent remembers across conversations
    r = await xiaoyuzhang.chat("What's my name?")
    print(r.content)  # "Your name is Xiaoyu"

    r = await xiaoyuzhang.chat("Help me write a hello world")
    # NOT stored to long-term memory (temporary request)

    # Session persists across restarts
    print(f"Session: {xiaoyuzhang.session_id}")

    await xiaoyuzhang.close()
    await gvanrossum.close()

asyncio.run(main())
```

## Two API Levels

| Level | API | Description |
|-------|-----|-------------|
| **Level 1** | `Agent` | Chat API with automatic memory (recommended) |
| **Level 2** | `Memory`, `ShortTermMemory` | Manual control for custom agents |

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
