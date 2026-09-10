# Getting Started with Momex

Momex is a Python package and API for structured, attributed memory. Import it
directly into your application; your application owns the agent and reply model.
No protocol adapter, separate service or agent framework is required.

## 1. Install and configure

Use Python 3.12 or later, preferably in a virtual environment:

```powershell
python -m pip install momex
```

Supply `MOMEX_LLM_API_KEY` securely through your environment or secret manager.
For an OpenAI-backed setup, set these non-secret options:

```powershell
$env:MOMEX_LLM_PROVIDER = "openai"
$env:MOMEX_LLM_MODEL = "gpt-4o"
$env:MOMEX_STORAGE_PATH = ".\momex_data"
```

`MomexConfig.from_env()` reads `MOMEX_*` variables and also loads a local/parent
`.env` without overriding exported values. Never commit credentials or `.env`
files. OpenAI/Azure embeddings can reuse compatible LLM credentials; other LLM
providers need separate embedding configuration.

You can instead use `MomexConfig.from_yaml("momex.yaml")` with a non-secret file:

```yaml
llm:
  provider: openai
  model: gpt-4o
storage:
  backend: sqlite
  path: .\momex_data
```

Omitted YAML keys are read from `MOMEX_LLM_API_KEY` and, when separately
configured, `MOMEX_EMBEDDING_API_KEY`. See the
[configuration guide](momex-usage.md#configuration) for other providers and
optional PostgreSQL settings.

**Data handling:** extraction, hybrid search, embeddings and model replies can
send input to configured providers. `infer=False` skips fact extraction, not
embedding API calls. Use only data you are authorized to share.

## 2. Store and retrieve cited evidence

Save this as `memory_demo.py` and run `python memory_demo.py`:

```python
import asyncio

from momex import Memory, MomexConfig, format_context


async def main():
    config = MomexConfig.from_env()
    async with Memory(collection="user:demo", config=config) as memory:
        added = await memory.add(
            [
                {
                    "role": "user",
                    "speaker": "Demo user",
                    "content": "I prefer Python for small automation projects.",
                    "source_id": "demo-preference-1",
                }
            ]
        )
        print("New memory IDs:", added.memory_ids)

        items = await memory.search(
            "Which language do I prefer?", limit=5, neighbors=1
        )
        context = format_context(items, token_budget=512)
        print(context.text)
        print("Citations:", context.citations)


asyncio.run(main())
```

Repeating the same source ID with the same payload skips duplicate ingestion.
Reusing it with different content or metadata raises an error. Keep IDs stable
when retrying imported messages.

The SDK returns evidence, not an answer. Results carry `collection`, `source_id`,
`memory_id` (for extracted knowledge), `sources` and `status`. Default searches
exclude expired, superseded and unconfirmed content. Historical `as_of` queries
and explicit visibility flags keep that evidence labeled. `format_context`
bounds the complete evidence text, including headers and citation labels.

User assertions are eligible for extraction by default. Assistant/tool turns
are stored as **unconfirmed context** unless reviewed with `confirmed=True` or
explicitly trusted using `write_policy="all"`. Generated replies must not
silently become confirmed user facts.

## 3. Use the API from your own agent

The usual flow is `memory.search()` -> `format_context()` -> your reply model ->
`memory.add()`. Your application controls prompts, model clients and resource
lifetime. Assistant writeback remains unconfirmed under the default `user`
policy.

The [SDK agent example](agent-example.md) provides a small reusable Python
function for this flow. It accepts your own async reply callback, adds no
service or packaged command, and can skip writeback for read-only turns.

## Further reading

- [Momex API](momex.md) and [usage/configuration](momex-usage.md)
- [SDK agent example](agent-example.md)
- [TypeAgent tutorial (upstream APIs)](typeagent-getting-started.md) and
  [TypeAgent high-level API](high-level-api.md), retained for legacy users
