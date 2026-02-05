# Environment Variables

Load configuration from environment variables using `MomexConfig.from_env()`:

```python
from momex import MomexConfig

config = MomexConfig.from_env()
```

## LLM Configuration

Required for knowledge extraction and query answering:

| Variable | Description |
|----------|-------------|
| `MOMEX_LLM_PROVIDER` | LLM provider: `openai`, `azure`, `anthropic`, `deepseek`, `qwen` |
| `MOMEX_LLM_MODEL` | Model name (e.g., `gpt-4o`, `claude-sonnet-4-20250514`) |
| `MOMEX_LLM_API_KEY` | API key for the LLM provider |
| `MOMEX_LLM_API_BASE` | Base URL (required for Azure) |
| `MOMEX_LLM_TEMPERATURE` | Temperature for responses (default: `0.0`) |

Example:
```bash
export MOMEX_LLM_PROVIDER=openai
export MOMEX_LLM_MODEL=gpt-4o
export MOMEX_LLM_API_KEY=sk-xxx
```

## Embedding Configuration

Optional. If not set, embeddings are auto-inferred from LLM config (works for OpenAI/Azure).

| Variable | Description |
|----------|-------------|
| `MOMEX_EMBEDDING_PROVIDER` | Embedding provider: `openai`, `azure` |
| `MOMEX_EMBEDDING_MODEL` | Model name (default: `text-embedding-3-small`) |
| `MOMEX_EMBEDDING_API_KEY` | API key (defaults to LLM key if same provider) |
| `MOMEX_EMBEDDING_API_BASE` | Base URL for embedding API |
| `MOMEX_EMBEDDING_DIMENSIONS` | Optional embedding dimension override |

Example (using Anthropic LLM with OpenAI embeddings):
```bash
# LLM
export MOMEX_LLM_PROVIDER=anthropic
export MOMEX_LLM_MODEL=claude-sonnet-4-20250514
export MOMEX_LLM_API_KEY=sk-ant-xxx

# Embedding (required because Anthropic doesn't support embeddings)
export MOMEX_EMBEDDING_PROVIDER=openai
export MOMEX_EMBEDDING_API_KEY=sk-xxx
```

## Storage Configuration

| Variable | Description |
|----------|-------------|
| `MOMEX_STORAGE_BACKEND` | Storage backend: `sqlite` (default), `postgres` |
| `MOMEX_STORAGE_PATH` | SQLite storage directory (default: `./momex_data`) |
| `MOMEX_STORAGE_POSTGRES_URL` | PostgreSQL connection URL |
| `MOMEX_STORAGE_POSTGRES_SCHEMA` | PostgreSQL schema name for collection isolation |

Example (SQLite):
```bash
export MOMEX_STORAGE_PATH=./my_data
```

Example (PostgreSQL):
```bash
export MOMEX_STORAGE_BACKEND=postgres
export MOMEX_STORAGE_POSTGRES_URL=postgresql://user:pass@localhost:5432/momex
```

## Complete Examples

### OpenAI (simplest)
```bash
export MOMEX_LLM_PROVIDER=openai
export MOMEX_LLM_MODEL=gpt-4o
export MOMEX_LLM_API_KEY=sk-xxx
# Embedding auto-inferred from LLM config
```

### Azure OpenAI
```bash
export MOMEX_LLM_PROVIDER=azure
export MOMEX_LLM_MODEL=gpt-4o
export MOMEX_LLM_API_KEY=xxx
export MOMEX_LLM_API_BASE=https://xxx.openai.azure.com
# Embedding auto-inferred from LLM config
```

### Anthropic + OpenAI Embedding
```bash
export MOMEX_LLM_PROVIDER=anthropic
export MOMEX_LLM_MODEL=claude-sonnet-4-20250514
export MOMEX_LLM_API_KEY=sk-ant-xxx

export MOMEX_EMBEDDING_PROVIDER=openai
export MOMEX_EMBEDDING_API_KEY=sk-xxx
```

### DeepSeek + OpenAI Embedding
```bash
export MOMEX_LLM_PROVIDER=deepseek
export MOMEX_LLM_MODEL=deepseek-chat
export MOMEX_LLM_API_KEY=xxx

export MOMEX_EMBEDDING_PROVIDER=openai
export MOMEX_EMBEDDING_API_KEY=sk-xxx
```

## Loading from .env File

It is recommended to put your environment variables in a file named `.env`:

```bash
# .env
MOMEX_LLM_PROVIDER=openai
MOMEX_LLM_MODEL=gpt-4o
MOMEX_LLM_API_KEY=sk-xxx
```

Momex automatically loads `.env` files from the current or parent directory.

## TypeAgent Environment Variables

For advanced use cases, TypeAgent also supports these environment variables:

- `OPENAI_API_KEY` - Used by embedding model if MOMEX_EMBEDDING_API_KEY not set
- `AZURE_OPENAI_API_KEY` - Used by Azure embedding model
- `OPENAI_MODEL` - Legacy TypeChat model setting

Momex environment variables take precedence over TypeAgent variables.
