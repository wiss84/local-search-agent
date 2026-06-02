# Configuration Guide

All framework configuration lives in a single `SearchAgentConfig` dataclass. This page covers every option, how they interact, and common patterns.

## How Configuration Works

`SearchAgentConfig` uses Python dataclasses with sensible defaults. You only need to set what differs from the defaults.

```python
from local_search_agent import SearchAgentConfig

# Minimal — uses Google, reads key from saved keys
config = SearchAgentConfig(
    document_dirs=["C:/my_docs"],
    workspace_name="finance",
    provider="google",
)

# Full explicit config
config = SearchAgentConfig(
    document_dirs=["C:/my_docs", "C:/more_docs"],
    workspace_name="finance",
    meilisearch_url="http://localhost:7700",
    meili_master_key="local_search_master_key",
    provider="google",
    api_key="YOUR_KEY",
    model_name="gemma-4-31b-it",
    host="127.0.0.1",
    port=8000,
    top_k=10,
    max_iterations=50,
    db_path="local_search_agent.db",
)
```

## API Key Resolution

The framework resolves API keys in this priority order:

1. `api_key` argument passed directly to `SearchAgentConfig`
2. Keys saved via `local-search config set-key` (stored in `keys.json` in your OS user config dir)
3. Environment variable (`GOOGLE_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`)

If none of these are found, the framework starts without error but raises a clear `ValueError` the moment a query is attempted.

Ollama always uses `None` — no key required.

## All Parameters

### Document Sources

| Parameter | Default | Description |
|-----------|---------|-------------|
| `document_dirs` | `[]` | List of directories to scan for documents. Supports multiple directories per workspace. |
| `workspace_name` | `"default"` | Logical name for this document collection. Also used as the Meilisearch index name unless `index_name` overrides it. |

### Meilisearch

| Parameter | Default | Description |
|-----------|---------|-------------|
| `meilisearch_url` | `http://localhost:7700` | URL of the Meilisearch instance. The binary manager starts Meilisearch at this address automatically. |
| `meili_master_key` | `local_search_master_key` | Meilisearch authentication key. The default is fine for local use. Change this for network-exposed deployments. |
| `index_name` | `None` | Override the Meilisearch index name. Defaults to `workspace_name`. Useful when you need the index name to differ from the workspace name. |

### LLM Provider

| Parameter | Default | Description |
|-----------|---------|-------------|
| `provider` | `"google"` | One of: `google`, `ollama`, `openai`, `anthropic` |
| `api_key` | `None` | Provider API key. See resolution order above. |
| `model_name` | `"gemma-4-31b-it"` | Model to use. Provider-specific. See model reference below. |

### File Server

| Parameter | Default | Description |
|-----------|---------|-------------|
| `host` | `127.0.0.1` | Bind address for the FastAPI file server. Use `0.0.0.0` to expose on the network. |
| `port` | `8000` | Port for the file server. |

### Agent Behaviour

| Parameter | Default | Description |
|-----------|---------|-------------|
| `top_k` | `5` | Number of search results the agent retrieves per `search_local_index` call. Higher values improve recall at the cost of more tokens. |
| `max_iterations` | `50` | Maximum number of search/fetch/reason cycles before the agent stops. Prevents runaway loops. |
| `max_retries` | `5` | HTTP retry count for agent tool calls (fetch, health checks). |

### Persistence

| Parameter | Default | Description |
|-----------|---------|-------------|
| `db_path` | user config dir | Path to the SQLite database that stores workspace registrations, document metadata, sync history, and chat sessions. Defaults to `local_search_agent.db` in your OS user config directory — the same location as `keys.json`, `models.json`, and `settings.json`. This means the database survives `pip install --upgrade` and is independent of your working directory. |

### Semantic Search

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_semantic` | `False` | Run ConceptCompiler and StructuralParser during ingestion. Adds one LLM call per document. |
| `enable_query_expansion` | `False` | Expand user queries with synonyms at search time. |
| `semantic_model` | `None` | Override the model used for concept extraction at ingest time. Defaults to the main `model_name`. |

The `semantic_model` field only controls which model runs concept extraction — the main agent still uses `model_name` for answering queries. This lets you use a cheaper or faster model for ingest without affecting query quality.

The semantic provider is configured separately via `local-search config set-semantic --provider` (CLI), the UI Semantic Search pane, or by setting `semantic_provider` in `settings.json`. It is not a field on `SearchAgentConfig` directly — the provider override is read from the shared `settings.json` at ingest time.

### Access Control (Experimental)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_access_control` | `False` | Enforce Windows/LDAP access control on file server endpoints. |
| `ldap_server` | `None` | LDAP server URL, e.g. `ldap://company.local`. Required if `enable_access_control=True`. |

---

## Model Reference

### Google (via Google AI Studio)

| Model | Notes |
|-------|-------|
| `gemma-4-31b-it` | Default. Dense, strong reasoning. |
| `gemma-4-26b-a4b-it` | MoE variant. Faster, same free tier. |
| `gemini-3.1-flash-lite` | Lightweight Gemini model. |

Free tier: ~15 requests/minute. Get a key at https://aistudio.google.com.

### Ollama (local)

| Model | Notes |
|-------|-------|
| `mistral` | Good general-purpose model, fast. |
| `llama3.2` | Meta's latest. |
| `qwen2.5` | Strong reasoning, multilingual. |
| `gemma4:e2b` | Google's smallest Gemma 4 — very fast. |

No API key. Requires Ollama installed and running. See [Installation Guide](installation.md#using-ollama-fully-local-no-api-key).

### OpenAI

| Model | Notes |
|-------|-------|
| `gpt-4o-mini` | Good balance of cost and quality. |
| `gpt-4o` | Best quality. |

Requires `OPENAI_API_KEY` or saved key.

### Anthropic

| Model | Notes |
|-------|-------|
| `claude-sonnet-4-20250514` | Strong reasoning, long context. |
| `claude-haiku-4-5-20251001` | Fast and cheap. |

Requires `ANTHROPIC_API_KEY` or saved key.

---

## Configuration Patterns

### Multiple workspaces, shared config

```python
from local_search_agent import SearchAgentFramework, SearchAgentConfig

config = SearchAgentConfig(
    provider="google",
    model_name="gemma-4-31b-it",
    workspace_name="default",  # placeholder
)

framework = SearchAgentFramework(config)
framework.create_workspace("finance", "/data/finance")
framework.create_workspace("hr",      "/data/hr")
framework.create_workspace("legal",   "/data/legal")

framework.ingest_workspace("finance")
framework.ingest_workspace("hr")
framework.ingest_workspace("legal")

framework.start_file_server()
framework.start_incremental_scheduler(interval_minutes=15)

response = framework.query("What is the parental leave policy?", workspace="hr")
```

### Fully local with Ollama

```python
config = SearchAgentConfig(
    document_dirs=["/data/docs"],
    workspace_name="local",
    provider="ollama",
    model_name="mistral",
    # No api_key needed
)
```

### Custom database location

```python
config = SearchAgentConfig(
    document_dirs=["/data/docs"],
    workspace_name="production",
    provider="google",
    db_path="/var/lib/local-search-agent/production.db",
)
```

The default location on each platform is:

| Platform | Default path |
|----------|--------------|
| Windows | `C:\Users\<name>\AppData\Roaming\local-search-agent\local_search_agent.db` |
| macOS | `~/Library/Application Support/local-search-agent/local_search_agent.db` |
| Linux | `~/.config/local-search-agent/local_search_agent.db` |

You can also override it via the `LSA_DB_PATH` environment variable or the `--db` CLI flag, without changing your Python code.

### Tune for large document sets

```python
config = SearchAgentConfig(
    document_dirs=["/data/large_corpus"],
    workspace_name="corpus",
    provider="google",
    top_k=10,           # more results per search
    max_iterations=30,  # more reasoning cycles
)
```

### Tune for low memory

```python
# Also adjust in constants.py:
# PDF_PAGES_PER_BATCH = 10
# DOCX_CHAR_SPLIT_THRESHOLD = 3000

config = SearchAgentConfig(
    document_dirs=["/data/docs"],
    workspace_name="lean",
    provider="ollama",
    model_name="gemma4:e2b",
    top_k=3,
    max_iterations=15,
    enable_semantic=False,
)
```

---

## Validation

Call `config.validate()` to check for problems early:

```python
config = SearchAgentConfig(...)
config.validate()   # raises ValueError for bad config, warns for missing key
```

The framework calls `validate()` internally at relevant points. A missing API key is only a warning at construction time — it becomes an error when a query is actually sent.
