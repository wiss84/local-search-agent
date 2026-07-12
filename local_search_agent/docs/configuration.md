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

### Multi-Tenant RBAC

| Parameter | Default | Description |
|-----------|---------|-------------|
| `identity_provider` | `None` | An `IdentityProvider` instance (`HeaderIdentityProvider`, `APIKeyIdentityProvider`, `JWTIdentityProvider`, or your own). `None` (the default) is today's single-user mode, completely unchanged — `AuthorizationMiddleware` is never added to the app. Setting this to any `IdentityProvider` opts into three-tier (`superadmin`/`admin`/`member`) enforcement on every protected route — `admin`/`member` are per-workspace grants, `superadmin` is unconditional. See [Role-Based Access Control](role_based_access_control.md) for the full guide, including which provider fits which deployment and a CLI walkthrough for bootstrapping keys and grants. This field is deliberately excluded from `config.to_dict()` (it may hold non-serializable state, e.g. a live `AuthDB` connection) — reconstruct it explicitly if you round-trip a config through `to_dict()`/`from_dict()`. |

### Watch Mode

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_watch_mode` | `False` | Use filesystem events (`watchdog`) instead of polling to trigger re-ingestion. See [Watch Mode](#watch-mode-1) below. Mutually exclusive with the polling scheduler. |
| `enrich_on_watch` | `True` | Whether watch-triggered re-ingests also run semantic enrichment (only relevant if `enable_semantic=True`). Set `False` to skip the LLM call on watch-triggered syncs for speed; you'll need a later manual or scheduled sync to backfill semantic fields for those files. |

### Re-ranking

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_reranking` | `True` | Re-rank Meilisearch BM25 results with a local cross-encoder (`flashrank`) for better relevance. Runs fully offline; the model downloads once (~17MB) on first use and is cached. |
| `rerank_candidate_multiplier` | `4` | Fetch `top_k x` this many candidates from Meilisearch before re-ranking down to `top_k`. Higher values improve quality at the cost of slightly more compute. |

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
# Set ingestion constants via the framework (persists to advanced_settings.json)
config = SearchAgentConfig(
    document_dirs=["/data/docs"],
    workspace_name="lean",
    provider="ollama",
    model_name="gemma4:e2b",
    top_k=3,
    max_iterations=15,
    enable_semantic=False,
)

framework = SearchAgentFramework(config)
framework.set_advanced_settings({
    "PDF_PAGES_PER_BATCH": 10,
    "DOCX_CHAR_SPLIT_THRESHOLD": 30000,
    "CHUNK_TARGET_CHARS": 8000,
})
framework.ingest_and_index()
```

Or equivalently via the CLI:

```bash
local-search config set-advanced --key PDF_PAGES_PER_BATCH --value 10
local-search config set-advanced --key DOCX_CHAR_SPLIT_THRESHOLD --value 30000
local-search config set-advanced --key CHUNK_TARGET_CHARS --value 8000
```

Or via the UI: **Settings → Advanced** tab.

---

## Advanced Settings

Ingestion and search constants can be overridden without touching source code. Overrides are stored in `advanced_settings.json` in your user config directory (same location as `keys.json` and `settings.json`), and take effect on the next ingest run. Constants not explicitly overridden continue to use their compiled-in defaults from `constants.py`.

### Where to set them

**UI** — **Settings → Advanced** tab. Each field shows the current default in grey. Modified fields are highlighted in amber. Click **Save** to persist; **Reset to Defaults** to clear all overrides.

**CLI:**

```bash
# Override one constant
local-search config set-advanced --key PDF_PAGES_PER_BATCH --value 10

# Reset all overrides back to compiled-in defaults
local-search config set-advanced --reset

# Verify effective values (overrides show [OVERRIDE])
local-search config show
```

**Python API:**

```python
# Read effective values
print(framework.get_advanced_settings())

# Write overrides
framework.set_advanced_settings({
    "PDF_PAGES_PER_BATCH": 10,
    "CHUNK_TARGET_CHARS": 8000,
})

# Reset
framework.set_advanced_settings({})
```

### Reference table

| Key | Category | Default | Description |
|-----|----------|---------|-------------|
| `CHUNK_MIN_CHARS` | Chunking | see constants.py | Minimum chars before chunking is applied |
| `CHUNK_TARGET_CHARS` | Chunking | — | Target chars per chunk |
| `CHUNK_MAX_CHARS` | Chunking | — | Hard cap chars per chunk |
| `CHUNK_OVERLAP_CHARS` | Chunking | — | Overlap between consecutive chunks |
| `TABLE_ROWS_PER_CHUNK` | Table/CSV | — | Rows per chunk for tabular data |
| `PDF_PAGES_PER_BATCH` | PDF/DOCX | — | Pages per processing batch |
| `PDF_SPLIT_THRESHOLD` | PDF/DOCX | — | Page count above which a PDF is split into batches |
| `PDF_FALLBACK_PAGES_PER_BATCH` | PDF/DOCX | — | Batch size used when the primary batch fails |
| `DOCX_CHAR_SPLIT_THRESHOLD` | PDF/DOCX | — | DOCX char count above which section-splitting is used |
| `TESSERACT_FALLBACK_MIN_CHARS` | OCR | — | Min chars from PyMuPDF before Tesseract is tried |
| `DEFAULT_TOP_K` | Search | — | Default results returned per search call |
| `DEFAULT_MAX_ITERATIONS` | Agent | — | Default max agent reasoning iterations |
| `SNIPPET_CONTEXT_CHARS` | Search | — | Chars of context around a match in result snippets |

Run `local-search config show` to see the live default values for your installed version.

> **Note:** Advanced settings affect ingestion behaviour. After changing chunking or batching constants, run `local-search ingest --force` (or use **Force Re-ingest** in the UI) to reprocess your documents with the new settings.

---

## Validation

Call `config.validate()` to check for problems early:

```python
config = SearchAgentConfig(...)
config.validate()   # raises ValueError for bad config, warns for missing key
```

The framework calls `validate()` internally at relevant points. A missing API key is only a warning at construction time — it becomes an error when a query is actually sent.

---

## Watch Mode

Watch Mode reacts to filesystem changes (file created, modified, or deleted) within seconds, instead of waiting for the next polling interval. It is the recommended way to keep an index fresh and replaces the older polling-based `IncrementalSyncScheduler` for most use cases.

### How it works

A `watchdog` observer watches all of a workspace's `document_dirs`. When a change is detected, a short debounce window (~2.5s) collapses bursts of OS-level events — a single file save, or a folder copy with many files — into one re-ingestion run, rather than firing a sync per individual event.

Each watch-triggered sync reuses the exact same delta logic and `IngestionPipeline` as a manual or scheduled sync. The only behavioural difference is *when* it fires, and whether semantic enrichment runs (`enrich_on_watch`).

### Enabling Watch Mode

**Python API:**

```python
config = SearchAgentConfig(
    document_dirs=["/data/finance"],
    workspace_name="finance",
    provider="google",
    enable_watch_mode=True,
    enrich_on_watch=True,   # default; set False to skip LLM calls on watch-triggered syncs
)
framework = SearchAgentFramework(config)
framework.start_watch_mode()
```

**CLI:**

```bash
local-search watch start --workspace finance --dirs "C:\Shares\FinanceDocs"
local-search watch status
local-search watch trigger --workspace finance   # force immediate sync, bypassing debounce
```

**UI:** Sidebar **Sync** button → select **Watch Mode** in the dropdown → **Save**. A quick on/off toggle sits next to the Sync button for turning automatic sync on or off without opening the modal.

### Watch Mode vs. the polling Scheduler

| | Watch Mode | Polling Scheduler *(deprecated)* |
|---|---|---|
| Trigger | Filesystem events (`watchdog`) | Fixed interval (default 15 min) |
| Reaction time | Seconds | Up to the full interval |
| Detects changes while app is closed | No — only reacts while running | No — only reacts while running |
| Mechanism | `WorkspaceWatcher` | `IncrementalSyncScheduler` (APScheduler) |

The two are mutually exclusive per workspace — enabling one via the UI Sync modal or sidebar toggle automatically stops the other for that workspace. The polling scheduler remains available for backward compatibility but is deprecated; new code should use Watch Mode.

### `enrich_on_watch`

If a workspace has `enable_semantic=True`, every new or changed document should ideally get the same concept/synonym enrichment as the rest of the index — otherwise query expansion only works for some documents. `enrich_on_watch` defaults to `True` for this reason. Set it to `False` only if you want watch-triggered syncs to skip the LLM call entirely (e.g. a slow/rate-limited free-tier provider with frequent file churn), accepting that those documents won't be enriched until a later manual or scheduled full sync.

```python
framework.set_watch_mode_settings(enable_watch_mode=True, enrich_on_watch=False)
settings = framework.get_watch_mode_settings()
# {'enable_watch_mode': True, 'enrich_on_watch': False}
```
