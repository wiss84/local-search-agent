# Python API Reference

## Import

```python
from local_search_agent import (
    SearchAgentFramework,
    SearchAgentConfig,
    DocumentNode,
    IngestionPipeline,
    IngestStats,
    MeilisearchClient,
    QueryBuilder,
    LocalSearchAgent,
    IncrementalSyncScheduler,
    IndexMonitor,
    IndexHealthSummary,
    WorkspaceManager,
    MetadataDB,
    Reranker,
    # Agent tool integration
    LocalSearchTool,
    ToolResult,
)
```

---

## LocalSearchTool

Use this when you want to integrate Local Search Agent into another AI agent or application. Instead of querying the framework directly, you wrap an indexed workspace as a tool that any external agent can call with a plain query string and get a clean answer back.

There are two steps: **index your documents once**, then **create the tool** and hand it to your agent.

### Step 1 — Index your documents and start the file server

Do this once, or whenever your documents change. It is separate from the tool itself.

```python
from local_search_agent import SearchAgentFramework, SearchAgentConfig

config = SearchAgentConfig(
    document_dirs=["C:\\Users\\username\\Desktop\\skills_documents"],
    workspace_name="skills",
    provider="google",
    api_key=None,               # reads from GOOGLE_API_KEY env var if omitted
    model_name="gemini-2.0-flash-lite",
)

framework = SearchAgentFramework(config)
framework.ingest_and_index()
framework.start_file_server()  # must be running before the tool is used
```

### Step 2 — Create the tool

Once the workspace is indexed, create a `LocalSearchTool` from the same config and pass it to your agent.

```python
from local_search_agent import LocalSearchTool

skill_tool = LocalSearchTool(config)

# Optional: pass return_raw=True to bypass LLM summarisation and return
# the full document text verbatim. Use this when the calling agent should
# reason over the raw content itself (e.g. skill files, memory files).
skill_tool = LocalSearchTool(config, return_raw=True)
```

### ToolResult

`skill_tool.run(query)` returns a `ToolResult`:

| Field | Type | Description |
|-------|------|-------------|
| `answer` | `str` | Answer synthesised by the internal agent |
| `sources` | `list[str]` | Titles of the source documents used |

`str(result)` returns just the answer, so the tool works as a drop-in wherever a string is expected.

### Usage with LangChain

```python
from langchain_core.tools import tool

@tool
def skill_search(query: str) -> str:
    """Search the skills knowledge base for coding patterns and techniques."""
    return skill_tool.run(query).answer

# Bind skill_search to your LangChain agent the same way as any other tool
```

### Usage without LangChain

```python
result = skill_tool.run("how do I handle rate limits in Python?")
print(result.answer)
print(result.sources)   # ["rate_limit_handler", "retry_patterns"]
```

### Multiple tools

Create one tool per directory. Each uses its own `workspace_name` which maps to a separate Meilisearch index. Each workspace needs to be indexed independently before the tool is used.

The file server is workspace-agnostic — it serves any document by `doc_id` regardless of which workspace it belongs to. This means you only need to call `start_file_server()` once, from any one of the framework instances.

```python
from local_search_agent import SearchAgentFramework, SearchAgentConfig, LocalSearchTool

skill_config      = SearchAgentConfig(document_dirs=["C:/skills"],     workspace_name="skills",     ...)
memory_config     = SearchAgentConfig(document_dirs=["C:/memory"],     workspace_name="memory",     ...)
finance_config    = SearchAgentConfig(document_dirs=["C:/finance"],    workspace_name="finance",    ...)
accounting_config = SearchAgentConfig(document_dirs=["C:/accounting"], workspace_name="accounting", ...)

# Index each workspace independently
SearchAgentFramework(skill_config).ingest_and_index()
SearchAgentFramework(memory_config).ingest_and_index()
SearchAgentFramework(finance_config).ingest_and_index()
SearchAgentFramework(accounting_config).ingest_and_index()

# Start the file server once — workspace does not matter here
SearchAgentFramework(skill_config).start_file_server()

# Create the tools
skill_tool      = LocalSearchTool(skill_config,      return_raw=True)
memory_tool     = LocalSearchTool(memory_config,     return_raw=True)
finance_tool    = LocalSearchTool(finance_config)
accounting_tool = LocalSearchTool(accounting_config)
```

---

## SearchAgentFramework

The main entry point. Wraps all components behind a single object.

```python
framework = SearchAgentFramework(config: SearchAgentConfig)
```

### File Server

```python
framework.start_file_server(port=None, block=False)
```
Start the FastAPI document server. Runs in a background thread unless `block=True`.

```python
framework.stop_file_server()
```
Signal the server to shut down.

### Ingestion

```python
stats = framework.ingest_and_index(force=False) -> IngestStats
```
Parse all documents in `config.document_dirs` and index into Meilisearch. Uses delta logic by default — only re-indexes files whose `modified_at` timestamp has changed. Pass `force=True` to re-index everything regardless.

```python
stats = framework.ingest_workspace(workspace_name, force=False) -> IngestStats
```
Run ingestion for a specific named workspace without changing `config.workspace_name`.

```python
framework.wipe_and_reingest(workspace_name=None)
```
Delete the Meilisearch index and all SQLite document records for a workspace, then force a full re-ingest from scratch.

### Agent Query

```python
response = framework.query(question, top_k=None, workspace=None) -> dict
```
Ask the agent a question. Returns a dict:

| Key | Type | Description |
|-----|------|-------------|
| `answer` | `str` | The agent's answer |
| `sources` | `list[dict]` | Source documents used |
| `iterations_used` | `int` | Number of search/fetch iterations |
| `truncated` | `bool` | True if max iterations was reached |
| `token_input` | `int` | Input token count |
| `token_output` | `int` | Output token count |

### Workspace Management

```python
framework.create_workspace(name, document_dir)
```
Register a new workspace pointing to a document directory.

```python
workspaces = framework.list_workspaces() -> list[dict]
```
Return all registered workspaces. Each dict has `name` and `document_dir`.

```python
framework.delete_workspace(name, wipe_index=False)
```
Remove a workspace registration. Pass `wipe_index=True` to also delete the Meilisearch index.

### Scheduler

⚠️ **DEPRECATED** — Use watch mode instead. `start_incremental_scheduler()` is kept for backward compatibility but should not be used for new code. Watch mode is event-driven and reacts instantly to file changes without polling delays.

```python
framework.start_incremental_scheduler(interval_minutes=15)
```
Start the APScheduler background job. Registers all existing workspaces automatically.

```python
framework.stop_incremental_scheduler()
```
Gracefully stop the scheduler.

```python
framework.add_workspace_to_scheduler(workspace_name, interval_minutes=None)
```
Add a specific workspace to a running scheduler.

```python
framework.trigger_sync_now(workspace_name=None)
```
Force an immediate sync outside the normal schedule.

### Watch Mode

Filesystem event-driven re-ingestion. Reacts to file changes instantly without polling delays. **Recommended over the polling scheduler.**

```python
framework.start_watch_mode()
```
Start watch mode in a background thread. All workspaces registered with `enable_watch_mode=True` in the config (or via `set_watch_mode_settings()`) are watched for file changes. File events are debounced by 2 seconds — rapid bursts of changes are collapsed into a single sync. Semantic enrichment is controlled by `enrich_on_watch` in the config or by calling `set_watch_mode_settings()`.

```python
framework.stop_watch_mode()
```
Gracefully stop watch mode.

```python
framework.add_workspace_to_watch_mode(workspace_name)
```
Add a specific workspace to watch mode. The workspace must already be registered. Uses the `enrich_on_watch` setting from `set_watch_mode_settings()` or the config.

```python
status = framework.get_watch_mode_status() -> dict
```
Return watch mode state. Example:

```python
{
    "running": True,
    "watched_directories": {
        "finance": 2,      # 2 directories being watched
        "legal": 1,        # 1 directory
    }
}
```

```python
framework.set_watch_mode_settings(enable_watch_mode: bool, enrich_on_watch: bool)
```
Configure watch mode behavior at runtime. Persists to the config so settings survive restarts.

- `enable_watch_mode`: Enable/disable watch mode globally
- `enrich_on_watch`: Whether watch-triggered syncs also run semantic enrichment (if enabled in the config). Set `False` to skip the LLM call for speed; you can always run a later full re-ingest with `force=True` to backfill semantic fields.

```python
settings = framework.get_watch_mode_settings() -> dict
```
Return current watch mode settings: `{"enable_watch_mode": bool, "enrich_on_watch": bool}`.

**Example:**

```python
# Start watching all registered workspaces
framework.set_watch_mode_settings(enable_watch_mode=True, enrich_on_watch=True)
framework.start_watch_mode()

# Later, add a new workspace to watch mode
framework.add_workspace_to_watch_mode("hr")

# Check status
print(framework.get_watch_mode_status())  # {'running': True, 'watched_directories': {'finance': 1, 'hr': 1}}

# Stop watching
framework.stop_watch_mode()
```

### Health Monitoring

```python
summary = framework.get_index_health() -> IndexHealthSummary
```
Return freshness status across all workspaces.

```python
status = framework.get_scheduler_status() -> dict
```
Return scheduler state: `running`, `scheduled_jobs`, next run times.

### Advanced Settings

```python
settings = framework.get_advanced_settings() -> dict
```
Return the effective value of every ingestion/search constant, merging any user overrides on top of the compiled-in defaults from `constants.py`. The returned dict always contains every key — overridden keys reflect the user-set value, unoverridden keys reflect the compiled-in default.

```python
effective = framework.set_advanced_settings(overrides: dict) -> dict
```
Persist ingestion/search constant overrides to `advanced_settings.json` in the user config directory. Overrides take effect on the **next** ingest run. Pass an empty dict `{}` to reset everything back to compiled-in defaults. Returns the effective constants after applying the overrides.

Valid keys: `CHUNK_MIN_CHARS`, `CHUNK_TARGET_CHARS`, `CHUNK_MAX_CHARS`, `CHUNK_OVERLAP_CHARS`, `TABLE_ROWS_PER_CHUNK`, `PDF_PAGES_PER_BATCH`, `PDF_SPLIT_THRESHOLD`, `PDF_FALLBACK_PAGES_PER_BATCH`, `DOCX_CHAR_SPLIT_THRESHOLD`, `TESSERACT_FALLBACK_MIN_CHARS`, `DEFAULT_TOP_K`, `DEFAULT_MAX_ITERATIONS`, `SNIPPET_CONTEXT_CHARS`.

Unknown keys and values that cannot be coerced to the expected numeric type are silently ignored.

**Example:**

```python
# Inspect current effective values
print(framework.get_advanced_settings())
# {'CHUNK_TARGET_CHARS': 10000, 'PDF_PAGES_PER_BATCH': 20, ...}

# Tune for a low-RAM machine
framework.set_advanced_settings({
    "PDF_PAGES_PER_BATCH": 10,
    "CHUNK_TARGET_CHARS": 8000,
})
framework.ingest_and_index(force=True)

# Reset all overrides back to compiled-in defaults
framework.set_advanced_settings({})
```

Advanced settings are shared across the CLI (`local-search config set-advanced`), the UI **Settings → Advanced** tab, and the Python API. They persist across restarts and `pip install --upgrade`.

### Semantic Settings

```python
framework.set_semantic_settings(
    enable_semantic: bool,
    enable_query_expansion: bool,
)
```
Update semantic feature flags at runtime. Persists to `settings.json` in the user config directory so the settings are shared across CLI, UI, and future Python API sessions.

```python
settings = framework.get_semantic_settings() -> dict
```
Return the current semantic feature flags as a dict with keys `enable_semantic`, `enable_query_expansion`.

**Example:**

```python
# Check current state
print(framework.get_semantic_settings())
# {'enable_semantic': False, 'enable_query_expansion': False}

# Enable concept extraction and query expansion
framework.set_semantic_settings(
    enable_semantic=True,
    enable_query_expansion=True,
)

# Re-ingest to apply semantic indexing to existing documents
framework.ingest_and_index(force=True)
```

To use a different model or provider for concept extraction, set it via the CLI or UI before ingesting:

```bash
# Use a cheaper model for concept extraction only
local-search config set-semantic --provider google --model gemma-4-26b-a4b-it
```

Or pass `semantic_model` directly in `SearchAgentConfig`:

```python
config = SearchAgentConfig(
    provider="google",
    model_name="gemma-4-31b-it",     # used for agent queries
    enable_semantic=True,
    semantic_model="gemma-4-26b-a4b-it",  # used for concept extraction only
)
```

> **Note:** `enable_semantic` only affects documents ingested *after* the flag is enabled. Use `force=True` on the next ingest to reprocess existing documents.

---

## SearchAgentConfig

All runtime configuration in one dataclass.

```python
from local_search_agent import SearchAgentConfig

config = SearchAgentConfig(
    document_dirs=["C:/my_docs"],
    workspace_name="finance",
    provider="google",
    api_key="YOUR_KEY",        # optional — reads from saved keys or env var if omitted
    model_name="gemma-4-31b-it",
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `document_dirs` | `list[str]` | `[]` | Directories to ingest |
| `workspace_name` | `str` | `"default"` | Workspace name and Meilisearch index name |
| `meilisearch_url` | `str` | `http://localhost:7700` | Meilisearch URL |
| `meili_master_key` | `str` | `local_search_master_key` | Meilisearch master key |
| `index_name` | `str \| None` | `None` | Override index name (defaults to `workspace_name`) |
| `provider` | `str` | `"google"` | LLM provider: `google`, `ollama`, `openai`, `anthropic` |
| `api_key` | `str \| None` | `None` | API key. Resolution order: this arg → saved keys → env var |
| `model_name` | `str` | `"gemma-4-31b-it"` | Model name. For Ollama: `"mistral"`, `"llama3.2"`, etc. |
| `host` | `str` | `127.0.0.1` | File server bind address |
| `port` | `int` | `8000` | File server port |
| `top_k` | `int` | `5` | Search results per query |
| `max_iterations` | `int` | `50` | Agent loop iteration cap |
| `max_retries` | `int` | `5` | HTTP retry count for agent tools |
| `db_path` | `str` | `local_search_agent.db` | SQLite database path |
| `enable_semantic` | `bool` | `False` | Run ConceptCompiler + StructuralParser at ingest |
| `enable_query_expansion` | `bool` | `False` | Expand queries with synonyms at search time |
| `semantic_model` | `str \| None` | `None` | Override model for concept extraction only. Defaults to `model_name`. The semantic provider is set separately via CLI/UI. |
| `enable_access_control` | `bool` | `False` | *(Experimental)* Enforce Windows/LDAP access control |
| `ldap_server` | `str \| None` | `None` | *(Experimental)* LDAP server URL |
| `enable_watch_mode` | `bool` | `False` | Use filesystem events instead of polling to trigger re-ingestion |
| `enrich_on_watch` | `bool` | `True` | Whether watch-triggered re-ingests also run semantic enrichment (if `enable_semantic=True`) |
| `enable_reranking` | `bool` | `True` | Re-rank Meilisearch BM25 results with a local cross-encoder (flashrank) for better relevance. Fully offline. |
| `rerank_candidate_multiplier` | `int` | `4` | Fetch `top_k × this many` candidates from Meilisearch before re-ranking down to `top_k`. Higher = better quality, more compute. |
| `identity_provider` | `IdentityProvider \| None` | `None` | Opt into multi-tenant RBAC. `None` is single-user mode, unchanged. See [Multi-Tenant RBAC](#multi-tenant-rbac) above and [Role-Based Access Control](role_based_access_control.md). Excluded from `to_dict()`. |

### Methods

```python
config.validate()
```
Validates the config. Raises `ValueError` for unknown providers or missing directories. Logs a warning if no API key is found (does not raise — the error surfaces at query time).

```python
config.to_dict() -> dict
```
Serialize to dict. `api_key` is excluded for safety.

```python
config = SearchAgentConfig.from_dict(data: dict)
```
Reconstruct from dict.

### Properties

```python
config.server_base_url       # "http://127.0.0.1:8000"
config.file_server_base_url  # "http://127.0.0.1:8000"
config.text_url(doc_id)      # "http://127.0.0.1:8000/text/<doc_id>"
config.docs_url(doc_id)      # "http://127.0.0.1:8000/docs/<doc_id>"
```

---

## DocumentNode

Represents one indexed document (or chunk of a document).

```python
from local_search_agent import DocumentNode

node = DocumentNode.from_file(
    source_path="/data/report.pdf",
    text="Cleaned document text...",
    workspace="finance",
)
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `doc_id` | `str` | Stable unique ID. For unchunked documents: `sha256(abs_path)[:16]`. For chunks: `sha256(abs_path:chunk:N)[:16]`. Stable across re-ingests as long as the file path and chunk index don't change. |
| `title` | `str` | Filename without extension |
| `text` | `str` | Cleaned Markdown text |
| `file_type` | `str` | Extension without dot: `pdf`, `docx`, etc. |
| `source_path` | `str` | Absolute path to the source file |
| `folder_path` | `str` | Parent directory of the source file |
| `workspace` | `str` | Workspace name |
| `modified_at` | `str` | ISO-8601 timestamp with UTC offset |
| `indexed_at` | `str` | ISO-8601 timestamp with UTC offset |
| `concepts` | `list[str]` | Semantic concepts (Experimental, populated by ConceptCompiler) |
| `synonyms` | `list[str]` | Synonyms (Experimental, populated by ConceptCompiler + StructuralParser) |

### Methods

```python
DocumentNode.make_doc_id(source_path) -> str   # static
DocumentNode.from_file(source_path, text, workspace, title=None) -> DocumentNode  # classmethod
node.to_dict() -> dict
DocumentNode.from_dict(data) -> DocumentNode   # classmethod
node.snippet(query, context_chars=300) -> str  # extract snippet around query match
```

---

## MeilisearchClient

Direct access to the Meilisearch index.

```python
from local_search_agent import MeilisearchClient

client = MeilisearchClient(
    url="http://localhost:7700",
    api_key="local_search_master_key",
    index_name="finance",
)
```

### Methods

```python
client.index_documents(nodes: list[DocumentNode])
client.search(query, top_k=5, filter_expr=None, snippet_chars=300) -> list[dict]
client.delete_document(doc_id)
client.delete_index()
client.is_healthy() -> bool
client.get_index_stats() -> dict
```

Search results are dicts with keys: `doc_id`, `title`, `text`, `snippet`, `file_type`, `folder_path`, `modified_at`, `workspace`, `score`.

---

## QueryBuilder

Construct Meilisearch filter expressions.

```python
from local_search_agent import QueryBuilder

filter_expr = QueryBuilder(
    workspace="finance",
    file_type="pdf",
    modified_after="2024-01-01",
    modified_before="2024-12-31",
).build()
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `workspace` | Filter by workspace name |
| `file_type` | Single extension or list: `"pdf"` or `["pdf", "docx"]` |
| `folder_path` | Filter by folder path |
| `modified_after` | ISO date string — only docs modified after this date |
| `modified_before` | ISO date string — only docs modified before this date |
| `raw` | Raw Meilisearch filter string (overrides all others) |

---

## Reranker

Cross-encoder re-ranking layer for improving BM25 relevance. Uses the local `flashrank` model (CPU-only) to re-score Meilisearch candidates on semantic similarity to the query.

```python
from local_search_agent import Reranker

reranker = Reranker()
```

### Why Re-ranking?

BM25 scores documents by term frequency — it has no semantic understanding. A cross-encoder re-ranker catches:
- Synonym / paraphrase mismatches (query: "fail", doc: "exception")
- BM25 over-ranking short chunks that happen to contain rare query terms
- Improved relevance ordering for the same `top_k` results

### Model & Caching

- **Default model**: `ms-marco-TinyBERT-L-2-v2` (~17MB, ~100ms per 10 queries on CPU)
- **First use**: Downloads to `<user_config_dir>/local-search-agent/models/flashrank/`
- **Subsequent uses**: Loads from disk only — no internet access needed

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | `str` | `ms-marco-TinyBERT-L-2-v2` | flashrank model name |
| `cache_dir` | `str \| None` | `<user_config_dir>/models/flashrank` | Model cache directory |
| `max_length` | `int` | `512` | Max token length per passage. Model's native limit. Shorter values = faster. |

### Methods

```python
scores = reranker.rerank(query: str, candidates: list[str], top_k: int = 5) -> list[tuple[str, float]]
```

Re-rank candidates against the query. Returns up to `top_k` results as `(candidate_text, score)` tuples, sorted by score (highest first). Scores are in range [0, 1].

**Example:**

```python
candidates = [
    "The system failed with error code 500.",
    "Database connection exception occurred.",
    "User login was successful.",
]

ranked = reranker.rerank("What error happened?", candidates, top_k=2)
# [(The system failed..., 0.92), (Database connection..., 0.87)]
```

### Integration with SearchAgentFramework

Reranking is built into the framework. Control via config:

```python
config = SearchAgentConfig(
    document_dirs=["/data/docs"],
    workspace_name="finance",
    enable_reranking=True,              # Enable re-ranking (default)
    rerank_candidate_multiplier=4,      # Fetch 4x top_k candidates, re-rank to top_k
)

framework = SearchAgentFramework(config)
```

The flow:
1. `MeilisearchClient.search()` fetches `top_k × rerank_candidate_multiplier` candidates from Meilisearch
2. `Reranker.rerank()` scores all candidates
3. Results are truncated to `top_k` and returned to the agent

---

## IngestionPipeline

Direct access to the ingestion pipeline.

```python
from local_search_agent import IngestionPipeline, SearchAgentConfig, MeilisearchClient
from local_search_agent import WorkspaceManager

config = SearchAgentConfig(document_dirs=["/data/docs"], workspace_name="docs")
wm = WorkspaceManager(db_path=config.db_path)
client = MeilisearchClient(url=config.meilisearch_url, api_key=config.meili_master_key, index_name="docs")

pipeline = IngestionPipeline(config=config, workspace_manager=wm, meili_client=client)
stats = pipeline.run(force=False)
print(stats)
```

### IngestStats

Returned by `pipeline.run()` and `framework.ingest_and_index()`.

| Field | Type | Description |
|-------|------|-------------|
| `total` | `int` | Files discovered |
| `indexed` | `int` | Files (chunks) successfully indexed |
| `skipped` | `int` | Files skipped (unchanged since last index) |
| `failed` | `int` | Files that failed to parse or index |
| `duration_s` | `float` | Total wall-clock time in seconds |
| `errors` | `list[str]` | Error messages for failed files |

`str(stats)` returns: `IngestStats(total=X, indexed=Y, skipped=Z, failed=W, duration=V.s)`

---

## WorkspaceManager

SQLite-backed registry of workspaces.

```python
from local_search_agent import WorkspaceManager

wm = WorkspaceManager(db_path="local_search_agent.db")
wm.create_workspace(name="finance", document_dir="/data/finance")
wm.list_workspaces()       # -> list[dict]
wm.get_workspace("finance") # -> dict | None
wm.delete_workspace("finance")
wm.document_needs_reindex(file_path, mtime_iso) # -> bool
```

---

## IncrementalSyncScheduler

APScheduler-backed background sync.

```python
from local_search_agent import IncrementalSyncScheduler

scheduler = IncrementalSyncScheduler(
    workspace_manager=wm,
    metadata_db=mdb,
    interval_minutes=15,
)
scheduler.start()
scheduler.add_workspace(config, interval_minutes=15)
scheduler.remove_workspace("finance")
scheduler.trigger_now("finance")
scheduler.stop(wait=True)
scheduler.get_status() -> dict
```

---

## IndexMonitor

Read-only freshness monitor.

```python
from local_search_agent import IndexMonitor

monitor = IndexMonitor(metadata_db=mdb, stale_threshold_minutes=30)
summary = monitor.get_health_summary()   # -> IndexHealthSummary
health  = monitor.get_workspace_health("finance")  # -> WorkspaceHealth | None
stale   = monitor.get_stale_workspaces() # -> list[WorkspaceHealth]
```

### IndexHealthSummary fields

`total_workspaces`, `healthy`, `stale`, `never_synced`, `error`, `running`, `total_docs`, `all_healthy`, `workspaces: list[WorkspaceHealth]`

### WorkspaceHealth fields

`workspace`, `status`, `doc_count`, `error_count`, `last_error`, `last_sync_at`, `next_sync_at`, `age_minutes`

---

## Multi-Tenant RBAC

See [Role-Based Access Control](role_based_access_control.md) for the full
conceptual guide (roles, choosing a provider, CLI walkthrough). This
section covers the Python API surface only.

### Identity and IdentityProvider

```python
from local_search_agent.auth.identity import Identity, IdentityProvider
```

`Identity` is a small dataclass: `subject: str`, `display_name: str = ""`,
`is_superadmin: bool = False`. `IdentityProvider` is a `typing.Protocol`
with one method, `resolve(request) -> Identity | None` — implement it to
write a custom provider; return `None` for anyone you can't verify.

### Built-in providers

```python
from local_search_agent.auth.header_provider import HeaderIdentityProvider
from local_search_agent.auth.api_key_provider import APIKeyIdentityProvider
from local_search_agent.auth.jwt_provider import JWTIdentityProvider
```

```python
HeaderIdentityProvider(
    header_name: str,
    trusted_proxy_ips: list[str] | None = None,
    display_name_header: str | None = None,
    superadmin_header: str | None = None,
)
```
Trusts a header set by an authenticating reverse proxy already in front
of this app. No cryptography — exactly as trustworthy as whatever sets
the header. See its module docstring for the full trust-boundary warning
before using it.

```python
APIKeyIdentityProvider(auth_db: AuthDB)
```
Issues and validates `lsa_<key_id>_<secret>` API keys (argon2-hashed at
rest) plus short-lived browser session cookies on top of them. Construct
with a shared `AuthDB` instance (`from local_search_agent.workspace.auth_db import AuthDB`).

```python
JWTIdentityProvider(
    issuer: str,
    audience: str,
    jwks_uri: str,
    algorithms: list[str] | None = None,        # default ["RS256"]; "none" is a hard error
    subject_claim: str = "sub",
    display_name_claim: str | None = "name",
    superadmin_claim: str | None = None,
    jwks_cache_ttl_seconds: int = 600,
    clock_skew_seconds: int = 60,
)
```
Validates a bearer JWT against your IdP's JWKS endpoint (Auth0, Okta,
Azure AD, Google Workspace, or any OIDC/OAuth2 issuer). Raises
`ProviderUnavailableError` (from `local_search_agent.auth.errors`) if the
JWKS endpoint itself is unreachable — a distinct failure mode from a
simply-invalid token, which resolves to `None` instead. See its module
docstring for the full algorithm-allow-list / clock-skew / JWKS-caching
rationale.

### Enabling it

```python
config.identity_provider = APIKeyIdentityProvider(AuthDB(db_path=config.db_path))
```

See [SearchAgentConfig's `identity_provider` field](#searchagentconfig)
below — `None` (the default) is single-user mode, completely unchanged.

### SearchAgentFramework methods

```python
framework.grant_workspace_access(
    workspaces: list[str],
    subject: str,
    role: str,          # "member" | "admin"
    granted_by: str,
) -> None
```
Grant `subject` a role across one or more workspaces in a single atomic
call — either every workspace gets the grant or none do.

```python
framework.revoke_workspace_access(subject: str, workspaces: list[str] | None = None) -> int
```
Revoke `subject`'s access. `workspaces=None` revokes everything for that
subject. Returns the number of grants removed.

```python
framework.list_workspace_access(subject: str | None = None, workspace: str | None = None) -> list[dict]
```
List grants, optionally filtered by subject and/or workspace.

```python
framework.get_workspace_role(subject: str, workspace: str) -> str | None
```
Return `subject`'s role in `workspace`, or `None` if they have no grant
(fail-closed — no grant means no access, not a default role).

```python
key_id, raw_key = framework.create_api_key(
    subject: str,
    created_by: str,
    display_name: str = "",
    is_superadmin: bool = False,
) -> tuple[str, str]
```
Generate a new API key (`APIKeyIdentityProvider` mode). `raw_key` is
returned exactly once — only its argon2 hash is persisted.

```python
framework.revoke_api_key(key_id: str) -> bool
```
Revoke an API key by its `key_id`. Returns `True` if an active key was
found and revoked.

```python
framework.list_api_keys(subject: str | None = None) -> list[dict]
```
List key metadata only (never the raw key or its hash), optionally
filtered by subject.

### Model/Provider Access Control

```python
framework.grant_model_access(role: str, provider: str, model_name: str, granted_by: str) -> None
framework.revoke_model_access(role: str, provider: str, model_name: str) -> bool
framework.list_model_access(role: str | None = None) -> list[dict]
```
Manage which provider+model combinations a role (`"member"` or
`"admin"`) may use for their own queries — a cost control, not a
workspace permission. A role with nothing granted has access to nothing
(fail-closed). See [Role-Based Access Control — Model / Provider Access
Control](role_based_access_control.md#model--provider-access-control)
for the full concept.

### Rate Limits & Concurrency

```python
framework.set_concurrency_limit(provider: str, limit: int, multi_tenant: bool) -> None
framework.delete_concurrency_limit(provider: str, multi_tenant: bool) -> bool
framework.get_concurrency_limits(multi_tenant: bool) -> dict[str, int]

framework.set_quota_override(
    provider: str, model_name: str, multi_tenant: bool,
    rpm: int | None = None, tpm: int | None = None, rpd: int | None = None,
) -> None
framework.delete_quota_override(provider: str, model_name: str, multi_tenant: bool) -> bool
framework.get_quota_overrides(multi_tenant: bool, provider: str | None = None) -> dict
```
Manage the max simultaneous in-flight LLM calls per provider
(concurrency) and RPM/TPM/RPD quota overrides per provider+model.
`multi_tenant` selects which of two completely independent namespaces to
read/write — pass `config.identity_provider is not None` for the natural
default matching this framework instance's own mode, or an explicit
value to manage the other namespace deliberately. Changes take effect on
the next call for that provider (no restart needed). See [Role-Based
Access Control — Rate Limits &
Concurrency](role_based_access_control.md#rate-limits--concurrency) for
the full concept, including why the two namespaces exist.

**Example:**

```python
from local_search_agent import SearchAgentFramework, SearchAgentConfig
from local_search_agent.auth.api_key_provider import APIKeyIdentityProvider
from local_search_agent.workspace.auth_db import AuthDB

config = SearchAgentConfig(workspace_name="finance", db_path="prod.db")
config.identity_provider = APIKeyIdentityProvider(AuthDB(db_path=config.db_path))
framework = SearchAgentFramework(config)

key_id, raw_key = framework.create_api_key(subject="alice@acme.com", created_by="admin@acme.com")
framework.grant_workspace_access(workspaces=["finance"], subject="alice@acme.com", role="admin", granted_by="admin@acme.com")

print(framework.get_workspace_role("alice@acme.com", "finance"))  # "admin"
```

### HTTP endpoints (multi-tenant mode only)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/ui/whoami` | GET | Identity + role introspection for frontend role-gating. Always mounted; reports `multi_tenant: false` in single-user mode. |
| `/api/auth/login` | POST | `APIKeyIdentityProvider` only — exchange a raw key for a session cookie. |
| `/api/auth/logout` | POST | `APIKeyIdentityProvider` only — clear the session cookie. |
| `/api/admin/grants` | GET / POST / DELETE | List / create / revoke workspace grants. Global admin only; granting/revoking `admin` itself requires superadmin. |
| `/api/admin/keys` | GET / POST / DELETE | List / create / revoke API keys. `APIKeyIdentityProvider` only, global admin only; creating/revoking another admin's key requires superadmin. |
| `/api/ui/models/allowed` | GET | Filtered provider/model list for the caller's current role. |
| `/api/ui/models/access` | GET / POST / DELETE | Manage the two role-level model allow-lists. Superadmin only. |
| `/api/ui/rate-limits`, `/rate-limits/concurrency`, `/rate-limits/quota` | GET / POST / DELETE | Concurrency + quota-override config for this deployment's own mode. Superadmin only. |
| `/health/ready` | GET | Readiness probe — verifies Meilisearch is reachable, distinct from the plain liveness `/health`. Not RBAC-specific but shipped alongside it; see [Architecture](architecture.md). |

---

## Complete Example

```python
from local_search_agent import SearchAgentFramework, SearchAgentConfig

config = SearchAgentConfig(
    document_dirs=["C:/company_docs/finance"],
    workspace_name="finance",
    provider="google",
    # api_key omitted — reads from saved keys set via CLI or UI
    model_name="gemma-4-31b-it",
    top_k=10,
)

framework = SearchAgentFramework(config)

# Ingest
stats = framework.ingest_and_index()
print(stats)

# Start server + scheduler
framework.start_file_server()
framework.start_incremental_scheduler(interval_minutes=15)

# Query
response = framework.query("What was the total cloud spend in Q3?")
print(response["answer"])
print(f"Sources: {len(response['sources'])}, Iterations: {response['iterations_used']}")

# Health
health = framework.get_index_health()
print(health.to_dict())
```
