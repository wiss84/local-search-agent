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
)
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

### Health Monitoring

```python
summary = framework.get_index_health() -> IndexHealthSummary
```
Return freshness status across all workspaces.

```python
status = framework.get_scheduler_status() -> dict
```
Return scheduler state: `running`, `scheduled_jobs`, next run times.

### Semantic Settings

```python
framework.set_semantic_settings(
    enable_semantic: bool,
    enable_query_expansion: bool,
    enable_link_graph: bool,
)
```
Update all three semantic feature flags at runtime. Persists to `settings.json` in the user config directory so the settings are shared across CLI, UI, and future Python API sessions. If `enable_link_graph` changes, the agent tool list is rebuilt on the next query.

```python
settings = framework.get_semantic_settings() -> dict[str, bool]
```
Return the current semantic feature flags as a dict with keys `enable_semantic`, `enable_query_expansion`, `enable_link_graph`.

**Example:**

```python
# Check current state
print(framework.get_semantic_settings())
# {'enable_semantic': False, 'enable_query_expansion': False, 'enable_link_graph': False}

# Enable concept extraction and query expansion
framework.set_semantic_settings(
    enable_semantic=True,
    enable_query_expansion=True,
    enable_link_graph=False,
)

# Re-ingest to apply semantic indexing to existing documents
framework.ingest_and_index(force=True)
```

> **Note:** `enable_semantic` and `enable_link_graph` only affect documents ingested *after* the flag is enabled. Existing documents in the index do not gain concept or link metadata retroactively. Use `force=True` on the next ingest to reprocess them.

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
| `enable_semantic` | `bool` | `False` | *(Experimental)* Run ConceptCompiler + StructuralParser at ingest |
| `enable_query_expansion` | `bool` | `False` | *(Experimental)* Expand queries with synonyms at search time |
| `enable_link_graph` | `bool` | `False` | *(Experimental)* Build cross-document topic links at ingest |
| `semantic_model` | `str \| None` | `None` | *(Experimental)* Override model for concept compilation |
| `enable_access_control` | `bool` | `False` | *(Experimental)* Enforce Windows/LDAP access control |
| `ldap_server` | `str \| None` | `None` | *(Experimental)* LDAP server URL |

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
| `doc_id` | `str` | `sha256(abs_path)[:16]` — stable, URL-safe ID |
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
