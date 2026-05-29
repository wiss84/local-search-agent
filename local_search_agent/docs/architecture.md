# Architecture Overview

## System Architecture

```
Local Documents → Ingestion Pipeline → Meilisearch (BM25)
                                             ↓
                               FastAPI File Server (/text, /docs)
                                             ↓
                                   LangGraph Agent Loop
                                             ↓
                               Answer with Source Citations

Background: APScheduler → Incremental Re-ingestion (delta only)
Monitoring:  IndexMonitor → Freshness tracking per workspace
UI:          pywebview + FastAPI + SSE → Desktop Dashboard
```

## Core Components

### 1. Meilisearch Manager (`core/meilisearch_manager.py`)

Manages the Meilisearch binary lifecycle.

- Downloads the correct platform binary (Windows/macOS/Linux) on first use
- Caches in user cache directory, versioned to prevent corruption
- File-locked download (safe for concurrent processes)
- Starts Meilisearch as a subprocess, waits for health check
- Registers `atexit` cleanup so Meilisearch stops when Python exits
- Logs to `<cache_dir>/logs/meilisearch.stdout.log`

### 2. Ingestion Pipeline (`ingestion/`)

Transforms raw documents into clean, indexed, searchable content.

**Pipeline stages:**
1. **Discovery** — walks `document_dirs`, yields supported file paths (skips hidden files/dirs)
2. **Delta check** — skips files whose `modified_at` hasn't changed since last index
3. **Parsing** — routes to the correct parser by extension; produces Markdown text
4. **Cleaning** — 6-step pipeline: control chars → Unicode normalize → watermarks → page numbers → broken words → whitespace
5. **Chunking** — splits documents exceeding `CHUNK_MIN_CHARS`; table-row chunking for CSV/Markdown tables, sliding-window with overlap for prose
6. **Semantic enrichment** (opt-in) — ConceptCompiler + StructuralParser at ingest time
7. **Indexing** — batches DocumentNodes into Meilisearch; polls task completion
8. **Registration** — records each document in WorkspaceManager SQLite

### 3. Search Index (`search/`)

Meilisearch running as a local binary, providing fast deterministic BM25 search.

- One index per workspace (true isolation)
- Searchable attributes: `title`, `text`, `concepts`, `synonyms`
- Filterable attributes: `file_type`, `folder_path`, `modified_at`, `workspace`
- Snippet extraction with context windows

### 4. File Server (`server/`)

FastAPI application serving documents via HTTP for agent access and human citation.

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Liveness check |
| `GET /health/indexes` | Index health summary |
| `GET /workspaces` | List registered workspaces |
| `GET /workspaces/{name}/docs` | List documents in workspace |
| `GET /workspaces/{name}/history` | Sync history log |
| `GET /text/{doc_id}` | Pre-cleaned Markdown text (agent-facing) |
| `GET /docs/{doc_id}` | Raw original file (human citation link) |

### 5. Agent Loop (`agent/`)

LangGraph ReAct-style agent that acts as a researcher over your local intranet.

```
START → call_llm → route → call_tools → call_llm → ... → END
```

**Tools:**
- `search_local_index` — BM25 search against Meilisearch
- `fetch_local_url` — retrieves full document text from file server
- `get_related_docs` — finds related documents via link graph (opt-in)

**Features:**
- Multi-provider: Google, Ollama, OpenAI, Anthropic
- `temperature=0` for deterministic responses
- Rate limit handling + retry logic
- Max iteration guard
- Both `query()` (blocking) and `stream()` (SSE) modes

### 6. Workspace Management (`workspace/`)

- `WorkspaceManager` — SQLite-backed registry of workspace names → document directories; delta-check logic
- `MetadataDB` — tracks sync jobs, history, doc counts, error counts

### 7. Scheduler (`scheduler/`)

APScheduler 3.x background jobs for incremental re-ingestion.

- One interval job per workspace (`coalesce=True`, `max_instances=1`)
- Sync state written to MetadataDB before and after every run
- Progress callbacks feed the UI's live ingest status bar

### 8. Desktop UI (`ui/`)

pywebview + Jinja2 templates + FastAPI backend + SSE streaming.

- `dashboard.py` — starts the FastAPI backend, opens the native OS webview window
- `api_routes.py` — all `/api/ui/*` routes; sessions, chat, ingest, scheduler, workspaces
- `store.py` — SQLite persistence for chat sessions, messages, token counts, UI config
- `_JSBridge` — exposes `pick_folder()` and `open_url()` to JavaScript via pywebview's JS API
- SSE stream delivers live tool events (`tool_start`, `tool_end`, `text_chunk`, `thinking`, `done`) to the frontend

### 9. Key Manager (`core/key_manager.py`)

Stores and resolves LLM provider API keys outside the project.

- Keys saved to `keys.json` in the OS user config directory (`platformdirs.user_config_dir`)
- Resolution order: explicit `api_key` argument → `keys.json` → environment variable
- CLI: `local-search config set-key / list-keys / delete-key`
- UI: **Set API Keys** button in the top bar
- Values are masked when read back (first 6 + `***` + last 4)
- Ollama always resolves to `None` — no key required

### 10. Semantic Layer (`semantic/`)

Optional features that enhance BM25 search without vectors.

- **ConceptCompiler** — LLM-driven concept/synonym extraction at ingest (one call per document's chunk\part)
- **StructuralParser** — pure regex extraction of headings, definitions, references, key-values
- **QueryExpander** — expands user query at search time using concepts from the index or LLM
- **LinkGraph** — SQLite cross-document relationship store; same_topic links built at ingest
- **SemanticEnricher** — orchestrates A + B + link graph

## Key Design Decisions

**No embeddings** — BM25 is deterministic and auditable. Semantic search is via structured metadata (concepts, synonyms), no vector math.

**Document = URL** — Every document has a stable `doc_id = sha256(abs_path)[:16]` served at `/text/{doc_id}` (agent) and `/docs/{doc_id}` (human). Stable, URL-safe, collision-resistant.

**Delta ingestion** — Files re-indexed only when `modified_at` changes. A 10,000-doc corpus with 50 changed files re-indexes only the 50.

**One index per workspace** — Workspace isolation is at the Meilisearch index level, not by filter. Cross-workspace search is not supported by design (Future Feature).

**All timestamps use `datetime.now().astimezone()`** — Local time with UTC offset embedded.

