# CLI Reference

## Command Structure

```
local-search [--db <path>] [--log-level LEVEL] <command> [subcommand] [options]
```

## Global Options

| Option | Default | Description |
|--------|---------|-------------|
| `--db` | `local_search_agent.db` | SQLite metadata database path. Also reads from `LSA_DB_PATH` env var. |
| `--log-level` | `INFO` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

---

## config

Manage saved API keys. Keys are stored in your user config directory outside the project — never in the repo.

### config set-key

```bash
local-search config set-key --provider <provider> --key <key>
```

| Option | Description |
|--------|-------------|
| `--provider` | One of: `google`, `openai`, `anthropic` |
| `--key` | Your API key |

```bash
local-search config set-key --provider google --key AIzaSy...
local-search config set-key --provider openai --key sk-...
local-search config set-key --provider anthropic --key sk-ant-...
```

Ollama does not use a key — omit it entirely.

### config list-keys

```bash
local-search config list-keys
```

Shows all saved providers with their keys masked (first 6 chars + `***` + last 4).

### config delete-key

```bash
local-search config delete-key --provider <provider>
```

Removes the saved key for a provider.

### config add-model

```bash
local-search config add-model --provider <provider> --model-name <name>
```

Adds a model name for a provider. Stored in `models.json` in your user config directory. The model will appear in the UI sidebar dropdown and is available to `local-search query --model`.

```bash
local-search config add-model --provider ollama --model-name gemma4:e2b
local-search config add-model --provider openai --model-name gpt-4o-mini
local-search config add-model --provider anthropic --model-name claude-haiku-4-5-20251001
```

### config delete-model

```bash
local-search config delete-model --provider <provider> --model-name <name>
```

Removes a model name for a provider.

### config list-models

```bash
local-search config list-models
```

Lists all saved model names per provider.

### config set-semantic

Enable or disable a semantic search feature. Settings are stored in `settings.json` in your user config directory and are shared across CLI, UI, and Python API.

```bash
local-search config set-semantic <feature> <value>
```

| Feature | Description |
|---------|-------------|
| `semantic` | ConceptCompiler + StructuralParser at ingest time |
| `query-expansion` | Expand queries with synonyms at search time |
| `link-graph` | Build cross-document topic links at ingest |

`<value>` accepts: `true`, `false`, `on`, `off`, `enable`, `disable`, `1`, `0`, `yes`, `no`

```bash
local-search config set-semantic semantic true
local-search config set-semantic query-expansion on
local-search config set-semantic link-graph false
```

### config show-semantic

```bash
local-search config show-semantic
```

Shows the current state of all three semantic feature flags.

### config show

```bash
local-search config show
```

Shows everything in one view: version, saved API keys (masked), models per provider, semantic settings, and LangSmith tracing status. Useful for debugging.

---

## setup

Download the Meilisearch binary for the current platform. Runs automatically on first use; call this explicitly to pre-download (In case of a bug).

```bash
local-search setup [--force]
```

| Option | Description |
|--------|-------------|
| `--force` | Re-download even if binary already exists |

---

## ui

Open the desktop dashboard.

```bash
local-search ui [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | `127.0.0.1` | Dashboard API server host. Also reads `LSA_HOST`. |
| `--port` | `8765` | Dashboard API server port. Also reads `LSA_PORT`. |
| `--provider` | `google` | LLM provider: `google`, `ollama`, `openai`, `anthropic`. Also reads `LSA_PROVIDER`. |
| `--model` | `gemma-4-31b-it` | Model name. Also reads `LSA_MODEL`. |
| `--meili-url` | `http://localhost:7700` | Meilisearch URL. Also reads `MEILI_URL`. |
| `--meili-key` | `local_search_master_key` | Meilisearch master key. Also reads `MEILI_MASTER_KEY`. |
| `--scheduler-interval` | `0` | Start ingestion scheduler with this interval in minutes. `0` = disabled. |
| `--headless` | off | Run API server only, no window (for debugging). |

---

## workspace

Manage workspaces (named collections of documents).

### workspace create

```bash
local-search workspace create <name> <dir>
```

Registers a new workspace pointing to a document directory. Creates sync tracking records in MetadataDB.

```bash
local-search workspace create finance "C:\Shares\FinanceDocs"
local-search workspace create hr ./hr_policies
local-search workspace create legal /mnt/legal_repository
```

### workspace list

```bash
local-search workspace list
```

Lists all registered workspaces with their document directories.

### workspace delete

```bash
local-search workspace delete <name> [--wipe]
```

Removes a workspace registration from SQLite. Does not delete your files.

| Option | Description |
|--------|-------------|
| `--wipe` | Also delete all documents from the Meilisearch index. |

```bash
local-search workspace delete old_project           # Remove registration only
local-search workspace delete old_project --wipe    # Remove registration + wipe index
```

---

## ingest

Parse and index documents into Meilisearch.

```bash
local-search ingest --workspace <name> --dirs <dir> [dir ...] [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--workspace` | `default` | Workspace to ingest into |
| `--dirs` | (required) | One or more source directories |
| `--meili-url` | `http://localhost:7700` | Meilisearch URL |
| `--meili-key` | `local_search_master_key` | Meilisearch master key |
| `--force` | off | Re-index all files, ignoring delta logic |
| `--wipe` | off | Delete the index and all DB records, then force full re-ingest |

`--force` re-indexes all files but keeps existing index data. `--wipe` deletes everything first, then starts from scratch.

```bash
local-search ingest --workspace finance --dirs "C:\Shares\FinanceDocs"
local-search ingest --workspace finance --dirs "C:\dir1" "C:\dir2"
local-search ingest --workspace finance --dirs "C:\Shares\FinanceDocs" --force
local-search ingest --workspace finance --dirs "C:\Shares\FinanceDocs" --wipe
```

---

## serve

Start the FastAPI file server that serves documents via HTTP for the agent.

```bash
local-search serve --workspace <name> [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--workspace` | `default` | Workspace to serve |
| `--host` | `127.0.0.1` | Server bind address |
| `--port` | `8000` | Server port |
| `--meili-url` | `http://localhost:7700` | Meilisearch URL |
| `--meili-key` | `local_search_master_key` | Meilisearch master key |
| `--dirs` | (none) | If provided, ingest these directories before starting |
| `--scheduler` | off | Start incremental sync scheduler alongside the server |
| `--interval` | `15` | Scheduler interval in minutes (only with `--scheduler`) |

```bash
local-search serve --workspace finance
local-search serve --workspace finance --scheduler --interval 15
local-search serve --workspace finance --dirs "C:\Shares\FinanceDocs" --scheduler --interval 15
```

---

## query

Ask the agent a question about your indexed documents.

```bash
local-search query [question] --workspace <name> [options]
```

Omit `question` to enter interactive mode.

| Option | Default | Description |
|--------|---------|-------------|
| `--workspace` | `default` | Workspace to query |
| `--provider` | `google` | LLM provider: `google`, `ollama`, `openai`, `anthropic` |
| `--model` | `gemma-4-31b-it` | Model name |
| `--api-key` | (from saved keys) | Override the API key for this query only. Resolution order: this flag → saved keys (`config set-key`) → env var. |
| `--meili-url` | `http://localhost:7700` | Meilisearch URL |
| `--meili-key` | `local_search_master_key` | Meilisearch master key |
| `--max-iterations` | `10` | Maximum agent loop iterations |
| `--top-k` | `5` | Number of search results per query |

```bash
local-search query "What was the AWS spend in Q3?" --workspace finance --provider google
local-search query "Vacation policy?" --workspace hr --provider ollama --model mistral
local-search query --workspace finance --provider google   # Interactive mode
```

---

## scheduler

Manage the incremental sync scheduler.

### scheduler status

Show which workspaces are scheduled and their next run times.

```bash
local-search scheduler status
```

### scheduler start

Start the scheduler as a foreground blocking process.

```bash
local-search scheduler start --workspace <name> [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--workspace` | `default` | Primary workspace to register |
| `--dirs` | (none) | Directories to register for this workspace |
| `--meili-url` | `http://localhost:7700` | Meilisearch URL |
| `--meili-key` | `local_search_master_key` | Meilisearch master key |
| `--interval` | `15` | Sync interval in minutes |

Press Ctrl+C to stop.

### scheduler trigger

Force an immediate sync for a workspace outside the normal schedule.

```bash
local-search scheduler trigger --workspace <name> [--force]
```

| Option | Description |
|--------|-------------|
| `--force` | Force full re-index (ignore delta logic) |

---

## health

Show index health and freshness across all registered workspaces.

```bash
local-search health [--stale-threshold <minutes>]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--stale-threshold` | `30` | Minutes after which a workspace is considered stale |

**Status values:**

| Icon | Status | Meaning |
|------|--------|---------|
| ✓ | healthy | Last sync within stale threshold |
| ⚠ | stale | Last sync older than threshold |
| ○ | never_synced | Workspace registered but never ingested |
| ✗ | error | Last sync failed |
| ↻ | running | Sync currently in progress |

---

## Environment Variables

| Variable | Used by |
|----------|---------|
| `GOOGLE_API_KEY` | `query`, `ui` with `--provider google` |
| `OPENAI_API_KEY` | `query`, `ui` with `--provider openai` |
| `ANTHROPIC_API_KEY` | `query`, `ui` with `--provider anthropic` |
| `MEILI_URL` | `ui` Meilisearch URL |
| `MEILI_MASTER_KEY` | `ui` Meilisearch master key |
| `LSA_DB_PATH` | All commands — database path |
| `LSA_HOST` | `ui` — dashboard host |
| `LSA_PORT` | `ui` — dashboard port |
| `LSA_PROVIDER` | `ui` — default provider |
| `LSA_MODEL` | `ui` — default model |
