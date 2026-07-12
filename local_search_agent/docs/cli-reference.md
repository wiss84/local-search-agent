# CLI Reference

## Command Structure

```
local-search [--db <path>] [--log-level LEVEL] <command> [subcommand] [options]
```

## Global Options

| Option | Default | Description |
|--------|---------|-------------|
| `--db` | user config dir | SQLite metadata database path. Defaults to `local_search_agent.db` in your OS user config directory (same location as `keys.json`). Also reads from `LSA_DB_PATH` env var. |
| `--log-level` | `INFO` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

---

## config

Manage saved API keys, models, and semantic settings. All settings are stored in your user config directory outside the project — never in the repo.

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

Adds a model name for a provider. Stored in `models.json` in your user config directory. The model will appear in the UI sidebar dropdown and is available to `local-search query --model`. Models added here also appear in the Semantic Model dropdown in the UI settings.

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

Configure semantic search features. All flags are optional and can be combined in one call. Settings are stored in `settings.json` in your user config directory and are shared across CLI, UI, and Python API.

```bash
local-search config set-semantic [--enable true|false] [--query-expansion true|false] [--provider <provider>] [--model <model>]
```

| Flag | Description |
|------|-------------|
| `--enable` | Enable or disable ConceptCompiler + StructuralParser at ingest time |
| `--query-expansion` | Enable or disable query expansion with synonyms at search time |
| `--provider` | Provider to use for concept extraction (overrides main provider). Use `none` to reset. |
| `--model` | Model to use for concept extraction (overrides main model). Use `none` to reset. |

```bash
# Enable semantic indexing and query expansion
local-search config set-semantic --enable true --query-expansion true

# Use a cheaper model for concept extraction
local-search config set-semantic --provider google --model gemma-4-26b-a4b-it

# Set everything in one call
local-search config set-semantic --enable true --query-expansion true --provider google --model gemma-4-26b-a4b-it

# Use a local Ollama model for concept extraction
local-search config set-semantic --provider ollama --model llama3.2

# Reset semantic model back to the main agent model
local-search config set-semantic --model none --provider none
```

### config show-semantic

```bash
local-search config show-semantic
```

Shows the current state of all semantic settings including the model override.

### config set-advanced

Override a compiled-in ingestion or search constant. Overrides are stored in `advanced_settings.json` in your user config directory and take effect on the next ingest run. Constants not overridden continue to use their compiled-in defaults.

```bash
local-search config set-advanced --key <CONSTANT_NAME> --value <number>
local-search config set-advanced --reset
```

| Option | Description |
|--------|-------------|
| `--key` | Name of the constant to override (see table below) |
| `--value` | New numeric value |
| `--reset` | Remove all overrides and revert to compiled-in defaults |

**Valid keys:**

| Key | Category | Description |
|-----|----------|-------------|
| `CHUNK_MIN_CHARS` | Chunking | Minimum chars before a document is chunked |
| `CHUNK_TARGET_CHARS` | Chunking | Target chars per chunk |
| `CHUNK_MAX_CHARS` | Chunking | Hard cap chars per chunk |
| `CHUNK_OVERLAP_CHARS` | Chunking | Overlap between consecutive chunks |
| `TABLE_ROWS_PER_CHUNK` | Table/CSV | Rows per chunk for tabular data |
| `PDF_PAGES_PER_BATCH` | PDF/DOCX | Pages per processing batch |
| `PDF_SPLIT_THRESHOLD` | PDF/DOCX | Page count above which a PDF is split into batches |
| `PDF_FALLBACK_PAGES_PER_BATCH` | PDF/DOCX | Batch size used when the primary batch fails |
| `DOCX_CHAR_SPLIT_THRESHOLD` | PDF/DOCX | DOCX char count above which section-splitting is used |
| `TESSERACT_FALLBACK_MIN_CHARS` | OCR | Minimum chars from PyMuPDF before Tesseract is tried |
| `DEFAULT_TOP_K` | Search | Default number of results returned per search call |
| `DEFAULT_MAX_ITERATIONS` | Agent | Default max agent reasoning iterations |
| `SNIPPET_CONTEXT_CHARS` | Search | Characters of context around a match in snippets |

```bash
# Use smaller PDF batches on a low-RAM machine
local-search config set-advanced --key PDF_PAGES_PER_BATCH --value 10

# Larger chunks for a corpus of long technical documents
local-search config set-advanced --key CHUNK_TARGET_CHARS --value 16000

# Return more search results per agent call
local-search config set-advanced --key DEFAULT_TOP_K --value 10

# Reset all overrides back to compiled-in defaults
local-search config set-advanced --reset
```

See `config show` to verify the effective values and identify which are overridden.



### config set-concurrency / delete-concurrency

Cap the max number of LLM calls for a provider allowed in flight at once, deployment-wide. For Ollama this is the framework-side mirror of Ollama's own `OLLAMA_NUM_PARALLEL` — set it based on your actual hardware's real capacity, since this framework can't introspect your VRAM itself. For cloud providers it's a burst control layered on top of (not instead of) any RPM/TPM override set via `set-rate-limit`. See [Role-Based Access Control — Rate Limits & Concurrency](role_based_access_control.md#rate-limits--concurrency) for the full concept.

```bash
local-search config set-concurrency --provider <provider> --limit <n> [--multi-tenant]
local-search config delete-concurrency --provider <provider> [--multi-tenant]
```

| Option | Description |
|--------|-------------|
| `--provider` | One of: `google`, `openai`, `anthropic`, `ollama` |
| `--limit` | Max simultaneous in-flight LLM calls (integer ≥ 1) |
| `--multi-tenant` | Edit the multi-tenant namespace instead of single-user's — these are completely independent settings, see below. |

```bash
local-search config set-concurrency --provider ollama --limit 2
local-search config delete-concurrency --provider ollama
```

### config set-rate-limit / delete-rate-limit

Set or remove an RPM/TPM/RPD override for a provider+model. Google gets auto-detected free-tier limits by default (overridable here); every other provider tracks nothing at all until you add an override — this is the only way OpenAI/Anthropic/Ollama get real quota tracking rather than blind retry-on-error.

```bash
local-search config set-rate-limit --provider <provider> --model-name <name> [--rpm <n>] [--tpm <n>] [--rpd <n>] [--multi-tenant]
local-search config delete-rate-limit --provider <provider> --model-name <name> [--multi-tenant]
```

| Option | Description |
|--------|-------------|
| `--provider` | One of: `google`, `openai`, `anthropic`, `ollama` |
| `--model-name` | Model name this override applies to |
| `--rpm` | Requests per minute |
| `--tpm` | Tokens per minute |
| `--rpd` | Requests per day |
| `--multi-tenant` | Edit the multi-tenant namespace instead of single-user's. |

At least one of `--rpm`/`--tpm`/`--rpd` is required. An omitted dimension means "don't track this", not "unlimited".

```bash
# A paid-tier OpenAI account with real, much-higher limits than the free tier
local-search config set-rate-limit --provider openai --model-name gpt-5 --rpm 500 --tpm 2000000
```

### config show-rate-limits

```bash
local-search config show-rate-limits [--multi-tenant]
```

Shows every configured concurrency limit and quota override for the given mode's namespace (single-user by default; add `--multi-tenant` for that namespace instead — they're stored independently and never overwrite each other).

### config show

```bash
local-search config show
```

Shows everything in one view: version, saved API keys (masked), models per provider, semantic settings, advanced settings (with `[OVERRIDE]` markers next to any user-set values), and LangSmith tracing status. Useful for debugging and verifying what values are actually in use.

---

## setup

Download the Meilisearch binary for the current platform. Runs automatically on first use; call this explicitly to pre-download.

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
| `--db` | user config dir | SQLite database path. Also reads `LSA_DB_PATH`. |
| `--provider` | `google` | LLM provider: `google`, `ollama`, `openai`, `anthropic`. Also reads `LSA_PROVIDER`. |
| `--model` | `gemma-4-31b-it` | Model name. Also reads `LSA_MODEL`. |
| `--meili-url` | `http://localhost:7700` | Meilisearch URL. Also reads `MEILI_URL`. |
| `--meili-key` | `local_search_master_key` | Meilisearch master key. Also reads `MEILI_MASTER_KEY`. |
| `--scheduler-interval` | `0` | Start ingestion scheduler with this interval in minutes. `0` = disabled. |
| `--headless` | off | Run API server only, no window (for debugging). |
| `--multi-tenant` | off | Enable multi-tenant RBAC (`APIKeyIdentityProvider`) against this same `--db`. Bootstrap keys/grants first with `auth create-key` and `grant-access` against the same `--db` path. See [Role-Based Access Control](role_based_access_control.md). |
| `--insecure-cookies` | off | Allow the session cookie over plain HTTP. Needed only when `--host` is a real LAN IP rather than `127.0.0.1`/`localhost` — browsers treat only `localhost` as a secure context, so a `Secure` cookie is silently dropped on any other plain-HTTP address (login will appear to do nothing). Use only on a trusted local network; never for anything internet-facing. See [Role-Based Access Control's cookie note](role_based_access_control.md#a-note-on-testing-across-devices-on-a-lan). |

---

## workspace

Manage workspaces (named collections of documents).

### workspace create

```bash
local-search workspace create <name> <dir> [--multi-tenant]
```

```bash
local-search workspace create finance "C:\Shares\FinanceDocs"
local-search workspace create hr ./hr_policies
local-search --db D:\mydata\search.db workspace create finance "C:\Shares\FinanceDocs"
```

| Option | Description |
|--------|-------------|
| `--multi-tenant` | Also provision a scoped, member-level Meilisearch key for this workspace (see [Role-Based Access Control](role_based_access_control.md)). Only meaningful if you're running this framework in multi-tenant mode elsewhere against this same `--db`; the workspace is fully usable either way. |

### workspace list

```bash
local-search workspace list
```

### workspace delete

```bash
local-search workspace delete <name> [--wipe] [--multi-tenant]
```

| Option | Description |
|--------|-------------|
| `--wipe` | Also delete all documents from the Meilisearch index. |
| `--multi-tenant` | Also delete this workspace's scoped Meilisearch key, if one was provisioned via `workspace create --multi-tenant`. |

---

## grant-access

Grant a subject a role across one or more workspaces, in a single atomic call — either every workspace gets the grant or none do. See [Role-Based Access Control](role_based_access_control.md) for the full concept walkthrough.

```bash
local-search grant-access --subject <email> --workspace <name> [<name> ...] --role <member|admin> [--granted-by <email>]
```

| Option | Description |
|--------|-------------|
| `--subject` | Stable identity being granted access, e.g. an email. |
| `--workspace` | One or more workspace names (space-separated). |
| `--role` | `member` or `admin`. |
| `--granted-by` | Identity performing the grant, for audit purposes. Defaults to the current OS user. |

```bash
local-search grant-access --subject alice@acme.com --workspace finance --role admin
local-search grant-access --subject bob@acme.com --workspace finance marketing --role member
```

---

## revoke-access

Revoke a subject's access to one or more workspaces, or all of it if `--workspace` is omitted.

```bash
local-search revoke-access --subject <email> [--workspace <name> ...]
```

```bash
local-search revoke-access --subject bob@acme.com --workspace finance
local-search revoke-access --subject bob@acme.com   # revokes everything
```

---

## list-access

List `workspace_members` grants, optionally filtered by subject and/or workspace (either, both, or neither).

```bash
local-search list-access [--subject <email>] [--workspace <name>]
```

```bash
local-search list-access
local-search list-access --workspace finance
local-search list-access --subject alice@acme.com
```

---

## grant-model-access / revoke-model-access / list-model-access

Manage which provider+model combinations a role may use for their own
queries — a cost control, not a workspace permission. Two flat
allow-lists, one per role (`member`/`admin`), not per-person or
per-workspace. A role with nothing granted has access to **nothing**
(fail-closed) — grant at least one model to each role before anyone in
that role tries to query. Superadmin always has access to every
configured model and is unaffected by any of this. See [Role-Based
Access Control — Model / Provider Access
Control](role_based_access_control.md#model--provider-access-control)
for the full concept.

```bash
local-search grant-model-access --role <member|admin> --provider <provider> --model-name <name> [--granted-by <email>]
local-search revoke-model-access --role <member|admin> --provider <provider> --model-name <name>
local-search list-model-access [--role <member|admin>]
```

| Option | Description |
|--------|-------------|
| `--role` | `member` or `admin`. |
| `--provider` | One of: `google`, `openai`, `anthropic`, `ollama`. |
| `--model-name` | Model name. |
| `--granted-by` | Identity performing the grant, for audit purposes (`grant-model-access` only). Defaults to the current OS user. |

```bash
local-search grant-model-access --role member --provider google --model-name gemma-4-31b-it
local-search grant-model-access --role admin --provider openai --model-name gpt-5
local-search list-model-access
local-search revoke-model-access --role member --provider google --model-name gemma-4-31b-it
```

---

## auth

Manage API keys for `APIKeyIdentityProvider` (see [Role-Based Access Control](role_based_access_control.md)). Not meaningful with `HeaderIdentityProvider` or `JWTIdentityProvider`, which never issue app-level keys.

### auth create-key

Generate a new API key for a subject. The raw key is shown exactly once — only its argon2 hash is persisted, so store it securely immediately; there is no way to retrieve it again later.

```bash
local-search auth create-key --subject <email> [--display-name <name>] [--superadmin] [--created-by <email>]
```

| Option | Description |
|--------|-------------|
| `--subject` | Stable identity, e.g. an email. |
| `--display-name` | Human-readable name for UI display only. |
| `--superadmin` | Mark this identity as a framework-level superadmin (rarely needed — most identities should use workspace grants instead). |
| `--created-by` | Identity creating the key, for audit purposes. Defaults to the current OS user. |

```bash
local-search auth create-key --subject alice@acme.com --display-name "Alice"
```

### auth revoke-key

```bash
local-search auth revoke-key <key_id>
```

Revokes an API key by its `key_id` (see `auth list-keys` for key IDs). Revoking a key stops that specific credential from authenticating; it does not touch the subject's workspace grants — see the RBAC guide's note on why these are kept separate.

### auth list-keys

```bash
local-search auth list-keys [--subject <email>]
```

Lists key metadata only (`key_id`, `subject`, `display_name`, status, `created_at`) — never the raw key or its hash.

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

```bash
local-search ingest --workspace finance --dirs "C:\Shares\FinanceDocs"
local-search ingest --workspace finance --dirs "C:\dir1" "C:\dir2" --force
local-search ingest --workspace finance --dirs "C:\Shares\FinanceDocs" --wipe
```

---

## serve

Start the FastAPI file server.

```bash
local-search serve --workspace <name> [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--workspace` | `default` | Workspace to serve |
| `--host` | `127.0.0.1` | Server bind address |
| `--port` | `8000` | Server port |
| `--dirs` | (none) | If provided, ingest these directories before starting |
| `--scheduler` | off | Start incremental sync scheduler alongside the server |
| `--interval` | `15` | Scheduler interval in minutes |

---

## query

Ask the agent a question.

```bash
local-search query [question] --workspace <name> [options]
```

Omit `question` to enter interactive mode.

| Option | Default | Description |
|--------|---------|-------------|
| `--workspace` | `default` | Workspace to query |
| `--provider` | `google` | LLM provider: `google`, `ollama`, `openai`, `anthropic` |
| `--model` | `gemma-4-31b-it` | Model name |
| `--api-key` | (from saved keys) | Override the API key for this query only |
| `--max-iterations` | `10` | Maximum agent loop iterations |
| `--top-k` | `5` | Number of search results per query |

```bash
local-search query "What was the AWS spend in Q3?" --workspace finance --provider google
local-search query --workspace finance   # Interactive mode
```

---

## watch

Filesystem event-driven re-ingestion. Reacts to file changes instantly without polling. **Recommended over the polling scheduler** for most use cases.

### watch start

```bash
local-search watch start --workspace <name> [--no-enrich]
```

Start watch mode as a foreground process. Blocks until interrupted (Ctrl+C). Watches all document directories in the workspace and re-indexes within seconds of file changes.

| Option | Description |
|--------|-------------|
| `--workspace` | Workspace to watch |
| `--no-enrich` | Skip semantic enrichment on watch-triggered syncs (faster, but you may need a later full re-ingest to backfill semantic fields) |

```bash
# Start watching a workspace with semantic enrichment (default)
local-search watch start --workspace finance

# Start watching but skip semantic enrichment for speed
local-search watch start --workspace finance --no-enrich
```

### watch status

```bash
local-search watch status
```

Show which workspaces and directories are currently being watched. Returns "Watch mode is not running." if no watcher is active.

```bash
local-search watch status
# Output:
# Watch mode running -- 2 workspace(s)
#   finance: 2 directory(ies)
#   legal: 1 directory(ies)
```

---

## scheduler

⚠️ **DEPRECATED** — Use `watch start` instead. Watch mode is event-driven and reacts instantly to file changes without polling delays.

The polling-based scheduler is kept for backward compatibility but is no longer recommended. It will be removed in a future major version.

### scheduler status

```bash
local-search scheduler status
```

*Deprecated.* Use `watch status` instead.

### scheduler start

```bash
local-search scheduler start --workspace <name> [--interval <minutes>]
```

*Deprecated.* Use `watch start` instead.

### scheduler trigger

```bash
local-search scheduler trigger --workspace <name> [--force]
```

*Deprecated.* Use `watch trigger` (not yet implemented) or manual sync.

---

## health

```bash
local-search health [--stale-threshold <minutes>]
```

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
