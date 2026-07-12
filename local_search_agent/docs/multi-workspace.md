# Multi-Workspace Guide

## What is a Workspace?

A workspace is a named, isolated collection of documents with its own Meilisearch index. Each workspace ingests from its own directory (or directories), maintains its own sync history, and can be queried independently.

Common patterns:
- One workspace per department: `finance`, `hr`, `legal`, `engineering`
- One workspace per project
- One workspace per document type: `contracts`, `reports`, `emails`

Workspaces are fully isolated — a query against `finance` never searches `hr`.

---

## Creating Workspaces

### CLI

```bash
local-search workspace create finance "C:\Shares\Finance"
local-search workspace create hr "C:\Shares\HR"
local-search workspace create legal "C:\Shares\Legal"
```

### Python API

```python
framework.create_workspace("finance", "C:/Shares/Finance")
framework.create_workspace("hr",      "C:/Shares/HR")
framework.create_workspace("legal",   "C:/Shares/Legal")
```

### UI

Open the workspace dropdown in the top bar → **New Workspace** → enter a name and pick a folder.

---

## Ingesting Multiple Workspaces

### CLI

Each workspace is ingested separately:

```bash
local-search ingest --workspace finance --dirs "C:\Shares\Finance"
local-search ingest --workspace hr      --dirs "C:\Shares\HR"
local-search ingest --workspace legal   --dirs "C:\Shares\Legal"
```

### Python API

```python
framework.create_workspace("finance", "C:/Shares/Finance")
framework.create_workspace("hr",      "C:/Shares/HR")
framework.create_workspace("legal",   "C:/Shares/Legal")

framework.ingest_workspace("finance")
framework.ingest_workspace("hr")
framework.ingest_workspace("legal")
```

---

## Querying a Specific Workspace

### CLI

```bash
local-search query "What is the parental leave policy?" --workspace hr --provider google
local-search query "What was Q3 cloud spend?"           --workspace finance --provider google
```

### Python API

```python
response = framework.query("What is the parental leave policy?", workspace="hr")
response = framework.query("What was Q3 cloud spend?",           workspace="finance")
```

### UI

Use the workspace selector in the top bar to switch between workspaces before sending a message.

---

## Managing Workspaces

### List all workspaces

```bash
local-search workspace list
```

```python
workspaces = framework.list_workspaces()
for ws in workspaces:
    print(ws["name"], ws["document_dir"])
```

### Delete a workspace (keep index)

Removes the registration from SQLite. The Meilisearch index is kept — useful if you want to re-register the workspace later without re-ingesting.

```bash
local-search workspace delete old_project
```

```python
framework.delete_workspace("old_project")
```

### Delete a workspace and wipe the index

```bash
local-search workspace delete old_project --wipe
```

```python
framework.delete_workspace("old_project", wipe_index=True)
```

### Wipe and re-ingest

Deletes everything (index + SQLite records) and runs a full fresh ingest:

```bash
local-search ingest --workspace finance --dirs "C:\Shares\Finance" --wipe
```

```python
framework.wipe_and_reingest("finance")
```

---

## Scheduler with Multiple Workspaces

The incremental scheduler registers and manages all workspaces automatically.

### CLI

```bash
# Start the server with the scheduler — all registered workspaces sync automatically
local-search serve --workspace finance --scheduler --interval 15
```

### Python API

```python
framework.start_file_server()
framework.start_incremental_scheduler(interval_minutes=15)
# All workspaces already registered in the DB are picked up automatically

# Add a new workspace to a running scheduler
framework.add_workspace_to_scheduler("legal", interval_minutes=30)

# Force immediate sync for one workspace
framework.trigger_sync_now("finance")
```

### Scheduler Status

```bash
local-search scheduler status
```

```python
status = framework.get_scheduler_status()
# {
#   "running": True,
#   "scheduled_jobs": [
#     {"workspace": "finance", "interval_minutes": 15, "next_run": "2025-05-26T14:30:00"},
#     {"workspace": "hr",      "interval_minutes": 15, "next_run": "2025-05-26T14:32:00"},
#   ]
# }
```

---

## Health Monitoring

```bash
local-search health
```

Output:
```
Workspace         Status     Docs    Last Sync              Next Sync
finance           ✓ healthy  1,842   2 minutes ago          13 minutes
hr                ✓ healthy    341   4 minutes ago          11 minutes
legal             ⚠ stale      892   2 hours ago            —
old_archive       ○ never       0    never                  —
```

```python
summary = framework.get_index_health()
print(f"Total workspaces: {summary.total_workspaces}")
print(f"Healthy: {summary.healthy}  Stale: {summary.stale}")

for ws in summary.workspaces:
    print(f"{ws.workspace:<20} {ws.status:<12} {ws.doc_count} docs")
```

**Status values:**

| Status | Meaning |
|--------|---------|
| `healthy` | Last sync within stale threshold (default 30 min) |
| `stale` | Last sync older than threshold — index may be out of date |
| `never_synced` | Workspace registered but never ingested |
| `error` | Last sync failed |
| `running` | Sync currently in progress |

---

## Multiple Source Directories per Workspace

A workspace can pull from more than one directory:

```bash
local-search ingest --workspace finance --dirs "C:\Finance\Reports" "C:\Finance\Contracts" "C:\Finance\Emails"
```

```python
config = SearchAgentConfig(
    document_dirs=[
        "C:/Finance/Reports",
        "C:/Finance/Contracts",
        "C:/Finance/Emails",
    ],
    workspace_name="finance",
    provider="google",
)
framework = SearchAgentFramework(config)
framework.ingest_and_index()
```

All directories feed into the same Meilisearch index. Documents from different directories are distinguished by their `folder_path` field, which can be used as a search filter.

---

## Access Control Across Workspaces

Everything above describes single-user mode, where anyone who can reach
the server can query and manage every workspace. If you're serving
multiple employees or teams from one shared deployment, layer
[multi-tenant RBAC](role_based_access_control.md) on top — it changes
nothing about how workspaces themselves work (still one Meilisearch index
per workspace, still fully isolated), it only adds a permission check in
front of them.

Access is granted per subject, per workspace, as `member` or `admin` —
not globally. A subject can be `admin` in `finance` and have no access to
`hr` at all:

```bash
local-search grant-access --subject alice@acme.com --workspace finance --role admin
local-search grant-access --subject alice@acme.com --workspace marketing --role member
# alice@acme.com has no grant for hr or legal — queries against those
# workspaces are denied, not silently empty-results.
```

A few actions from the sections above become **admin-only** once RBAC is
on, rather than available to anyone: `workspace create` and
`workspace delete` are global-admin-only (see
[Role-Based Access Control](role_based_access_control.md#concepts) for
why these can't be scoped to "admin of my workspace only"), while
ingesting/syncing an *existing* workspace (`ingest`, `watch start`,
`scheduler start`) requires `admin` on that specific workspace. Querying
a workspace (`query`, and the CLI walkthrough's read operations above)
only requires `member`.

This is entirely opt-in — set `identity_provider` on your config (or pass
`--multi-tenant` to `local-search ui`) to turn it on; everything on this
page works exactly as written if you never do.

---

## Complete Multi-Workspace Example

```python
from local_search_agent import SearchAgentFramework, SearchAgentConfig

config = SearchAgentConfig(
    provider="google",
    model_name="gemma-4-31b-it",
    workspace_name="finance",  # primary workspace
)

framework = SearchAgentFramework(config)

# Create and ingest all workspaces
workspaces = {
    "finance": "C:/Shares/Finance",
    "hr":      "C:/Shares/HR",
    "legal":   "C:/Shares/Legal",
}
for name, directory in workspaces.items():
    framework.create_workspace(name, directory)
    stats = framework.ingest_workspace(name)
    print(f"{name}: {stats}")

# Start server and scheduler
framework.start_file_server()
framework.start_incremental_scheduler(interval_minutes=15)

# Query specific workspaces
hr_response     = framework.query("What is the remote work policy?",   workspace="hr")
legal_response  = framework.query("Summarise the NDA template.",       workspace="legal")
finance_response = framework.query("What was the Q3 AWS spend?",       workspace="finance")

print(hr_response["answer"])
```
