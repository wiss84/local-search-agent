# Troubleshooting

## Installation Issues

### `pip install` fails with build errors

Some dependencies (particularly `docling` and `lxml`) require build tools on Linux.

```bash
# Ubuntu / Debian
sudo apt-get install build-essential python3-dev libxml2-dev libxslt1-dev

# Then retry
pip install local-search-agent
```

On macOS, install Xcode Command Line Tools:

```bash
xcode-select --install
```

### `pywebview` fails to open the UI on Linux

pywebview requires WebKitGTK. Install it:

```bash
# Ubuntu / Debian
sudo apt-get install python3-gi python3-gi-cairo gir1.2-gtk-3.0 gir1.2-webkit2-4.0

# Fedora
sudo dnf install python3-gobject webkit2gtk3
```

---

## Meilisearch Issues

### Meilisearch binary download fails

The framework downloads the Meilisearch binary on first use. If the download fails (network, proxy, permissions):

1. Check your internet connection
2. If you're behind a proxy, set `HTTPS_PROXY` environment variable
3. Try downloading manually from https://github.com/meilisearch/meilisearch/releases and placing it at the path shown in the error message
4. Re-run `local-search setup --force`

### Meilisearch won't start

Check if another process is using port 7700:

```bash
# Windows
netstat -ano | findstr :7700

# macOS / Linux
lsof -i :7700
```

Kill the conflicting process or change the port:

```python
config = SearchAgentConfig(
    meilisearch_url="http://localhost:7701",
    ...
)
```

Check the Meilisearch log for details:
- **Windows**: `C:\Users\<name>\AppData\Local\local-search-agent\Cache\<version>\logs\meilisearch.stdout.log`
- **macOS**: `~/Library/Caches/local-search-agent/<version>/logs/meilisearch.stdout.log`
- **Linux**: `~/.cache/local-search-agent/<version>/logs/meilisearch.stdout.log`

### `meilisearch_python_sdk` import error

Make sure you are not importing from the legacy `meilisearch` package. The correct package is `meilisearch-python-sdk`. If you have both installed:

```bash
pip uninstall meilisearch
pip install meilisearch-python-sdk
```

---

## Ingestion Issues

### Files are being skipped that I know have changed

The delta check uses the filesystem `modified_at` timestamp. Files copied with tools that preserve the original timestamp (e.g. `robocopy /COPYALL`, `rsync -a`) appear unchanged to the pipeline. Use `--force` to bypass delta logic:

```bash
local-search ingest --workspace finance --dirs "C:\my_docs" --force
```

### PDF text is garbled or empty

Scanned PDFs without a text layer produce no output from Docling. Check that the PDF is searchable (open in a PDF reader and try to select text). For scanned PDFs you need OCR pre-processing before ingestion.

Heavily image-based PDFs (e.g. brochures, slide-heavy reports) may also produce poor extractions. Docling handles most standard business PDFs well.

### Ingestion is very slow

Normal for large initial loads. Docling performs layout analysis which is CPU-intensive. Expect roughly 1–5 seconds per page depending on PDF complexity. For large corpora run the initial ingestion once and let the scheduler handle incremental updates.

If `enable_semantic=True`, each document also requires an LLM call — this adds significant time on large corpora.

### `IngestStats` shows many failures

Check `stats.errors` for the specific files:

```python
stats = framework.ingest_and_index()
for error in stats.errors:
    print(error)
```

Common causes: corrupted files, password-protected PDFs, files locked by another process, encoding issues in `.txt` files.

---

## Query Issues

### No API key found error

```
ValueError: No API key found for provider 'google'.
Run: local-search config set-key --provider google --key YOUR_KEY
```

Set your key:

```bash
local-search config set-key --provider google --key YOUR_KEY
```

Or in the UI: click **Set API Keys** in the top bar.

### Agent returns "I couldn't find any relevant documents"

This usually means either:

1. **The workspace wasn't ingested** — run `local-search ingest` first, then check `local-search health`
2. **Wrong workspace** — make sure `--workspace` matches the name you ingested into
3. **The file server isn't running** — the agent needs the file server to fetch document content. Run `local-search serve --workspace <name>` in a separate terminal, or use the UI which starts it automatically
4. **Query terminology mismatch** — BM25 is keyword-based. Try rephrasing with exact terms from your documents. Enable `enable_query_expansion=True` for synonym-based matching

### Agent hits max iterations without a complete answer

The agent loop reached `max_iterations` before finishing. This can happen on complex multi-step questions or very large document sets. Try:

```python
config = SearchAgentConfig(
    max_iterations=30,  # increase from default 10
    top_k=10,           # retrieve more results per search
    ...
)
```

### Ollama connection refused

Ollama must be running before you start a query. Check:

```bash
ollama list          # lists installed models
ollama serve         # start Ollama if not already running as a service
```

Also verify the model is pulled:

```bash
ollama pull mistral
```

---

## UI Issues

### The UI window is blank or white

This is usually a pywebview / WebView2 issue on Windows. Make sure the WebView2 Runtime is installed:

- Download from https://developer.microsoft.com/en-us/microsoft-edge/webview2/
- Install the Evergreen Bootstrapper

On macOS / Linux the system WebKit is used — if the window is blank, check the terminal for JavaScript errors.

### The UI opens but the chat doesn't respond

The UI backend is a FastAPI server. If it fails to start, the frontend will show an error or be unresponsive. Check the terminal output for errors when you ran `local-search ui`.

### The workspace dropdown is empty in the UI

You need to create at least one workspace. Either use the **New Workspace** button in the UI, or create one via CLI and restart the UI:

```bash
local-search workspace create my_workspace "C:\my_docs"
local-search ui
```

---

## Multi-Tenant RBAC Issues

See [Role-Based Access Control](role_based_access_control.md) for the full
conceptual guide. These only apply if you've set `identity_provider` /
passed `--multi-tenant` — single-user mode is unaffected by anything
below.

### "Authentication required" / redirected to the login page unexpectedly

Usually means the session is no longer valid — it expired (2h idle / 24h
hard cap), someone (a superadmin) revoked the API key it was created
from, or the workspace grant behind it was revoked. Revoking a key
force-logs-out any active session tied to that subject immediately, not
just the key itself — the very next action redirects straight to login,
the same as clicking Sign Out. Log in again with a valid key.

### A button/action is greyed out or missing

The frontend hides or disables controls the caller's current role can't
use (`data-requires-role` for member/admin, a separate
`data-requires-superadmin` for the stricter tier — workspace create/
delete, force re-ingest, wipe & re-ingest, concurrency settings). This is
a UX convenience, not the real boundary — the same restriction is
enforced server-side regardless, so this isn't a bug to work around, it's
accurately reflecting what that role is actually allowed to do. Check
[Role-Based Access Control — Roles and grants](role_based_access_control.md#roles-and-grants)
for exactly which tier a given action needs.

### "The model X/Y is not allowed for your role"

Model/Provider Access Control's allow-list has nothing granted for that
role yet — a role with zero rows has access to **nothing**, fail-closed,
the same as workspace access. A superadmin needs to grant at least one
provider/model to `member` and `admin` via Settings → Model Manager →
"Model access by role" (or `GET/POST /api/ui/models/access` directly)
before anyone in that role can query at all. See [Role-Based Access
Control — Model / Provider Access Control](role_based_access_control.md#model--provider-access-control).

### A query seems stuck on "N requests ahead of you"

A concurrency limit is configured for that provider (Settings → Model
Manager → Concurrency, superadmin-only) and every slot is currently in
use. This resolves on its own once a slot frees up; if it never clears
within 120 seconds the request fails with a clear error instead of
hanging forever. If this happens often, the configured limit is probably
too low for actual demand — raise it via `local-search config
set-concurrency` or the same Settings panel. See [Role-Based Access
Control — Rate Limits & Concurrency](role_based_access_control.md#rate-limits--concurrency).

### A plain admin can't create/delete a workspace, force re-ingest, or wipe & re-ingest

These four actions are superadmin-only, not workspace-admin — this is
intentional (see [Roles and grants](role_based_access_control.md#roles-and-grants)
for why), not a bug. Ordinary incremental ingest stays available to any
workspace admin; only the heavier/destructive variants and workspace
provisioning moved to superadmin.

---

## Logs and Diagnostics

### Enable debug logging

```bash
local-search --log-level DEBUG ui
local-search --log-level DEBUG ingest --workspace finance --dirs "C:\my_docs"
```

### Check index health

```bash
local-search health
```

### Check scheduler status

```bash
local-search scheduler status
```

### Check saved API keys

```bash
local-search config list-keys
```

### SQLite database inspection

The SQLite database (`local_search_agent.db` by default) contains workspace registrations, document records, and sync history. You can inspect it with any SQLite browser (e.g. DB Browser for SQLite).

---

## Getting Help

If you encounter a bug or unexpected behaviour:

1. Check this page first
2. Run with `--log-level DEBUG` and capture the output
3. Open an issue at https://github.com/wiss84/local-search-agent/issues with the log output and steps to reproduce
