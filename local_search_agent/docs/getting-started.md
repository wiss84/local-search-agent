# Getting Started

## What is Local Search Agent?

Local Search Agent gives your AI agent a search engine for your local files. Point it at a folder of documents — PDFs, Word files, Excel sheets, HTML pages, plain text, CSV, Markdown — and ask questions in natural language. The agent searches, fetch the relevent doc(s), reads, and reasons over your documents the same way a researcher would use the web, but entirely on your machine.

No embeddings. No vector stores. No cloud upload. BM25 search via Meilisearch, a LangGraph agent loop with tools, and a native desktop UI.

## Prerequisites

- Python 3.11+
- One of the following:
  - A free Google AI Studio API key (https://aistudio.google.com)
  - An OpenAI or Anthropic API key (paid)
  - Ollama installed for fully local, zero-cost usage

## Quick Start (UI)

#### 1. Install

```bash
pip install local-search-agent
```

#### 2. Set your API key

```bash
local-search config set-key --provider google --key YOUR_KEY
```

Or use Ollama instead — see: [Installation Guide](installation.md)

#### 3. Open the UI

```bash
local-search ui
```

Meilisearch downloads and starts automatically on first run. The desktop window opens and you're ready to go. Create a workspace, point it at a folder, ingest, and start asking questions.

---

## CLI Quick Start

If you prefer the terminal, you have two options:

**Option A: Watch Mode (recommended)**  — Reacts to file changes instantly without polling.

#### Create a workspace

```bash
local-search workspace create my_workspace "C:\my_docs"
```

#### Ingest your documents

```bash
local-search ingest --workspace my_workspace --dirs "C:\my_docs"
```

Output:
```
Ingesting ['C:\my_docs'] into workspace 'my_workspace' ...
Done. IngestStats(total=12, indexed=12, skipped=0, failed=0, duration=18.4s)
```

#### Start watch mode (blocks; Ctrl+C to stop)

```bash
local-search watch start --workspace my_workspace
```

#### In another terminal, start the file server

```bash
local-search serve --workspace my_workspace
```

#### Ask a question (in a third terminal)

```bash
local-search query "What was the AWS spend in Q3?" --workspace my_workspace --provider google
```

---

**Option B: Polling Scheduler (deprecated)**  — Falls back to polling at fixed intervals. Use watch mode instead.

Skip ahead to the Python API Quick Start or consult the [CLI Reference](cli-reference.md#scheduler) for deprecated scheduler commands.

---

## Python API Quick Start

```python
from local_search_agent import SearchAgentFramework, SearchAgentConfig

config = SearchAgentConfig(
    document_dirs=["C:/my_docs"],
    workspace_name="my_workspace",
    provider="google",
    api_key="YOUR_KEY",
    model_name="gemma-4-31b-it",
)

framework = SearchAgentFramework(config)
framework.ingest_and_index()
framework.start_file_server()

# Option A: Watch mode (recommended) — reacts to file changes instantly
framework.set_watch_mode_settings(enable_watch_mode=True, enrich_on_watch=True)
framework.start_watch_mode()  # non-blocking, runs in background thread

# Option B: Polling scheduler (deprecated) — use watch mode instead
# framework.start_incremental_scheduler(interval_minutes=15)

response = framework.query("What was the AWS spend in Q3?")
print(response["answer"])

# Later, stop watch mode
# framework.stop_watch_mode()
```

---

## Supported File Types

| Extension | Parser |
|-----------|--------|
| `.pdf` | Docling |
| `.docx` | Docling |
| `.html`, `.htm` | BeautifulSoup4 |
| `.xlsx` | openpyxl |
| `.pptx` | python-pptx |
| `.txt`, `.md` | TextParser |
| `.csv` | CSVParser |
| `.json` | JSONParser |
| `.xml` | XMLParser |
| `.eml` | EMLParser |

## Next Steps

- [CLI Reference](cli-reference.md) — all commands and flags
- [Python API Reference](api-reference.md) — full API documentation
- [Configuration Guide](configuration.md) — all config options, including watch mode and re-ranking
- [Multi-Workspace Guide](multi-workspace.md) — managing multiple document collections
- [Semantic Search](semantic-search.md) — optional AI-powered search enhancement

---

## Key Features

| Feature | Description |
|---------|-------------|
| **BM25 Search** | Deterministic, auditable keyword search via Meilisearch |
| **Re-ranking** | Cross-encoder re-scoring (flashrank) improves relevance of top results. Enabled by default. |
| **Watch Mode** | Filesystem-driven re-ingestion. Reacts to file changes within seconds. |
| **Semantic Search** | Optional AI-powered concept extraction + query expansion (ConceptCompiler, StructuralParser) |
| **Multi-format** | PDFs, Word, Excel, HTML, CSV, JSON, Markdown, plain text, email |
| **PDF OCR** | Tiered strategy: native text → Tesseract → RapidOCR (automatic fallback) |
| **Local-first** | No cloud upload. All computation happens on your machine. |
| **Desktop UI** | Native OS window (pywebview) for easy document management and queries |
