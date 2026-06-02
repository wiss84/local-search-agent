# Local Search Agent

**Give your AI agent a search engine for your local files.**

---

## What is this?

Local Search Agent is a Python framework that gives your AI agent a search engine for your local files and lets it search, fetch, and reason over your local documents — the same way a researcher searches the web, but entirely on your machine.

Point it at a folder. Ask a question. The agent searches your documents, reads the relevant ones, and gives you an answer with citations — no cloud upload, no API calls to external search services, no embeddings, no vector stores.

```
"What was the AWS spend in Q3?"  →  agent searches index  →  fetches relevant docs  →  answers with sources
```

---


## Why not RAG?

Traditional RAG (Retrieval-Augmented Generation) has a fundamental problem: it converts your documents into embeddings and stores them in a vector database. That means:

- **Stale indexes** — embeddings go out of date silently. You never know if the agent is reading your latest documents or a six-month-old snapshot
- **Black-box retrieval** — you can't see why a document was retrieved or not. Debugging poor answers is guesswork
- **Chunking anxiety** — split too small and you lose context. Split too large and retrieval quality degrades. There's no right answer
- **Infrastructure overhead** — a vector database is another service to run, maintain, and pay for
- **Semantic drift** — embeddings are sensitive to how questions are phrased. A question about "cloud expenditure" may never match a document that says "AWS spend"

Local Search Agent takes a different approach: **BM25 keyword search via Meilisearch, structured metadata, and a LangGraph agent loop with tools**. The agent searches your document index the same way a developer searches Stack Overflow — with real queries, real results, and full transparency into what was retrieved and why.

The result is deterministic, auditable, and fast. You can see exactly what the agent fetched for every answer.

---

## How it works

```
1. INGEST     Your documents → parsed, cleaned, chunked, indexed into Meilisearch
2. SERVE      FastAPI file server makes documents available to the agent via HTTP
3. SEARCH     LangGraph agent loop: search_local_index → fetch_local_url → reason
4. ANSWER     Agent returns an answer with inline source citations
```

Everything runs locally. Meilisearch downloads automatically on first use, no manual setup.

---

## Screenshots

### Desktop UI
![Local Search Agent UI](https://raw.githubusercontent.com/wiss84/local-search-agent/main/local_search_agent/docs/assets/local_search_agent_ui.webp)

### CLI Interactive Mode
![Local Search Agent CLI](https://raw.githubusercontent.com/wiss84/local-search-agent/main/local_search_agent/docs/assets/local_search_agent_cli.webp)

### Python API
![Local Search Agent Python API](https://raw.githubusercontent.com/wiss84/local-search-agent/main/local_search_agent/docs/assets/local_search_agent_api.webp)

---

## Video Demos

- **Native UI** — Watch the [UI design and configuration video demo](https://youtu.be/J-POiSDbArs)
- **CLI AGENT** — Watch the [Terminal document querying video demo](https://youtu.be/ZIiN4NG5g3U)
- **Python API** — Watch the [Local Search Agent API Integration video demo](https://youtu.be/JfoLKScLi1Y)

---

## Install

```bash
pip install local-search-agent
```

## Set your API key

```bash
# Google AI Studio (free tier — recommended) or paid from openai or anthropic
local-search config set-key --provider google --key YOUR_KEY

# Or use Ollama for a fully local, zero-cost setup (no key needed)
# Install from https://ollama.com 
# Download any model that support function calling and system instructions: 
`ollama pull gemma4:e2b` (7.2GB) 
`ollama pull gemma4:e4b` (9.6GB) 
`ollama pull nemotron-3-nano:4b` (2.8GB Highly recommended)

```

---

## Quick Start

### Desktop UI

```bash
local-search ui
```

The desktop UI open:
1. Create a workspace, name it, point it at a directory of files. The "Database path" field is optional — leave it blank to use the default location shown in the hint, or paste a custom path and click "Set & Restart".
2. Ingest (parse, clean, chunk).
3. Get a free google api key from ai-studio.
4. Set your api key at the top bar's right corner, or add a paid key for anthropic\openai .
Note: For paid models or ollama, you will need to set model name via the config button at the top  bar's right corner.
5. click Ingest from the left sidebar.
6. watch the progress bar at the bottom bar, wait until all files marked as completed.
7. Start asking questions.

### CLI

```bash
# Create a workspace and ingest documents
local-search workspace create finance "C:\my_docs"
local-search ingest --workspace finance --dirs "C:\my_docs"

# Start the file server (keep this running)
local-search serve --workspace finance

# Ask a question
local-search query "What was the AWS spend in Q3?" --workspace finance --provider google

# Use interactive mode
local-search query --workspace finance --provider google
```

### Python API

```python
from local_search_agent import SearchAgentFramework, SearchAgentConfig

config = SearchAgentConfig(
    document_dirs=["C:/my_docs"],
    workspace_name="finance",
    provider="google",
    # db_path defaults to your OS user config dir — same location as keys.json
    # override only if you need a custom location:
    # db_path="D:/mydata/search.db",
)

framework = SearchAgentFramework(config)
framework.ingest_and_index()
framework.start_file_server()

response = framework.query("What was the AWS spend in Q3?")
print(response["answer"])
```

### Agent Tool Integration

Wrap an indexed workspace as a tool and plug it into any external AI agent — LangChain, LangGraph, Google Gemini SDK, or any framework that calls a function.

```python
from local_search_agent import SearchAgentFramework, SearchAgentConfig, LocalSearchTool

config = SearchAgentConfig(
    document_dirs=["C:/skills"],
    workspace_name="skills",
    provider="google",
    model_name="gemini-2.0-flash-lite",  # cheap model for retrieval
)

# Index once
framework = SearchAgentFramework(config)
framework.ingest_and_index()
framework.start_file_server()

# Create the tool
skill_tool = LocalSearchTool(config)

# Use inside a LangChain / LangGraph agent
from langchain_core.tools import tool

@tool
def search_skills(query: str) -> str:
    """Search the skills knowledge base for coding patterns and techniques."""
    return skill_tool.run(query).answer
```

Pass `return_raw=True` to bypass the internal LLM summarisation and return the full document text verbatim — useful when the calling agent should reason over the raw content itself:

```python
skill_tool = LocalSearchTool(config, return_raw=True)
```

See the [Python API Reference](https://github.com/wiss84/local-search-agent/blob/main/local_search_agent/docs/api-reference.md#localsearchtool) for the full `LocalSearchTool` documentation.

---

## Supported File Types

| Format | Extension |
|--------|-----------|
| PDF | `.pdf` |
| Word | `.docx` |
| Excel | `.xlsx` |
| PowerPoint | `.pptx` |
| HTML | `.html`, `.htm` |
| Plain text | `.txt`, `.md` |
| CSV | `.csv` |
| JSON | `.json` |
| XML | `.xml` |
| Email | `.eml` |

---

## Key Features

- **One command install** — `pip install local-search-agent`. Meilisearch downloads automatically
- **No embeddings, no vector stores** — BM25 search with structured metadata. Fast, deterministic, auditable
- **Native desktop UI** — pywebview window with live streaming agent responses, workspace management, and chat history
- **Multi-provider LLM** — Google, Ollama (local), OpenAI, Anthropic
- **Multi-workspace** — isolate document collections by department, project, channel, or topic. Each workspace is its own search index
- **Incremental sync** — background scheduler re-indexes only changed files. A 10,000-document corpus with 50 changes re-indexes only the 50
- **Full CLI parity** — everything you can do in the UI you can do from the terminal
- **Python API** — embed the framework directly in your own application
- **Cross-platform** — Windows, macOS, Linux

---

## Documentation

| Guide | Description |
|-------|-------------|
| [Getting Started](https://github.com/wiss84/local-search-agent/blob/main/local_search_agent/docs/getting-started.md) | First steps, quick start for UI, CLI, and Python API |
| [Installation](https://github.com/wiss84/local-search-agent/blob/main/local_search_agent/docs/installation.md) | Full install guide, API keys, Ollama setup, platform notes |
| [Architecture](https://github.com/wiss84/local-search-agent/blob/main/local_search_agent/docs/architecture.md) | Full architrecture, design guide |
| [CLI Reference](https://github.com/wiss84/local-search-agent/blob/main/local_search_agent/docs/cli-reference.md) | All commands and flags |
| [Python API Reference](https://github.com/wiss84/local-search-agent/blob/main/local_search_agent/docs/api-reference.md) | Full API documentation |
| [Configuration](https://github.com/wiss84/local-search-agent/blob/main/local_search_agent/docs/configuration.md) | All config options and patterns |
| [Ingestion](https://github.com/wiss84/local-search-agent/blob/main/local_search_agent/docs/ingestion.md) | How ingestion works, supported formats, chunking, scheduler |
| [Multi-Workspace](https://github.com/wiss84/local-search-agent/blob/main/local_search_agent/docs/multi-workspace.md) | Managing multiple document collections |
| [Semantic Search](https://github.com/wiss84/local-search-agent/blob/main/local_search_agent/docs/semantic-search.md) | Experimental: concept extraction, query expansion |
| [Troubleshooting](https://github.com/wiss84/local-search-agent/blob/main/local_search_agent/docs/troubleshooting.md) | Common issues and fixes |

---

## Contributing

Contributions are welcome. Clone the repo and install in editable mode with dev dependencies:

```bash
git clone https://github.com/wiss84/local-search-agent.git
cd local-search-agent
pip install -e ".[dev]"
```

Run tests before submitting a PR:

```bash
pytest tests/ -v --cov=local_search_agent --cov-report=term-missing
ruff check .
ruff format .
```

---

## License

MIT — see [LICENSE](https://github.com/wiss84/local-search-agent/blob/main/LICENSE) for details.

---

Built by [Wissam Metawee](https://github.com/wiss84)
