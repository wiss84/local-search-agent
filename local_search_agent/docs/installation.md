# Installation Guide

## System Requirements

- **Python**: 3.11 or higher
- **RAM**: 4 GB minimum (8 GB recommended)
- **Disk**: 2 GB minimum (Meilisearch binary + index storage)
- **OS**: Windows 10/11, macOS 12+, Linux (Ubuntu 20.04+)

## Install

```bash
pip install local-search-agent
```

This installs everything needed for both single-user mode and optional
[multi-tenant RBAC](role_based_access_control.md) — `argon2-cffi`
(API-key hashing), `cryptography` (Meilisearch scoped-key encryption at
rest), and `PyJWT[crypto]` (JWT validation) are core dependencies, not an
extra you opt into separately. Nothing further to install even if you
never turn RBAC on; these packages simply sit unused until you set
`identity_provider` on `SearchAgentConfig`.

## First Run: Meilisearch Downloads Automatically

On first use the framework downloads the Meilisearch binary for your platform and caches it in your user cache directory. This happens automatically — you don't need to do anything.

The binary is cached at:
- **Windows**: `C:\Users\<name>\AppData\Local\local-search-agent\Cache\<version>\meilisearch.exe`
- **macOS**: `~/Library/Caches/local-search-agent/<version>/meilisearch`
- **Linux**: `~/.cache/local-search-agent/<version>/meilisearch`

You can also trigger the download explicitly:

```bash
local-search setup
```

## Set Your API Key

Use the CLI to save your key — it is stored securely in your user config directory, outside the project folder, and never uploaded anywhere:

```bash
# Google AI Studio (free tier)
local-search config set-key --provider google --key YOUR_KEY

# OpenAI
local-search config set-key --provider openai --key YOUR_KEY

# Anthropic
local-search config set-key --provider anthropic --key YOUR_KEY
```

Get a free Google AI Studio key at https://aistudio.google.com.

To check which keys are saved (values are masked):
```bash
local-search config list-keys
```

To remove a key:
```bash
local-search config delete-key --provider google
```

Keys are stored at:
- **Windows**: `C:\Users\<name>\AppData\Roaming\local-search-agent\keys.json`
- **macOS**: `~/Library/Application Support/local-search-agent/keys.json`
- **Linux**: `~/.config/local-search-agent/keys.json`

If you use the desktop UI, you can set keys there instead via the **Set API Keys** button in the top bar.

For the Python API, pass the key directly:
```python
config = SearchAgentConfig(provider="google", api_key="YOUR_KEY")
```

## Using Ollama (Fully Local, No API Key)

Ollama runs models locally — no key needed, no cloud calls.

**Windows:**
```bash
winget install Ollama.Ollama
```

**macOS / Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```
After the ollama app is downloaded and running, go to settings, set the `Context Length` to 16k

Restart the terminal, Then pull a model (must support function calling and system instructions):

```bash
ollama pull mistral       # good general-purpose model
ollama pull llama3.2      # Meta's latest
ollama pull gemma4:e2b    # Google's smallest Gemma 4
```

Ollama starts its server automatically at `localhost:11434`. The framework connects to it via `langchain-ollama` with no extra configuration.

## Verify Installation

```bash
local-search --help
```

## For Contributors (GitHub Clone)

```bash
git clone https://github.com/wiss84/local-search-agent
cd local-search-agent
pip install -e ".[dev]"
```

This installs the package in editable mode plus all development tools (pytest, ruff, mypy).

## Platform Notes

### Windows
- Long path support may need to be enabled for very deep directory structures
- The Meilisearch binary runs as a background process and stops automatically when the framework exits

### macOS
- Xcode Command Line Tools may be required: `xcode-select --install`
- On Apple Silicon (M1/M2/M3) the arm64 binary is downloaded automatically

### Linux
- Build essentials may be needed for some Python dependencies: `sudo apt-get install build-essential`
- The Meilisearch binary is marked executable automatically after download

## Upgrading

```bash
pip install --upgrade local-search-agent
```

Run `local-search setup --force` after upgrading if the Meilisearch version changed.

## Uninstalling

```bash
pip uninstall local-search-agent
```

This removes the Python package but not your saved keys, SQLite database, or Meilisearch cache. To remove everything:
- Delete `local_search_agent.db`
- Delete the cache directory listed above for your platform
- Delete the keys.json file listed above for your platform
