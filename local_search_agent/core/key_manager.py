"""
Key manager — stores and retrieves LLM provider API keys.

Keys are saved to a JSON file in the user config directory
(via platformdirs), completely outside the project and package:

  Windows : C:\\Users\\<name>\\AppData\\Roaming\\local-search-agent\\keys.json
  macOS   : ~/Library/Application Support/local-search-agent/keys.json
  Linux   : ~/.config/local-search-agent/keys.json

Priority order when resolving a key at runtime:
  1. Explicitly passed api_key argument in SearchAgentConfig
  2. keys.json (managed by this module / CLI / UI)
  3. Environment variable (GOOGLE_API_KEY, OPENAI_API_KEY, etc.)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from platformdirs import user_config_dir

_APP_NAME = "local-search-agent"

_PROVIDER_ENV_VARS: dict[str, str] = {
    "google":    "GOOGLE_API_KEY",
    "openai":    "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "ollama":    "",
}

SUPPORTED_PROVIDERS = list(_PROVIDER_ENV_VARS.keys())

# LangSmith fixed constants — always the same values
LANGSMITH_TRACING  = "true"
LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"


def apply_langsmith_env() -> bool:
    """
    If a LangSmith API key is saved, set all four LangChain env vars so
    LangChain picks them up automatically at runtime.

    Returns True if LangSmith tracing was activated, False otherwise.
    """
    keys = _load()
    api_key = keys.get("langsmith_api_key")
    project  = keys.get("langsmith_project", "local-search-agent")
    if not api_key:
        return False
    os.environ["LANGCHAIN_TRACING_V2"]  = LANGSMITH_TRACING
    os.environ["LANGCHAIN_ENDPOINT"]    = LANGSMITH_ENDPOINT
    os.environ["LANGCHAIN_API_KEY"]     = api_key
    os.environ["LANGCHAIN_PROJECT"]     = project
    return True


def set_langsmith(api_key: str, project: str) -> None:
    """Save LangSmith credentials and immediately activate them in os.environ."""
    if not api_key.strip():
        raise ValueError("LangSmith API key must not be empty.")
    keys = _load()
    keys["langsmith_api_key"] = api_key.strip()
    keys["langsmith_project"]  = project.strip() or "local-search-agent"
    _save(keys)
    apply_langsmith_env()


def delete_langsmith() -> bool:
    """Remove saved LangSmith credentials and clear env vars."""
    keys = _load()
    changed = False
    for k in ("langsmith_api_key", "langsmith_project"):
        if k in keys:
            del keys[k]
            changed = True
    if changed:
        _save(keys)
        for var in ("LANGCHAIN_TRACING_V2", "LANGCHAIN_ENDPOINT",
                    "LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT"):
            os.environ.pop(var, None)
    return changed


def get_langsmith() -> dict:
    """Return saved LangSmith config (api_key masked, project plain)."""
    keys = _load()
    api_key = keys.get("langsmith_api_key", "")
    project  = keys.get("langsmith_project", "")
    if api_key and len(api_key) > 8:
        masked = api_key[:6] + "*" * (len(api_key) - 10) + api_key[-4:]
    elif api_key:
        masked = "****"
    else:
        masked = ""
    return {"api_key_masked": masked, "project": project, "configured": bool(api_key)}


def _keys_path() -> Path:
    config_dir = Path(user_config_dir(_APP_NAME))
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "keys.json"


def _load() -> dict[str, str]:
    path = _keys_path()
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, str)}
    except (json.JSONDecodeError, OSError):
        return {}


def _save(keys: dict[str, str]) -> None:
    path = _keys_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(keys, f, indent=2)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def set_key(provider: str, key: str) -> None:
    """
    Save an API key for a provider.
    Overwrites any existing key for that provider.
    """
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unknown provider {provider!r}. "
            f"Supported: {', '.join(SUPPORTED_PROVIDERS)}"
        )
    if provider == "ollama":
        raise ValueError("Ollama does not use an API key.")
    if not key or not key.strip():
        raise ValueError("API key must not be empty.")
    keys = _load()
    keys[provider] = key.strip()
    _save(keys)


def get_key(provider: str) -> Optional[str]:
    """
    Retrieve the saved API key for a provider from keys.json.
    Returns None if no key is saved.
    """
    return _load().get(provider)


def delete_key(provider: str) -> bool:
    """
    Remove the saved key for a provider.
    Returns True if a key was deleted, False if none existed.
    """
    keys = _load()
    if provider not in keys:
        return False
    del keys[provider]
    _save(keys)
    return True


def list_keys() -> dict[str, str]:
    """
    Return all saved keys with the value partially masked.
    e.g. {"google": "AIzaSy********************xyz"}
    """
    raw = _load()
    masked: dict[str, str] = {}
    for provider, key in raw.items():
        if len(key) <= 8:
            masked[provider] = "****"
        else:
            masked[provider] = key[:6] + "*" * (len(key) - 10) + key[-4:]
    return masked


def resolve_key(provider: str, explicit_key: Optional[str] = None) -> Optional[str]:
    """
    Resolve the API key for a provider using priority order:
      1. explicit_key (passed directly by caller)
      2. keys.json
      3. environment variable

    Returns None for ollama (no key needed).
    """
    if provider == "ollama":
        return None

    if explicit_key:
        return explicit_key

    saved = get_key(provider)
    if saved:
        return saved

    env_var = _PROVIDER_ENV_VARS.get(provider, "")
    if env_var:
        return os.environ.get(env_var) or None

    return None


def keys_file_path() -> str:
    """Return the path to the keys.json file (for display purposes)."""
    return str(_keys_path())


# ---------------------------------------------------------------------------
# Default models seeded per provider on first use
# ---------------------------------------------------------------------------

_DEFAULT_MODELS: dict[str, list[str]] = {
    "google": [
        "gemini-3.1-flash-lite",
        "gemma-4-26b-a4b-it",
        "gemma-4-31b-it",
    ],
    "openai": [],
    "anthropic": [],
    "ollama": [],
}


def _models_path() -> Path:
    config_dir = Path(user_config_dir(_APP_NAME))
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "models.json"


def _load_models() -> dict[str, list[str]]:
    path = _models_path()
    if not path.exists():
        # Seed defaults on first use
        _save_models(_DEFAULT_MODELS)
        return {k: list(v) for k, v in _DEFAULT_MODELS.items()}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Ensure all providers present
        for provider, defaults in _DEFAULT_MODELS.items():
            if provider not in data:
                data[provider] = list(defaults)
        return data
    except (json.JSONDecodeError, OSError):
        return {k: list(v) for k, v in _DEFAULT_MODELS.items()}


def _save_models(models: dict[str, list[str]]) -> None:
    path = _models_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(models, f, indent=2)


def get_models(provider: str | None = None) -> dict[str, list[str]] | list[str]:
    """
    Return stored models.
    If provider is given, return the list for that provider.
    If provider is None, return the full dict.
    """
    data = _load_models()
    if provider is not None:
        return data.get(provider, [])
    return data


def add_model(provider: str, model_name: str) -> None:
    """Add a model name for a provider. No-op if already present."""
    if provider not in _DEFAULT_MODELS:
        raise ValueError(f"Unknown provider {provider!r}.")
    if not model_name.strip():
        raise ValueError("Model name must not be empty.")
    data = _load_models()
    if model_name.strip() not in data[provider]:
        data[provider].append(model_name.strip())
        _save_models(data)


def delete_model(provider: str, model_name: str) -> bool:
    """Remove a model name for a provider. Returns True if removed."""
    data = _load_models()
    if provider not in data or model_name not in data[provider]:
        return False
    data[provider].remove(model_name)
    _save_models(data)
    return True


def models_file_path() -> str:
    """Return the path to the models.json file (for display purposes)."""
    return str(_models_path())


# ---------------------------------------------------------------------------
# Semantic settings — stored in settings.json in the user config dir
# so CLI, UI, and Python API all read from the same source of truth.
# ---------------------------------------------------------------------------

_SEMANTIC_DEFAULTS: dict[str, bool] = {
    "enable_semantic":        False,
    "enable_query_expansion": False,
    "enable_link_graph":      False,
}


def _settings_path() -> Path:
    config_dir = Path(user_config_dir(_APP_NAME))
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "settings.json"


def _load_settings() -> dict:
    path = _settings_path()
    if not path.exists():
        return dict(_SEMANTIC_DEFAULTS)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Ensure all keys present
        for k, v in _SEMANTIC_DEFAULTS.items():
            if k not in data:
                data[k] = v
        return data
    except (json.JSONDecodeError, OSError):
        return dict(_SEMANTIC_DEFAULTS)


def _save_settings(settings: dict) -> None:
    path = _settings_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)


def get_semantic_settings() -> dict[str, bool]:
    """
    Return all three semantic feature flags.

    Returns
    -------
    dict with keys: enable_semantic, enable_query_expansion, enable_link_graph
    """
    s = _load_settings()
    return {
        "enable_semantic":        bool(s.get("enable_semantic",        False)),
        "enable_query_expansion": bool(s.get("enable_query_expansion", False)),
        "enable_link_graph":      bool(s.get("enable_link_graph",      False)),
    }


def set_semantic_setting(key: str, value: bool) -> None:
    """
    Set a single semantic feature flag.

    Parameters
    ----------
    key   : One of: enable_semantic, enable_query_expansion, enable_link_graph
    value : True to enable, False to disable
    """
    if key not in _SEMANTIC_DEFAULTS:
        raise ValueError(
            f"Unknown semantic setting {key!r}. "
            f"Valid keys: {', '.join(_SEMANTIC_DEFAULTS)}"
        )
    settings = _load_settings()
    settings[key] = bool(value)
    _save_settings(settings)


def set_all_semantic_settings(
    enable_semantic: bool,
    enable_query_expansion: bool,
    enable_link_graph: bool,
) -> None:
    """Set all three semantic flags in one atomic write."""
    _save_settings({
        "enable_semantic":        bool(enable_semantic),
        "enable_query_expansion": bool(enable_query_expansion),
        "enable_link_graph":      bool(enable_link_graph),
    })


def settings_file_path() -> str:
    """Return the path to the settings.json file (for display purposes)."""
    return str(_settings_path())
