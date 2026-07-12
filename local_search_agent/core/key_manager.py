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
    "google": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "ollama": "",
}

SUPPORTED_PROVIDERS = list(_PROVIDER_ENV_VARS.keys())

# LangSmith fixed constants — always the same values
LANGSMITH_TRACING = "true"
LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"


def apply_langsmith_env() -> bool:
    """
    If a LangSmith API key is saved, set all four LangChain env vars so
    LangChain picks them up automatically at runtime.

    Returns True if LangSmith tracing was activated, False otherwise.
    """
    keys = _load()
    api_key = keys.get("langsmith_api_key")
    project = keys.get("langsmith_project", "local-search-agent")
    if not api_key:
        return False
    os.environ["LANGCHAIN_TRACING_V2"] = LANGSMITH_TRACING
    os.environ["LANGCHAIN_ENDPOINT"] = LANGSMITH_ENDPOINT
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = project
    return True


def set_langsmith(api_key: str, project: str) -> None:
    """Save LangSmith credentials and immediately activate them in os.environ."""
    if not api_key.strip():
        raise ValueError("LangSmith API key must not be empty.")
    keys = _load()
    keys["langsmith_api_key"] = api_key.strip()
    keys["langsmith_project"] = project.strip() or "local-search-agent"
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
        for var in (
            "LANGCHAIN_TRACING_V2",
            "LANGCHAIN_ENDPOINT",
            "LANGCHAIN_API_KEY",
            "LANGCHAIN_PROJECT",
        ):
            os.environ.pop(var, None)
    return changed


def get_langsmith() -> dict:
    """Return saved LangSmith config (api_key masked, project plain)."""
    keys = _load()
    api_key = keys.get("langsmith_api_key", "")
    project = keys.get("langsmith_project", "")
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
            f"Unknown provider {provider!r}. Supported: {', '.join(SUPPORTED_PROVIDERS)}"
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

_SEMANTIC_DEFAULTS: dict = {
    "enable_semantic": False,
    "enable_query_expansion": False,
    "semantic_provider": "",
    "semantic_model": "",
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


def get_semantic_settings() -> dict:
    """
    Return semantic feature flags and model overrides.

    Returns
    -------
    dict with keys: enable_semantic, enable_query_expansion,
                    semantic_provider, semantic_model
    """
    s = _load_settings()
    return {
        "enable_semantic": bool(s.get("enable_semantic", False)),
        "enable_query_expansion": bool(s.get("enable_query_expansion", False)),
        "semantic_provider": s.get("semantic_provider", ""),
        "semantic_model": s.get("semantic_model", ""),
    }


def set_semantic_setting(key: str, value: bool) -> None:
    """
    Set a single semantic feature flag.

    Parameters
    ----------
    key   : One of: enable_semantic, enable_query_expansion
    value : True to enable, False to disable
    """
    if key not in _SEMANTIC_DEFAULTS:
        raise ValueError(
            f"Unknown semantic setting {key!r}. Valid keys: {', '.join(_SEMANTIC_DEFAULTS)}"
        )
    settings = _load_settings()
    settings[key] = bool(value)
    _save_settings(settings)


def set_all_semantic_settings(
    enable_semantic: bool,
    enable_query_expansion: bool,
    semantic_provider: str = "",
    semantic_model: str = "",
) -> None:
    """Set all semantic settings in one atomic write."""
    _save_settings(
        {
            "enable_semantic": bool(enable_semantic),
            "enable_query_expansion": bool(enable_query_expansion),
            "semantic_provider": semantic_provider.strip(),
            "semantic_model": semantic_model.strip(),
        }
    )


def settings_file_path() -> str:
    """Return the path to the settings.json file (for display purposes)."""
    return str(_settings_path())


# ---------------------------------------------------------------------------
# Watch mode settings — stored in settings.json alongside semantic settings.
# ---------------------------------------------------------------------------

_WATCH_MODE_DEFAULTS: dict = {
    "enable_watch_mode": False,
    "enrich_on_watch": True,
}


def get_watch_mode_settings() -> dict:
    """
    Return watch-mode feature flags.

    Returns
    -------
    dict with keys: enable_watch_mode, enrich_on_watch
    """
    s = _load_settings()
    return {
        "enable_watch_mode": bool(s.get("enable_watch_mode", False)),
        "enrich_on_watch": bool(s.get("enrich_on_watch", True)),
    }


def set_all_watch_mode_settings(enable_watch_mode: bool, enrich_on_watch: bool) -> None:
    """Set all watch-mode settings in one atomic write."""
    s = _load_settings()
    s["enable_watch_mode"] = bool(enable_watch_mode)
    s["enrich_on_watch"] = bool(enrich_on_watch)
    _save_settings(s)


# ---------------------------------------------------------------------------
# Re-ranking settings — stored in settings.json.
# ---------------------------------------------------------------------------


def get_reranking_settings() -> dict:
    """
    Return re-ranking feature flags.

    Returns
    -------
    dict with keys: enable_reranking, rerank_candidate_multiplier
    """
    s = _load_settings()
    return {
        "enable_reranking": bool(s.get("enable_reranking", True)),
        "rerank_candidate_multiplier": int(s.get("rerank_candidate_multiplier", 4)),
    }


def set_all_reranking_settings(enable_reranking: bool, rerank_candidate_multiplier: int) -> None:
    """Set all re-ranking settings in one atomic write."""
    s = _load_settings()
    s["enable_reranking"] = bool(enable_reranking)
    s["rerank_candidate_multiplier"] = int(rerank_candidate_multiplier)
    _save_settings(s)


# ---------------------------------------------------------------------------
# Custom DB path — stored in settings.json so the UI remembers it across restarts
# ---------------------------------------------------------------------------


def get_saved_db_path() -> Optional[str]:
    """Return the user-saved custom db_path, or None if using the default."""
    s = _load_settings()
    return s.get("db_path") or None


def set_saved_db_path(path: Optional[str]) -> None:
    """
    Persist a custom db_path to settings.json.
    Pass None to clear it (revert to default).
    """
    s = _load_settings()
    if path:
        s["db_path"] = str(path)
    else:
        s.pop("db_path", None)
    _save_settings(s)


# ---------------------------------------------------------------------------
# Advanced / ingestion tuning settings — stored in advanced_settings.json
# in the user config dir so they survive pip upgrades.
# Each key corresponds to a constant in constants.py; unset keys fall back
# to the hardcoded constant values at runtime.
# ---------------------------------------------------------------------------

# Keys and their Python types, mirroring constants.py
_ADVANCED_SETTING_KEYS: dict[str, type] = {
    # Chunking
    "CHUNK_MIN_CHARS": int,
    "CHUNK_TARGET_CHARS": int,
    "CHUNK_MAX_CHARS": int,
    "CHUNK_OVERLAP_CHARS": int,
    # Table chunking
    "TABLE_ROWS_PER_CHUNK": int,
    # PDF / DOCX batching
    "PDF_PAGES_PER_BATCH": int,
    "PDF_SPLIT_THRESHOLD": int,
    "PDF_FALLBACK_PAGES_PER_BATCH": int,
    "DOCX_CHAR_SPLIT_THRESHOLD": int,
    # OCR fallback
    "TESSERACT_FALLBACK_MIN_CHARS": int,
    # Agent
    "DEFAULT_TOP_K": int,
    "DEFAULT_MAX_ITERATIONS": int,
    # Search
    "SNIPPET_CONTEXT_CHARS": int,
}


def _advanced_path() -> Path:
    config_dir = Path(user_config_dir(_APP_NAME))
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "advanced_settings.json"


def _load_advanced() -> dict:
    path = _advanced_path()
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_advanced(data: dict) -> None:
    path = _advanced_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_advanced_settings() -> dict:
    """
    Return all user-overridden advanced settings.
    Only keys that have been explicitly set are returned; callers should
    fall back to constants.py for any missing key.
    """
    return dict(_load_advanced())


def set_advanced_settings(overrides: dict) -> None:
    """
    Persist a full set of advanced setting overrides.
    Pass an empty dict to reset everything to defaults.
    Values are coerced to their expected types; unknown keys are ignored.
    """
    cleaned: dict = {}
    for key, expected_type in _ADVANCED_SETTING_KEYS.items():
        if key in overrides and overrides[key] is not None and str(overrides[key]).strip() != "":
            try:
                cleaned[key] = expected_type(overrides[key])
            except (ValueError, TypeError):
                pass  # silently skip bad values
    _save_advanced(cleaned)


def advanced_settings_file_path() -> str:
    """Return the path to the advanced_settings.json file (for display)."""
    return str(_advanced_path())


def get_effective_constants() -> dict:
    """
    Return the effective value of every advanced setting, merging user
    overrides on top of the compiled-in constants from constants.py.
    This is the single source of truth that ingestion and search code
    should read from when they want to respect user overrides.
    """
    from local_search_agent.core import constants as _C

    defaults = {
        "CHUNK_MIN_CHARS": _C.CHUNK_MIN_CHARS,
        "CHUNK_TARGET_CHARS": _C.CHUNK_TARGET_CHARS,
        "CHUNK_MAX_CHARS": _C.CHUNK_MAX_CHARS,
        "CHUNK_OVERLAP_CHARS": _C.CHUNK_OVERLAP_CHARS,
        "TABLE_ROWS_PER_CHUNK": _C.TABLE_ROWS_PER_CHUNK,
        "PDF_PAGES_PER_BATCH": _C.PDF_PAGES_PER_BATCH,
        "PDF_SPLIT_THRESHOLD": _C.PDF_SPLIT_THRESHOLD,
        "PDF_FALLBACK_PAGES_PER_BATCH": _C.PDF_FALLBACK_PAGES_PER_BATCH,
        "DOCX_CHAR_SPLIT_THRESHOLD": _C.DOCX_CHAR_SPLIT_THRESHOLD,
        "TESSERACT_FALLBACK_MIN_CHARS": _C.TESSERACT_FALLBACK_MIN_CHARS,
        "DEFAULT_TOP_K": _C.DEFAULT_TOP_K,
        "DEFAULT_MAX_ITERATIONS": _C.DEFAULT_MAX_ITERATIONS,
        "SNIPPET_CONTEXT_CHARS": _C.SNIPPET_CONTEXT_CHARS,
    }
    overrides = _load_advanced()
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Rate limits & concurrency -- stored in rate_limits.json (separate from
# settings.json since it's dict-of-dicts, same reasoning models.json is its
# own file rather than living in the flat settings.json key space).
#
# Namespaced by mode ("single_user" vs "multi_tenant") within ONE file,
# rather than sharing one flat namespace -- these are genuinely
# independent settings surfaces, not the same setting seen two ways:
# a person might run this framework single-user on their own laptop AND
# separately run/test a multi-tenant deployment using the SAME OS user
# account on the SAME machine (platformdirs' user_config_dir() is keyed
# per-OS-user, not per app-mode) -- without this split, toggling
# --multi-tenant on to test RBAC would silently read/overwrite the exact
# same file a single-user setup already relies on, and vice versa. Every
# get/set/delete function below takes an explicit multi_tenant: bool
# rather than trying to auto-detect which one is "active" -- callers
# already know their own mode (ui/api_routes.py has
# app_state.config.identity_provider; agent/agent.py has config itself;
# the CLI takes an explicit --multi-tenant flag) and guessing here would
# just be a second, redundant place for that logic to drift out of sync.
#
# Two independent things live in each namespace, both fully admin-
# configurable so a company running paid-tier models with much higher
# limits than the free tier can set their own real numbers rather than
# being stuck with the free-tier defaults auto-detected in
# agent/rate_limit_handler.py:
#
#   concurrency      : { provider: max_simultaneous_llm_calls }
#                       Caps how many LLM calls for that provider may be
#                       in flight at once, deployment-wide. For Ollama
#                       this is the framework-side mirror of
#                       OLLAMA_NUM_PARALLEL -- the admin sets this based on
#                       their own hardware's real capacity, this module has
#                       no way to introspect VRAM itself. Absent = no cap
#                       (today's behavior, unchanged until an admin opts
#                       in by setting one). The UI only ever exposes this
#                       in multi-tenant mode (superadmin-only) -- a
#                       single-user desktop install has no separate
#                       "deployment" to protect from itself, so there's
#                       nothing for a concurrency cap to usefully do
#                       there; the single_user namespace still exists
#                       structurally for symmetry/completeness, it's just
#                       never written to by the UI in that mode.
#
#   quota_overrides  : { provider: { model_name: {rpm, tpm, rpd} } }
#                       Overrides agent/rate_limit_handler.py's
#                       auto-detected Google free-tier limits, and is the
#                       ONLY way non-Google providers get any RPM/TPM
#                       tracking at all (they otherwise run in
#                       retry-only mode with no quota tracking whatsoever
#                       -- see that module's own docstring). Any of
#                       rpm/tpm/rpd may be omitted; an omitted field means
#                       "don't track this dimension" for that model, not
#                       "unlimited" -- there is no such thing as tracking
#                       toward an unlimited number. Unlike concurrency,
#                       this stays visible and editable in single-user
#                       mode too (a solo user on a paid-tier account
#                       benefits from real RPM/TPM tracking just as much
#                       as a company would).
# ---------------------------------------------------------------------------

_RATE_LIMIT_NAMESPACES = ("single_user", "multi_tenant")

_RATE_LIMIT_DEFAULTS: dict = {
    "concurrency": {},
    "quota_overrides": {},
}


def _rate_limit_mode_key(multi_tenant: bool) -> str:
    return "multi_tenant" if multi_tenant else "single_user"


def _rate_limits_path() -> Path:
    config_dir = Path(user_config_dir(_APP_NAME))
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "rate_limits.json"


def _load_rate_limits_file() -> dict:
    """Load the whole file (both namespaces), filling in any missing
    namespace/keys with defaults."""
    path = _rate_limits_path()
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            data = {}
    else:
        data = {}
    for ns in _RATE_LIMIT_NAMESPACES:
        if ns not in data or not isinstance(data[ns], dict):
            data[ns] = {}
        for key, default in _RATE_LIMIT_DEFAULTS.items():
            if key not in data[ns] or not isinstance(data[ns][key], dict):
                data[ns][key] = dict(default)
    return data


def _save_rate_limits_file(data: dict) -> None:
    path = _rate_limits_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_concurrency_limits(multi_tenant: bool) -> dict[str, int]:
    """Return {provider: max_simultaneous_llm_calls} for every provider
    that has an explicit limit configured IN THIS MODE's namespace. A
    provider absent from this dict has no concurrency cap at all."""
    ns = _rate_limit_mode_key(multi_tenant)
    return dict(_load_rate_limits_file()[ns]["concurrency"])


def set_concurrency_limit(provider: str, limit: int, multi_tenant: bool) -> None:
    """Set the max simultaneous in-flight LLM calls for a provider, in THIS MODE's namespace."""
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unknown provider {provider!r}. Supported: {', '.join(SUPPORTED_PROVIDERS)}"
        )
    limit = int(limit)
    if limit < 1:
        raise ValueError("Concurrency limit must be at least 1.")
    data = _load_rate_limits_file()
    ns = _rate_limit_mode_key(multi_tenant)
    data[ns]["concurrency"][provider] = limit
    _save_rate_limits_file(data)


def delete_concurrency_limit(provider: str, multi_tenant: bool) -> bool:
    """Remove a provider's concurrency cap in THIS MODE's namespace (reverts to unbounded). Returns True if one existed."""
    data = _load_rate_limits_file()
    ns = _rate_limit_mode_key(multi_tenant)
    if provider not in data[ns]["concurrency"]:
        return False
    del data[ns]["concurrency"][provider]
    _save_rate_limits_file(data)
    return True


def get_quota_overrides(multi_tenant: bool, provider: Optional[str] = None) -> dict:
    """
    Return configured RPM/TPM/RPD overrides from THIS MODE's namespace.
    provider=None returns the full {provider: {model_name: {...}}} dict;
    a specific provider returns just that provider's {model_name: {...}}.
    """
    ns = _rate_limit_mode_key(multi_tenant)
    overrides = _load_rate_limits_file()[ns]["quota_overrides"]
    if provider is not None:
        return dict(overrides.get(provider, {}))
    return {k: dict(v) for k, v in overrides.items()}


def set_quota_override(
    provider: str,
    model_name: str,
    multi_tenant: bool,
    rpm: Optional[int] = None,
    tpm: Optional[int] = None,
    rpd: Optional[int] = None,
) -> None:
    """
    Set (or replace) the RPM/TPM/RPD override for one provider+model, in
    THIS MODE's namespace. At least one of rpm/tpm/rpd must be given. Any
    omitted field means "don't track this dimension", not "unlimited" --
    pass an explicit very-high number if that's genuinely the intent.
    """
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unknown provider {provider!r}. Supported: {', '.join(SUPPORTED_PROVIDERS)}"
        )
    if not model_name.strip():
        raise ValueError("Model name must not be empty.")
    if rpm is None and tpm is None and rpd is None:
        raise ValueError("At least one of rpm/tpm/rpd must be provided.")
    entry: dict = {}
    if rpm is not None:
        entry["rpm"] = int(rpm)
    if tpm is not None:
        entry["tpm"] = int(tpm)
    if rpd is not None:
        entry["rpd"] = int(rpd)
    data = _load_rate_limits_file()
    ns = _rate_limit_mode_key(multi_tenant)
    data[ns]["quota_overrides"].setdefault(provider, {})[model_name.strip()] = entry
    _save_rate_limits_file(data)


def delete_quota_override(provider: str, model_name: str, multi_tenant: bool) -> bool:
    """Remove a provider+model's RPM/TPM/RPD override from THIS MODE's namespace. Returns True if one existed."""
    data = _load_rate_limits_file()
    ns = _rate_limit_mode_key(multi_tenant)
    provider_overrides = data[ns]["quota_overrides"].get(provider, {})
    if model_name not in provider_overrides:
        return False
    del provider_overrides[model_name]
    _save_rate_limits_file(data)
    return True


def rate_limits_file_path() -> str:
    """Return the path to the rate_limits.json file (for display purposes).
    Both namespaces live in this one file -- see the module-level comment above."""
    return str(_rate_limits_path())
