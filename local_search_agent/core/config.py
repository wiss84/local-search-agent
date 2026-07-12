"""
SearchAgentConfig: all runtime configuration in one place.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from local_search_agent.core.constants import (
    DEFAULT_HOST,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MEILI_MASTER_KEY,
    DEFAULT_MEILI_URL,
    DEFAULT_PORT,
)

if TYPE_CHECKING:
    from local_search_agent.auth.identity import IdentityProvider


def _default_top_k() -> int:
    """Resolve the default top_k, respecting any advanced_settings.json override."""
    from local_search_agent.core.key_manager import get_effective_constants

    return get_effective_constants()["DEFAULT_TOP_K"]


def _default_max_iterations() -> int:
    """Resolve the default max_iterations, respecting any advanced_settings.json override."""
    from local_search_agent.core.key_manager import get_effective_constants

    return get_effective_constants()["DEFAULT_MAX_ITERATIONS"]


def _default_db_path() -> str:
    """
    Resolve the default SQLite database path.

    Stored in the same user-config directory as keys.json / models.json /
    settings.json so it survives pip upgrades and is independent of the
    current working directory.

      Windows : C:\\Users\\<name>\\AppData\\Roaming\\local-search-agent\\local_search_agent.db
      macOS   : ~/Library/Application Support/local-search-agent/local_search_agent.db
      Linux   : ~/.config/local-search-agent/local_search_agent.db
    """
    from platformdirs import user_config_dir

    config_dir = Path(user_config_dir("local-search-agent"))
    config_dir.mkdir(parents=True, exist_ok=True)
    return str(config_dir / "local_search_agent.db")


@dataclass
class SearchAgentConfig:
    """
    Central configuration object for the Local Search Agent framework.

    Parameters
    ----------
    document_dirs        : One or more local directories to ingest.
    workspace_name       : Logical name for this corpus / Meilisearch index.
    meilisearch_url      : URL of the Meilisearch instance.
    meili_master_key     : Meilisearch master API key.
    index_name           : Override Meilisearch index name (defaults to workspace_name).
    provider             : LLM provider: "google" | "ollama" | "openai" | "anthropic".
    api_key              : Provider API key.
                           Resolution order:
                             1. This argument (explicit — highest priority)
                             2. keys.json managed by `local-search config set-key`
                             3. Environment variable (GOOGLE_API_KEY, OPENAI_API_KEY, etc.)
                           Ollama always uses None (no key required).
    model_name           : Model name. Google default: "gemma-4-31b-it".
                           For Ollama, pass the model tag e.g. "mistral", "llama3.2".
    host / port          : FastAPI server bind address.
    top_k                : Search results per query.
    max_iterations       : Agent loop iteration cap.
    db_path              : SQLite database path.

    Semantic search
    --------------------------------
    enable_semantic        : Run ConceptCompiler (A) + StructuralParser (B) at ingest.
    enable_query_expansion : Expand queries with synonyms before searching (C).
    semantic_model         : Override LLM model for concept compilation.

    Watch mode
    --------------------------------
    enable_watch_mode    : Use filesystem events (watchdog) instead of polling to
                           trigger re-ingestion. Deprecated alternative: the polling
                           IncrementalSyncScheduler (start_incremental_scheduler).
    enrich_on_watch       : Whether watch-triggered re-ingests also run semantic
                           enrichment (only relevant if enable_semantic is True).
                           Defaults to True so watch-triggered docs stay consistent
                           with the rest of the workspace.

    Re-ranking
    --------------------------------
    enable_reranking             : Re-rank Meilisearch BM25 results with a local
                                   cross-encoder (flashrank) for better relevance.
    rerank_candidate_multiplier  : Fetch top_k * this many candidates from Meilisearch
                                   before re-ranking down to top_k.

    Access control
    --------------------------------
    enable_access_control  : Enforce Windows/LDAP access control on file endpoints.
    ldap_server            : LDAP server URL (e.g. "ldap://company.local").
    """

    # --- Document sources ---
    document_dirs: list[str] = field(default_factory=list)
    workspace_name: str = "default"

    # --- Meilisearch ---
    meilisearch_url: str = DEFAULT_MEILI_URL
    meili_master_key: str = DEFAULT_MEILI_MASTER_KEY
    index_name: Optional[str] = None

    # --- LLM Provider ---
    provider: str = "google"
    api_key: Optional[str] = None
    model_name: str = "gemma-4-31b-it"

    # --- File server ---
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT  # Dashboard / file-server API endpoint
    file_server_port: int = 8000  # Separate, fixed port for the text/docs file server

    # --- Agent behaviour ---
    top_k: int = field(default_factory=_default_top_k)
    max_iterations: int = field(default_factory=_default_max_iterations)
    max_retries: int = DEFAULT_MAX_RETRIES

    # --- Persistence ---
    db_path: str = field(default_factory=_default_db_path)

    # --- Semantic search ---
    enable_semantic: bool = False
    enable_query_expansion: bool = False
    semantic_model: Optional[str] = None

    # --- Access control ---
    enable_access_control: bool = False
    ldap_server: Optional[str] = None

    # --- Watch mode (replaces polling-based incremental scheduler) ---
    enable_watch_mode: bool = False
    enrich_on_watch: bool = True

    # --- Re-ranking ---
    enable_reranking: bool = True
    rerank_candidate_multiplier: int = 4

    # --- Multi-tenant RBAC (see docs/role_based_access_control.md) ---
    # None (the default) = single-user mode, completely unchanged:
    # AuthorizationMiddleware is never added to the app. Set this to any
    # IdentityProvider implementation to opt into multi-tenant enforcement.
    identity_provider: Optional["IdentityProvider"] = None

    # cookie_secure=True (default) marks the APIKeyIdentityProvider browser
    # session cookie Secure -- browsers then refuse to store or send it over
    # anything but HTTPS, EXCEPT on localhost/127.0.0.1, which browsers treat
    # as a secure context even over plain HTTP. That exception is why a
    # single-machine `local-search ui --multi-tenant` test works out of the
    # box, and why the exact same setup silently fails to log anyone in the
    # moment `--host` is a real LAN IP instead: the login POST succeeds
    # server-side (a session row really is created), but the browser drops
    # the Set-Cookie header entirely, so every subsequent request looks
    # unauthenticated again -- indistinguishable from "login did nothing"
    # without knowing this mechanism exists. Set False only for a
    # LAN-without-TLS deployment (e.g. `local-search ui --multi-tenant
    # --insecure-cookies`) where you understand the session cookie will
    # then be sent in cleartext over that network. Never set False for
    # anything reachable from the open internet -- terminate real TLS (see
    # docs/production-deployment.md's reverse-proxy section) instead.
    cookie_secure: bool = True

    # ------------------------------------------------------------------
    # Post-init
    # ------------------------------------------------------------------

    def __post_init__(self):
        if isinstance(self.document_dirs, str):
            self.document_dirs = [self.document_dirs]
        if self.api_key is None:
            from local_search_agent.core.key_manager import resolve_key

            self.api_key = resolve_key(self.provider)
        if self.index_name is None:
            self.index_name = self.workspace_name
        # Load semantic settings from settings.json if not explicitly set
        # (explicit means the caller passed a non-default value)
        if not any([self.enable_semantic, self.enable_query_expansion]):
            from local_search_agent.core.key_manager import get_semantic_settings

            s = get_semantic_settings()
            self.enable_semantic = s["enable_semantic"]
            self.enable_query_expansion = s["enable_query_expansion"]
            # Only set semantic_model override if not already explicitly provided
            if s.get("semantic_model") and not self.semantic_model:
                self.semantic_model = s["semantic_model"]
            # Note: semantic_provider is stored in settings but never overwrites
            # the main config.provider — it is only used by _get_enricher directly
        # Load watch-mode settings from settings.json if left at defaults
        if not self.enable_watch_mode and self.enrich_on_watch:
            from local_search_agent.core.key_manager import get_watch_mode_settings

            w = get_watch_mode_settings()
            self.enable_watch_mode = w["enable_watch_mode"]
            self.enrich_on_watch = w["enrich_on_watch"]

        # Load re-ranking settings from settings.json if left at defaults
        if self.enable_reranking and self.rerank_candidate_multiplier == 4:
            from local_search_agent.core.key_manager import get_reranking_settings

            r = get_reranking_settings()
            self.enable_reranking = r["enable_reranking"]
            self.rerank_candidate_multiplier = r["rerank_candidate_multiplier"]

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def server_base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def file_server_base_url(self) -> str:
        return f"http://{self.host}:{self.file_server_port}"

    def text_url(self, doc_id: str) -> str:
        return f"{self.file_server_base_url}/text/{doc_id}"

    def docs_url(self, doc_id: str) -> str:
        return f"{self.file_server_base_url}/docs/{doc_id}"

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        # asdict() deep-copies every field recursively. identity_provider may
        # hold a non-deepcopy-safe object (e.g. threading.Lock inside AuthDB),
        # so it must be excluded *before* asdict() runs, not popped after —
        # popping after is too late, the deepcopy already blew up by then.
        saved_provider = self.identity_provider
        self.identity_provider = None
        try:
            d = asdict(self)
        finally:
            self.identity_provider = saved_provider
        d.pop("api_key", None)
        d.pop("identity_provider", None)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "SearchAgentConfig":
        return cls(**data)

    def validate(self) -> None:
        if self.provider not in ("google", "ollama", "openai", "anthropic"):
            raise ValueError(f"Unknown provider: {self.provider!r}.")
        if self.provider != "ollama" and not self.api_key:
            import logging as _logging

            _logging.getLogger(__name__).warning(
                "No API key found for provider %r. "
                "Run: local-search config set-key --provider %s --key YOUR_KEY",
                self.provider,
                self.provider,
            )
        for d in self.document_dirs:
            if not os.path.isdir(d):
                raise ValueError(f"document_dir does not exist: {d!r}")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
