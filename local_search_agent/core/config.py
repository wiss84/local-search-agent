"""
SearchAgentConfig: all runtime configuration in one place.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Optional

from local_search_agent.core.constants import (
    DEFAULT_HOST,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MEILI_MASTER_KEY,
    DEFAULT_MEILI_URL,
    DEFAULT_PORT,
    DEFAULT_TOP_K,
)


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

    Semantic search (Experimental)
    --------------------------------
    enable_semantic        : Run ConceptCompiler (A) + StructuralParser (B) at ingest.
    enable_query_expansion : Expand queries with synonyms before searching (C).
    enable_link_graph      : Build cross-document same_topic links at ingest.
    semantic_model         : Override LLM model for concept compilation.

    Access control (Experimental)
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
    top_k: int = DEFAULT_TOP_K
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    max_retries: int = DEFAULT_MAX_RETRIES

    # --- Persistence ---
    db_path: str = "local_search_agent.db"

    # --- Phase 5: Semantic search ---
    enable_semantic: bool = False
    enable_query_expansion: bool = False
    enable_link_graph: bool = False
    semantic_model: Optional[str] = None

    # --- Phase 5: Access control ---
    enable_access_control: bool = False
    ldap_server: Optional[str] = None

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
        if not any([self.enable_semantic, self.enable_query_expansion, self.enable_link_graph]):
            from local_search_agent.core.key_manager import get_semantic_settings

            s = get_semantic_settings()
            self.enable_semantic = s["enable_semantic"]
            self.enable_query_expansion = s["enable_query_expansion"]
            self.enable_link_graph = s["enable_link_graph"]

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
        d = asdict(self)
        d.pop("api_key", None)
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
