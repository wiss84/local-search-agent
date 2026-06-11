"""
SearchAgentFramework: the main entry point for users of this library.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import uvicorn

from local_search_agent.core.config import SearchAgentConfig
from local_search_agent.server.fastapi_app import build_app
from local_search_agent.workspace.metadata_db import MetadataDB
from local_search_agent.workspace.workspace_manager import WorkspaceManager

logger = logging.getLogger(__name__)


class SearchAgentFramework:
    """
    Top-level orchestrator for the Local Search Agent framework.

    Typical usage
    -------------
    ```python
    from local_search_agent import SearchAgentFramework, SearchAgentConfig

    config = SearchAgentConfig(
        document_dirs=["C:/shares/company_data"],
        workspace_name="workspace_name",
        provider="google",
        api_key="YOUR_KEY",  # or omit to auto-resolve from saved keys / env var
        model_name="gemma-4-31b-it",
    )
    framework = SearchAgentFramework(config)
    framework.ingest_and_index()
    framework.start_file_server()
    framework.start_incremental_scheduler(interval_minutes=15)
    response = framework.query("What was AWS spend in Q3 2024?")
    print(response)
    ```
    """

    def __init__(self, config: SearchAgentConfig):
        self.config = config
        self._server_thread: Optional[threading.Thread] = None
        self._server: Optional[uvicorn.Server] = None
        self._workspace_manager = WorkspaceManager(db_path=config.db_path)
        self._metadata_db = MetadataDB(db_path=config.db_path)
        self._meili_client = None
        self._scheduler = None
        self._meili_manager = None
        # Activate LangSmith tracing if credentials are saved
        from local_search_agent.core.key_manager import apply_langsmith_env

        if apply_langsmith_env():
            logger.info("LangSmith tracing activated.")
        logger.info("SearchAgentFramework initialised (workspace=%r)", config.workspace_name)

    # ------------------------------------------------------------------
    # Lazy helpers
    # ------------------------------------------------------------------

    def _ensure_meilisearch(self) -> None:
        """Start the local Meilisearch binary if it is not already running."""
        if self._meili_manager is None:
            from local_search_agent.core.meilisearch_manager import MeilisearchManager

            self._meili_manager = MeilisearchManager(
                url=self.config.meilisearch_url,
                master_key=self.config.meili_master_key,
            )
        self._meili_manager.start()

    def _get_meili_client(self):
        self._ensure_meilisearch()
        if self._meili_client is None:
            from local_search_agent.search.meilisearch_client import MeilisearchClient

            self._meili_client = MeilisearchClient(
                url=self.config.meilisearch_url,
                api_key=self.config.meili_master_key,
                index_name=self.config.index_name,
            )
        return self._meili_client

    def _get_scheduler(self):
        if self._scheduler is None:
            from local_search_agent.scheduler.incremental_sync import IncrementalSyncScheduler

            self._scheduler = IncrementalSyncScheduler(
                workspace_manager=self._workspace_manager,
                metadata_db=self._metadata_db,
                interval_minutes=15,
            )
        return self._scheduler

    # ------------------------------------------------------------------
    # Phase 1: File Server
    # ------------------------------------------------------------------

    def start_file_server(self, port: Optional[int] = None, block: bool = False) -> None:
        """Start the FastAPI file server in background (or blocking if block=True)."""
        port = port or self.config.port
        app = build_app(
            config=self.config,
            workspace_manager=self._workspace_manager,
            metadata_db=self._metadata_db,
        )

        uv_config = uvicorn.Config(
            app=app,
            host=self.config.host,
            port=port,
            log_level="info",
        )
        self._server = uvicorn.Server(uv_config)

        if block:
            logger.info("Starting file server on %s:%d (blocking)", self.config.host, port)
            self._server.run()
        else:
            self._server_thread = threading.Thread(
                target=self._server.run,
                daemon=True,
                name="local-search-file-server",
            )
            self._server_thread.start()
            logger.info("File server started in background on %s:%d", self.config.host, port)

    def stop_file_server(self) -> None:
        if self._server:
            self._server.should_exit = True
            logger.info("File server stop requested.")

    # ------------------------------------------------------------------
    # Phase 2: Ingestion
    # ------------------------------------------------------------------

    def ingest_and_index(self, force: bool = False):
        """Parse all documents in config.document_dirs and index into Meilisearch."""
        from local_search_agent.ingestion.pipeline import IngestionPipeline

        for doc_dir in self.config.document_dirs:
            self._workspace_manager.create_workspace(
                name=self.config.workspace_name,
                document_dir=doc_dir,
            )

        # Ensure sync job record exists for this workspace
        self._metadata_db.upsert_sync_job(workspace=self.config.workspace_name)

        pipeline = IngestionPipeline(
            config=self.config,
            workspace_manager=self._workspace_manager,
            meili_client=self._get_meili_client(),
        )
        stats = pipeline.run(force=force)
        logger.info("ingest_and_index complete: %s", stats)

        # Update health/sync tracking so local-search health shows correct doc counts
        self._metadata_db.set_sync_complete(
            workspace=self.config.workspace_name,
            doc_count=stats.indexed + stats.skipped,
            error_count=stats.failed,
            next_sync_at=None,  # No recurring scheduler outside serve --scheduler
            last_error=stats.errors[0] if stats.errors else None,
        )
        return stats

    # ------------------------------------------------------------------
    # Phase 3: Agent Query
    # ------------------------------------------------------------------

    def _resolve_document_dirs(self, workspace: Optional[str] = None) -> None:
        """
        Populate config.document_dirs from the workspace DB when it is empty.

        This ensures the system prompt always shows the correct document paths
        regardless of how the framework was constructed (CLI, Python API, or UI).
        When document_dirs was passed explicitly it is left untouched.
        """
        if self.config.document_dirs:
            return
        name = workspace or self.config.workspace_name
        ws = self._workspace_manager.get_workspace(name)
        if ws and ws.get("document_dir"):
            self.config.document_dirs = [ws["document_dir"]]
            logger.debug(
                "Resolved document_dirs from workspace DB: %r → %r",
                name,
                self.config.document_dirs,
            )

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        workspace: Optional[str] = None,
    ) -> dict:
        """Ask the agent a question against the indexed documents."""
        from local_search_agent.agent.agent import LocalSearchAgent

        self._resolve_document_dirs(workspace)
        agent = LocalSearchAgent(
            config=self.config,
            meili_client=self._get_meili_client(),
        )
        return agent.query(question=question, top_k=top_k, workspace=workspace)

    def query_raw_state(self, question: str, workspace: Optional[str] = None) -> dict:
        """Run the agent and return the raw LangGraph state (messages list)."""
        from local_search_agent.agent.agent import LocalSearchAgent

        self._resolve_document_dirs(workspace)
        agent = LocalSearchAgent(
            config=self.config,
            meili_client=self._get_meili_client(),
        )
        return agent.query_raw_state(question=question, workspace=workspace)

    # ------------------------------------------------------------------
    # Phase 4: Multi-workspace management
    # ------------------------------------------------------------------

    def create_workspace(self, name: str, document_dir: str) -> None:
        """Register a new named workspace pointing to a document directory."""
        self._workspace_manager.create_workspace(name=name, document_dir=document_dir)
        self._metadata_db.upsert_sync_job(workspace=name)
        logger.info("Workspace created: %r → %r", name, document_dir)

    def list_workspaces(self) -> list[dict]:
        """Return all registered workspaces."""
        return self._workspace_manager.list_workspaces()

    def delete_workspace(self, name: str, wipe_index: bool = False) -> None:
        """
        Remove a workspace registration.

        Parameters
        ----------
        name       : Workspace name to delete.
        wipe_index : If True, also delete all documents from the Meilisearch index.
                     Does not delete the index itself, only its documents.
        """
        if wipe_index:
            try:
                meili = self._get_meili_client()
                meili.delete_index()
                logger.info("Meilisearch index %r deleted.", name)
            except Exception as e:
                logger.warning("Could not delete Meilisearch index %r: %s", name, e)
        self._workspace_manager.delete_workspace(name)
        if self._scheduler:
            self._scheduler.remove_workspace(name)
        logger.info("Workspace deleted: %r (wipe_index=%s)", name, wipe_index)

    def ingest_workspace(self, workspace_name: str, force: bool = False):
        """
        Run ingestion for a specific named workspace (not just the default one in config).

        Useful when managing multiple workspaces without recreating the framework.
        """
        from local_search_agent.core.config import SearchAgentConfig
        from local_search_agent.ingestion.pipeline import IngestionPipeline
        from local_search_agent.search.meilisearch_client import MeilisearchClient

        ws = self._workspace_manager.get_workspace(workspace_name)
        if ws is None:
            raise ValueError(
                f"Workspace {workspace_name!r} is not registered. Call create_workspace() first."
            )

        # Build a per-workspace config derived from the main config
        ws_config = SearchAgentConfig(
            document_dirs=[ws["document_dir"]],
            workspace_name=workspace_name,
            meilisearch_url=self.config.meilisearch_url,
            meili_master_key=self.config.meili_master_key,
            index_name=workspace_name,  # Each workspace = its own Meilisearch index
            provider=self.config.provider,
            api_key=self.config.api_key,
            model_name=self.config.model_name,
            db_path=self.config.db_path,
            top_k=self.config.top_k,
            max_iterations=self.config.max_iterations,
        )

        meili_client = MeilisearchClient(
            url=ws_config.meilisearch_url,
            api_key=ws_config.meili_master_key,
            index_name=workspace_name,
        )

        pipeline = IngestionPipeline(
            config=ws_config,
            workspace_manager=self._workspace_manager,
            meili_client=meili_client,
        )
        stats = pipeline.run(force=force)
        logger.info("ingest_workspace(%r) complete: %s", workspace_name, stats)
        return stats

    def wipe_and_reingest(self, workspace_name: Optional[str] = None) -> None:
        """
        Clear all indexed documents for a workspace and run a full re-ingest.

        Wipes the Meilisearch index and SQLite document records, then
        re-ingests from scratch.  Equivalent to the UI's wipe button.

        Parameters
        ----------
        workspace_name : Workspace to wipe. Defaults to config.workspace_name.
        """
        import sqlite3

        name = workspace_name or self.config.workspace_name

        # 1. Delete Meilisearch index (will be recreated on next ingest)
        try:
            meili = self._get_meili_client()
            meili.delete_index()
            logger.info("Meilisearch index %r deleted.", name)
        except Exception as e:
            logger.warning("Could not delete Meilisearch index %r: %s", name, e)

        # 2. Wipe SQLite document records for this workspace
        conn = sqlite3.connect(self.config.db_path)
        try:
            cur = conn.execute("DELETE FROM documents WHERE workspace = ?", (name,))
            conn.commit()
            logger.info("Wiped %d SQLite document records for workspace %r.", cur.rowcount, name)
        finally:
            conn.close()

        # 3. Force full re-ingest
        logger.info("Starting force re-ingest for workspace %r.", name)
        if name == self.config.workspace_name:
            return self.ingest_and_index(force=True)
        else:
            return self.ingest_workspace(name, force=True)

    # ------------------------------------------------------------------
    # Phase 4: Incremental scheduler
    # ------------------------------------------------------------------

    def start_incremental_scheduler(self, interval_minutes: int = 15) -> None:
        """
        Start the APScheduler background job for incremental re-ingestion.

        Registers the current framework config's workspace and starts
        the scheduler. Additional workspaces can be added via
        add_workspace_to_scheduler().

        Parameters
        ----------
        interval_minutes : How often to check for changed files (default 15).
        """
        scheduler = self._get_scheduler()
        scheduler._default_interval = interval_minutes

        # Start the scheduler first so add_workspace can register real APScheduler jobs
        scheduler.start()

        # Register the primary workspace
        scheduler.add_workspace(self.config, interval_minutes=interval_minutes)

        # Also register any other workspaces already in the DB
        for ws in self._workspace_manager.list_workspaces():
            if ws["name"] != self.config.workspace_name:
                # Build a minimal config for this workspace
                from local_search_agent.core.config import SearchAgentConfig

                ws_config = SearchAgentConfig(
                    document_dirs=[ws["document_dir"]],
                    workspace_name=ws["name"],
                    meilisearch_url=self.config.meilisearch_url,
                    meili_master_key=self.config.meili_master_key,
                    index_name=ws["name"],
                    provider=self.config.provider,
                    api_key=self.config.api_key,
                    model_name=self.config.model_name,
                    db_path=self.config.db_path,
                )
                scheduler.add_workspace(ws_config, interval_minutes=interval_minutes)

        logger.info(
            "Incremental scheduler started (interval=%dm, workspaces=%d).",
            interval_minutes,
            len(scheduler._workspace_configs),
        )

    def stop_incremental_scheduler(self) -> None:
        """Gracefully stop the incremental scheduler."""
        if self._scheduler:
            self._scheduler.stop()

    def add_workspace_to_scheduler(
        self,
        workspace_name: str,
        interval_minutes: Optional[int] = None,
    ) -> None:
        """
        Add an already-registered workspace to the incremental scheduler.

        Parameters
        ----------
        workspace_name   : Name of an existing workspace (must be in WorkspaceManager).
        interval_minutes : Sync interval override. Defaults to scheduler's default.
        """
        ws = self._workspace_manager.get_workspace(workspace_name)
        if ws is None:
            raise ValueError(
                f"Workspace {workspace_name!r} not found. Call create_workspace() first."
            )

        from local_search_agent.core.config import SearchAgentConfig

        ws_config = SearchAgentConfig(
            document_dirs=[ws["document_dir"]],
            workspace_name=workspace_name,
            meilisearch_url=self.config.meilisearch_url,
            meili_master_key=self.config.meili_master_key,
            index_name=workspace_name,
            provider=self.config.provider,
            api_key=self.config.api_key,
            model_name=self.config.model_name,
            db_path=self.config.db_path,
        )
        self._get_scheduler().add_workspace(ws_config, interval_minutes=interval_minutes)
        logger.info("Workspace %r added to scheduler.", workspace_name)

    def trigger_sync_now(self, workspace_name: Optional[str] = None) -> None:
        """Force an immediate sync for a workspace outside the normal schedule."""
        name = workspace_name or self.config.workspace_name
        self._get_scheduler().trigger_now(name)

    # ------------------------------------------------------------------
    # Phase 4: Health monitoring
    # ------------------------------------------------------------------

    def get_index_health(self):
        """
        Return an IndexHealthSummary across all registered workspaces.

        Returns
        -------
        IndexHealthSummary with per-workspace status, doc counts, and staleness info.
        """
        from local_search_agent.scheduler.monitor import IndexMonitor

        monitor = IndexMonitor(self._metadata_db)
        return monitor.get_health_summary()

    def get_scheduler_status(self) -> dict:
        """Return current scheduler status (running, jobs, next run times)."""
        if self._scheduler is None:
            return {"running": False, "registered_workspaces": [], "scheduled_jobs": []}
        return self._scheduler.get_status()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_semantic_settings(
        self,
        enable_semantic: bool,
        enable_query_expansion: bool,
    ) -> None:
        """
        Update semantic feature flags at runtime and persist them to settings.json.

        Settings are written to the user config directory so they are shared
        across CLI, UI, and future Python API sessions.

        Parameters
        ----------
        enable_semantic        : Run ConceptCompiler + StructuralParser at ingest.
        enable_query_expansion : Expand queries with synonyms at search time.

        Example
        -------
        ```python
        framework.set_semantic_settings(
            enable_semantic=True,
            enable_query_expansion=True,
        )
        ```
        """
        from local_search_agent.core.key_manager import set_all_semantic_settings

        self.config.enable_semantic = enable_semantic
        self.config.enable_query_expansion = enable_query_expansion

        set_all_semantic_settings(
            enable_semantic=enable_semantic,
            enable_query_expansion=enable_query_expansion,
        )

        logger.info(
            "Semantic settings updated: enable_semantic=%s, enable_query_expansion=%s",
            enable_semantic,
            enable_query_expansion,
        )

    def get_semantic_settings(self) -> dict[str, bool]:
        """
        Return the current semantic feature flag settings.

        Returns
        -------
        dict with keys: ``enable_semantic``, ``enable_query_expansion``
        """
        return {
            "enable_semantic": self.config.enable_semantic,
            "enable_query_expansion": self.config.enable_query_expansion,
        }

    def get_advanced_settings(self) -> dict:
        """
        Return the effective ingestion / search constants, merging any
        user overrides on top of the compiled-in defaults from constants.py.

        The returned dict always contains every key; overridden keys are
        the user-set values, non-overridden keys are the constants.py defaults.

        Returns
        -------
        dict with keys matching ``_ADVANCED_SETTING_KEYS`` in key_manager.py:
        ``CHUNK_MIN_CHARS``, ``CHUNK_TARGET_CHARS``, ``CHUNK_MAX_CHARS``,
        ``CHUNK_OVERLAP_CHARS``, ``TABLE_ROWS_PER_CHUNK``,
        ``PDF_PAGES_PER_BATCH``, ``PDF_SPLIT_THRESHOLD``,
        ``PDF_FALLBACK_PAGES_PER_BATCH``, ``DOCX_CHAR_SPLIT_THRESHOLD``,
        ``TESSERACT_FALLBACK_MIN_CHARS``, ``DEFAULT_TOP_K``,
        ``DEFAULT_MAX_ITERATIONS``, ``SNIPPET_CONTEXT_CHARS``.

        Example
        -------
        ```python
        settings = framework.get_advanced_settings()
        print(settings["PDF_PAGES_PER_BATCH"])   # 20 (or user override)
        ```
        """
        from local_search_agent.core.key_manager import get_effective_constants

        return get_effective_constants()

    def set_advanced_settings(self, overrides: dict) -> dict:
        """
        Persist ingestion / search constant overrides to ``advanced_settings.json``
        in the user config directory.

        Overrides take effect on the **next** ingest run.  Pass an empty dict
        to reset everything back to the compiled-in defaults.

        Unknown keys and values that cannot be coerced to the expected numeric
        type are silently ignored.

        Parameters
        ----------
        overrides : dict
            Mapping of constant name → new value.  Valid keys::

                CHUNK_MIN_CHARS, CHUNK_TARGET_CHARS, CHUNK_MAX_CHARS,
                CHUNK_OVERLAP_CHARS, TABLE_ROWS_PER_CHUNK,
                PDF_PAGES_PER_BATCH, PDF_SPLIT_THRESHOLD,
                PDF_FALLBACK_PAGES_PER_BATCH, DOCX_CHAR_SPLIT_THRESHOLD,
                TESSERACT_FALLBACK_MIN_CHARS, DEFAULT_TOP_K,
                DEFAULT_MAX_ITERATIONS, SNIPPET_CONTEXT_CHARS

        Returns
        -------
        dict — the effective constants after applying the overrides (same as
        ``get_advanced_settings()``).

        Examples
        --------
        ```python
        # Lower batch size for machines with limited RAM
        framework.set_advanced_settings({
            "PDF_PAGES_PER_BATCH": 10,
            "CHUNK_TARGET_CHARS": 8000,
        })
        framework.ingest_and_index(force=True)

        # Reset everything back to defaults
        framework.set_advanced_settings({})
        ```
        """
        from local_search_agent.core.key_manager import (
            get_effective_constants,
        )
        from local_search_agent.core.key_manager import (
            set_advanced_settings as _set,
        )

        _set(overrides)
        effective = get_effective_constants()
        logger.info("Advanced settings updated: %s", overrides)
        return effective
