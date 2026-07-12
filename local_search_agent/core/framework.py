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
from local_search_agent.workspace.auth_db import AuthDB
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
        self._auth_db = AuthDB(db_path=config.db_path)
        self._meili_client = None
        self._scheduler = None
        self._watcher = None
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
        """
        DEPRECATED: the polling-based IncrementalSyncScheduler is kept for
        backward compatibility. New code should use start_watch_mode() /
        _get_watcher() instead, which reacts to filesystem events directly
        rather than polling on a fixed interval.
        """
        if self._scheduler is None:
            from local_search_agent.scheduler.incremental_sync import IncrementalSyncScheduler

            self._scheduler = IncrementalSyncScheduler(
                workspace_manager=self._workspace_manager,
                metadata_db=self._metadata_db,
                interval_minutes=15,
            )
        return self._scheduler

    def _get_watcher(self):
        if self._watcher is None:
            from local_search_agent.scheduler.watch_mode import WorkspaceWatcher

            self._watcher = WorkspaceWatcher(
                workspace_manager=self._workspace_manager,
                metadata_db=self._metadata_db,
            )
        return self._watcher

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
        if self._watcher:
            self._watcher.remove_workspace(name)
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
        DEPRECATED (polling-based, use 'watch'): Start the APScheduler background
        job for incremental re-ingestion.

        Kept for backward compatibility. New code should prefer
        start_watch_mode(), which reacts to filesystem changes via watchdog
        instead of polling on a fixed interval.

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
        """DEPRECATED (polling-based, use 'watch'): Gracefully stop the incremental scheduler."""
        if self._scheduler:
            self._scheduler.stop()

    # ------------------------------------------------------------------
    # Watch mode (filesystem-event-driven, replaces the polling scheduler)
    # ------------------------------------------------------------------

    def start_watch_mode(self) -> None:
        """
        Start filesystem-event-driven watching for incremental re-ingestion.

        Reacts to file creates/modifies/deletes within config.document_dirs
        almost immediately (after a short debounce window), instead of
        waiting for a fixed polling interval like start_incremental_scheduler().

        Registers the current framework config's workspace, plus any other
        workspaces already in the DB, mirroring start_incremental_scheduler().
        Whether watch-triggered syncs run semantic enrichment is controlled
        by config.enrich_on_watch.

        Example
        -------
        ```python
        framework.start_watch_mode()
        ```
        """
        watcher = self._get_watcher()
        watcher.start()

        # Register the primary workspace
        watcher.add_workspace(self.config)

        # Also register any other workspaces already in the DB
        for ws in self._workspace_manager.list_workspaces():
            if ws["name"] != self.config.workspace_name:
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
                    enrich_on_watch=self.config.enrich_on_watch,
                )
                watcher.add_workspace(ws_config)

        logger.info(
            "Watch mode started (workspaces=%d, enrich_on_watch=%s).",
            len(watcher._workspace_configs),
            self.config.enrich_on_watch,
        )

    def stop_watch_mode(self) -> None:
        """Gracefully stop watch mode and its filesystem observer thread."""
        if self._watcher:
            self._watcher.stop()

    def add_workspace_to_watch_mode(self, workspace_name: str) -> None:
        """
        Add an already-registered workspace to watch mode.

        Parameters
        ----------
        workspace_name : Name of an existing workspace (must be in WorkspaceManager).
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
            enrich_on_watch=self.config.enrich_on_watch,
        )
        self._get_watcher().add_workspace(ws_config)
        logger.info("Workspace %r added to watch mode.", workspace_name)

    def get_watch_mode_status(self) -> dict:
        """Return current watch-mode status (running, watched workspaces/directories)."""
        if self._watcher is None:
            return {"running": False, "registered_workspaces": [], "watched_directories": {}}
        return self._watcher.get_status()

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
        """
        Force an immediate sync for a workspace outside the normal schedule.

        If watch mode is running and the workspace is registered with it,
        the watcher's trigger_now() is used (debounce-bypassing). Otherwise
        falls back to the polling scheduler's trigger_now() for backward
        compatibility.
        """
        name = workspace_name or self.config.workspace_name
        if self._watcher is not None and name in self._watcher._workspace_configs:
            self._watcher.trigger_now(name)
        else:
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

    def set_watch_mode_settings(self, enable_watch_mode: bool, enrich_on_watch: bool) -> None:
        """
        Update watch-mode feature flags at runtime and persist them to settings.json.

        Parameters
        ----------
        enable_watch_mode : Whether watch mode should be used (informational here;
                            actually starting/stopping it is done via
                            start_watch_mode()/stop_watch_mode()).
        enrich_on_watch    : Whether watch-triggered re-ingests also run semantic
                            enrichment (only relevant if enable_semantic is True).

        Example
        -------
        ```python
        framework.set_watch_mode_settings(enable_watch_mode=True, enrich_on_watch=False)
        ```
        """
        from local_search_agent.core.key_manager import set_all_watch_mode_settings

        self.config.enable_watch_mode = enable_watch_mode
        self.config.enrich_on_watch = enrich_on_watch

        set_all_watch_mode_settings(
            enable_watch_mode=enable_watch_mode,
            enrich_on_watch=enrich_on_watch,
        )

        logger.info(
            "Watch mode settings updated: enable_watch_mode=%s, enrich_on_watch=%s",
            enable_watch_mode,
            enrich_on_watch,
        )

    def get_watch_mode_settings(self) -> dict[str, bool]:
        """
        Return the current watch-mode feature flag settings.

        Returns
        -------
        dict with keys: ``enable_watch_mode``, ``enrich_on_watch``
        """
        return {
            "enable_watch_mode": self.config.enable_watch_mode,
            "enrich_on_watch": self.config.enrich_on_watch,
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

    # ------------------------------------------------------------------
    # Multi-tenant RBAC: workspace_members grants
    # (see upcoming_features/04-multi-tenant-rbac-mode.md)
    # ------------------------------------------------------------------

    def grant_workspace_access(
        self,
        workspaces: list[str],
        subject: str,
        role: str,
        granted_by: str,
    ) -> None:
        """
        Grant `subject` a role across one or more workspaces in a single
        atomic call — either every workspace gets the grant or none do.

        Parameters
        ----------
        workspaces : Workspace names to grant access to.
        subject    : Stable identity (e.g. email) — see Identity.subject.
        role       : 'member' | 'admin'.
        granted_by : Identity.subject of whoever is performing the grant
                    (recorded for audit purposes in workspace_members.granted_by).

        Example
        -------
        ```python
        framework.grant_workspace_access(
            workspaces=["finance", "marketing"],
            subject="alice@acme.com",
            role="member",
            granted_by="admin@acme.com",
        )
        ```
        """
        self._auth_db.grant_access_bulk(
            workspaces=workspaces, subject=subject, role=role, granted_by=granted_by
        )

    def revoke_workspace_access(self, subject: str, workspaces: Optional[list[str]] = None) -> int:
        """
        Revoke `subject`'s access. If `workspaces` is None, revokes every
        grant for that subject; otherwise revokes only the listed workspaces.

        Returns the number of grants removed.
        """
        return self._auth_db.revoke_access(subject=subject, workspaces=workspaces)

    def list_workspace_access(
        self, subject: Optional[str] = None, workspace: Optional[str] = None
    ) -> list[dict]:
        """List grants, optionally filtered by subject and/or workspace."""
        return self._auth_db.list_access(subject=subject, workspace=workspace)

    def get_workspace_role(self, subject: str, workspace: str) -> Optional[str]:
        """Return `subject`'s role in `workspace`, or None if they have no grant (fail-closed)."""
        return self._auth_db.get_role(subject=subject, workspace=workspace)

    # ------------------------------------------------------------------
    # Multi-tenant RBAC: API key management (APIKeyIdentityProvider)
    # ------------------------------------------------------------------

    def _get_api_key_provider(self):
        from local_search_agent.auth.api_key_provider import APIKeyIdentityProvider

        return APIKeyIdentityProvider(self._auth_db)

    def create_api_key(
        self,
        subject: str,
        created_by: str,
        display_name: str = "",
        is_superadmin: bool = False,
    ) -> tuple[str, str]:
        """
        Generate a new API key for `subject` (APIKeyIdentityProvider mode).

        Returns (key_id, raw_key). raw_key is shown exactly once here —
        only its argon2 hash is persisted. The caller (CLI/admin API) is
        responsible for displaying it to the operator and never logging it.
        """
        return self._get_api_key_provider().create_key(
            subject=subject,
            created_by=created_by,
            display_name=display_name,
            is_superadmin=is_superadmin,
        )

    def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key by its key_id. Returns True if an active key was found and revoked."""
        return self._get_api_key_provider().revoke_key(key_id)

    def list_api_keys(self, subject: Optional[str] = None) -> list[dict]:
        """List API key metadata (never the raw key or its hash), optionally filtered by subject."""
        return self._get_api_key_provider().list_keys(subject=subject)

    # ------------------------------------------------------------------
    # Model / Provider Access Control (which provider+model combinations
    # each role may use for their own queries -- a cost control, not a
    # workspace permission. See docs/role_based_access_control.md.)
    # ------------------------------------------------------------------

    def grant_model_access(
        self, role: str, provider: str, model_name: str, granted_by: str
    ) -> None:
        """
        Grant `role` ('member' or 'admin') permission to use `provider`/
        `model_name` for their own queries. A role with nothing granted
        has access to nothing (fail-closed) -- grant at least one model to
        each role before anyone tries to query. Superadmin always has
        access to every configured model and is never affected by this.

        Example
        -------
        ```python
        framework.grant_model_access("member", "google", "gemma-4-31b-it", granted_by="admin@acme.com")
        ```
        """
        self._auth_db.grant_model_access(
            role=role, provider=provider, model_name=model_name, granted_by=granted_by
        )

    def revoke_model_access(self, role: str, provider: str, model_name: str) -> bool:
        """Remove a single role/provider/model allow-list entry. Returns True if one existed."""
        return self._auth_db.revoke_model_access(
            role=role, provider=provider, model_name=model_name
        )

    def list_model_access(self, role: Optional[str] = None) -> list[dict]:
        """List model-access grant rows, optionally filtered by role ('member'/'admin')."""
        return self._auth_db.list_model_access(role=role)

    # ------------------------------------------------------------------
    # Rate Limits & Concurrency (deployment-wide LLM call caps and
    # RPM/TPM/RPD quota tracking. See docs/role_based_access_control.md.)
    #
    # multi_tenant : Single-user and multi-tenant settings are stored in
    # independent namespaces in the same rate_limits.json -- pass
    # `config.identity_provider is not None` for the natural default that
    # matches whichever mode this framework instance is actually running
    # in, or an explicit bool to manage the other namespace deliberately
    # (e.g. provisioning a multi-tenant deployment's limits from a
    # single-user-mode script before it's ever run with --multi-tenant).
    # ------------------------------------------------------------------

    def set_concurrency_limit(self, provider: str, limit: int, multi_tenant: bool) -> None:
        """
        Cap the max number of simultaneous in-flight LLM calls for a
        provider. For Ollama this is the framework-side mirror of
        Ollama's own OLLAMA_NUM_PARALLEL -- set it based on your actual
        hardware's real capacity, this framework can't introspect your
        VRAM itself. Takes effect on the next call for that provider (no
        restart needed).

        Example
        -------
        ```python
        framework.set_concurrency_limit("ollama", 2, multi_tenant=False)
        ```
        """
        from local_search_agent.agent.rate_limit_handler import reset_shared_rate_limit_handlers
        from local_search_agent.core.key_manager import set_concurrency_limit as _set

        _set(provider, limit, multi_tenant)
        reset_shared_rate_limit_handlers()

    def delete_concurrency_limit(self, provider: str, multi_tenant: bool) -> bool:
        """Remove a provider's concurrency cap (reverts to unbounded). Returns True if one existed."""
        from local_search_agent.agent.rate_limit_handler import reset_shared_rate_limit_handlers
        from local_search_agent.core.key_manager import delete_concurrency_limit as _delete

        result = _delete(provider, multi_tenant)
        reset_shared_rate_limit_handlers()
        return result

    def get_concurrency_limits(self, multi_tenant: bool) -> dict[str, int]:
        """Return {provider: max_simultaneous_llm_calls} for the given mode's namespace."""
        from local_search_agent.core.key_manager import get_concurrency_limits as _get

        return _get(multi_tenant)

    def set_quota_override(
        self,
        provider: str,
        model_name: str,
        multi_tenant: bool,
        rpm: Optional[int] = None,
        tpm: Optional[int] = None,
        rpd: Optional[int] = None,
    ) -> None:
        """
        Set (or replace) the RPM/TPM/RPD override for one provider+model.
        Google gets auto-detected free-tier limits by default (overridable
        here); every other provider tracks nothing at all until an
        override is set here -- this is the only way OpenAI/Anthropic/
        Ollama get real sliding-window quota tracking rather than blind
        retry-on-error. At least one of rpm/tpm/rpd is required; an
        omitted dimension means "don't track this", not "unlimited".

        Example
        -------
        ```python
        # A paid-tier OpenAI account with real, much-higher limits than the free tier
        framework.set_quota_override("openai", "gpt-5", multi_tenant=True, rpm=500, tpm=2_000_000)
        ```
        """
        from local_search_agent.agent.rate_limit_handler import reset_shared_rate_limit_handlers
        from local_search_agent.core.key_manager import set_quota_override as _set

        _set(provider, model_name, multi_tenant, rpm=rpm, tpm=tpm, rpd=rpd)
        reset_shared_rate_limit_handlers()

    def delete_quota_override(self, provider: str, model_name: str, multi_tenant: bool) -> bool:
        """Remove a provider+model's RPM/TPM/RPD override. Returns True if one existed."""
        from local_search_agent.agent.rate_limit_handler import reset_shared_rate_limit_handlers
        from local_search_agent.core.key_manager import delete_quota_override as _delete

        result = _delete(provider, model_name, multi_tenant)
        reset_shared_rate_limit_handlers()
        return result

    def get_quota_overrides(self, multi_tenant: bool, provider: Optional[str] = None) -> dict:
        """
        Return configured RPM/TPM/RPD overrides for the given mode's
        namespace. provider=None returns the full {provider: {model_name:
        {...}}} dict; a specific provider returns just that provider's
        {model_name: {...}}.
        """
        from local_search_agent.core.key_manager import get_quota_overrides as _get

        return _get(multi_tenant, provider)
