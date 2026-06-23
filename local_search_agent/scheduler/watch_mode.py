"""
Watch mode for the Local Search Agent framework.

Replaces (or complements) the polling-based IncrementalSyncScheduler with
filesystem event-driven re-indexing using `watchdog`. Instead of waiting up
to `interval_minutes` for a poll, a workspace re-indexes within a couple of
seconds of a file being created, modified, or deleted.

Design
------
- One `watchdog.observers.Observer` per framework process, with one
  scheduled watch per workspace document_dir.
- All filesystem events for a workspace are debounced: rapid bursts of
  events (e.g. a single file save firing multiple OS-level notifications,
  or a folder copy) are collapsed into a single sync trigger, fired only
  after `debounce_seconds` of inactivity.
- Each triggered sync reuses the exact same IngestionPipeline + delta logic
  (document_needs_reindex) as the polling scheduler and manual sync — the
  only behavioural difference is *when* a sync fires and whether semantic
  enrichment runs for that particular sync (controlled by `enrich`).
- MetadataDB sync state/history is updated identically to the polling
  scheduler, so `get_index_health()` and sync history work the same way
  regardless of which mechanism triggered the sync. The one deliberate
  difference: `next_sync_at` is always recorded as None here, since watch
  mode is event-driven and has no fixed next-run time.

This module does not replace IncrementalSyncScheduler in code — both can
exist side by side. SearchAgentFramework.start_watch_mode() is the
recommended path going forward; start_incremental_scheduler() is kept for
backward compatibility (see deprecation note there).

Usage
-----
    from local_search_agent.scheduler.watch_mode import WorkspaceWatcher

    watcher = WorkspaceWatcher(workspace_manager=wm, metadata_db=metadata_db)
    watcher.start()                  # starts the underlying Observer thread
    watcher.add_workspace(config)    # config.enrich_on_watch controls enrichment
    watcher.trigger_now("finance")   # force immediate sync, bypassing debounce
    watcher.stop()
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Debounce window: collapse bursts of filesystem events into one sync.
DEFAULT_DEBOUNCE_SECONDS = 2.5


class _DebouncedHandler:
    """
    Debounces bursts of filesystem events for one workspace into a single
    delayed call to `on_settle`. Not a watchdog FileSystemEventHandler
    itself — see `_build_handler` for the actual watchdog subclass, kept
    separate so this class has no hard dependency on watchdog being
    importable at module load time.
    """

    def __init__(self, workspace: str, on_settle, debounce_seconds: float):
        self._workspace = workspace
        self._on_settle = on_settle
        self._debounce_seconds = debounce_seconds
        self._timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()

    def notify(self) -> None:
        """Called on every filesystem event; (re)starts the debounce timer."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self._debounce_seconds, self._fire)
            self._timer.daemon = True
            self._timer.start()

    def _fire(self) -> None:
        try:
            self._on_settle(self._workspace)
        except Exception:
            logger.exception("Watch-mode debounced sync failed for workspace %r.", self._workspace)

    def cancel(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None


def _build_handler(debounced: "_DebouncedHandler"):
    """
    Build a watchdog FileSystemEventHandler bound to a _DebouncedHandler.
    Imports watchdog lazily so it is only required when watch mode is
    actually used.
    """
    from watchdog.events import FileSystemEventHandler

    class _Handler(FileSystemEventHandler):
        def on_any_event(self, event):
            # Ignore directory-only events; we only care about file content
            # changes (create/modify/delete/move of an actual file).
            if event.is_directory:
                return
            debounced.notify()

    return _Handler()


class WorkspaceWatcher:
    """
    watchdog-backed, event-driven incremental sync.

    Parameters
    ----------
    workspace_manager : WorkspaceManager instance
    metadata_db       : MetadataDB instance for sync state tracking
    debounce_seconds  : Seconds of filesystem inactivity before a sync fires
                        (default 2.5s). Collapses bursts of OS events from a
                        single save or folder copy into one sync.
    """

    def __init__(
        self,
        workspace_manager,
        metadata_db,
        debounce_seconds: float = DEFAULT_DEBOUNCE_SECONDS,
    ):
        self._wm = workspace_manager
        self._db = metadata_db
        self._debounce_seconds = debounce_seconds
        self._observer = None  # Lazy init
        self._running = False
        # workspace_name -> SearchAgentConfig
        self._workspace_configs: dict[str, object] = {}
        # workspace_name -> optional progress_callback
        self._workspace_callbacks: dict[str, object] = {}
        # workspace_name -> _DebouncedHandler
        self._debouncers: dict[str, _DebouncedHandler] = {}
        # workspace_name -> list of watchdog ObservedWatch handles
        self._watches: dict[str, list] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _get_observer(self):
        """Lazily create the watchdog Observer."""
        if self._observer is None:
            try:
                from watchdog.observers import Observer
            except ImportError as e:
                raise ImportError(
                    "watchdog is not installed. Run: pip install 'watchdog>=5.0.0,<7.0.0'"
                ) from e
            self._observer = Observer()
        return self._observer

    def start(self) -> None:
        """Start the underlying watchdog Observer thread. Non-blocking."""
        if self._running:
            logger.warning("WorkspaceWatcher already running.")
            return
        observer = self._get_observer()
        observer.start()
        self._running = True
        logger.info("WorkspaceWatcher started (debounce: %.1fs).", self._debounce_seconds)

    def stop(self, wait: bool = True) -> None:
        """Stop the watcher and cancel any pending debounce timers."""
        if not self._running:
            return
        for debouncer in self._debouncers.values():
            debouncer.cancel()
        if self._observer is not None:
            self._observer.stop()
            if wait:
                self._observer.join(timeout=5)
        self._running = False
        logger.info("WorkspaceWatcher stopped.")

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Workspace management
    # ------------------------------------------------------------------

    def add_workspace(self, config, progress_callback=None) -> None:  # config: SearchAgentConfig
        """
        Register a workspace's document_dirs with the watcher.

        If the workspace is already registered, its watches are replaced
        (handles config changes, e.g. document_dirs or enrich_on_watch).
        """
        workspace = config.workspace_name
        self.remove_workspace(workspace)  # clean any prior watches first

        self._workspace_configs[workspace] = config
        self._workspace_callbacks[workspace] = progress_callback

        debouncer = _DebouncedHandler(
            workspace=workspace,
            on_settle=self._run_sync,
            debounce_seconds=self._debounce_seconds,
        )
        self._debouncers[workspace] = debouncer

        # Ensure sync_job record exists, mirroring the polling scheduler so
        # health/history views behave consistently regardless of trigger type.
        # next_sync_at is intentionally None: watch mode is event-driven and
        # has no fixed next-run time.
        self._db.upsert_sync_job(workspace=workspace, next_sync_at=None)

        if not self._running:
            logger.info(
                "Workspace %r registered for watch mode (watcher not yet started).", workspace
            )
            return

        self._schedule_workspace_watches(workspace, config, debouncer)

    def remove_workspace(self, workspace: str) -> None:
        """Stop watching a workspace's directories and clear its state."""
        self._workspace_configs.pop(workspace, None)
        self._workspace_callbacks.pop(workspace, None)
        debouncer = self._debouncers.pop(workspace, None)
        if debouncer is not None:
            debouncer.cancel()
        watches = self._watches.pop(workspace, None)
        if watches and self._observer is not None:
            for watch in watches:
                try:
                    self._observer.unschedule(watch)
                except Exception:
                    pass

    def trigger_now(self, workspace: str) -> None:
        """Force an immediate sync, bypassing the debounce window."""
        if workspace not in self._workspace_configs:
            raise ValueError(
                f"Workspace {workspace!r} is not registered with the watcher. "
                "Call add_workspace() first."
            )
        # Cancel any pending debounced timer so we don't double-sync.
        debouncer = self._debouncers.get(workspace)
        if debouncer is not None:
            debouncer.cancel()
        logger.info("Triggering immediate watch-mode sync for workspace %r.", workspace)
        self._run_sync(workspace)

    # ------------------------------------------------------------------
    # Scheduling helpers
    # ------------------------------------------------------------------

    def _schedule_workspace_watches(self, workspace: str, config, debouncer) -> None:
        observer = self._get_observer()
        handler = _build_handler(debouncer)
        watches = []
        for doc_dir in config.document_dirs:
            try:
                watch = observer.schedule(handler, path=doc_dir, recursive=True)
                watches.append(watch)
            except Exception as e:
                logger.warning(
                    "Could not watch directory %r for workspace %r: %s", doc_dir, workspace, e
                )
        self._watches[workspace] = watches
        logger.info(
            "Watching %d directory(ies) for workspace %r (enrich_on_watch=%s).",
            len(watches),
            workspace,
            getattr(config, "enrich_on_watch", True),
        )

    def _run_sync(self, workspace: str) -> None:
        """
        Execute one watch-triggered sync for a workspace.

        Mirrors IncrementalSyncScheduler._run_sync, with two differences:
          - force is always False (watch mode only ever reacts to actual changes)
          - `enrich` is taken from config.enrich_on_watch instead of always True
        """
        from local_search_agent.ingestion.pipeline import IngestionPipeline
        from local_search_agent.search.meilisearch_client import MeilisearchClient

        config = self._workspace_configs.get(workspace)
        if config is None:
            logger.warning(
                "Watch-mode sync fired for unregistered workspace %r; ignoring.", workspace
            )
            return
        progress_callback = self._workspace_callbacks.get(workspace)
        enrich = bool(getattr(config, "enrich_on_watch", True))

        logger.info("Starting watch-mode sync for workspace %r (enrich=%s) ...", workspace, enrich)
        history_id = self._db.record_sync_start(workspace)
        self._db.set_sync_running(workspace)

        start_time = time.monotonic()
        error: Optional[str] = None

        try:
            meili_client = MeilisearchClient(
                url=config.meilisearch_url,
                api_key=config.meili_master_key,
                index_name=config.index_name or workspace,
            )

            pipeline = IngestionPipeline(
                config=config,
                workspace_manager=self._wm,
                meili_client=meili_client,
            )

            stats = pipeline.run(force=False, enrich=enrich, progress_callback=progress_callback)
            duration = time.monotonic() - start_time

            logger.info("Watch-mode sync complete for workspace %r: %s", workspace, stats)

            self._db.record_sync_finish(
                history_id=history_id,
                indexed=stats.indexed,
                skipped=stats.skipped,
                failed=stats.failed,
                duration_s=duration,
                errors=stats.errors,
            )
            self._db.set_sync_complete(
                workspace=workspace,
                doc_count=stats.indexed + stats.skipped,
                error_count=stats.failed,
                next_sync_at=None,  # event-driven; no fixed next-sync time
                last_error=stats.errors[0] if stats.errors else None,
            )

        except Exception as e:
            duration = time.monotonic() - start_time
            error = str(e)
            logger.exception("Watch-mode sync failed for workspace %r: %s", workspace, e)

            self._db.record_sync_finish(
                history_id=history_id,
                indexed=0,
                skipped=0,
                failed=1,
                duration_s=duration,
                errors=[error],
            )
            self._db.set_sync_complete(
                workspace=workspace,
                doc_count=0,
                error_count=1,
                next_sync_at=None,
                last_error=error,
            )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return a summary of the watcher's current state."""
        watched = {workspace: len(watches) for workspace, watches in self._watches.items()}
        return {
            "running": self._running,
            "registered_workspaces": list(self._workspace_configs.keys()),
            "watched_directories": watched,
            "debounce_seconds": self._debounce_seconds,
        }
