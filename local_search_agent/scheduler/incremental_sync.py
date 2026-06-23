"""
Incremental sync scheduler for the Local Search Agent framework.

DEPRECATED: this polling-based scheduler is kept for backward compatibility.
New code should prefer Watch Mode (local_search_agent.scheduler.watch_mode),
which reacts to filesystem events via `watchdog` instead of polling on a
fixed interval. See docs/watch-mode.md (or watch-mode section of the docs)
for migration guidance.

Runs a background APScheduler job that re-indexes changed documents
across all registered workspaces on a configurable interval.

Design
------
- One APScheduler BackgroundScheduler instance per framework process.
- Each registered workspace gets its own interval job.
- Jobs use the IngestionPipeline with force=False (delta logic only processes
  files whose modified_at has changed since last index).
- Each workspace gets its own MeilisearchClient with its own index_name
  (= workspace name). This gives true workspace isolation in Meilisearch.
- MetadataDB tracks sync state and history for monitoring.
- The scheduler is designed to survive individual workspace sync failures
  (one workspace error never cancels other workspace jobs).

APScheduler version
-------------------
Uses APScheduler 3.x (3.11.2+). Version 4 is still alpha.
Interval jobs via BlockingScheduler or BackgroundScheduler.

Usage
-----
    from local_search_agent.scheduler.incremental_sync import IncrementalSyncScheduler

    scheduler = IncrementalSyncScheduler(
        workspace_manager=wm,
        metadata_db=metadata_db,
        interval_minutes=15,
    )
    scheduler.start()                        # non-blocking background thread
    scheduler.add_workspace(config)          # add a workspace to watch
    scheduler.trigger_now("finance")         # force immediate sync
    scheduler.stop()                         # graceful shutdown
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


class IncrementalSyncScheduler:
    """
    APScheduler-backed incremental sync scheduler.

    Manages one interval job per workspace. Each job runs the
    IngestionPipeline with force=False so only changed files are processed.

    Parameters
    ----------
    workspace_manager : WorkspaceManager instance
    metadata_db       : MetadataDB instance for sync state tracking
    interval_minutes  : Default sync interval for new workspaces (default 15)
    """

    def __init__(
        self,
        workspace_manager,
        metadata_db,
        interval_minutes: int = 15,
    ):
        import warnings

        warnings.warn(
            "IncrementalSyncScheduler (polling-based) is deprecated. "
            "Use local_search_agent.scheduler.watch_mode.WorkspaceWatcher "
            "(or framework.start_watch_mode()) instead, which reacts to "
            "filesystem changes via watchdog instead of polling on a fixed interval.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._wm = workspace_manager
        self._db = metadata_db
        self._default_interval = interval_minutes
        self._scheduler = None  # Lazy init
        self._running = False
        # workspace_name → SearchAgentConfig
        self._workspace_configs: dict[str, object] = {}
        # workspace_name → optional progress_callback
        self._workspace_callbacks: dict[str, object] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _get_scheduler(self):
        """Lazily create the APScheduler BackgroundScheduler."""
        if self._scheduler is None:
            try:
                from apscheduler.executors.pool import ThreadPoolExecutor
                from apscheduler.schedulers.background import BackgroundScheduler
            except ImportError as e:
                raise ImportError(
                    "APScheduler is not installed. Run: pip install 'apscheduler>=3.11.2,<4.0'"
                ) from e

            self._scheduler = BackgroundScheduler(
                executors={"default": ThreadPoolExecutor(max_workers=4)},
                job_defaults={
                    "coalesce": True,  # If a job is overdue, run it once (not N times)
                    "max_instances": 1,  # Never run the same workspace job concurrently
                    "misfire_grace_time": 60,
                },
            )
        return self._scheduler

    def start(self) -> None:
        """Start the background scheduler. Non-blocking."""
        if self._running:
            logger.warning("Scheduler already running.")
            return

        scheduler = self._get_scheduler()
        scheduler.start()
        self._running = True
        logger.info(
            "IncrementalSyncScheduler started (default interval: %dm).",
            self._default_interval,
        )

    def stop(self, wait: bool = True) -> None:
        """
        Stop the background scheduler gracefully.

        Parameters
        ----------
        wait : If True, wait for running jobs to complete before shutting down.
        """
        if not self._running:
            return
        if self._scheduler:
            self._scheduler.shutdown(wait=wait)
        self._running = False
        logger.info("IncrementalSyncScheduler stopped.")

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Workspace management
    # ------------------------------------------------------------------

    def add_workspace(
        self,
        config,  # SearchAgentConfig
        interval_minutes: Optional[int] = None,
        progress_callback=None,
    ) -> None:
        """
        Register a workspace for incremental sync and add an APScheduler job.

        If the workspace is already registered, updates the interval.

        Parameters
        ----------
        config           : SearchAgentConfig for this workspace.
        interval_minutes : Override the default interval for this workspace.
        """
        interval = interval_minutes or self._default_interval
        workspace = config.workspace_name
        self._workspace_configs[workspace] = config
        self._workspace_callbacks[workspace] = progress_callback

        # Ensure sync_job record exists in MetadataDB
        next_sync = (datetime.now().astimezone() + timedelta(minutes=interval)).isoformat()
        self._db.upsert_sync_job(workspace=workspace, next_sync_at=next_sync)

        if not self._running:
            logger.info("Workspace %r registered for sync (scheduler not yet started).", workspace)
            return

        self._schedule_workspace_job(
            workspace, config, interval, self._workspace_callbacks.get(workspace)
        )

    def remove_workspace(self, workspace: str) -> None:
        """Remove a workspace from the sync schedule."""
        self._workspace_configs.pop(workspace, None)
        if self._scheduler:
            job_id = self._job_id(workspace)
            try:
                self._scheduler.remove_job(job_id)
                logger.info("Removed sync job for workspace %r.", workspace)
            except Exception:
                pass  # Job may not exist

    def trigger_now(self, workspace: str) -> None:
        """Force an immediate sync for a workspace (outside the normal schedule)."""
        config = self._workspace_configs.get(workspace)
        if config is None:
            raise ValueError(
                f"Workspace {workspace!r} is not registered with the scheduler. "
                "Call add_workspace() first."
            )
        logger.info("Triggering immediate sync for workspace %r.", workspace)
        self._run_sync(workspace, config)

    # ------------------------------------------------------------------
    # Scheduling helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _job_id(workspace: str) -> str:
        return f"incremental_sync_{workspace}"

    def _schedule_workspace_job(
        self, workspace: str, config, interval: int, progress_callback=None
    ) -> None:
        """Add or replace the APScheduler interval job for a workspace."""
        scheduler = self._get_scheduler()
        job_id = self._job_id(workspace)

        # Remove existing job if present (handles re-registration with new interval)
        try:
            scheduler.remove_job(job_id)
        except Exception:
            pass

        scheduler.add_job(
            func=self._run_sync,
            trigger="interval",
            minutes=interval,
            id=job_id,
            name=f"Sync workspace: {workspace}",
            args=[workspace, config, progress_callback],
        )
        logger.info("Scheduled sync job for workspace %r every %d minutes.", workspace, interval)

    def _run_sync(self, workspace: str, config, progress_callback=None) -> None:
        """
        Execute one incremental sync for a workspace.

        Called by APScheduler on each interval tick, or directly by trigger_now().
        Updates MetadataDB with sync state before and after.
        """
        from local_search_agent.ingestion.pipeline import IngestionPipeline
        from local_search_agent.search.meilisearch_client import MeilisearchClient

        logger.info("Starting incremental sync for workspace %r ...", workspace)
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

            stats = pipeline.run(force=False, progress_callback=progress_callback)
            duration = time.monotonic() - start_time

            logger.info("Sync complete for workspace %r: %s", workspace, stats)

            # Calculate next sync time
            interval = self._default_interval
            job_id = self._job_id(workspace)
            if self._scheduler:
                try:
                    job = self._scheduler.get_job(job_id)
                    if job and job.trigger:
                        # Extract interval from trigger if available
                        interval_attr = getattr(job.trigger, "interval", None)
                        if interval_attr:
                            interval = int(interval_attr.total_seconds() / 60)
                except Exception:
                    pass

            next_sync = (datetime.now().astimezone() + timedelta(minutes=interval)).isoformat()

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
                next_sync_at=next_sync,
                last_error=stats.errors[0] if stats.errors else None,
            )

        except Exception as e:
            duration = time.monotonic() - start_time
            error = str(e)
            logger.exception("Sync failed for workspace %r: %s", workspace, e)

            next_sync = (
                datetime.now().astimezone() + timedelta(minutes=self._default_interval)
            ).isoformat()

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
                next_sync_at=next_sync,
                last_error=error,
            )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return a summary of the scheduler's current state."""
        jobs = []
        if self._scheduler:
            for job in self._scheduler.get_jobs():
                next_run = job.next_run_time
                jobs.append(
                    {
                        "job_id": job.id,
                        "name": job.name,
                        "next_run_at": next_run.isoformat() if next_run else None,
                    }
                )
        return {
            "running": self._running,
            "registered_workspaces": list(self._workspace_configs.keys()),
            "scheduled_jobs": jobs,
        }
