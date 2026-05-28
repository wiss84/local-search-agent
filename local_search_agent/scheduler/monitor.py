"""
Index freshness monitor for the Local Search Agent framework.

Responsibilities
----------------
- Track which workspaces are stale (last sync too old or never synced)
- Provide a health summary across all workspaces
- Expose a /health/indexes endpoint payload for the FastAPI server (Phase 5)

The monitor is purely read-only against MetadataDB. It does NOT trigger
re-ingestion — that is the scheduler's job. It answers the question:
"Are my indexes fresh enough to trust?"

Usage
-----
    from local_search_agent.scheduler.monitor import IndexMonitor

    monitor = IndexMonitor(metadata_db)
    summary = monitor.get_health_summary()
    stale = monitor.get_stale_workspaces(older_than_minutes=30)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class WorkspaceHealth:
    """Health status for a single workspace."""

    workspace: str
    status: str  # 'healthy' | 'stale' | 'never_synced' | 'error' | 'running'
    last_sync_at: Optional[str]
    next_sync_at: Optional[str]
    doc_count: int
    error_count: int
    last_error: Optional[str]
    age_minutes: Optional[float]  # Minutes since last sync (None if never synced)

    def is_healthy(self) -> bool:
        return self.status == "healthy"


@dataclass
class IndexHealthSummary:
    """Aggregate health summary across all workspaces."""

    total_workspaces: int = 0
    healthy: int = 0
    stale: int = 0
    never_synced: int = 0
    error: int = 0
    running: int = 0
    total_docs: int = 0
    workspaces: list[WorkspaceHealth] = field(default_factory=list)

    @property
    def all_healthy(self) -> bool:
        return self.stale == 0 and self.never_synced == 0 and self.error == 0

    def to_dict(self) -> dict:
        return {
            "total_workspaces": self.total_workspaces,
            "healthy": self.healthy,
            "stale": self.stale,
            "never_synced": self.never_synced,
            "error": self.error,
            "running": self.running,
            "total_docs": self.total_docs,
            "all_healthy": self.all_healthy,
            "workspaces": [
                {
                    "workspace": w.workspace,
                    "status": w.status,
                    "last_sync_at": w.last_sync_at,
                    "next_sync_at": w.next_sync_at,
                    "doc_count": w.doc_count,
                    "error_count": w.error_count,
                    "age_minutes": round(w.age_minutes, 1) if w.age_minutes is not None else None,
                }
                for w in self.workspaces
            ],
        }


class IndexMonitor:
    """
    Read-only monitor for workspace index freshness.

    Parameters
    ----------
    metadata_db      : MetadataDB instance
    stale_threshold  : Minutes after which a workspace is considered stale (default 30)
    """

    def __init__(self, metadata_db, stale_threshold_minutes: int = 30):
        self._db = metadata_db
        self._stale_threshold = stale_threshold_minutes

    def get_workspace_health(self, workspace: str) -> Optional[WorkspaceHealth]:
        """Return health status for a single workspace."""
        job = self._db.get_sync_job(workspace)
        if job is None:
            return None
        return self._job_to_health(job)

    def get_stale_workspaces(
        self, older_than_minutes: Optional[int] = None
    ) -> list[WorkspaceHealth]:
        """Return all workspaces that need re-ingestion."""
        threshold = older_than_minutes or self._stale_threshold
        stale_jobs = self._db.get_stale_workspaces(older_than_minutes=threshold)
        return [self._job_to_health(j) for j in stale_jobs]

    def get_health_summary(self) -> IndexHealthSummary:
        """Return aggregate health across all registered workspaces."""
        jobs = self._db.list_sync_jobs()
        summary = IndexHealthSummary(total_workspaces=len(jobs))

        for job in jobs:
            health = self._job_to_health(job)
            summary.workspaces.append(health)
            summary.total_docs += job.get("doc_count", 0)

            if health.status == "healthy":
                summary.healthy += 1
            elif health.status == "stale":
                summary.stale += 1
            elif health.status == "never_synced":
                summary.never_synced += 1
            elif health.status == "error":
                summary.error += 1
            elif health.status == "running":
                summary.running += 1

        return summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _job_to_health(self, job: dict) -> WorkspaceHealth:
        """Convert a sync_job row dict to a WorkspaceHealth object."""
        last_sync_at = job.get("last_sync_at")
        db_status = job.get("sync_status", "idle")
        age_minutes: Optional[float] = None

        if last_sync_at:
            try:
                last_sync_dt = datetime.fromisoformat(last_sync_at)
                now = datetime.now().astimezone()
                age_minutes = (now - last_sync_dt).total_seconds() / 60.0
            except ValueError:
                pass

        # Determine health status
        if db_status == "running":
            status = "running"
        elif last_sync_at is None:
            status = "never_synced"
        elif db_status == "error":
            status = "error"
        elif age_minutes is not None and age_minutes > self._stale_threshold:
            status = "stale"
        else:
            status = "healthy"

        return WorkspaceHealth(
            workspace=job["workspace"],
            status=status,
            last_sync_at=last_sync_at,
            next_sync_at=job.get("next_sync_at"),
            doc_count=job.get("doc_count", 0),
            error_count=job.get("error_count", 0),
            last_error=job.get("last_error"),
            age_minutes=age_minutes,
        )
