"""
SQLite schema extensions for Phase 4 scheduler and workspace metadata.

This module extends the WorkspaceManager's database with two additional tables:

  sync_jobs      : Tracks scheduler state per workspace
                   (last_sync, next_sync, sync_status, doc_count, error_count)

  sync_history   : Append-only log of every sync run
                   (workspace, started_at, finished_at, indexed, skipped, failed)

The WorkspaceManager already owns the SQLite connection and schema init.
MetadataDB is a pure query/write helper — it does NOT own the connection.
It receives the db_path and opens its own connections as needed (thread-safe).

Design decision
---------------
Kept separate from workspace_manager.py to maintain clear separation of concerns:
  workspace_manager.py  → document registry (what's indexed)
  metadata_db.py        → scheduler state (when/how sync ran)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sync_jobs (
    workspace       TEXT PRIMARY KEY,
    last_sync_at    TEXT,           -- ISO-8601 local-tz timestamp of last completed sync
    next_sync_at    TEXT,           -- ISO-8601 local-tz timestamp of next scheduled sync
    sync_status     TEXT NOT NULL DEFAULT 'idle',
                                    -- 'idle' | 'running' | 'error'
    last_error      TEXT,           -- Error message from last failed sync (if any)
    doc_count       INTEGER NOT NULL DEFAULT 0,
    error_count     INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sync_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace       TEXT NOT NULL,
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    indexed         INTEGER NOT NULL DEFAULT 0,
    skipped         INTEGER NOT NULL DEFAULT 0,
    failed          INTEGER NOT NULL DEFAULT 0,
    duration_s      REAL NOT NULL DEFAULT 0.0,
    errors          TEXT NOT NULL DEFAULT '[]'  -- JSON list of error strings
);
"""


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat()


class MetadataDB:
    """
    Scheduler state and sync history database helper.

    Thread-safe: all writes use a threading.Lock.

    Parameters
    ----------
    db_path : Path to the SQLite database file (same file as WorkspaceManager).
    """

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)
        logger.debug("MetadataDB schema initialised at %r", self._db_path)

    # ------------------------------------------------------------------
    # sync_jobs CRUD
    # ------------------------------------------------------------------

    def upsert_sync_job(
        self,
        workspace: str,
        next_sync_at: Optional[str] = None,
        status: str = "idle",
    ) -> None:
        """Create or update a sync job record for a workspace."""
        now = _now_iso()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sync_jobs
                    (workspace, next_sync_at, sync_status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(workspace) DO UPDATE SET
                    next_sync_at = COALESCE(excluded.next_sync_at, next_sync_at),
                    sync_status  = excluded.sync_status,
                    updated_at   = excluded.updated_at
                """,
                (workspace, next_sync_at, status, now, now),
            )

    def set_sync_running(self, workspace: str) -> None:
        """Mark a workspace sync as currently running."""
        now = _now_iso()
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE sync_jobs SET sync_status='running', updated_at=? WHERE workspace=?",
                (now, workspace),
            )

    def set_sync_complete(
        self,
        workspace: str,
        doc_count: int,
        error_count: int,
        next_sync_at: str,
        last_error: Optional[str] = None,
    ) -> None:
        """Mark a workspace sync as complete (idle or error)."""
        now = _now_iso()
        status = "error" if error_count > 0 else "idle"
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE sync_jobs SET
                    last_sync_at = ?,
                    next_sync_at = ?,
                    sync_status  = ?,
                    last_error   = ?,
                    doc_count    = ?,
                    error_count  = ?,
                    updated_at   = ?
                WHERE workspace = ?
                """,
                (now, next_sync_at, status, last_error, doc_count, error_count, now, workspace),
            )

    def get_sync_job(self, workspace: str) -> Optional[dict]:
        """Return the sync job record for a workspace, or None if not found."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM sync_jobs WHERE workspace=?", (workspace,)).fetchone()
        return dict(row) if row else None

    def list_sync_jobs(self) -> list[dict]:
        """Return all sync job records ordered by workspace name."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM sync_jobs ORDER BY workspace").fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # sync_history
    # ------------------------------------------------------------------

    def record_sync_start(self, workspace: str) -> int:
        """
        Insert a new sync_history record for a starting sync.
        Returns the row id for later update via record_sync_finish().
        """
        now = _now_iso()
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO sync_history (workspace, started_at, errors)
                VALUES (?, ?, '[]')
                """,
                (workspace, now),
            )
            return cur.lastrowid

    def record_sync_finish(
        self,
        history_id: int,
        indexed: int,
        skipped: int,
        failed: int,
        duration_s: float,
        errors: list[str],
    ) -> None:
        """Update an existing sync_history record with completion data."""
        now = _now_iso()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE sync_history SET
                    finished_at = ?,
                    indexed     = ?,
                    skipped     = ?,
                    failed      = ?,
                    duration_s  = ?,
                    errors      = ?
                WHERE id = ?
                """,
                (now, indexed, skipped, failed, duration_s, json.dumps(errors), history_id),
            )

    def get_sync_history(self, workspace: str, limit: int = 20) -> list[dict]:
        """Return the most recent sync history records for a workspace."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM sync_history
                WHERE workspace=?
                ORDER BY id DESC LIMIT ?
                """,
                (workspace, limit),
            ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["errors"] = json.loads(d.get("errors") or "[]")
            result.append(d)
        return result

    # ------------------------------------------------------------------
    # Freshness monitoring
    # ------------------------------------------------------------------

    def get_stale_workspaces(self, older_than_minutes: int = 30) -> list[dict]:
        """
        Return workspaces whose last_sync_at is older than `older_than_minutes`
        or have never been synced. Used by the monitor to flag stale indexes.
        """
        from datetime import timedelta

        threshold = (
            datetime.now().astimezone() - timedelta(minutes=older_than_minutes)
        ).isoformat()

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM sync_jobs
                WHERE last_sync_at IS NULL OR last_sync_at < ?
                ORDER BY workspace
                """,
                (threshold,),
            ).fetchall()
        return [dict(r) for r in rows]
