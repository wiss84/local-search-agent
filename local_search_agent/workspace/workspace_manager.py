"""
WorkspaceManager: registry of workspaces and their indexed documents.

Responsibilities
----------------
- Persist workspace metadata (name, document_dir, created_at) in SQLite.
- Hold an in-memory map of doc_id → DocumentNode for fast O(1) lookups
  by the FastAPI file server.
- Provide CRUD operations for both workspaces and documents.

Thread safety
-------------
All SQLite writes are protected by a threading.Lock.
The in-memory document cache (dict) is also protected by the same lock.
This is safe because the file server is I/O-bound and single-node.

Timestamps
----------
All timestamps use the system local timezone via datetime.now().astimezone(),
so they reflect the machine owner's timezone with an embedded UTC offset.
This is consistent with DocumentNode's _local_now_iso() helper.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime
from typing import Optional

from local_search_agent.core.document_node import DocumentNode

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS workspaces (
    name         TEXT PRIMARY KEY,
    document_dir TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,
    workspace    TEXT NOT NULL,
    title        TEXT NOT NULL,
    file_type    TEXT NOT NULL,
    source_path  TEXT NOT NULL,
    folder_path  TEXT NOT NULL,
    modified_at  TEXT NOT NULL,
    indexed_at   TEXT NOT NULL,
    text         TEXT NOT NULL DEFAULT '',
    concepts     TEXT NOT NULL DEFAULT '[]',
    synonyms     TEXT NOT NULL DEFAULT '[]',
    FOREIGN KEY (workspace) REFERENCES workspaces(name)
);
"""


def _now_iso() -> str:
    """Current local time with UTC offset as ISO-8601 string."""
    return datetime.now().astimezone().isoformat()


class WorkspaceManager:
    """
    Manages workspaces and their document registries.

    The `text` field of DocumentNode is NOT stored in SQLite (it can be
    tens of MB for large corpora).  It lives only in the in-memory cache,
    populated when documents are registered via register_document().

    This means after a server restart, ingest_and_index() must be called
    again to repopulate the text cache.  Phase 4 will add optional text
    persistence for large corpora.
    """

    def __init__(self, db_path: str = "local_search_agent.db"):
        self._db_path = db_path
        self._lock = threading.Lock()
        # In-memory doc store: doc_id → DocumentNode (includes text)
        self._doc_cache: dict[str, DocumentNode] = {}
        self._init_db()

    # ------------------------------------------------------------------
    # DB init
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)
            # Backfill: add `text` column if upgrading from an older schema
            try:
                conn.execute("ALTER TABLE documents ADD COLUMN text TEXT NOT NULL DEFAULT ''")
            except sqlite3.OperationalError:
                pass  # Column already exists (new install or already migrated)
            # Index on source_path — used by document_needs_reindex() to look up
            # whether any chunk of a file is already indexed, without knowing the
            # chunk doc_ids in advance.
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_source_path ON documents(source_path)"
            )
        logger.debug("WorkspaceManager DB initialised at %r", self._db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Workspace CRUD
    # ------------------------------------------------------------------

    def create_workspace(self, name: str, document_dir: str) -> None:
        """Register a new workspace. Silently updates document_dir if name already exists."""
        now = _now_iso()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO workspaces (name, document_dir, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET document_dir=excluded.document_dir, updated_at=excluded.updated_at
                """,
                (name, document_dir, now, now),
            )
        logger.info("Workspace registered: %r → %r", name, document_dir)

    def list_workspaces(self) -> list[dict]:
        """Return all registered workspaces as plain dicts."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM workspaces ORDER BY name").fetchall()
        return [dict(row) for row in rows]

    def get_workspace(self, name: str) -> Optional[dict]:
        """Return a single workspace dict or None if not found."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM workspaces WHERE name = ?", (name,)).fetchone()
        return dict(row) if row else None

    def delete_workspace(self, name: str) -> None:
        """Remove workspace and all its document records. Does NOT delete files."""
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM documents WHERE workspace = ?", (name,))
            conn.execute("DELETE FROM workspaces WHERE name = ?", (name,))
        # Evict from cache
        with self._lock:
            evict = [k for k, v in self._doc_cache.items() if v.workspace == name]
            for k in evict:
                del self._doc_cache[k]
        logger.info("Workspace deleted: %r (%d docs evicted from cache)", name, len(evict))

    # ------------------------------------------------------------------
    # Document CRUD
    # ------------------------------------------------------------------

    def register_document(self, node: DocumentNode) -> None:
        """
        Persist document metadata and text to SQLite and cache the full DocumentNode
        (including text) in memory for fast file-server lookups.
        """
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents
                    (doc_id, workspace, title, file_type, source_path, folder_path,
                     modified_at, indexed_at, text, concepts, synonyms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    title=excluded.title,
                    file_type=excluded.file_type,
                    source_path=excluded.source_path,
                    folder_path=excluded.folder_path,
                    modified_at=excluded.modified_at,
                    indexed_at=excluded.indexed_at,
                    text=excluded.text,
                    concepts=excluded.concepts,
                    synonyms=excluded.synonyms
                """,
                (
                    node.doc_id,
                    node.workspace,
                    node.title,
                    node.file_type,
                    node.source_path,
                    node.folder_path,
                    node.modified_at,
                    node.indexed_at,
                    node.text,
                    json.dumps(node.concepts),
                    json.dumps(node.synonyms),
                ),
            )
            self._doc_cache[node.doc_id] = node
        logger.debug("Document registered: %r (%r)", node.doc_id, node.title)

    def get_document(self, doc_id: str) -> Optional[DocumentNode]:
        """
        Retrieve a DocumentNode by doc_id.

        Checks in-memory cache first (O(1)).
        Falls back to SQLite with full text if the node was evicted or the server restarted.
        """
        # Fast path: in-memory cache
        if doc_id in self._doc_cache:
            return self._doc_cache[doc_id]

        # Cold path: SQLite (text is now persisted)
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        if row is None:
            return None

        node = DocumentNode(
            doc_id=row["doc_id"],
            title=row["title"],
            text=row["text"],
            file_type=row["file_type"],
            source_path=row["source_path"],
            folder_path=row["folder_path"],
            workspace=row["workspace"],
            modified_at=row["modified_at"],
            indexed_at=row["indexed_at"],
            concepts=json.loads(row["concepts"]),
            synonyms=json.loads(row["synonyms"]),
        )
        # Repopulate in-memory cache so subsequent lookups are fast
        self._doc_cache[node.doc_id] = node
        return node

    def list_documents(self, workspace_name: str) -> Optional[list[dict]]:
        """
        Return metadata for all documents in a workspace.
        Returns None if the workspace doesn't exist.
        """
        if self.get_workspace(workspace_name) is None:
            return None
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT doc_id, title, file_type, source_path, folder_path, modified_at, indexed_at "
                "FROM documents WHERE workspace = ? ORDER BY title",
                (workspace_name,),
            ).fetchall()
        return [dict(row) for row in rows]

    def document_needs_reindex(self, source_path: str, modified_at: str) -> bool:
        """
        Return True if the file at source_path has changed since it was last indexed.
        Used by the incremental ingestion pipeline (Phase 4).

        Queries by source_path rather than doc_id because chunked documents
        store chunk-specific doc_ids (sha256(path:chunk:N)) — the base file
        doc_id (sha256(path)) is never written to the DB.  Any chunk row for
        this path carries the same modified_at, so one LIMIT 1 lookup is enough.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT modified_at FROM documents WHERE source_path = ? LIMIT 1",
                (source_path,),
            ).fetchone()
        if row is None:
            return True  # Never indexed
        return row["modified_at"] != modified_at
