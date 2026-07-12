"""
SQLite schema + CRUD for multi-tenant RBAC (see docs/role_based_access_control.md).

This module extends the shared local_search_agent.db with four additional tables:

  workspace_members : One row per (workspace, subject) grant. The single
                       source of truth AuthorizationMiddleware checks on
                       every workspace-scoped request. Fail-closed: no row,
                       no access.

  activity_log       : Append-only audit trail ("who did what, when, in
                        which workspace"). Never UPDATEd or DELETEd except
                        via purge_activity_log()'s bulk retention sweep.

  auth_sessions       : Browser session tokens for APIKeyIdentityProvider
                        (see "Browser session flow" in the design doc).
                        Header/JWT identity providers never populate this
                        table — their session is the company's own SSO.

  auth_attempts       : Short-retention table for rate-limiting the
                        key-validation path. Deliberately separate from
                        activity_log (security mechanism, not audit trail).

  meili_keys          : One scoped, member-level Meilisearch API key per
                        workspace (Phase 7 -- data-layer defense in depth).
                        Stores the key Fernet-encrypted; see
                        auth/meili_key_crypto.py for the encrypt/decrypt
                        boundary. AuthDB itself never handles a raw
                        Meilisearch key.

  model_access_by_role : Two flat allow-lists (one per non-superadmin
                        role) of which (provider, model_name)
                        combinations that role may use. Role-level, not
                        per-subject and not per-workspace -- every member
                        anywhere shares one allow-list, every admin
                        anywhere shares a (presumably broader) one.
                        Superadmin is never a value stored here; it
                        bypasses this table entirely in code. A role with
                        zero rows has access to nothing -- fail-closed,
                        same principle as workspace_members.

Like MetadataDB, AuthDB does NOT own the SQLite connection/schema-init
lifecycle for the whole DB — it is a pure query/write helper that opens
its own connections against a shared db_path (thread-safe).

Design decision
----------------
Kept separate from workspace_manager.py and metadata_db.py for the same
reason metadata_db.py is separate from workspace_manager.py: distinct
concern (identity/authorization/audit vs. document registry vs. scheduler
state), same underlying SQLite file.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

VALID_ROLES = ("member", "admin")

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS workspace_members (
    workspace   TEXT NOT NULL,
    subject     TEXT NOT NULL,
    role        TEXT NOT NULL,          -- 'member' | 'admin'
    granted_at  TEXT NOT NULL,
    granted_by  TEXT NOT NULL,
    PRIMARY KEY (workspace, subject)
);

CREATE TABLE IF NOT EXISTS activity_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    subject     TEXT NOT NULL,
    workspace   TEXT,                   -- NULL for login/logout
    action      TEXT NOT NULL,          -- 'login' | 'logout' | 'search' | 'ingest' |
                                         -- 'delete_conversation' | 'workspace_create' |
                                         -- 'workspace_delete' | 'workspace_wipe' |
                                         -- 'grant_access' | 'revoke_access'
    detail      TEXT,                   -- e.g. query text, doc count
    ip_address  TEXT,
    timestamp   TEXT NOT NULL,
    success     INTEGER NOT NULL        -- 1/0 — log denied attempts too
);

CREATE TABLE IF NOT EXISTS auth_sessions (
    token_hash    TEXT PRIMARY KEY,     -- sha256 of the opaque session token
    subject       TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    expires_at    TEXT NOT NULL,
    ip_address    TEXT,
    user_agent    TEXT,
    display_name  TEXT NOT NULL DEFAULT '',
    is_superadmin INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS auth_attempts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    subject      TEXT,                  -- attempted subject, may be unresolved/unknown
    ip_address   TEXT,
    attempted_at TEXT NOT NULL,
    success      INTEGER NOT NULL       -- 1/0
);

CREATE TABLE IF NOT EXISTS api_keys (
    key_id        TEXT PRIMARY KEY,     -- short, non-secret public identifier (embedded in the
                                         -- raw key itself: lsa_<key_id>_<secret> — lets verification
                                         -- do an indexed PK lookup instead of scanning every row
                                         -- and argon2-verifying each one).
    subject       TEXT NOT NULL,
    key_hash      TEXT NOT NULL,        -- argon2 hash of the secret portion only, never the full raw key
    display_name  TEXT NOT NULL DEFAULT '',
    is_superadmin INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL,
    created_by    TEXT NOT NULL,
    revoked_at    TEXT                  -- NULL = active
);

CREATE TABLE IF NOT EXISTS meili_keys (
    workspace      TEXT PRIMARY KEY,    -- one scoped member-level key per workspace
    key_uid        TEXT NOT NULL,       -- Meilisearch's own key uid, needed to delete the key later
    encrypted_key  TEXT NOT NULL,       -- Fernet-encrypted raw Meilisearch key (see auth/meili_key_crypto.py)
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_access_by_role (
    role         TEXT NOT NULL,   -- 'member' | 'admin' -- never 'superadmin', which
                                   -- bypasses this table entirely in code, same
                                   -- pattern as every other superadmin bypass here
    provider     TEXT NOT NULL,
    model_name   TEXT NOT NULL,
    granted_by   TEXT NOT NULL,
    granted_at   TEXT NOT NULL,
    PRIMARY KEY (role, provider, model_name)
);
"""

_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_activity_subject ON activity_log(subject, timestamp);
CREATE INDEX IF NOT EXISTS idx_activity_workspace ON activity_log(workspace, timestamp);
CREATE INDEX IF NOT EXISTS idx_workspace_members_subject ON workspace_members(subject);
CREATE INDEX IF NOT EXISTS idx_auth_sessions_expires ON auth_sessions(expires_at);
CREATE INDEX IF NOT EXISTS idx_auth_attempts_subject ON auth_attempts(subject, attempted_at);
CREATE INDEX IF NOT EXISTS idx_auth_attempts_ip ON auth_attempts(ip_address, attempted_at);
CREATE INDEX IF NOT EXISTS idx_api_keys_subject ON api_keys(subject);
"""


def _now_iso() -> str:
    """Current local time with UTC offset as ISO-8601 string (matches other modules)."""
    return datetime.now().astimezone().isoformat()


def _validate_role(role: str) -> None:
    if role not in VALID_ROLES:
        raise ValueError(f"Invalid role {role!r}; must be one of {VALID_ROLES}")


class AuthDB:
    """
    Identity/authorization/audit database helper for multi-tenant RBAC.

    Thread-safe: all writes use a threading.Lock.

    Parameters
    ----------
    db_path : Path to the SQLite database file (same file as WorkspaceManager
              and MetadataDB).
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
            conn.executescript(_INDEX_SQL)
            # Backfill (Phase 4: browser session flow) — pre-existing auth_sessions
            # tables from Phase 1 won't have these columns yet. Nullable-safe
            # defaults so old rows (if any survive past their expiry sweep)
            # just resolve to an empty display_name / non-superadmin identity,
            # same backfill pattern as chat_sessions.created_by in store.py.
            for stmt in (
                "ALTER TABLE auth_sessions ADD COLUMN display_name TEXT NOT NULL DEFAULT ''",
                "ALTER TABLE auth_sessions ADD COLUMN is_superadmin INTEGER NOT NULL DEFAULT 0",
            ):
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError:
                    pass  # Column already exists
        logger.debug("AuthDB schema initialised at %r", self._db_path)

    # ------------------------------------------------------------------
    # workspace_members
    # ------------------------------------------------------------------

    def grant_access(self, workspace: str, subject: str, role: str, granted_by: str) -> None:
        """Grant (or update) a single workspace/subject/role row."""
        _validate_role(role)
        now = _now_iso()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO workspace_members (workspace, subject, role, granted_at, granted_by)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(workspace, subject) DO UPDATE SET
                    role = excluded.role,
                    granted_at = excluded.granted_at,
                    granted_by = excluded.granted_by
                """,
                (workspace, subject, role, now, granted_by),
            )
        logger.info(
            "Access granted: subject=%r workspace=%r role=%r by=%r",
            subject,
            workspace,
            role,
            granted_by,
        )

    def grant_access_bulk(
        self, workspaces: list[str], subject: str, role: str, granted_by: str
    ) -> None:
        """
        Grant the same subject/role across multiple workspaces in a single
        transaction — either all rows are written or none are, so a typo
        in one workspace name mid-list doesn't leave the grant half-applied.
        """
        _validate_role(role)
        if not workspaces:
            return
        now = _now_iso()
        with self._lock, self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO workspace_members (workspace, subject, role, granted_at, granted_by)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(workspace, subject) DO UPDATE SET
                    role = excluded.role,
                    granted_at = excluded.granted_at,
                    granted_by = excluded.granted_by
                """,
                [(ws, subject, role, now, granted_by) for ws in workspaces],
            )
        logger.info(
            "Access granted (bulk): subject=%r workspaces=%r role=%r by=%r",
            subject,
            workspaces,
            role,
            granted_by,
        )

    def revoke_access(self, subject: str, workspaces: Optional[list[str]] = None) -> int:
        """
        Revoke a subject's access. If `workspaces` is None, revokes every
        grant for that subject; otherwise revokes only the listed workspaces
        (single transaction, same all-or-nothing property as grant_access_bulk).

        Returns the number of rows deleted.
        """
        with self._lock, self._connect() as conn:
            if workspaces is None:
                cur = conn.execute("DELETE FROM workspace_members WHERE subject = ?", (subject,))
            else:
                if not workspaces:
                    return 0
                placeholders = ",".join("?" for _ in workspaces)
                cur = conn.execute(
                    f"DELETE FROM workspace_members WHERE subject = ? AND workspace IN ({placeholders})",
                    (subject, *workspaces),
                )
            deleted = cur.rowcount
        logger.info(
            "Access revoked: subject=%r workspaces=%r (%d rows)", subject, workspaces, deleted
        )
        return deleted

    def get_role(self, subject: str, workspace: str) -> Optional[str]:
        """Return the subject's role in a workspace, or None if no grant exists (fail-closed)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT role FROM workspace_members WHERE workspace = ? AND subject = ?",
                (workspace, subject),
            ).fetchone()
        return row["role"] if row else None

    def list_access(
        self, subject: Optional[str] = None, workspace: Optional[str] = None
    ) -> list[dict]:
        """
        List grants, filtered by subject and/or workspace.
        With neither filter, returns every grant (admin overview).
        """
        clauses = []
        params: list[str] = []
        if subject is not None:
            clauses.append("subject = ?")
            params.append(subject)
        if workspace is not None:
            clauses.append("workspace = ?")
            params.append(workspace)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM workspace_members {where} ORDER BY workspace, subject",
                params,
            ).fetchall()
        return [dict(r) for r in rows]

    def is_member(self, subject: str, workspace: str) -> bool:
        """True if the subject has any role (member or admin) in the workspace."""
        return self.get_role(subject, workspace) is not None

    def is_global_admin(self, subject: str) -> bool:
        """
        True if the subject holds an 'admin' role in at least one workspace.

        Used for actions the design doc calls "global + admin-only" —
        settings, API keys, LangSmith, workspace creation — where
        app_state.config is one shared object, so there's no way to scope
        "admin of my workspace only"; any admin of any workspace qualifies.
        Not to be confused with Identity.is_superadmin, which is a
        framework-level escape hatch checked separately by the caller.
        """
        return any(row["role"] == "admin" for row in self.list_access(subject=subject))

    # ------------------------------------------------------------------
    # activity_log
    # ------------------------------------------------------------------

    def log_activity(
        self,
        subject: str,
        action: str,
        workspace: Optional[str] = None,
        detail: Optional[str] = None,
        ip_address: Optional[str] = None,
        success: bool = True,
    ) -> int:
        """
        Append a row to the audit trail. Never raises on its own account
        into the caller's request path in a way that should block the
        request — callers should log-and-continue on failure here, since
        losing an audit row is preferable to failing the underlying action.

        Returns the inserted row id.
        """
        now = _now_iso()
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO activity_log (subject, workspace, action, detail, ip_address, timestamp, success)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (subject, workspace, action, detail, ip_address, now, 1 if success else 0),
            )
            return cur.lastrowid

    def get_activity_log(
        self,
        subject: Optional[str] = None,
        workspace: Optional[str] = None,
        limit: int = 200,
    ) -> list[dict]:
        """Return recent activity rows, newest first, optionally filtered. Admin-only at the API layer."""
        clauses = []
        params: list = []
        if subject is not None:
            clauses.append("subject = ?")
            params.append(subject)
        if workspace is not None:
            clauses.append("workspace = ?")
            params.append(workspace)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM activity_log {where} ORDER BY id DESC LIMIT ?",
                params,
            ).fetchall()
        return [dict(r) for r in rows]

    def purge_activity_log(self, older_than_days: int) -> int:
        """
        Delete activity_log rows older than `older_than_days`. Returns the
        number of rows deleted. Intended to be called periodically (CLI
        command or wired into the existing watch-mode/scheduler background
        loop) — this table holds potentially sensitive search history and
        should not grow unbounded.
        """
        threshold = (datetime.now().astimezone() - timedelta(days=older_than_days)).isoformat()
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM activity_log WHERE timestamp < ?", (threshold,))
            deleted = cur.rowcount
        logger.info("Purged %d activity_log rows older than %d days", deleted, older_than_days)
        return deleted

    # ------------------------------------------------------------------
    # auth_sessions (APIKeyIdentityProvider browser sessions)
    # ------------------------------------------------------------------

    def create_session(
        self,
        token_hash: str,
        subject: str,
        expires_at: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        display_name: str = "",
        is_superadmin: bool = False,
    ) -> None:
        """Insert a new session row. token_hash must already be a sha256 hex digest — never the raw token."""
        now = _now_iso()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO auth_sessions
                    (token_hash, subject, created_at, expires_at, ip_address, user_agent,
                     display_name, is_superadmin)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    token_hash,
                    subject,
                    now,
                    expires_at,
                    ip_address,
                    user_agent,
                    display_name,
                    1 if is_superadmin else 0,
                ),
            )

    def get_session(self, token_hash: str) -> Optional[dict]:
        """Return the session row for a token hash, or None if not found (expiry is the caller's concern)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM auth_sessions WHERE token_hash = ?", (token_hash,)
            ).fetchone()
        return dict(row) if row else None

    def extend_session(self, token_hash: str, new_expires_at: str) -> None:
        """Slide the expiry forward on activity."""
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE auth_sessions SET expires_at = ? WHERE token_hash = ?",
                (new_expires_at, token_hash),
            )

    def delete_session(self, token_hash: str) -> None:
        """Immediate revocation (logout) — deletes the row, not just a client-side cookie clear."""
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM auth_sessions WHERE token_hash = ?", (token_hash,))

    def delete_expired_sessions(self) -> int:
        """Sweep expired sessions. Returns number deleted. Safe to call periodically."""
        now = _now_iso()
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM auth_sessions WHERE expires_at < ?", (now,))
            deleted = cur.rowcount
        return deleted

    def delete_sessions_for_subject(self, subject: str) -> int:
        """Revoke every session belonging to a subject (e.g. on password/API-key rotation). Returns rows deleted."""
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM auth_sessions WHERE subject = ?", (subject,))
            return cur.rowcount

    # ------------------------------------------------------------------
    # auth_attempts (rate limiting)
    # ------------------------------------------------------------------

    def record_attempt(
        self, subject: Optional[str], ip_address: Optional[str], success: bool
    ) -> None:
        """Record a key-validation attempt for rate-limiting purposes."""
        now = _now_iso()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO auth_attempts (subject, ip_address, attempted_at, success) VALUES (?, ?, ?, ?)",
                (subject, ip_address, now, 1 if success else 0),
            )

    def count_recent_failed_attempts(
        self,
        subject: Optional[str] = None,
        ip_address: Optional[str] = None,
        window_minutes: int = 15,
    ) -> int:
        """
        Count failed attempts within the trailing window, by subject and/or
        IP. At least one of subject/ip_address must be provided.
        """
        if subject is None and ip_address is None:
            raise ValueError("Must provide at least one of subject or ip_address")
        threshold = (datetime.now().astimezone() - timedelta(minutes=window_minutes)).isoformat()
        clauses = ["success = 0", "attempted_at >= ?"]
        params: list = [threshold]
        if subject is not None:
            clauses.append("subject = ?")
            params.append(subject)
        if ip_address is not None:
            clauses.append("ip_address = ?")
            params.append(ip_address)
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) AS c FROM auth_attempts WHERE {' AND '.join(clauses)}",
                params,
            ).fetchone()
        return row["c"]

    def purge_old_attempts(self, older_than_days: int = 7) -> int:
        """Sweep old auth_attempts rows (short-retention, security-mechanism-only table)."""
        threshold = (datetime.now().astimezone() - timedelta(days=older_than_days)).isoformat()
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM auth_attempts WHERE attempted_at < ?", (threshold,))
            return cur.rowcount

    # ------------------------------------------------------------------
    # api_keys (APIKeyIdentityProvider key storage)
    # ------------------------------------------------------------------

    def create_api_key(
        self,
        key_id: str,
        subject: str,
        key_hash: str,
        created_by: str,
        display_name: str = "",
        is_superadmin: bool = False,
    ) -> None:
        """
        Persist a newly generated API key's metadata + argon2 hash.
        The raw key is never passed to or stored by this method — callers
        (APIKeyIdentityProvider.create_key) hash it before calling this.
        """
        now = _now_iso()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO api_keys
                    (key_id, subject, key_hash, display_name, is_superadmin, created_at, created_by, revoked_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    key_id,
                    subject,
                    key_hash,
                    display_name,
                    1 if is_superadmin else 0,
                    now,
                    created_by,
                ),
            )
        logger.info("API key created: key_id=%r subject=%r by=%r", key_id, subject, created_by)

    def get_api_key(self, key_id: str) -> Optional[dict]:
        """
        Return the api_keys row for a key_id (indexed PK lookup), or None if
        no such key_id exists. Revocation is the caller's concern (check
        revoked_at), same pattern as get_session()'s expiry handling.
        """
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM api_keys WHERE key_id = ?", (key_id,)).fetchone()
        return dict(row) if row else None

    def revoke_api_key(self, key_id: str) -> bool:
        """
        Mark a key as revoked (sets revoked_at, does not delete the row —
        keeps it visible in list_api_keys for audit purposes). Returns True
        if a matching, not-already-revoked key was found.
        """
        now = _now_iso()
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "UPDATE api_keys SET revoked_at = ? WHERE key_id = ? AND revoked_at IS NULL",
                (now, key_id),
            )
            revoked = cur.rowcount > 0
        if revoked:
            logger.info("API key revoked: key_id=%r", key_id)
        return revoked

    def list_api_keys(self, subject: Optional[str] = None) -> list[dict]:
        """List API key metadata (including key_hash — callers that display
        this to admins should strip it), optionally filtered by subject."""
        if subject is not None:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM api_keys WHERE subject = ? ORDER BY created_at DESC",
                    (subject,),
                ).fetchall()
        else:
            with self._connect() as conn:
                rows = conn.execute("SELECT * FROM api_keys ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # meili_keys (scoped, member-level Meilisearch API keys — Phase 7)
    # ------------------------------------------------------------------

    def store_meili_key(self, workspace: str, key_uid: str, encrypted_key: str) -> None:
        """
        Upsert the scoped Meilisearch key for a workspace. encrypted_key must
        already be Fernet-encrypted (see auth/meili_key_crypto.py) -- this
        method never sees or stores a raw key.
        """
        now = _now_iso()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO meili_keys (workspace, key_uid, encrypted_key, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(workspace) DO UPDATE SET
                    key_uid = excluded.key_uid,
                    encrypted_key = excluded.encrypted_key,
                    created_at = excluded.created_at
                """,
                (workspace, key_uid, encrypted_key, now),
            )
        logger.info("Scoped Meilisearch key stored: workspace=%r key_uid=%r", workspace, key_uid)

    def get_meili_key_row(self, workspace: str) -> Optional[dict]:
        """
        Return the raw meili_keys row (key_uid + still-encrypted key) for a
        workspace, or None if no scoped key has been provisioned for it --
        callers should fall back to the service-level master key in that
        case (e.g. a workspace created before this feature existed).
        Decryption is the caller's concern (see auth/meili_key_crypto.py) --
        this class never handles plaintext Meilisearch keys.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM meili_keys WHERE workspace = ?", (workspace,)
            ).fetchone()
        return dict(row) if row else None

    def delete_meili_key(self, workspace: str) -> bool:
        """Delete the stored scoped-key row for a workspace (e.g. on workspace delete). Returns True if a row existed."""
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM meili_keys WHERE workspace = ?", (workspace,))
            return cur.rowcount > 0

    # ------------------------------------------------------------------
    # model_access_by_role (Model / Provider Access Control)
    # ------------------------------------------------------------------

    def grant_model_access(
        self, role: str, provider: str, model_name: str, granted_by: str
    ) -> None:
        """Grant (or refresh) a single role/provider/model allow-list row."""
        _validate_role(role)
        now = _now_iso()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO model_access_by_role (role, provider, model_name, granted_by, granted_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(role, provider, model_name) DO UPDATE SET
                    granted_by = excluded.granted_by,
                    granted_at = excluded.granted_at
                """,
                (role, provider, model_name, granted_by, now),
            )
        logger.info(
            "Model access granted: role=%r provider=%r model=%r by=%r",
            role,
            provider,
            model_name,
            granted_by,
        )

    def revoke_model_access(self, role: str, provider: str, model_name: str) -> bool:
        """Remove a single role/provider/model allow-list row. Returns True if a row existed."""
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM model_access_by_role WHERE role = ? AND provider = ? AND model_name = ?",
                (role, provider, model_name),
            )
            revoked = cur.rowcount > 0
        if revoked:
            logger.info(
                "Model access revoked: role=%r provider=%r model=%r", role, provider, model_name
            )
        return revoked

    def list_model_access(self, role: Optional[str] = None) -> list[dict]:
        """List allow-list rows, optionally filtered to one role. With no filter, returns every row (admin overview)."""
        if role is not None:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM model_access_by_role WHERE role = ? ORDER BY provider, model_name",
                    (role,),
                ).fetchall()
        else:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM model_access_by_role ORDER BY role, provider, model_name"
                ).fetchall()
        return [dict(r) for r in rows]

    def is_model_allowed(self, role: str, provider: str, model_name: str) -> bool:
        """
        True if (provider, model_name) is on `role`'s allow-list.
        Fail-closed: a role with zero rows here has access to nothing,
        same principle as get_role()'s "no row, no access". Does NOT check
        Identity.is_superadmin -- callers must check that themselves
        before calling this, same pattern as is_global_admin()'s own
        docstring (superadmin is a framework-level bypass, never a value
        stored in this table).
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM model_access_by_role WHERE role = ? AND provider = ? AND model_name = ?",
                (role, provider, model_name),
            ).fetchone()
        return row is not None

    def role_allowed_models(self, role: str) -> dict[str, list[str]]:
        """
        Return `role`'s allowed models grouped by provider -- the same
        shape as key_manager.get_models() (provider -> list[model_name]),
        so callers can intersect against the full configured set with a
        single dict comprehension rather than reshaping rows themselves.
        """
        grouped: dict[str, list[str]] = {}
        for row in self.list_model_access(role=role):
            grouped.setdefault(row["provider"], []).append(row["model_name"])
        return grouped
