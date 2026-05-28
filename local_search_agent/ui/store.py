"""
UIStore: persistence layer for the desktop dashboard.

Manages three tables inside the existing local_search_agent.db:

    chat_sessions  — one row per conversation session (workspace-scoped)
    chat_messages  — one row per message turn (user or assistant)
    ui_config      — flat key/value store for UI state (theme, last provider, etc.)

All writes are protected by the same threading.Lock used by WorkspaceManager
when UIStore is initialised with a shared connection.  When constructed
standalone it creates its own lock.

Schema is created idempotently on __init__ (CREATE TABLE IF NOT EXISTS),
so this is safe to construct alongside an existing WorkspaceManager instance
that already owns the same db_path.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id   TEXT PRIMARY KEY,
    workspace    TEXT NOT NULL,
    title        TEXT NOT NULL DEFAULT '',
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_messages (
    message_id   TEXT PRIMARY KEY,
    session_id   TEXT NOT NULL,
    role         TEXT NOT NULL,
    content      TEXT NOT NULL,
    tool_calls   TEXT NOT NULL DEFAULT '[]',
    thinking     TEXT NOT NULL DEFAULT '',
    token_query  INTEGER NOT NULL DEFAULT 0,
    token_reply  INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id)
);

CREATE TABLE IF NOT EXISTS ui_config (
    key          TEXT PRIMARY KEY,
    value        TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
"""

# Index for fast session listing per workspace
_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_chat_sessions_workspace
    ON chat_sessions(workspace, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session
    ON chat_messages(session_id, created_at ASC);
"""


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex


class UIStore:
    """
    Persistence layer for dashboard UI state.

    Parameters
    ----------
    db_path : Path to the SQLite database (same file as WorkspaceManager).
    lock    : Optional shared threading.Lock.  Pass WorkspaceManager._lock
              to serialise all writes through a single lock.  If None, a
              new lock is created (safe for standalone use).
    """

    def __init__(
        self,
        db_path: str = "local_search_agent.db",
        lock: Optional[threading.Lock] = None,
    ):
        self._db_path = db_path
        self._lock = lock if lock is not None else threading.Lock()
        self._init_db()

    # ------------------------------------------------------------------
    # DB init
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")   # concurrent readers + one writer
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)
            conn.executescript(_INDEX_SQL)
        logger.debug("UIStore DB initialised at %r", self._db_path)

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    def create_session(self, workspace: str, title: str = "") -> dict:
        """Create a new chat session and return it as a dict."""
        now = _now_iso()
        session_id = _new_id()
        # Auto-title: "Chat YYYY-MM-DD HH:MM" if none provided
        if not title:
            title = "Chat " + datetime.now().strftime("%Y-%m-%d %H:%M")
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO chat_sessions (session_id, workspace, title, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (session_id, workspace, title, now, now),
            )
        logger.debug("Session created: %r (workspace=%r)", session_id, workspace)
        return {"session_id": session_id, "workspace": workspace,
                "title": title, "created_at": now, "updated_at": now}

    def list_sessions(self, workspace: str, limit: int = 50) -> list[dict]:
        """Return sessions for a workspace, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM chat_sessions WHERE workspace = ?"
                " ORDER BY updated_at DESC LIMIT ?",
                (workspace, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_session(self, session_id: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM chat_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        return dict(row) if row else None

    def rename_session(self, session_id: str, title: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE session_id = ?",
                (title, _now_iso(), session_id),
            )

    def delete_session(self, session_id: str) -> None:
        """Delete a session and all its messages."""
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
        logger.debug("Session deleted: %r", session_id)

    def _touch_session(self, session_id: str, conn: sqlite3.Connection) -> None:
        """Update updated_at on the parent session (call inside an open transaction)."""
        conn.execute(
            "UPDATE chat_sessions SET updated_at = ? WHERE session_id = ?",
            (_now_iso(), session_id),
        )

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tool_calls: Optional[list] = None,
        thinking: str = "",
        token_query: int = 0,
        token_reply: int = 0,
    ) -> dict:
        """
        Append a message to a session.

        Parameters
        ----------
        session_id  : Parent session ID.
        role        : "user" or "assistant".
        content     : Full markdown text of the message.
        tool_calls  : List of dicts {tool, input, output, duration_ms}.
        thinking    : Raw thinking block text (assistant only).
        token_query : Tokens in the query for this turn.
        token_reply : Tokens in the reply for this turn.
        """
        now = _now_iso()
        message_id = _new_id()
        tool_calls_json = json.dumps(tool_calls or [])
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO chat_messages
                   (message_id, session_id, role, content, tool_calls, thinking,
                    token_query, token_reply, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (message_id, session_id, role, content, tool_calls_json,
                 thinking, token_query, token_reply, now),
            )
            self._touch_session(session_id, conn)
        return {
            "message_id": message_id, "session_id": session_id, "role": role,
            "content": content, "tool_calls": tool_calls or [],
            "thinking": thinking, "token_query": token_query,
            "token_reply": token_reply, "created_at": now,
        }

    def list_messages(self, session_id: str) -> list[dict]:
        """Return all messages for a session, oldest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM chat_messages WHERE session_id = ? ORDER BY created_at ASC",
                (session_id,),
            ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["tool_calls"] = json.loads(d["tool_calls"])
            result.append(d)
        return result

    def session_token_totals(self, session_id: str) -> dict:
        """Return summed token counts for a session."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT SUM(token_query) as tq, SUM(token_reply) as tr"
                " FROM chat_messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        tq = row["tq"] or 0
        tr = row["tr"] or 0
        return {"token_query": tq, "token_reply": tr, "token_total": tq + tr}

    # ------------------------------------------------------------------
    # UI config key/value store
    # ------------------------------------------------------------------

    def get_config(self, key: str, default=None):
        """Return the decoded value for a config key, or default if absent."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM ui_config WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except (json.JSONDecodeError, TypeError):
            return row["value"]

    def set_config(self, key: str, value) -> None:
        """Upsert a config value (value is JSON-encoded)."""
        now = _now_iso()
        encoded = json.dumps(value)
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO ui_config (key, value, updated_at) VALUES (?, ?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (key, encoded, now),
            )

    def get_all_config(self) -> dict:
        """Return all config as a flat dict {key: decoded_value}."""
        with self._connect() as conn:
            rows = conn.execute("SELECT key, value FROM ui_config").fetchall()
        result = {}
        for r in rows:
            try:
                result[r["key"]] = json.loads(r["value"])
            except (json.JSONDecodeError, TypeError):
                result[r["key"]] = r["value"]
        return result
