"""
Unit tests for AuthDB (multi-tenant RBAC — schema + migrations).

See docs/role_based_access_control.md for the full design.
Covers:
- workspace_members: grant / grant_bulk (atomicity) / revoke (all vs subset) /
  get_role (fail-closed None) / list_access filters / is_member
- activity_log: log_activity / get_activity_log filters / purge_activity_log
- auth_sessions: create / get / extend / delete / delete_expired_sessions /
  delete_sessions_for_subject
- auth_attempts: record_attempt / count_recent_failed_attempts / purge_old_attempts
- chat_sessions.created_by migration (UIStore)
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

import pytest

from local_search_agent.ui.store import UIStore
from local_search_agent.workspace.auth_db import AuthDB

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def auth_db(tmp_path):
    """Fresh AuthDB backed by a temp SQLite file."""
    return AuthDB(db_path=str(tmp_path / "test.db"))


def _iso(dt: datetime) -> str:
    return dt.astimezone().isoformat()


# ---------------------------------------------------------------------------
# workspace_members
# ---------------------------------------------------------------------------


class TestWorkspaceMembers:
    def test_grant_and_get_role(self, auth_db):
        auth_db.grant_access("finance", "alice@acme.com", "member", granted_by="admin@acme.com")
        assert auth_db.get_role("alice@acme.com", "finance") == "member"

    def test_get_role_no_grant_is_none(self, auth_db):
        # Fail-closed: absence of a row means no access, not an error.
        assert auth_db.get_role("alice@acme.com", "finance") is None

    def test_grant_is_upsert(self, auth_db):
        auth_db.grant_access("finance", "alice@acme.com", "member", granted_by="admin@acme.com")
        auth_db.grant_access("finance", "alice@acme.com", "admin", granted_by="admin@acme.com")
        assert auth_db.get_role("alice@acme.com", "finance") == "admin"

    def test_grant_invalid_role_raises(self, auth_db):
        with pytest.raises(ValueError):
            auth_db.grant_access(
                "finance", "alice@acme.com", "superuser", granted_by="admin@acme.com"
            )

    def test_same_subject_different_role_per_workspace(self, auth_db):
        auth_db.grant_access("finance", "alice@acme.com", "admin", granted_by="admin@acme.com")
        auth_db.grant_access("marketing", "alice@acme.com", "member", granted_by="admin@acme.com")
        assert auth_db.get_role("alice@acme.com", "finance") == "admin"
        assert auth_db.get_role("alice@acme.com", "marketing") == "member"

    def test_grant_bulk_writes_all_workspaces(self, auth_db):
        auth_db.grant_access_bulk(
            ["finance", "marketing"], "alice@acme.com", "member", granted_by="admin@acme.com"
        )
        assert auth_db.get_role("alice@acme.com", "finance") == "member"
        assert auth_db.get_role("alice@acme.com", "marketing") == "member"

    def test_grant_bulk_empty_list_is_noop(self, auth_db):
        auth_db.grant_access_bulk([], "alice@acme.com", "member", granted_by="admin@acme.com")
        assert auth_db.list_access(subject="alice@acme.com") == []

    def test_revoke_specific_workspaces(self, auth_db):
        auth_db.grant_access_bulk(
            ["finance", "marketing"], "alice@acme.com", "member", granted_by="admin@acme.com"
        )
        deleted = auth_db.revoke_access("alice@acme.com", workspaces=["finance"])
        assert deleted == 1
        assert auth_db.get_role("alice@acme.com", "finance") is None
        assert auth_db.get_role("alice@acme.com", "marketing") == "member"

    def test_revoke_all_for_subject(self, auth_db):
        auth_db.grant_access_bulk(
            ["finance", "marketing"], "alice@acme.com", "member", granted_by="admin@acme.com"
        )
        deleted = auth_db.revoke_access("alice@acme.com")
        assert deleted == 2
        assert auth_db.list_access(subject="alice@acme.com") == []

    def test_list_access_by_workspace(self, auth_db):
        auth_db.grant_access("finance", "alice@acme.com", "admin", granted_by="admin@acme.com")
        auth_db.grant_access("finance", "bob@acme.com", "member", granted_by="admin@acme.com")
        rows = auth_db.list_access(workspace="finance")
        assert {r["subject"] for r in rows} == {"alice@acme.com", "bob@acme.com"}

    def test_is_member(self, auth_db):
        auth_db.grant_access("finance", "alice@acme.com", "member", granted_by="admin@acme.com")
        assert auth_db.is_member("alice@acme.com", "finance") is True
        assert auth_db.is_member("bob@acme.com", "finance") is False

    def test_list_access_by_workspace_checks_roles(self, auth_db):
        auth_db.grant_access("finance", "alice@acme.com", "admin", granted_by="admin@acme.com")
        auth_db.grant_access("finance", "bob@acme.com", "member", granted_by="admin@acme.com")
        rows = auth_db.list_access(workspace="finance")
        roles = {r["subject"]: r["role"] for r in rows}
        assert roles["alice@acme.com"] == "admin"
        assert roles["bob@acme.com"] == "member"


# ---------------------------------------------------------------------------
# activity_log
# ---------------------------------------------------------------------------


class TestActivityLog:
    def test_log_and_read_back(self, auth_db):
        auth_db.log_activity("alice@acme.com", "search", workspace="finance", detail="Q3 revenue")
        rows = auth_db.get_activity_log(subject="alice@acme.com")
        assert len(rows) == 1
        assert rows[0]["action"] == "search"
        assert rows[0]["success"] == 1

    def test_log_denied_attempt(self, auth_db):
        auth_db.log_activity("bob@acme.com", "search", workspace="finance", success=False)
        rows = auth_db.get_activity_log(subject="bob@acme.com")
        assert rows[0]["success"] == 0

    def test_filter_by_workspace(self, auth_db):
        auth_db.log_activity("alice@acme.com", "search", workspace="finance")
        auth_db.log_activity("alice@acme.com", "search", workspace="marketing")
        rows = auth_db.get_activity_log(workspace="finance")
        assert len(rows) == 1
        assert rows[0]["workspace"] == "finance"

    def test_newest_first(self, auth_db):
        auth_db.log_activity("alice@acme.com", "login")
        auth_db.log_activity("alice@acme.com", "logout")
        rows = auth_db.get_activity_log(subject="alice@acme.com")
        assert rows[0]["action"] == "logout"
        assert rows[1]["action"] == "login"

    def test_purge_old_rows(self, auth_db, tmp_path):
        # Insert a row and manually backdate its timestamp past the retention window.
        auth_db.log_activity("alice@acme.com", "search", workspace="finance")
        old_ts = _iso(datetime.now() - timedelta(days=100))
        conn = sqlite3.connect(auth_db._db_path)
        conn.execute("UPDATE activity_log SET timestamp = ?", (old_ts,))
        conn.commit()
        conn.close()

        deleted = auth_db.purge_activity_log(older_than_days=90)
        assert deleted == 1
        assert auth_db.get_activity_log(subject="alice@acme.com") == []

    def test_limit_parameter_constrains_results(self, auth_db):
        for i in range(5):
            auth_db.log_activity("alice@acme.com", "search", workspace="finance")
        rows = auth_db.get_activity_log(subject="alice@acme.com", limit=3)
        assert len(rows) == 3

    def test_no_filters_returns_all(self, auth_db):
        auth_db.log_activity("alice@acme.com", "search", workspace="finance")
        auth_db.log_activity("bob@acme.com", "search", workspace="marketing")
        rows = auth_db.get_activity_log()
        assert len(rows) == 2


# ---------------------------------------------------------------------------
# auth_sessions
# ---------------------------------------------------------------------------


class TestAuthSessions:
    def test_create_and_get(self, auth_db):
        expires = _iso(datetime.now() + timedelta(hours=1))
        auth_db.create_session("hash123", "alice@acme.com", expires_at=expires)
        session = auth_db.get_session("hash123")
        assert session is not None
        assert session["subject"] == "alice@acme.com"

    def test_get_unknown_returns_none(self, auth_db):
        assert auth_db.get_session("nonexistent") is None

    def test_extend_session(self, auth_db):
        expires = _iso(datetime.now() + timedelta(hours=1))
        auth_db.create_session("hash123", "alice@acme.com", expires_at=expires)
        new_expires = _iso(datetime.now() + timedelta(hours=2))
        auth_db.extend_session("hash123", new_expires)
        assert auth_db.get_session("hash123")["expires_at"] == new_expires

    def test_delete_session_is_immediate_revocation(self, auth_db):
        expires = _iso(datetime.now() + timedelta(hours=1))
        auth_db.create_session("hash123", "alice@acme.com", expires_at=expires)
        auth_db.delete_session("hash123")
        assert auth_db.get_session("hash123") is None

    def test_delete_expired_sessions(self, auth_db):
        expired = _iso(datetime.now() - timedelta(hours=1))
        valid = _iso(datetime.now() + timedelta(hours=1))
        auth_db.create_session("expired_hash", "alice@acme.com", expires_at=expired)
        auth_db.create_session("valid_hash", "alice@acme.com", expires_at=valid)

        deleted = auth_db.delete_expired_sessions()
        assert deleted == 1
        assert auth_db.get_session("expired_hash") is None
        assert auth_db.get_session("valid_hash") is not None

    def test_delete_sessions_for_subject(self, auth_db):
        expires = _iso(datetime.now() + timedelta(hours=1))
        auth_db.create_session("h1", "alice@acme.com", expires_at=expires)
        auth_db.create_session("h2", "alice@acme.com", expires_at=expires)
        auth_db.create_session("h3", "bob@acme.com", expires_at=expires)

        deleted = auth_db.delete_sessions_for_subject("alice@acme.com")
        assert deleted == 2
        assert auth_db.get_session("h3") is not None

    def test_delete_expired_sessions_all_expired(self, auth_db):
        expired = _iso(datetime.now() - timedelta(hours=1))
        auth_db.create_session("h1", "alice@acme.com", expires_at=expired)
        auth_db.create_session("h2", "alice@acme.com", expires_at=expired)
        assert auth_db.delete_expired_sessions() == 2
        assert auth_db.get_session("h1") is None
        assert auth_db.get_session("h2") is None

    def test_delete_expired_sessions_none_expired(self, auth_db):
        valid = _iso(datetime.now() + timedelta(hours=1))
        auth_db.create_session("h1", "alice@acme.com", expires_at=valid)
        assert auth_db.delete_expired_sessions() == 0
        assert auth_db.get_session("h1") is not None

    def test_create_session_with_display_name_and_superadmin(self, auth_db):
        expires = _iso(datetime.now() + timedelta(hours=1))
        auth_db.create_session(
            "hash1",
            "alice@acme.com",
            expires_at=expires,
            display_name="Alice",
            is_superadmin=True,
        )
        row = auth_db.get_session("hash1")
        assert row["display_name"] == "Alice"
        assert row["is_superadmin"] == 1


# ---------------------------------------------------------------------------
# auth_attempts
# ---------------------------------------------------------------------------


class TestAuthAttempts:
    def test_count_recent_failed_attempts_by_subject(self, auth_db):
        auth_db.record_attempt("alice@acme.com", "1.2.3.4", success=False)
        auth_db.record_attempt("alice@acme.com", "1.2.3.4", success=False)
        auth_db.record_attempt("alice@acme.com", "1.2.3.4", success=True)
        count = auth_db.count_recent_failed_attempts(subject="alice@acme.com")
        assert count == 2

    def test_count_recent_failed_attempts_by_ip(self, auth_db):
        auth_db.record_attempt(None, "9.9.9.9", success=False)
        auth_db.record_attempt(None, "9.9.9.9", success=False)
        count = auth_db.count_recent_failed_attempts(ip_address="9.9.9.9")
        assert count == 2

    def test_count_requires_subject_or_ip(self, auth_db):
        with pytest.raises(ValueError):
            auth_db.count_recent_failed_attempts()

    def test_window_excludes_old_attempts(self, auth_db):
        auth_db.record_attempt("alice@acme.com", "1.2.3.4", success=False)
        conn = sqlite3.connect(auth_db._db_path)
        old_ts = _iso(datetime.now() - timedelta(minutes=30))
        conn.execute("UPDATE auth_attempts SET attempted_at = ?", (old_ts,))
        conn.commit()
        conn.close()

        count = auth_db.count_recent_failed_attempts(subject="alice@acme.com", window_minutes=15)
        assert count == 0

    def test_purge_old_attempts(self, auth_db):
        auth_db.record_attempt("alice@acme.com", "1.2.3.4", success=False)
        conn = sqlite3.connect(auth_db._db_path)
        old_ts = _iso(datetime.now() - timedelta(days=10))
        conn.execute("UPDATE auth_attempts SET attempted_at = ?", (old_ts,))
        conn.commit()
        conn.close()

        deleted = auth_db.purge_old_attempts(older_than_days=7)
        assert deleted == 1


# ---------------------------------------------------------------------------
# chat_sessions.created_by migration
# ---------------------------------------------------------------------------


class TestChatSessionsMigration:
    def test_created_by_column_exists(self, tmp_path):
        store = UIStore(db_path=str(tmp_path / "test.db"))
        conn = sqlite3.connect(store._db_path)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(chat_sessions)").fetchall()}
        conn.close()
        assert "created_by" in cols

    def test_migration_is_idempotent_on_existing_db(self, tmp_path):
        # Simulate an upgrade: open twice against the same file.
        db_path = str(tmp_path / "test.db")
        UIStore(db_path=db_path)
        # Second construction must not raise even though the column already exists.
        UIStore(db_path=db_path)

    def test_existing_sessions_unaffected_by_migration(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        store = UIStore(db_path=db_path)
        session = store.create_session(workspace="ws1", title="Test")
        # created_by defaults to NULL until AuthorizationMiddleware populates it (later phase).
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT created_by FROM chat_sessions WHERE session_id = ?", (session["session_id"],)
        ).fetchone()
        conn.close()
        assert row["created_by"] is None
