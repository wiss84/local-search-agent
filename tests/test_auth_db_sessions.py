"""
Unit tests for AuthDB auth_sessions cleanup methods.

Covers:
- delete_expired_sessions removes only expired rows
- delete_expired_sessions returns count of deleted rows
- delete_expired_sessions is safe to call on empty table
- delete_sessions_for_subject removes only that subject's sessions
- delete_sessions_for_subject returns row count
- delete_sessions_for_subject with non-existent subject returns 0
- create_session + get_session round-trip via hash lookup
- extend_session updates expires_at
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from local_search_agent.workspace.auth_db import AuthDB


@pytest.fixture
def auth_db(tmp_path):
    db_path = str(tmp_path / "test_auth.db")
    return AuthDB(db_path=db_path)


def _make_session(auth_db, subject="alice", token_hash="abc123", offset_minutes=60):
    expires = (datetime.now().astimezone() + timedelta(minutes=offset_minutes)).isoformat()
    auth_db.create_session(
        token_hash=token_hash,
        subject=subject,
        expires_at=expires,
        display_name="Alice",
        is_superadmin=False,
    )


class TestDeleteExpiredSessions:
    def test_removes_expired_rows(self, auth_db):
        _make_session(auth_db, token_hash="expired1", offset_minutes=-10)
        _make_session(auth_db, token_hash="expired2", offset_minutes=-30)
        _make_session(auth_db, token_hash="active1", offset_minutes=60)

        deleted = auth_db.delete_expired_sessions()

        assert deleted == 2
        assert auth_db.get_session("expired1") is None
        assert auth_db.get_session("expired2") is None
        assert auth_db.get_session("active1") is not None

    def test_returns_zero_when_no_expired(self, auth_db):
        _make_session(auth_db, token_hash="active1", offset_minutes=60)

        deleted = auth_db.delete_expired_sessions()

        assert deleted == 0
        assert auth_db.get_session("active1") is not None

    def test_safe_on_empty_table(self, auth_db):
        deleted = auth_db.delete_expired_sessions()
        assert deleted == 0


class TestDeleteSessionsForSubject:
    def test_removes_all_sessions_for_subject(self, auth_db):
        _make_session(auth_db, subject="alice", token_hash="alice1")
        _make_session(auth_db, subject="alice", token_hash="alice2")
        _make_session(auth_db, subject="bob", token_hash="bob1")

        deleted = auth_db.delete_sessions_for_subject("alice")

        assert deleted == 2
        assert auth_db.get_session("alice1") is None
        assert auth_db.get_session("alice2") is None
        assert auth_db.get_session("bob1") is not None

    def test_returns_zero_for_missing_subject(self, auth_db):
        _make_session(auth_db, subject="alice", token_hash="alice1")

        deleted = auth_db.delete_sessions_for_subject("nobody")

        assert deleted == 0
        assert auth_db.get_session("alice1") is not None

    def test_empty_result_on_empty_table(self, auth_db):
        deleted = auth_db.delete_sessions_for_subject("anyone")
        assert deleted == 0


class TestSessionRoundTrip:
    def test_create_and_get_session(self, auth_db):
        _make_session(auth_db, token_hash="tok_abc", subject="alice")
        row = auth_db.get_session("tok_abc")
        assert row is not None
        assert row["subject"] == "alice"
        assert row["token_hash"] == "tok_abc"

    def test_get_missing_session_returns_none(self, auth_db):
        assert auth_db.get_session("nonexistent") is None

    def test_extend_session_updates_expiry(self, auth_db):
        _make_session(auth_db, token_hash="tok_ext", offset_minutes=5)
        original = auth_db.get_session("tok_ext")["expires_at"]

        new_expiry = (datetime.now().astimezone() + timedelta(hours=2)).isoformat()
        auth_db.extend_session("tok_ext", new_expiry)

        updated = auth_db.get_session("tok_ext")
        assert updated["expires_at"] == new_expiry
        assert updated["expires_at"] != original
