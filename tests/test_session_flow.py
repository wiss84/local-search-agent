"""
Tests for multi-tenant RBAC: browser session flow.

Covers:
- AuthDB.auth_sessions: display_name/is_superadmin persisted + migration
  idempotency (mirrors the chat_sessions.created_by migration test pattern)
- APIKeyIdentityProvider.login() / resolve_session() / logout(): valid,
  invalid, expired, sliding expiry (extends but caps at max lifetime)
- APIKeyIdentityProvider.resolve(): cookie-first, bearer-header fallback
- End-to-end via TestClient + AuthorizationMiddleware + the real
  /api/auth/login and /api/auth/logout routes: login sets a cookie,
  a subsequent request authenticates via that cookie alone (no bearer
  header), logout immediately invalidates it.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from local_search_agent.auth.api_key_provider import (
    SESSION_COOKIE_NAME,
    APIKeyIdentityProvider,
)
from local_search_agent.auth.authorization_middleware import AuthorizationMiddleware
from local_search_agent.auth.session_routes import build_auth_router
from local_search_agent.workspace.auth_db import AuthDB

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def auth_db(tmp_path):
    return AuthDB(db_path=str(tmp_path / "test.db"))


@pytest.fixture
def provider(auth_db):
    return APIKeyIdentityProvider(auth_db)


class _FakeRequest:
    def __init__(self, headers: dict = None, cookies: dict = None):
        self.headers = headers or {}
        self.cookies = cookies or {}


# ---------------------------------------------------------------------------
# AuthDB: auth_sessions schema (display_name / is_superadmin) + migration
# ---------------------------------------------------------------------------


class TestAuthSessionsSchema:
    def test_create_session_persists_display_name_and_superadmin(self, auth_db):
        expires = (datetime.now().astimezone() + timedelta(hours=1)).isoformat()
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

    def test_create_session_without_new_fields_defaults(self, auth_db):
        # Phase 1's original call shape — must keep working unchanged.
        expires = (datetime.now().astimezone() + timedelta(hours=1)).isoformat()
        auth_db.create_session("hash1", "alice@acme.com", expires_at=expires)
        row = auth_db.get_session("hash1")
        assert row["display_name"] == ""
        assert row["is_superadmin"] == 0

    def test_migration_idempotent_on_existing_db(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        AuthDB(db_path=db_path)
        # Second construction against the same file must not raise even
        # though display_name/is_superadmin already exist.
        AuthDB(db_path=db_path)

    def test_columns_present_via_pragma(self, auth_db):
        conn = sqlite3.connect(auth_db._db_path)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(auth_sessions)").fetchall()}
        conn.close()
        assert "display_name" in cols
        assert "is_superadmin" in cols


# ---------------------------------------------------------------------------
# APIKeyIdentityProvider: login / resolve_session / logout
# ---------------------------------------------------------------------------


class TestLogin:
    def test_login_valid_key_returns_token_and_expiry(self, provider):
        _, raw_key = provider.create_key(subject="alice@acme.com", created_by="admin")
        result = provider.login(raw_key)
        assert result is not None
        token, expires_at = result
        assert token
        assert isinstance(expires_at, datetime)

    def test_login_invalid_key_returns_none(self, provider):
        assert provider.login("lsa_bad_key") is None

    def test_login_revoked_key_returns_none(self, provider):
        key_id, raw_key = provider.create_key(subject="alice@acme.com", created_by="admin")
        provider.revoke_key(key_id)
        assert provider.login(raw_key) is None

    def test_login_creates_session_with_correct_subject(self, provider, auth_db):
        _, raw_key = provider.create_key(subject="alice@acme.com", created_by="admin")
        token, _ = provider.login(raw_key)
        identity = provider.resolve_session(token)
        assert identity is not None
        assert identity.subject == "alice@acme.com"

    def test_raw_token_never_equals_stored_hash(self, provider, auth_db):
        _, raw_key = provider.create_key(subject="alice@acme.com", created_by="admin")
        token, _ = provider.login(raw_key)
        # Can't directly introspect the hash without the internal helper,
        # but we can confirm the raw token isn't usable as a dict key lookup
        # against get_session (which expects a hash, not the raw token).
        assert auth_db.get_session(token) is None


class TestResolveSession:
    def test_valid_session_resolves_identity(self, provider):
        _, raw_key = provider.create_key(
            subject="alice@acme.com", created_by="admin", display_name="Alice"
        )
        token, _ = provider.login(raw_key)
        identity = provider.resolve_session(token)
        assert identity.subject == "alice@acme.com"
        assert identity.display_name == "Alice"

    def test_unknown_token_returns_none(self, provider):
        assert provider.resolve_session("nonexistent-token") is None

    def test_empty_token_returns_none(self, provider):
        assert provider.resolve_session("") is None

    def test_expired_session_returns_none_and_is_deleted(self, provider, auth_db):
        _, raw_key = provider.create_key(subject="alice@acme.com", created_by="admin")
        token, _ = provider.login(raw_key)

        # Manually backdate expires_at past now.
        conn = sqlite3.connect(auth_db._db_path)
        past = (datetime.now().astimezone() - timedelta(minutes=5)).isoformat()
        conn.execute("UPDATE auth_sessions SET expires_at = ?", (past,))
        conn.commit()
        conn.close()

        assert provider.resolve_session(token) is None
        # Swept on the way out.
        conn = sqlite3.connect(auth_db._db_path)
        count = conn.execute("SELECT COUNT(*) FROM auth_sessions").fetchone()[0]
        conn.close()
        assert count == 0

    def test_sliding_expiry_extends_on_activity(self, provider, auth_db):
        _, raw_key = provider.create_key(subject="alice@acme.com", created_by="admin")
        token, original_expiry = provider.login(raw_key)

        # Backdate created_at/expires_at so there's clear room to extend
        # without hitting the max-lifetime cap.
        conn = sqlite3.connect(auth_db._db_path)
        created = (datetime.now().astimezone() - timedelta(hours=1)).isoformat()
        near_expiry = (datetime.now().astimezone() + timedelta(minutes=5)).isoformat()
        conn.execute(
            "UPDATE auth_sessions SET created_at = ?, expires_at = ?", (created, near_expiry)
        )
        conn.commit()
        conn.close()

        provider.resolve_session(token)

        # get_session expects hash; verify via raw SQL instead
        conn = sqlite3.connect(auth_db._db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT expires_at FROM auth_sessions WHERE subject = ?", ("alice@acme.com",)
        ).fetchone()
        conn.close()
        new_expiry = datetime.fromisoformat(row["expires_at"])
        original_expiry_dt = datetime.fromisoformat(near_expiry)
        assert new_expiry > original_expiry_dt

    def test_sliding_expiry_capped_at_max_lifetime(self, provider, auth_db):
        _, raw_key = provider.create_key(subject="alice@acme.com", created_by="admin")
        token, _ = provider.login(raw_key)

        # Session created 23.9 hours ago — close to the 24h hard cap.
        # Sliding expiry (+2h idle timeout) would normally push expires_at
        # well past the 24h cap; it must be clamped to created_at + 24h instead.
        conn = sqlite3.connect(auth_db._db_path)
        created_at = datetime.now().astimezone() - timedelta(hours=23, minutes=54)
        near_expiry = datetime.now().astimezone() + timedelta(minutes=1)
        conn.execute(
            "UPDATE auth_sessions SET created_at = ?, expires_at = ?",
            (created_at.isoformat(), near_expiry.isoformat()),
        )
        conn.commit()
        conn.close()

        identity = provider.resolve_session(token)
        assert identity is not None

        # Inspect the row directly (get_session needs a hash we don't have
        # here, so query the table by subject instead).
        conn = sqlite3.connect(auth_db._db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM auth_sessions WHERE subject = ?", ("alice@acme.com",)
        ).fetchone()
        conn.close()
        new_expiry = datetime.fromisoformat(row["expires_at"])
        max_allowed = created_at + timedelta(hours=24)
        assert new_expiry <= max_allowed + timedelta(
            seconds=2
        )  # small tolerance for test execution time


class TestLogout:
    def test_logout_invalidates_session(self, provider):
        _, raw_key = provider.create_key(subject="alice@acme.com", created_by="admin")
        token, _ = provider.login(raw_key)
        assert provider.resolve_session(token) is not None
        provider.logout(token)
        assert provider.resolve_session(token) is None

    def test_logout_unknown_token_is_noop(self, provider):
        provider.logout("nonexistent-token")  # must not raise


# ---------------------------------------------------------------------------
# resolve(): cookie-first, bearer-header fallback
# ---------------------------------------------------------------------------


class TestResolveCookieFirst:
    def test_valid_cookie_used(self, provider):
        _, raw_key = provider.create_key(subject="alice@acme.com", created_by="admin")
        token, _ = provider.login(raw_key)
        request = _FakeRequest(cookies={SESSION_COOKIE_NAME: token})
        identity = provider.resolve(request)
        assert identity is not None
        assert identity.subject == "alice@acme.com"

    def test_invalid_cookie_falls_through_to_bearer(self, provider):
        _, raw_key = provider.create_key(subject="alice@acme.com", created_by="admin")
        request = _FakeRequest(
            cookies={SESSION_COOKIE_NAME: "bogus-token"},
            headers={"Authorization": f"Bearer {raw_key}"},
        )
        identity = provider.resolve(request)
        assert identity is not None
        assert identity.subject == "alice@acme.com"

    def test_no_cookie_uses_bearer(self, provider):
        _, raw_key = provider.create_key(subject="alice@acme.com", created_by="admin")
        request = _FakeRequest(headers={"Authorization": f"Bearer {raw_key}"})
        identity = provider.resolve(request)
        assert identity is not None

    def test_neither_returns_none(self, provider):
        request = _FakeRequest()
        assert provider.resolve(request) is None


# ---------------------------------------------------------------------------
# End-to-end: /api/auth/login sets a cookie, authenticates, /logout clears it
# ---------------------------------------------------------------------------


class _FakeConfig:
    def __init__(self, identity_provider):
        self.identity_provider = identity_provider


def _build_e2e_app(config, auth_db) -> FastAPI:
    app = FastAPI()
    app.add_middleware(AuthorizationMiddleware, config=config, auth_db=auth_db)
    app.include_router(
        build_auth_router(
            config.identity_provider,
            cookie_secure=False,
            cookie_httponly=False,
            cookie_samesite="lax",
        )
    )

    @app.get("/api/ui/sessions")
    async def sessions(workspace: str, request: Request):
        return JSONResponse({"subject": request.state.identity.subject, "role": request.state.role})

    return app


class TestEndToEndSessionFlow:
    def test_login_then_authenticated_request_via_cookie_only(self, provider, auth_db):
        config = _FakeConfig(identity_provider=provider)
        app = _build_e2e_app(config, auth_db)
        client = TestClient(app)

        _, raw_key = provider.create_key(subject="alice@acme.com", created_by="admin")
        auth_db.grant_access("finance", "alice@acme.com", "member", granted_by="admin")

        login_resp = client.post("/api/auth/login", json={"api_key": raw_key})
        assert login_resp.status_code == 200
        assert SESSION_COOKIE_NAME in login_resp.cookies
        # Raw session token must never appear in the JSON body.
        assert raw_key not in login_resp.text
        assert login_resp.json() == {"ok": True}

        # No Authorization header at all — cookie alone must authenticate.
        resp = client.get("/api/ui/sessions", params={"workspace": "finance"})
        assert resp.status_code == 200
        assert resp.json()["subject"] == "alice@acme.com"

    def test_login_wrong_key_returns_401(self, provider, auth_db):
        config = _FakeConfig(identity_provider=provider)
        app = _build_e2e_app(config, auth_db)
        client = TestClient(app)

        resp = client.post("/api/auth/login", json={"api_key": "lsa_wrong_key"})
        assert resp.status_code == 401

    def test_logout_invalidates_cookie_immediately(self, provider, auth_db):
        config = _FakeConfig(identity_provider=provider)
        app = _build_e2e_app(config, auth_db)
        client = TestClient(app)

        _, raw_key = provider.create_key(subject="alice@acme.com", created_by="admin")
        auth_db.grant_access("finance", "alice@acme.com", "member", granted_by="admin")

        client.post("/api/auth/login", json={"api_key": raw_key})
        assert client.get("/api/ui/sessions", params={"workspace": "finance"}).status_code == 200

        logout_resp = client.post("/api/auth/logout")
        assert logout_resp.status_code == 200

        # Same client (cookie jar still holds whatever's left after
        # delete_cookie) — the session row is gone either way, so this
        # must now fail even if the browser somehow still sent the cookie.
        resp = client.get("/api/ui/sessions", params={"workspace": "finance"})
        assert resp.status_code == 401

    def test_logout_without_prior_login_is_idempotent(self, provider, auth_db):
        config = _FakeConfig(identity_provider=provider)
        app = _build_e2e_app(config, auth_db)
        client = TestClient(app)
        resp = client.post("/api/auth/logout")
        assert resp.status_code == 200

    def test_login_sets_secure_cookie_headers(self, provider, auth_db):
        config = _FakeConfig(identity_provider=provider)
        app = FastAPI()
        app.add_middleware(AuthorizationMiddleware, config=config, auth_db=auth_db)
        app.include_router(build_auth_router(config.identity_provider))

        client = TestClient(app)

        _, raw_key = provider.create_key(subject="alice@acme.com", created_by="admin")
        auth_db.grant_access("finance", "alice@acme.com", "member", granted_by="admin")

        login_resp = client.post("/api/auth/login", json={"api_key": raw_key})
        set_cookie = login_resp.headers.get("set-cookie", "")
        assert "HttpOnly" in set_cookie
        assert "Secure" in set_cookie
        assert "SameSite=strict" in set_cookie
