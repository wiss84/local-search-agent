"""
Tests for two gaps in (activity logging):

1. Login rate limiting -- APIKeyIdentityProvider.login() now checks
   AuthDB.count_recent_failed_attempts()/record_attempt() before/after
   verifying a key (security checklist item, see api_key_provider.py's
   _MAX_FAILED_LOGIN_ATTEMPTS/_LOGIN_ATTEMPT_WINDOW_MINUTES).

2. Session ownership -- route_policy.py gained RoutePolicy.workspace_from_session_id
   for the /api/ui/sessions/{session_id}... family (workspace only knowable
   via a DB lookup on the session row), resolved by
   AuthorizationMiddleware._resolve_workspace_from_session() using a
   session_lookup callback (UIStore.get_session_workspace in production).
   chat_sessions.created_by is now populated on create and used by
   list_sessions()'s ownership filter (?all=true lets admins bypass it).
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from local_search_agent.auth.api_key_provider import APIKeyIdentityProvider
from local_search_agent.auth.authorization_middleware import AuthorizationMiddleware
from local_search_agent.ui.api_routes import build_ui_router
from local_search_agent.ui.store import UIStore
from local_search_agent.workspace.auth_db import AuthDB
from local_search_agent.workspace.workspace_manager import WorkspaceManager


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test.db")


@pytest.fixture
def auth_db(db_path):
    return AuthDB(db_path=db_path)


@pytest.fixture
def provider(auth_db):
    return APIKeyIdentityProvider(auth_db)


# ---------------------------------------------------------------------------
# 1. Login rate limiting
# ---------------------------------------------------------------------------


class TestLoginRateLimiting:
    def test_blocks_after_max_failed_attempts_from_same_ip(self, provider):
        for _ in range(10):
            assert provider.login("lsa_bad_bad", ip_address="1.2.3.4") is None

        # 11th attempt from the same IP is now rate-limited *before* even
        # checking the key -- a valid key would still be rejected.
        _, raw_key = provider.create_key(subject="alice@acme.com", created_by="root")
        assert provider.login(raw_key, ip_address="1.2.3.4") is None

    def test_different_ip_is_unaffected(self, provider):
        for _ in range(10):
            provider.login("lsa_bad_bad", ip_address="1.2.3.4")

        _, raw_key = provider.create_key(subject="alice@acme.com", created_by="root")
        result = provider.login(raw_key, ip_address="9.9.9.9")
        assert result is not None

    def test_successful_logins_do_not_count_against_the_limit(self, provider):
        _, raw_key = provider.create_key(subject="alice@acme.com", created_by="root")
        for _ in range(10):
            assert provider.login(raw_key, ip_address="5.5.5.5") is not None
        # Still succeeds -- only *failed* attempts count toward the limit.
        assert provider.login(raw_key, ip_address="5.5.5.5") is not None

    def test_no_ip_address_skips_rate_limit_check(self, provider):
        # ip_address=None (e.g. a test harness or a proxy that stripped it)
        # -- can't rate-limit what you can't key on, so this just falls
        # through to normal key verification rather than raising.
        _, raw_key = provider.create_key(subject="alice@acme.com", created_by="root")
        assert provider.login(raw_key) is not None


# ---------------------------------------------------------------------------
# 2. Session ownership / workspace_from_session_id
# ---------------------------------------------------------------------------


class _FakeFramework:
    def delete_workspace(self, name, wipe_index=False):
        pass


class _FakeConfig:
    def __init__(self, db_path, identity_provider=None):
        self.db_path = db_path
        self.identity_provider = identity_provider


class _FakeAppState:
    def __init__(self, db_path, identity_provider=None):
        self.config = _FakeConfig(db_path, identity_provider=identity_provider)
        self.auth_db = AuthDB(db_path=db_path)
        self.workspace_manager = WorkspaceManager(db_path=db_path)
        self.store = UIStore(db_path=db_path)
        self.framework = _FakeFramework()
        self.scheduler = None
        self.watcher = None

    def get_agent(self, workspace=None):
        raise RuntimeError("no agent configured in tests")


def _build_app(app_state) -> FastAPI:
    app = FastAPI()
    if app_state.config.identity_provider is not None:
        app.add_middleware(
            AuthorizationMiddleware,
            config=app_state.config,
            auth_db=app_state.auth_db,
            session_lookup=app_state.store.get_session_workspace,
        )
    app.include_router(build_ui_router(app_state))
    return app


def _bearer(raw_key):
    return {"Authorization": f"Bearer {raw_key}"}


@pytest.fixture
def mt_app_state(db_path):
    app_state = _FakeAppState(db_path)
    provider = APIKeyIdentityProvider(app_state.auth_db)
    app_state.config.identity_provider = provider
    return app_state


class TestSessionRouteAuthorization:
    def test_member_can_delete_own_session(self, mt_app_state):
        provider = mt_app_state.config.identity_provider
        _, raw_key = provider.create_key(subject="bob@acme.com", created_by="root")
        mt_app_state.auth_db.grant_access("finance", "bob@acme.com", "member", granted_by="root")
        session = mt_app_state.store.create_session(workspace="finance", created_by="bob@acme.com")

        app = _build_app(mt_app_state)
        client = TestClient(app)
        resp = client.delete(f"/api/ui/sessions/{session['session_id']}", headers=_bearer(raw_key))
        assert resp.status_code == 200
        assert mt_app_state.store.get_session(session["session_id"]) is None

        rows = mt_app_state.auth_db.get_activity_log(subject="bob@acme.com")
        assert len(rows) == 1
        assert rows[0]["action"] == "delete_conversation"
        assert rows[0]["workspace"] == "finance"

    def test_member_cannot_delete_someone_elses_session(self, mt_app_state):
        """The actual boundary: a member is no longer blocked from
        deleting entirely (that was the gap this was fixed for), but they
        still can't delete a session someone ELSE created."""
        provider = mt_app_state.config.identity_provider
        _, alice_key = provider.create_key(subject="alice@acme.com", created_by="root")
        mt_app_state.auth_db.grant_access("finance", "alice@acme.com", "member", granted_by="root")
        session = mt_app_state.store.create_session(workspace="finance", created_by="bob@acme.com")

        app = _build_app(mt_app_state)
        client = TestClient(app)
        resp = client.delete(
            f"/api/ui/sessions/{session['session_id']}", headers=_bearer(alice_key)
        )
        assert resp.status_code == 403
        assert mt_app_state.store.get_session(session["session_id"]) is not None

    def test_admin_can_delete_any_session_in_the_workspace(self, mt_app_state):
        provider = mt_app_state.config.identity_provider
        _, admin_key = provider.create_key(subject="admin@acme.com", created_by="root")
        mt_app_state.auth_db.grant_access("finance", "admin@acme.com", "admin", granted_by="root")
        session = mt_app_state.store.create_session(workspace="finance", created_by="bob@acme.com")

        app = _build_app(mt_app_state)
        client = TestClient(app)
        resp = client.delete(
            f"/api/ui/sessions/{session['session_id']}", headers=_bearer(admin_key)
        )
        assert resp.status_code == 200
        assert mt_app_state.store.get_session(session["session_id"]) is None

    def test_member_can_delete_legacy_session_with_no_recorded_owner(self, mt_app_state):
        """A pre-migration row with created_by=NULL has unknown ownership --
        fail open (same policy the GET /sessions ownership filter already
        uses) rather than permanently blocking deletion of old data nobody
        can prove they own."""
        provider = mt_app_state.config.identity_provider
        _, raw_key = provider.create_key(subject="alice@acme.com", created_by="root")
        mt_app_state.auth_db.grant_access("finance", "alice@acme.com", "member", granted_by="root")
        session = mt_app_state.store.create_session(workspace="finance")  # created_by=None

        app = _build_app(mt_app_state)
        client = TestClient(app)
        resp = client.delete(f"/api/ui/sessions/{session['session_id']}", headers=_bearer(raw_key))
        assert resp.status_code == 200

    def test_member_can_view_and_rename_session_in_their_workspace(self, mt_app_state):
        provider = mt_app_state.config.identity_provider
        _, raw_key = provider.create_key(subject="bob@acme.com", created_by="root")
        mt_app_state.auth_db.grant_access("finance", "bob@acme.com", "member", granted_by="root")
        session = mt_app_state.store.create_session(workspace="finance")

        app = _build_app(mt_app_state)
        client = TestClient(app)
        resp = client.get(f"/api/ui/sessions/{session['session_id']}", headers=_bearer(raw_key))
        assert resp.status_code == 200

        resp = client.patch(
            f"/api/ui/sessions/{session['session_id']}",
            json={"title": "Renamed"},
            headers=_bearer(raw_key),
        )
        assert resp.status_code == 200

    def test_no_grant_in_workspace_denies_session_access(self, mt_app_state):
        provider = mt_app_state.config.identity_provider
        _, raw_key = provider.create_key(subject="mallory@acme.com", created_by="root")
        session = mt_app_state.store.create_session(workspace="finance")

        app = _build_app(mt_app_state)
        client = TestClient(app)
        resp = client.get(f"/api/ui/sessions/{session['session_id']}", headers=_bearer(raw_key))
        assert resp.status_code == 403

    def test_nonexistent_session_id_denies_rather_than_404(self, mt_app_state):
        # session_lookup returns None for an unknown id -> workspace None ->
        # 403, indistinguishable from "exists but you lack access" per the
        # design doc's no-information-leakage principle.
        provider = mt_app_state.config.identity_provider
        _, raw_key = provider.create_key(subject="alice@acme.com", created_by="root")
        mt_app_state.auth_db.grant_access("finance", "alice@acme.com", "admin", granted_by="root")

        app = _build_app(mt_app_state)
        client = TestClient(app)
        resp = client.delete("/api/ui/sessions/does-not-exist", headers=_bearer(raw_key))
        assert resp.status_code == 403

    def test_single_user_mode_delete_session_unaffected(self, db_path):
        app_state = _FakeAppState(db_path)  # no identity_provider
        session = app_state.store.create_session(workspace="default")

        app = _build_app(app_state)
        client = TestClient(app)
        resp = client.delete(f"/api/ui/sessions/{session['session_id']}")
        assert resp.status_code == 200
        assert app_state.store.get_session(session["session_id"]) is None


class TestSessionOwnershipFiltering:
    def test_member_sees_only_own_sessions_by_default(self, mt_app_state):
        provider = mt_app_state.config.identity_provider
        _, alice_key = provider.create_key(subject="alice@acme.com", created_by="root")
        _, bob_key = provider.create_key(subject="bob@acme.com", created_by="root")
        mt_app_state.auth_db.grant_access("finance", "alice@acme.com", "member", granted_by="root")
        mt_app_state.auth_db.grant_access("finance", "bob@acme.com", "member", granted_by="root")

        app = _build_app(mt_app_state)
        client = TestClient(app)
        client.post("/api/ui/sessions", json={"workspace": "finance"}, headers=_bearer(alice_key))
        client.post("/api/ui/sessions", json={"workspace": "finance"}, headers=_bearer(bob_key))

        resp = client.get(
            "/api/ui/sessions", params={"workspace": "finance"}, headers=_bearer(alice_key)
        )
        assert resp.status_code == 200
        sessions = resp.json()["sessions"]
        assert len(sessions) == 1
        assert sessions[0]["created_by"] == "alice@acme.com"

    def test_admin_sees_all_sessions_with_all_true(self, mt_app_state):
        provider = mt_app_state.config.identity_provider
        _, alice_key = provider.create_key(subject="alice@acme.com", created_by="root")
        _, admin_key = provider.create_key(subject="admin@acme.com", created_by="root")
        mt_app_state.auth_db.grant_access("finance", "alice@acme.com", "member", granted_by="root")
        mt_app_state.auth_db.grant_access("finance", "admin@acme.com", "admin", granted_by="root")

        app = _build_app(mt_app_state)
        client = TestClient(app)
        client.post("/api/ui/sessions", json={"workspace": "finance"}, headers=_bearer(alice_key))
        client.post("/api/ui/sessions", json={"workspace": "finance"}, headers=_bearer(admin_key))

        resp = client.get(
            "/api/ui/sessions",
            params={"workspace": "finance", "all": "true"},
            headers=_bearer(admin_key),
        )
        assert len(resp.json()["sessions"]) == 2

        # Without ?all=true, admin still only sees their own.
        resp = client.get(
            "/api/ui/sessions", params={"workspace": "finance"}, headers=_bearer(admin_key)
        )
        assert len(resp.json()["sessions"]) == 1

    def test_single_user_mode_sees_all_sessions_unfiltered(self, db_path):
        app_state = _FakeAppState(db_path)
        app_state.store.create_session(workspace="default")
        app_state.store.create_session(workspace="default")

        app = _build_app(app_state)
        client = TestClient(app)
        resp = client.get("/api/ui/sessions", params={"workspace": "default"})
        assert len(resp.json()["sessions"]) == 2

    def test_legacy_null_created_by_visible_to_everyone(self, mt_app_state):
        # Simulate a pre-migration row with created_by left NULL.
        mt_app_state.store.create_session(workspace="finance")  # created_by=None

        provider = mt_app_state.config.identity_provider
        _, raw_key = provider.create_key(subject="alice@acme.com", created_by="root")
        mt_app_state.auth_db.grant_access("finance", "alice@acme.com", "member", granted_by="root")

        app = _build_app(mt_app_state)
        client = TestClient(app)
        resp = client.get(
            "/api/ui/sessions", params={"workspace": "finance"}, headers=_bearer(raw_key)
        )
        assert len(resp.json()["sessions"]) == 1
