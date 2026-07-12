"""
Tests for multi-tenant RBAC: activity logging.

AuthDB.log_activity()/get_activity_log()
this file covers the
*call sites* added in this phase: api_routes.py's _log_activity() helper
plus the query/ingest/ingest-wipe/workspace-create/workspace-delete routes
that call it, grants_routes.py's grant/revoke, and login/logout via
APIKeyIdentityProvider.

Deliberately NOT covered here (documented gap, not silent): DELETE
/api/ui/sessions/{session_id} ("delete_conversation") isn't in
route_policy.py's ROUTE_POLICIES yet -- its workspace lives on the session
row, not the request, and resolving that requires the per-route ownership
lookup workspace_resolution.py's docstring calls out as a follow-up to
this same phase. request.state.identity is never set for that route today
(AuthorizationMiddleware passes unmatched routes straight through), so
_log_activity() would silently no-op there even if wired in -- not worth
faking.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from local_search_agent.auth.api_key_provider import APIKeyIdentityProvider
from local_search_agent.auth.authorization_middleware import AuthorizationMiddleware
from local_search_agent.auth.grants_routes import build_grants_router
from local_search_agent.ui.api_routes import build_ui_router
from local_search_agent.ui.store import UIStore
from local_search_agent.workspace.auth_db import AuthDB
from local_search_agent.workspace.workspace_manager import WorkspaceManager

# ---------------------------------------------------------------------------
# Fakes / fixtures
# ---------------------------------------------------------------------------


class _FakeFramework:
    """Stand-in for SearchAgentFramework -- just enough for delete_workspace()."""

    def __init__(self):
        self.deleted: list[tuple[str, bool]] = []

    def delete_workspace(self, name: str, wipe_index: bool = False) -> None:
        self.deleted.append((name, wipe_index))


class _FakeConfig:
    def __init__(self, db_path: str, identity_provider=None):
        self.db_path = db_path
        self.identity_provider = identity_provider
        self.meilisearch_url = "http://localhost:7700"
        self.meili_master_key = "local_search_master_key"
        # Needed since Model/Provider Access Control (Option B) added a
        # provider/model allow-list check to POST /api/ui/query that reads
        # app_state.config.provider/model_name as the fallback "effective
        # model" when a request doesn't specify its own override.
        self.provider = "google"
        self.model_name = "gemma-4-31b-it"


class _FakeAppState:
    """
    Minimal app_state double -- real AuthDB/WorkspaceManager/UIStore against
    a tmp SQLite file (all three are thin, cheap SQLite wrappers, same
    pattern as other test files in this suite), plus a fake framework/agent
    so nothing here needs a live Meilisearch or LLM provider.
    """

    def __init__(self, db_path: str, identity_provider=None):
        self.config = _FakeConfig(db_path, identity_provider=identity_provider)
        self.auth_db = AuthDB(db_path=db_path)
        self.workspace_manager = WorkspaceManager(db_path=db_path)
        self.store = UIStore(db_path=db_path)
        self.framework = _FakeFramework()
        self.scheduler = None
        self.watcher = None

    def get_agent(self, workspace=None, meili_api_key=None, provider=None, model_name=None):
        raise RuntimeError("no agent configured in tests -- query route should still log first")


def _build_app(app_state: _FakeAppState) -> FastAPI:
    app = FastAPI()
    if app_state.config.identity_provider is not None:
        app.add_middleware(
            AuthorizationMiddleware,
            config=app_state.config,
            auth_db=app_state.auth_db,
        )
    app.include_router(build_ui_router(app_state))
    return app


def _bearer(raw_key: str) -> dict:
    return {"Authorization": f"Bearer {raw_key}"}


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test.db")


@pytest.fixture
def app_state(db_path):
    return _FakeAppState(db_path)


@pytest.fixture
def mt_app_state(app_state):
    """Same app_state, but with an APIKeyIdentityProvider wired into config
    (multi-tenant mode) -- sharing app_state.auth_db, same as production's
    AppState.auth_db / build_dashboard_app() wiring."""
    provider = APIKeyIdentityProvider(app_state.auth_db)
    app_state.config.identity_provider = provider
    return app_state


# ---------------------------------------------------------------------------
# Single-user mode: no identity_provider configured -> silent no-op
# ---------------------------------------------------------------------------


class TestSingleUserModeNoOps:
    def test_query_does_not_write_activity_log(self, app_state):
        app = _build_app(app_state)
        client = TestClient(app)
        session = app_state.store.create_session(workspace="default")

        resp = client.post(
            "/api/ui/query",
            json={
                "session_id": session["session_id"],
                "question": "hello",
                "workspace": "default",
            },
        )
        assert resp.status_code == 200
        assert app_state.auth_db.get_activity_log() == []

    def test_workspace_create_does_not_write_activity_log(self, app_state, tmp_path):
        app = _build_app(app_state)
        client = TestClient(app)
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()

        resp = client.post(
            "/api/ui/workspaces", json={"name": "solo", "document_dirs": [str(docs_dir)]}
        )
        assert resp.status_code == 200
        assert app_state.auth_db.get_activity_log() == []


# ---------------------------------------------------------------------------
# Multi-tenant mode: query / ingest / ingest-wipe / workspace create+delete
# ---------------------------------------------------------------------------


class TestQueryActivityLogging:
    def test_search_logged_with_subject_workspace_and_question(self, mt_app_state):
        provider = mt_app_state.config.identity_provider
        _, raw_key = provider.create_key(subject="alice@acme.com", created_by="root")
        mt_app_state.auth_db.grant_access("finance", "alice@acme.com", "member", granted_by="root")
        # Model/Provider Access Control (Option B): the query route now
        # checks the effective provider/model against the caller's role's
        # allow-list before it ever reaches the (fake, error-raising) agent
        # -- grant the member role access to the fake config's default
        # provider/model so this test still exercises activity logging
        # rather than tripping the (correctly working) 403 gate.
        mt_app_state.auth_db.grant_model_access(
            "member", "google", "gemma-4-31b-it", granted_by="root"
        )
        session = mt_app_state.store.create_session(workspace="finance")

        app = _build_app(mt_app_state)
        client = TestClient(app)
        resp = client.post(
            "/api/ui/query",
            json={
                "session_id": session["session_id"],
                "question": "what were Q3 revenues?",
                "workspace": "finance",
            },
            headers=_bearer(raw_key),
        )
        assert resp.status_code == 200

        rows = mt_app_state.auth_db.get_activity_log(subject="alice@acme.com")
        assert len(rows) == 1
        assert rows[0]["action"] == "search"
        assert rows[0]["workspace"] == "finance"
        assert rows[0]["detail"] == "what were Q3 revenues?"
        assert rows[0]["success"] == 1

    def test_member_without_grant_denied_and_not_logged(self, mt_app_state):
        provider = mt_app_state.config.identity_provider
        _, raw_key = provider.create_key(subject="mallory@acme.com", created_by="root")
        session = mt_app_state.store.create_session(workspace="finance")

        app = _build_app(mt_app_state)
        client = TestClient(app)
        resp = client.post(
            "/api/ui/query",
            json={
                "session_id": session["session_id"],
                "question": "leak this",
                "workspace": "finance",
            },
            headers=_bearer(raw_key),
        )
        assert resp.status_code == 403
        assert mt_app_state.auth_db.get_activity_log() == []


class TestIngestActivityLogging:
    def test_ingest_logged_as_admin_action_with_force_detail(self, mt_app_state):
        provider = mt_app_state.config.identity_provider
        # force=True is superadmin-only (tightened after discussing where
        # workspace-admin-vs-superadmin should sit for an open-source
        # framework any company/team/solo user might deploy -- see
        # test_workspace_and_ingest_tightening.py for the dedicated
        # coverage of that restriction itself). This test is about
        # activity logging, so it just needs a caller who's actually
        # allowed to reach the force=True branch.
        _, raw_key = provider.create_key(
            subject="admin@acme.com", created_by="root", is_superadmin=True
        )

        app = _build_app(mt_app_state)
        client = TestClient(app)
        resp = client.post(
            "/api/ui/ingest",
            json={"workspace": "finance", "force": True},
            headers=_bearer(raw_key),
        )
        assert resp.status_code == 200

        rows = mt_app_state.auth_db.get_activity_log(subject="admin@acme.com")
        assert len(rows) == 1
        assert rows[0]["action"] == "ingest"
        assert rows[0]["workspace"] == "finance"
        assert rows[0]["detail"] == "force=True"

    def test_ingest_wipe_logged_as_workspace_wipe(self, mt_app_state):
        provider = mt_app_state.config.identity_provider
        # Wipe & re-ingest is always superadmin-only (see
        # test_workspace_and_ingest_tightening.py) -- same reasoning as
        # above, this test is only about activity logging.
        _, raw_key = provider.create_key(
            subject="admin@acme.com", created_by="root", is_superadmin=True
        )

        app = _build_app(mt_app_state)
        client = TestClient(app)
        resp = client.post(
            "/api/ui/ingest/wipe",
            json={"workspace": "finance"},
            headers=_bearer(raw_key),
        )
        assert resp.status_code == 200

        rows = mt_app_state.auth_db.get_activity_log(subject="admin@acme.com")
        assert len(rows) == 1
        assert rows[0]["action"] == "workspace_wipe"
        assert rows[0]["workspace"] == "finance"

    def test_member_cannot_trigger_ingest(self, mt_app_state):
        provider = mt_app_state.config.identity_provider
        _, raw_key = provider.create_key(subject="bob@acme.com", created_by="root")
        mt_app_state.auth_db.grant_access("finance", "bob@acme.com", "member", granted_by="root")

        app = _build_app(mt_app_state)
        client = TestClient(app)
        resp = client.post(
            "/api/ui/ingest",
            json={"workspace": "finance"},
            headers=_bearer(raw_key),
        )
        assert resp.status_code == 403
        assert mt_app_state.auth_db.get_activity_log() == []


class TestWorkspaceCreateDeleteActivityLogging:
    def test_workspace_create_logged_global_admin(self, mt_app_state, tmp_path):
        provider = mt_app_state.config.identity_provider
        # Workspace create is superadmin-only (tightened -- requires a
        # document_dirs path that already exists on the server's own disk,
        # inherently a provisioning action; see
        # test_workspace_and_ingest_tightening.py for dedicated coverage of
        # the restriction itself). This test is only about activity
        # logging, so it just needs a caller who's actually allowed through.
        _, raw_key = provider.create_key(
            subject="root@acme.com", created_by="bootstrap", is_superadmin=True
        )
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()

        app = _build_app(mt_app_state)
        client = TestClient(app)
        resp = client.post(
            "/api/ui/workspaces",
            json={"name": "legal", "document_dirs": [str(docs_dir)]},
            headers=_bearer(raw_key),
        )
        assert resp.status_code == 200

        rows = mt_app_state.auth_db.get_activity_log(subject="root@acme.com")
        assert len(rows) == 1
        assert rows[0]["action"] == "workspace_create"
        assert rows[0]["workspace"] == "legal"

    def test_workspace_delete_logged_before_delegating_to_framework(self, mt_app_state):
        provider = mt_app_state.config.identity_provider
        # Workspace delete is superadmin-only too -- see above.
        _, raw_key = provider.create_key(
            subject="admin@acme.com", created_by="root", is_superadmin=True
        )
        mt_app_state.workspace_manager.create_workspace(name="finance", document_dir="/tmp")

        app = _build_app(mt_app_state)
        client = TestClient(app)
        resp = client.request(
            "DELETE",
            "/api/ui/workspaces/finance",
            params={"wipe": "true"},
            headers=_bearer(raw_key),
        )
        assert resp.status_code == 200
        assert mt_app_state.framework.deleted == [("finance", True)]

        rows = mt_app_state.auth_db.get_activity_log(subject="admin@acme.com")
        assert len(rows) == 1
        assert rows[0]["action"] == "workspace_delete"
        assert rows[0]["workspace"] == "finance"
        assert rows[0]["detail"] == "wipe=True"

    def test_workspace_create_requires_global_admin_not_member(self, mt_app_state, tmp_path):
        provider = mt_app_state.config.identity_provider
        _, raw_key = provider.create_key(subject="bob@acme.com", created_by="root")
        mt_app_state.auth_db.grant_access("finance", "bob@acme.com", "member", granted_by="root")
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()

        app = _build_app(mt_app_state)
        client = TestClient(app)
        resp = client.post(
            "/api/ui/workspaces",
            json={"name": "legal", "document_dirs": [str(docs_dir)]},
            headers=_bearer(raw_key),
        )
        assert resp.status_code == 403
        assert mt_app_state.auth_db.get_activity_log() == []

    def test_workspace_create_requires_superadmin_not_just_global_admin(
        self, mt_app_state, tmp_path
    ):
        """A plain workspace admin (global_admin-eligible, but not
        superadmin) must still be denied -- workspace create/delete was
        tightened past global_admin specifically."""
        provider = mt_app_state.config.identity_provider
        _, raw_key = provider.create_key(subject="carol@acme.com", created_by="root")
        mt_app_state.auth_db.grant_access("finance", "carol@acme.com", "admin", granted_by="root")
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()

        app = _build_app(mt_app_state)
        client = TestClient(app)
        resp = client.post(
            "/api/ui/workspaces",
            json={"name": "legal", "document_dirs": [str(docs_dir)]},
            headers=_bearer(raw_key),
        )
        assert resp.status_code == 403
        assert mt_app_state.auth_db.get_activity_log() == []


# ---------------------------------------------------------------------------
# grants_routes.py: grant_access / revoke_access
# ---------------------------------------------------------------------------


class _FakeIdentity:
    def __init__(self, subject, is_superadmin=True):
        self.subject = subject
        # Defaults to True -- these tests exercise grant/revoke logging
        # itself, not the separate superadmin-vs-admin restriction on
        # granting/revoking the admin role (see test_admin_panel_routes.py
        # for those); every grant/revoke here uses role="member", which
        # any global admin (superadmin or not) can perform.
        self.is_superadmin = is_superadmin


class TestGrantsActivityLogging:
    @pytest.fixture
    def auth_db(self, db_path):
        return AuthDB(db_path=db_path)

    def _app_with_injected_identity(self, auth_db, subject):
        app = FastAPI()
        app.include_router(build_grants_router(auth_db))

        @app.middleware("http")
        async def inject_identity(request: Request, call_next):
            request.state.identity = _FakeIdentity(subject)
            return await call_next(request)

        return app

    def test_grant_logs_one_row_per_workspace(self, auth_db):
        app = self._app_with_injected_identity(auth_db, "root@acme.com")
        client = TestClient(app)
        resp = client.post(
            "/api/admin/grants",
            json={
                "subject": "alice@acme.com",
                "workspaces": ["finance", "marketing"],
                "role": "member",
            },
        )
        assert resp.status_code == 200

        rows = auth_db.get_activity_log(subject="root@acme.com")
        assert len(rows) == 2
        assert {r["action"] for r in rows} == {"grant_access"}
        assert {r["workspace"] for r in rows} == {"finance", "marketing"}
        assert all("alice@acme.com" in r["detail"] for r in rows)

    def test_invalid_role_does_not_log(self, auth_db):
        app = self._app_with_injected_identity(auth_db, "root@acme.com")
        client = TestClient(app)
        resp = client.post(
            "/api/admin/grants",
            json={"subject": "alice@acme.com", "workspaces": ["finance"], "role": "owner"},
        )
        assert resp.status_code == 400
        assert auth_db.get_activity_log() == []

    def test_revoke_logs_revoke_access(self, auth_db):
        auth_db.grant_access("finance", "alice@acme.com", "member", granted_by="root")
        app = self._app_with_injected_identity(auth_db, "root@acme.com")
        client = TestClient(app)
        resp = client.request(
            "DELETE",
            "/api/admin/grants",
            json={"subject": "alice@acme.com", "workspaces": ["finance"]},
        )
        assert resp.status_code == 200

        rows = auth_db.get_activity_log(subject="root@acme.com")
        assert len(rows) == 1
        assert rows[0]["action"] == "revoke_access"
        assert "alice@acme.com" in rows[0]["detail"]


# ---------------------------------------------------------------------------
# APIKeyIdentityProvider: login / logout
# ---------------------------------------------------------------------------


class TestLoginLogoutActivityLogging:
    @pytest.fixture
    def auth_db(self, db_path):
        return AuthDB(db_path=db_path)

    @pytest.fixture
    def provider(self, auth_db):
        return APIKeyIdentityProvider(auth_db)

    def test_login_logs_login_action(self, provider, auth_db):
        _, raw_key = provider.create_key(subject="alice@acme.com", created_by="root")
        result = provider.login(raw_key, ip_address="10.0.0.5")
        assert result is not None

        rows = auth_db.get_activity_log(subject="alice@acme.com")
        assert len(rows) == 1
        assert rows[0]["action"] == "login"
        assert rows[0]["ip_address"] == "10.0.0.5"

    def test_failed_login_does_not_log(self, provider, auth_db):
        assert provider.login("lsa_bad_key") is None
        assert auth_db.get_activity_log() == []

    def test_logout_logs_logout_action_for_valid_session(self, provider, auth_db):
        _, raw_key = provider.create_key(subject="alice@acme.com", created_by="root")
        token, _ = provider.login(raw_key)
        provider.logout(token)

        rows = auth_db.get_activity_log(subject="alice@acme.com")
        actions = [r["action"] for r in rows]
        assert actions == ["logout", "login"]  # newest first

    def test_logout_unknown_token_does_not_log(self, provider, auth_db):
        provider.logout("nonexistent-token")  # must not raise
        assert auth_db.get_activity_log() == []
