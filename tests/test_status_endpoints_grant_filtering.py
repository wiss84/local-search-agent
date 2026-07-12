"""
Regression tests for the grant-filtering fix on GET /api/ui/ingest/status,
GET /api/ui/scheduler/status, and GET /api/ui/watch/status.

Confirmed gap (see upcoming_features/08-security-hardening-and-remaining-work.md,
section 1): these three routes return a summary across *every* registered
workspace in one call with no check at all -- the same "listing across
many workspaces" shape problem as GET /api/ui/workspaces (which is
filtered by grant via bespoke handler logic rather than a RoutePolicy
entry, since RoutePolicy only expresses a single-workspace check).

These tests exercise api_routes.py's build_ui_router() directly against a
minimal fake app_state, bypassing AuthorizationMiddleware entirely (these
routes are deliberately not in ROUTE_POLICIES, so the middleware never
touches them -- identity has to be resolved inside the handler itself via
_filter_workspaces_by_grant()).
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from local_search_agent.auth.api_key_provider import APIKeyIdentityProvider
from local_search_agent.ui import api_routes
from local_search_agent.ui.api_routes import build_ui_router
from local_search_agent.workspace.auth_db import AuthDB

# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------


@pytest.fixture
def auth_db(tmp_path):
    return AuthDB(db_path=str(tmp_path / "test.db"))


@pytest.fixture
def provider(auth_db):
    return APIKeyIdentityProvider(auth_db)


class _FakeConfig:
    def __init__(self, identity_provider=None):
        self.identity_provider = identity_provider


class _FakeScheduler:
    def get_status(self):
        return {
            "running": True,
            "registered_workspaces": ["finance", "marketing"],
            "scheduled_jobs": [
                {"job_id": "incremental_sync_finance", "name": "f", "next_run_at": None},
                {"job_id": "incremental_sync_marketing", "name": "m", "next_run_at": None},
            ],
        }


class _FakeWatcher:
    def get_status(self):
        return {
            "running": True,
            "registered_workspaces": ["finance", "marketing"],
            "watched_directories": {"finance": 1, "marketing": 2},
            "debounce_seconds": 2.5,
        }


class _FakeAppState:
    def __init__(self, auth_db, identity_provider=None):
        self.config = _FakeConfig(identity_provider)
        self.auth_db = auth_db
        self.scheduler = _FakeScheduler()
        self.watcher = _FakeWatcher()
        self.store = None
        self.workspace_manager = None
        self.framework = None


def _build_app(app_state) -> FastAPI:
    app = FastAPI()
    app.include_router(build_ui_router(app_state))
    return app


def _bearer(raw_key: str) -> dict:
    return {"Authorization": f"Bearer {raw_key}"}


@pytest.fixture(autouse=True)
def _reset_ingest_registry():
    """The ingest progress registry is module-level global state -- reset
    it before and after every test so tests don't leak into each other."""
    api_routes._ingest_registry.clear()
    yield
    api_routes._ingest_registry.clear()


def _seed_ingest_registry():
    api_routes._ingest_registry["finance"] = api_routes._IngestProgress(
        workspace="finance", status="running"
    )
    api_routes._ingest_registry["marketing"] = api_routes._IngestProgress(
        workspace="marketing", status="done"
    )


# ---------------------------------------------------------------------------
# Single-user mode: no filtering at all, no auth required
# ---------------------------------------------------------------------------


class TestSingleUserModeUnfiltered:
    def test_ingest_status_returns_everything(self, auth_db):
        _seed_ingest_registry()
        app_state = _FakeAppState(auth_db, identity_provider=None)
        client = TestClient(_build_app(app_state))
        resp = client.get("/api/ui/ingest/status")
        assert resp.status_code == 200
        names = {w["workspace"] for w in resp.json()["workspaces"]}
        assert names == {"finance", "marketing"}

    def test_scheduler_status_returns_everything(self, auth_db):
        app_state = _FakeAppState(auth_db, identity_provider=None)
        client = TestClient(_build_app(app_state))
        resp = client.get("/api/ui/scheduler/status")
        assert resp.status_code == 200
        assert set(resp.json()["registered_workspaces"]) == {"finance", "marketing"}

    def test_watch_status_returns_everything(self, auth_db):
        app_state = _FakeAppState(auth_db, identity_provider=None)
        client = TestClient(_build_app(app_state))
        resp = client.get("/api/ui/watch/status")
        assert resp.status_code == 200
        assert set(resp.json()["registered_workspaces"]) == {"finance", "marketing"}
        assert set(resp.json()["watched_directories"].keys()) == {"finance", "marketing"}


# ---------------------------------------------------------------------------
# Multi-tenant mode: unauthenticated denied
# ---------------------------------------------------------------------------


class TestMultiTenantNoCredential:
    @pytest.mark.parametrize(
        "path", ["/api/ui/ingest/status", "/api/ui/scheduler/status", "/api/ui/watch/status"]
    )
    def test_no_credential_returns_401(self, auth_db, provider, path):
        app_state = _FakeAppState(auth_db, identity_provider=provider)
        client = TestClient(_build_app(app_state))
        resp = client.get(path)
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Multi-tenant mode: non-superadmin filtered to their grants only
# ---------------------------------------------------------------------------


class TestMultiTenantFiltering:
    def test_ingest_status_filtered_to_granted_workspace(self, auth_db, provider):
        _seed_ingest_registry()
        auth_db.grant_access("finance", "alice@acme.com", "member", granted_by="root")
        _, raw = provider.create_key(subject="alice@acme.com", created_by="root")

        app_state = _FakeAppState(auth_db, identity_provider=provider)
        client = TestClient(_build_app(app_state))
        resp = client.get("/api/ui/ingest/status", headers=_bearer(raw))
        assert resp.status_code == 200
        names = {w["workspace"] for w in resp.json()["workspaces"]}
        assert names == {"finance"}

    def test_scheduler_status_filtered_to_granted_workspace(self, auth_db, provider):
        auth_db.grant_access("finance", "alice@acme.com", "member", granted_by="root")
        _, raw = provider.create_key(subject="alice@acme.com", created_by="root")

        app_state = _FakeAppState(auth_db, identity_provider=provider)
        client = TestClient(_build_app(app_state))
        resp = client.get("/api/ui/scheduler/status", headers=_bearer(raw))
        assert resp.status_code == 200
        body = resp.json()
        assert body["registered_workspaces"] == ["finance"]
        job_ids = {j["job_id"] for j in body["scheduled_jobs"]}
        assert job_ids == {"incremental_sync_finance"}

    def test_watch_status_filtered_to_granted_workspace(self, auth_db, provider):
        auth_db.grant_access("finance", "alice@acme.com", "member", granted_by="root")
        _, raw = provider.create_key(subject="alice@acme.com", created_by="root")

        app_state = _FakeAppState(auth_db, identity_provider=provider)
        client = TestClient(_build_app(app_state))
        resp = client.get("/api/ui/watch/status", headers=_bearer(raw))
        assert resp.status_code == 200
        body = resp.json()
        assert body["registered_workspaces"] == ["finance"]
        assert set(body["watched_directories"].keys()) == {"finance"}

    def test_member_with_no_grants_sees_nothing(self, auth_db, provider):
        _seed_ingest_registry()
        _, raw = provider.create_key(subject="bob@acme.com", created_by="root")

        app_state = _FakeAppState(auth_db, identity_provider=provider)
        client = TestClient(_build_app(app_state))
        resp = client.get("/api/ui/ingest/status", headers=_bearer(raw))
        assert resp.status_code == 200
        assert resp.json()["workspaces"] == []


# ---------------------------------------------------------------------------
# Multi-tenant mode: superadmin bypasses filtering, sees everything
# ---------------------------------------------------------------------------


class TestMultiTenantSuperadminBypass:
    def test_superadmin_sees_all_workspaces(self, auth_db, provider):
        _seed_ingest_registry()
        _, raw = provider.create_key(subject="boss@acme.com", created_by="root", is_superadmin=True)

        app_state = _FakeAppState(auth_db, identity_provider=provider)
        client = TestClient(_build_app(app_state))
        resp = client.get("/api/ui/ingest/status", headers=_bearer(raw))
        assert resp.status_code == 200
        names = {w["workspace"] for w in resp.json()["workspaces"]}
        assert names == {"finance", "marketing"}
