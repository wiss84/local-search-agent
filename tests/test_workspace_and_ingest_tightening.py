"""
Regression tests for tightening workspace create/delete and destructive
ingest actions (force re-ingest, wipe & re-ingest) to superadmin_only.

Context: this framework is open source (PyPI + GitHub), used by anyone
from a solo user to a small team to a company deployment. Workspace
create/delete requires a document_dirs path that already exists on the
SERVER's own disk -- inherently a provisioning action only whoever
actually deployed/set up the server can act on meaningfully, not
something every workspace-level admin should be able to trigger. Force
re-ingest and wipe & re-ingest are similarly reserved: an ordinary
workspace admin can still trigger normal incremental ingest freely, but
the heavier, more destructive variants are superadmin-only.

Covers:
1. RoutePolicy entries for POST/DELETE /api/ui/workspaces.
2. The body-conditional check inside POST /api/ui/ingest (force=True) and
   the always-on check inside POST /api/ui/ingest/wipe -- neither of these
   is expressible as a plain RoutePolicy entry since both routes are also
   reachable with a non-restricted body shape (force=False for /ingest;
   /ingest/wipe has no non-destructive variant at all, but is still a
   request-level check inside the handler, not scope-based routing).
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from local_search_agent.auth.route_policy import match_policy
from local_search_agent.ui.api_routes import build_ui_router

# ---------------------------------------------------------------------------
# Layer 1: RoutePolicy entries for workspace create/delete
# ---------------------------------------------------------------------------


class TestWorkspaceCreateDeleteIsSuperadminOnly:
    def test_create_workspace_is_superadmin_only(self):
        policy = match_policy("POST", "/api/ui/workspaces")
        assert policy is not None
        assert policy.scope == "superadmin_only"

    def test_delete_workspace_is_superadmin_only(self):
        policy = match_policy("DELETE", "/api/ui/workspaces/finance")
        assert policy is not None
        assert policy.scope == "superadmin_only"


# ---------------------------------------------------------------------------
# Layer 2: POST /api/ui/ingest and /api/ui/ingest/wipe's handler-level checks
# ---------------------------------------------------------------------------


class _FakeConfig:
    identity_provider = None


class _FakeAppState:
    def __init__(self):
        self.config = _FakeConfig()
        self.auth_db = None
        self.workspace_manager = None


class _FakeIdentity:
    def __init__(self, subject, is_superadmin=False):
        self.subject = subject
        self.is_superadmin = is_superadmin


def _identity_stub_middleware(identity):
    async def middleware(request: Request, call_next):
        request.state.identity = identity
        request.state.role = "admin"
        return await call_next(request)

    return middleware


def _build_app(identity=None):
    app_state = _FakeAppState()
    app = FastAPI()
    if identity is not None:
        app.middleware("http")(_identity_stub_middleware(identity))
    app.include_router(build_ui_router(app_state))
    return app


class TestForceIngestRestrictedToSuperadmin:
    def test_single_user_mode_bypasses_check(self):
        """No identity at all (single-user desktop mode) -- force ingest
        still works exactly as before; this feature is multi-tenant-only."""
        app = _build_app(identity=None)
        client = TestClient(app)
        resp = client.post("/api/ui/ingest", json={"workspace": "finance", "force": True})
        assert resp.status_code == 200

    def test_plain_admin_denied_force_ingest(self):
        identity = _FakeIdentity("alice@acme.com", is_superadmin=False)
        app = _build_app(identity=identity)
        client = TestClient(app)
        resp = client.post("/api/ui/ingest", json={"workspace": "finance", "force": True})
        assert resp.status_code == 403

    def test_plain_admin_allowed_ordinary_ingest(self):
        """force=False (the default, ordinary incremental sync) is NOT
        restricted -- only the force=True variant is."""
        identity = _FakeIdentity("alice@acme.com", is_superadmin=False)
        app = _build_app(identity=identity)
        client = TestClient(app)
        resp = client.post("/api/ui/ingest", json={"workspace": "finance", "force": False})
        assert resp.status_code == 200

    def test_superadmin_allowed_force_ingest(self):
        identity = _FakeIdentity("boss@acme.com", is_superadmin=True)
        app = _build_app(identity=identity)
        client = TestClient(app)
        resp = client.post("/api/ui/ingest", json={"workspace": "finance", "force": True})
        assert resp.status_code == 200


class TestWipeAndReingestRestrictedToSuperadmin:
    def test_single_user_mode_bypasses_check(self):
        app = _build_app(identity=None)
        client = TestClient(app)
        resp = client.post("/api/ui/ingest/wipe", json={"workspace": "finance"})
        assert resp.status_code == 200

    def test_plain_admin_denied(self):
        identity = _FakeIdentity("alice@acme.com", is_superadmin=False)
        app = _build_app(identity=identity)
        client = TestClient(app)
        resp = client.post("/api/ui/ingest/wipe", json={"workspace": "finance"})
        assert resp.status_code == 403

    def test_superadmin_allowed(self):
        identity = _FakeIdentity("boss@acme.com", is_superadmin=True)
        app = _build_app(identity=identity)
        client = TestClient(app)
        resp = client.post("/api/ui/ingest/wipe", json={"workspace": "finance"})
        assert resp.status_code == 200
