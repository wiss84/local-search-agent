"""
Unit + integration tests for multi-tenant RBAC: AuthorizationMiddleware.

Uses a minimal FastAPI test app with routes mirroring route_policy.py's
ROUTE_POLICIES (same TestClient pattern as test_server.py), plus real
AuthDB + APIKeyIdentityProvider instances (no mocking of the auth stack
itself — only the identity provider's transport is a test double via a
Bearer header, exactly like production).

Covers:
- resolve_workspace(): path / query / body precedence
- match_policy(): protected vs unprotected routes
- AuthorizationMiddleware: every fail-closed branch (no provider on
  config → no-op; no credential → 401; bad credential → 401; no grant →
  403; insufficient role → 403; global_admin scope; superadmin bypass)
- request.state.identity/workspace/role populated for downstream handlers
- POST body still readable downstream after the middleware reads it for
  workspace resolution (body-replay regression check)
- End-to-end: bearer-key auth → granted workspace succeeds → non-granted
  workspace denied → revoke_workspace_access() → the SAME still-valid API
  key is immediately denied on the next request (proves per-request DB
  checks, not cached grants, per the design doc's core claim)
"""

from __future__ import annotations

import re

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from pydantic import BaseModel

from local_search_agent.auth.api_key_provider import APIKeyIdentityProvider
from local_search_agent.auth.authorization_middleware import AuthorizationMiddleware
from local_search_agent.auth.errors import ProviderUnavailableError
from local_search_agent.auth.route_policy import match_policy
from local_search_agent.auth.workspace_resolution import resolve_workspace
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


class _FakeConfig:
    """Minimal stand-in for SearchAgentConfig — middleware only reads .identity_provider."""

    def __init__(self, identity_provider):
        self.identity_provider = identity_provider


class _QueryBody(BaseModel):
    session_id: str
    question: str
    workspace: str


def _build_test_app(config, auth_db) -> FastAPI:
    """
    A small app with routes mirroring the real ROUTE_POLICIES shapes:
    - POST /api/ui/query           (member, workspace-from-body)
    - GET  /api/ui/sessions        (member, workspace-from-query)
    - GET  /workspaces/{ws}/docs   (member, workspace-from-path)
    - POST /api/ui/ingest          (admin, workspace-from-body)
    - POST /api/ui/workspaces      (admin, global_admin)
    - GET  /health                 (unprotected — not in ROUTE_POLICIES)
    """
    app = FastAPI()
    app.add_middleware(AuthorizationMiddleware, config=config, auth_db=auth_db)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/api/ui/query")
    async def query(body: _QueryBody, request: Request):
        # Downstream handler reads the Pydantic body AND raw request.state —
        # proves the body wasn't left exhausted by the middleware's read.
        return JSONResponse(
            {
                "workspace_seen_by_handler": body.workspace,
                "identity_subject": request.state.identity.subject,
                "resolved_workspace": request.state.workspace,
                "role": request.state.role,
            }
        )

    @app.get("/api/ui/sessions")
    async def sessions(workspace: str, request: Request):
        return JSONResponse({"workspace": workspace, "role": request.state.role})

    @app.get("/workspaces/{workspace_name}/docs")
    async def workspace_docs(workspace_name: str, request: Request):
        return JSONResponse({"workspace": workspace_name, "role": request.state.role})

    @app.post("/api/ui/ingest")
    async def ingest(request: Request):
        body = await request.json()
        return JSONResponse({"workspace": body["workspace"], "role": request.state.role})

    @app.post("/api/ui/workspaces")
    async def create_workspace(request: Request):
        return JSONResponse({"role": request.state.role})

    @app.post("/api/ui/keys")
    async def set_key(request: Request):
        # Still genuinely global_admin-scoped in the real ROUTE_POLICIES
        # (unlike /api/ui/workspaces, tightened to superadmin_only) -- used
        # by TestGlobalAdminScope so those tests keep exercising actual
        # global_admin semantics regardless of what /api/ui/workspaces's
        # own scope is.
        return JSONResponse({"role": request.state.role})

    return app


@pytest.fixture
def app_and_client(auth_db, provider):
    config = _FakeConfig(identity_provider=provider)
    app = _build_test_app(config, auth_db)
    return app, TestClient(app), config


def _bearer(raw_key: str) -> dict:
    return {"Authorization": f"Bearer {raw_key}"}


# ---------------------------------------------------------------------------
# resolve_workspace() unit tests
# ---------------------------------------------------------------------------


class _FakeRequest:
    """Minimal Starlette-Request-like stand-in for direct resolve_workspace() tests."""

    def __init__(self, path="/", query_params=None, method="GET", json_body=None):
        self.url = type("URL", (), {"path": path})()
        self.query_params = query_params or {}
        self.method = method
        self.state = type("State", (), {})()
        self._json_body = json_body

    async def body(self):
        import json

        return json.dumps(self._json_body).encode() if self._json_body is not None else b""


class TestResolveWorkspace:
    async def test_path_takes_precedence(self):
        pattern = re.compile(r"^/workspaces/(?P<workspace>[^/]+)/docs$")
        req = _FakeRequest(path="/workspaces/finance/docs", query_params={"workspace": "marketing"})
        result = await resolve_workspace(req, path_pattern=pattern)
        assert result == "finance"

    async def test_query_used_when_no_path_match(self):
        req = _FakeRequest(path="/api/ui/sessions", query_params={"workspace": "finance"})
        result = await resolve_workspace(req)
        assert result == "finance"

    async def test_body_used_as_fallback(self):
        req = _FakeRequest(path="/api/ui/query", method="POST", json_body={"workspace": "finance"})
        result = await resolve_workspace(req)
        assert result == "finance"

    async def test_no_match_returns_none(self):
        req = _FakeRequest(path="/api/ui/query", method="POST", json_body={})
        result = await resolve_workspace(req)
        assert result is None

    async def test_body_cached_on_request_state(self):
        req = _FakeRequest(path="/api/ui/query", method="POST", json_body={"workspace": "finance"})
        await resolve_workspace(req)
        assert req.state.json_body == {"workspace": "finance"}


# ---------------------------------------------------------------------------
# match_policy() unit tests
# ---------------------------------------------------------------------------


class TestMatchPolicy:
    def test_matches_known_route(self):
        policy = match_policy("POST", "/api/ui/query")
        assert policy is not None
        assert policy.required_role == "member"
        assert policy.scope == "workspace"

    def test_unprotected_route_returns_none(self):
        assert match_policy("GET", "/health") is None

    def test_wrong_method_returns_none(self):
        assert match_policy("DELETE", "/api/ui/query") is None

    def test_global_admin_scope_route(self):
        # /api/ui/workspaces was tightened to superadmin_only (see
        # TestSuperadminOnlyScope below) -- /api/ui/keys is still a real,
        # unchanged global_admin example.
        policy = match_policy("POST", "/api/ui/keys")
        assert policy.scope == "global_admin"
        assert policy.required_role == "admin"

    def test_superadmin_only_scope_route(self):
        policy = match_policy("POST", "/api/ui/workspaces")
        assert policy is not None
        assert policy.scope == "superadmin_only"
        assert policy.required_role == "admin"

    def test_workspace_path_group_route(self):
        policy = match_policy("GET", "/workspaces/finance/docs")
        assert policy is not None
        assert policy.workspace_path_group is True


# ---------------------------------------------------------------------------
# AuthorizationMiddleware: no-op when identity_provider is None
# ---------------------------------------------------------------------------


class TestMiddlewareNoOp:
    def test_no_identity_provider_passes_through_unprotected_route(self, auth_db):
        config = _FakeConfig(identity_provider=None)
        app = _build_test_app(config, auth_db)
        client = TestClient(app)
        # No Authorization header at all — should still succeed since the
        # middleware no-ops entirely when identity_provider is unset.
        assert client.get("/health").status_code == 200


# ---------------------------------------------------------------------------
# AuthorizationMiddleware: fail-closed branches
# ---------------------------------------------------------------------------


class TestFailClosed:
    def test_unprotected_route_always_passes(self, app_and_client):
        _, client, _ = app_and_client
        assert client.get("/health").status_code == 200

    def test_missing_credential_denied_401(self, app_and_client):
        _, client, _ = app_and_client
        resp = client.post(
            "/api/ui/query", json={"session_id": "s1", "question": "hi", "workspace": "finance"}
        )
        assert resp.status_code == 401

    def test_bad_credential_denied_401(self, app_and_client):
        _, client, _ = app_and_client
        resp = client.post(
            "/api/ui/query",
            json={"session_id": "s1", "question": "hi", "workspace": "finance"},
            headers=_bearer("lsa_bad_key_totally_wrong"),
        )
        assert resp.status_code == 401

    def test_valid_credential_no_grant_denied_403(self, app_and_client, provider):
        _, client, _ = app_and_client
        _, raw_key = provider.create_key(subject="alice@acme.com", created_by="admin")
        resp = client.post(
            "/api/ui/query",
            json={"session_id": "s1", "question": "hi", "workspace": "finance"},
            headers=_bearer(raw_key),
        )
        assert resp.status_code == 403

    def test_member_role_denied_from_admin_route(self, app_and_client, provider, auth_db):
        _, client, _ = app_and_client
        _, raw_key = provider.create_key(subject="alice@acme.com", created_by="admin")
        auth_db.grant_access("finance", "alice@acme.com", "member", granted_by="admin")
        resp = client.post(
            "/api/ui/ingest", json={"workspace": "finance"}, headers=_bearer(raw_key)
        )
        assert resp.status_code == 403

    def test_admin_role_allowed_on_admin_route(self, app_and_client, provider, auth_db):
        _, client, _ = app_and_client
        _, raw_key = provider.create_key(subject="alice@acme.com", created_by="admin")
        auth_db.grant_access("finance", "alice@acme.com", "admin", granted_by="admin")
        resp = client.post(
            "/api/ui/ingest", json={"workspace": "finance"}, headers=_bearer(raw_key)
        )
        assert resp.status_code == 200
        assert resp.json()["role"] == "admin"

    def test_member_role_allowed_on_member_route(self, app_and_client, provider, auth_db):
        _, client, _ = app_and_client
        _, raw_key = provider.create_key(subject="alice@acme.com", created_by="admin")
        auth_db.grant_access("finance", "alice@acme.com", "member", granted_by="admin")
        resp = client.post(
            "/api/ui/query",
            json={"session_id": "s1", "question": "hi", "workspace": "finance"},
            headers=_bearer(raw_key),
        )
        assert resp.status_code == 200

    def test_no_workspace_resolvable_denied_403(self, app_and_client, provider, auth_db):
        _, client, _ = app_and_client
        _, raw_key = provider.create_key(subject="alice@acme.com", created_by="admin")
        auth_db.grant_access("finance", "alice@acme.com", "member", granted_by="admin")
        # No workspace anywhere in the request (body key omitted via raw send is hard with
        # a Pydantic model requiring it; use ingest's plain-dict body instead)
        resp = client.post("/api/ui/ingest", json={}, headers=_bearer(raw_key))
        assert resp.status_code in (403, 422)  # 422 if FastAPI itself rejects malformed body

    def test_provider_unavailable_returns_503(self, app_and_client, provider):
        _, client, _ = app_and_client
        provider.resolve = lambda req: (_ for _ in []).throw(ProviderUnavailableError("IdP down"))
        resp = client.post(
            "/api/ui/query",
            json={"session_id": "s1", "question": "hi", "workspace": "finance"},
        )
        assert resp.status_code == 503

    def test_unexpected_exception_returns_503(self, app_and_client, provider):
        _, client, _ = app_and_client
        provider.resolve = lambda req: (_ for _ in []).throw(RuntimeError("boom"))
        resp = client.post(
            "/api/ui/query",
            json={"session_id": "s1", "question": "hi", "workspace": "finance"},
        )
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Workspace resolution source: path / query / body — via the real middleware
# ---------------------------------------------------------------------------


class TestWorkspaceResolutionSources:
    def test_resolves_from_path(self, app_and_client, provider, auth_db):
        _, client, _ = app_and_client
        _, raw_key = provider.create_key(subject="alice@acme.com", created_by="admin")
        auth_db.grant_access("finance", "alice@acme.com", "member", granted_by="admin")
        resp = client.get("/workspaces/finance/docs", headers=_bearer(raw_key))
        assert resp.status_code == 200
        assert resp.json()["workspace"] == "finance"

    def test_resolves_from_query(self, app_and_client, provider, auth_db):
        _, client, _ = app_and_client
        _, raw_key = provider.create_key(subject="alice@acme.com", created_by="admin")
        auth_db.grant_access("finance", "alice@acme.com", "member", granted_by="admin")
        resp = client.get(
            "/api/ui/sessions", params={"workspace": "finance"}, headers=_bearer(raw_key)
        )
        assert resp.status_code == 200

    def test_resolves_from_body(self, app_and_client, provider, auth_db):
        _, client, _ = app_and_client
        _, raw_key = provider.create_key(subject="alice@acme.com", created_by="admin")
        auth_db.grant_access("finance", "alice@acme.com", "member", granted_by="admin")
        resp = client.post(
            "/api/ui/query",
            json={"session_id": "s1", "question": "hi", "workspace": "finance"},
            headers=_bearer(raw_key),
        )
        assert resp.status_code == 200
        assert resp.json()["resolved_workspace"] == "finance"


# ---------------------------------------------------------------------------
# Body-replay regression: downstream handler must still read the POST body
# after the middleware consumed it for workspace resolution.
# ---------------------------------------------------------------------------


class TestBodyReplay:
    def test_downstream_handler_still_reads_pydantic_body(self, app_and_client, provider, auth_db):
        _, client, _ = app_and_client
        _, raw_key = provider.create_key(subject="alice@acme.com", created_by="admin")
        auth_db.grant_access("finance", "alice@acme.com", "member", granted_by="admin")
        resp = client.post(
            "/api/ui/query",
            json={"session_id": "s1", "question": "what is our Q3 spend?", "workspace": "finance"},
            headers=_bearer(raw_key),
        )
        assert resp.status_code == 200
        # The handler's Pydantic model successfully parsed a non-empty body —
        # if the stream had been left exhausted by the middleware, FastAPI
        # would 422 on a missing/empty body instead of reaching our handler.
        assert resp.json()["workspace_seen_by_handler"] == "finance"


# ---------------------------------------------------------------------------
# global_admin scope + superadmin bypass
# ---------------------------------------------------------------------------


class TestGlobalAdminScope:
    """Uses /api/ui/keys (still genuinely global_admin-scoped) rather than
    /api/ui/workspaces, which was tightened to superadmin_only -- see
    TestSuperadminOnlyScope below for that route's own coverage."""

    def test_admin_anywhere_allowed(self, app_and_client, provider, auth_db):
        _, client, _ = app_and_client
        _, raw_key = provider.create_key(subject="alice@acme.com", created_by="admin")
        auth_db.grant_access("marketing", "alice@acme.com", "admin", granted_by="admin")
        resp = client.post("/api/ui/keys", json={}, headers=_bearer(raw_key))
        assert resp.status_code == 200

    def test_member_only_denied(self, app_and_client, provider, auth_db):
        _, client, _ = app_and_client
        _, raw_key = provider.create_key(subject="alice@acme.com", created_by="admin")
        auth_db.grant_access("marketing", "alice@acme.com", "member", granted_by="admin")
        resp = client.post("/api/ui/keys", json={}, headers=_bearer(raw_key))
        assert resp.status_code == 403

    def test_no_grants_at_all_denied(self, app_and_client, provider):
        _, client, _ = app_and_client
        _, raw_key = provider.create_key(subject="alice@acme.com", created_by="admin")
        resp = client.post("/api/ui/keys", json={}, headers=_bearer(raw_key))
        assert resp.status_code == 403

    def test_superadmin_bypasses_grant_requirement(self, app_and_client, provider):
        _, client, _ = app_and_client
        _, raw_key = provider.create_key(
            subject="root@acme.com", created_by="admin", is_superadmin=True
        )
        resp = client.post("/api/ui/keys", json={}, headers=_bearer(raw_key))
        assert resp.status_code == 200

    def test_superadmin_bypasses_workspace_grant(self, app_and_client, provider):
        _, client, _ = app_and_client
        _, raw_key = provider.create_key(
            subject="root@acme.com", created_by="admin", is_superadmin=True
        )
        resp = client.post(
            "/api/ui/ingest", json={"workspace": "finance"}, headers=_bearer(raw_key)
        )
        assert resp.status_code == 200
        assert resp.json()["role"] == "admin"


class TestSuperadminOnlyScope:
    """/api/ui/workspaces (create) -- tightened past global_admin: being
    'admin' in one or more workspaces is NOT enough, only
    Identity.is_superadmin passes. See route_policy.py's comment on why
    (workspace create requires a document_dirs path that already exists on
    the server's own disk -- provisioning, not day-to-day administration).
    """

    def test_plain_admin_not_enough(self, app_and_client, provider, auth_db):
        _, client, _ = app_and_client
        _, raw_key = provider.create_key(subject="alice@acme.com", created_by="admin")
        auth_db.grant_access("marketing", "alice@acme.com", "admin", granted_by="admin")
        resp = client.post("/api/ui/workspaces", json={}, headers=_bearer(raw_key))
        assert resp.status_code == 403

    def test_member_denied(self, app_and_client, provider, auth_db):
        _, client, _ = app_and_client
        _, raw_key = provider.create_key(subject="alice@acme.com", created_by="admin")
        auth_db.grant_access("marketing", "alice@acme.com", "member", granted_by="admin")
        resp = client.post("/api/ui/workspaces", json={}, headers=_bearer(raw_key))
        assert resp.status_code == 403

    def test_no_grants_at_all_denied(self, app_and_client, provider):
        _, client, _ = app_and_client
        _, raw_key = provider.create_key(subject="alice@acme.com", created_by="admin")
        resp = client.post("/api/ui/workspaces", json={}, headers=_bearer(raw_key))
        assert resp.status_code == 403

    def test_superadmin_allowed(self, app_and_client, provider):
        _, client, _ = app_and_client
        _, raw_key = provider.create_key(
            subject="root@acme.com", created_by="admin", is_superadmin=True
        )
        resp = client.post("/api/ui/workspaces", json={}, headers=_bearer(raw_key))
        assert resp.status_code == 200
        assert resp.json()["role"] == "admin"


# ---------------------------------------------------------------------------
# End-to-end: immediate revocation (the design doc's core claim)
# ---------------------------------------------------------------------------


class TestEndToEndRevocation:
    def test_revocation_takes_effect_on_next_request_same_credential(
        self, app_and_client, provider, auth_db
    ):
        """
        1. Bearer-key auth resolves an identity (stand-in for "login").
        2. A request against a granted workspace succeeds.
        3. The same request against a non-granted workspace is denied.
        4. Admin revokes the grant.
        5. The SAME still-valid API key immediately gets denied on the
           very next request — proving revocation doesn't wait for the
           credential itself to expire (per-request DB check, not a
           cached grant baked into a token).
        """
        _, client, _ = app_and_client
        _, raw_key = provider.create_key(subject="alice@acme.com", created_by="admin")

        # Not yet granted anywhere.
        resp = client.post(
            "/api/ui/ingest", json={"workspace": "finance"}, headers=_bearer(raw_key)
        )
        assert resp.status_code == 403

        # Grant admin access to finance.
        auth_db.grant_access("finance", "alice@acme.com", "admin", granted_by="root")
        resp = client.post(
            "/api/ui/ingest", json={"workspace": "finance"}, headers=_bearer(raw_key)
        )
        assert resp.status_code == 200

        # Different, non-granted workspace still denied.
        resp = client.post(
            "/api/ui/ingest", json={"workspace": "marketing"}, headers=_bearer(raw_key)
        )
        assert resp.status_code == 403

        # Revoke — the API key itself is untouched (still valid, unexpired).
        auth_db.revoke_access("alice@acme.com", workspaces=["finance"])

        # Same credential, same workspace, immediately denied.
        resp = client.post(
            "/api/ui/ingest", json={"workspace": "finance"}, headers=_bearer(raw_key)
        )
        assert resp.status_code == 403

        # The key itself still verifies fine — only the grant was revoked.
        assert provider.verify_key(raw_key) is not None
