"""
Tests for the admin panel's backend surface: /api/admin/grants and
/api/admin/keys, plus their route_policy.py registration.

Covers:
- route_policy.py: GET/POST/DELETE for both /api/admin/grants and
  /api/admin/keys are registered as global_admin-scoped (a real gap was
  caught here during development — GET /api/admin/grants was originally
  missing from ROUTE_POLICIES; this guards against that regressing).
- build_grants_router(): grant/revoke/list against a real AuthDB,
  including granted_by being taken from request.state.identity.
- build_admin_keys_router(): create/revoke/list against a real
  APIKeyIdentityProvider, including the raw key being returned exactly
  once and never persisted.
- End-to-end via AuthorizationMiddleware + TestClient: an admin (of any
  workspace) can use both routers; a member-only or ungranted identity is
  denied 403; the raw key from a successful create never appears in a
  subsequent list response.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from local_search_agent.auth.admin_keys_routes import build_admin_keys_router
from local_search_agent.auth.api_key_provider import APIKeyIdentityProvider
from local_search_agent.auth.authorization_middleware import AuthorizationMiddleware
from local_search_agent.auth.grants_routes import build_grants_router
from local_search_agent.auth.route_policy import match_policy
from local_search_agent.workspace.auth_db import AuthDB

# ---------------------------------------------------------------------------
# route_policy.py registration
# ---------------------------------------------------------------------------


class TestRoutePolicyRegistration:
    @pytest.mark.parametrize("method", ["POST", "DELETE", "GET"])
    def test_grants_endpoint_registered(self, method):
        policy = match_policy(method, "/api/admin/grants")
        assert policy is not None, f"{method} /api/admin/grants missing from ROUTE_POLICIES"
        assert policy.scope == "global_admin"
        assert policy.required_role == "admin"

    @pytest.mark.parametrize("method", ["POST", "DELETE", "GET"])
    def test_keys_endpoint_registered(self, method):
        policy = match_policy(method, "/api/admin/keys")
        assert policy is not None, f"{method} /api/admin/keys missing from ROUTE_POLICIES"
        assert policy.scope == "global_admin"
        assert policy.required_role == "admin"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def auth_db(tmp_path):
    return AuthDB(db_path=str(tmp_path / "test.db"))


@pytest.fixture
def provider(auth_db):
    return APIKeyIdentityProvider(auth_db)


class _FakeIdentity:
    def __init__(self, subject, is_superadmin=True):
        self.subject = subject
        # Defaults to True: these are pre-existing CRUD unit tests for the
        # routers themselves (not tests of the superadmin-vs-admin
        # restriction added later), and "root@acme.com" was already the
        # implied trusted operator in this file's naming.
        self.is_superadmin = is_superadmin


class _FakeConfig:
    def __init__(self, identity_provider):
        self.identity_provider = identity_provider


def _build_app(config, auth_db, provider) -> FastAPI:
    app = FastAPI()
    app.add_middleware(AuthorizationMiddleware, config=config, auth_db=auth_db)
    app.include_router(build_grants_router(auth_db))
    app.include_router(build_admin_keys_router(provider))
    return app


def _bearer(raw_key: str) -> dict:
    return {"Authorization": f"Bearer {raw_key}"}


# ---------------------------------------------------------------------------
# build_grants_router() — direct unit tests (bypassing middleware)
# ---------------------------------------------------------------------------


class TestGrantsRouterUnit:
    def test_grant_uses_identity_subject_as_granted_by(self, auth_db):
        app = FastAPI()
        app.include_router(build_grants_router(auth_db))

        @app.middleware("http")
        async def inject_identity(request: Request, call_next):
            request.state.identity = _FakeIdentity("root@acme.com")
            return await call_next(request)

        client = TestClient(app)
        resp = client.post(
            "/api/admin/grants",
            json={"subject": "alice@acme.com", "workspaces": ["finance"], "role": "member"},
        )
        assert resp.status_code == 200
        rows = auth_db.list_access(subject="alice@acme.com")
        assert rows[0]["granted_by"] == "root@acme.com"

    def test_grant_invalid_role_returns_400(self, auth_db):
        app = FastAPI()
        app.include_router(build_grants_router(auth_db))

        @app.middleware("http")
        async def inject_identity(request: Request, call_next):
            request.state.identity = _FakeIdentity("root@acme.com")
            return await call_next(request)

        client = TestClient(app)
        resp = client.post(
            "/api/admin/grants",
            json={"subject": "alice@acme.com", "workspaces": ["finance"], "role": "owner"},
        )
        assert resp.status_code == 400

    def test_revoke_returns_deleted_count(self, auth_db):
        auth_db.grant_access("finance", "alice@acme.com", "member", granted_by="root")
        app = FastAPI()
        app.include_router(build_grants_router(auth_db))

        @app.middleware("http")
        async def inject_identity(request: Request, call_next):
            request.state.identity = _FakeIdentity("root@acme.com")
            return await call_next(request)

        client = TestClient(app)
        resp = client.request(
            "DELETE",
            "/api/admin/grants",
            json={"subject": "alice@acme.com", "workspaces": ["finance"]},
        )
        assert resp.status_code == 200
        assert resp.json()["revoked"] == 1

    def test_list_grants(self, auth_db):
        auth_db.grant_access("finance", "alice@acme.com", "admin", granted_by="root")
        app = FastAPI()
        app.include_router(build_grants_router(auth_db))
        client = TestClient(app)
        resp = client.get("/api/admin/grants", params={"workspace": "finance"})
        assert resp.status_code == 200
        assert resp.json()["grants"][0]["subject"] == "alice@acme.com"


# ---------------------------------------------------------------------------
# build_admin_keys_router() — direct unit tests
# ---------------------------------------------------------------------------


class TestAdminKeysRouterUnit:
    def test_create_key_returns_raw_key_once(self, provider):
        app = FastAPI()
        app.include_router(build_admin_keys_router(provider))

        @app.middleware("http")
        async def inject_identity(request: Request, call_next):
            request.state.identity = _FakeIdentity("root@acme.com")
            return await call_next(request)

        client = TestClient(app)
        resp = client.post("/api/admin/keys", json={"subject": "alice@acme.com"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["raw_key"].startswith("lsa_")
        assert body["key_id"]

    def test_created_by_comes_from_identity(self, provider, auth_db):
        app = FastAPI()
        app.include_router(build_admin_keys_router(provider))

        @app.middleware("http")
        async def inject_identity(request: Request, call_next):
            request.state.identity = _FakeIdentity("root@acme.com")
            return await call_next(request)

        client = TestClient(app)
        resp = client.post("/api/admin/keys", json={"subject": "alice@acme.com"})
        key_id = resp.json()["key_id"]
        row = auth_db.get_api_key(key_id)
        assert row["created_by"] == "root@acme.com"

    def test_list_keys_never_includes_raw_key_or_hash(self, provider):
        app = FastAPI()
        app.include_router(build_admin_keys_router(provider))

        @app.middleware("http")
        async def inject_identity(request: Request, call_next):
            request.state.identity = _FakeIdentity("root@acme.com")
            return await call_next(request)

        client = TestClient(app)
        create_resp = client.post("/api/admin/keys", json={"subject": "alice@acme.com"})
        raw_key = create_resp.json()["raw_key"]

        list_resp = client.get("/api/admin/keys")
        assert list_resp.status_code == 200
        assert raw_key not in list_resp.text
        assert "key_hash" not in list_resp.json()["keys"][0]

    def test_revoke_key(self, provider):
        app = FastAPI()
        app.include_router(build_admin_keys_router(provider))

        @app.middleware("http")
        async def inject_identity(request: Request, call_next):
            request.state.identity = _FakeIdentity("root@acme.com")
            return await call_next(request)

        client = TestClient(app)
        key_id = client.post("/api/admin/keys", json={"subject": "alice@acme.com"}).json()["key_id"]
        resp = client.request("DELETE", "/api/admin/keys", json={"key_id": key_id})
        assert resp.status_code == 200
        assert resp.json()["revoked"] is True


# ---------------------------------------------------------------------------
# End-to-end through AuthorizationMiddleware
# ---------------------------------------------------------------------------


class TestEndToEndAdminPanel:
    def test_admin_of_any_workspace_can_grant(self, provider, auth_db):
        config = _FakeConfig(identity_provider=provider)
        app = _build_app(config, auth_db, provider)
        client = TestClient(app)

        _, admin_key = provider.create_key(subject="root@acme.com", created_by="bootstrap")
        auth_db.grant_access("marketing", "root@acme.com", "admin", granted_by="bootstrap")

        resp = client.post(
            "/api/admin/grants",
            json={"subject": "alice@acme.com", "workspaces": ["finance"], "role": "member"},
            headers=_bearer(admin_key),
        )
        assert resp.status_code == 200
        assert auth_db.get_role("alice@acme.com", "finance") == "member"

    def test_member_only_identity_denied(self, provider, auth_db):
        config = _FakeConfig(identity_provider=provider)
        app = _build_app(config, auth_db, provider)
        client = TestClient(app)

        _, member_key = provider.create_key(subject="bob@acme.com", created_by="root")
        auth_db.grant_access("finance", "bob@acme.com", "member", granted_by="root")

        resp = client.post(
            "/api/admin/grants",
            json={"subject": "alice@acme.com", "workspaces": ["finance"], "role": "member"},
            headers=_bearer(member_key),
        )
        assert resp.status_code == 403

    def test_no_grants_at_all_denied(self, provider, auth_db):
        config = _FakeConfig(identity_provider=provider)
        app = _build_app(config, auth_db, provider)
        client = TestClient(app)

        _, raw_key = provider.create_key(subject="nobody@acme.com", created_by="root")
        resp = client.get("/api/admin/keys", headers=_bearer(raw_key))
        assert resp.status_code == 403

    def test_no_credential_denied(self, provider, auth_db):
        config = _FakeConfig(identity_provider=provider)
        app = _build_app(config, auth_db, provider)
        client = TestClient(app)
        resp = client.get("/api/admin/grants")
        assert resp.status_code == 401

    def test_admin_can_create_and_list_keys(self, provider, auth_db):
        config = _FakeConfig(identity_provider=provider)
        app = _build_app(config, auth_db, provider)
        client = TestClient(app)

        _, admin_key = provider.create_key(subject="root@acme.com", created_by="bootstrap")
        auth_db.grant_access("marketing", "root@acme.com", "admin", granted_by="bootstrap")

        resp = client.post(
            "/api/admin/keys", json={"subject": "carol@acme.com"}, headers=_bearer(admin_key)
        )
        assert resp.status_code == 200

        list_resp = client.get("/api/admin/keys", headers=_bearer(admin_key))
        subjects = [k["subject"] for k in list_resp.json()["keys"]]
        assert "carol@acme.com" in subjects
