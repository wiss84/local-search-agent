"""
Tests for multi-tenant RBAC: whoami endpoint + frontend role-gating.

Covers:
- GET /api/ui/whoami: single-user mode (no identity_provider), no
  credential (401), authenticated with no workspace param, authenticated
  with a workspace param (grant present / absent), superadmin bypass
- Template rendering: index.html (and the new _script_role_gating.html
  partial) render through the real Jinja2 environment without error, and
  the expected data-requires-role hooks / JS function are present in the
  rendered output — a lightweight sanity check that the frontend wiring
  didn't break template stitching, not a JS behavior test.
"""

from __future__ import annotations

import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from local_search_agent.auth.api_key_provider import APIKeyIdentityProvider
from local_search_agent.auth.whoami_route import build_whoami_router
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
    def __init__(self, identity_provider):
        self.identity_provider = identity_provider


def _build_app(config, auth_db) -> FastAPI:
    app = FastAPI()
    app.include_router(build_whoami_router(config, auth_db))
    return app


# ---------------------------------------------------------------------------
# whoami: single-user mode (no identity_provider)
# ---------------------------------------------------------------------------


class TestWhoamiSingleUserMode:
    def test_no_identity_provider_returns_multi_tenant_false(self, auth_db):
        config = _FakeConfig(identity_provider=None)
        client = TestClient(_build_app(config, auth_db))
        resp = client.get("/api/ui/whoami")
        assert resp.status_code == 200
        body = resp.json()
        assert body["multi_tenant"] is False
        assert body["subject"] is None
        assert body["role"] is None

    def test_no_identity_provider_ignores_workspace_param(self, auth_db):
        config = _FakeConfig(identity_provider=None)
        client = TestClient(_build_app(config, auth_db))
        resp = client.get("/api/ui/whoami", params={"workspace": "finance"})
        assert resp.status_code == 200
        assert resp.json()["multi_tenant"] is False


# ---------------------------------------------------------------------------
# whoami: multi-tenant mode
# ---------------------------------------------------------------------------


class TestWhoamiMultiTenant:
    def test_no_credential_returns_401(self, provider, auth_db):
        config = _FakeConfig(identity_provider=provider)
        client = TestClient(_build_app(config, auth_db))
        resp = client.get("/api/ui/whoami")
        assert resp.status_code == 401

    def test_bad_credential_returns_401(self, provider, auth_db):
        config = _FakeConfig(identity_provider=provider)
        client = TestClient(_build_app(config, auth_db))
        resp = client.get("/api/ui/whoami", headers={"Authorization": "Bearer lsa_bad_key"})
        assert resp.status_code == 401

    def test_valid_credential_no_workspace_param(self, provider, auth_db):
        config = _FakeConfig(identity_provider=provider)
        client = TestClient(_build_app(config, auth_db))
        _, raw_key = provider.create_key(
            subject="alice@acme.com", created_by="admin", display_name="Alice"
        )
        resp = client.get("/api/ui/whoami", headers={"Authorization": f"Bearer {raw_key}"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["multi_tenant"] is True
        assert body["subject"] == "alice@acme.com"
        assert body["display_name"] == "Alice"
        assert body["role"] is None  # no workspace specified — can't resolve a role
        assert body["workspace"] is None

    def test_valid_credential_workspace_with_grant(self, provider, auth_db):
        config = _FakeConfig(identity_provider=provider)
        client = TestClient(_build_app(config, auth_db))
        _, raw_key = provider.create_key(subject="alice@acme.com", created_by="admin")
        auth_db.grant_access("finance", "alice@acme.com", "admin", granted_by="root")

        resp = client.get(
            "/api/ui/whoami",
            params={"workspace": "finance"},
            headers={"Authorization": f"Bearer {raw_key}"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["role"] == "admin"
        assert body["workspace"] == "finance"

    def test_valid_credential_workspace_without_grant(self, provider, auth_db):
        config = _FakeConfig(identity_provider=provider)
        client = TestClient(_build_app(config, auth_db))
        _, raw_key = provider.create_key(subject="alice@acme.com", created_by="admin")

        resp = client.get(
            "/api/ui/whoami",
            params={"workspace": "finance"},
            headers={"Authorization": f"Bearer {raw_key}"},
        )
        assert resp.status_code == 200
        assert resp.json()["role"] is None

    def test_different_role_per_workspace(self, provider, auth_db):
        config = _FakeConfig(identity_provider=provider)
        client = TestClient(_build_app(config, auth_db))
        _, raw_key = provider.create_key(subject="alice@acme.com", created_by="admin")
        auth_db.grant_access("finance", "alice@acme.com", "admin", granted_by="root")
        auth_db.grant_access("marketing", "alice@acme.com", "member", granted_by="root")
        headers = {"Authorization": f"Bearer {raw_key}"}

        resp1 = client.get("/api/ui/whoami", params={"workspace": "finance"}, headers=headers)
        resp2 = client.get("/api/ui/whoami", params={"workspace": "marketing"}, headers=headers)
        assert resp1.json()["role"] == "admin"
        assert resp2.json()["role"] == "member"

    def test_superadmin_gets_admin_role_without_grant(self, provider, auth_db):
        config = _FakeConfig(identity_provider=provider)
        client = TestClient(_build_app(config, auth_db))
        _, raw_key = provider.create_key(
            subject="root@acme.com", created_by="admin", is_superadmin=True
        )
        resp = client.get(
            "/api/ui/whoami",
            params={"workspace": "finance"},
            headers={"Authorization": f"Bearer {raw_key}"},
        )
        assert resp.json()["role"] == "admin"
        assert resp.json()["is_superadmin"] is True


# ---------------------------------------------------------------------------
# Template rendering sanity check
# ---------------------------------------------------------------------------


TEMPLATES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "local_search_agent",
    "ui",
    "templates",
)


class TestRoleGatingTemplateWiring:
    def test_index_html_renders_without_error(self):
        from jinja2 import Environment, FileSystemLoader

        env = Environment(loader=FileSystemLoader(TEMPLATES_DIR), autoescape=False)
        template = env.get_template("index.html")
        html = template.render(port=8765, file_server_port=8000, version="test")
        assert "<html" in html

    def test_data_requires_role_present_on_admin_buttons(self):
        from jinja2 import Environment, FileSystemLoader

        env = Environment(loader=FileSystemLoader(TEMPLATES_DIR), autoescape=False)
        template = env.get_template("index.html")
        html = template.render(port=8765, file_server_port=8000, version="test")

        for button_id in (
            "btn-ingest",
            "btn-scheduler",
            "btn-settings",
        ):
            # Crude but effective: the button's id and the data-requires-role
            # attribute should appear close together in the same tag.
            idx = html.index(f'id="{button_id}"')
            snippet = html[max(0, idx - 200) : idx + 200]
            assert 'data-requires-role="admin"' in snippet, f"{button_id} missing role gate"

    def test_data_requires_superadmin_present_on_tightened_buttons(self):
        """Workspace create/delete and force re-ingest/wipe & re-ingest
        were tightened past ordinary admin to superadmin-only (see
        route_policy.py + api_routes.py) -- these buttons must carry the
        stricter data-requires-superadmin gate, not the old
        data-requires-role="admin" one.
        """
        from jinja2 import Environment, FileSystemLoader

        env = Environment(loader=FileSystemLoader(TEMPLATES_DIR), autoescape=False)
        template = env.get_template("index.html")
        html = template.render(port=8765, file_server_port=8000, version="test")

        for button_id in (
            "btn-delete-workspace",
            "btn-new-workspace",
            "btn-force-ingest",
            "btn-wipe-ingest",
        ):
            idx = html.index(f'id="{button_id}"')
            snippet = html[max(0, idx - 200) : idx + 200]
            assert 'data-requires-superadmin="true"' in snippet, (
                f"{button_id} missing superadmin gate"
            )

    def test_role_gating_js_function_defined(self):
        with open(os.path.join(TEMPLATES_DIR, "_script_role_gating.html"), encoding="utf-8") as f:
            js = f.read()
        assert "function applyRoleGating" in js
        assert "function refreshRoleGating" in js
        assert "/whoami" in js
        assert "data-requires-superadmin" in js
        assert "state.isSuperadmin" in js

    def test_boot_calls_refresh_role_gating(self):
        with open(os.path.join(TEMPLATES_DIR, "_script_models_boot.html"), encoding="utf-8") as f:
            js = f.read()
        assert "refreshRoleGating()" in js

    def test_workspace_change_calls_refresh_role_gating(self):
        with open(
            os.path.join(TEMPLATES_DIR, "_script_health_workspaces_sessions.html"),
            encoding="utf-8",
        ) as f:
            js = f.read()
        assert "refreshRoleGating()" in js

    def test_dynamic_session_delete_button_has_no_admin_role_gate(self):
        """Session delete was changed from admin-only to ownership-checked
        (any member may delete a session they created; the server-side
        handler enforces created_by == subject, not a role tier -- see
        api_routes.py's delete_session() and route_policy.py's own
        comment on why this couldn't be expressed as a RoutePolicy entry).
        The button must NOT carry the old admin-only client-side gate
        anymore, or a member would never see a delete button for their own
        sessions despite the backend now allowing it.
        """
        with open(
            os.path.join(TEMPLATES_DIR, "_script_health_workspaces_sessions.html"),
            encoding="utf-8",
        ) as f:
            js = f.read()
        assert "btn-delete-session" in js
        idx = js.index("btn-delete-session")
        snippet = js[max(0, idx - 200) : idx + 200]
        assert 'data-requires-role="admin"' not in snippet


# ---------------------------------------------------------------------------
# Regression guard: whoami must be mounted unconditionally in dashboard.py
# ---------------------------------------------------------------------------
# A real bug shipped here: build_whoami_router(...) was originally called
# only inside `if app_state.config.identity_provider is not None:`. Since
# boot() calls /api/ui/whoami on every launch regardless, single-user
# installs (identity_provider=None, the default) got a 404 — which
# refreshRoleGating()'s catch block then (correctly, for genuine
# multi-tenant failures) treats as fail-closed, silently disabling every
# gated button for people who never opted into multi-tenant mode at all.
# Caught by manual click-through, not by TestWhoamiSingleUserMode above,
# because that class tests build_whoami_router() in isolation — its
# internal None-handling was always correct. The bug was entirely in
# *when dashboard.py mounts it*, so the regression guard has to inspect
# the mounting call site itself, not just the router's behavior.


class TestWhoamiAlwaysMounted:
    def test_whoami_router_included_unconditionally_in_dashboard(self):
        import inspect

        from local_search_agent.ui.dashboard import build_dashboard_app

        source = inspect.getsource(build_dashboard_app)
        lines = source.splitlines()
        target_idx = next(
            i
            for i, line in enumerate(lines)
            if "build_whoami_router(" in line and "include_router" in line
        )
        indent = len(lines[target_idx]) - len(lines[target_idx].lstrip())
        # Top-level statements inside build_dashboard_app sit at 4 spaces;
        # anything nested inside the `if identity_provider is not None:`
        # block sits at 8+. If this assertion fails, the bug is back.
        assert indent <= 4, (
            "whoami router's include_router() call appears nested inside a "
            "conditional block — this regresses to a 404 for single-user "
            "installs (identity_provider=None) on every UI load."
        )
