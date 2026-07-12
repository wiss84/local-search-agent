"""
Tests for Model / Provider Access Control (Option B: true per-request
model selection -- see upcoming_features/08-security-hardening-and-remaining-work.md,
section 2, and the follow-up conversation that chose Option B over
Option A).

Covers three layers:
1. AuthDB's model_access_by_role CRUD (grant/revoke/list/is_model_allowed/
   role_allowed_models) -- the data layer everything else sits on.
2. route_policy.py's new entries for GET /api/ui/models/allowed and the
   GET/POST/DELETE /api/ui/models/access management routes.
3. POST /api/ui/query's actual enforcement decision -- denies a request
   whose effective provider/model isn't on the caller's current role's
   allow-list, allows one that is, and bypasses the check entirely for
   superadmins and single-user mode (no identity_provider configured).

Layer 3 uses a fake, fast LocalSearchAgent stand-in (via monkeypatching
AppState.get_agent) rather than a real LLM/Meilisearch connection, so
these tests stay fast and deterministic -- the enforcement check itself
runs (and can deny) before any agent is ever built, so only the "allowed"
path needs a working fake agent to drain the SSE stream without hitting
a real network call.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from local_search_agent.auth.route_policy import match_policy
from local_search_agent.ui.api_routes import build_ui_router
from local_search_agent.ui.store import UIStore
from local_search_agent.workspace.auth_db import AuthDB

# ---------------------------------------------------------------------------
# Layer 1: AuthDB's model_access_by_role CRUD
# ---------------------------------------------------------------------------


@pytest.fixture
def auth_db(tmp_path):
    return AuthDB(db_path=str(tmp_path / "test.db"))


class TestModelAccessCRUD:
    def test_role_with_no_rows_allows_nothing(self, auth_db):
        assert auth_db.is_model_allowed("member", "google", "gemma-4-31b-it") is False
        assert auth_db.role_allowed_models("member") == {}

    def test_grant_then_allowed(self, auth_db):
        auth_db.grant_model_access("member", "google", "gemma-4-31b-it", granted_by="root")
        assert auth_db.is_model_allowed("member", "google", "gemma-4-31b-it") is True
        assert auth_db.is_model_allowed("member", "google", "some-other-model") is False

    def test_revoke_removes_access(self, auth_db):
        auth_db.grant_model_access("member", "google", "gemma-4-31b-it", granted_by="root")
        revoked = auth_db.revoke_model_access("member", "google", "gemma-4-31b-it")
        assert revoked is True
        assert auth_db.is_model_allowed("member", "google", "gemma-4-31b-it") is False

    def test_revoke_nonexistent_returns_false(self, auth_db):
        assert auth_db.revoke_model_access("member", "google", "nope") is False

    def test_roles_are_independent(self, auth_db):
        """member and admin allow-lists are separate; granting one never
        implicitly grants the other."""
        auth_db.grant_model_access("admin", "openai", "gpt-5", granted_by="root")
        assert auth_db.is_model_allowed("admin", "openai", "gpt-5") is True
        assert auth_db.is_model_allowed("member", "openai", "gpt-5") is False

    def test_role_allowed_models_groups_by_provider(self, auth_db):
        auth_db.grant_model_access("member", "google", "gemma-4-31b-it", granted_by="root")
        auth_db.grant_model_access("member", "google", "gemini-3.1-flash-lite", granted_by="root")
        auth_db.grant_model_access("member", "ollama", "mistral", granted_by="root")
        grouped = auth_db.role_allowed_models("member")
        assert set(grouped["google"]) == {"gemma-4-31b-it", "gemini-3.1-flash-lite"}
        assert grouped["ollama"] == ["mistral"]

    def test_invalid_role_rejected(self, auth_db):
        with pytest.raises(ValueError):
            auth_db.grant_model_access("superadmin", "google", "x", granted_by="root")

    def test_list_model_access_filters_by_role(self, auth_db):
        auth_db.grant_model_access("member", "google", "a", granted_by="root")
        auth_db.grant_model_access("admin", "google", "b", granted_by="root")
        member_rows = auth_db.list_model_access(role="member")
        assert len(member_rows) == 1
        assert member_rows[0]["model_name"] == "a"
        all_rows = auth_db.list_model_access()
        assert len(all_rows) == 2


# ---------------------------------------------------------------------------
# Layer 2: RoutePolicy entries
# ---------------------------------------------------------------------------


class TestModelAccessRoutePolicyEntries:
    def test_models_allowed_is_member_workspace_scoped(self):
        policy = match_policy("GET", "/api/ui/models/allowed")
        assert policy is not None
        assert policy.scope == "workspace"
        assert policy.required_role == "member"

    @pytest.mark.parametrize("method", ["GET", "POST", "DELETE"])
    def test_models_access_is_global_admin_scoped(self, method):
        policy = match_policy(method, "/api/ui/models/access")
        assert policy is not None
        assert policy.scope == "global_admin"
        assert policy.required_role == "admin"

    def test_models_allowed_does_not_collide_with_plain_models_route(self):
        """GET /api/ui/models (Model Manager) and GET /api/ui/models/allowed
        must resolve to different policies -- a regex without a $ anchor
        on /models could otherwise swallow the /allowed sub-path too."""
        plain = match_policy("GET", "/api/ui/models")
        allowed = match_policy("GET", "/api/ui/models/allowed")
        assert plain.scope == "global_admin"
        assert allowed.scope == "workspace"


# ---------------------------------------------------------------------------
# Layer 3: POST /api/ui/query's enforcement decision
# ---------------------------------------------------------------------------


class _FakeAgent:
    """Minimal LocalSearchAgent stand-in -- no real LLM/Meilisearch call."""

    def _get_tools(self):
        return []

    def stream(self, question, workspace=None):
        yield {"type": "text", "text": "fake answer"}
        yield {
            "type": "done",
            "token_input": 1,
            "token_output": 1,
            "iterations_used": 1,
            "truncated": False,
            "sources": [],
        }


class _FakeConfig:
    def __init__(self, identity_provider=None, provider="google", model_name="gemma-4-31b-it"):
        self.identity_provider = identity_provider
        self.provider = provider
        self.model_name = model_name


class _FakeAppState:
    def __init__(self, store, auth_db, identity_provider=None):
        self.store = store
        self.auth_db = auth_db
        self.config = _FakeConfig(identity_provider=identity_provider)

    def get_agent(self, workspace=None, meili_api_key=None, provider=None, model_name=None):
        return _FakeAgent()


def _identity_stub_middleware(identity, role):
    """A tiny Starlette middleware standing in for AuthorizationMiddleware
    -- sets request.state exactly as it would have after resolving this
    identity/role for a workspace-scoped request, without needing a real
    IdentityProvider or workspace_members grant plumbed through."""

    async def middleware(request: Request, call_next):
        request.state.identity = identity
        request.state.role = role
        return await call_next(request)

    return middleware


class _FakeIdentity:
    def __init__(self, subject, is_superadmin=False):
        self.subject = subject
        self.is_superadmin = is_superadmin


@pytest.fixture
def store(tmp_path):
    return UIStore(db_path=str(tmp_path / "test.db"))


def _build_app(app_state, identity=None, role=None):
    app = FastAPI()
    if identity is not None:
        app.middleware("http")(_identity_stub_middleware(identity, role))
    app.include_router(build_ui_router(app_state))
    return app


class TestQueryModelEnforcement:
    def test_single_user_mode_bypasses_check_entirely(self, store, auth_db):
        """No identity_provider at all -- Option B/section 2 doesn't apply,
        per the design doc's explicit early-exit requirement."""
        app_state = _FakeAppState(store, auth_db, identity_provider=None)
        session = store.create_session(workspace="finance", title="t")
        app = _build_app(app_state)  # no identity middleware at all
        client = TestClient(app)
        resp = client.post(
            "/api/ui/query",
            json={"session_id": session["session_id"], "question": "hi", "workspace": "finance"},
        )
        assert resp.status_code == 200

    def test_superadmin_bypasses_check(self, store, auth_db):
        app_state = _FakeAppState(store, auth_db, identity_provider=object())
        session = store.create_session(workspace="finance", title="t")
        identity = _FakeIdentity("boss@acme.com", is_superadmin=True)
        app = _build_app(app_state, identity=identity, role="admin")
        client = TestClient(app)
        resp = client.post(
            "/api/ui/query",
            json={"session_id": session["session_id"], "question": "hi", "workspace": "finance"},
        )
        assert resp.status_code == 200

    def test_member_denied_model_not_on_allow_list(self, store, auth_db):
        """Zero rows granted for 'member' -- fail-closed, same as every
        other allow-list in this system."""
        app_state = _FakeAppState(store, auth_db, identity_provider=object())
        session = store.create_session(workspace="finance", title="t")
        identity = _FakeIdentity("alice@acme.com", is_superadmin=False)
        app = _build_app(app_state, identity=identity, role="member")
        client = TestClient(app)
        resp = client.post(
            "/api/ui/query",
            json={"session_id": session["session_id"], "question": "hi", "workspace": "finance"},
        )
        assert resp.status_code == 403

    def test_member_allowed_when_default_model_is_granted(self, store, auth_db):
        auth_db.grant_model_access("member", "google", "gemma-4-31b-it", granted_by="root")
        app_state = _FakeAppState(store, auth_db, identity_provider=object())
        session = store.create_session(workspace="finance", title="t")
        identity = _FakeIdentity("alice@acme.com", is_superadmin=False)
        app = _build_app(app_state, identity=identity, role="member")
        client = TestClient(app)
        resp = client.post(
            "/api/ui/query",
            json={"session_id": session["session_id"], "question": "hi", "workspace": "finance"},
        )
        assert resp.status_code == 200

    def test_option_b_per_request_override_checked_not_shared_default(self, store, auth_db):
        """The whole point of Option B: a request can ask for a DIFFERENT
        model than the shared app_state.config default, and that explicit
        override -- not the shared default -- is what gets checked."""
        # Member is granted access to "openai/gpt-5-mini" specifically,
        # NOT the shared default (google/gemma-4-31b-it).
        auth_db.grant_model_access("member", "openai", "gpt-5-mini", granted_by="root")
        app_state = _FakeAppState(store, auth_db, identity_provider=object())
        session = store.create_session(workspace="finance", title="t")
        identity = _FakeIdentity("alice@acme.com", is_superadmin=False)
        app = _build_app(app_state, identity=identity, role="member")
        client = TestClient(app)

        # Explicit override matching the grant -> allowed.
        resp_allowed = client.post(
            "/api/ui/query",
            json={
                "session_id": session["session_id"],
                "question": "hi",
                "workspace": "finance",
                "provider": "openai",
                "model_name": "gpt-5-mini",
            },
        )
        assert resp_allowed.status_code == 200

        # No override -> falls back to the shared default, which this
        # member was never granted -> denied.
        resp_denied = client.post(
            "/api/ui/query",
            json={"session_id": session["session_id"], "question": "hi", "workspace": "finance"},
        )
        assert resp_denied.status_code == 403

    def test_admin_and_member_allow_lists_are_independent(self, store, auth_db):
        """A subject who is 'admin' in the workspace they're querying gets
        checked against the admin allow-list, not the member one, even if
        the two differ."""
        auth_db.grant_model_access("admin", "openai", "gpt-5", granted_by="root")
        # Deliberately do NOT grant member access to the same model.
        app_state = _FakeAppState(store, auth_db, identity_provider=object())
        session = store.create_session(workspace="finance", title="t")
        identity = _FakeIdentity("alice@acme.com", is_superadmin=False)
        app = _build_app(app_state, identity=identity, role="admin")
        client = TestClient(app)
        resp = client.post(
            "/api/ui/query",
            json={
                "session_id": session["session_id"],
                "question": "hi",
                "workspace": "finance",
                "provider": "openai",
                "model_name": "gpt-5",
            },
        )
        assert resp.status_code == 200
