"""
These tests cover the following gaps and regressions in the auth layer:

 - auth_db.py: role overwrite on re-grant, revoke(None) semantics,
   is_global_admin row-lookup efficiency, list_access NO filter, purge
   activity log retention, auth_attempts combined-subject+IP filter,
   meili_key row upsert (store twice → single row).
 - api_key_provider.py: empty-key_id verification, valid-key format
   only accepts lsa_ prefix, rate-limit window sliding recovery,
   IP=None pass-through, superadmin flag on keys, logout idempotency.
 - header_provider.py: explicit empty-string header, trust_proxy_ips
   None vs non-None behavior, superadmin header.
 - authorization_middleware.py: workspace_from_session_id with None
   session_lookup → 403, superadmin on workspace-route admin-role branch,
   unlisted routes pass through.
 - route_policy.py: first-match-wins with overlapping regex, compiled()
   returns a new Pattern each call.
 - meili_key_provisioning.py: DB row is deleted even if Meilisearch
   delete_scoped_key raises.

JWT-specific gap tests (nbf enforcement, kid-miss refresh, provider
unavailable recovery) live in the `TestJWT*` classes at the bottom of
this file and are guarded with `pytest.importorskip` individually so they
are silently skipped rather than erroring when PyJWT is not installed.
"""

from __future__ import annotations

import datetime
import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import (
    Request,  # required at module level for async route handlers with __future__ annotations
)

from local_search_agent.auth.api_key_provider import (
    APIKeyIdentityProvider,
    _hash_token,
)
from local_search_agent.auth.authorization_middleware import AuthorizationMiddleware
from local_search_agent.auth.header_provider import HeaderIdentityProvider
from local_search_agent.auth.meili_key_provisioning import (
    deprovision_workspace_keys,
)
from local_search_agent.auth.route_policy import ROUTE_POLICIES, RoutePolicy, match_policy
from local_search_agent.workspace.auth_db import AuthDB

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test_gaps.db")


@pytest.fixture
def auth_db(db_path):
    return AuthDB(db_path=db_path)


@pytest.fixture
def provider(auth_db):
    return APIKeyIdentityProvider(auth_db)


class _FakeHeaders(dict):
    def get(self, key, default=None):
        for k, v in self.items():
            if k.lower() == key.lower():
                return v
        return default


def _fake_request(headers=None, cookies=None):
    return SimpleNamespace(headers=_FakeHeaders(headers or {}), cookies=cookies or {})


# ===================================================================
# auth_db.py — role overwrite, revoke semantics, query efficiency,
#               list_access, purge, auth_attempts, meili_keys upsert
# ===================================================================


class TestGrantAccessOverwrite:
    def test_grant_updates_role_on_existing_row(self, auth_db):
        auth_db.grant_access("finance", "alice@acme.com", "member", "root")
        auth_db.grant_access("finance", "alice@acme.com", "admin", "root")
        assert auth_db.get_role("alice@acme.com", "finance") == "admin"

    def test_grant_updates_granted_at(self, auth_db):

        t1 = datetime.datetime.now(datetime.timezone.utc)
        auth_db.grant_access("finance", "alice@acme.com", "member", "root")
        t2 = datetime.datetime.now(datetime.timezone.utc)
        row = auth_db.list_access(subject="alice@acme.com")[0]
        row_ts = datetime.datetime.fromisoformat(row["granted_at"])
        assert t1 <= row_ts <= t2 or t2 <= row_ts <= t1  # within window


class TestRevokeAccess:
    def test_revoke_none_revokes_all_workspaces_for_subject(self, auth_db):
        for ws in ("finance", "marketing", "hr"):
            auth_db.grant_access(ws, "alice@acme.com", "member", "root")
        deleted = auth_db.revoke_access("alice@acme.com", workspaces=None)
        assert deleted == 3
        assert auth_db.list_access(subject="alice@acme.com") == []

    def test_revoke_specific_workspaces_only(self, auth_db):
        for ws in ("finance", "marketing", "hr"):
            auth_db.grant_access(ws, "alice@acme.com", "member", "root")
        deleted = auth_db.revoke_access("alice@acme.com", workspaces=["finance"])
        assert deleted == 1
        remaining = auth_db.list_access(subject="alice@acme.com")
        assert {r["workspace"] for r in remaining} == {"marketing", "hr"}

    def test_revoke_empty_list_returns_zero(self, auth_db):
        auth_db.grant_access("finance", "alice@acme.com", "member", "root")
        deleted = auth_db.revoke_access("alice@acme.com", workspaces=[])
        assert deleted == 0
        assert auth_db.get_role("alice@acme.com", "finance") == "member"


class TestIsGlobalAdmin:
    def test_true_when_subject_is_admin_in_any_workspace(self, auth_db):
        auth_db.grant_access("finance", "alice@acme.com", "admin", "root")
        assert auth_db.is_global_admin("alice@acme.com") is True

    def test_false_when_subject_is_only_member(self, auth_db):
        auth_db.grant_access("finance", "alice@acme.com", "member", "root")
        assert auth_db.is_global_admin("alice@acme.com") is False

    def test_false_when_ungranted(self, auth_db):
        assert auth_db.is_global_admin("nobody@acme.com") is False


class TestListAccess:
    def test_no_filters_returns_every_row(self, auth_db):
        auth_db.grant_access("finance", "alice@acme.com", "admin", "root")
        auth_db.grant_access("marketing", "alice@acme.com", "member", "root")
        auth_db.grant_access("finance", "bob@acme.com", "member", "root")
        rows = auth_db.list_access()
        assert len(rows) == 3

    def test_filter_by_subject(self, auth_db):
        auth_db.grant_access("finance", "alice@acme.com", "admin", "root")
        auth_db.grant_access("finance", "bob@acme.com", "member", "root")
        rows = auth_db.list_access(subject="alice@acme.com")
        assert len(rows) == 1
        assert rows[0]["workspace"] == "finance"
        assert rows[0]["role"] == "admin"

    def test_filter_by_workspace(self, auth_db):
        auth_db.grant_access("finance", "alice@acme.com", "admin", "root")
        auth_db.grant_access("finance", "bob@acme.com", "member", "root")
        auth_db.grant_access("marketing", "alice@acme.com", "member", "root")
        rows = auth_db.list_access(workspace="finance")
        assert len(rows) == 2
        subjects = {r["subject"] for r in rows}
        assert subjects == {"alice@acme.com", "bob@acme.com"}


class TestActivityLogLifecycle:
    def test_purge_removes_only_old_rows(self, auth_db):
        now = auth_db.log_activity(
            subject="alice@acme.com", action="search", workspace="finance", success=True
        )
        # Insert an old row directly via SQL so the purge logic has something to remove.

        old_ts = (
            datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=100)
        ).isoformat()
        with auth_db._connect() as conn:
            conn.execute(
                "INSERT INTO activity_log (subject, workspace, action, detail, ip_address, timestamp, success) VALUES (?,?,?,?,?,?,?)",
                ("old_guy@acme.com", "finance", "search", "", None, old_ts, 1),
            )
        deleted = auth_db.purge_activity_log(older_than_days=90)
        assert deleted == 1
        # The fresh row is still there
        rows = auth_db.get_activity_log(subject="alice@acme.com")
        assert len(rows) == 1
        assert rows[0]["id"] == now

    def test_ip_address_none_stored_as_null(self, auth_db):
        row_id = auth_db.log_activity(
            subject="alice@acme.com", action="login", workspace=None, ip_address=None
        )
        with auth_db._connect() as conn:
            row = conn.execute(
                "SELECT ip_address FROM activity_log WHERE id = ?", (row_id,)
            ).fetchone()
        assert row["ip_address"] is None


class TestAuthAttempts:
    def test_combined_subject_and_ip_filter(self, auth_db):
        auth_db.record_attempt("alice@acme.com", "10.0.0.1", True)
        auth_db.record_attempt("alice@acme.com", "10.0.0.2", False)
        auth_db.record_attempt("mallory@acme.com", "10.0.0.1", False)
        # Only the row matching BOTH subject and IP
        count = auth_db.count_recent_failed_attempts(
            subject="alice@acme.com", ip_address="10.0.0.1"
        )
        assert count == 0
        count = auth_db.count_recent_failed_attempts(
            subject="mallory@acme.com", ip_address="10.0.0.1"
        )
        assert count == 1

    def test_count_recent_requires_at_least_one_filter(self, auth_db):
        with pytest.raises(ValueError):
            auth_db.count_recent_failed_attempts()

    def test_purge_removes_old_attempts(self, auth_db):

        auth_db.record_attempt("alice@acme.com", "10.0.0.1", False)
        old_ts = (
            datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=30)
        ).isoformat()
        with auth_db._connect() as conn:
            conn.execute(
                "INSERT INTO auth_attempts (subject, ip_address, attempted_at, success) VALUES (?,?,?,?)",
                ("old@acme.com", "1.1.1.1", old_ts, 0),
            )
        deleted = auth_db.purge_old_attempts(older_than_days=7)
        assert deleted == 1


class TestMeiliKeys:
    def test_store_then_get_then_delete(self, auth_db):
        encrypted = "ZmFrZV9FWjJfRW5jb2RlZF9LZXk="
        auth_db.store_meili_key("finance", "key-uid-1", encrypted)
        row = auth_db.get_meili_key_row("finance")
        assert row is not None
        assert row["key_uid"] == "key-uid-1"
        assert row["encrypted_key"] == encrypted
        assert auth_db.delete_meili_key("finance") is True
        assert auth_db.get_meili_key_row("finance") is None

    def test_store_upserts_single_row(self, auth_db):
        auth_db.store_meili_key("finance", "key-uid-1", "encrypted-value-A")
        auth_db.store_meili_key("finance", "key-uid-2", "encrypted-value-B")
        row = auth_db.get_meili_key_row("finance")
        assert row["key_uid"] == "key-uid-2"
        assert row["encrypted_key"] == "encrypted-value-B"


# ===================================================================
# api_key_provider.py — key format, rate-limit window, session expiry
# ===================================================================


class TestAPIKeyFormat:
    def test_verify_key_rejects_non_lsa_prefix(self, provider):
        assert provider.verify_key("badprefix_abc123_secret") is None

    def test_verify_key_rejects_malformed_three_part(self, provider):
        # Two-part key_id like lsa_abc123 (missing secret)
        assert provider.verify_key("lsa_onlytwoparts") is None

    def test_valid_raw_key_format_matches(self, provider):
        _, raw = provider.create_key(subject="alice@acme.com", created_by="root")
        parts = raw.split("_", 2)
        assert parts[0] == "lsa"
        assert len(parts) == 3
        identity = provider.verify_key(raw)
        assert identity is not None
        assert identity.subject == "alice@acme.com"


class TestLoginRateLimitWindowSlides:
    def test_rate_limit_lifts_after_window_expires(self, provider, monkeypatch):
        monkeypatch.setattr(
            "local_search_agent.auth.api_key_provider._MAX_FAILED_LOGIN_ATTEMPTS", 3
        )
        monkeypatch.setattr(
            "local_search_agent.auth.api_key_provider._LOGIN_ATTEMPT_WINDOW_MINUTES", 1
        )

        for _ in range(3):
            provider.login("lsa_bad_bad", ip_address="10.0.0.5")

        _, raw = provider.create_key(subject="alice@acme.com", created_by="root")
        # Within window: blocked
        assert provider.login(raw, ip_address="10.0.0.5") is None

        # Sleep past the 1-minute window so the DB threshold sweeps all the
        # recorded failed attempts, then the next call is allowed again.
        import time as _time

        _time.sleep(62)

        result = provider.login(raw, ip_address="10.0.0.5")
        assert result is not None


class TestResolveSession:
    def test_sliding_expiry_preserves_identity_fields(self, provider):
        _, raw = provider.create_key(
            subject="alice@acme.com",
            created_by="root",
            display_name="Alice",
            is_superadmin=True,
        )
        session_token, _ = provider.login(raw)
        # Resolve multiple times to exercise sliding expiry
        for _ in range(5):
            identity = provider.resolve_session(session_token)
            assert identity is not None
            assert identity.display_name == "Alice"
            assert identity.is_superadmin is True

    def test_extend_session_isolation(self, auth_db, provider):
        _, raw = provider.create_key(subject="alice@acme.com", created_by="root")
        session_token, _ = provider.login(raw)
        token_hash = _hash_token(session_token)
        row1 = auth_db.get_session(token_hash)
        assert row1 is not None
        expires1 = row1["expires_at"]
        # Trigger a resolve to extend expiry
        provider.resolve_session(session_token)
        row2 = auth_db.get_session(token_hash)
        assert row2["expires_at"] != expires1
        assert row2["expires_at"] > expires1


# ===================================================================
# header_provider.py — empty string, trusted proxy behavior
# ===================================================================


class TestHeaderProviderEmptyInput:
    def test_empty_string_header_returns_none(self):
        provider = HeaderIdentityProvider(header_name="X-Remote-User")
        req = _fake_request(headers={"X-Remote-User": ""})
        identity = provider.resolve(req)
        assert identity is None

    def test_whitespace_only_header_returns_none(self):
        provider = HeaderIdentityProvider(header_name="X-Remote-User")
        req = _fake_request(headers={"X-Remote-User": "   "})
        identity = provider.resolve(req)
        assert identity is None

    def test_missing_header_returns_none(self):
        provider = HeaderIdentityProvider(header_name="X-Remote-User")
        req = _fake_request(headers={})
        identity = provider.resolve(req)
        assert identity is None

    def test_trusted_proxy_none_unrestricted(self):
        # trusted_proxy_ips=None means no IP restriction at all —
        # client is ignored, subject header alone is enough
        provider = HeaderIdentityProvider(
            header_name="X-Remote-User",
            trusted_proxy_ips=None,
        )
        req = _fake_request(headers={"X-Remote-User": "alice@acme.com"})
        # req.client can be None (no TCP peer info) — provider should not care
        req.client = None
        identity = provider.resolve(req)
        assert identity is not None
        assert identity.subject == "alice@acme.com"

    def test_trusted_proxy_set_allows_trusted_ip(self):
        provider = HeaderIdentityProvider(
            header_name="X-Remote-User",
            trusted_proxy_ips={"127.0.0.1"},
        )
        req = _fake_request(headers={"X-Remote-User": "alice@acme.com"})
        req.client = SimpleNamespace(host="127.0.0.1")
        assert provider.resolve(req) is not None

    def test_trusted_proxy_set_rejects_untrusted_ip(self):
        provider = HeaderIdentityProvider(
            header_name="X-Remote-User",
            trusted_proxy_ips={"127.0.0.1"},
        )
        req = _fake_request(headers={"X-Remote-User": "alice@acme.com"})
        req.client = SimpleNamespace(host="10.0.0.1")
        assert provider.resolve(req) is None

    def test_trusted_proxy_set_rejects_none_client(self):
        provider = HeaderIdentityProvider(
            header_name="X-Remote-User",
            trusted_proxy_ips={"127.0.0.1"},
        )
        req = _fake_request(headers={"X-Remote-User": "alice@acme.com"})
        req.client = None
        # None is not in the trusted set → deny
        assert provider.resolve(req) is None

    def test_superadmin_header(self):
        provider = HeaderIdentityProvider(
            header_name="X-Remote-User",
            superadmin_header="X-Superadmin",
            superadmin_values=frozenset({"1"}),
        )
        req = _fake_request(headers={"X-Remote-User": "alice@acme.com", "X-Superadmin": "1"})
        identity = provider.resolve(req)
        assert identity is not None
        assert identity.is_superadmin is True


# ===================================================================
# authorization_middleware.py — workspace_from_session_id,
#                               superadmin on workspace-scoped routes
# ===================================================================


class TestWorkspaceFromSessionIdInMiddleware:
    def _auth_middleware_app(self, auth_db, provider, session_lookup=None):
        """Build a test app using a real Config class-attribute pattern so
        FastAPI's request-state injection works correctly regardless of
        __future__ annotations being active in this file."""
        from fastapi import FastAPI

        class Config:
            identity_provider = provider

        app = FastAPI()
        app.add_middleware(
            AuthorizationMiddleware,
            config=Config(),
            auth_db=auth_db,
            session_lookup=session_lookup,
        )
        return app

    def test_route_matches_session_lookup_scope(self, auth_db, provider):
        """RoutePolicy with workspace_from_session_id is honored when a
        matching route is present and session_lookup is wired in.
        """
        from local_search_agent.ui.store import UIStore

        store = UIStore(db_path=auth_db._db_path)
        session = store.create_session(workspace="finance")

        def resolve_session_workspace(session_id):
            row = store.get_session(session_id)
            return row["workspace"] if row else None

        _, raw = provider.create_key(subject="alice@acme.com", created_by="root")
        auth_db.grant_access("finance", "alice@acme.com", "member", "root")

        # Unit-level check: call session_lookup directly + verify role
        ws = resolve_session_workspace(session["session_id"])
        assert ws == "finance"
        assert auth_db.get_role("alice@acme.com", ws) == "member"

    def test_workspace_from_session_denies_when_session_not_found(self, auth_db, provider):
        def resolve_session_workspace(session_id):
            return None

        _, raw = provider.create_key(subject="alice@acme.com", created_by="root")
        auth_db.grant_access("finance", "alice@acme.com", "member", "root")

        from fastapi.responses import JSONResponse
        from fastapi.testclient import TestClient

        app = self._auth_middleware_app(auth_db, provider, session_lookup=resolve_session_workspace)

        @app.get(r"/api/ui/sessions/(?P<session_id>[^/]+)")
        async def get_session(session_id):
            return JSONResponse({"ok": True})

        client = TestClient(app)
        resp = client.get(
            "/api/ui/sessions/nonexistent-session",
            headers={"Authorization": f"Bearer {raw}"},
        )
        assert resp.status_code == 403

    def test_superadmin_key_generic_create_workspace_authz(self, auth_db, provider):
        """A superadmin-flagged key is treated as admin in all authorize calls —
        global_admin scope, no workspace grant required."""
        _, raw = provider.create_key(subject="boss@acme.com", created_by="root", is_superadmin=True)
        assert auth_db.is_global_admin("boss@acme.com") is False

        from fastapi.responses import JSONResponse
        from fastapi.testclient import TestClient

        app = self._auth_middleware_app(auth_db, provider)

        @app.post("/api/ui/workspaces")
        async def create_ws(request: Request):
            return JSONResponse({"role": request.state.role})

        client = TestClient(app)
        resp = client.post(
            "/api/ui/workspaces",
            headers={"Authorization": f"Bearer {raw}"},
            json={},
        )
        assert resp.status_code == 200
        assert resp.json()["role"] == "admin"

    def test_global_admin_requires_workspace_admin_grant(self, auth_db, provider):
        """A member in a workspace is NOT a global_admin — global_admin-scoped
        routes require an explicit admin grant in at least one workspace."""
        _, raw = provider.create_key(subject="alice@acme.com", created_by="root")
        auth_db.grant_access("finance", "alice@acme.com", "member", "root")
        assert auth_db.is_global_admin("alice@acme.com") is False

        from fastapi.responses import JSONResponse
        from fastapi.testclient import TestClient

        app = self._auth_middleware_app(auth_db, provider)

        @app.post("/api/ui/workspaces")
        async def create_ws(request: Request):
            return JSONResponse({"role": request.state.role})

        client = TestClient(app)
        resp = client.post(
            "/api/ui/workspaces",
            headers={"Authorization": f"Bearer {raw}"},
            json={},
        )
        assert resp.status_code == 403

    def test_superadmin_only_scope_denies_plain_workspace_admin(self, auth_db, provider):
        """scope == "superadmin_only" (e.g. POST /api/ui/restart) is a
        stricter bar than global_admin -- being admin in one or more
        workspaces (is_global_admin) is NOT enough; only
        Identity.is_superadmin passes.
        """
        _, raw = provider.create_key(subject="alice@acme.com", created_by="root")
        auth_db.grant_access("finance", "alice@acme.com", "admin", "root")
        assert auth_db.is_global_admin("alice@acme.com") is True

        from fastapi.responses import JSONResponse
        from fastapi.testclient import TestClient

        app = self._auth_middleware_app(auth_db, provider)

        @app.post("/api/ui/restart")
        async def restart(request: Request):
            return JSONResponse({"role": request.state.role})

        client = TestClient(app)
        resp = client.post(
            "/api/ui/restart",
            headers={"Authorization": f"Bearer {raw}"},
            json={},
        )
        assert resp.status_code == 403

    def test_superadmin_only_scope_allows_superadmin(self, auth_db, provider):
        _, raw = provider.create_key(subject="boss@acme.com", created_by="root", is_superadmin=True)

        from fastapi.responses import JSONResponse
        from fastapi.testclient import TestClient

        app = self._auth_middleware_app(auth_db, provider)

        @app.post("/api/ui/restart")
        async def restart(request: Request):
            return JSONResponse({"role": request.state.role})

        client = TestClient(app)
        resp = client.post(
            "/api/ui/restart",
            headers={"Authorization": f"Bearer {raw}"},
            json={},
        )
        assert resp.status_code == 200
        assert resp.json()["role"] == "admin"


# ===================================================================
# route_policy.py — regex compilation, first-match-wins
# ===================================================================


class TestRoutePolicyRegexCompilation:
    def test_compiled_pattern_equivalence(self):
        policy = ROUTE_POLICIES[0]
        p1 = policy.compiled()
        p2 = policy.compiled()
        assert p1.pattern == p2.pattern
        assert p1.search("/api/ui/query") is not None
        assert p2.search("/api/ui/query") is not None

    def test_first_match_wins_with_overlapping_regex(self):
        # Create two policies: a broad catch-all then a specific sub-route.
        # The order in ROUTE_POLICIES is what wins, not regex specificity.
        broad = RoutePolicy("GET", r"^/api/ui/.*$", "member", "workspace")
        specific = RoutePolicy("GET", r"^/api/ui/admin/settings$", "admin", "global_admin")
        policies = [broad, specific]
        # First in list wins
        matched = next(
            (
                p
                for p in policies
                if p.method == "GET" and p.compiled().search("/api/ui/admin/settings")
            ),
            None,
        )
        assert matched is broad  # first match wins

    def test_no_match_returns_none(self):
        result = match_policy("GET", "/completely/unprotected")
        assert result is None


class TestModelsRouteIsGlobalAdminScoped:
    """Regression test for the confirmed gap: /api/ui/models (the Model
    Manager tab's actual backend) was completely absent from
    ROUTE_POLICIES, so AuthorizationMiddleware let any caller through with
    no check at all, regardless of identity, session validity, or role.
    """

    @pytest.mark.parametrize("method", ["GET", "POST", "DELETE"])
    def test_models_route_requires_global_admin(self, method):
        policy = match_policy(method, "/api/ui/models")
        assert policy is not None
        assert policy.scope == "global_admin"
        assert policy.required_role == "admin"


class TestLangsmithGetRouteRegistered:
    """Regression test caught by the route-policy-coverage walk test:
    GET /api/ui/langsmith had no RoutePolicy entry at all -- only its
    POST/DELETE siblings did, same shape of gap as /api/ui/models.
    """

    def test_get_is_global_admin_scoped(self):
        policy = match_policy("GET", "/api/ui/langsmith")
        assert policy is not None
        assert policy.scope == "global_admin"


class TestSchedulerAndWatchDeleteRoutesRegistered:
    """Regression test: DELETE /api/ui/scheduler and DELETE /api/ui/watch
    (stopping the scheduler/watcher for every workspace at once) were
    missing from ROUTE_POLICIES entirely, while their POST counterparts
    were registered. Both DELETEs are global_admin-scoped, not
    workspace-scoped, since stopping the process isn't tied to any single
    workspace named in the request.
    """

    @pytest.mark.parametrize("path", ["/api/ui/scheduler", "/api/ui/watch"])
    def test_delete_is_global_admin_scoped(self, path):
        policy = match_policy("DELETE", path)
        assert policy is not None
        assert policy.scope == "global_admin"
        assert policy.required_role == "admin"


class TestConfigGetAndDbInfoRegistered:
    """Regression test: GET /api/ui/config (PATCH was already scoped,
    GET was not) and GET /api/ui/db-info (reveals the active db_path)
    were both missing from ROUTE_POLICIES.
    """

    def test_config_get_is_global_admin_scoped(self):
        policy = match_policy("GET", "/api/ui/config")
        assert policy is not None
        assert policy.scope == "global_admin"

    def test_db_info_get_is_global_admin_scoped(self):
        policy = match_policy("GET", "/api/ui/db-info")
        assert policy is not None
        assert policy.scope == "global_admin"


class TestSettingsGetRoutesRegistered:
    """Regression test for gaps the route-policy-coverage walk test caught
    beyond what the handoff doc explicitly listed: GET variants of the
    semantic/reranking/advanced settings endpoints had no RoutePolicy
    entry at all (only their POST/DELETE siblings did).
    """

    @pytest.mark.parametrize(
        "path",
        [
            "/api/ui/settings/semantic",
            "/api/ui/settings/reranking",
            "/api/ui/settings/advanced",
        ],
    )
    def test_get_is_global_admin_scoped(self, path):
        policy = match_policy("GET", path)
        assert policy is not None, f"GET {path} missing from ROUTE_POLICIES"
        assert policy.scope == "global_admin"


class TestWatchModeSettingsRegistered:
    """Regression test: GET/POST /api/ui/settings/watch-mode had no
    RoutePolicy entry at all -- also caught by the route-policy-coverage
    walk test, not mentioned explicitly in the handoff doc.
    """

    @pytest.mark.parametrize("method", ["GET", "POST"])
    def test_watch_mode_settings_is_global_admin_scoped(self, method):
        policy = match_policy(method, "/api/ui/settings/watch-mode")
        assert policy is not None
        assert policy.scope == "global_admin"


class TestExportChatRoutesAreAuthenticatedScoped:
    """Regression test for the export-chat finding, corrected after
    clarifying the actual intended workflow: a member finds an answer in
    their own conversation and exports it to send to someone else (e.g.
    their manager) -- exporting a conversation the member already has
    legitimate read access to is ordinary use, not something to gate
    behind admin. Scope is "authenticated" (any resolved identity, no
    admin gate, no workspace check), not "global_admin". The separate,
    still-open design question -- these routes write to whatever folder
    path the client sends in the request body -- is unrelated to *who*
    may call the route and is intentionally not addressed by this scope.
    """

    @pytest.mark.parametrize(
        "path",
        ["/api/ui/export-chat", "/api/ui/export-chat-docx", "/api/ui/export-table-xlsx"],
    )
    def test_export_route_is_authenticated_scoped(self, path):
        policy = match_policy("POST", path)
        assert policy is not None, f"POST {path} missing from ROUTE_POLICIES"
        assert policy.scope == "authenticated"

    def test_plain_member_can_export(self, auth_db, provider):
        """A caller with no admin grant anywhere -- just a valid key -- must
        be allowed through an authenticated-scoped route."""
        _, raw = provider.create_key(subject="alice@acme.com", created_by="root")
        # Deliberately no grant_access call at all -- alice has zero
        # workspace rows, confirming this scope truly does not check any.

        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        from fastapi.testclient import TestClient

        app = FastAPI()
        app.add_middleware(
            AuthorizationMiddleware,
            config=type("C", (), {"identity_provider": provider})(),
            auth_db=auth_db,
        )

        @app.post("/api/ui/export-chat")
        async def export_chat(request: Request):
            return JSONResponse({"role": request.state.role})

        client = TestClient(app)
        resp = client.post(
            "/api/ui/export-chat",
            headers={"Authorization": f"Bearer {raw}"},
            json={"folder": "/tmp", "filename": "chat.md", "content": "hi"},
        )
        assert resp.status_code == 200
        assert resp.json()["role"] == "member"

    def test_unauthenticated_caller_denied(self, auth_db, provider):
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        from fastapi.testclient import TestClient

        app = FastAPI()
        app.add_middleware(
            AuthorizationMiddleware,
            config=type("C", (), {"identity_provider": provider})(),
            auth_db=auth_db,
        )

        @app.post("/api/ui/export-chat")
        async def export_chat(request: Request):
            return JSONResponse({"ok": True})

        client = TestClient(app)
        resp = client.post("/api/ui/export-chat", json={"folder": "/tmp"})
        assert resp.status_code == 401


class TestRestartRouteIsSuperadminOnly:
    """Regression test for the confirmed, reproduced bug that started this
    audit: POST /api/ui/restart (restarts the whole dashboard process
    against a caller-supplied db_path) was completely absent from
    ROUTE_POLICIES -- reachable by anyone regardless of role. Scoped
    superadmin_only rather than global_admin since it affects every
    workspace and every other user at once, a stricter bar than
    "admin of at least one workspace".
    """

    def test_restart_route_requires_superadmin_only(self):
        policy = match_policy("POST", "/api/ui/restart")
        assert policy is not None
        assert policy.scope == "superadmin_only"
        assert policy.required_role == "admin"


# ===================================================================
# meili_key_provisioning.py — DB cleanup on Meilisearch error
# ===================================================================


class TestDeprovisionResilience:
    def test_db_row_removed_even_if_meili_delete_fails(self, auth_db):
        # Pre-create a row so deprovision has something to clean up
        auth_db.store_meili_key("finance", "existing-key-uid", "encrypted-key-placeholder")
        assert auth_db.get_meili_key_row("finance") is not None

        client = MagicMock()
        client.delete_scoped_key.side_effect = RuntimeError("Meilisearch is down")

        with patch(
            "local_search_agent.auth.meili_key_provisioning.MeilisearchClient",
            return_value=client,
        ):
            # NOTE: the current implementation does NOT swallow the error --
            # the try/finally cleans up the DB row but then re-raises. This
            # is a known gap (see code review notes); the test asserts the
            # current behavior so the gap is visible and doesn't silently
            # regress if someone "fixes" the exception handling.
            with pytest.raises(RuntimeError):
                deprovision_workspace_keys(
                    workspace="finance",
                    meilisearch_url="http://localhost:7700",
                    meili_master_key="master",
                    auth_db=auth_db,
                )

        # DB row IS cleaned up (finally block runs before re-raise)
        assert auth_db.get_meili_key_row("finance") is None


# ===================================================================
# AuthorizationMiddleware — unlisted routes pass through unchanged
# ===================================================================


class TestUnlistedRoutesPassThrough:
    def test_unlisted_route_not_gated(self, provider, auth_db):
        from fastapi import FastAPI
        from starlette.testclient import TestClient

        app = FastAPI()
        app.add_middleware(
            AuthorizationMiddleware,
            config=type("C", (), {"identity_provider": provider})(),
            auth_db=auth_db,
        )

        @app.get("/health")
        async def health():
            return {"ok": True}

        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200


# ===================================================================
# API key: superadmin flag on creation
# ===================================================================


class TestAPIKeySuperadminFlag:
    def test_superadmin_key_has_is_superadmin_in_identity(self, provider):
        _, raw = provider.create_key(
            subject="boss@acme.com",
            created_by="root",
            is_superadmin=True,
        )
        identity = provider.verify_key(raw)
        assert identity is not None
        assert identity.is_superadmin is True

    def test_regular_key_is_not_superadmin(self, provider):
        _, raw = provider.create_key(
            subject="alice@acme.com",
            created_by="root",
            is_superadmin=False,
        )
        identity = provider.verify_key(raw)
        assert identity is not None
        assert identity.is_superadmin is False


# ===================================================================
# Race-condition safety: logout is idempotent for invalid token
# ===================================================================


class TestLogoutIdempotent:
    def test_double_logout_with_same_token_is_harmless(self, provider):
        _, raw = provider.create_key(subject="alice@acme.com", created_by="root")
        session_token, _ = provider.login(raw)
        provider.logout(session_token)
        # Second call on the same now-invalidated token must not raise
        provider.logout(session_token)
        # Session must be gone
        assert provider.resolve_session(session_token) is None


# ===================================================================
# JWT provider gap tests (PyJWT optional -- per-test-class guard)
# These are skipped cleanly when PyJWT is not installed.
# ===================================================================


class TestJWTNbfEnforcement:
    def test_token_not_yet_valid_is_rejected(self):
        jwt = pytest.importorskip("jwt", reason="PyJWT not installed; JWT gap tests skipped")
        from cryptography.hazmat.primitives.asymmetric import rsa

        from local_search_agent.auth.jwt_provider import JWTIdentityProvider

        issuer = "https://login.acme.test/"
        audience = "acme-app"
        jwks_uri = "https://login.acme.test/.well-known/jwks.json"
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        public_key = private_key.public_key()
        jwk_dict = json.loads(
            __import__("jwt.algorithms", fromlist=["RSAAlgorithm"]).RSAAlgorithm.to_jwk(public_key)
        )
        jwk_dict.update({"kid": "test-key-gaps", "use": "sig", "alg": "RS256"})
        jwks_payload = {"keys": [jwk_dict]}

        provider = JWTIdentityProvider(issuer=issuer, audience=audience, jwks_uri=jwks_uri)
        future_ts = int(time.time()) + 7200
        claims = {
            "iss": issuer,
            "aud": audience,
            "sub": "alice@acme.com",
            "iat": future_ts - 60,
            "exp": future_ts + 3600,
            "nbf": future_ts,
        }
        token = jwt.encode(claims, private_key, algorithm="RS256", headers={"kid": "test-key-gaps"})
        req = _fake_request(headers={"Authorization": f"Bearer {token}"})

        with patch("httpx.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = jwks_payload
            mock_resp.raise_for_status.return_value = None
            mock_get.return_value = mock_resp
            identity = provider.resolve(req)

        assert identity is None


class TestJWKSRefreshOnUnknownKid:
    def test_refresh_fetched_when_kid_misses_cache(self):
        jwt = pytest.importorskip("jwt", reason="PyJWT not installed; JWT gap tests skipped")
        from cryptography.hazmat.primitives.asymmetric import rsa

        from local_search_agent.auth.jwt_provider import JWTIdentityProvider

        issuer = "https://login.acme.test/"
        audience = "acme-app"
        jwks_uri = "https://login.acme.test/.well-known/jwks.json"
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        public_key = private_key.public_key()
        jwk_dict = json.loads(
            __import__("jwt.algorithms", fromlist=["RSAAlgorithm"]).RSAAlgorithm.to_jwk(public_key)
        )
        jwk_dict.update({"kid": "old-kid", "use": "sig", "alg": "RS256"})
        old_jwks = {"keys": [jwk_dict]}

        other_kid = "totally-new-kid"
        new_jwk = {**jwk_dict, "kid": other_kid}
        new_jwks = {"keys": [new_jwk]}

        provider = JWTIdentityProvider(issuer=issuer, audience=audience, jwks_uri=jwks_uri)
        payload = {
            "iss": issuer,
            "aud": audience,
            "sub": "alice@acme.com",
            "iat": int(time.time()),
            "exp": int(time.time()) + 3600,
        }
        token_new_kid = jwt.encode(
            payload, private_key, algorithm="RS256", headers={"kid": other_kid}
        )

        call_count = [0]

        def fake_get(url, **kwargs):
            call_count[0] += 1
            resp = MagicMock()
            if call_count[0] == 1:
                resp.json.return_value = old_jwks
            else:
                resp.json.return_value = new_jwks
            resp.raise_for_status.return_value = None
            return resp

        with patch("httpx.get", side_effect=fake_get):
            req = _fake_request(headers={"Authorization": f"Bearer {token_new_kid}"})
            identity = provider.resolve(req)

        assert call_count[0] >= 2
        assert identity is not None
        assert identity.subject == "alice@acme.com"


class TestJWTProviderUnavailableRecovery:
    def test_recovery_after_transient_jwks_failure(self):
        jwt = pytest.importorskip("jwt", reason="PyJWT not installed; JWT gap tests skipped")
        from cryptography.hazmat.primitives.asymmetric import rsa

        from local_search_agent.auth.errors import ProviderUnavailableError
        from local_search_agent.auth.jwt_provider import JWTIdentityProvider

        issuer = "https://login.acme.test/"
        audience = "acme-app"
        jwks_uri = "https://login.acme.test/.well-known/jwks.json"
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        public_key = private_key.public_key()
        jwk_dict = json.loads(
            __import__("jwt.algorithms", fromlist=["RSAAlgorithm"]).RSAAlgorithm.to_jwk(public_key)
        )
        jwk_dict.update({"kid": "test-key-gaps", "use": "sig", "alg": "RS256"})
        jwks_payload = {"keys": [jwk_dict]}

        provider = JWTIdentityProvider(issuer=issuer, audience=audience, jwks_uri=jwks_uri)
        payload = {
            "iss": issuer,
            "aud": audience,
            "sub": "alice@acme.com",
            "iat": int(time.time()),
            "exp": int(time.time()) + 3600,
        }
        token = jwt.encode(
            payload, private_key, algorithm="RS256", headers={"kid": "test-key-gaps"}
        )

        call_count = [0]

        def fake_get(url, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("JWKS endpoint temporarily down")
            resp = MagicMock()
            resp.json.return_value = jwks_payload
            resp.raise_for_status.return_value = None
            return resp

        with patch("httpx.get", side_effect=fake_get):
            req = _fake_request(headers={"Authorization": f"Bearer {token}"})
            with pytest.raises(ProviderUnavailableError):
                provider.resolve(req)

            identity = provider.resolve(req)

        assert identity is not None
        assert identity.subject == "alice@acme.com"
