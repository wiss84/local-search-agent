"""
Regression test from upcoming_features/08-security-hardening-and-remaining-work.md,
section 1, "How to verify" item 3:

"a test that walks every route actually registered on the FastAPI app
(app.routes) and asserts each one is either on a small, explicit
allowlist (/health, /help/*, static assets, login/logout) or has a
matching RoutePolicy entry -- turns 'did we forget one' into something
caught automatically instead of relying on manual re-reads."

This builds the real dashboard app (ui/dashboard.py's build_dashboard_app)
in multi-tenant mode -- with a fake, lightweight AppState substitute that
provides just enough surface for route *registration* to succeed (no real
Meilisearch, file server, or pywebview involved; those are only touched
at request time by handlers this test never calls) -- then walks every
registered route+method and checks it against route_policy.py's
ROUTE_POLICIES or a small explicit allowlist of routes that are
deliberately unprotected (pre-authentication endpoints, static assets,
health/ping checks).

If a new route is ever added to the app without a corresponding
ROUTE_POLICIES entry or an allowlist addition here, this test fails --
this is exactly the class of bug that let GET/POST/DELETE /api/ui/models
and POST /api/ui/restart slip through unprotected.
"""

from __future__ import annotations

import pytest
from starlette.routing import Mount, Route

from local_search_agent.auth.api_key_provider import APIKeyIdentityProvider
from local_search_agent.auth.route_policy import match_policy
from local_search_agent.ui.dashboard import build_dashboard_app
from local_search_agent.workspace.auth_db import AuthDB

# ---------------------------------------------------------------------------
# Explicit allowlist: routes deliberately NOT covered by ROUTE_POLICIES.
# Anything else must have a matching RoutePolicy entry.
# ---------------------------------------------------------------------------

# (method, path) pairs, not path alone -- some paths (e.g. /api/ui/workspaces)
# have one allowlisted method and one RoutePolicy-checked method, so
# allowlisting by path alone would silently exempt the wrong method too.
_ALLOWLISTED_METHOD_PATHS = {
    ("GET", "/"),  # index shell -- itself gates on a valid session cookie inside the handler
    ("GET", "/login"),  # login page -- must be reachable pre-authentication
    ("GET", "/ping"),  # pywebview ready-poll, no sensitive data
    ("GET", "/api/ui/whoami"),  # identity introspection for frontend role-gating
    ("GET", "/api/ui/health"),  # read-only Meilisearch/server health poll, no sensitive data
    ("GET", "/openapi.json"),  # FastAPI's auto-generated schema -- endpoint shapes, no data
    ("POST", "/api/auth/login"),  # must be reachable pre-authentication
    ("POST", "/api/auth/logout"),  # idempotent, no sensitive data, safe pre-authentication
    # Deliberate RoutePolicy exceptions per route_policy.py's own module
    # docstring: these summarise/list across many workspaces at once ("which
    # of these does the caller have any role in"), a different shape of
    # check than RoutePolicy expresses, so they're filtered via bespoke
    # handler-level grant logic instead of a RoutePolicy entry. Note POST
    # /api/ui/workspaces (create) is deliberately NOT here -- it's a real
    # RoutePolicy entry (global_admin) and must stay checked.
    ("GET", "/api/ui/workspaces"),
    ("GET", "/api/ui/ingest/status"),
    ("GET", "/api/ui/scheduler/status"),
    ("GET", "/api/ui/watch/status"),
}

# Path prefixes that are always unprotected regardless of method (static
# assets have nothing sensitive; docs/help pages are public reference
# material).
_ALLOWLISTED_PREFIXES = (
    "/assets",
    "/help",
)


class _FakeStore:
    def get_session_workspace(self, session_id):
        return None


class _FakeConfig:
    def __init__(self, identity_provider):
        self.identity_provider = identity_provider
        self.cookie_secure = True
        self.port = 8765
        self.file_server_port = 8000


class _FakeAppState:
    """
    Minimal stand-in for ui.dashboard.AppState -- provides just the
    attributes build_dashboard_app() reads at *registration* time (not
    request time), so this test can build the real route table without
    spinning up Meilisearch, a file server, or pywebview.
    """

    def __init__(self, config, auth_db):
        self.config = config
        self.auth_db = auth_db
        self.store = _FakeStore()
        self.scheduler = None
        self.watcher = None
        self.workspace_manager = None
        self.framework = None


@pytest.fixture
def app(tmp_path):
    auth_db = AuthDB(db_path=str(tmp_path / "test.db"))
    provider = APIKeyIdentityProvider(auth_db)
    config = _FakeConfig(identity_provider=provider)
    app_state = _FakeAppState(config, auth_db)
    return build_dashboard_app(app_state)


def _iter_routes(app):
    """Yield (method, path) for every concrete Route on the app (skips Mounts,
    which are checked separately via the prefix allowlist)."""
    for route in app.routes:
        if isinstance(route, Mount):
            continue
        if not isinstance(route, Route):
            continue
        for method in route.methods or []:
            if method in ("HEAD", "OPTIONS"):
                continue
            yield method, route.path


def _is_allowlisted(method: str, path: str) -> bool:
    if (method, path) in _ALLOWLISTED_METHOD_PATHS:
        return True
    return any(path.startswith(prefix) for prefix in _ALLOWLISTED_PREFIXES)


class TestEveryRouteIsPolicyCoveredOrAllowlisted:
    def test_no_unprotected_unlisted_routes(self, app):
        uncovered = []
        for method, path in _iter_routes(app):
            if _is_allowlisted(method, path):
                continue
            if match_policy(method, path) is None:
                uncovered.append((method, path))
        assert uncovered == [], (
            f"Found route(s) with no RoutePolicy entry and not on the allowlist: {uncovered}. "
            "Add a RoutePolicy entry in route_policy.py, or if this route is genuinely meant "
            "to be unprotected, add it to _ALLOWLISTED_METHOD_PATHS/_ALLOWLISTED_PREFIXES above."
        )

    def test_mounts_are_all_allowlisted(self, app):
        """Any StaticFiles/Mount on the app must be under an allowlisted prefix."""
        for route in app.routes:
            if isinstance(route, Mount):
                assert any(route.path.startswith(prefix) for prefix in _ALLOWLISTED_PREFIXES), (
                    f"Mount at {route.path!r} is not covered by the allowlist prefixes."
                )
