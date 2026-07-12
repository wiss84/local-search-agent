"""
AuthorizationMiddleware: enforces workspace_members grants on every
workspace-scoped request.

Sibling to AccessControlMiddleware
(server/middleware/access_control.py) — same fail-closed philosophy, same
"trust an already-authenticated identity, don't issue it" framework
posture — but a different concern: AccessControlMiddleware checks
Windows-ACL/LDAP file permissions on two doc-serving endpoints;
AuthorizationMiddleware checks workspace_members role grants on every
route listed in route_policy.py's ROUTE_POLICIES.

Opt-in, zero ceremony for existing installs
---------------------------------------------
This middleware is only ever added to the app when
`config.identity_provider is not None` (see fastapi_app.py / dashboard.py's
build_app functions) — single-user desktop installs with no identity
provider configured see no behavior change at all, same pattern as
`enable_access_control`.

Fail-closed, every branch
--------------------------
- IdentityProvider.resolve() raises (ProviderUnavailableError) → 503,
  never falls back to "unauthenticated."
- IdentityProvider.resolve() returns None (no/bad credential) → 401.
- Route is workspace-scoped but resolve_workspace() can't determine a
  workspace → 403 (can't verify, so deny — never "skip the check").
- No workspace_members row for (subject, workspace) → 403.
- Row exists but role is below what the route requires → 403.
- scope == "global_admin" and subject has no 'admin' row anywhere and
  isn't Identity.is_superadmin → 403.
- scope == "superadmin_only" and subject isn't Identity.is_superadmin
  (holding 'admin' in one or more workspaces is not enough) → 403.
- scope == "authenticated" only requires a resolved identity — no
  workspace or admin check at all, so there is no denial branch beyond
  the identity-resolution failures already listed above.

No information leakage
------------------------
Every denial in this middleware returns the same generic body regardless
of *why* — "workspace doesn't exist" and "workspace exists but you lack
access" are indistinguishable externally. The *why* only ever appears in the server log via
`_log_denial()`, and never includes the raw credential (Authorization
header is never logged).
"""

from __future__ import annotations

import logging
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from local_search_agent.auth.errors import (
    AuthError,
    IdentityResolutionError,
    ProviderUnavailableError,
)
from local_search_agent.auth.route_policy import RoutePolicy, match_policy
from local_search_agent.auth.workspace_resolution import resolve_workspace

logger = logging.getLogger(__name__)

_ROLE_RANK = {"member": 0, "admin": 1}

_DENY_FORBIDDEN_BODY = {
    "error": "Forbidden",
    "detail": "You do not have access to this resource.",
}


def _forbidden() -> JSONResponse:
    # Fresh instance per call — JSONResponse bodies shouldn't be reused across requests.
    return JSONResponse(_DENY_FORBIDDEN_BODY, status_code=403)


def _unauthorized() -> JSONResponse:
    return JSONResponse(
        {"error": "Unauthorized", "detail": "Authentication required."}, status_code=401
    )


def _unavailable() -> JSONResponse:
    return JSONResponse(
        {"error": "ServiceUnavailable", "detail": "Identity provider is currently unavailable."},
        status_code=503,
    )


class AuthorizationMiddleware(BaseHTTPMiddleware):
    """
    Parameters
    ----------
    app               : The ASGI application.
    config            : SearchAgentConfig — read for config.identity_provider.
                        If None, this middleware should not even be added
                        (see build_app functions), but dispatch() also
                        no-ops defensively if it somehow is.
    auth_db           : AuthDB instance — the single source of truth for
                        workspace_members / is_global_admin checks.
    session_lookup    : Optional callable(session_id: str) -> Optional[str],
                        used only for RoutePolicy entries with
                        workspace_from_session_id=True (e.g.
                        DELETE /api/ui/sessions/{id}) — the workspace for
                        those routes isn't present in the request itself,
                        only as a column on the session row. Wired to
                        app_state.store.get_session(...)["workspace"] by
                        dashboard.py. If None, any route needing it fails
                        closed (returns 403) rather than skipping the check.
    """

    def __init__(self, app, config=None, auth_db=None, session_lookup=None):
        super().__init__(app)
        self._config = config
        self._auth_db = auth_db
        self._session_lookup = session_lookup

    async def dispatch(self, request: Request, call_next) -> Response:
        provider = getattr(self._config, "identity_provider", None) if self._config else None
        if provider is None or self._auth_db is None:
            return await call_next(request)

        policy = match_policy(request.method, request.url.path)
        if policy is None:
            return await call_next(request)  # route isn't in ROUTE_POLICIES — unprotected

        # -- Resolve identity (fail-closed on any provider error) -----------
        try:
            identity = provider.resolve(request)
        except AuthError as e:
            self._log_denial(None, policy, request, exception_type=type(e).__name__)
            return _unavailable() if isinstance(e, ProviderUnavailableError) else _unauthorized()
        except Exception as e:
            # Anything not our own exception hierarchy is treated as a
            # provider failure, not a "no credential" case — fail closed
            # with 503 rather than silently downgrading to unauthenticated.
            self._log_denial(None, policy, request, exception_type=type(e).__name__)
            return _unavailable()

        if identity is None:
            self._log_denial(None, policy, request, exception_type=IdentityResolutionError.__name__)
            return _unauthorized()

        # -- Authorize -------------------------------------------------------
        if policy.scope == "authenticated":
            # No workspace binding and no admin gate -- just "is there a
            # valid identity at all". Used for routes like the export-chat
            # family where the payload (messages/content) was already
            # assembled client-side from data the caller already had a
            # legitimate, already-checked read on (their own session's
            # messages) -- there's nothing further to authorize here beyond
            # "someone is logged in". See route_policy.py's comment
            # alongside these entries for the separate, still-open design
            # question this scope deliberately does NOT address (the
            # server-side folder path).
            request.state.identity = identity
            request.state.workspace = None
            request.state.role = "member"
            return await call_next(request)

        if policy.scope == "superadmin_only":
            if identity.is_superadmin:
                request.state.identity = identity
                request.state.workspace = None
                request.state.role = "admin"
                return await call_next(request)
            self._log_denial(identity.subject, policy, request)
            return _forbidden()

        if policy.scope == "global_admin":
            if identity.is_superadmin or self._auth_db.is_global_admin(identity.subject):
                request.state.identity = identity
                request.state.workspace = None
                request.state.role = "admin"
                return await call_next(request)
            self._log_denial(identity.subject, policy, request)
            return _forbidden()

        # scope == "workspace"
        if policy.workspace_from_session_id:
            workspace = self._resolve_workspace_from_session(request, policy)
        else:
            path_pattern = policy.compiled() if policy.workspace_path_group else None
            workspace = await resolve_workspace(request, path_pattern=path_pattern)
        if workspace is None:
            # Can't determine which workspace this request is about —
            # deny rather than let it through unchecked.
            self._log_denial(identity.subject, policy, request)
            return _forbidden()

        if identity.is_superadmin:
            role = "admin"
        else:
            role = self._auth_db.get_role(identity.subject, workspace)

        if role is None:
            self._log_denial(identity.subject, policy, request, workspace=workspace)
            return _forbidden()

        if _ROLE_RANK.get(role, -1) < _ROLE_RANK.get(policy.required_role, 0):
            self._log_denial(identity.subject, policy, request, workspace=workspace)
            return _forbidden()

        request.state.identity = identity
        request.state.workspace = workspace
        request.state.role = role
        request.state.meili_key = self._resolve_meili_key(workspace, role)
        return await call_next(request)

    def _resolve_meili_key(self, workspace: str, role: str) -> Optional[str]:
        """
        Data-layer defense in depth: for
        member-role requests, look up and decrypt the workspace's scoped,
        search-only Meilisearch key so downstream handlers (api_routes.py's
        /query -> AppState.get_agent) use it instead of the service-level
        master key. Admin requests deliberately get None here (fall back to
        the master key) -- per the doc's stated trade-off, scoped keys only
        protect member-level access; admin already has destructive
        capability across the app layer, so this doesn't widen the blast
        radius of an admin compromise.

        Returns None (safe fallback to the master key, not a denial -- this
        is defense in depth on top of the role check that already passed,
        not a second authorization decision) if: role is admin, no scoped
        key has been provisioned for this workspace yet (e.g. a workspace
        created before this feature existed), or decryption fails (e.g.
        LSA_FERNET_KEY rotated without following the documented procedure).
        """
        if role != "member":
            return None
        row = self._auth_db.get_meili_key_row(workspace)
        if row is None:
            return None
        try:
            from local_search_agent.auth.meili_key_crypto import decrypt_meili_key

            return decrypt_meili_key(row["encrypted_key"])
        except Exception:
            logger.warning(
                "Could not decrypt scoped Meilisearch key for workspace=%r; "
                "falling back to the service-level master key.",
                workspace,
            )
            return None

    def _resolve_workspace_from_session(
        self, request: Request, policy: RoutePolicy
    ) -> Optional[str]:
        """
        For RoutePolicy entries with workspace_from_session_id=True (e.g.
        DELETE /api/ui/sessions/{id}) -- extract session_id from the path,
        then look up its owning workspace via self._session_lookup. Returns
        None (fail-closed, same as resolve_workspace()'s contract) if the
        pattern doesn't match, no session_lookup was configured, or the
        session_id doesn't resolve to a real session -- a nonexistent
        session and an existing-but-ungranted one are deliberately
        indistinguishable externally, per the "no information leakage"
        principle.
        """
        if self._session_lookup is None:
            return None
        match = policy.compiled().search(request.url.path)
        if not match:
            return None
        try:
            session_id = match.group("session_id")
        except IndexError:
            return None
        if not session_id:
            return None
        return self._session_lookup(session_id)

    def _log_denial(
        self,
        subject: Optional[str],
        policy: RoutePolicy,
        request: Request,
        workspace: Optional[str] = None,
        exception_type: Optional[str] = None,
    ) -> None:
        """
        Log denial context without secrets — never the Authorization header
        or raw credential, per the design doc's audit-logging principle.
        """
        logger.warning(
            "AuthorizationMiddleware denied: subject=%r workspace=%r action=%s %s "
            "required_role=%r scope=%r exception_type=%s",
            subject,
            workspace,
            request.method,
            request.url.path,
            policy.required_role,
            policy.scope,
            exception_type,
        )
