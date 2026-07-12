"""
GET /api/ui/whoami — identity+role introspection for frontend role-gating.

Real enforcement is always AuthorizationMiddleware server-side — this endpoint exists purely so the
frontend can avoid showing a `member` a button that would 403 on click
(see ui/templates/_script_role_gating.html for the JS side).

Deliberately NOT gated by AuthorizationMiddleware's ROUTE_POLICIES: this
endpoint must be answerable even before a workspace is chosen (the
frontend calls it once on boot, then again whenever the active workspace
changes, passing ?workspace=... at that point) — a hard workspace
requirement here would break the very page-load flow it exists to
support. It still requires a valid identity when multi-tenant mode is on;
it just doesn't require a workspace grant to answer "who are you."

Single-user desktop mode (config.identity_provider is None): responds
with {"multi_tenant": false, ...} rather than erroring — the frontend's
refreshRoleGating() checks this flag and no-ops entirely, so this route
existing at all causes zero behavior change for today's installs.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse


def build_whoami_router(config, auth_db) -> APIRouter:
    """
    Build the /api/ui/whoami route.

    Parameters
    ----------
    config  : SearchAgentConfig — read for config.identity_provider.
    auth_db : Shared AuthDB instance, reused (not reconstructed per
              request — AuthDB.__init__ touches the DB for schema init,
              which would be wasteful on every single call here).
    """
    router = APIRouter(tags=["auth"])

    @router.get("/api/ui/whoami")
    async def whoami(request: Request, workspace: Optional[str] = None):
        provider = config.identity_provider
        if provider is None:
            response = JSONResponse({"multi_tenant": False, "subject": None, "role": None})
            # Never let a WebView2/browser cache serve a stale identity
            # check across separate process runs -- a persistent disk cache
            # returning a prior session's response here would show/hide the
            # wrong UI regardless of what mode is actually running now.
            response.headers["Cache-Control"] = "no-store"
            return response

        try:
            identity = provider.resolve(request)
        except Exception:
            # Fail-closed, same posture as AuthorizationMiddleware -- a
            # provider error here must never be reported as "no role"
            # (which the frontend could misread as "role: member-ish");
            # it's reported as unauthenticated instead.
            identity = None

        if identity is None:
            response = JSONResponse(
                {"error": "Unauthorized", "detail": "Authentication required."}, status_code=401
            )
            response.headers["Cache-Control"] = "no-store"
            return response

        role = None
        if workspace:
            role = (
                "admin" if identity.is_superadmin else auth_db.get_role(identity.subject, workspace)
            )

        response = JSONResponse(
            {
                "multi_tenant": True,
                "subject": identity.subject,
                "display_name": identity.display_name,
                "is_superadmin": identity.is_superadmin,
                "workspace": workspace,
                "role": role,
            }
        )
        response.headers["Cache-Control"] = "no-store"
        return response

    return router
