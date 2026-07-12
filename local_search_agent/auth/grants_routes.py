"""
Admin grants REST API: POST/DELETE/GET /api/admin/grants.

See the UI/programmatic equivalent of the CLI's grant-access/revoke-access/
list-access commands. Both call the same AuthDB methods (thin wrappers,
never duplicate logic), same pattern as every other CLI/API pair in this
codebase.

Already declared in route_policy.py's ROUTE_POLICIES
(scope="global_admin") — this router does NOT re-check authorization
itself. It trusts request.state.identity, which AuthorizationMiddleware
only ever sets after confirming the caller is a global admin (admin in at
least one workspace, or Identity.is_superadmin). This router being
reachable at all already implies that check passed.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel


class GrantRequest(BaseModel):
    subject: str
    workspaces: list[str]
    role: str


class RevokeRequest(BaseModel):
    subject: str
    workspaces: Optional[list[str]] = None


def build_grants_router(auth_db) -> APIRouter:
    """Build the /api/admin/grants router bound to a shared AuthDB instance."""
    router = APIRouter(prefix="/api/admin", tags=["admin"])

    @router.post("/grants")
    async def grant(body: GrantRequest, request: Request):
        caller = request.state.identity
        # A plain (non-superadmin) global admin may only grant the
        # `member` role -- granting `admin` itself is the actual privilege
        # -escalation lever in this system (see admin_keys_routes.py's
        # _is_elevated_subject docstring: admin status is always derived
        # live from these grants, never baked into a key), so it gets its
        # own, tighter check rather than trusting scope="global_admin"
        # alone. This is the real enforcement point for "an admin can't
        # create admin peers" -- restricting key creation alone would not
        # be enough, since an existing ordinary key automatically starts
        # acting as admin the moment its subject is granted the role here.
        if body.role == "admin" and not caller.is_superadmin:
            raise HTTPException(
                403,
                detail="Operation denied: only a superadmin can grant the admin role.",
            )
        granted_by = caller.subject
        try:
            auth_db.grant_access_bulk(
                workspaces=body.workspaces,
                subject=body.subject,
                role=body.role,
                granted_by=granted_by,
            )
        except ValueError as e:
            # e.g. invalid role — AuthDB._validate_role() raised.
            return JSONResponse({"error": "BadRequest", "detail": str(e)}, status_code=400)
        for ws in body.workspaces:
            auth_db.log_activity(
                subject=granted_by,
                action="grant_access",
                workspace=ws,
                detail=f"subject={body.subject} role={body.role}",
                ip_address=request.client.host if request.client else None,
            )
        return JSONResponse({"ok": True})

    @router.delete("/grants")
    async def revoke(body: RevokeRequest, request: Request):
        caller = request.state.identity
        # Same reasoning as grant() above: revoking someone's admin-level
        # access is itself an admin-tier action (demoting/removing an
        # admin), not something a peer admin should be able to do to
        # another admin. Only block when the subject currently holds
        # `admin` in one of the workspaces actually being touched here
        # (or in any workspace, when workspaces=None means "revoke
        # everything") -- a plain admin can still freely revoke a
        # member's access, which is the common case.
        if not caller.is_superadmin:
            existing = auth_db.list_access(subject=body.subject)
            target_workspaces = set(body.workspaces) if body.workspaces else None
            has_admin_grant = any(
                row["role"] == "admin"
                and (target_workspaces is None or row["workspace"] in target_workspaces)
                for row in existing
            )
            if has_admin_grant:
                raise HTTPException(
                    403,
                    detail="Operation denied: only a superadmin can revoke admin-level access.",
                )
        deleted = auth_db.revoke_access(subject=body.subject, workspaces=body.workspaces)
        revoked_by = caller.subject
        auth_db.log_activity(
            subject=revoked_by,
            action="revoke_access",
            workspace=None,
            detail=f"subject={body.subject} workspaces={body.workspaces} deleted={deleted}",
            ip_address=request.client.host if request.client else None,
        )
        return JSONResponse({"ok": True, "revoked": deleted})

    @router.get("/grants")
    async def list_grants(
        request: Request, subject: Optional[str] = None, workspace: Optional[str] = None
    ):
        rows = auth_db.list_access(subject=subject, workspace=workspace)
        return JSONResponse({"grants": rows})

    return router
