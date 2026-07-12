"""
Admin API-key management REST API: POST/DELETE/GET /api/admin/keys.

Mirrors the CLI's `auth create-key` / `auth revoke-key` / `auth list-keys`
commands (see cli/commands.py) — both are thin wrappers around
APIKeyIdentityProvider, never duplicate logic. Only meaningful when
config.identity_provider is an APIKeyIdentityProvider instance; the admin
panel that calls this only renders in that case in the frontend.

Protected via route_policy.py's ROUTE_POLICIES (scope="global_admin"),
same trust boundary as grants_routes.py: this router does not re-check
authorization itself, it trusts request.state.identity already having
been confirmed as a global admin by AuthorizationMiddleware.

Distinct from /api/ui/keys, which manages LLM provider API keys
(Google/OpenAI/Anthropic) — an unrelated, pre-existing concept.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel


class CreateKeyRequest(BaseModel):
    subject: str
    display_name: str = ""
    # No is_superadmin field -- deliberately not exposed via this HTTP
    # endpoint at all (not just gated). Minting a framework-level
    # superadmin key is CLI-only (`local-search auth create-key --subject
    # ... --superadmin`), which already implies direct machine/filesystem
    # access -- a meaningfully higher bar than a browser click, and not
    # something a workspace admin should be able to grant themselves or
    # anyone else via the dashboard, regardless of how the checkbox is
    # labeled or gated.


class RevokeKeyRequest(BaseModel):
    key_id: str


def _is_elevated_subject(auth_db, subject: str) -> bool:
    """
    True if `subject` currently holds `admin` in any workspace, or holds
    any active (non-revoked) superadmin key.

    Used to gate key create/revoke for callers who are global-admins but
    not superadmins: they may only touch keys for subjects who are, and
    will remain, ordinary members everywhere. This is deliberately checked
    dynamically rather than trusting anything stored on a key itself --
    admin status is always derived live from workspace_members grants,
    never baked into a key (the only thing baked into a key is
    is_superadmin), so this must re-check current grants/keys on every
    call rather than caching the answer.
    """
    if any(row["role"] == "admin" for row in auth_db.list_access(subject=subject)):
        return True
    if any(
        row.get("is_superadmin") and not row.get("revoked_at")
        for row in auth_db.list_api_keys(subject=subject)
    ):
        return True
    return False


def build_admin_keys_router(provider) -> APIRouter:
    """Build the /api/admin/keys router bound to a specific APIKeyIdentityProvider instance."""
    router = APIRouter(prefix="/api/admin", tags=["admin"])

    @router.post("/keys")
    async def create_key(body: CreateKeyRequest, request: Request):
        caller = request.state.identity
        # route_policy.py's scope="global_admin" only requires the caller
        # be an admin of *some* workspace (or a superadmin) -- see
        # AuthDB.is_global_admin()'s own docstring for why that's the
        # existing, deliberate trust boundary for RBAC administration in
        # general. A plain (non-superadmin) admin may still only create
        # keys for subjects who are, and will remain, ordinary members --
        # not for anyone who already holds admin or superadmin anywhere,
        # which would let admins mint credentials for their peers.
        if not caller.is_superadmin and _is_elevated_subject(provider._auth_db, body.subject):
            raise HTTPException(
                403,
                detail=(
                    "Operation denied: only a superadmin can create or revoke a key "
                    "for a subject who holds admin access anywhere, or already holds "
                    "a superadmin key."
                ),
            )
        created_by = caller.subject
        key_id, raw_key = provider.create_key(
            subject=body.subject,
            created_by=created_by,
            display_name=body.display_name,
            is_superadmin=False,
        )
        # Raw key returned exactly once, same as the CLI's `auth create-key`
        # — the frontend must display it once and never persist it.
        return JSONResponse({"key_id": key_id, "raw_key": raw_key})

    @router.delete("/keys")
    async def revoke_key(body: RevokeKeyRequest, request: Request):
        caller = request.state.identity
        key_row = provider._auth_db.get_api_key(body.key_id)
        if key_row is not None and not caller.is_superadmin:
            if _is_elevated_subject(provider._auth_db, key_row["subject"]):
                raise HTTPException(
                    403,
                    detail=(
                        "Operation denied: only a superadmin can revoke a key belonging "
                        "to anyone who holds admin access anywhere, or a superadmin key."
                    ),
                )
        revoked = provider.revoke_key(body.key_id)
        return JSONResponse({"ok": True, "revoked": revoked})

    @router.get("/keys")
    async def list_keys(subject: Optional[str] = None):
        rows = provider.list_keys(subject=subject)
        return JSONResponse({"keys": rows})

    return router
