"""
Browser session HTTP endpoints: POST /api/auth/login, POST /api/auth/logout.

Only meaningful for APIKeyIdentityProvider — Header/JWT identity providers never need this, since their session is
the company's own SSO (see api_key_provider.py's module docstring). This
router is only mounted when config.identity_provider is specifically an
APIKeyIdentityProvider instance (see ui/dashboard.py's build_dashboard_app).

Deliberately NOT in route_policy.py's ROUTE_POLICIES — these endpoints
must be reachable pre-authentication (that's the whole point of a login
endpoint), so AuthorizationMiddleware passes them straight through
unchecked, same as any other unprotected route.

Cookie flags
-------------
HttpOnly (unreadable to JS, mitigates XSS token theft), Secure (HTTPS
only), SameSite=Strict (mitigates CSRF).
The raw session token is set directly by set_cookie() and never
appears in the JSON response body.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from local_search_agent.auth.api_key_provider import SESSION_COOKIE_NAME, APIKeyIdentityProvider

logger = logging.getLogger(__name__)


class LoginRequest(BaseModel):
    api_key: str


def build_auth_router(
    provider: APIKeyIdentityProvider,
    *,
    cookie_secure: bool = True,
    cookie_httponly: bool = True,
    cookie_samesite: str = "strict",
) -> APIRouter:
    """Build the /api/auth router bound to a specific APIKeyIdentityProvider instance."""
    router = APIRouter(prefix="/api/auth", tags=["auth"])

    @router.post("/login")
    async def login(body: LoginRequest, request: Request):
        client_ip = request.client.host if request.client else None
        user_agent = request.headers.get("User-Agent")

        result = provider.login(raw_key=body.api_key, ip_address=client_ip, user_agent=user_agent)
        if result is None:
            # Generic message — never reveal whether the key format was
            # wrong, the key_id was unknown, or the key was revoked.
            return JSONResponse(
                {"error": "Unauthorized", "detail": "Invalid API key."}, status_code=401
            )

        raw_token, expires_at = result
        resp = JSONResponse({"ok": True})
        resp.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=raw_token,
            httponly=cookie_httponly,
            secure=cookie_secure,
            samesite=cookie_samesite,
            expires=int(expires_at.timestamp()),
        )
        return resp

    @router.post("/logout")
    async def logout(request: Request):
        token = request.cookies.get(SESSION_COOKIE_NAME)
        if token:
            provider.logout(token)
        # Always 200 + clear the cookie, even if there was no valid session
        # to begin with — logout is idempotent from the caller's perspective.
        resp = JSONResponse({"ok": True})
        resp.delete_cookie(SESSION_COOKIE_NAME)
        return resp

    return router
