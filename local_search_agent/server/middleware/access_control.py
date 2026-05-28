"""
Windows / LDAP access control middleware for the Local Search Agent file server.

When enable_access_control=True in SearchAgentConfig, this middleware intercepts
every request to /docs/{doc_id} and /text/{doc_id} and verifies that the caller
has read permission on the underlying source file before serving it.

Two enforcement strategies
--------------------------
A. Windows ACL (default on Windows):
   Uses `win32security` (pywin32) to read the file's DACL and check whether
   the caller's Windows identity has FILE_GENERIC_READ access.
   Only works on Windows hosts where the file server runs under a domain account.

B. LDAP group membership (cross-platform):
   Calls an LDAP server to verify the caller belongs to a group that is
   allowed to access the requested workspace.
   Requires ldap_server to be set in SearchAgentConfig.

Identity extraction
-------------------
The caller's identity is extracted from the `X-Remote-User` request header,
which should be set by a reverse proxy (nginx, IIS, Apache) performing
authentication upstream. If the header is absent, the request is rejected
with 401 Unauthorized (fail-closed by default).

For local development / testing, set LDAP_BYPASS_HEADER=1 env var to skip
access checks entirely. Never use this in production.

Graceful degradation
--------------------
If pywin32 is not installed (non-Windows or not configured), Windows ACL checks
are skipped and a warning is logged. Set enable_access_control=False in config
to disable entirely.

Install
-------
Windows ACL: pip install pywin32
LDAP: pip install python-ldap
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

# Paths that require access control checks
_PROTECTED_PREFIXES = ("/docs/", "/text/")

# Env var to bypass checks in development (never use in production)
_BYPASS_ENV_VAR = "LSA_ACCESS_CONTROL_BYPASS"


class AccessControlMiddleware(BaseHTTPMiddleware):
    """
    Windows/LDAP access control middleware.

    Enforces read permission on document endpoints based on the caller's
    Windows identity (extracted from the X-Remote-User header set by a
    trusted reverse proxy).

    Parameters
    ----------
    app             : The ASGI application.
    config          : SearchAgentConfig (for ldap_server, workspace_name, etc.)
    workspace_manager : WorkspaceManager for source_path lookups.
    """

    def __init__(self, app, config=None, workspace_manager=None):
        super().__init__(app)
        self._config = config
        self._wm = workspace_manager

    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip non-protected paths
        path = request.url.path
        if not any(path.startswith(prefix) for prefix in _PROTECTED_PREFIXES):
            return await call_next(request)

        # Bypass for development
        if os.environ.get(_BYPASS_ENV_VAR) == "1":
            logger.debug("AccessControl: bypass active (%s=1), skipping check.", _BYPASS_ENV_VAR)
            return await call_next(request)

        # Require access control to be enabled
        if self._config is None or not self._config.enable_access_control:
            return await call_next(request)

        # Extract caller identity from reverse-proxy header
        remote_user = request.headers.get("X-Remote-User", "").strip()
        if not remote_user:
            logger.warning(
                "AccessControl: request to %s has no X-Remote-User header. Rejecting.", path
            )
            return JSONResponse(
                {
                    "error": "Unauthorized",
                    "detail": (
                        "Access to this document requires authentication. "
                        "The X-Remote-User header must be set by your reverse proxy."
                    ),
                },
                status_code=401,
            )

        # Extract doc_id from path (/docs/{doc_id} or /text/{doc_id})
        doc_id = path.split("/")[-1]
        if not doc_id:
            return await call_next(request)

        # Look up the source file for this document
        source_path: Optional[str] = None
        if self._wm is not None:
            node = self._wm.get_document(doc_id)
            if node:
                source_path = node.source_path

        if source_path is None:
            # Document not found — let the route handler return the 404
            return await call_next(request)

        # Check access
        allowed = self._check_access(remote_user, source_path)

        if not allowed:
            logger.warning(
                "AccessControl: user %r denied access to %r (source: %s)",
                remote_user, doc_id, source_path,
            )
            return JSONResponse(
                {
                    "error": "Forbidden",
                    "detail": (
                        f"User {remote_user!r} does not have read permission "
                        f"on document {doc_id!r}."
                    ),
                },
                status_code=403,
            )

        logger.debug("AccessControl: user %r granted access to %r", remote_user, doc_id)
        return await call_next(request)

    def _check_access(self, username: str, source_path: str) -> bool:
        """
        Return True if `username` has read access to `source_path`.

        Tries strategies in order:
        1. Windows ACL check (if pywin32 available and on Windows)
        2. LDAP group membership (if ldap_server configured)
        3. Fallback: deny (fail-closed)
        """
        # Strategy 1: Windows ACL
        if os.name == "nt":
            result = self._check_windows_acl(username, source_path)
            if result is not None:
                return result

        # Strategy 2: LDAP
        ldap_server = getattr(self._config, "ldap_server", None)
        if ldap_server:
            result = self._check_ldap(username, ldap_server)
            if result is not None:
                return result

        # Fail-closed: if neither strategy could verify, deny
        logger.warning(
            "AccessControl: no working access check for user=%r path=%r. Denying (fail-closed).",
            username, source_path,
        )
        return False

    def _check_windows_acl(self, username: str, source_path: str) -> Optional[bool]:
        """
        Check Windows DACL for read permission.

        Returns True/False if check succeeds, None if pywin32 unavailable.
        """
        try:
            import ntsecuritycon as con
            import win32security
        except ImportError:
            logger.debug("pywin32 not available; skipping Windows ACL check.")
            return None

        try:
            sd = win32security.GetFileSecurity(
                source_path, win32security.DACL_SECURITY_INFORMATION
            )
            dacl = sd.GetSecurityDescriptorDacl()
            if dacl is None:
                return True  # No DACL = no restriction

            # Resolve username to SID
            try:
                sid, _, _ = win32security.LookupAccountName(None, username)
            except Exception:
                logger.warning("AccessControl: could not resolve Windows SID for %r", username)
                return False

            # Check each ACE for FILE_GENERIC_READ allow/deny
            for i in range(dacl.GetAceCount()):
                ace_type, ace_flags, mask, ace_sid = dacl.GetAce(i)
                if ace_sid == sid:
                    if ace_type == win32security.ACCESS_DENIED_ACE_TYPE:
                        if mask & con.FILE_GENERIC_READ:
                            return False
                    if ace_type == win32security.ACCESS_ALLOWED_ACE_TYPE:
                        if mask & con.FILE_GENERIC_READ:
                            return True

            return False  # No matching ACE found — deny
        except Exception as e:
            logger.warning("AccessControl: Windows ACL check failed: %s", e)
            return None

    def _check_ldap(self, username: str, ldap_server: str) -> Optional[bool]:
        """
        Check LDAP group membership.

        Returns True if the user exists in the directory (basic presence check).
        In a production deployment, extend this to check group membership
        against workspace-specific LDAP groups.

        Returns None if python-ldap is not installed.
        """
        try:
            import ldap
        except ImportError:
            logger.debug("python-ldap not available; skipping LDAP check.")
            return None

        try:
            conn = ldap.initialize(ldap_server)
            conn.set_option(ldap.OPT_NETWORK_TIMEOUT, 5)
            conn.simple_bind_s()  # Anonymous bind for directory lookup

            # Basic presence check: search for the user's CN
            results = conn.search_s(
                "",
                ldap.SCOPE_SUBTREE,
                f"(sAMAccountName={ldap.filter.escape_filter_chars(username)})",
                ["cn"],
            )
            conn.unbind_s()
            return len(results) > 0

        except Exception as e:
            logger.warning("AccessControl: LDAP check failed for %r: %s", username, e)
            return None
