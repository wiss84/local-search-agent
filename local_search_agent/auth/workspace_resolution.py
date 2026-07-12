"""
resolve_workspace: the single canonical place a workspace identifier is
extracted from a request.

"Workspace identifier must be resolved from one canonical place per request, enforced
by a single shared dependency/decorator used by every route — not
reimplemented per-endpoint." This module IS that one place —
AuthorizationMiddleware calls resolve_workspace() and nothing else along
the authorization path parses a workspace name out of a request by hand.

Precedence (first match wins)
------------------------------
1. Path parameter — e.g. /workspaces/{workspace_name}/docs,
   /api/ui/workspaces/{workspace_name}. Extracted via each RoutePolicy
   entry's compiled regex (see route_policy.py) rather than FastAPI's own
   path-param parsing, because BaseHTTPMiddleware.dispatch() runs before
   Starlette's router matches the request — path_params isn't populated
   yet at this point in the ASGI pipeline.
2. Query parameter — `?workspace=...` (e.g. GET /api/ui/sessions).
3. JSON body key `workspace` — for POST/PATCH/DELETE requests whose
   workspace lives in the payload (e.g. {"workspace": "finance", ...}).
   Middleware must read and cache the body (request.state.json_body) so
   the downstream route handler can still call request.json()/read the
   parsed Pydantic model without a second, empty read of the exhausted
   ASGI receive stream.

Known limitation (documented, not silently ignored)
------------------------------------------------------
Some endpoints only carry a *resource id* that indirectly belongs to a
workspace (e.g. DELETE /api/ui/sessions/{session_id} — the workspace is a
column on that session's row, not present anywhere in the request itself).
Resolving those requires a DB lookup ("Activity logging... wire into... grant/revoke") is where
per-route ownership lookups get threaded through, alongside activity_log
writes. For now, a RoutePolicy for such a route should be marked
`resolve_from_body=False, resolve_from_path=False, resolve_from_query=False`
sparingly and reviewed case-by-case — resolve_workspace() returning None
for a route that IS workspace-scoped means AuthorizationMiddleware denies
by default (fail-closed), not that it silently skips the check.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from starlette.requests import Request


async def resolve_workspace(
    request: "Request",
    path_pattern: Optional[re.Pattern] = None,
) -> Optional[str]:
    """
    Extract the workspace identifier for this request, or None if it
    cannot be determined from path, query, or (cached) JSON body.

    Parameters
    ----------
    request      : The incoming Starlette/FastAPI request.
    path_pattern : Compiled regex with a named group `workspace`, matched
                   against request.url.path. Supplied by the matching
                   RoutePolicy entry (see route_policy.py) — this function
                   itself has no knowledge of route shapes.

    Side effect: on a JSON body fallback, caches the parsed body on
    request.state.json_body so downstream handlers (and repeated calls to
    this function within the same request) don't re-read the exhausted
    ASGI body stream.
    """
    # 1. Path parameter
    if path_pattern is not None:
        match = path_pattern.search(request.url.path)
        if match:
            try:
                workspace = match.group("workspace")
            except IndexError:
                workspace = None
            if workspace:
                return workspace

    # 2. Query parameter
    workspace = request.query_params.get("workspace")
    if workspace:
        return workspace

    # 3. JSON body (cached on request.state so it's only read once)
    if request.method in ("POST", "PATCH", "PUT", "DELETE"):
        body = getattr(request.state, "json_body", None)
        if body is None:
            raw = await request.body()
            if raw:
                try:
                    body = json.loads(raw)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    body = {}
            else:
                body = {}
            request.state.json_body = body
        if isinstance(body, dict):
            workspace = body.get("workspace")
            if workspace:
                return workspace

    return None
