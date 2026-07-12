"""
RoutePolicy: the data-driven table of which routes AuthorizationMiddleware
protects, and what each one requires.

Kept as one declarative list rather than scattered per-endpoint
decorators, per the doc's "not reimplemented per-endpoint" requirement —
adding RBAC to a new route means adding one entry here, not touching the
route handler.

Two scopes
----------
"workspace" : Requires a grant in workspace_members for the specific
              workspace resolved by resolve_workspace(). Most routes.
"global_admin" : Requires the caller to be `admin` in at least one
              workspace (or Identity.is_superadmin) — used for the
              handful of genuinely global actions the design doc calls
              out as "global + admin-only" (settings, API keys, LangSmith,
              workspace create — see the Roles table's note on why these
              can't be scoped to "admin of my workspace only" while
              app_state.config stays one shared object).
"superadmin_only" : Requires Identity.is_superadmin — stricter than
              "global_admin", reserved for the handful of actions that
              affect the entire running deployment at once (every
              workspace, every other user), where "admin of at least one
              workspace" is too broad a bar (e.g. POST /api/ui/restart,
              which restarts the whole dashboard process against a
              caller-supplied db_path).
"authenticated" : Requires only a resolved identity — no workspace
              check, no admin gate. Used where the route's payload was
              already assembled client-side from data the caller had a
              legitimate, separately-checked read on (e.g. the
              export-chat family, which just serialises a messages list
              the caller already fetched under its own workspace grant).
              A member exporting their own conversation is ordinary,
              intended use, not something to gate behind admin.

Known scope boundary (see workspace_resolution.py's docstring): most
routes whose workspace can only be found via a DB lookup on a resource id
are now covered (see the `/api/ui/sessions/{session_id}...` entries below,
resolved via `workspace_from_session_id` + `AuthorizationMiddleware`'s
`session_lookup` callback). Anything still missing from this table passes
through unauthorized-checked; that remains a known, documented gap, not a
silent one.

One deliberate exception to "not listed here = unprotected": `GET
/workspaces` (server/fastapi_app.py) and `GET /api/ui/workspaces`
(ui/api_routes.py) are both filtered by grant, but via bespoke
handler-level logic rather than a RoutePolicy entry -- a workspace
*listing* needs "which of these many workspaces does the caller have any
role in at all", a different shape of check than RoutePolicy's "does the
caller have role X in workspace Y", which doesn't fit this table's model.
Don't assume either route is unprotected just because it's absent here.

Same reasoning, same exception: `GET /api/ui/ingest/status`, `GET
/api/ui/scheduler/status`, and `GET /api/ui/watch/status`
(ui/api_routes.py) each summarise state across every registered workspace
in one response and are filtered via `_filter_workspaces_by_grant()`
rather than a RoutePolicy entry, for the same shape-of-check reason.

One more exception, different shape: `POST /api/ui/ingest` is listed
below at "admin"/"workspace" scope, which covers the ordinary
incremental-sync case -- but that single route also handles
force=True ("Force Re-ingest"), which the handler itself further
restricts to superadmin_only. RoutePolicy only matches on method+path,
not request body content, so a body-conditional restriction like this
can't be expressed as a table entry here; see trigger_ingest()'s own
docstring in api_routes.py for the actual check. `POST /api/ui/ingest/wipe`
is a separate route and IS fully superadmin_only, but only inside its own
handler (not reflected in this table either, for the same
can't-express-conditionally reason) -- don't assume either route's
severity from this table alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RoutePolicy:
    method: str  # HTTP method, e.g. "GET", "POST" — matched exactly
    path_regex: str  # regex matched against request.url.path (re.search)
    required_role: str  # "member" | "admin"
    scope: str  # "workspace" | "global_admin"
    workspace_path_group: bool = False  # True if path_regex has a (?P<workspace>...) group
    # True if path_regex has a (?P<session_id>...) group instead of a
    # workspace group -- the workspace isn't present anywhere in the
    # request itself, only resolvable via a DB lookup on the session row
    # (see AuthorizationMiddleware._resolve_workspace_from_session).
    # Mutually exclusive with workspace_path_group.
    workspace_from_session_id: bool = False

    def compiled(self) -> re.Pattern:
        return re.compile(self.path_regex)


# ---------------------------------------------------------------------------
# The policy table
# ---------------------------------------------------------------------------
# Ordered roughly by the Roles table in the design doc. Anything NOT listed
# here is NOT protected by AuthorizationMiddleware (passes straight through)
# — e.g. GET /health, GET /help/*, static assets.

ROUTE_POLICIES: list[RoutePolicy] = [
    # -- search / query (member) --------------------------------------------
    RoutePolicy("POST", r"^/api/ui/query$", "member", "workspace"),
    # -- view doc list / sessions (member) -----------------------------------
    RoutePolicy("GET", r"^/api/ui/sessions$", "member", "workspace"),
    RoutePolicy("POST", r"^/api/ui/sessions$", "member", "workspace"),
    # -- per-session routes: workspace only knowable via a DB lookup on the
    #    session row itself, not present anywhere in the request (see
    #    RoutePolicy.workspace_from_session_id / AuthorizationMiddleware's
    #    session_lookup callback). Delete is member-level at the ROUTE
    #    level (any workspace member may reach the handler), but the
    #    HANDLER itself further restricts a non-admin caller to deleting
    #    only sessions they created (created_by == subject) -- an admin
    #    may delete any session in the workspace, matching the same
    #    "admin sees/manages everyone's, member sees/manages their own"
    #    pattern GET /api/ui/sessions already uses for its own ?all=true
    #    admin-visibility toggle. RoutePolicy can't express "member, but
    #    only their own row" -- it only knows role tiers, not row
    #    ownership -- so this is another handler-level exception, same
    #    shape as the POST /api/ui/ingest force=True carve-out documented
    #    above; see delete_session()'s own docstring in api_routes.py.
    RoutePolicy(
        "GET",
        r"^/api/ui/sessions/(?P<session_id>[^/]+)$",
        "member",
        "workspace",
        workspace_from_session_id=True,
    ),
    RoutePolicy(
        "PATCH",
        r"^/api/ui/sessions/(?P<session_id>[^/]+)$",
        "member",
        "workspace",
        workspace_from_session_id=True,
    ),
    RoutePolicy(
        "DELETE",
        r"^/api/ui/sessions/(?P<session_id>[^/]+)$",
        "member",
        "workspace",
        workspace_from_session_id=True,
    ),
    RoutePolicy(
        "GET",
        r"^/api/ui/sessions/(?P<session_id>[^/]+)/messages$",
        "member",
        "workspace",
        workspace_from_session_id=True,
    ),
    RoutePolicy(
        "GET",
        r"^/api/ui/sessions/(?P<session_id>[^/]+)/tokens$",
        "member",
        "workspace",
        workspace_from_session_id=True,
    ),
    RoutePolicy(
        "GET",
        r"^/workspaces/(?P<workspace>[^/]+)/docs$",
        "member",
        "workspace",
        workspace_path_group=True,
    ),
    RoutePolicy(
        "GET",
        r"^/workspaces/(?P<workspace>[^/]+)/history$",
        "member",
        "workspace",
        workspace_path_group=True,
    ),
    RoutePolicy(
        "GET",
        r"^/api/ui/workspaces/(?P<workspace>[^/]+)/docs$",
        "member",
        "workspace",
        workspace_path_group=True,
    ),
    # -- ingest / sync workspace documents (admin) ---------------------------
    RoutePolicy("POST", r"^/api/ui/ingest$", "admin", "workspace"),
    RoutePolicy("POST", r"^/api/ui/ingest/wipe$", "admin", "workspace"),
    RoutePolicy("POST", r"^/api/ui/scheduler$", "admin", "workspace"),
    # DELETE stops the scheduler process entirely (every registered
    # workspace at once) -- unlike POST, there's no single workspace this
    # action is scoped to, so "global_admin" (not "workspace", which
    # wouldn't even have a workspace value to resolve from this request)
    # is the correct scope here.
    RoutePolicy("DELETE", r"^/api/ui/scheduler$", "admin", "global_admin"),
    RoutePolicy("POST", r"^/api/ui/watch$", "admin", "workspace"),
    # Same reasoning as DELETE /api/ui/scheduler above -- stops the watcher
    # for every registered workspace at once.
    RoutePolicy("DELETE", r"^/api/ui/watch$", "admin", "global_admin"),
    RoutePolicy(
        "POST",
        r"^/api/ui/watch/trigger/(?P<workspace>[^/]+)$",
        "admin",
        "workspace",
        workspace_path_group=True,
    ),
    # -- create / delete / wipe workspace (superadmin only) -------------------
    # Both require a document_dirs path that already exists on the SERVER's
    # own disk -- provisioning, not day-to-day workspace administration.
    # That's inherently something only whoever actually deployed/set up the
    # server can act on meaningfully (they're the one who knows a real
    # filesystem path there), not every subject a superadmin happens to
    # have granted 'admin' in one workspace. Tightened from global_admin
    # after discussing where the line should sit for an open-source
    # framework any company/team/solo user might deploy -- day-to-day
    # actions that don't need filesystem knowledge (ingest trigger,
    # watch-mode toggle, scheduler) stay at the workspace-admin tier;
    # actions that inherently require already knowing a real server-side
    # path move to superadmin_only.
    RoutePolicy("POST", r"^/api/ui/workspaces$", "admin", "superadmin_only"),
    RoutePolicy(
        "DELETE",
        r"^/api/ui/workspaces/(?P<workspace>[^/]+)$",
        "admin",
        "superadmin_only",
        workspace_path_group=True,
    ),
    # -- grant/revoke/list workspace_members access (admin, global) --------
    RoutePolicy("POST", r"^/api/admin/grants$", "admin", "global_admin"),
    RoutePolicy("DELETE", r"^/api/admin/grants$", "admin", "global_admin"),
    RoutePolicy("GET", r"^/api/admin/grants$", "admin", "global_admin"),
    RoutePolicy("POST", r"^/api/admin/keys$", "admin", "global_admin"),
    RoutePolicy("DELETE", r"^/api/admin/keys$", "admin", "global_admin"),
    RoutePolicy("GET", r"^/api/admin/keys$", "admin", "global_admin"),
    # -- provider/model, semantic/reranking/advanced settings, API keys,
    #    LangSmith — global + admin-only (see doc's note on why these can't
    #    be scoped to "admin of my workspace only") -------------------------
    RoutePolicy("PATCH", r"^/api/ui/config$", "admin", "global_admin"),
    RoutePolicy("GET", r"^/api/ui/config$", "admin", "global_admin"),
    RoutePolicy("GET", r"^/api/ui/db-info$", "admin", "global_admin"),
    # Restarts the whole dashboard process against a caller-supplied
    # db_path -- affects every workspace and every other user at once, so
    # this is superadmin_only rather than global_admin (a workspace-scoped
    # admin shouldn't be able to restart the entire deployment).
    RoutePolicy("POST", r"^/api/ui/restart$", "admin", "superadmin_only"),
    RoutePolicy("GET", r"^/api/ui/models$", "admin", "global_admin"),
    RoutePolicy("POST", r"^/api/ui/models$", "admin", "global_admin"),
    RoutePolicy("DELETE", r"^/api/ui/models$", "admin", "global_admin"),
    # Model/Provider Access Control (Option B -- per-request model
    # selection). /models/allowed resolves the caller's CURRENT role for
    # the workspace it's asked about (query param, same generic mechanism
    # as GET /api/ui/sessions) -- a subject who's admin in one workspace
    # and member in another gets a different filtered list depending on
    # which workspace this call is about, matching whichever role applies
    # to the query they're about to make. The two allow-list-management
    # routes are global (not per-workspace, mirrors /api/ui/models itself).
    RoutePolicy("GET", r"^/api/ui/models/allowed$", "member", "workspace"),
    RoutePolicy("GET", r"^/api/ui/models/access$", "admin", "global_admin"),
    RoutePolicy("POST", r"^/api/ui/models/access$", "admin", "global_admin"),
    RoutePolicy("DELETE", r"^/api/ui/models/access$", "admin", "global_admin"),
    # Concurrency & quota (RPM/TPM/RPD) config -- superadmin_only, not
    # global_admin: unlike Model Manager/Model Access (which ordinary
    # workspace admins can at least see), these numbers directly control
    # provider spend and shared-account rate-limit headroom for the whole
    # deployment -- hidden entirely from ordinary admins, not just
    # uneditable by them.
    RoutePolicy("GET", r"^/api/ui/rate-limits$", "admin", "superadmin_only"),
    RoutePolicy("POST", r"^/api/ui/rate-limits/concurrency$", "admin", "superadmin_only"),
    RoutePolicy("DELETE", r"^/api/ui/rate-limits/concurrency$", "admin", "superadmin_only"),
    RoutePolicy("POST", r"^/api/ui/rate-limits/quota$", "admin", "superadmin_only"),
    RoutePolicy("DELETE", r"^/api/ui/rate-limits/quota$", "admin", "superadmin_only"),
    RoutePolicy("GET", r"^/api/ui/keys$", "admin", "global_admin"),
    RoutePolicy("POST", r"^/api/ui/keys$", "admin", "global_admin"),
    RoutePolicy("DELETE", r"^/api/ui/keys$", "admin", "global_admin"),
    RoutePolicy("GET", r"^/api/ui/langsmith$", "admin", "global_admin"),
    RoutePolicy("POST", r"^/api/ui/langsmith$", "admin", "global_admin"),
    RoutePolicy("DELETE", r"^/api/ui/langsmith$", "admin", "global_admin"),
    RoutePolicy("GET", r"^/api/ui/settings/semantic$", "admin", "global_admin"),
    RoutePolicy("POST", r"^/api/ui/settings/semantic$", "admin", "global_admin"),
    RoutePolicy("GET", r"^/api/ui/settings/reranking$", "admin", "global_admin"),
    RoutePolicy("POST", r"^/api/ui/settings/reranking$", "admin", "global_admin"),
    RoutePolicy("GET", r"^/api/ui/settings/advanced$", "admin", "global_admin"),
    RoutePolicy("POST", r"^/api/ui/settings/advanced$", "admin", "global_admin"),
    RoutePolicy("DELETE", r"^/api/ui/settings/advanced$", "admin", "global_admin"),
    # Found during the same audit that caught /api/ui/models: GET/POST
    # /api/ui/settings/watch-mode (enable_watch_mode / enrich_on_watch
    # feature flags) had no RoutePolicy entry at all, same shape of gap.
    RoutePolicy("GET", r"^/api/ui/settings/watch-mode$", "admin", "global_admin"),
    RoutePolicy("POST", r"^/api/ui/settings/watch-mode$", "admin", "global_admin"),
    # These write a file to whatever folder path the client sends in the
    # request body -- that server-side-folder-path design question is a
    # separate, known, and deliberately open issue (not addressed by this
    # scope choice, see "authenticated" above and the current-session
    # handoff notes). Scope is "authenticated", not "global_admin": the
    # export payload is a messages list the caller already assembled
    # client-side from their own, already-granted conversation -- gating
    # export behind admin would block the exact normal-member workflow
    # (member finds an answer, exports it, sends the file to their
    # manager) the feature exists for.
    RoutePolicy("POST", r"^/api/ui/export-chat$", "member", "authenticated"),
    RoutePolicy("POST", r"^/api/ui/export-chat-docx$", "member", "authenticated"),
    RoutePolicy("POST", r"^/api/ui/export-table-xlsx$", "member", "authenticated"),
]


def match_policy(method: str, path: str) -> Optional[RoutePolicy]:
    """Return the first RoutePolicy whose method+path_regex matches, or None if unprotected."""
    for policy in ROUTE_POLICIES:
        if policy.method != method:
            continue
        if policy.compiled().search(path):
            return policy
    return None
