"""
FastAPI routes for the Desktop Dashboard UI.

All routes are mounted under /api/ui on the existing FastAPI app, or on a
dedicated app instance when the dashboard runs standalone.

SSE stream contract (POST /api/ui/query):
------------------------------------------
Each event has a named type and a JSON data payload:

    event: thinking
    data: {"text": "..."}

    event: queued
    data: {"waiting_ahead": 2}
    (emitted when an LLM call inside this query has to wait for a free
    concurrency slot -- see agent/rate_limit_handler.py's ConcurrencyGate.
    Purely informational; may appear zero or more times per query.)

    event: tool_start
    data: {"tool": "search_local_index", "input": {...}, "call_id": "abc"}

    event: tool_end
    data: {"tool": "search_local_index", "output": "...", "duration_ms": 320, "call_id": "abc"}

    event: text_chunk
    data: {"text": "..."}

    event: done
    data: {"token_query": 1240, "token_reply": 890, "message_id": "..."}

    event: error
    data: {"message": "Rate limit exceeded — retry after 60 s"}

The frontend assembles text_chunk events into the assistant message bubble and
feeds tool_start/tool_end events into the live tool drawer.  On done it writes
the completed message to chat_messages via add_message().
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Live ingest progress — shared state written by the background thread,
# read by the /ingest/status endpoint. One slot per workspace.
# ---------------------------------------------------------------------------


@dataclass
class _IngestProgress:
    workspace: str
    status: str = "idle"  # "running" | "done" | "error"
    files_total: int = 0
    files_processed: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    chunks_indexed: int = 0
    current_file: str = ""
    error: str = ""
    failed_files: list = field(default_factory=list)
    started_at: float = field(default_factory=time.monotonic)
    finished_at: float = 0.0

    @property
    def elapsed_s(self) -> float:
        end = self.finished_at or time.monotonic()
        return round(end - self.started_at, 1)

    @staticmethod
    def _fmt_elapsed(seconds: float) -> str:
        """Format elapsed seconds as human-readable string."""
        s = int(seconds)
        if s < 60:
            return f"{s}s"
        elif s < 3600:
            m, sec = divmod(s, 60)
            return f"{m}m {sec}s"
        elif s < 86400:
            h, rem = divmod(s, 3600)
            m, sec = divmod(rem, 60)
            return f"{h}h {m}m {sec}s"
        else:
            d, rem = divmod(s, 86400)
            h, rem2 = divmod(rem, 3600)
            m, sec = divmod(rem2, 60)
            return f"{d}d {h}h {m}m {sec}s"

    def to_dict(self) -> dict:
        return {
            "workspace": self.workspace,
            "status": self.status,
            "total": self.files_total,
            "processed": self.files_processed,
            "skipped": self.files_skipped,
            "failed": self.files_failed,
            "indexed": self.chunks_indexed,
            "current_file": self.current_file,
            "error": self.error,
            "elapsed_s": self.elapsed_s,
            "elapsed_fmt": self._fmt_elapsed(self.elapsed_s),
            "failed_files": self.failed_files,
            "doc_count": self.chunks_indexed,
        }


# Module-level registry: workspace → _IngestProgress
# Protected by a lock because the background thread writes and the
# async route handler reads from a different thread.
_ingest_lock = threading.Lock()
_ingest_registry: dict[str, _IngestProgress] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise_dirs(ws: dict) -> list[str]:
    """
    WorkspaceManager stores document_dir as a single TEXT column.
    Return a clean list of non-empty directory paths regardless of which
    key is present (document_dirs list vs document_dir string).
    """
    dirs = ws.get("document_dirs") or [ws.get("document_dir", "")]
    return [d for d in dirs if d]


def _filter_workspaces_by_grant(app_state, request: Request):
    """
    Resolve which workspace names the caller is allowed to see, for routes
    that return a summary across *every* registered workspace at once
    (ingest/status, scheduler/status, watch/status).

    These routes have the same "listing across many workspaces" shape
    problem noted on GET /api/ui/workspaces's own docstring: RoutePolicy
    only expresses "does the caller have role X in workspace Y", not
    "which of these many workspaces does the caller have any role in at
    all" -- so these routes are deliberately NOT route_policy.py entries,
    and identity has to be resolved here directly instead of read off
    request.state (AuthorizationMiddleware never touches them).

    Returns
    -------
    None
        if single-user mode (no identity_provider configured) or the
        caller is a superadmin -- meaning "don't filter, return everything".
    set[str]
        the workspace names the caller actually holds a grant in,
        otherwise (multi-tenant mode, non-superadmin caller).

    Raises HTTPException(401) if multi-tenant mode is on and no identity
    resolves for this request -- these routes leak in-progress file paths
    and directory names across every workspace, so unauthenticated access
    is denied outright rather than merely unfiltered.
    """
    identity_provider = getattr(app_state.config, "identity_provider", None)
    if identity_provider is None:
        return None
    try:
        identity = identity_provider.resolve(request)
    except Exception:
        identity = None
    if identity is None:
        raise HTTPException(401, detail="Authentication required.")
    if identity.is_superadmin:
        return None
    return {row["workspace"] for row in app_state.auth_db.list_access(subject=identity.subject)}


def _log_activity(
    app_state,
    request: Request,
    action: str,
    workspace: Optional[str] = None,
    detail: Optional[str] = None,
    success: bool = True,
) -> None:
    """
    Write one activity_log row via app_state.auth_db, per
    upcoming_features/04-multi-tenant-rbac-mode.md's "Activity logging"
    section (Phase 6).

    No-ops silently (does not raise) when:
      - request.state.identity isn't set -- either single-user mode (no
        identity_provider configured) or the route isn't in
        route_policy.py's ROUTE_POLICIES, so AuthorizationMiddleware never
        resolved an identity for this request. Multi-tenant activity
        logging only makes sense once there's a subject to attribute the
        row to.
      - the underlying auth_db write itself fails, per the design doc:
        "losing an audit row is preferable to failing the underlying
        action" -- logged as a warning, never re-raised into the caller's
        request path.
    """
    identity = getattr(request.state, "identity", None)
    if identity is None:
        return
    ip_address = request.client.host if request.client else None
    try:
        app_state.auth_db.log_activity(
            subject=identity.subject,
            action=action,
            workspace=workspace,
            detail=detail,
            ip_address=ip_address,
            success=success,
        )
    except Exception:
        logger.warning("Failed to write activity_log row for action=%r", action, exc_info=True)


# ---------------------------------------------------------------------------
# Pydantic request bodies
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    session_id: str
    question: str
    workspace: Optional[str] = None
    # Model/Provider Access Control (Option B): per-request model
    # selection. None means "use the shared deployment-wide default"
    # (app_state.config.provider / model_name), exactly today's behavior.
    # Whichever provider/model is actually used -- explicit here or the
    # shared default -- is checked against the caller's current role's
    # allow-list before the query runs; see the /query handler below.
    provider: Optional[str] = None
    model_name: Optional[str] = None


class NewSessionRequest(BaseModel):
    workspace: str
    title: Optional[str] = ""


class RenameSessionRequest(BaseModel):
    title: str


class ConfigPatchRequest(BaseModel):
    key: str
    value: object


class SchedulerSetRequest(BaseModel):
    workspace: str
    interval_minutes: int


class WatchSetRequest(BaseModel):
    workspace: str


class WatchModeSettingsRequest(BaseModel):
    enable_watch_mode: bool
    enrich_on_watch: bool


class ModelAddRequest(BaseModel):
    provider: str
    model_name: str


class ModelDeleteRequest(BaseModel):
    provider: str
    model_name: str


class ModelAccessRequest(BaseModel):
    role: str
    provider: str
    model_name: str


class ConcurrencyLimitRequest(BaseModel):
    provider: str
    limit: int


class ConcurrencyDeleteRequest(BaseModel):
    provider: str


class QuotaOverrideRequest(BaseModel):
    provider: str
    model_name: str
    rpm: Optional[int] = None
    tpm: Optional[int] = None
    rpd: Optional[int] = None


class QuotaOverrideDeleteRequest(BaseModel):
    provider: str
    model_name: str


class SemanticSettingsRequest(BaseModel):
    enable_semantic: bool
    enable_query_expansion: bool
    semantic_provider: str = ""
    semantic_model: str = ""


class RerankSettingsRequest(BaseModel):
    enable_reranking: bool
    rerank_candidate_multiplier: int = 4


class LangSmithRequest(BaseModel):
    api_key: str
    project: str = "local-search-agent"


class ApiKeyRequest(BaseModel):
    provider: str
    key: str


class ApiKeyDeleteRequest(BaseModel):
    provider: str


class IngestRequest(BaseModel):
    workspace: str
    force: bool = False


class AdvancedSettingsRequest(BaseModel):
    overrides: dict  # keys from _ADVANCED_SETTING_KEYS, values are ints/floats or None


class ExportDocxRequest(BaseModel):
    # None (the default) means the caller has no folder to write to --
    # i.e. a plain browser tab with no pywebview bridge to pick one,
    # whether that browser happens to be running via RDP on this same
    # machine or on a genuinely separate one (the two are indistinguishable
    # from here -- see the handler's own docstring). folder is only ever
    # set by the desktop app's pick_folder() flow.
    folder: Optional[str] = None
    filename: str = "chat.docx"
    messages: list[dict]


class ExportTableXlsxRequest(BaseModel):
    folder: Optional[str] = None
    filename: str = "table.xlsx"
    headers: list[str]
    rows: list[list] = []


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def build_ui_router(app_state) -> APIRouter:
    """
    Build and return the /api/ui APIRouter.

    Parameters
    ----------
    app_state : object with attributes:
        .config            SearchAgentConfig
        .store             UIStore
        .workspace_manager WorkspaceManager
        .framework         SearchAgentFramework
        .scheduler         IncrementalSyncScheduler (may be None)
        .auth_db           AuthDB (used by _log_activity, the grant-filtered
                           listing routes, and Model/Provider Access
                           Control's model_access_by_role checks)
    """
    router = APIRouter(prefix="/api/ui", tags=["ui"])

    # ----------------------------------------------------------------
    # Sessions
    # ----------------------------------------------------------------

    @router.post("/sessions")
    async def create_session(body: NewSessionRequest, request: Request) -> JSONResponse:
        identity = getattr(request.state, "identity", None)
        session = app_state.store.create_session(
            workspace=body.workspace,
            title=body.title or "",
            created_by=identity.subject if identity else None,
        )
        return JSONResponse(session)

    @router.get("/sessions")
    async def list_sessions(
        workspace: str, request: Request, limit: int = 50, all: bool = False
    ) -> JSONResponse:
        sessions = app_state.store.list_sessions(workspace=workspace, limit=limit)
        identity = getattr(request.state, "identity", None)
        role = getattr(request.state, "role", None)
        # Ownership filter: a member sees only sessions they created (plus
        # any pre-migration rows with created_by=NULL, since ownership is
        # simply unknown for those -- fail open on old data rather than
        # hiding it unexpectedly). Single-user mode (identity is None) is
        # unaffected -- no filtering at all. An admin can pass ?all=true to
        # see every session in the workspace, matching the Roles table's
        # "delete conversations: admin" scope (admins need visibility to
        # manage other members' sessions, not just their own).
        if identity is not None and not (all and role == "admin"):
            sessions = [s for s in sessions if s.get("created_by") in (None, identity.subject)]
        return JSONResponse({"sessions": sessions})

    @router.get("/sessions/{session_id}")
    async def get_session(session_id: str) -> JSONResponse:
        session = app_state.store.get_session(session_id)
        if not session:
            raise HTTPException(404, detail=f"Session {session_id!r} not found.")
        return JSONResponse(session)

    @router.patch("/sessions/{session_id}")
    async def rename_session(session_id: str, body: RenameSessionRequest) -> JSONResponse:
        session = app_state.store.get_session(session_id)
        if not session:
            raise HTTPException(404, detail=f"Session {session_id!r} not found.")
        app_state.store.rename_session(session_id, body.title)
        return JSONResponse({"ok": True})

    @router.delete("/sessions/{session_id}")
    async def delete_session(session_id: str, request: Request) -> JSONResponse:
        """
        Delete a conversation. A member may delete only sessions they
        created; an admin may delete any session in the workspace (see
        route_policy.py's own comment on why this ownership check lives
        here rather than as a RoutePolicy entry -- RoutePolicy only
        expresses role tiers, not per-row ownership). Single-user mode (no
        identity_provider, identity is None) is unrestricted, as always.
        """
        identity = getattr(request.state, "identity", None)
        role = getattr(request.state, "role", None)
        if identity is not None and role != "admin":
            session = app_state.store.get_session(session_id)
            # A pre-migration row with created_by=NULL has unknown
            # ownership -- fail open (same policy GET /sessions already
            # uses for its own ownership filter) rather than blocking
            # deletion of old data nobody can prove they own.
            if session and session.get("created_by") not in (None, identity.subject):
                raise HTTPException(403, detail="You can only delete conversations you created.")
        # Logged before the delete runs, per the design doc's audit-logging
        # principle -- a crash mid-delete should still leave a record of
        # intent. request.state.workspace was already resolved by
        # AuthorizationMiddleware's session_lookup for this route (see
        # RoutePolicy.workspace_from_session_id), so no extra DB lookup here.
        _log_activity(
            app_state,
            request,
            "delete_conversation",
            workspace=getattr(request.state, "workspace", None),
            detail=f"session_id={session_id}",
        )
        app_state.store.delete_session(session_id)
        return JSONResponse({"ok": True})

    @router.get("/sessions/{session_id}/messages")
    async def get_messages(session_id: str) -> JSONResponse:
        session = app_state.store.get_session(session_id)
        if not session:
            raise HTTPException(404, detail=f"Session {session_id!r} not found.")
        messages = app_state.store.list_messages(session_id)
        return JSONResponse({"messages": messages})

    @router.get("/sessions/{session_id}/tokens")
    async def get_session_tokens(session_id: str) -> JSONResponse:
        return JSONResponse(app_state.store.session_token_totals(session_id))

    # ----------------------------------------------------------------
    # Query — SSE stream
    # ----------------------------------------------------------------

    @router.post("/query")
    async def query(body: QueryRequest, request: Request) -> StreamingResponse:
        """
        Stream the agent response as SSE events.

        The agent itself is synchronous (LangGraph + LangChain).  We run it in
        a thread pool executor so the event loop stays unblocked.  Tool call
        events are emitted by temporarily wrapping the agent's tool map.
        """
        session = app_state.store.get_session(body.session_id)
        if not session:
            raise HTTPException(404, detail=f"Session {body.session_id!r} not found.")

        workspace = body.workspace or session["workspace"]

        # Model/Provider Access Control (Option B): whichever provider/model
        # will actually be used for this query -- the request's own
        # override if given, otherwise the shared deployment-wide default
        # -- must be on the caller's CURRENT role's allow-list. Superadmin
        # and single-user mode (no identity_provider at all) both bypass
        # this entirely, matching every other RBAC check in this codebase.
        # request.state.role is whichever role AuthorizationMiddleware
        # already resolved for THIS workspace (this route is
        # workspace-scoped in route_policy.py), so a subject who's admin
        # in one workspace and only member in another is checked against
        # the correct tier for the workspace they're actually querying.
        effective_provider = body.provider or app_state.config.provider
        effective_model = body.model_name or app_state.config.model_name

        identity = getattr(request.state, "identity", None)
        if identity is not None and not identity.is_superadmin:
            role = getattr(request.state, "role", None)
            if not app_state.auth_db.is_model_allowed(role, effective_provider, effective_model):
                raise HTTPException(
                    403,
                    detail=(
                        f"The model {effective_provider}/{effective_model} is not allowed "
                        f"for your role."
                    ),
                )

        _log_activity(app_state, request, "search", workspace=workspace, detail=body.question)

        meili_api_key = getattr(request.state, "meili_key", None)

        return StreamingResponse(
            _run_agent_streaming(
                app_state=app_state,
                session_id=body.session_id,
                question=body.question,
                workspace=workspace,
                request=request,
                meili_api_key=meili_api_key,
                provider=body.provider,
                model_name=body.model_name,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # ----------------------------------------------------------------
    # Config (provider, model, theme, etc.)
    # ----------------------------------------------------------------

    @router.get("/config")
    async def get_config() -> JSONResponse:
        return JSONResponse(app_state.store.get_all_config())

    @router.patch("/config")
    async def set_config(body: ConfigPatchRequest) -> JSONResponse:
        app_state.store.set_config(body.key, body.value)
        if body.key == "global.provider":
            app_state.config.provider = body.value
            app_state.config.api_key = None
            app_state.config.__post_init__()
            app_state.invalidate_agents()
        elif body.key == "global.model":
            app_state.config.model_name = body.value
            app_state.invalidate_agents()
        return JSONResponse({"ok": True})

    # ----------------------------------------------------------------
    # API Keys
    # ----------------------------------------------------------------

    @router.get("/keys")
    async def get_keys() -> JSONResponse:
        """Return all saved API keys, masked."""
        from local_search_agent.core.key_manager import list_keys

        return JSONResponse({"keys": list_keys()})

    @router.post("/keys")
    async def set_key(body: ApiKeyRequest) -> JSONResponse:
        """Save an API key for a provider."""
        from local_search_agent.core.key_manager import set_key

        try:
            set_key(body.provider, body.key)
            # Hot-reload: if active provider matches, refresh the config's api_key
            if app_state.config.provider == body.provider:
                app_state.config.api_key = body.key
                app_state.invalidate_agents()
            return JSONResponse({"ok": True})
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.delete("/keys")
    async def delete_key(body: ApiKeyDeleteRequest) -> JSONResponse:
        """Remove a saved API key for a provider."""
        from local_search_agent.core.key_manager import delete_key

        deleted = delete_key(body.provider)
        if app_state.config.provider == body.provider:
            app_state.config.api_key = None
            app_state.invalidate_agents()
        return JSONResponse({"ok": True, "deleted": deleted})

    # ----------------------------------------------------------------
    # LangSmith
    # ----------------------------------------------------------------

    @router.get("/langsmith")
    async def get_langsmith() -> JSONResponse:
        """Return current LangSmith config (api_key masked)."""
        from local_search_agent.core.key_manager import get_langsmith

        return JSONResponse(get_langsmith())

    @router.post("/langsmith")
    async def set_langsmith(body: LangSmithRequest) -> JSONResponse:
        """Save LangSmith credentials and activate tracing immediately."""
        from local_search_agent.core.key_manager import set_langsmith

        try:
            set_langsmith(api_key=body.api_key, project=body.project)
            return JSONResponse({"ok": True})
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.delete("/langsmith")
    async def delete_langsmith() -> JSONResponse:
        """Remove LangSmith credentials and deactivate tracing."""
        from local_search_agent.core.key_manager import delete_langsmith

        deleted = delete_langsmith()
        return JSONResponse({"ok": True, "deleted": deleted})

    # ----------------------------------------------------------------
    # Models
    # ----------------------------------------------------------------

    @router.get("/models")
    async def get_models() -> JSONResponse:
        """Return all stored models grouped by provider."""
        from local_search_agent.core.key_manager import get_models

        return JSONResponse({"models": get_models()})

    @router.post("/models")
    async def add_model(body: ModelAddRequest) -> JSONResponse:
        """Add a model name for a provider."""
        from local_search_agent.core.key_manager import add_model

        try:
            add_model(body.provider, body.model_name)
            from local_search_agent.core.key_manager import get_models

            return JSONResponse({"ok": True, "models": get_models()})
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.delete("/models")
    async def delete_model(body: ModelDeleteRequest) -> JSONResponse:
        """Remove a model name for a provider."""
        from local_search_agent.core.key_manager import delete_model, get_models

        deleted = delete_model(body.provider, body.model_name)
        return JSONResponse({"ok": True, "deleted": deleted, "models": get_models()})

    # ----------------------------------------------------------------
    # Model / Provider Access Control (Option B: true per-request model
    # selection -- see the /query handler above for the actual
    # enforcement point; these two routes are the allow-list read/write
    # surface the frontend uses to build its model picker and the admin
    # UI panel that manages the two role-level allow-lists).
    # ----------------------------------------------------------------

    @router.get("/models/allowed")
    async def get_allowed_models(request: Request) -> JSONResponse:
        """
        Return the provider/model options the caller's CURRENT role (for
        the workspace given via ?workspace=) may use, intersected with
        what's actually configured in models.json -- so a stale allow-list
        row for a since-deleted model never appears as a phantom option.
        This is a UX filter on top of the real enforcement in POST /query,
        not a substitute for it (that check re-verifies server-side
        regardless of what this endpoint returned).

        Single-user mode (no identity_provider) and superadmin callers get
        every configured provider/model, unfiltered -- this feature is
        multi-tenant-only, and superadmin always bypasses every grant in
        this system by design.
        """
        from local_search_agent.core.key_manager import get_models

        all_models = get_models()  # {provider: [model_name, ...]}

        identity_provider = getattr(app_state.config, "identity_provider", None)
        if identity_provider is None:
            return JSONResponse({"models": all_models})

        identity = getattr(request.state, "identity", None)
        if identity is None or identity.is_superadmin:
            return JSONResponse({"models": all_models})

        role = getattr(request.state, "role", "member")
        allowed = app_state.auth_db.role_allowed_models(role)
        filtered = {
            provider: [m for m in models if m in allowed.get(provider, [])]
            for provider, models in all_models.items()
        }
        return JSONResponse({"models": filtered})

    @router.get("/models/access")
    async def get_model_access() -> JSONResponse:
        """Return the full model_access_by_role allow-lists, grouped by
        role, for the admin UI panel that manages them."""
        return JSONResponse(
            {
                "member": app_state.auth_db.role_allowed_models("member"),
                "admin": app_state.auth_db.role_allowed_models("admin"),
            }
        )

    @router.post("/models/access")
    async def grant_model_access_route(body: ModelAccessRequest, request: Request) -> JSONResponse:
        """Add one (role, provider, model_name) row to the allow-list."""
        identity = getattr(request.state, "identity", None)
        granted_by = identity.subject if identity else "system"
        try:
            app_state.auth_db.grant_model_access(
                body.role, body.provider, body.model_name, granted_by
            )
        except ValueError as e:
            raise HTTPException(400, detail=str(e))
        return JSONResponse({"ok": True})

    @router.delete("/models/access")
    async def revoke_model_access_route(body: ModelAccessRequest) -> JSONResponse:
        """Remove one (role, provider, model_name) row from the allow-list."""
        revoked = app_state.auth_db.revoke_model_access(body.role, body.provider, body.model_name)
        return JSONResponse({"ok": True, "revoked": revoked})

    # ----------------------------------------------------------------
    # Rate limits & concurrency (07-concurrency-and-model-serving) --
    # fully admin-configurable so a company on paid-tier accounts with
    # much higher real limits than the free-tier defaults can set their
    # own numbers, and so Ollama's concurrency can be tuned to whatever
    # the admin's own hardware actually supports (this framework has no
    # way to introspect VRAM itself). Every route here is superadmin_only
    # by design (route_policy.py) -- unlike Model Manager/Model Access,
    # which ordinary admins can at least see, these numbers are hidden
    # from ordinary admins entirely, not just uneditable.
    # ----------------------------------------------------------------

    @router.get("/rate-limits")
    async def get_rate_limits() -> JSONResponse:
        """Return the full configured concurrency limits and quota
        overrides FOR THIS DEPLOYMENT's OWN MODE, for the superadmin-only
        settings panel. Single-user and multi-tenant settings are
        independent namespaces (see key_manager.py) -- this always
        reflects whichever mode this running process is actually in.
        """
        from local_search_agent.core.key_manager import (
            get_concurrency_limits,
            get_quota_overrides,
        )

        multi_tenant = app_state.config.identity_provider is not None
        return JSONResponse(
            {
                "concurrency": get_concurrency_limits(multi_tenant),
                "quota_overrides": get_quota_overrides(multi_tenant),
            }
        )

    @router.post("/rate-limits/concurrency")
    async def set_concurrency(body: ConcurrencyLimitRequest) -> JSONResponse:
        """Set the max simultaneous in-flight LLM calls for a provider,
        in THIS deployment's own mode's namespace."""
        from local_search_agent.agent.rate_limit_handler import (
            reset_shared_rate_limit_handlers,
        )
        from local_search_agent.core.key_manager import set_concurrency_limit

        multi_tenant = app_state.config.identity_provider is not None
        try:
            set_concurrency_limit(body.provider, body.limit, multi_tenant)
        except ValueError as e:
            raise HTTPException(400, detail=str(e))
        # Take effect immediately rather than only after a restart -- see
        # reset_shared_rate_limit_handlers()'s own docstring.
        reset_shared_rate_limit_handlers()
        return JSONResponse({"ok": True})

    @router.delete("/rate-limits/concurrency")
    async def delete_concurrency(body: ConcurrencyDeleteRequest) -> JSONResponse:
        """Remove a provider's concurrency cap (reverts to unbounded), in
        THIS deployment's own mode's namespace."""
        from local_search_agent.agent.rate_limit_handler import (
            reset_shared_rate_limit_handlers,
        )
        from local_search_agent.core.key_manager import delete_concurrency_limit

        multi_tenant = app_state.config.identity_provider is not None
        deleted = delete_concurrency_limit(body.provider, multi_tenant)
        reset_shared_rate_limit_handlers()
        return JSONResponse({"ok": True, "deleted": deleted})

    @router.post("/rate-limits/quota")
    async def set_quota(body: QuotaOverrideRequest) -> JSONResponse:
        """Set (or replace) the RPM/TPM/RPD override for one provider+model,
        in THIS deployment's own mode's namespace."""
        from local_search_agent.agent.rate_limit_handler import (
            reset_shared_rate_limit_handlers,
        )
        from local_search_agent.core.key_manager import set_quota_override

        multi_tenant = app_state.config.identity_provider is not None
        try:
            set_quota_override(
                body.provider,
                body.model_name,
                multi_tenant,
                rpm=body.rpm,
                tpm=body.tpm,
                rpd=body.rpd,
            )
        except ValueError as e:
            raise HTTPException(400, detail=str(e))
        reset_shared_rate_limit_handlers()
        return JSONResponse({"ok": True})

    @router.delete("/rate-limits/quota")
    async def delete_quota(body: QuotaOverrideDeleteRequest) -> JSONResponse:
        """Remove a provider+model's RPM/TPM/RPD override, in THIS
        deployment's own mode's namespace."""
        from local_search_agent.agent.rate_limit_handler import (
            reset_shared_rate_limit_handlers,
        )
        from local_search_agent.core.key_manager import delete_quota_override

        multi_tenant = app_state.config.identity_provider is not None
        deleted = delete_quota_override(body.provider, body.model_name, multi_tenant)
        reset_shared_rate_limit_handlers()
        return JSONResponse({"ok": True, "deleted": deleted})

    # ----------------------------------------------------------------
    # Semantic / agent settings
    # ----------------------------------------------------------------

    @router.get("/settings/semantic")
    async def get_semantic_settings() -> JSONResponse:
        """Return current semantic feature flags from settings.json."""
        from local_search_agent.core.key_manager import get_semantic_settings

        return JSONResponse(get_semantic_settings())

    @router.post("/settings/semantic")
    async def set_semantic_settings(body: SemanticSettingsRequest) -> JSONResponse:
        """
        Update semantic feature flags at runtime.
        Persists to settings.json so CLI, UI, and Python API all share state.
        """
        from local_search_agent.core.key_manager import set_all_semantic_settings

        app_state.config.enable_semantic = body.enable_semantic
        app_state.config.enable_query_expansion = body.enable_query_expansion
        app_state.config.semantic_model = body.semantic_model or None
        app_state.invalidate_agents()

        set_all_semantic_settings(
            enable_semantic=body.enable_semantic,
            enable_query_expansion=body.enable_query_expansion,
            semantic_provider=body.semantic_provider,
            semantic_model=body.semantic_model,
        )

        return JSONResponse(
            {
                "ok": True,
                "enable_semantic": app_state.config.enable_semantic,
                "enable_query_expansion": app_state.config.enable_query_expansion,
                "semantic_provider": body.semantic_provider,
                "semantic_model": body.semantic_model,
            }
        )

    @router.get("/settings/reranking")
    async def get_reranking_settings() -> JSONResponse:
        """Return current re-ranking feature flags from config."""
        return JSONResponse(
            {
                "enable_reranking": app_state.config.enable_reranking,
                "rerank_candidate_multiplier": app_state.config.rerank_candidate_multiplier,
            }
        )

    @router.post("/settings/reranking")
    async def set_reranking_settings(body: RerankSettingsRequest) -> JSONResponse:
        """
        Update re-ranking settings at runtime and persist to settings.json.
        Takes effect immediately for all subsequent searches.
        """
        from local_search_agent.core.key_manager import set_all_reranking_settings

        app_state.config.enable_reranking = body.enable_reranking
        app_state.config.rerank_candidate_multiplier = body.rerank_candidate_multiplier
        app_state.invalidate_agents()
        set_all_reranking_settings(
            enable_reranking=body.enable_reranking,
            rerank_candidate_multiplier=body.rerank_candidate_multiplier,
        )
        return JSONResponse(
            {
                "ok": True,
                "enable_reranking": body.enable_reranking,
                "rerank_candidate_multiplier": body.rerank_candidate_multiplier,
            }
        )

    # ----------------------------------------------------------------
    # Advanced (ingestion tuning) settings
    # ----------------------------------------------------------------

    @router.get("/settings/advanced")
    async def get_advanced_settings() -> JSONResponse:
        """Return user-overridden advanced settings plus effective defaults."""
        from local_search_agent.core.key_manager import (
            get_advanced_settings,
            get_effective_constants,
        )

        return JSONResponse(
            {
                "overrides": get_advanced_settings(),
                "effective": get_effective_constants(),
            }
        )

    @router.post("/settings/advanced")
    async def set_advanced_settings(body: AdvancedSettingsRequest) -> JSONResponse:
        """
        Persist advanced setting overrides. Pass an empty dict to reset all to defaults.

        DEFAULT_TOP_K and DEFAULT_MAX_ITERATIONS are also applied live to the
        running app_state.config (mirroring /settings/reranking) so no UI/agent
        restart is needed — the next query picks up the new value immediately.
        All other advanced settings (chunking, PDF/DOCX batching, snippet length)
        are already re-read from advanced_settings.json on every ingest/search
        call and never required a restart.
        """
        from local_search_agent.core.key_manager import (
            get_effective_constants,
            set_advanced_settings,
        )

        set_advanced_settings(body.overrides)
        effective = get_effective_constants()
        app_state.config.top_k = effective["DEFAULT_TOP_K"]
        app_state.config.max_iterations = effective["DEFAULT_MAX_ITERATIONS"]
        app_state.invalidate_agents()
        return JSONResponse({"ok": True, "effective": effective})

    @router.delete("/settings/advanced")
    async def reset_advanced_settings() -> JSONResponse:
        """Reset all advanced settings to compiled-in defaults. Applies live, no restart."""
        from local_search_agent.core.key_manager import (
            get_effective_constants,
            set_advanced_settings,
        )

        set_advanced_settings({})
        effective = get_effective_constants()
        app_state.config.top_k = effective["DEFAULT_TOP_K"]
        app_state.config.max_iterations = effective["DEFAULT_MAX_ITERATIONS"]
        app_state.invalidate_agents()
        return JSONResponse({"ok": True, "effective": effective})

    # ----------------------------------------------------------------
    # Ingestion
    # ----------------------------------------------------------------

    @router.post("/ingest")
    async def trigger_ingest(body: IngestRequest, request: Request) -> JSONResponse:
        """
        Trigger a manual re-ingest for a workspace.
        Runs in a background thread so the HTTP response returns immediately.
        The frontend polls /api/ui/ingest/status for progress.

        force=True ("Force Re-ingest" in the UI) reprocesses every file
        regardless of modification time -- a much heavier, slower operation
        than the default incremental sync. Tightened to superadmin_only:
        a workspace admin can still trigger the ordinary incremental ingest
        (force=False) freely, but re-processing an entire workspace from
        scratch is reserved for whoever runs the deployment, same
        provisioning-vs-administration line drawn for workspace
        create/delete in route_policy.py. Single-user mode (no
        identity_provider) and superadmin both bypass this, matching every
        other RBAC check in this codebase.
        """
        identity = getattr(request.state, "identity", None)
        if body.force and identity is not None and not identity.is_superadmin:
            raise HTTPException(403, detail="Force re-ingest is restricted to superadmin.")
        _log_activity(
            app_state, request, "ingest", workspace=body.workspace, detail=f"force={body.force}"
        )
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, _run_ingest, app_state, body.workspace, body.force)
        return JSONResponse(
            {
                "ok": True,
                "message": f"Ingestion started for {body.workspace!r} (force={body.force}).",
            }
        )

    @router.post("/ingest/wipe")
    async def ingest_wipe(body: IngestRequest, request: Request) -> JSONResponse:
        """
        Wipe all indexed documents for the workspace (Meilisearch + SQLite),
        then immediately kick off a force re-ingest with live progress tracking.

        Always superadmin_only -- this is the single most destructive
        action a workspace admin could otherwise trigger (deletes every
        indexed document before rebuilding), so unlike ordinary ingest
        there's no non-destructive variant to leave open to workspace
        admins. Single-user mode (no identity_provider) bypasses this,
        matching every other RBAC check in this codebase.
        """
        identity = getattr(request.state, "identity", None)
        if identity is not None and not identity.is_superadmin:
            raise HTTPException(403, detail="Wipe & re-ingest is restricted to superadmin.")
        # Logged before the destructive wipe is scheduled -- per the design
        # doc's audit-logging principle, a crash mid-wipe should still leave
        # a record of intent. The actual SQLite/Meilisearch deletion happens
        # a moment later in _run_ingest's background thread, which has no
        # request context (and therefore no identity) to log with directly.
        _log_activity(app_state, request, "workspace_wipe", workspace=body.workspace)
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, _run_ingest, app_state, body.workspace, True, True)
        return JSONResponse(
            {
                "ok": True,
                "message": f"Wipe and re-ingest started for {body.workspace!r}.",
            }
        )

    @router.get("/ingest/status")
    async def ingest_status(request: Request) -> JSONResponse:
        with _ingest_lock:
            workspaces = [p.to_dict() for p in _ingest_registry.values()]
        granted = _filter_workspaces_by_grant(app_state, request)
        if granted is not None:
            workspaces = [w for w in workspaces if w["workspace"] in granted]
        return JSONResponse({"workspaces": workspaces})

    # ----------------------------------------------------------------
    # Scheduler
    # ----------------------------------------------------------------

    @router.post("/scheduler")
    async def set_scheduler(body: SchedulerSetRequest) -> JSONResponse:
        # Auto-start the scheduler on first use if it wasn't started at boot
        if app_state.scheduler is None:
            app_state.start_scheduler(interval_minutes=body.interval_minutes)
        try:
            ws = app_state.workspace_manager.get_workspace(body.workspace)
            if ws is None:
                raise HTTPException(404, detail=f"Workspace {body.workspace!r} not found.")
            from local_search_agent.core.config import SearchAgentConfig

            ws_config = SearchAgentConfig(
                workspace_name=body.workspace,
                document_dirs=_normalise_dirs(ws),
                meilisearch_url=app_state.config.meilisearch_url,
                meili_master_key=app_state.config.meili_master_key,
                provider=app_state.config.provider,
                db_path=app_state.config.db_path,
            )

            # Build a progress callback that writes into the shared ingest registry
            # so the status bar shows scheduler syncs just like manual ingests.
            def _make_callback(ws_name):
                def _cb(indexed, skipped, failed, total, current_file):
                    import os as _os
                    import time as _time

                    with _ingest_lock:
                        prog = _ingest_registry.get(ws_name)
                        if prog is None or prog.status not in ("running",):
                            prog = _IngestProgress(
                                workspace=ws_name,
                                status="running",
                                started_at=_time.monotonic(),
                            )
                            _ingest_registry[ws_name] = prog
                        prog.files_total = total
                        prog.files_processed = indexed + skipped + failed
                        prog.files_skipped = skipped
                        prog.files_failed = failed
                        prog.current_file = (
                            _os.path.basename(current_file)
                            if current_file not in ("", "__done__")
                            else ""
                        )
                        if current_file == "__done__":
                            prog.status = "done"
                            prog.finished_at = _time.monotonic()

                return _cb

            app_state.scheduler.add_workspace(
                ws_config,
                interval_minutes=body.interval_minutes,
                progress_callback=_make_callback(body.workspace),
            )
            app_state.store.set_config(
                f"scheduler.{body.workspace}.interval_minutes", body.interval_minutes
            )
            return JSONResponse({"ok": True, "interval_minutes": body.interval_minutes})
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, detail=str(e))

    @router.delete("/scheduler")
    async def stop_scheduler() -> JSONResponse:
        if app_state.scheduler is None or not app_state.scheduler.is_running:
            return JSONResponse({"ok": True, "message": "Scheduler was not running."})
        app_state.scheduler.stop(wait=False)
        app_state.scheduler = None
        return JSONResponse({"ok": True, "message": "Scheduler stopped."})

    @router.get("/scheduler/status")
    async def scheduler_status(request: Request) -> JSONResponse:
        if app_state.scheduler is None:
            return JSONResponse({"running": False, "jobs": []})
        status = app_state.scheduler.get_status()
        granted = _filter_workspaces_by_grant(app_state, request)
        if granted is not None:
            status["registered_workspaces"] = [
                w for w in status["registered_workspaces"] if w in granted
            ]
            status["scheduled_jobs"] = [
                j
                for j in status["scheduled_jobs"]
                if j["job_id"].removeprefix("incremental_sync_") in granted
            ]
        return JSONResponse(status)

    # ----------------------------------------------------------------
    # Watch mode (filesystem-event-driven, replaces the polling scheduler)
    # ----------------------------------------------------------------

    @router.post("/watch")
    async def add_workspace_to_watch(body: WatchSetRequest) -> JSONResponse:
        """
        Add a workspace to watch mode. Auto-starts the watcher on first use.
        """
        if app_state.watcher is None:
            app_state.start_watch_mode()
        try:
            ws = app_state.workspace_manager.get_workspace(body.workspace)
            if ws is None:
                raise HTTPException(404, detail=f"Workspace {body.workspace!r} not found.")
            from local_search_agent.core.config import SearchAgentConfig
            from local_search_agent.core.key_manager import get_watch_mode_settings

            watch_settings = get_watch_mode_settings()
            ws_config = SearchAgentConfig(
                workspace_name=body.workspace,
                document_dirs=_normalise_dirs(ws),
                meilisearch_url=app_state.config.meilisearch_url,
                meili_master_key=app_state.config.meili_master_key,
                provider=app_state.config.provider,
                db_path=app_state.config.db_path,
                enrich_on_watch=watch_settings["enrich_on_watch"],
            )

            # Reuse the same progress-callback shape as the scheduler so the
            # status bar shows watch-triggered syncs just like manual ingests.
            def _make_callback(ws_name):
                def _cb(indexed, skipped, failed, total, current_file):
                    import os as _os
                    import time as _time

                    with _ingest_lock:
                        prog = _ingest_registry.get(ws_name)
                        if prog is None or prog.status not in ("running",):
                            prog = _IngestProgress(
                                workspace=ws_name,
                                status="running",
                                started_at=_time.monotonic(),
                            )
                            _ingest_registry[ws_name] = prog
                        prog.files_total = total
                        prog.files_processed = indexed + skipped + failed
                        prog.files_skipped = skipped
                        prog.files_failed = failed
                        prog.current_file = (
                            _os.path.basename(current_file)
                            if current_file not in ("", "__done__")
                            else ""
                        )
                        if current_file == "__done__":
                            prog.status = "done"
                            prog.finished_at = _time.monotonic()

                return _cb

            app_state.watcher.add_workspace(
                ws_config,
                progress_callback=_make_callback(body.workspace),
            )
            return JSONResponse({"ok": True, "workspace": body.workspace})
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, detail=str(e))

    @router.delete("/watch")
    async def stop_watch_mode() -> JSONResponse:
        if app_state.watcher is None or not app_state.watcher.is_running:
            return JSONResponse({"ok": True, "message": "Watch mode was not running."})
        app_state.watcher.stop(wait=False)
        app_state.watcher = None
        return JSONResponse({"ok": True, "message": "Watch mode stopped."})

    @router.get("/watch/status")
    async def watch_status(request: Request) -> JSONResponse:
        if app_state.watcher is None:
            return JSONResponse(
                {"running": False, "registered_workspaces": [], "watched_directories": {}}
            )
        status = app_state.watcher.get_status()
        granted = _filter_workspaces_by_grant(app_state, request)
        if granted is not None:
            status["registered_workspaces"] = [
                w for w in status["registered_workspaces"] if w in granted
            ]
            status["watched_directories"] = {
                w: c for w, c in status["watched_directories"].items() if w in granted
            }
        return JSONResponse(status)

    @router.post("/watch/trigger/{workspace_name}")
    async def trigger_watch_sync(workspace_name: str) -> JSONResponse:
        """Force an immediate sync for a workspace registered with watch mode."""
        if app_state.watcher is None:
            raise HTTPException(400, detail="Watch mode is not running.")
        try:
            app_state.watcher.trigger_now(workspace_name)
            return JSONResponse({"ok": True, "workspace": workspace_name})
        except ValueError as e:
            raise HTTPException(404, detail=str(e))

    @router.get("/settings/watch-mode")
    async def get_watch_mode_settings_route() -> JSONResponse:
        """Return current watch-mode feature flags from settings.json."""
        from local_search_agent.core.key_manager import get_watch_mode_settings

        return JSONResponse(get_watch_mode_settings())

    @router.post("/settings/watch-mode")
    async def set_watch_mode_settings_route(body: WatchModeSettingsRequest) -> JSONResponse:
        """
        Update watch-mode feature flags (enable_watch_mode, enrich_on_watch).
        Persists to settings.json. Does not itself start/stop the watcher —
        use POST/DELETE /api/ui/watch for that.
        """
        from local_search_agent.core.key_manager import set_all_watch_mode_settings

        app_state.config.enable_watch_mode = body.enable_watch_mode
        app_state.config.enrich_on_watch = body.enrich_on_watch

        set_all_watch_mode_settings(
            enable_watch_mode=body.enable_watch_mode,
            enrich_on_watch=body.enrich_on_watch,
        )

        return JSONResponse(
            {
                "ok": True,
                "enable_watch_mode": body.enable_watch_mode,
                "enrich_on_watch": body.enrich_on_watch,
            }
        )

    @router.get("/db-info")
    async def db_info() -> JSONResponse:
        """Return the current database path so the UI can display it as a hint."""
        return JSONResponse({"db_path": app_state.config.db_path})

    @router.post("/restart")
    async def restart_with_db(request: Request) -> JSONResponse:
        """
        Save a new db_path and restart the UI process.
        The actual restart is delegated to the pywebview JS bridge so it
        runs in the main thread. We trigger it by calling window.pywebview.api
        via a deferred JS evaluation after the HTTP response is sent.
        """
        body = await request.json()
        new_db_path = body.get("db_path", "").strip()
        if not new_db_path:
            raise HTTPException(400, detail="db_path is required.")
        # Validate the parent directory exists
        import pathlib

        parent = pathlib.Path(new_db_path).parent
        if not parent.exists():
            raise HTTPException(400, detail=f"Directory does not exist: {parent}")
        # Persist immediately so the new process picks it up even if pywebview restart fails
        from local_search_agent.core.key_manager import set_saved_db_path

        set_saved_db_path(new_db_path)
        # Schedule the actual process restart
        import os as _os
        import subprocess as _subprocess
        import sys as _sys
        import threading

        # Build the relaunch command using the same executable (works on all platforms)
        _cmd = [_sys.executable, "-m", "local_search_agent.ui.dashboard", "--db", new_db_path]

        def _do_restart():
            import time

            time.sleep(0.8)  # let the HTTP response reach the browser first
            # Spawn a short-lived helper process that waits for this process to die
            # and release its port before starting the new UI process.
            # Using a helper avoids the port-collision race on all platforms.
            helper = f"import time, subprocess; time.sleep(2); subprocess.Popen({_cmd!r})"
            _subprocess.Popen([_sys.executable, "-c", helper])
            _os._exit(0)

        threading.Thread(target=_do_restart, daemon=True).start()
        return JSONResponse({"ok": True, "restarting": True, "db_path": new_db_path})

    # ----------------------------------------------------------------
    # Workspaces (UI-facing summary)
    # ----------------------------------------------------------------

    @router.get("/workspaces")
    async def list_workspaces(request: Request) -> JSONResponse:
        workspaces = app_state.workspace_manager.list_workspaces()
        # Normalise: DB stores document_dir (singular TEXT column).
        # Add document_dirs list so the frontend always sees the same shape.
        for ws in workspaces:
            ws["document_dirs"] = _normalise_dirs(ws)

        # Multi-tenant mode: only return workspaces the caller actually has
        # a workspace_members grant in. This route is deliberately NOT a
        # route_policy.py entry -- RoutePolicy expresses "does this caller
        # have role X in workspace Y", but a *listing* needs "which of these
        # many workspaces does the caller have any role in at all", a
        # different shape of check that doesn't fit the single-workspace
        # RoutePolicy model. AuthorizationMiddleware therefore never touches
        # this route at all (see route_policy.py's own docstring on routes
        # not in its table), so identity has to be resolved here directly
        # rather than read off request.state -- otherwise every workspace's
        # existence leaks to literally any caller, authenticated or not,
        # once multi-tenant mode is on, and the frontend's dropdown shows
        # workspaces that immediately 403 the moment they're clicked.
        identity_provider = getattr(app_state.config, "identity_provider", None)
        if identity_provider is not None:
            try:
                identity = identity_provider.resolve(request)
            except Exception:
                identity = None
            if identity is None:
                raise HTTPException(401, detail="Authentication required.")
            if not identity.is_superadmin:
                granted = {
                    row["workspace"]
                    for row in app_state.auth_db.list_access(subject=identity.subject)
                }
                workspaces = [ws for ws in workspaces if ws["name"] in granted]

        return JSONResponse({"workspaces": workspaces})

    @router.delete("/workspaces/{workspace_name}")
    async def delete_workspace(
        workspace_name: str, request: Request, wipe: bool = False
    ) -> JSONResponse:
        """
        Delete a workspace registration from SQLite.
        Pass ?wipe=true to also delete its Meilisearch index.
        """
        ws = app_state.workspace_manager.get_workspace(workspace_name)
        if ws is None:
            raise HTTPException(404, detail=f"Workspace {workspace_name!r} not found.")
        # Logged before the delete runs, per the design doc's audit-logging
        # principle -- a crash mid-delete should still leave a record of intent.
        _log_activity(
            app_state, request, "workspace_delete", workspace=workspace_name, detail=f"wipe={wipe}"
        )
        if app_state.config.identity_provider is not None:
            from local_search_agent.auth.meili_key_provisioning import deprovision_workspace_keys

            deprovision_workspace_keys(
                workspace=workspace_name,
                meilisearch_url=app_state.config.meilisearch_url,
                meili_master_key=app_state.config.meili_master_key,
                auth_db=app_state.auth_db,
            )
        try:
            app_state.framework.delete_workspace(name=workspace_name, wipe_index=wipe)
            return JSONResponse({"ok": True, "workspace": workspace_name, "index_wiped": wipe})
        except Exception as e:
            raise HTTPException(500, detail=str(e))

    @router.post("/workspaces")
    async def create_workspace(request: Request) -> JSONResponse:
        import os

        body = await request.json()
        name = body.get("name", "").strip()
        dirs = body.get("document_dirs", [])
        if not name:
            raise HTTPException(400, detail="Workspace name is required.")
        if not dirs:
            raise HTTPException(400, detail="At least one document_dir is required.")
        for d in dirs:
            if not os.path.isdir(d):
                raise HTTPException(400, detail=f"Directory does not exist: {d!r}")
        try:
            # WorkspaceManager.create_workspace takes one dir at a time;
            # call it for each dir — it upserts so the last one wins as the
            # canonical document_dir (multi-dir support is a future schema change).
            for d in dirs:
                app_state.workspace_manager.create_workspace(name=name, document_dir=d)
            _log_activity(app_state, request, "workspace_create", workspace=name)
            if app_state.config.identity_provider is not None:
                # Non-fatal by design -- see meili_key_provisioning.py's
                # module docstring. A workspace is fully usable without a
                # scoped key; member requests just fall back to the
                # service-level master key until this is retried.
                from local_search_agent.auth.meili_key_provisioning import (
                    provision_workspace_keys,
                )

                provision_workspace_keys(
                    workspace=name,
                    meilisearch_url=app_state.config.meilisearch_url,
                    meili_master_key=app_state.config.meili_master_key,
                    auth_db=app_state.auth_db,
                )
            return JSONResponse({"ok": True, "workspace": name})
        except Exception as e:
            raise HTTPException(500, detail=str(e))

    # ----------------------------------------------------------------
    # Export chat
    # ----------------------------------------------------------------

    @router.post("/export-chat")
    async def export_chat(request: Request) -> Response:
        """
        Export the current chat as Markdown.

        Two delivery modes, chosen per-request by whether `folder` is
        present in the body -- not a server-side setting:

        - `folder` given (desktop app, via pywebview's pick_folder()):
          write to that path on THIS process's own disk and open it with
          the OS default app, exactly as before. Correct because the
          caller in this mode is always the same machine this process
          runs on.
        - `folder` absent (any plain browser tab -- no pywebview bridge to
          pick a folder with, whether that tab happens to be an
          employee's RDP session on this same server or a genuinely
          separate machine reached over the network): return the content
          directly as the HTTP response body with `Content-Disposition:
          attachment`, and let the BROWSER's own download mechanism
          deliver it. This is correct in both of those sub-cases, since a
          browser's download always lands wherever the person actually is
          sitting -- unlike writing to a path on this process's disk,
          which is only ever the right place when that disk belongs to
          the same machine the person is looking at.

        The frontend already does this exact same fallback for Markdown
        by building the file client-side and never calling this endpoint
        at all when there's no pywebview bridge (see exportChatMarkdown()
        in ui/templates/_script_toolbar_ingest.html) -- this branch exists
        for any other caller of this endpoint (API/CLI use, or future
        frontend changes) so the same correct behavior isn't tied to one
        specific frontend code path.
        """
        body = await request.json()
        folder = (body.get("folder") or "").strip()
        filename = body.get("filename", "chat.md").strip()
        content = body.get("content", "")

        # Sanitise filename — strip any path separators the JS might have snuck in
        filename = os.path.basename(filename) or "chat.md"

        if not folder:
            return Response(
                content=content.encode("utf-8"),
                media_type="text/markdown",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )

        if not os.path.isdir(folder):
            raise HTTPException(400, detail=f"Invalid folder: {folder!r}")
        filepath = os.path.join(folder, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        # Open the file with the OS default app (Notepad / TextEdit / gedit)
        import subprocess
        import sys

        try:
            if sys.platform == "win32":
                os.startfile(filepath)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", filepath])
            else:
                subprocess.Popen(["xdg-open", filepath])
        except Exception as e:
            logger.warning("Could not open exported file: %s", e)

        return JSONResponse({"ok": True, "path": filepath})

    # ----------------------------------------------------------------
    # Export chat — Word (.docx)
    # ----------------------------------------------------------------

    @router.post("/export-chat-docx")
    async def export_chat_docx(body: ExportDocxRequest) -> Response:
        """Same dual delivery mode as export_chat() above, keyed off
        whether `folder` is present -- see that handler's docstring."""
        from local_search_agent.ui.export_docx import build_docx

        filename = os.path.basename(body.filename) or "chat.docx"
        if not filename.lower().endswith(".docx"):
            filename += ".docx"

        try:
            docx_bytes = build_docx(body.messages, app_state.workspace_manager)
        except Exception as e:
            logger.exception("Word export failed: %s", e)
            raise HTTPException(500, detail=f"Word export failed: {e}")

        if not body.folder:
            return Response(
                content=docx_bytes,
                media_type=(
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                ),
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )

        folder = body.folder.strip()
        if not os.path.isdir(folder):
            raise HTTPException(400, detail=f"Invalid folder: {folder!r}")
        filepath = os.path.join(folder, filename)

        with open(filepath, "wb") as f:
            f.write(docx_bytes)

        import subprocess
        import sys

        try:
            if sys.platform == "win32":
                os.startfile(filepath)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", filepath])
            else:
                subprocess.Popen(["xdg-open", filepath])
        except Exception as e:
            logger.warning("Could not open exported Word file: %s", e)

        return JSONResponse({"ok": True, "path": filepath})

    # ----------------------------------------------------------------
    # Export a single table from a chat answer — Excel (.xlsx)
    # ----------------------------------------------------------------

    @router.post("/export-table-xlsx")
    async def export_table_xlsx(body: ExportTableXlsxRequest) -> Response:
        """Same dual delivery mode as export_chat() above, keyed off
        whether `folder` is present -- see that handler's docstring."""
        from local_search_agent.ui.export_xlsx import build_xlsx

        if not body.headers:
            raise HTTPException(400, detail="Table has no headers to export.")

        filename = os.path.basename(body.filename) or "table.xlsx"
        if not filename.lower().endswith(".xlsx"):
            filename += ".xlsx"

        try:
            xlsx_bytes = build_xlsx(body.headers, body.rows)
        except Exception as e:
            logger.exception("Excel export failed: %s", e)
            raise HTTPException(500, detail=f"Excel export failed: {e}")

        if not body.folder:
            return Response(
                content=xlsx_bytes,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )

        folder = body.folder.strip()
        if not os.path.isdir(folder):
            raise HTTPException(400, detail=f"Invalid folder: {folder!r}")
        filepath = os.path.join(folder, filename)

        with open(filepath, "wb") as f:
            f.write(xlsx_bytes)

        import subprocess
        import sys

        try:
            if sys.platform == "win32":
                os.startfile(filepath)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", filepath])
            else:
                subprocess.Popen(["xdg-open", filepath])
        except Exception as e:
            logger.warning("Could not open exported Excel file: %s", e)

        return JSONResponse({"ok": True, "path": filepath})

    # ----------------------------------------------------------------
    # Health (used by the status bar)
    # ----------------------------------------------------------------

    @router.get("/health")
    async def health() -> JSONResponse:
        import logging as _logging

        import httpx

        # httpx logs every outbound request at INFO; suppress for the health poll
        _httpx_log = _logging.getLogger("httpx")
        _prev_level = _httpx_log.level
        _httpx_log.setLevel(_logging.WARNING)
        status: dict = {"server": "ok"}
        meili_url = app_state.config.meilisearch_url
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(f"{meili_url}/health")
                if r.status_code == 200:
                    status["meilisearch"] = "ok"
                else:
                    status["meilisearch"] = f"error:{r.status_code}"
                    logger.warning("Meilisearch health returned HTTP %d", r.status_code)
        except httpx.ConnectError:
            status["meilisearch"] = "offline"
            logger.debug("Meilisearch unreachable at %s", meili_url)
        except Exception as e:
            status["meilisearch"] = "offline"
            logger.warning("Meilisearch health check error: %s", e)
        finally:
            _httpx_log.setLevel(_prev_level)
        return JSONResponse(status)

    return router


# ---------------------------------------------------------------------------
# Streaming agent runner
# ---------------------------------------------------------------------------


async def _run_agent_streaming(
    app_state,
    session_id: str,
    question: str,
    workspace: str,
    request: Request,
    meili_api_key: Optional[str] = None,
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
) -> AsyncIterator[str]:
    """
    Async generator yielding SSE-formatted strings.

    The synchronous LangGraph agent runs in a daemon thread and pushes
    events onto a thread-safe queue.  This coroutine drains the queue
    into SSE events, checking for client disconnect between each drain.

    meili_api_key : Per-request scoped Meilisearch key from
                    request.state.meili_key (Phase 7 -- see
                    AuthorizationMiddleware._resolve_meili_key and
                    AppState.get_agent). None for single-user mode and
                    admin-role multi-tenant requests, both of which use
                    the service-level master key instead.
    provider, model_name : Optional per-request model override (Model/
                    Provider Access Control, Option B). Already validated
                    against the caller's role's allow-list by the /query
                    handler before this generator is ever constructed --
                    this function just forwards them to
                    AppState.get_agent(), it does no enforcement of its
                    own.
    """
    import queue as _queue
    import threading

    q: _queue.Queue = _queue.Queue()
    DONE_SENTINEL = object()
    ERROR_SENTINEL = object()

    # Persist user message immediately so it's in history even if agent errors
    app_state.store.add_message(session_id=session_id, role="user", content=question)

    def _agent_thread():
        from local_search_agent.agent.rate_limit_handler import set_queued_callback

        # Notifies the SSE loop below when an LLM call has to wait for a
        # concurrency slot -- see rate_limit_handler.py's ConcurrencyGate
        # and set_queued_callback()'s own docstrings. Thread-local, set
        # fresh at the start of every request's own dedicated thread, so
        # it's correct even though the underlying LocalSearchAgent/
        # RateLimitHandler instance may be shared across concurrent
        # requests (that sharing is the whole point of the gate).
        set_queued_callback(
            lambda waiting_ahead: q.put(("queued", {"waiting_ahead": waiting_ahead}))
        )
        try:
            agent = app_state.get_agent(
                workspace, meili_api_key=meili_api_key, provider=provider, model_name=model_name
            )
            agent._get_tools()  # ensure graph + tools are initialised

            for event in agent.stream(question=question, workspace=workspace):
                etype = event["type"]

                if etype == "thinking":
                    q.put(("thinking", {"text": event["text"]}))

                elif etype == "tool_start":
                    q.put(
                        (
                            "tool_start",
                            {
                                "tool": event["tool"],
                                "input": event["input"],
                                "call_id": event["call_id"],
                            },
                        )
                    )

                elif etype == "tool_end":
                    q.put(
                        (
                            "tool_end",
                            {
                                "tool": event["tool"],
                                "output": event["output"],
                                "duration_ms": event["duration_ms"],
                                "call_id": event["call_id"],
                            },
                        )
                    )

                elif etype == "token_update":
                    q.put(
                        (
                            "token_update",
                            {
                                "token_query": event["token_input"],
                                "token_reply": event["token_output"],
                            },
                        )
                    )

                elif etype == "text":
                    q.put(("text_chunk", {"text": event["text"]}))

                elif etype == "done":
                    q.put(
                        (
                            "done",
                            {
                                "token_query": event["token_input"],
                                "token_reply": event["token_output"],
                                "iterations_used": event["iterations_used"],
                            },
                        )
                    )

        except Exception as e:
            logger.exception("Agent thread error: %s", e)
            q.put((ERROR_SENTINEL, str(e)))
        finally:
            set_queued_callback(None)
            q.put((DONE_SENTINEL, None))

    threading.Thread(target=_agent_thread, daemon=True).start()

    assembled_text: list[str] = []
    assembled_tool_calls: list[dict] = []
    assembled_thinking: str = ""
    loop = asyncio.get_event_loop()

    while True:
        if await request.is_disconnected():
            logger.info("SSE client disconnected for session %r", session_id)
            break

        try:
            event_type, data = await loop.run_in_executor(None, lambda: q.get(timeout=0.1))
        except Exception:
            continue

        if event_type is DONE_SENTINEL:
            break

        if event_type is ERROR_SENTINEL:
            yield _sse_event("error", {"message": str(data)})
            break

        if event_type == "thinking":
            assembled_thinking = data["text"]
            yield _sse_event("thinking", data)

        elif event_type == "queued":
            # Emitted by ConcurrencyGate.acquire()'s on_wait callback
            # (see rate_limit_handler.py) -- an LLM call inside this
            # query is waiting for a free concurrency slot. Purely
            # informational for the UI ("N requests ahead of you"); no
            # store write, no assembled state to update.
            yield _sse_event("queued", data)

        elif event_type == "tool_start":
            yield _sse_event("tool_start", data)
            assembled_tool_calls.append(
                {
                    "tool": data["tool"],
                    "input": data["input"],
                    "call_id": data["call_id"],
                    "output": None,
                    "duration_ms": None,
                }
            )

        elif event_type == "tool_end":
            yield _sse_event("tool_end", data)
            for tc in assembled_tool_calls:
                if tc.get("call_id") == data.get("call_id"):
                    tc["output"] = data["output"]
                    tc["duration_ms"] = data["duration_ms"]
                    break

        elif event_type == "token_update":
            yield _sse_event("token_update", data)

        elif event_type == "text_chunk":
            assembled_text.append(data["text"])
            yield _sse_event("text_chunk", data)

        elif event_type == "done":
            full_content = "".join(assembled_text)
            msg = app_state.store.add_message(
                session_id=session_id,
                role="assistant",
                content=full_content,
                tool_calls=assembled_tool_calls,
                thinking=assembled_thinking,
                token_query=data.get("token_query", 0),
                token_reply=data.get("token_reply", 0),
            )
            yield _sse_event(
                "done",
                {
                    "token_query": data.get("token_query", 0),
                    "token_reply": data.get("token_reply", 0),
                    "message_id": msg["message_id"],
                    "iterations_used": data.get("iterations_used", 0),
                },
            )


def _sse_event(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


# ---------------------------------------------------------------------------
# Background ingest helper (called from thread pool)
# ---------------------------------------------------------------------------


def _run_ingest(app_state, workspace: str, force: bool = False, wipe: bool = False) -> None:
    """Run ingestion synchronously in a background thread, reporting live progress.

    Parameters
    ----------
    force : Re-index all files regardless of modification time.
    wipe  : Delete the Meilisearch index and all SQLite document records for
            the workspace before ingesting.  Implies force=True.
    """
    if wipe:
        force = True

    # Register progress slot
    progress = _IngestProgress(workspace=workspace, status="running", started_at=time.monotonic())
    with _ingest_lock:
        _ingest_registry[workspace] = progress

    def _callback(indexed: int, skipped: int, failed: int, total: int, current_file: str) -> None:
        with _ingest_lock:
            progress.files_total = total
            progress.files_processed = indexed + skipped + failed
            progress.files_skipped = skipped
            progress.files_failed = failed
            progress.current_file = (
                os.path.basename(current_file) if current_file not in ("", "__done__") else ""
            )
            if current_file == "__done__":
                progress.status = "done"
                progress.finished_at = time.monotonic()

    try:
        logger.info(
            "Manual ingest triggered for workspace %r (force=%s, wipe=%s)",
            workspace,
            force,
            wipe,
        )
        ws = app_state.workspace_manager.get_workspace(workspace)
        if ws is None:
            logger.error("Ingest: workspace %r not found", workspace)
            with _ingest_lock:
                progress.status = "error"
                progress.error = f"Workspace {workspace!r} not found"
            return

        from local_search_agent.core.config import SearchAgentConfig
        from local_search_agent.ingestion.pipeline import IngestionPipeline
        from local_search_agent.search.meilisearch_client import MeilisearchClient

        ws_config = SearchAgentConfig(
            workspace_name=workspace,
            document_dirs=_normalise_dirs(ws),
            meilisearch_url=app_state.config.meilisearch_url,
            meili_master_key=app_state.config.meili_master_key,
            provider=app_state.config.provider,
            db_path=app_state.config.db_path,
            semantic_model=app_state.config.semantic_model,
        )
        mc = MeilisearchClient(
            url=ws_config.meilisearch_url,
            api_key=ws_config.meili_master_key,
            index_name=ws_config.index_name or workspace,
        )

        # ── Wipe step ────────────────────────────────────────────────────
        if wipe:
            import sqlite3 as _sqlite3

            try:
                mc.delete_index()
                logger.info("Wipe: Meilisearch index deleted for %r.", workspace)
            except Exception as e:
                logger.warning("Wipe: could not delete Meilisearch index for %r: %s", workspace, e)
            try:
                with _sqlite3.connect(app_state.config.db_path) as conn:
                    deleted = conn.execute(
                        "DELETE FROM documents WHERE workspace = ?", (workspace,)
                    ).rowcount
                    conn.commit()
                logger.info("Wipe: removed %d SQLite document records for %r.", deleted, workspace)
            except Exception as e:
                logger.warning("Wipe: could not clear SQLite records for %r: %s", workspace, e)
        # ─────────────────────────────────────────────────────────────────

        pipeline = IngestionPipeline(
            config=ws_config,
            workspace_manager=app_state.workspace_manager,
            meili_client=mc,
        )
        stats = pipeline.run(force=force, progress_callback=_callback)
        with _ingest_lock:
            progress.chunks_indexed = stats.indexed
            progress.failed_files = [os.path.basename(f) for f in stats.errors if f]
        logger.info("Manual ingest complete for %r: %s", workspace, stats)

    except Exception as e:
        logger.exception("Manual ingest failed for %r: %s", workspace, e)
        with _ingest_lock:
            progress.status = "error"
            progress.error = str(e)
            progress.finished_at = time.monotonic()
