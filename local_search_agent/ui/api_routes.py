"""
FastAPI routes for the Desktop Dashboard UI.

All routes are mounted under /api/ui on the existing FastAPI app, or on a
dedicated app instance when the dashboard runs standalone.

SSE stream contract (POST /api/ui/query):
------------------------------------------
Each event has a named type and a JSON data payload:

    event: thinking
    data: {"text": "..."}

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
from fastapi.responses import JSONResponse, StreamingResponse
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


# ---------------------------------------------------------------------------
# Pydantic request bodies
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    session_id: str
    question: str
    workspace: Optional[str] = None


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
    """
    router = APIRouter(prefix="/api/ui", tags=["ui"])

    # ----------------------------------------------------------------
    # Sessions
    # ----------------------------------------------------------------

    @router.post("/sessions")
    async def create_session(body: NewSessionRequest) -> JSONResponse:
        session = app_state.store.create_session(
            workspace=body.workspace,
            title=body.title or "",
        )
        return JSONResponse(session)

    @router.get("/sessions")
    async def list_sessions(workspace: str, limit: int = 50) -> JSONResponse:
        sessions = app_state.store.list_sessions(workspace=workspace, limit=limit)
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
    async def delete_session(session_id: str) -> JSONResponse:
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

        return StreamingResponse(
            _run_agent_streaming(
                app_state=app_state,
                session_id=body.session_id,
                question=body.question,
                workspace=workspace,
                request=request,
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
            app_state._agent = None
        elif body.key == "global.model":
            app_state.config.model_name = body.value
            app_state._agent = None
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
                app_state._agent = None
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
            app_state._agent = None
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
        """Persist advanced setting overrides. Pass an empty dict to reset all to defaults."""
        from local_search_agent.core.key_manager import (
            get_effective_constants,
            set_advanced_settings,
        )

        set_advanced_settings(body.overrides)
        return JSONResponse({"ok": True, "effective": get_effective_constants()})

    @router.delete("/settings/advanced")
    async def reset_advanced_settings() -> JSONResponse:
        """Reset all advanced settings to compiled-in defaults."""
        from local_search_agent.core.key_manager import (
            get_effective_constants,
            set_advanced_settings,
        )

        set_advanced_settings({})
        return JSONResponse({"ok": True, "effective": get_effective_constants()})

    # ----------------------------------------------------------------
    # Ingestion
    # ----------------------------------------------------------------

    @router.post("/ingest")
    async def trigger_ingest(body: IngestRequest) -> JSONResponse:
        """
        Trigger a manual re-ingest for a workspace.
        Runs in a background thread so the HTTP response returns immediately.
        The frontend polls /api/ui/ingest/status for progress.
        """
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, _run_ingest, app_state, body.workspace, body.force)
        return JSONResponse(
            {
                "ok": True,
                "message": f"Ingestion started for {body.workspace!r} (force={body.force}).",
            }
        )

    @router.post("/ingest/wipe")
    async def ingest_wipe(body: IngestRequest) -> JSONResponse:
        """
        Wipe all indexed documents for the workspace (Meilisearch + SQLite),
        then immediately kick off a force re-ingest with live progress tracking.
        """
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, _run_ingest, app_state, body.workspace, True, True)
        return JSONResponse(
            {
                "ok": True,
                "message": f"Wipe and re-ingest started for {body.workspace!r}.",
            }
        )

    @router.get("/ingest/status")
    async def ingest_status() -> JSONResponse:
        with _ingest_lock:
            workspaces = [p.to_dict() for p in _ingest_registry.values()]
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
    async def scheduler_status() -> JSONResponse:
        if app_state.scheduler is None:
            return JSONResponse({"running": False, "jobs": []})
        return JSONResponse(app_state.scheduler.get_status())

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
    async def watch_status() -> JSONResponse:
        if app_state.watcher is None:
            return JSONResponse(
                {"running": False, "registered_workspaces": [], "watched_directories": {}}
            )
        return JSONResponse(app_state.watcher.get_status())

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
    async def list_workspaces() -> JSONResponse:
        workspaces = app_state.workspace_manager.list_workspaces()
        # Normalise: DB stores document_dir (singular TEXT column).
        # Add document_dirs list so the frontend always sees the same shape.
        for ws in workspaces:
            ws["document_dirs"] = _normalise_dirs(ws)
        return JSONResponse({"workspaces": workspaces})

    @router.delete("/workspaces/{workspace_name}")
    async def delete_workspace(workspace_name: str, wipe: bool = False) -> JSONResponse:
        """
        Delete a workspace registration from SQLite.
        Pass ?wipe=true to also delete its Meilisearch index.
        """
        ws = app_state.workspace_manager.get_workspace(workspace_name)
        if ws is None:
            raise HTTPException(404, detail=f"Workspace {workspace_name!r} not found.")
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
            return JSONResponse({"ok": True, "workspace": name})
        except Exception as e:
            raise HTTPException(500, detail=str(e))

    # ----------------------------------------------------------------
    # Export chat
    # ----------------------------------------------------------------

    @router.post("/export-chat")
    async def export_chat(request: Request) -> JSONResponse:
        import os

        body = await request.json()
        folder = body.get("folder", "").strip()
        filename = body.get("filename", "chat.md").strip()
        content = body.get("content", "")

        if not folder or not os.path.isdir(folder):
            raise HTTPException(400, detail=f"Invalid folder: {folder!r}")

        # Sanitise filename — strip any path separators the JS might have snuck in
        filename = os.path.basename(filename) or "chat.md"
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
) -> AsyncIterator[str]:
    """
    Async generator yielding SSE-formatted strings.

    The synchronous LangGraph agent runs in a daemon thread and pushes
    events onto a thread-safe queue.  This coroutine drains the queue
    into SSE events, checking for client disconnect between each drain.
    """
    import queue as _queue
    import threading

    q: _queue.Queue = _queue.Queue()
    DONE_SENTINEL = object()
    ERROR_SENTINEL = object()

    # Persist user message immediately so it's in history even if agent errors
    app_state.store.add_message(session_id=session_id, role="user", content=question)

    def _agent_thread():
        try:
            agent = app_state.get_agent(workspace)
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
