"""
Desktop Dashboard launcher.

Starts the FastAPI backend on a background thread, then opens a pywebview
window pointing at it.  The window uses the native OS webview (WebView2 on
Windows, WKWebView on macOS, WebKitGTK on Linux) — no Electron, no Chromium.

Usage
-----
    python -m local_search_agent.ui.dashboard
    local-search ui                              # via CLI

Environment / .env variables used
----------------------------------
    GOOGLE_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY
    MEILI_URL          (default http://localhost:7700)
    MEILI_MASTER_KEY   (default local_search_master_key)
    LSA_DB_PATH        (default local_search_agent.db)
    LSA_HOST           (default 127.0.0.1)
    LSA_PORT           (default 8765  — separate from the file server on 8000)
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import threading
import time
from typing import Optional

from local_search_agent.core.constants import __version__

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AppState — single object wiring all backend components together
# ---------------------------------------------------------------------------


class AppState:
    """
    Holds all backend singletons.  One instance lives for the lifetime of the
    dashboard process.  API route handlers receive this via closure.

    Lazy properties
    ---------------
    get_agent(workspace)  — builds/returns a LocalSearchAgent for the workspace.
                            Agent is rebuilt only when provider or model changes
                            (api_routes.py sets self._agent = None on config patch).
    """

    def __init__(self, config):
        from local_search_agent.core.framework import SearchAgentFramework
        from local_search_agent.ui.store import UIStore
        from local_search_agent.workspace.metadata_db import MetadataDB
        from local_search_agent.workspace.workspace_manager import WorkspaceManager

        self.config = config

        self.workspace_manager = WorkspaceManager(db_path=config.db_path)
        self._metadata_db = MetadataDB(db_path=config.db_path)

        # Share the workspace_manager's lock so all SQLite writes are serialised
        self.store = UIStore(
            db_path=config.db_path,
            lock=getattr(self.workspace_manager, "_lock", None),
        )

        self.framework = SearchAgentFramework(config)
        self.framework.start_file_server(port=config.file_server_port)
        self.framework._ensure_meilisearch()

        self.scheduler: Optional[object] = None  # set by start_scheduler()

        self._agent = None
        self._agent_workspace: Optional[str] = None

    # ------------------------------------------------------------------
    # Agent
    # ------------------------------------------------------------------

    def get_agent(self, workspace: Optional[str] = None):
        """Return a LocalSearchAgent for the given workspace, building if needed."""
        from local_search_agent.agent.agent import LocalSearchAgent
        from local_search_agent.search.meilisearch_client import MeilisearchClient

        target_workspace = workspace or self.config.workspace_name

        # Rebuild if workspace changed or config was patched
        if self._agent is None or self._agent_workspace != target_workspace:
            ws_config = self._config_for_workspace(target_workspace)
            mc = MeilisearchClient(
                url=ws_config.meilisearch_url,
                api_key=ws_config.meili_master_key,
                index_name=ws_config.index_name or target_workspace,
            )
            self._agent = LocalSearchAgent(
                config=ws_config,
                meili_client=mc,
                workspace_manager=self.workspace_manager,
            )
            self._agent_workspace = target_workspace
            logger.info("Agent built for workspace %r", target_workspace)

        return self._agent

    def _config_for_workspace(self, workspace: str):
        """Return a SearchAgentConfig scoped to a specific workspace."""
        from local_search_agent.core.config import SearchAgentConfig

        ws = self.workspace_manager.get_workspace(workspace)
        document_dirs = []
        if ws:
            # DB stores document_dir (singular); handle both shapes defensively.
            document_dirs = ws.get("document_dirs") or [ws.get("document_dir", "")]
            document_dirs = [d for d in document_dirs if d]  # drop empty strings

        return SearchAgentConfig(
            workspace_name=workspace,
            document_dirs=document_dirs,
            meilisearch_url=self.config.meilisearch_url,
            meili_master_key=self.config.meili_master_key,
            provider=self.config.provider,
            api_key=self.config.api_key,
            model_name=self.config.model_name,
            host=self.config.host,
            port=self.config.port,
            file_server_port=self.config.file_server_port,
            top_k=self.config.top_k,
            max_iterations=self.config.max_iterations,
            max_retries=self.config.max_retries,
            db_path=self.config.db_path,
        )

    # ------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------

    def start_scheduler(self, interval_minutes: int = 15) -> None:
        from local_search_agent.scheduler.incremental_sync import IncrementalSyncScheduler

        self.scheduler = IncrementalSyncScheduler(
            workspace_manager=self.workspace_manager,
            metadata_db=self._metadata_db,
            interval_minutes=interval_minutes,
        )
        self.scheduler.start()
        logger.info("Scheduler started (interval=%dm)", interval_minutes)


# ---------------------------------------------------------------------------
# FastAPI app builder
# ---------------------------------------------------------------------------


def build_dashboard_app(app_state: AppState):
    """
    Build the FastAPI app that serves the dashboard API + Jinja2-rendered frontend.

    The frontend is split into component templates under ui/templates/.
    index.html uses Jinja2 {% include %} tags to stitch them together.
    The port is injected as a template variable so _script.html can build
    the correct API base URL without hardcoding it.
    """
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles
    from jinja2 import Environment, FileSystemLoader

    from local_search_agent.ui.api_routes import build_ui_router

    app = FastAPI(
        title="Local Search Agent — Dashboard",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount UI API routes
    ui_router = build_ui_router(app_state)
    app.include_router(ui_router)

    # Serve static assets (icons, logos)
    assets_dir = os.path.join(os.path.dirname(__file__), "assets")
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    # Jinja2 environment pointing at ui/templates/
    templates_dir = os.path.join(os.path.dirname(__file__), "templates")
    jinja_env = Environment(
        loader=FileSystemLoader(templates_dir),
        autoescape=False,  # HTML templates — autoescape would mangle SVG/JS
    )

    @app.get("/")
    async def serve_index():
        template = jinja_env.get_template("index.html")
        html = template.render(
            port=app_state.config.port,
            file_server_port=app_state.config.file_server_port,
            version=__version__,
        )
        return HTMLResponse(content=html)

    # Health check for the pywebview ready-poll
    @app.get("/ping")
    async def ping():
        return {"ok": True}

    return app


# ---------------------------------------------------------------------------
# Server thread
# ---------------------------------------------------------------------------
def _wait_for_server(host: str, port: int, timeout: float = 15.0) -> bool:
    """Block until the server accepts TCP connections or timeout elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def _start_server_thread(app, host: str, port: int) -> threading.Thread:
    """Start uvicorn in a daemon thread and return it."""
    import uvicorn

    def _run():
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="warning",  # keep console clean; dashboard has its own logs
            access_log=False,  # suppress per-request access lines (e.g. health poll spam)
        )

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


# ---------------------------------------------------------------------------
# pywebview window
# ---------------------------------------------------------------------------


def _open_window(url: str) -> None:
    """Open the pywebview window.  Blocks until the window is closed."""
    import webview  # type: ignore

    def _on_loaded():
        logger.debug("pywebview window loaded")

    window = webview.create_window(
        title="Local Search Agent",
        url=url,
        width=1280,
        height=800,
        min_size=(900, 600),
        resizable=True,
        on_top=False,
        background_color="#0f1117",  # matches the dark theme background
    )
    window.events.loaded += _on_loaded

    webview.start(
        debug=os.environ.get("LSA_DEBUG_WEBVIEW", "").lower() in ("1", "true"),
    )


# ---------------------------------------------------------------------------
# Folder dialog helper — called from JS via pywebview's JS API
# ---------------------------------------------------------------------------


class _JSBridge:
    """
    Exposed to the frontend as window.pywebview.api

    Methods here are callable from JavaScript via:
        window.pywebview.api.pick_folder().then(path => ...)
    """

    def restart_with_db(self, new_db_path: str) -> None:
        """
        Save new_db_path to settings.json, spawn a fresh `local-search ui` process
        with that path, then exit the current process.
        Called from JS: window.pywebview.api.restart_with_db(path)
        """
        import subprocess
        import sys

        from local_search_agent.core.key_manager import set_saved_db_path

        set_saved_db_path(new_db_path.strip() or None)

        # Build the restart command — same executable, same args minus any old --db
        cmd = [sys.executable, "-m", "local_search_agent.ui.dashboard"]
        if new_db_path.strip():
            cmd += ["--db", new_db_path.strip()]

        logger.info("Restarting UI with db_path=%r", new_db_path)
        # Spawn a short-lived helper that waits 2s for this process to exit
        # and release port 8765 before starting the new UI process.
        delayed_cmd = [
            sys.executable,
            "-c",
            f"import time; time.sleep(2); import subprocess; subprocess.Popen({cmd!r})",
        ]
        subprocess.Popen(delayed_cmd, close_fds=True)

        # Give the HTTP response a moment to reach the browser, then exit
        import threading

        threading.Timer(0.8, lambda: os._exit(0)).start()

    def pick_folder(self) -> Optional[str]:
        """Open the native OS folder picker and return the selected path."""
        import webview

        try:
            dialog_type = webview.FileDialog.FOLDER
        except AttributeError:
            dialog_type = webview.FOLDER_DIALOG
        result = webview.windows[0].create_file_dialog(
            dialog_type=dialog_type,
        )
        if result:
            return result[0]
        return None

    def open_url(self, url: str) -> None:
        """Open a URL or local file path using the OS default application."""
        import subprocess
        import sys

        try:
            if sys.platform == "win32":
                # Convert http://localhost:8000/docs/<id> → actual file path if needed
                # For web URLs just open in the default browser
                subprocess.Popen(["cmd", "/c", "start", "", url], shell=False)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", url])
            else:
                subprocess.Popen(["xdg-open", url])
        except Exception as e:
            logger.warning("open_url failed for %r: %s", url, e)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run(
    host: str = "127.0.0.1",
    port: int = 8765,
    db_path: str = None,
    provider: str = "google",
    model: str = "gemma-4-31b-it",
    meili_url: str = "http://localhost:7700",
    meili_key: str = "local_search_master_key",
    scheduler_interval: int = 0,  # 0 = don't start scheduler
    open_window: bool = True,
    file_server_port: int = 8000,
) -> None:
    from local_search_agent.core.config import SearchAgentConfig, _default_db_path
    from local_search_agent.core.key_manager import get_saved_db_path

    if db_path is None:
        db_path = get_saved_db_path() or _default_db_path()

    config = SearchAgentConfig(
        workspace_name="default",
        document_dirs=[],
        meilisearch_url=meili_url,
        meili_master_key=meili_key,
        provider=provider,
        model_name=model,
        host=host,
        port=port,
        file_server_port=file_server_port,
        db_path=db_path,
    )

    # Restore persisted provider/model from ui_config if available
    from local_search_agent.ui.store import UIStore

    store = UIStore(db_path=db_path)
    saved_provider = store.get_config("global.provider")
    saved_model = store.get_config("global.model")
    if saved_provider:
        config.provider = saved_provider
    if saved_model:
        config.model_name = saved_model
    # Re-read API key after possible provider change
    # Semantic settings are loaded automatically in __post_init__ from settings.json
    config.__post_init__()

    app_state = AppState(config)

    if scheduler_interval > 0:
        app_state.start_scheduler(interval_minutes=scheduler_interval)

    app = build_dashboard_app(app_state)

    logger.info("Starting dashboard server on http://%s:%d", host, port)
    _start_server_thread(app, host, port)

    if not _wait_for_server(host, port, timeout=15.0):
        raise RuntimeError(f"Dashboard server did not start within 15 s on {host}:{port}")
    logger.info("Dashboard server ready.")

    url = f"http://{host}:{port}"

    if open_window:
        # Register the JS bridge so JS can call pick_folder()
        bridge = _JSBridge()
        # pywebview's js_api is set at window-creation time in _open_window
        # We patch _open_window inline to pass js_api
        _open_window_with_bridge(url, bridge)
    else:
        # Headless mode (e.g. CI or tests that only need the API)
        logger.info("Headless mode — window not opened.")
        # Keep process alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


def _open_window_with_bridge(url: str, bridge: _JSBridge) -> None:
    """Open pywebview window with JS bridge attached."""
    import webview  # type: ignore

    _window = webview.create_window(
        title="Local Agent Search Engine",
        url=url,
        js_api=bridge,
        width=1440,
        height=860,
        min_size=(1100, 650),
        resizable=True,
        background_color="#0f1117",
    )

    webview.start(
        debug=os.environ.get("LSA_DEBUG_WEBVIEW", "").lower() in ("1", "true"),
    )


# ---------------------------------------------------------------------------
# __main__ entry point
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="local-search ui",
        description="Open the Local Search Agent desktop dashboard.",
    )
    p.add_argument("--host", default=os.environ.get("LSA_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("LSA_PORT", "8765")))
    p.add_argument(
        "--db",
        default=os.environ.get("LSA_DB_PATH") or None,  # None → _default_db_path() in run()
    )
    p.add_argument(
        "--provider",
        default=os.environ.get("LSA_PROVIDER", "google"),
        choices=["google", "ollama", "openai", "anthropic"],
    )
    p.add_argument("--model", default=os.environ.get("LSA_MODEL", "gemma-4-31b-it"))
    p.add_argument("--meili-url", default=os.environ.get("MEILI_URL", "http://localhost:7700"))
    p.add_argument(
        "--meili-key", default=os.environ.get("MEILI_MASTER_KEY", "local_search_master_key")
    )
    p.add_argument(
        "--scheduler-interval",
        type=int,
        default=0,
        help="Start scheduler with this interval in minutes (0 = disabled).",
    )
    p.add_argument(
        "--headless", action="store_true", help="Run API server only, no window (for debugging)."
    )
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run(
        host=args.host,
        port=args.port,
        db_path=args.db,
        provider=args.provider,
        model=args.model,
        meili_url=args.meili_url,
        meili_key=args.meili_key,
        scheduler_interval=args.scheduler_interval,
        open_window=not args.headless,
    )
