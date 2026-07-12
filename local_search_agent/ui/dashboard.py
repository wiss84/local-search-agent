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

from fastapi import Request

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
    get_agent(workspace, provider=None, model_name=None) — builds/returns
                            a LocalSearchAgent for the given
                            workspace/provider/model combination, caching
                            each combination independently so concurrent
                            requests under different roles/allow-lists
                            (Model/Provider Access Control, Option B) can
                            use different models at once. invalidate_agents()
                            drops every cached agent (api_routes.py calls
                            it on any settings change that affects all of
                            them regardless of provider/model).
    """

    def __init__(self, config):
        from local_search_agent.core.framework import SearchAgentFramework
        from local_search_agent.ui.store import UIStore
        from local_search_agent.workspace.auth_db import AuthDB
        from local_search_agent.workspace.metadata_db import MetadataDB
        from local_search_agent.workspace.workspace_manager import WorkspaceManager

        self.config = config

        self.workspace_manager = WorkspaceManager(db_path=config.db_path)
        self._metadata_db = MetadataDB(db_path=config.db_path)

        # Always constructed (schema is CREATE TABLE IF NOT EXISTS, so this is a
        # no-op cost-wise for single-user installs) so api_routes.py handlers
        # always have somewhere to write activity_log rows via _log_activity()
        # -- writes are themselves gated on request.state.identity being present,
        # which only happens when identity_provider is configured and the route
        # is in ROUTE_POLICIES. Shared with build_dashboard_app()'s
        # AuthorizationMiddleware/admin routers rather than each constructing
        # its own AuthDB against the same db_path.
        self.auth_db = AuthDB(db_path=config.db_path)

        # Share the workspace_manager's lock so all SQLite writes are serialised
        self.store = UIStore(
            db_path=config.db_path,
            lock=getattr(self.workspace_manager, "_lock", None),
        )

        self.framework = SearchAgentFramework(config)
        self.framework.start_file_server(port=config.file_server_port)
        self.framework._ensure_meilisearch()

        self.scheduler: Optional[object] = None  # set by start_scheduler() [deprecated]
        self.watcher: Optional[object] = None  # set by start_watch_mode()

        self._agents: dict[tuple, object] = {}

    # ------------------------------------------------------------------
    # Agent
    # ------------------------------------------------------------------

    def get_agent(
        self,
        workspace: Optional[str] = None,
        meili_api_key: Optional[str] = None,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        """
        Return a LocalSearchAgent for the given workspace/provider/model,
        building and caching it if needed.

        provider, model_name : Optional per-request overrides (Model/
                        Provider Access Control, Option B -- true
                        per-request model selection). None means "use
                        the shared deployment-wide default"
                        (self.config.provider / self.config.model_name),
                        exactly today's behavior for single-user mode and
                        any caller who doesn't specify one. Passing an
                        explicit value lets two concurrent requests --
                        e.g. a member and an admin, or two different
                        admins -- use two different, independently
                        allowed models at the same time without
                        rebuilding or evicting each other's cached agent.
                        Enforcement of *which* provider/model a given role
                        may request happens one layer up, in
                        api_routes.py's /query handler, before this
                        method is ever called -- this method itself has
                        no concept of roles or allow-lists, it just builds
                        whatever agent it's asked for.

        meili_api_key : Optional per-request scoped Meilisearch key (Phase 7,
                        see auth/meili_key_provisioning.py + AuthorizationMiddleware's
                        request.state.meili_key) to use instead of
                        ws_config.meili_master_key. None means "use the
                        service-level master key" -- single-user mode and
                        admin-role multi-tenant requests both pass None.

        Caching: keyed on (workspace, meili_api_key, provider, model_name)
                        -- a dict of independent slots rather than one
                        shared slot, specifically so concurrent
                        different-model requests don't rebuild or evict
                        each other's agent. invalidate_agents() clears the
                        whole dict; called whenever a setting that affects
                        every agent regardless of provider/model changes
                        (config PATCH, semantic/reranking/advanced
                        settings) -- correctness (never silently reusing
                        an agent built under stale settings) takes
                        priority over cache-hit-rate here, same principle
                        this cache already followed before Option B.
        """
        from local_search_agent.agent.agent import LocalSearchAgent
        from local_search_agent.search.meilisearch_client import MeilisearchClient

        target_workspace = workspace or self.config.workspace_name
        effective_provider = provider or self.config.provider
        effective_model_name = model_name or self.config.model_name

        cache_key = (target_workspace, meili_api_key, effective_provider, effective_model_name)
        agent = self._agents.get(cache_key)
        if agent is None:
            ws_config = self._config_for_workspace(
                target_workspace, provider=provider, model_name=model_name
            )
            effective_key = meili_api_key or ws_config.meili_master_key
            mc = MeilisearchClient(
                url=ws_config.meilisearch_url,
                api_key=effective_key,
                index_name=ws_config.index_name or target_workspace,
            )
            agent = LocalSearchAgent(
                config=ws_config,
                meili_client=mc,
                workspace_manager=self.workspace_manager,
            )
            self._agents[cache_key] = agent
            logger.info(
                "Agent built for workspace %r provider=%r model=%r (scoped_key=%s)",
                target_workspace,
                effective_provider,
                effective_model_name,
                bool(meili_api_key),
            )

        return agent

    def invalidate_agents(self) -> None:
        """
        Drop every cached agent, regardless of workspace/provider/model,
        so the next get_agent() call for any of them rebuilds from
        current settings. Called whenever something that affects every
        agent changes -- config PATCH (the shared default provider/model),
        semantic/reranking/advanced settings -- since none of those are
        part of the cache key, and a stale cached agent would otherwise
        silently keep running under the old settings.
        """
        self._agents = {}

    def _config_for_workspace(
        self,
        workspace: str,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        """
        Return a SearchAgentConfig scoped to a specific workspace.

        provider, model_name : Optional overrides applied to the plain
        dict *before* SearchAgentConfig(**d) runs, so __post_init__
        resolves api_key against the overridden provider (not the shared
        deployment-wide one) -- setting them on the already-constructed
        object afterward would be too late, api_key resolution already
        happened by then.
        """
        from dataclasses import asdict

        from local_search_agent.core.config import SearchAgentConfig

        ws = self.workspace_manager.get_workspace(workspace)
        document_dirs = []
        if ws:
            document_dirs = ws.get("document_dirs") or [ws.get("document_dir", "")]
            document_dirs = [d for d in document_dirs if d]

        # asdict() deep-copies every field recursively — identity_provider
        # may hold a non-deepcopy-safe object (e.g. threading.Lock inside
        # AuthDB), so it must be excluded before asdict() runs, then
        # reattached to the new config afterward (it should be shared
        # across per-workspace configs, not deep-copied).
        saved_provider = self.config.identity_provider
        self.config.identity_provider = None
        try:
            d = asdict(self.config)
        finally:
            self.config.identity_provider = saved_provider
        d.pop("api_key", None)  # let __post_init__ re-resolve from keys.json / env
        d.pop("index_name", None)  # let __post_init__ set index_name = workspace_name
        d.pop("identity_provider", None)
        d["workspace_name"] = workspace
        d["document_dirs"] = document_dirs
        if provider is not None:
            d["provider"] = provider
        if model_name is not None:
            d["model_name"] = model_name
        ws_config = SearchAgentConfig(**d)
        ws_config.identity_provider = saved_provider
        return ws_config

    # ------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------

    def start_scheduler(self, interval_minutes: int = 15) -> None:
        """DEPRECATED (polling-based, use start_watch_mode): start the APScheduler-backed scheduler."""
        from local_search_agent.scheduler.incremental_sync import IncrementalSyncScheduler

        self.scheduler = IncrementalSyncScheduler(
            workspace_manager=self.workspace_manager,
            metadata_db=self._metadata_db,
            interval_minutes=interval_minutes,
        )
        self.scheduler.start()
        logger.info("Scheduler started (interval=%dm)", interval_minutes)

    # ------------------------------------------------------------------
    # Watch mode
    # ------------------------------------------------------------------

    def start_watch_mode(self) -> None:
        """Start the watchdog-backed WorkspaceWatcher (filesystem-event-driven sync)."""
        from local_search_agent.scheduler.watch_mode import WorkspaceWatcher

        self.watcher = WorkspaceWatcher(
            workspace_manager=self.workspace_manager,
            metadata_db=self._metadata_db,
        )
        self.watcher.start()
        logger.info("Watch mode started.")


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
    from fastapi.responses import HTMLResponse, RedirectResponse
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

    if app_state.config.identity_provider is not None:
        from local_search_agent.auth.authorization_middleware import AuthorizationMiddleware
        from local_search_agent.auth.grants_routes import build_grants_router

        # Reuse AppState's AuthDB instance rather than constructing a second
        # one against the same db_path -- api_routes.py's _log_activity()
        # writes through app_state.auth_db, so this keeps a single AuthDB
        # object as the source of truth for the whole process.
        shared_auth_db = app_state.auth_db

        app.add_middleware(
            AuthorizationMiddleware,
            config=app_state.config,
            auth_db=shared_auth_db,
            session_lookup=app_state.store.get_session_workspace,
        )
        logger.info("Authorization middleware enabled on dashboard app (multi-tenant RBAC).")

        app.include_router(build_grants_router(shared_auth_db))

        from local_search_agent.auth.api_key_provider import APIKeyIdentityProvider

        if isinstance(app_state.config.identity_provider, APIKeyIdentityProvider):
            from local_search_agent.auth.admin_keys_routes import build_admin_keys_router
            from local_search_agent.auth.session_routes import build_auth_router

            app.include_router(
                build_auth_router(
                    app_state.config.identity_provider,
                    cookie_secure=app_state.config.cookie_secure,
                )
            )
            app.include_router(build_admin_keys_router(app_state.config.identity_provider))
            logger.info("Browser session flow enabled (/api/auth/login, /api/auth/logout).")
    else:
        shared_auth_db = None

    # whoami must ALWAYS be mounted, regardless of identity_provider — it
    # handles the None case internally (returns {"multi_tenant": false}),
    # and the frontend's boot() calls it unconditionally on every launch.
    # Mounting it only inside the block above means single-user desktop
    # installs get a 404 on load, which refreshRoleGating() then
    # (correctly, for the multi-tenant case) treats as fail-closed —
    # silently disabling every gated button for people who never opted
    # into multi-tenant mode at all. Not a hypothetical: this exact bug
    # shipped and was caught by manual click-through, not the test suite.
    from local_search_agent.auth.whoami_route import build_whoami_router

    app.include_router(build_whoami_router(app_state.config, shared_auth_db))

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
    async def serve_index(request: Request):
        from local_search_agent.auth.api_key_provider import (
            SESSION_COOKIE_NAME,
            APIKeyIdentityProvider,
        )

        provider = app_state.config.identity_provider
        if isinstance(provider, APIKeyIdentityProvider):
            token = request.cookies.get(SESSION_COOKIE_NAME)
            if not token or provider.resolve_session(token) is None:
                # No valid session — send them to the login page instead of
                # flashing the full app shell first. `next` round-trips back
                # here (or wherever they originally tried to reach) after login.
                return RedirectResponse(url=f"/login?next={request.url.path}", status_code=307)

        template = jinja_env.get_template("index.html")
        html = template.render(
            port=app_state.config.port,
            file_server_port=app_state.config.file_server_port,
            version=__version__,
        )
        return HTMLResponse(content=html)

    @app.get("/login")
    async def serve_login():
        from local_search_agent.auth.api_key_provider import APIKeyIdentityProvider

        if not isinstance(app_state.config.identity_provider, APIKeyIdentityProvider):
            # Login page only makes sense for APIKeyIdentityProvider —
            # Header/JWT modes never reach this (their SSO/proxy already
            # manages the session before traffic hits the app), and
            # single-user mode has no concept of login at all.
            return RedirectResponse(url="/", status_code=307)
        template = jinja_env.get_template("login.html")
        return HTMLResponse(content=template.render())

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
    multi_tenant: bool = False,
    insecure_cookies: bool = False,
) -> None:
    from local_search_agent.core.config import SearchAgentConfig, _default_db_path
    from local_search_agent.core.key_manager import get_saved_db_path

    if db_path is None:
        db_path = get_saved_db_path() or _default_db_path()

    if insecure_cookies:
        logger.warning(
            "--insecure-cookies is set: the multi-tenant session cookie will be sent "
            "over plain HTTP. Only use this on a trusted local network you control "
            "(e.g. testing multi-tenant mode across two laptops on the same LAN/hotspot "
            "without a TLS-terminating reverse proxy in front). Never use it for anything "
            "reachable from the open internet -- see docs/production-deployment.md instead."
        )

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
        cookie_secure=not insecure_cookies,
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

    if multi_tenant:
        from local_search_agent.auth.api_key_provider import APIKeyIdentityProvider
        from local_search_agent.workspace.auth_db import AuthDB

        auth_db = AuthDB(db_path=db_path)
        config.identity_provider = APIKeyIdentityProvider(auth_db)
        logger.info(
            "Multi-tenant RBAC enabled (APIKeyIdentityProvider) against db_path=%r", db_path
        )

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
    p.add_argument(
        "--multi-tenant",
        action="store_true",
        dest="multi_tenant",
        help="Enable multi-tenant RBAC (APIKeyIdentityProvider) against this same --db.",
    )
    p.add_argument(
        "--insecure-cookies",
        action="store_true",
        dest="insecure_cookies",
        help=(
            "Allow the multi-tenant session cookie over plain HTTP -- needed when "
            "--host is a real LAN IP rather than 127.0.0.1/localhost, since browsers "
            "otherwise silently refuse to store a Secure cookie over non-HTTPS. Only "
            "use this on a trusted local network; never for anything internet-facing "
            "(use a TLS reverse proxy instead, see docs/production-deployment.md)."
        ),
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
        multi_tenant=args.multi_tenant,
        insecure_cookies=args.insecure_cookies,
    )
