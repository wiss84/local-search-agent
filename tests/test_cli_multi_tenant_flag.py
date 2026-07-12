"""
Tests for the `local-search ui --multi-tenant` CLI flag wiring.

This reaches through
the real launch path (`local-search ui`) instead of only through the
Python API.

`run()` starts a live uvicorn server + Meilisearch connection when called
normally, so these tests mock the heavy pieces (AppState, build_dashboard_app,
server thread, readiness poll) and assert on how they're *called* — the
actual server behavior (AuthorizationMiddleware, whoami, sessions) is
already covered by test_authorization_middleware.py, test_session_flow.py,
etc. This file only proves the CLI flag reaches SearchAgentConfig.identity_provider
correctly.
"""

from __future__ import annotations

from io import StringIO
from unittest.mock import MagicMock, patch


def _run_cli(args: list[str]) -> tuple[int, str, str]:
    from local_search_agent.cli.commands import main

    stdout_buf, stderr_buf = StringIO(), StringIO()
    exit_code = 0
    with (
        patch("sys.argv", ["local-search"] + args),
        patch("sys.stdout", stdout_buf),
        patch("sys.stderr", stderr_buf),
    ):
        try:
            main()
        except SystemExit as e:
            exit_code = e.code if isinstance(e.code, int) else 1
    return exit_code, stdout_buf.getvalue(), stderr_buf.getvalue()


class TestCLIFlagWiring:
    def test_multi_tenant_flag_reaches_run(self, tmp_path):
        db = str(tmp_path / "test.db")
        with patch("local_search_agent.ui.dashboard.run") as mock_run:
            _run_cli(["--db", db, "ui", "--headless", "--multi-tenant"])
        mock_run.assert_called_once()
        assert mock_run.call_args.kwargs["multi_tenant"] is True
        assert mock_run.call_args.kwargs["db_path"] == db

    def test_no_flag_defaults_to_false(self, tmp_path):
        db = str(tmp_path / "test.db")
        with patch("local_search_agent.ui.dashboard.run") as mock_run:
            _run_cli(["--db", db, "ui", "--headless"])
        assert mock_run.call_args.kwargs["multi_tenant"] is False


class TestRunSetsIdentityProvider:
    def test_multi_tenant_true_sets_identity_provider(self, tmp_path):
        from local_search_agent.auth.api_key_provider import APIKeyIdentityProvider
        from local_search_agent.ui import dashboard

        db_path = str(tmp_path / "test.db")
        captured_config = {}

        def fake_app_state(config):
            captured_config["config"] = config
            return MagicMock()

        with (
            patch.object(dashboard, "AppState", side_effect=fake_app_state),
            patch.object(dashboard, "build_dashboard_app", return_value=MagicMock()),
            patch.object(dashboard, "_start_server_thread"),
            patch.object(dashboard, "_wait_for_server", return_value=True),
            patch.object(dashboard.time, "sleep", side_effect=KeyboardInterrupt),
        ):
            dashboard.run(
                db_path=db_path,
                open_window=False,
                multi_tenant=True,
                # Avoid the headless infinite loop — patch time.sleep to raise
                # KeyboardInterrupt on first call so run() returns immediately.
            )

        config = captured_config["config"]
        assert isinstance(config.identity_provider, APIKeyIdentityProvider)

    def test_multi_tenant_false_leaves_identity_provider_none(self, tmp_path):
        from local_search_agent.ui import dashboard

        db_path = str(tmp_path / "test.db")
        captured_config = {}

        def fake_app_state(config):
            captured_config["config"] = config
            return MagicMock()

        with (
            patch.object(dashboard, "AppState", side_effect=fake_app_state),
            patch.object(dashboard, "build_dashboard_app", return_value=MagicMock()),
            patch.object(dashboard, "_start_server_thread"),
            patch.object(dashboard, "_wait_for_server", return_value=True),
            patch.object(dashboard.time, "sleep", side_effect=KeyboardInterrupt),
        ):
            dashboard.run(db_path=db_path, open_window=False, multi_tenant=False)

        assert captured_config["config"].identity_provider is None

    def test_identity_provider_uses_same_db_path(self, tmp_path):
        from local_search_agent.ui import dashboard

        db_path = str(tmp_path / "test.db")
        captured_config = {}

        def fake_app_state(config):
            captured_config["config"] = config
            return MagicMock()

        with (
            patch.object(dashboard, "AppState", side_effect=fake_app_state),
            patch.object(dashboard, "build_dashboard_app", return_value=MagicMock()),
            patch.object(dashboard, "_start_server_thread"),
            patch.object(dashboard, "_wait_for_server", return_value=True),
            patch.object(dashboard.time, "sleep", side_effect=KeyboardInterrupt),
        ):
            dashboard.run(db_path=db_path, open_window=False, multi_tenant=True)

        provider = captured_config["config"].identity_provider
        # The provider's AuthDB must point at the exact same db_path the
        # rest of the app uses — this is what lets an existing single-user
        # install's workspaces/documents be reused unchanged when opting
        # into multi-tenant mode (the whole point of building this instead
        # of a separate throwaway test database/script).
        assert provider._auth_db._db_path == db_path


# ---------------------------------------------------------------------------
# Regression: GET / must actually serve HTML, not a validation error
# ---------------------------------------------------------------------------
# A real bug shipped here: `from __future__ import annotations` in
# dashboard.py makes all annotations lazy strings. `Request` was imported
# *inside* build_dashboard_app() (a local/nested import) rather than at
# module level, so when FastAPI tried to resolve serve_index's
# `request: Request` annotation via the function's __globals__, it
# couldn't find `Request` there and silently treated it as a plain
# required query parameter named "request" instead of the special
# injected type — GET / returned a 422 validation error
# ({"detail":[{"loc":["query","request"],"msg":"Field required"...}]})
# instead of the app shell, in BOTH single-user and multi-tenant mode
# (the bug is at route-registration time, independent of identity_provider).
# Caught by hand (blank pywebview window / raw JSON in browser), not by
# any earlier test, because no earlier test exercised GET / through a
# real TestClient against the actual build_dashboard_app().


class TestServeIndexRouteSignature:
    def _minimal_app_state(self, tmp_path, identity_provider=None):
        from types import SimpleNamespace

        from local_search_agent.core.config import SearchAgentConfig

        config = SearchAgentConfig(
            workspace_name="default",
            document_dirs=[],
            db_path=str(tmp_path / "test.db"),
            port=8765,
            file_server_port=8000,
        )
        config.identity_provider = identity_provider
        # build_dashboard_app only touches app_state.config directly at
        # module scope; build_ui_router(app_state) is mocked out below so
        # the rest of AppState's real attributes (framework, workspace_manager,
        # etc.) never need to exist for this test.
        return SimpleNamespace(config=config)

    def test_get_root_single_user_mode_returns_html_not_422(self, tmp_path):
        from fastapi import APIRouter

        from local_search_agent.ui import dashboard

        app_state = self._minimal_app_state(tmp_path, identity_provider=None)
        with patch("local_search_agent.ui.api_routes.build_ui_router", return_value=APIRouter()):
            app = dashboard.build_dashboard_app(app_state)

        from fastapi.testclient import TestClient

        client = TestClient(app)
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.status_code != 422
        assert "<html" in resp.text.lower()

    def test_get_root_multi_tenant_no_session_redirects_not_422(self, tmp_path):
        from fastapi import APIRouter

        from local_search_agent.auth.api_key_provider import APIKeyIdentityProvider
        from local_search_agent.ui import dashboard
        from local_search_agent.ui.store import UIStore
        from local_search_agent.workspace.auth_db import AuthDB

        db_path = str(tmp_path / "test.db")
        provider = APIKeyIdentityProvider(AuthDB(db_path=db_path))
        app_state = self._minimal_app_state(tmp_path, identity_provider=provider)
        app_state.config.db_path = db_path
        app_state.auth_db = AuthDB(db_path=db_path)
        app_state.store = UIStore(db_path=db_path)

        with patch("local_search_agent.ui.api_routes.build_ui_router", return_value=APIRouter()):
            app = dashboard.build_dashboard_app(app_state)

        from fastapi.testclient import TestClient

        client = TestClient(app, follow_redirects=False)
        resp = client.get("/")
        # No session cookie -> redirect to /login, never a 422.
        assert resp.status_code in (302, 307)
        assert resp.status_code != 422
