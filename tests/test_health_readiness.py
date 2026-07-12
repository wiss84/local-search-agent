"""
Tests for (production deployment infra): GET /health/ready.

See docs/production-deployment.md's "Health checks"
section and server/fastapi_app.py's _check_meilisearch_reachable /
health_ready docstrings for why this is a separate endpoint from the
existing GET /health liveness probe, not a change to it.

No real Meilisearch process is started here -- the "reachable"/"degraded"
cases are exercised by monkeypatching httpx.AsyncClient; the "unreachable"
case uses a real (but never-listening) loopback port so the underlying
httpx connection failure is genuine, not mocked.
"""

from __future__ import annotations

import httpx as httpx_module
from fastapi.testclient import TestClient

from local_search_agent.core.config import SearchAgentConfig
from local_search_agent.server import fastapi_app as fastapi_app_module
from local_search_agent.server.fastapi_app import build_app
from local_search_agent.workspace.workspace_manager import WorkspaceManager

# Port 1 is a well-known privileged port nothing binds to in test
# environments -- connections to it fail fast (connection refused) rather
# than hanging for the full timeout, which keeps this test deterministic
# and quick without any mocking.
_UNREACHABLE_MEILI_URL = "http://127.0.0.1:1"


def _build_client(tmp_path, meilisearch_url: str = _UNREACHABLE_MEILI_URL) -> TestClient:
    config = SearchAgentConfig(
        workspace_name="test_ws",
        db_path=str(tmp_path / "test.db"),
        provider="ollama",
        meilisearch_url=meilisearch_url,
    )
    wm = WorkspaceManager(db_path=config.db_path)
    app = build_app(config=config, workspace_manager=wm)
    return TestClient(app)


class _FakeAsyncResponse:
    def __init__(self, json_data: dict, status_code: int = 200):
        self._json_data = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx_module.HTTPStatusError(
                f"HTTP {self.status_code}", request=None, response=self
            )

    def json(self):
        return self._json_data


class _FakeAsyncClient:
    """Minimal async-context-manager stand-in for httpx.AsyncClient."""

    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def get(self, url, *args, **kwargs):
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def _patch_async_client(monkeypatch, response) -> None:
    monkeypatch.setattr(
        fastapi_app_module.httpx,
        "AsyncClient",
        lambda *a, **kw: _FakeAsyncClient(response),
    )


# ---------------------------------------------------------------------------
# GET /health -- must remain unaffected (regression guard)
# ---------------------------------------------------------------------------


class TestHealthUnaffected:
    def test_health_still_returns_200_without_meilisearch(self, tmp_path):
        client = _build_client(tmp_path)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_health_does_not_call_meilisearch(self, tmp_path, monkeypatch):
        # If /health ever starts depending on _check_meilisearch_reachable,
        # this fails loudly instead of silently changing liveness semantics
        # for every existing deployment/test that relies on it being
        # unconditional.
        called = {"count": 0}

        async def _boom(*a, **kw):
            called["count"] += 1
            raise AssertionError("GET /health must never call Meilisearch")

        monkeypatch.setattr(fastapi_app_module, "_check_meilisearch_reachable", _boom)
        client = _build_client(tmp_path)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert called["count"] == 0


# ---------------------------------------------------------------------------
# GET /health/ready -- Meilisearch reachable / degraded
# ---------------------------------------------------------------------------


class TestHealthReadyReachable:
    def test_returns_200_when_meilisearch_available(self, tmp_path, monkeypatch):
        _patch_async_client(monkeypatch, _FakeAsyncResponse({"status": "available"}))
        client = _build_client(tmp_path, meilisearch_url="http://fake-meili:7700")
        resp = client.get("/health/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["meilisearch"] is True
        assert "version" in data

    def test_returns_503_when_meilisearch_reports_non_available_status(self, tmp_path, monkeypatch):
        _patch_async_client(monkeypatch, _FakeAsyncResponse({"status": "initializing"}))
        client = _build_client(tmp_path, meilisearch_url="http://fake-meili:7700")
        resp = client.get("/health/ready")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["meilisearch"] is False

    def test_returns_503_when_meilisearch_http_error(self, tmp_path, monkeypatch):
        _patch_async_client(
            monkeypatch, _FakeAsyncResponse({"status": "available"}, status_code=500)
        )
        client = _build_client(tmp_path, meilisearch_url="http://fake-meili:7700")
        resp = client.get("/health/ready")
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# GET /health/ready -- Meilisearch unreachable
# ---------------------------------------------------------------------------


class TestHealthReadyUnreachable:
    def test_returns_503_when_connection_refused(self, tmp_path):
        client = _build_client(tmp_path, meilisearch_url=_UNREACHABLE_MEILI_URL)
        resp = client.get("/health/ready")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["meilisearch"] is False

    def test_returns_503_on_timeout(self, tmp_path, monkeypatch):
        _patch_async_client(monkeypatch, httpx_module.TimeoutException("simulated timeout"))
        client = _build_client(tmp_path, meilisearch_url="http://fake-meili:7700")
        resp = client.get("/health/ready")
        assert resp.status_code == 503

    def test_returns_503_on_malformed_json(self, tmp_path, monkeypatch):
        class _BadJsonResponse(_FakeAsyncResponse):
            def json(self):
                raise ValueError("not json")

        _patch_async_client(monkeypatch, _BadJsonResponse({}))
        client = _build_client(tmp_path, meilisearch_url="http://fake-meili:7700")
        resp = client.get("/health/ready")
        assert resp.status_code == 503

    def test_never_raises_unhandled_exception_as_500(self, tmp_path, monkeypatch):
        # Any unexpected exception type must still resolve to a clean 503,
        # not an unhandled-exception 500 -- fail-closed applies to
        # _check_meilisearch_reachable's own error handling, not just the
        # expected/documented failure modes above.
        _patch_async_client(monkeypatch, RuntimeError("totally unexpected"))
        client = _build_client(tmp_path, meilisearch_url="http://fake-meili:7700")
        resp = client.get("/health/ready")
        assert resp.status_code == 503
