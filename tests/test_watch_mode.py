"""
Unit tests for Watch Mode (filesystem-event-driven incremental sync).

Mirrors test_scheduler.py's structure and conventions:
- tmp_path-based MetadataDB / WorkspaceManager fixtures
- The underlying library (watchdog.observers.Observer) is mocked so no real
  filesystem-watching threads are spun up during tests.
- Debounce timing uses a very small debounce_seconds value with real
  threading.Timer, since the delay is short enough not to slow down CI,
  rather than mocking threading.Timer itself.

Tests cover:
- IngestionPipeline.run(enrich=...) parameter behavior
- _DebouncedHandler collapsing bursts of events into one call
- WorkspaceWatcher workspace registration, trigger_now, status, remove
- WorkspaceWatcher._run_sync respecting config.enrich_on_watch
- SearchAgentConfig watch-mode fields and settings persistence
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from local_search_agent.workspace.metadata_db import MetadataDB

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db(tmp_path):
    return MetadataDB(db_path=str(tmp_path / "test.db"))


@pytest.fixture
def wm(tmp_path):
    from local_search_agent.workspace.workspace_manager import WorkspaceManager

    return WorkspaceManager(db_path=str(tmp_path / "test.db"))


def _make_config(tmp_path, workspace="test_ws", enrich_on_watch=True):
    from local_search_agent.core.config import SearchAgentConfig

    return SearchAgentConfig(
        document_dirs=[str(tmp_path)],
        workspace_name=workspace,
        provider="ollama",
        db_path=str(tmp_path / "test.db"),
        enrich_on_watch=enrich_on_watch,
    )


def _make_watcher(wm, db, debounce_seconds=0.05):
    from local_search_agent.scheduler.watch_mode import WorkspaceWatcher

    return WorkspaceWatcher(
        workspace_manager=wm,
        metadata_db=db,
        debounce_seconds=debounce_seconds,
    )


# ---------------------------------------------------------------------------
# SearchAgentConfig — watch-mode fields
# ---------------------------------------------------------------------------


class TestConfigWatchModeFields:
    def test_defaults(self, tmp_path):
        config = _make_config(tmp_path)
        assert config.enrich_on_watch is True
        assert isinstance(config.enable_watch_mode, bool)

    def test_enrich_on_watch_explicit_false(self, tmp_path):
        config = _make_config(tmp_path, enrich_on_watch=False)
        assert config.enrich_on_watch is False

    def test_enable_reranking_default_true(self, tmp_path):
        config = _make_config(tmp_path)
        assert config.enable_reranking is True
        assert config.rerank_candidate_multiplier >= 1


# ---------------------------------------------------------------------------
# key_manager — watch-mode settings persistence
# ---------------------------------------------------------------------------


class TestWatchModeSettingsPersistence:
    def test_get_defaults_when_unset(self, tmp_path, monkeypatch):
        from local_search_agent.core import key_manager

        monkeypatch.setattr(key_manager, "_settings_path", lambda: tmp_path / "settings.json")
        settings = key_manager.get_watch_mode_settings()
        assert settings["enable_watch_mode"] is False
        assert settings["enrich_on_watch"] is True

    def test_set_and_get_roundtrip(self, tmp_path, monkeypatch):
        from local_search_agent.core import key_manager

        monkeypatch.setattr(key_manager, "_settings_path", lambda: tmp_path / "settings.json")
        key_manager.set_all_watch_mode_settings(enable_watch_mode=True, enrich_on_watch=False)
        settings = key_manager.get_watch_mode_settings()
        assert settings["enable_watch_mode"] is True
        assert settings["enrich_on_watch"] is False

    def test_set_does_not_clobber_other_keys(self, tmp_path, monkeypatch):
        from local_search_agent.core import key_manager

        monkeypatch.setattr(key_manager, "_settings_path", lambda: tmp_path / "settings.json")
        key_manager.set_all_semantic_settings(
            enable_semantic=True,
            enable_query_expansion=False,
            semantic_provider="google",
            semantic_model="gemma-4-31b-it",
        )
        key_manager.set_all_watch_mode_settings(enable_watch_mode=True, enrich_on_watch=False)
        sem = key_manager.get_semantic_settings()
        assert sem["enable_semantic"] is True


# ---------------------------------------------------------------------------
# IngestionPipeline.run(enrich=...)
# ---------------------------------------------------------------------------


class TestPipelineEnrichParameter:
    def _make_pipeline(self, tmp_path, enable_semantic=True):
        from local_search_agent.core.config import SearchAgentConfig
        from local_search_agent.ingestion.pipeline import IngestionPipeline
        from local_search_agent.workspace.workspace_manager import WorkspaceManager

        config = SearchAgentConfig(
            document_dirs=[str(tmp_path)],
            workspace_name="test_ws",
            provider="ollama",
            db_path=str(tmp_path / "test.db"),
            enable_semantic=enable_semantic,
        )
        wm = WorkspaceManager(db_path=str(tmp_path / "test.db"))
        mc = MagicMock()
        return IngestionPipeline(config=config, workspace_manager=wm, meili_client=mc)

    def test_enrich_false_skips_enrich_batch(self, tmp_path):
        pipeline = self._make_pipeline(tmp_path, enable_semantic=True)
        fake_file = str(tmp_path / "a.txt")
        with patch.object(pipeline, "_enrich_batch") as mock_enrich:
            with patch.object(pipeline, "_walk", return_value=[fake_file]):
                with patch.object(pipeline, "_parse_file", return_value=[MagicMock()]):
                    with patch.object(pipeline, "_flush_batch"):
                        pipeline.run(enrich=False, force=True)
            mock_enrich.assert_not_called()

    def test_enrich_true_calls_enrich_batch_when_nodes_present(self, tmp_path):
        pipeline = self._make_pipeline(tmp_path, enable_semantic=True)
        fake_file = str(tmp_path / "a.txt")
        with patch.object(pipeline, "_enrich_batch") as mock_enrich:
            with patch.object(pipeline, "_walk", return_value=[fake_file]):
                with patch.object(pipeline, "_parse_file", return_value=[MagicMock()]):
                    with patch.object(pipeline, "_flush_batch"):
                        pipeline.run(enrich=True, force=True)
            mock_enrich.assert_called_once()

    def test_enrich_default_is_true(self, tmp_path):
        import inspect

        from local_search_agent.ingestion.pipeline import IngestionPipeline

        sig = inspect.signature(IngestionPipeline.run)
        assert sig.parameters["enrich"].default is True


# ---------------------------------------------------------------------------
# _DebouncedHandler
# ---------------------------------------------------------------------------


class TestDebouncedHandler:
    def test_single_notify_fires_once(self):
        from local_search_agent.scheduler.watch_mode import _DebouncedHandler

        calls = []
        fired = threading.Event()

        def on_settle(ws):
            calls.append(ws)
            fired.set()

        handler = _DebouncedHandler(workspace="ws1", on_settle=on_settle, debounce_seconds=0.1)
        handler.notify()
        assert fired.wait(timeout=1.0), "on_settle callback did not fire within timeout"
        assert calls == ["ws1"]

    def test_burst_of_notifies_collapses_to_one_call(self):
        from local_search_agent.scheduler.watch_mode import _DebouncedHandler

        calls = []
        fired = threading.Event()

        def on_settle(ws):
            calls.append(ws)
            fired.set()

        handler = _DebouncedHandler(workspace="ws1", on_settle=on_settle, debounce_seconds=0.1)
        for _ in range(10):
            handler.notify()
            time.sleep(0.01)
        assert fired.wait(timeout=1.0), "on_settle callback did not fire within timeout"
        assert calls == ["ws1"]

    def test_cancel_prevents_fire(self):
        from local_search_agent.scheduler.watch_mode import _DebouncedHandler

        calls = []
        fired = threading.Event()

        def on_settle(ws):
            calls.append(ws)
            fired.set()

        handler = _DebouncedHandler(workspace="ws1", on_settle=on_settle, debounce_seconds=0.1)
        handler.notify()
        handler.cancel()
        assert not fired.wait(timeout=1.0), "on_settle callback fired after cancel"
        assert calls == []

    def test_exception_in_callback_is_caught(self):
        from local_search_agent.scheduler.watch_mode import _DebouncedHandler

        def _boom(ws):
            raise RuntimeError("boom")

        handler = _DebouncedHandler(workspace="ws1", on_settle=_boom, debounce_seconds=0.1)
        handler.notify()
        time.sleep(0.4)  # should not raise / crash the test process


# ---------------------------------------------------------------------------
# WorkspaceWatcher
# ---------------------------------------------------------------------------


class TestWorkspaceWatcher:
    def test_start_and_stop(self, db, wm):
        watcher = _make_watcher(wm, db)
        with patch("watchdog.observers.Observer.start"):
            with patch("watchdog.observers.Observer.stop"):
                with patch("watchdog.observers.Observer.join"):
                    watcher.start()
                    assert watcher.is_running
                    watcher.stop()
                    assert not watcher.is_running

    def test_add_workspace_creates_sync_job(self, db, wm, tmp_path):
        watcher = _make_watcher(wm, db)
        config = _make_config(tmp_path, "finance")

        with patch("watchdog.observers.Observer.start"):
            watcher.start()
            with patch("watchdog.observers.Observer.schedule"):
                watcher.add_workspace(config)
                job = db.get_sync_job("finance")
                assert job is not None

    def test_add_workspace_before_start_does_not_raise(self, db, wm, tmp_path):
        watcher = _make_watcher(wm, db)
        config = _make_config(tmp_path, "finance")
        watcher.add_workspace(config)  # watcher not started yet
        assert "finance" in watcher._workspace_configs

    def test_remove_workspace(self, db, wm, tmp_path):
        watcher = _make_watcher(wm, db)
        config = _make_config(tmp_path, "finance")

        with patch("watchdog.observers.Observer.start"):
            watcher.start()
            with patch("watchdog.observers.Observer.schedule"):
                watcher.add_workspace(config)
            watcher.remove_workspace("finance")
            assert "finance" not in watcher._workspace_configs
            assert "finance" not in watcher._debouncers

    def test_trigger_now_calls_run_sync(self, db, wm, tmp_path):
        watcher = _make_watcher(wm, db)
        config = _make_config(tmp_path, "finance")
        watcher._workspace_configs["finance"] = config
        watcher._debouncers["finance"] = MagicMock()

        with patch.object(watcher, "_run_sync") as mock_run:
            watcher.trigger_now("finance")
            mock_run.assert_called_once_with("finance")

    def test_trigger_now_unknown_workspace_raises(self, db, wm):
        watcher = _make_watcher(wm, db)
        with pytest.raises(ValueError, match="not registered"):
            watcher.trigger_now("nonexistent")

    def test_get_status_not_running(self, db, wm):
        watcher = _make_watcher(wm, db)
        status = watcher.get_status()
        assert status["running"] is False
        assert status["registered_workspaces"] == []

    def test_get_status_reports_registered_workspaces(self, db, wm, tmp_path):
        watcher = _make_watcher(wm, db)
        config = _make_config(tmp_path, "finance")
        with patch("watchdog.observers.Observer.start"):
            watcher.start()
            with patch("watchdog.observers.Observer.schedule"):
                watcher.add_workspace(config)
        status = watcher.get_status()
        assert "finance" in status["registered_workspaces"]


# ---------------------------------------------------------------------------
# WorkspaceWatcher._run_sync — enrich_on_watch behavior
# ---------------------------------------------------------------------------


class TestRunSyncEnrichBehavior:
    def test_run_sync_passes_enrich_true_when_enrich_on_watch_true(self, db, wm, tmp_path):
        watcher = _make_watcher(wm, db)
        config = _make_config(tmp_path, "finance", enrich_on_watch=True)
        watcher._workspace_configs["finance"] = config

        mock_stats = MagicMock(indexed=1, skipped=0, failed=0, errors=[])
        with patch("local_search_agent.search.meilisearch_client.MeilisearchClient"):
            with patch("local_search_agent.ingestion.pipeline.IngestionPipeline") as MockPipeline:
                MockPipeline.return_value.run.return_value = mock_stats
                watcher._run_sync("finance")
                _, kwargs = MockPipeline.return_value.run.call_args
                assert kwargs["enrich"] is True
                assert kwargs["force"] is False

    def test_run_sync_passes_enrich_false_when_enrich_on_watch_false(self, db, wm, tmp_path):
        watcher = _make_watcher(wm, db)
        config = _make_config(tmp_path, "finance", enrich_on_watch=False)
        watcher._workspace_configs["finance"] = config

        mock_stats = MagicMock(indexed=1, skipped=0, failed=0, errors=[])
        with patch("local_search_agent.search.meilisearch_client.MeilisearchClient"):
            with patch("local_search_agent.ingestion.pipeline.IngestionPipeline") as MockPipeline:
                MockPipeline.return_value.run.return_value = mock_stats
                watcher._run_sync("finance")
                _, kwargs = MockPipeline.return_value.run.call_args
                assert kwargs["enrich"] is False

    def test_run_sync_unregistered_workspace_is_noop(self, db, wm):
        watcher = _make_watcher(wm, db)
        # Should log a warning and return, not raise.
        watcher._run_sync("nonexistent")

    def test_run_sync_records_history_on_success(self, db, wm, tmp_path):
        watcher = _make_watcher(wm, db)
        config = _make_config(tmp_path, "finance")
        watcher._workspace_configs["finance"] = config
        db.upsert_sync_job(workspace="finance", next_sync_at=None)

        mock_stats = MagicMock(indexed=3, skipped=1, failed=0, errors=[])
        with patch("local_search_agent.search.meilisearch_client.MeilisearchClient"):
            with patch("local_search_agent.ingestion.pipeline.IngestionPipeline") as MockPipeline:
                MockPipeline.return_value.run.return_value = mock_stats
                watcher._run_sync("finance")

        job = db.get_sync_job("finance")
        assert job["sync_status"] == "idle"
        assert job["doc_count"] == 4  # indexed + skipped

    def test_run_sync_records_error_on_exception(self, db, wm, tmp_path):
        watcher = _make_watcher(wm, db)
        config = _make_config(tmp_path, "finance")
        watcher._workspace_configs["finance"] = config
        db.upsert_sync_job(workspace="finance", next_sync_at=None)

        with patch(
            "local_search_agent.search.meilisearch_client.MeilisearchClient",
            side_effect=RuntimeError("connection refused"),
        ):
            watcher._run_sync("finance")

        job = db.get_sync_job("finance")
        assert job["sync_status"] == "error"
        assert "connection refused" in job["last_error"]
