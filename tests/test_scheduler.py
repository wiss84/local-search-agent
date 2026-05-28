"""
Unit tests for Phase 4: scheduler, monitor, and MetadataDB.

All APScheduler calls are mocked — no real timers or threads needed.
Tests cover:
- MetadataDB schema init, sync_job CRUD, sync_history, staleness query
- IndexMonitor health classification logic
- IncrementalSyncScheduler workspace registration, trigger_now, status
- FastAPI /health/indexes and /workspaces/{name}/history endpoints
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from local_search_agent.scheduler.monitor import IndexMonitor
from local_search_agent.workspace.metadata_db import MetadataDB

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db(tmp_path):
    return MetadataDB(db_path=str(tmp_path / "test.db"))


@pytest.fixture
def monitor(db):
    return IndexMonitor(metadata_db=db, stale_threshold_minutes=30)


# ---------------------------------------------------------------------------
# MetadataDB — sync_jobs
# ---------------------------------------------------------------------------


class TestMetadataDBSyncJobs:
    def test_upsert_creates_new_job(self, db):
        db.upsert_sync_job("finance")
        job = db.get_sync_job("finance")
        assert job is not None
        assert job["workspace"] == "finance"
        assert job["sync_status"] == "idle"

    def test_upsert_is_idempotent(self, db):
        db.upsert_sync_job("finance")
        db.upsert_sync_job("finance")
        jobs = db.list_sync_jobs()
        assert len([j for j in jobs if j["workspace"] == "finance"]) == 1

    def test_upsert_sets_next_sync(self, db):
        ts = "2026-01-01T12:00:00+00:00"
        db.upsert_sync_job("hr", next_sync_at=ts)
        job = db.get_sync_job("hr")
        assert job["next_sync_at"] == ts

    def test_set_sync_running(self, db):
        db.upsert_sync_job("finance")
        db.set_sync_running("finance")
        job = db.get_sync_job("finance")
        assert job["sync_status"] == "running"

    def test_set_sync_complete_idle(self, db):
        db.upsert_sync_job("finance")
        db.set_sync_complete(
            workspace="finance",
            doc_count=100,
            error_count=0,
            next_sync_at="2026-01-01T13:00:00+00:00",
        )
        job = db.get_sync_job("finance")
        assert job["sync_status"] == "idle"
        assert job["doc_count"] == 100
        assert job["last_sync_at"] is not None

    def test_set_sync_complete_error_when_failures(self, db):
        db.upsert_sync_job("finance")
        db.set_sync_complete(
            workspace="finance",
            doc_count=95,
            error_count=5,
            next_sync_at="2026-01-01T13:00:00+00:00",
            last_error="Parser failed for report.pdf",
        )
        job = db.get_sync_job("finance")
        assert job["sync_status"] == "error"
        assert job["last_error"] == "Parser failed for report.pdf"

    def test_get_nonexistent_returns_none(self, db):
        assert db.get_sync_job("no_such_ws") is None

    def test_list_sync_jobs_returns_all(self, db):
        for name in ["alpha", "beta", "gamma"]:
            db.upsert_sync_job(name)
        jobs = db.list_sync_jobs()
        names = {j["workspace"] for j in jobs}
        assert {"alpha", "beta", "gamma"} == names


# ---------------------------------------------------------------------------
# MetadataDB — sync_history
# ---------------------------------------------------------------------------


class TestMetadataDBSyncHistory:
    def test_record_start_returns_id(self, db):
        hid = db.record_sync_start("finance")
        assert isinstance(hid, int)
        assert hid > 0

    def test_record_finish_updates_row(self, db):
        hid = db.record_sync_start("finance")
        db.record_sync_finish(
            history_id=hid,
            indexed=50,
            skipped=30,
            failed=2,
            duration_s=4.5,
            errors=["some error"],
        )
        history = db.get_sync_history("finance", limit=5)
        assert len(history) == 1
        record = history[0]
        assert record["indexed"] == 50
        assert record["skipped"] == 30
        assert record["failed"] == 2
        assert abs(record["duration_s"] - 4.5) < 0.01
        assert "some error" in record["errors"]
        assert record["finished_at"] is not None

    def test_history_limit_respected(self, db):
        for _ in range(10):
            hid = db.record_sync_start("finance")
            db.record_sync_finish(hid, 1, 0, 0, 0.1, [])
        history = db.get_sync_history("finance", limit=5)
        assert len(history) == 5

    def test_history_ordered_most_recent_first(self, db):
        ids = []
        for _ in range(3):
            hid = db.record_sync_start("finance")
            db.record_sync_finish(hid, 1, 0, 0, 0.1, [])
            ids.append(hid)
        history = db.get_sync_history("finance")
        returned_ids = [r["id"] for r in history]
        assert returned_ids == sorted(returned_ids, reverse=True)


# ---------------------------------------------------------------------------
# MetadataDB — staleness
# ---------------------------------------------------------------------------


class TestMetadataDBStaleness:
    def test_never_synced_workspace_is_stale(self, db):
        db.upsert_sync_job("finance")
        stale = db.get_stale_workspaces(older_than_minutes=30)
        names = [s["workspace"] for s in stale]
        assert "finance" in names

    def test_recently_synced_workspace_not_stale(self, db):
        db.upsert_sync_job("finance")
        # Set last_sync_at to 1 minute ago
        recent = (datetime.now().astimezone() - timedelta(minutes=1)).isoformat()
        db.set_sync_complete("finance", 100, 0, recent)
        # Override last_sync_at directly by completing it "now"
        stale = db.get_stale_workspaces(older_than_minutes=30)
        names = [s["workspace"] for s in stale]
        assert "finance" not in names

    def test_old_sync_is_stale(self, db):
        db.upsert_sync_job("old_ws")
        # Manually insert an old last_sync_at
        old_ts = (datetime.now().astimezone() - timedelta(hours=2)).isoformat()
        with db._connect() as conn:
            conn.execute(
                "UPDATE sync_jobs SET last_sync_at=? WHERE workspace=?",
                (old_ts, "old_ws"),
            )
        stale = db.get_stale_workspaces(older_than_minutes=30)
        names = [s["workspace"] for s in stale]
        assert "old_ws" in names


# ---------------------------------------------------------------------------
# IndexMonitor
# ---------------------------------------------------------------------------


class TestIndexMonitor:
    def test_never_synced_status(self, db, monitor):
        db.upsert_sync_job("finance")
        health = monitor.get_workspace_health("finance")
        assert health.status == "never_synced"
        assert health.age_minutes is None

    def test_healthy_status_after_recent_sync(self, db, monitor):
        db.upsert_sync_job("finance")
        db.set_sync_complete("finance", 100, 0, "2099-01-01T00:00:00+00:00")
        health = monitor.get_workspace_health("finance")
        assert health.status == "healthy"
        assert health.doc_count == 100

    def test_stale_status_after_old_sync(self, db, monitor):
        db.upsert_sync_job("finance")
        old_ts = (datetime.now().astimezone() - timedelta(hours=2)).isoformat()
        with db._connect() as conn:
            conn.execute(
                "UPDATE sync_jobs SET last_sync_at=?, sync_status='idle' WHERE workspace=?",
                (old_ts, "finance"),
            )
        health = monitor.get_workspace_health("finance")
        assert health.status == "stale"
        assert health.age_minutes > 30

    def test_running_status(self, db, monitor):
        db.upsert_sync_job("finance")
        db.set_sync_running("finance")
        health = monitor.get_workspace_health("finance")
        assert health.status == "running"

    def test_error_status(self, db, monitor):
        db.upsert_sync_job("finance")
        db.set_sync_complete("finance", 0, 5, "2099-01-01T00:00:00+00:00", "parse failed")
        # set status to error directly
        with db._connect() as conn:
            conn.execute("UPDATE sync_jobs SET sync_status='error' WHERE workspace=?", ("finance",))
        health = monitor.get_workspace_health("finance")
        assert health.status == "error"

    def test_unknown_workspace_returns_none(self, db, monitor):
        assert monitor.get_workspace_health("nonexistent") is None

    def test_health_summary_aggregates_correctly(self, db, monitor):
        db.upsert_sync_job("ws1")
        db.upsert_sync_job("ws2")
        db.set_sync_complete("ws2", 50, 0, "2099-01-01T00:00:00+00:00")

        summary = monitor.get_health_summary()
        assert summary.total_workspaces == 2
        assert summary.never_synced == 1
        assert summary.healthy == 1
        assert summary.total_docs == 50

    def test_health_summary_all_healthy_flag(self, db, monitor):
        db.upsert_sync_job("ws1")
        db.set_sync_complete("ws1", 10, 0, "2099-01-01T00:00:00+00:00")
        summary = monitor.get_health_summary()
        assert summary.all_healthy is True

    def test_to_dict_is_serialisable(self, db, monitor):
        db.upsert_sync_job("finance")
        summary = monitor.get_health_summary()
        d = summary.to_dict()
        import json

        json.dumps(d)  # Should not raise


# ---------------------------------------------------------------------------
# IncrementalSyncScheduler
# ---------------------------------------------------------------------------


class TestIncrementalSyncScheduler:
    def _make_scheduler(self, db, tmp_path):
        from local_search_agent.scheduler.incremental_sync import IncrementalSyncScheduler
        from local_search_agent.workspace.workspace_manager import WorkspaceManager

        wm = WorkspaceManager(db_path=str(tmp_path / "test.db"))
        return IncrementalSyncScheduler(
            workspace_manager=wm,
            metadata_db=db,
            interval_minutes=15,
        ), wm

    def _make_config(self, tmp_path, workspace="test_ws"):
        from local_search_agent.core.config import SearchAgentConfig

        return SearchAgentConfig(
            document_dirs=[str(tmp_path)],
            workspace_name=workspace,
            provider="ollama",
            db_path=str(tmp_path / "test.db"),
        )

    def test_start_and_stop(self, db, tmp_path):
        scheduler, _ = self._make_scheduler(db, tmp_path)
        with patch("apscheduler.schedulers.background.BackgroundScheduler.start"):
            with patch("apscheduler.schedulers.background.BackgroundScheduler.shutdown"):
                scheduler.start()
                assert scheduler.is_running
                scheduler.stop()
                assert not scheduler.is_running

    def test_add_workspace_creates_sync_job(self, db, tmp_path):
        scheduler, _ = self._make_scheduler(db, tmp_path)
        config = self._make_config(tmp_path, "finance")

        with patch("apscheduler.schedulers.background.BackgroundScheduler.start"):
            scheduler.start()
            with patch.object(scheduler, "_schedule_workspace_job"):
                scheduler.add_workspace(config)
                job = db.get_sync_job("finance")
                assert job is not None

    def test_remove_workspace(self, db, tmp_path):
        scheduler, _ = self._make_scheduler(db, tmp_path)
        config = self._make_config(tmp_path, "finance")

        with patch("apscheduler.schedulers.background.BackgroundScheduler.start"):
            scheduler.start()
            with patch.object(scheduler, "_schedule_workspace_job"):
                scheduler.add_workspace(config)
            scheduler.remove_workspace("finance")
            assert "finance" not in scheduler._workspace_configs

    def test_trigger_now_calls_run_sync(self, db, tmp_path):
        scheduler, _ = self._make_scheduler(db, tmp_path)
        config = self._make_config(tmp_path, "finance")
        scheduler._workspace_configs["finance"] = config

        with patch.object(scheduler, "_run_sync") as mock_run:
            scheduler.trigger_now("finance")
            mock_run.assert_called_once_with("finance", config)

    def test_trigger_now_unknown_workspace_raises(self, db, tmp_path):
        scheduler, _ = self._make_scheduler(db, tmp_path)
        with pytest.raises(ValueError, match="not registered"):
            scheduler.trigger_now("nonexistent")

    def test_get_status_not_running(self, db, tmp_path):
        scheduler, _ = self._make_scheduler(db, tmp_path)
        status = scheduler.get_status()
        assert status["running"] is False
        assert status["registered_workspaces"] == []


# ---------------------------------------------------------------------------
# FastAPI /health/indexes and /workspaces/{name}/history
# ---------------------------------------------------------------------------


class TestHealthIndexesEndpoint:
    def _make_client(self, tmp_path):
        from fastapi.testclient import TestClient

        from local_search_agent.core.config import SearchAgentConfig
        from local_search_agent.server.fastapi_app import build_app
        from local_search_agent.workspace.metadata_db import MetadataDB
        from local_search_agent.workspace.workspace_manager import WorkspaceManager

        db_path = str(tmp_path / "test.db")
        config = SearchAgentConfig(
            workspace_name="test_ws",
            provider="ollama",
            db_path=db_path,
        )
        wm = WorkspaceManager(db_path=db_path)
        mdb = MetadataDB(db_path=db_path)
        wm.create_workspace("test_ws", str(tmp_path))
        mdb.upsert_sync_job("test_ws")

        app = build_app(config=config, workspace_manager=wm, metadata_db=mdb)
        return TestClient(app), mdb

    def test_health_indexes_returns_200(self, tmp_path):
        client, _ = self._make_client(tmp_path)
        resp = client.get("/health/indexes")
        assert resp.status_code == 200

    def test_health_indexes_contains_workspaces(self, tmp_path):
        client, _ = self._make_client(tmp_path)
        data = client.get("/health/indexes").json()
        assert "total_workspaces" in data
        assert data["total_workspaces"] >= 1

    def test_health_indexes_without_metadata_db_returns_503(self, tmp_path):
        from fastapi.testclient import TestClient

        from local_search_agent.core.config import SearchAgentConfig
        from local_search_agent.server.fastapi_app import build_app
        from local_search_agent.workspace.workspace_manager import WorkspaceManager

        db_path = str(tmp_path / "test.db")
        config = SearchAgentConfig(workspace_name="ws", provider="ollama", db_path=db_path)
        wm = WorkspaceManager(db_path=db_path)
        app = build_app(config=config, workspace_manager=wm, metadata_db=None)
        tc = TestClient(app)
        resp = tc.get("/health/indexes")
        assert resp.status_code == 503

    def test_workspace_history_endpoint(self, tmp_path):
        client, mdb = self._make_client(tmp_path)
        hid = mdb.record_sync_start("test_ws")
        mdb.record_sync_finish(hid, 10, 5, 0, 2.1, [])

        resp = client.get("/workspaces/test_ws/history")
        assert resp.status_code == 200
        data = resp.json()
        assert data["workspace"] == "test_ws"
        assert len(data["history"]) >= 1
        assert data["history"][0]["indexed"] == 10

    def test_workspace_history_unknown_workspace_404(self, tmp_path):
        client, _ = self._make_client(tmp_path)
        resp = client.get("/workspaces/no_such_ws/history")
        assert resp.status_code == 404
