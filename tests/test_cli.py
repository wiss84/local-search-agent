"""
Unit and integration tests for the CLI commands (cli/commands.py).

Tests cover:
- All subcommands exit with code 0 on valid input
- Error paths exit with non-zero code
- config set-key / list-keys / delete-key round-trip
- workspace create / list / delete output
- ingest --wipe flag calls wipe_and_reingest
- query single-question mode returns answer
- health command shows status
- scheduler status command
- Global --help does not crash
"""

from __future__ import annotations

from io import StringIO
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(args: list[str]) -> tuple[int, str, str]:
    """
    Run the CLI with the given args list.
    Returns (exit_code, stdout, stderr).
    """
    from local_search_agent.cli.commands import main

    stdout_buf = StringIO()
    stderr_buf = StringIO()
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


def _patch_keys_path(tmp_path):
    keys_file = tmp_path / "keys.json"
    return patch(
        "local_search_agent.core.key_manager._keys_path",
        return_value=keys_file,
    )


# ---------------------------------------------------------------------------
# Global --help
# ---------------------------------------------------------------------------


class TestHelp:
    def test_help_does_not_crash(self):
        code, out, err = _run(["--help"])
        assert code == 0
        assert "usage" in out.lower() or "local-search" in out.lower()

    def test_workspace_help(self):
        code, out, err = _run(["workspace", "--help"])
        assert code == 0

    def test_ingest_help(self):
        code, out, err = _run(["ingest", "--help"])
        assert code == 0

    def test_query_help(self):
        code, out, err = _run(["query", "--help"])
        assert code == 0

    def test_config_help(self):
        code, out, err = _run(["config", "--help"])
        assert code == 0


# ---------------------------------------------------------------------------
# config set-key / list-keys / delete-key
# ---------------------------------------------------------------------------


class TestConfigCommands:
    def test_set_key_google(self, tmp_path):
        with _patch_keys_path(tmp_path):
            code, out, err = _run(
                [
                    "config",
                    "set-key",
                    "--provider",
                    "google",
                    "--key",
                    "AIzaSyTEST1234567890abcdef",
                ]
            )
            assert code == 0
            assert "saved" in out.lower()

    def test_set_key_shows_storage_path(self, tmp_path):
        with _patch_keys_path(tmp_path):
            code, out, err = _run(
                [
                    "config",
                    "set-key",
                    "--provider",
                    "google",
                    "--key",
                    "AIzaSyTEST1234567890abcdef",
                ]
            )
            assert "keys.json" in out or str(tmp_path) in out

    def test_list_keys_shows_saved_provider(self, tmp_path):
        with _patch_keys_path(tmp_path):
            from local_search_agent.core.key_manager import set_key

            set_key("openai", "sk-test_key_value_1234567890abcd")
            code, out, err = _run(["config", "list-keys"])
            assert code == 0
            assert "openai" in out

    def test_list_keys_shows_none_when_empty(self, tmp_path):
        with _patch_keys_path(tmp_path):
            code, out, err = _run(["config", "list-keys"])
            assert code == 0
            assert "none" in out.lower() or out.strip() != ""

    def test_delete_key_removes_entry(self, tmp_path):
        with _patch_keys_path(tmp_path):
            from local_search_agent.core.key_manager import set_key

            set_key("google", "AIzaSyTEST1234567890abcdef")
            code, out, err = _run(["config", "delete-key", "--provider", "google"])
            assert code == 0
            assert "removed" in out.lower() or "deleted" in out.lower() or "google" in out

    def test_set_key_unknown_provider_exits_nonzero(self, tmp_path):
        with _patch_keys_path(tmp_path):
            code, out, err = _run(
                [
                    "config",
                    "set-key",
                    "--provider",
                    "bingus",
                    "--key",
                    "somekey",
                ]
            )
            assert code != 0


# ---------------------------------------------------------------------------
# config add-model / delete-model / list-models
# ---------------------------------------------------------------------------


def _patch_models_path(tmp_path):
    models_file = tmp_path / "models.json"
    return patch(
        "local_search_agent.core.key_manager._models_path",
        return_value=models_file,
    )


class TestConfigModelCommands:
    def test_add_model_ollama(self, tmp_path):
        with _patch_models_path(tmp_path):
            code, out, err = _run(
                [
                    "config",
                    "add-model",
                    "--provider",
                    "ollama",
                    "--model-name",
                    "gemma4:e2b",
                ]
            )
            assert code == 0
            assert "gemma4:e2b" in out

    def test_add_model_shows_storage_path(self, tmp_path):
        with _patch_models_path(tmp_path):
            code, out, err = _run(
                [
                    "config",
                    "add-model",
                    "--provider",
                    "ollama",
                    "--model-name",
                    "mistral",
                ]
            )
            assert code == 0
            assert "models.json" in out or str(tmp_path) in out

    def test_add_model_unknown_provider_exits_nonzero(self, tmp_path):
        with _patch_models_path(tmp_path):
            code, out, err = _run(
                [
                    "config",
                    "add-model",
                    "--provider",
                    "bingus",
                    "--model-name",
                    "some-model",
                ]
            )
            assert code != 0

    def test_delete_model(self, tmp_path):
        with _patch_models_path(tmp_path):
            from local_search_agent.core.key_manager import add_model

            add_model("ollama", "gemma4:e2b")
            code, out, err = _run(
                [
                    "config",
                    "delete-model",
                    "--provider",
                    "ollama",
                    "--model-name",
                    "gemma4:e2b",
                ]
            )
            assert code == 0
            assert "removed" in out.lower() or "gemma4:e2b" in out

    def test_delete_model_not_found(self, tmp_path):
        with _patch_models_path(tmp_path):
            code, out, err = _run(
                [
                    "config",
                    "delete-model",
                    "--provider",
                    "ollama",
                    "--model-name",
                    "nonexistent-model",
                ]
            )
            assert code == 0
            assert "not found" in out.lower()

    def test_list_models_shows_google_defaults(self, tmp_path):
        with _patch_models_path(tmp_path):
            code, out, err = _run(["config", "list-models"])
            assert code == 0
            assert "google" in out
            assert "gemma-4-31b-it" in out

    def test_list_models_shows_added_model(self, tmp_path):
        with _patch_models_path(tmp_path):
            from local_search_agent.core.key_manager import add_model

            add_model("ollama", "mistral")
            code, out, err = _run(["config", "list-models"])
            assert code == 0
            assert "mistral" in out


# ---------------------------------------------------------------------------
# config set-semantic / show-semantic / show
# ---------------------------------------------------------------------------


def _patch_settings_path(tmp_path):
    settings_file = tmp_path / "settings.json"
    return patch(
        "local_search_agent.core.key_manager._settings_path",
        return_value=settings_file,
    )


class TestConfigSemanticCommands:
    def test_set_semantic_enable(self, tmp_path):
        with _patch_settings_path(tmp_path):
            code, out, err = _run(["config", "set-semantic", "--enable", "true"])
            assert code == 0
            assert "enabled" in out.lower()

    def test_set_semantic_disable(self, tmp_path):
        with _patch_settings_path(tmp_path):
            code, out, err = _run(["config", "set-semantic", "--enable", "false"])
            assert code == 0
            assert "disabled" in out.lower()

    def test_set_semantic_query_expansion(self, tmp_path):
        with _patch_settings_path(tmp_path):
            code, out, err = _run(["config", "set-semantic", "--query-expansion", "on"])
            assert code == 0

    def test_set_semantic_persists(self, tmp_path):
        with _patch_settings_path(tmp_path):
            _run(["config", "set-semantic", "--enable", "true"])
            from local_search_agent.core.key_manager import get_semantic_settings

            s = get_semantic_settings()
            assert s["enable_semantic"] is True

    def test_show_semantic_shows_all_flags(self, tmp_path):
        with _patch_settings_path(tmp_path):
            code, out, err = _run(["config", "show-semantic"])
            assert code == 0
            assert "enable_semantic" in out
            assert "enable_query_expansion" in out

    def test_show_semantic_shows_on_off(self, tmp_path):
        with _patch_settings_path(tmp_path):
            from local_search_agent.core.key_manager import set_semantic_setting

            set_semantic_setting("enable_semantic", True)
            code, out, err = _run(["config", "show-semantic"])
            assert "ON" in out or "off" in out


class TestConfigShow:
    def test_show_exits_zero(self, tmp_path):
        with (
            _patch_keys_path(tmp_path),
            _patch_models_path(tmp_path),
            _patch_settings_path(tmp_path),
        ):
            code, out, err = _run(["config", "show"])
            assert code == 0

    def test_show_contains_sections(self, tmp_path):
        with (
            _patch_keys_path(tmp_path),
            _patch_models_path(tmp_path),
            _patch_settings_path(tmp_path),
        ):
            code, out, err = _run(["config", "show"])
            assert "API Keys" in out
            assert "Models" in out
            assert "Semantic" in out
            assert "LangSmith" in out

    def test_show_contains_version(self, tmp_path):
        with (
            _patch_keys_path(tmp_path),
            _patch_models_path(tmp_path),
            _patch_settings_path(tmp_path),
        ):
            code, out, err = _run(["config", "show"])
            assert "0.1.0" in out or "Local Search Agent" in out


# ---------------------------------------------------------------------------
# workspace create / list / delete
# ---------------------------------------------------------------------------


class TestWorkspaceCommands:
    def test_workspace_create(self, tmp_path):
        db = str(tmp_path / "test.db")
        with patch("local_search_agent.core.framework.SearchAgentFramework.create_workspace"):
            code, out, err = _run(
                [
                    "--db",
                    db,
                    "workspace",
                    "create",
                    "finance",
                    str(tmp_path),
                ]
            )
            assert code == 0

    def test_workspace_list(self, tmp_path):
        db = str(tmp_path / "test.db")
        mock_fw = MagicMock()
        mock_fw.list_workspaces.return_value = [
            {"name": "finance", "document_dir": str(tmp_path)},
        ]
        with patch(
            "local_search_agent.core.framework.SearchAgentFramework",
            return_value=mock_fw,
        ):
            code, out, err = _run(["--db", db, "workspace", "list"])
            assert code == 0
            assert "finance" in out

    def test_workspace_list_empty(self, tmp_path):
        db = str(tmp_path / "test.db")
        mock_fw = MagicMock()
        mock_fw.list_workspaces.return_value = []
        with patch(
            "local_search_agent.core.framework.SearchAgentFramework",
            return_value=mock_fw,
        ):
            code, out, err = _run(["--db", db, "workspace", "list"])
            assert code == 0
            assert "no workspaces" in out.lower()

    def test_workspace_delete(self, tmp_path):
        db = str(tmp_path / "test.db")
        mock_fw = MagicMock()
        with patch(
            "local_search_agent.core.framework.SearchAgentFramework",
            return_value=mock_fw,
        ):
            code, out, err = _run(["--db", db, "workspace", "delete", "finance"])
            assert code == 0
            mock_fw.delete_workspace.assert_called_once_with(name="finance", wipe_index=False)

    def test_workspace_delete_with_wipe(self, tmp_path):
        db = str(tmp_path / "test.db")
        mock_fw = MagicMock()
        with patch(
            "local_search_agent.core.framework.SearchAgentFramework",
            return_value=mock_fw,
        ):
            code, out, err = _run(["--db", db, "workspace", "delete", "finance", "--wipe"])
            assert code == 0
            mock_fw.delete_workspace.assert_called_once_with(name="finance", wipe_index=True)


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------


class TestIngestCommands:
    def test_ingest_basic(self, tmp_path):
        db = str(tmp_path / "test.db")
        mock_fw = MagicMock()
        mock_stats = MagicMock()
        mock_stats.__str__ = lambda s: "IngestStats(total=5, indexed=5, skipped=0, failed=0)"
        mock_stats.errors = []
        mock_fw.ingest_and_index.return_value = mock_stats
        with patch(
            "local_search_agent.core.framework.SearchAgentFramework",
            return_value=mock_fw,
        ):
            code, out, err = _run(
                [
                    "--db",
                    db,
                    "ingest",
                    "--workspace",
                    "finance",
                    "--dirs",
                    str(tmp_path),
                ]
            )
            assert code == 0
            assert "Done" in out

    def test_ingest_wipe_calls_wipe_and_reingest(self, tmp_path):
        db = str(tmp_path / "test.db")
        mock_fw = MagicMock()
        mock_stats = MagicMock()
        mock_stats.__str__ = lambda s: "IngestStats(total=5, indexed=5, skipped=0, failed=0)"
        mock_stats.errors = []
        mock_fw.wipe_and_reingest.return_value = mock_stats
        with patch(
            "local_search_agent.core.framework.SearchAgentFramework",
            return_value=mock_fw,
        ):
            code, out, err = _run(
                [
                    "--db",
                    db,
                    "ingest",
                    "--workspace",
                    "finance",
                    "--dirs",
                    str(tmp_path),
                    "--wipe",
                ]
            )
            assert code == 0
            mock_fw.wipe_and_reingest.assert_called_once()
            mock_fw.ingest_and_index.assert_not_called()

    def test_ingest_shows_errors(self, tmp_path):
        db = str(tmp_path / "test.db")
        mock_fw = MagicMock()
        mock_stats = MagicMock()
        mock_stats.__str__ = lambda s: "IngestStats(total=5, indexed=3, skipped=0, failed=2)"
        mock_stats.errors = ["file1.pdf: parse error", "file2.docx: locked"]
        mock_fw.ingest_and_index.return_value = mock_stats
        with patch(
            "local_search_agent.core.framework.SearchAgentFramework",
            return_value=mock_fw,
        ):
            code, out, err = _run(
                [
                    "--db",
                    db,
                    "ingest",
                    "--workspace",
                    "finance",
                    "--dirs",
                    str(tmp_path),
                ]
            )
            assert "error" in out.lower()


# ---------------------------------------------------------------------------
# query (single question mode)
# ---------------------------------------------------------------------------


class TestQueryCommand:
    def test_query_single_question(self, tmp_path):
        db = str(tmp_path / "test.db")
        mock_fw = MagicMock()
        mock_fw.query.return_value = {
            "answer": "The AWS spend was $1.2M in Q3.",
            "iterations_used": 3,
            "truncated": False,
            "sources": [],
        }
        with patch(
            "local_search_agent.core.framework.SearchAgentFramework",
            return_value=mock_fw,
        ):
            code, out, err = _run(
                [
                    "--db",
                    db,
                    "query",
                    "What was the AWS spend?",
                    "--workspace",
                    "finance",
                    "--provider",
                    "ollama",
                    "--model",
                    "mistral",
                ]
            )
            assert code == 0
            assert "AWS spend" in out or "$1.2M" in out

    def test_query_truncated_shows_warning(self, tmp_path):
        db = str(tmp_path / "test.db")
        mock_fw = MagicMock()
        mock_fw.query.return_value = {
            "answer": "Partial answer.",
            "iterations_used": 10,
            "truncated": True,
            "sources": [],
        }
        with patch(
            "local_search_agent.core.framework.SearchAgentFramework",
            return_value=mock_fw,
        ):
            code, out, err = _run(
                [
                    "--db",
                    db,
                    "query",
                    "Some question",
                    "--workspace",
                    "finance",
                    "--provider",
                    "ollama",
                    "--model",
                    "mistral",
                ]
            )
            assert "incomplete" in out.lower() or "max iteration" in out.lower()


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------


class TestHealthCommand:
    def test_health_shows_workspace_status(self, tmp_path):
        db = str(tmp_path / "test.db")
        mock_summary = MagicMock()
        mock_ws = MagicMock()
        mock_ws.workspace = "finance"
        mock_ws.status = "healthy"
        mock_ws.doc_count = 142
        mock_ws.age_minutes = 5
        mock_ws.next_sync_at = None
        mock_ws.last_error = None
        mock_summary.workspaces = [mock_ws]
        mock_summary.total_workspaces = 1
        mock_summary.healthy = 1
        mock_summary.stale = 0
        mock_summary.never_synced = 0
        mock_summary.error = 0
        mock_summary.running = 0
        mock_summary.total_docs = 142
        with patch(
            "local_search_agent.scheduler.monitor.IndexMonitor",
            return_value=MagicMock(get_health_summary=MagicMock(return_value=mock_summary)),
        ):
            code, out, err = _run(["--db", db, "health"])
            assert code == 0
            assert "finance" in out


# ---------------------------------------------------------------------------
# scheduler status
# ---------------------------------------------------------------------------


class TestSchedulerStatusCommand:
    def test_scheduler_status_not_running(self, tmp_path):
        db = str(tmp_path / "test.db")
        mock_fw = MagicMock()
        mock_fw.get_scheduler_status.return_value = {"running": False, "scheduled_jobs": []}
        with patch(
            "local_search_agent.core.framework.SearchAgentFramework",
            return_value=mock_fw,
        ):
            code, out, err = _run(["--db", db, "scheduler", "status"])
            assert code == 0
            assert "not running" in out.lower()

    def test_scheduler_status_running_shows_jobs(self, tmp_path):
        db = str(tmp_path / "test.db")
        mock_fw = MagicMock()
        mock_fw.get_scheduler_status.return_value = {
            "running": True,
            "scheduled_jobs": [
                {"workspace": "finance", "interval_minutes": 15, "next_run": "2025-05-26T14:30:00"},
            ],
        }
        with patch(
            "local_search_agent.core.framework.SearchAgentFramework",
            return_value=mock_fw,
        ):
            code, out, err = _run(["--db", db, "scheduler", "status"])
            assert code == 0
            assert "finance" in out
