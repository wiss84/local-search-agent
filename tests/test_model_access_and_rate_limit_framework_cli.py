"""
Tests for the SearchAgentFramework Python API methods and CLI commands for
Model/Provider Access Control and Rate Limits & Concurrency (added after
noticing these features had HTTP/CLI-partial but no Framework-level Python
API surface, and Model Access had no CLI at all -- see
docs/role_based_access_control.md).
"""

from __future__ import annotations

from io import StringIO
from unittest.mock import patch

import pytest


def _run(args: list[str]) -> tuple[int, str, str]:
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


@pytest.fixture
def framework(tmp_path):
    from local_search_agent.core.config import SearchAgentConfig
    from local_search_agent.core.framework import SearchAgentFramework

    config = SearchAgentConfig(workspace_name="default", db_path=str(tmp_path / "test.db"))
    return SearchAgentFramework(config)


@pytest.fixture
def isolated_rate_limits(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "local_search_agent.core.key_manager.user_config_dir", lambda app: str(tmp_path)
    )
    monkeypatch.chdir(tmp_path)
    yield tmp_path


# ---------------------------------------------------------------------------
# SearchAgentFramework: Model / Provider Access Control
# ---------------------------------------------------------------------------


class TestFrameworkModelAccessMethods:
    def test_grant_then_list(self, framework):
        framework.grant_model_access("member", "google", "gemma-4-31b-it", granted_by="root")
        rows = framework.list_model_access(role="member")
        assert len(rows) == 1
        assert rows[0]["provider"] == "google"
        assert rows[0]["model_name"] == "gemma-4-31b-it"

    def test_revoke(self, framework):
        framework.grant_model_access("member", "google", "gemma-4-31b-it", granted_by="root")
        assert framework.revoke_model_access("member", "google", "gemma-4-31b-it") is True
        assert framework.list_model_access(role="member") == []

    def test_revoke_nonexistent_returns_false(self, framework):
        assert framework.revoke_model_access("member", "google", "nope") is False

    def test_list_all_roles(self, framework):
        framework.grant_model_access("member", "google", "a", granted_by="root")
        framework.grant_model_access("admin", "openai", "b", granted_by="root")
        assert len(framework.list_model_access()) == 2


# ---------------------------------------------------------------------------
# SearchAgentFramework: Rate Limits & Concurrency
# ---------------------------------------------------------------------------


class TestFrameworkRateLimitMethods:
    def test_set_get_delete_concurrency(self, framework, isolated_rate_limits):
        framework.set_concurrency_limit("ollama", 2, multi_tenant=False)
        assert framework.get_concurrency_limits(multi_tenant=False) == {"ollama": 2}
        assert framework.delete_concurrency_limit("ollama", multi_tenant=False) is True
        assert framework.get_concurrency_limits(multi_tenant=False) == {}

    def test_set_get_delete_quota_override(self, framework, isolated_rate_limits):
        framework.set_quota_override("openai", "gpt-5", multi_tenant=True, rpm=500, tpm=2_000_000)
        overrides = framework.get_quota_overrides(multi_tenant=True, provider="openai")
        assert overrides == {"gpt-5": {"rpm": 500, "tpm": 2_000_000}}
        assert framework.delete_quota_override("openai", "gpt-5", multi_tenant=True) is True
        assert framework.get_quota_overrides(multi_tenant=True, provider="openai") == {}

    def test_single_user_and_multi_tenant_independent_via_framework(
        self, framework, isolated_rate_limits
    ):
        framework.set_concurrency_limit("ollama", 2, multi_tenant=False)
        framework.set_concurrency_limit("ollama", 8, multi_tenant=True)
        assert framework.get_concurrency_limits(multi_tenant=False) == {"ollama": 2}
        assert framework.get_concurrency_limits(multi_tenant=True) == {"ollama": 8}


# ---------------------------------------------------------------------------
# CLI: grant-model-access / revoke-model-access / list-model-access
# ---------------------------------------------------------------------------


class TestModelAccessCLI:
    def test_grant_then_list(self, tmp_path):
        db = str(tmp_path / "test.db")
        code, out, err = _run(
            [
                "--db",
                db,
                "grant-model-access",
                "--role",
                "member",
                "--provider",
                "google",
                "--model-name",
                "gemma-4-31b-it",
            ]
        )
        assert code == 0
        assert "granted" in out.lower()

        code, out, err = _run(["--db", db, "list-model-access"])
        assert code == 0
        assert "gemma-4-31b-it" in out

    def test_revoke(self, tmp_path):
        db = str(tmp_path / "test.db")
        _run(
            [
                "--db",
                db,
                "grant-model-access",
                "--role",
                "member",
                "--provider",
                "google",
                "--model-name",
                "gemma-4-31b-it",
            ]
        )
        code, out, err = _run(
            [
                "--db",
                db,
                "revoke-model-access",
                "--role",
                "member",
                "--provider",
                "google",
                "--model-name",
                "gemma-4-31b-it",
            ]
        )
        assert code == 0
        assert "revoked" in out.lower()

    def test_revoke_nonexistent_reports_not_found(self, tmp_path):
        db = str(tmp_path / "test.db")
        code, out, err = _run(
            [
                "--db",
                db,
                "revoke-model-access",
                "--role",
                "member",
                "--provider",
                "google",
                "--model-name",
                "nope",
            ]
        )
        assert code == 0
        assert "no grant found" in out.lower()

    def test_list_empty(self, tmp_path):
        db = str(tmp_path / "test.db")
        code, out, err = _run(["--db", db, "list-model-access"])
        assert code == 0
        assert "no model-access grants" in out.lower()

    def test_list_filtered_by_role(self, tmp_path):
        db = str(tmp_path / "test.db")
        _run(
            [
                "--db",
                db,
                "grant-model-access",
                "--role",
                "member",
                "--provider",
                "google",
                "--model-name",
                "a",
            ]
        )
        _run(
            [
                "--db",
                db,
                "grant-model-access",
                "--role",
                "admin",
                "--provider",
                "openai",
                "--model-name",
                "b",
            ]
        )
        code, out, err = _run(["--db", db, "list-model-access", "--role", "member"])
        assert code == 0
        assert "google" in out
        assert "openai" not in out


# ---------------------------------------------------------------------------
# CLI: --help doesn't crash for the new commands
# ---------------------------------------------------------------------------


class TestNewCommandsHelp:
    @pytest.mark.parametrize(
        "cmd",
        [
            "grant-model-access",
            "revoke-model-access",
            "list-model-access",
        ],
    )
    def test_help(self, cmd):
        code, out, err = _run([cmd, "--help"])
        assert code == 0
