"""
Direct unit tests for individual cmd_* functions in local_search_agent/cli/commands.py.

These tests call cmd_* functions directly with an argparse.Namespace (or
MagicMock acting as one) rather than going through the CLI parser, isolating
each command from the argparse machinery.
"""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _args(**kwargs) -> argparse.Namespace:
    """Build an argparse.Namespace with sane defaults for all commands."""
    defaults = {
        "force": False,
        "provider": "ollama",
        "limit": 1,
        "multi_tenant": False,
        "model_name": None,
        "rpm": None,
        "tpm": None,
        "rpd": None,
        "name": "test-workspace",
        "dir": "/tmp",
        "db": ":memory:",
        "meili_url": "http://localhost:7700",
        "meili_key": "master",
        "wipe": False,
        "workspace": "default",
        "subject": "user@example.com",
        "role": "member",
        "granted_by": "tester",
        "display_name": None,
        "superadmin": False,
        "created_by": "tester",
        "key_id": "key-123",
        "host": "127.0.0.1",
        "port": 8765,
        "model": "gemma-4-31b-it",
        "scheduler_interval": 0,
        "headless": False,
        "insecure_cookies": False,
        "question": None,
        "api_key": None,
        "max_iterations": 10,
        "top_k": 5,
        "dirs": None,
        "stale_threshold": 30,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# cmd_setup
# ---------------------------------------------------------------------------


class TestCmdSetup:
    def test_calls_run_setup_with_force_true(self):
        from local_search_agent.cli.commands import cmd_setup

        with patch("local_search_agent.core.meilisearch_manager.run_setup") as mock_run:
            args = _args(force=True)
            cmd_setup(args)
            mock_run.assert_called_once_with(force=True)

    def test_calls_run_setup_with_force_false(self):
        from local_search_agent.cli.commands import cmd_setup

        with patch("local_search_agent.core.meilisearch_manager.run_setup") as mock_run:
            args = _args(force=False)
            cmd_setup(args)
            mock_run.assert_called_once_with(force=False)


# ---------------------------------------------------------------------------
# cmd_config_set_concurrency
# ---------------------------------------------------------------------------


class TestCmdConfigSetConcurrency:
    def test_valid_limit_sets_concurrency(self, capsys):
        from local_search_agent.cli.commands import cmd_config_set_concurrency

        with patch("local_search_agent.core.key_manager.set_concurrency_limit") as mock_set:
            with patch(
                "local_search_agent.agent.rate_limit_handler.reset_shared_rate_limit_handlers"
            ) as mock_reset:
                args = _args(provider="ollama", limit=4, multi_tenant=False)
                cmd_config_set_concurrency(args)
                mock_set.assert_called_once_with("ollama", 4, False)
                mock_reset.assert_called_once()
                captured = capsys.readouterr()
                assert "4" in captured.out
                assert "single-user" in captured.out

    def test_limit_less_than_one_exits(self):
        from local_search_agent.cli.commands import cmd_config_set_concurrency

        with patch(
            "local_search_agent.core.key_manager.set_concurrency_limit",
            side_effect=ValueError("limit < 1"),
        ):
            args = _args(provider="ollama", limit=0, multi_tenant=False)
            try:
                cmd_config_set_concurrency(args)
            except SystemExit as e:
                assert e.code == 1
            else:
                raise AssertionError("Expected SystemExit(1)")


# ---------------------------------------------------------------------------
# cmd_config_delete_concurrency
# ---------------------------------------------------------------------------


class TestCmdConfigDeleteConcurrency:
    def test_successful_deletion_prints_message(self, capsys):
        from local_search_agent.cli.commands import cmd_config_delete_concurrency

        with patch(
            "local_search_agent.core.key_manager.delete_concurrency_limit",
            return_value=True,
        ):
            with patch(
                "local_search_agent.agent.rate_limit_handler.reset_shared_rate_limit_handlers"
            ) as mock_reset:
                args = _args(provider="google", multi_tenant=True)
                cmd_config_delete_concurrency(args)
                mock_reset.assert_called_once()
                captured = capsys.readouterr()
                assert "removed" in captured.out.lower()
                assert "multi-tenant" in captured.out

    def test_no_existing_limit_prints_not_set(self, capsys):
        from local_search_agent.cli.commands import cmd_config_delete_concurrency

        with patch(
            "local_search_agent.core.key_manager.delete_concurrency_limit",
            return_value=False,
        ):
            with patch(
                "local_search_agent.agent.rate_limit_handler.reset_shared_rate_limit_handlers"
            ) as mock_reset:
                args = _args(provider="ollama", multi_tenant=False)
                cmd_config_delete_concurrency(args)
                mock_reset.assert_called_once()
                captured = capsys.readouterr()
                assert "no concurrency limit" in captured.out.lower()


# ---------------------------------------------------------------------------
# cmd_config_set_rate_limit
# ---------------------------------------------------------------------------


class TestCmdConfigSetRateLimit:
    def test_no_limits_provided_exits(self):
        from local_search_agent.cli.commands import cmd_config_set_rate_limit

        args = _args(provider="google", model_name="gemma-4-31b-it", rpm=None, tpm=None, rpd=None)
        try:
            cmd_config_set_rate_limit(args)
        except SystemExit as e:
            assert e.code == 1
        else:
            raise AssertionError("Expected SystemExit(1)")

    def test_valid_limits_sets_override(self, capsys):
        from local_search_agent.cli.commands import cmd_config_set_rate_limit

        with patch("local_search_agent.core.key_manager.set_quota_override") as mock_set:
            with patch(
                "local_search_agent.agent.rate_limit_handler.reset_shared_rate_limit_handlers"
            ) as mock_reset:
                args = _args(
                    provider="openai",
                    model_name="gpt-4o",
                    rpm=100,
                    tpm=50000,
                    rpd=1000,
                    multi_tenant=False,
                )
                cmd_config_set_rate_limit(args)
                mock_set.assert_called_once_with(
                    "openai", "gpt-4o", False, rpm=100, tpm=50000, rpd=1000
                )
                mock_reset.assert_called_once()
                captured = capsys.readouterr()
                assert "Rate limit override set" in captured.out


# ---------------------------------------------------------------------------
# cmd_config_delete_rate_limit
# ---------------------------------------------------------------------------


class TestCmdConfigDeleteRateLimit:
    def test_successful_deletion_prints_message(self, capsys):
        from local_search_agent.cli.commands import cmd_config_delete_rate_limit

        with patch(
            "local_search_agent.core.key_manager.delete_quota_override",
            return_value=True,
        ):
            with patch(
                "local_search_agent.agent.rate_limit_handler.reset_shared_rate_limit_handlers"
            ) as mock_reset:
                args = _args(provider="google", model_name="gemma-4-31b-it", multi_tenant=False)
                cmd_config_delete_rate_limit(args)
                mock_reset.assert_called_once()
                captured = capsys.readouterr()
                assert "removed" in captured.out.lower()
                assert "gemma-4-31b-it" in captured.out

    def test_no_existing_override_prints_not_set(self, capsys):
        from local_search_agent.cli.commands import cmd_config_delete_rate_limit

        with patch(
            "local_search_agent.core.key_manager.delete_quota_override",
            return_value=False,
        ):
            with patch(
                "local_search_agent.agent.rate_limit_handler.reset_shared_rate_limit_handlers"
            ) as mock_reset:
                args = _args(provider="anthropic", model_name="claude-4-sonnet", multi_tenant=False)
                cmd_config_delete_rate_limit(args)
                mock_reset.assert_called_once()
                captured = capsys.readouterr()
                assert "no rate limit override" in captured.out.lower()


# ---------------------------------------------------------------------------
# cmd_config_show_rate_limits
# ---------------------------------------------------------------------------


class TestCmdConfigShowRateLimits:
    def test_output_includes_header(self, capsys):
        from local_search_agent.cli.commands import cmd_config_show_rate_limits

        with patch("local_search_agent.core.key_manager.get_concurrency_limits", return_value={}):
            with patch("local_search_agent.core.key_manager.get_quota_overrides", return_value={}):
                args = _args(multi_tenant=False)
                cmd_config_show_rate_limits(args)
                captured = capsys.readouterr()
                assert "Rate limits & concurrency" in captured.out
                assert "single-user" in captured.out


# ---------------------------------------------------------------------------
# cmd_workspace_create
# ---------------------------------------------------------------------------


class TestCmdWorkspaceCreate:
    def test_creates_workspace_and_upserts_sync_job(self, capsys):
        from local_search_agent.cli.commands import cmd_workspace_create

        mock_wm = MagicMock()
        mock_mdb = MagicMock()
        with (
            patch(
                "local_search_agent.workspace.workspace_manager.WorkspaceManager",
                return_value=mock_wm,
            ),
            patch(
                "local_search_agent.workspace.metadata_db.MetadataDB",
                return_value=mock_mdb,
            ),
        ):
            args = _args(name="finance", dir="/tmp/finance", multi_tenant=False)
            cmd_workspace_create(args)
            mock_wm.create_workspace.assert_called_once_with(
                name="finance", document_dir="/tmp/finance"
            )
            mock_mdb.upsert_sync_job.assert_called_once_with(workspace="finance")
            captured = capsys.readouterr()
            assert "'finance'" in captured.out
            assert "/tmp/finance" in captured.out

    def test_multi_tenant_calls_provision_workspace_keys(self, capsys):
        from local_search_agent.cli.commands import cmd_workspace_create

        mock_wm = MagicMock()
        mock_mdb = MagicMock()
        mock_key_uid = "key-uid-abc"
        with (
            patch(
                "local_search_agent.workspace.workspace_manager.WorkspaceManager",
                return_value=mock_wm,
            ),
            patch(
                "local_search_agent.workspace.metadata_db.MetadataDB",
                return_value=mock_mdb,
            ),
            patch(
                "local_search_agent.auth.meili_key_provisioning.provision_workspace_keys",
                return_value=mock_key_uid,
            ) as mock_provision,
            patch("local_search_agent.workspace.auth_db.AuthDB") as mock_auth_db_cls,
        ):
            mock_auth_db = MagicMock()
            mock_auth_db_cls.return_value = mock_auth_db
            args = _args(
                name="finance",
                dir="/tmp/finance",
                multi_tenant=True,
                meili_url="http://localhost:7700",
                meili_key="master",
            )
            cmd_workspace_create(args)
            mock_provision.assert_called_once_with(
                workspace="finance",
                meilisearch_url="http://localhost:7700",
                meili_master_key="master",
                auth_db=mock_auth_db,
            )
            captured = capsys.readouterr()
            assert "Scoped member-level Meilisearch key provisioned" in captured.out
            assert "uid=key-uid-abc" in captured.out

    def test_multi_tenant_provision_failure_prints_warning(self, capsys):
        from local_search_agent.cli.commands import cmd_workspace_create

        mock_wm = MagicMock()
        mock_mdb = MagicMock()
        with (
            patch(
                "local_search_agent.workspace.workspace_manager.WorkspaceManager",
                return_value=mock_wm,
            ),
            patch(
                "local_search_agent.workspace.metadata_db.MetadataDB",
                return_value=mock_mdb,
            ),
            patch(
                "local_search_agent.auth.meili_key_provisioning.provision_workspace_keys",
                return_value=None,
            ),
            patch("local_search_agent.workspace.auth_db.AuthDB"),
        ):
            args = _args(
                name="finance",
                dir="/tmp/finance",
                multi_tenant=True,
                meili_url="http://localhost:7700",
                meili_key="master",
            )
            cmd_workspace_create(args)
            captured = capsys.readouterr()
            assert "Could not provision" in captured.out


# ---------------------------------------------------------------------------
# cmd_workspace_list
# ---------------------------------------------------------------------------


class TestCmdWorkspaceList:
    def test_no_workspaces_registered_message(self, capsys):
        from local_search_agent.cli.commands import cmd_workspace_list

        mock_fw = MagicMock()
        mock_fw.list_workspaces.return_value = []
        with (
            patch(
                "local_search_agent.core.framework.SearchAgentFramework",
                return_value=mock_fw,
            ),
            patch("local_search_agent.core.config.SearchAgentConfig"),
        ):
            args = _args()
            cmd_workspace_list(args)
            captured = capsys.readouterr()
            assert "No workspaces registered" in captured.out

    def test_lists_workspaces(self, capsys):
        from local_search_agent.cli.commands import cmd_workspace_list

        mock_fw = MagicMock()
        mock_fw.list_workspaces.return_value = [
            {"name": "finance", "document_dir": "/tmp/finance"},
            {"name": "legal", "document_dir": "/tmp/legal"},
        ]
        with (
            patch(
                "local_search_agent.core.framework.SearchAgentFramework",
                return_value=mock_fw,
            ),
            patch("local_search_agent.core.config.SearchAgentConfig"),
        ):
            args = _args()
            cmd_workspace_list(args)
            captured = capsys.readouterr()
            assert "finance" in captured.out
            assert "legal" in captured.out


# ---------------------------------------------------------------------------
# cmd_workspace_delete
# ---------------------------------------------------------------------------


class TestCmdWorkspaceDelete:
    def test_calls_delete_workspace_with_correct_args(self, capsys):
        from local_search_agent.cli.commands import cmd_workspace_delete

        mock_fw = MagicMock()
        with (
            patch(
                "local_search_agent.core.framework.SearchAgentFramework",
                return_value=mock_fw,
            ),
            patch("local_search_agent.core.config.SearchAgentConfig"),
            patch("local_search_agent.auth.meili_key_provisioning.deprovision_workspace_keys"),
            patch("local_search_agent.workspace.auth_db.AuthDB"),
        ):
            args = _args(name="finance", wipe=False, multi_tenant=False)
            cmd_workspace_delete(args)
            mock_fw.delete_workspace.assert_called_once_with(name="finance", wipe_index=False)

    def test_calls_delete_workspace_with_wipe(self, capsys):
        from local_search_agent.cli.commands import cmd_workspace_delete

        mock_fw = MagicMock()
        with (
            patch(
                "local_search_agent.core.framework.SearchAgentFramework",
                return_value=mock_fw,
            ),
            patch("local_search_agent.core.config.SearchAgentConfig"),
            patch("local_search_agent.auth.meili_key_provisioning.deprovision_workspace_keys"),
            patch("local_search_agent.workspace.auth_db.AuthDB"),
        ):
            args = _args(name="finance", wipe=True, multi_tenant=False)
            cmd_workspace_delete(args)
            mock_fw.delete_workspace.assert_called_once_with(name="finance", wipe_index=True)

    def test_multi_tenant_calls_deprovision(self, capsys):
        from local_search_agent.cli.commands import cmd_workspace_delete

        mock_fw = MagicMock()
        with (
            patch(
                "local_search_agent.core.framework.SearchAgentFramework",
                return_value=mock_fw,
            ),
            patch("local_search_agent.core.config.SearchAgentConfig"),
            patch(
                "local_search_agent.auth.meili_key_provisioning.deprovision_workspace_keys"
            ) as mock_deprovision,
            patch("local_search_agent.workspace.auth_db.AuthDB") as mock_auth_db_cls,
        ):
            mock_auth_db = MagicMock()
            mock_auth_db_cls.return_value = mock_auth_db
            args = _args(
                name="finance",
                wipe=False,
                multi_tenant=True,
                meili_url="http://localhost:7700",
                meili_key="master",
            )
            cmd_workspace_delete(args)
            mock_deprovision.assert_called_once_with(
                workspace="finance",
                meilisearch_url="http://localhost:7700",
                meili_master_key="master",
                auth_db=mock_auth_db,
            )


# ---------------------------------------------------------------------------
# cmd_grant_access / cmd_revoke_access / cmd_list_access
# ---------------------------------------------------------------------------


class TestCmdGrantAccess:
    def test_calls_framework_grant_workspace_access(self, capsys):
        from local_search_agent.cli.commands import cmd_grant_access

        mock_fw = MagicMock()
        with (
            patch(
                "local_search_agent.core.framework.SearchAgentFramework",
                return_value=mock_fw,
            ),
            patch("local_search_agent.core.config.SearchAgentConfig"),
        ):
            args = _args(
                subject="user@example.com",
                workspace=["finance", "legal"],
                role="admin",
                granted_by="admin",
            )
            cmd_grant_access(args)
            mock_fw.grant_workspace_access.assert_called_once_with(
                workspaces=["finance", "legal"],
                subject="user@example.com",
                role="admin",
                granted_by="admin",
            )
            captured = capsys.readouterr()
            assert "Granted" in captured.out
            assert "user@example.com" in captured.out


class TestCmdRevokeAccess:
    def test_calls_framework_revoke_workspace_access(self, capsys):
        from local_search_agent.cli.commands import cmd_revoke_access

        mock_fw = MagicMock()
        mock_fw.revoke_workspace_access.return_value = 2
        with (
            patch(
                "local_search_agent.core.framework.SearchAgentFramework",
                return_value=mock_fw,
            ),
            patch("local_search_agent.core.config.SearchAgentConfig"),
        ):
            args = _args(subject="user@example.com", workspace=["finance"])
            cmd_revoke_access(args)
            mock_fw.revoke_workspace_access.assert_called_once_with(
                subject="user@example.com", workspaces=["finance"]
            )
            captured = capsys.readouterr()
            assert "Revoked" in captured.out

    def test_revoke_all_when_no_workspace(self, capsys):
        from local_search_agent.cli.commands import cmd_revoke_access

        mock_fw = MagicMock()
        mock_fw.revoke_workspace_access.return_value = 3
        with (
            patch(
                "local_search_agent.core.framework.SearchAgentFramework",
                return_value=mock_fw,
            ),
            patch("local_search_agent.core.config.SearchAgentConfig"),
        ):
            args = _args(subject="user@example.com", workspace=None)
            cmd_revoke_access(args)
            mock_fw.revoke_workspace_access.assert_called_once_with(
                subject="user@example.com", workspaces=None
            )
            captured = capsys.readouterr()
            assert "all access" in captured.out


class TestCmdListAccess:
    def test_no_grants_prints_message(self, capsys):
        from local_search_agent.cli.commands import cmd_list_access

        mock_fw = MagicMock()
        mock_fw.list_workspace_access.return_value = []
        with (
            patch(
                "local_search_agent.core.framework.SearchAgentFramework",
                return_value=mock_fw,
            ),
            patch("local_search_agent.core.config.SearchAgentConfig"),
        ):
            args = _args()
            cmd_list_access(args)
            captured = capsys.readouterr()
            assert "No grants found" in captured.out


# ---------------------------------------------------------------------------
# cmd_grant_model_access / cmd_revoke_model_access / cmd_list_model_access
# ---------------------------------------------------------------------------


class TestCmdGrantModelAccess:
    def test_calls_framework_grant_model_access(self, capsys):
        from local_search_agent.cli.commands import cmd_grant_model_access

        mock_fw = MagicMock()
        with (
            patch(
                "local_search_agent.core.framework.SearchAgentFramework",
                return_value=mock_fw,
            ),
            patch("local_search_agent.core.config.SearchAgentConfig"),
        ):
            args = _args(
                subject="user@example.com",
                role="member",
                provider="google",
                model_name="gemma-4-31b-it",
                granted_by="admin",
                workspace=None,
            )
            cmd_grant_model_access(args)
            mock_fw.grant_model_access.assert_called_once_with(
                role="member",
                provider="google",
                model_name="gemma-4-31b-it",
                granted_by="admin",
            )
            captured = capsys.readouterr()
            assert "Granted" in captured.out


class TestCmdRevokeModelAccess:
    def test_calls_framework_revoke_model_access_success(self, capsys):
        from local_search_agent.cli.commands import cmd_revoke_model_access

        mock_fw = MagicMock()
        mock_fw.revoke_model_access.return_value = True
        with (
            patch(
                "local_search_agent.core.framework.SearchAgentFramework",
                return_value=mock_fw,
            ),
            patch("local_search_agent.core.config.SearchAgentConfig"),
        ):
            args = _args(role="member", provider="openai", model_name="gpt-4o")
            cmd_revoke_model_access(args)
            mock_fw.revoke_model_access.assert_called_once_with(
                role="member", provider="openai", model_name="gpt-4o"
            )
            captured = capsys.readouterr()
            assert "Revoked" in captured.out

    def test_no_grant_prints_not_found(self, capsys):
        from local_search_agent.cli.commands import cmd_revoke_model_access

        mock_fw = MagicMock()
        mock_fw.revoke_model_access.return_value = False
        with (
            patch(
                "local_search_agent.core.framework.SearchAgentFramework",
                return_value=mock_fw,
            ),
            patch("local_search_agent.core.config.SearchAgentConfig"),
        ):
            args = _args(role="member", provider="openai", model_name="gpt-4o")
            cmd_revoke_model_access(args)
            captured = capsys.readouterr()
            assert "No grant found" in captured.out


class TestCmdListModelAccess:
    def test_no_grants_prints_message(self, capsys):
        from local_search_agent.cli.commands import cmd_list_model_access

        mock_fw = MagicMock()
        mock_fw.list_model_access.return_value = []
        with (
            patch(
                "local_search_agent.core.framework.SearchAgentFramework",
                return_value=mock_fw,
            ),
            patch("local_search_agent.core.config.SearchAgentConfig"),
        ):
            args = _args(role=None)
            cmd_list_model_access(args)
            captured = capsys.readouterr()
            assert "No model-access grants found" in captured.out


# ---------------------------------------------------------------------------
# cmd_auth_create_key / cmd_auth_revoke_key / cmd_auth_list_keys
# ---------------------------------------------------------------------------


class TestCmdAuthCreateKey:
    def test_calls_create_api_key_and_prints_raw_key(self, capsys):
        from local_search_agent.cli.commands import cmd_auth_create_key

        mock_fw = MagicMock()
        mock_fw.create_api_key.return_value = ("key-id-123", "raw-key-xyz")
        with (
            patch(
                "local_search_agent.core.framework.SearchAgentFramework",
                return_value=mock_fw,
            ),
            patch("local_search_agent.core.config.SearchAgentConfig"),
        ):
            args = _args(
                subject="user@example.com",
                display_name=None,
                superadmin=False,
                created_by="admin",
            )
            cmd_auth_create_key(args)
            mock_fw.create_api_key.assert_called_once_with(
                subject="user@example.com",
                created_by="admin",
                display_name="",
                is_superadmin=False,
            )
            captured = capsys.readouterr()
            assert "key-id-123" in captured.out
            assert "raw-key-xyz" in captured.out
            assert "will not be shown again" in captured.out


class TestCmdAuthRevokeKey:
    def test_revoked_true_prints_success(self, capsys):
        from local_search_agent.cli.commands import cmd_auth_revoke_key

        mock_fw = MagicMock()
        mock_fw.revoke_api_key.return_value = True
        with (
            patch(
                "local_search_agent.core.framework.SearchAgentFramework",
                return_value=mock_fw,
            ),
            patch("local_search_agent.core.config.SearchAgentConfig"),
        ):
            args = _args(key_id="key-123")
            cmd_auth_revoke_key(args)
            captured = capsys.readouterr()
            assert "revoked" in captured.out.lower()
            assert "key-123" in captured.out

    def test_revoked_false_prints_not_found(self, capsys):
        from local_search_agent.cli.commands import cmd_auth_revoke_key

        mock_fw = MagicMock()
        mock_fw.revoke_api_key.return_value = False
        with (
            patch(
                "local_search_agent.core.framework.SearchAgentFramework",
                return_value=mock_fw,
            ),
            patch("local_search_agent.core.config.SearchAgentConfig"),
        ):
            args = _args(key_id="key-456")
            cmd_auth_revoke_key(args)
            captured = capsys.readouterr()
            assert "No active key found" in captured.out
            assert "key-456" in captured.out


class TestCmdAuthListKeys:
    def test_no_keys_prints_message(self, capsys):
        from local_search_agent.cli.commands import cmd_auth_list_keys

        mock_fw = MagicMock()
        mock_fw.list_api_keys.return_value = []
        with (
            patch(
                "local_search_agent.core.framework.SearchAgentFramework",
                return_value=mock_fw,
            ),
            patch("local_search_agent.core.config.SearchAgentConfig"),
        ):
            args = _args(subject=None)
            cmd_auth_list_keys(args)
            captured = capsys.readouterr()
            assert "No API keys found" in captured.out


# ---------------------------------------------------------------------------
# cmd_ingest
# ---------------------------------------------------------------------------


class TestCmdIngest:
    def test_wipe_calls_wipe_and_reingest(self, capsys):
        from local_search_agent.cli.commands import cmd_ingest

        mock_fw = MagicMock()
        mock_stats = MagicMock()
        mock_stats.__str__ = lambda s: "IngestStats(total=5, indexed=5, skipped=0, failed=0)"
        mock_stats.errors = []
        mock_fw.wipe_and_reingest.return_value = mock_stats
        with (
            patch(
                "local_search_agent.core.framework.SearchAgentFramework",
                return_value=mock_fw,
            ),
            patch("local_search_agent.core.config.SearchAgentConfig"),
        ):
            args = _args(
                workspace="finance",
                dirs=["/tmp/data"],
                meili_url="http://localhost:7700",
                meili_key="master",
                provider="ollama",
                wipe=True,
                force=False,
            )
            cmd_ingest(args)
            mock_fw.wipe_and_reingest.assert_called_once_with(workspace_name="finance")
            mock_fw.ingest_and_index.assert_not_called()
            captured = capsys.readouterr()
            assert "Wiping index" in captured.out

    def test_no_wipe_calls_ingest_and_index(self, capsys):
        from local_search_agent.cli.commands import cmd_ingest

        mock_fw = MagicMock()
        mock_stats = MagicMock()
        mock_stats.__str__ = lambda s: "IngestStats(total=3, indexed=3, skipped=0, failed=0)"
        mock_stats.errors = []
        mock_fw.ingest_and_index.return_value = mock_stats
        with (
            patch(
                "local_search_agent.core.framework.SearchAgentFramework",
                return_value=mock_fw,
            ),
            patch("local_search_agent.core.config.SearchAgentConfig"),
        ):
            args = _args(
                workspace="finance",
                dirs=["/tmp/data"],
                meili_url="http://localhost:7700",
                meili_key="master",
                provider="ollama",
                wipe=False,
                force=True,
            )
            cmd_ingest(args)
            mock_fw.ingest_and_index.assert_called_once_with(force=True)
            mock_fw.wipe_and_reingest.assert_not_called()
            captured = capsys.readouterr()
            assert "Done" in captured.out


# ---------------------------------------------------------------------------
# cmd_health
# ---------------------------------------------------------------------------


class TestCmdHealth:
    def test_no_workspaces_registered_message(self, capsys):
        from local_search_agent.cli.commands import cmd_health

        mock_summary = MagicMock()
        mock_summary.total_workspaces = 0
        mock_summary.healthy = 0
        mock_summary.stale = 0
        mock_summary.never_synced = 0
        mock_summary.error = 0
        mock_summary.running = 0
        mock_summary.total_docs = 0
        mock_summary.workspaces = []
        mock_summary.all_healthy = True

        with (
            patch(
                "local_search_agent.scheduler.monitor.IndexMonitor",
                return_value=MagicMock(get_health_summary=MagicMock(return_value=mock_summary)),
            ),
            patch("local_search_agent.workspace.metadata_db.MetadataDB"),
        ):
            args = _args()
            cmd_health(args)
            captured = capsys.readouterr()
            assert "No workspaces registered" in captured.out


# ---------------------------------------------------------------------------
# cmd_ui
# ---------------------------------------------------------------------------


class TestCmdUi:
    def test_calls_dashboard_run_with_correct_args(self):
        from local_search_agent.cli.commands import cmd_ui

        with patch("local_search_agent.ui.dashboard.run") as mock_run:
            args = _args(
                host="0.0.0.0",
                port=9000,
                provider="ollama",
                model="mistral",
                meili_url="http://localhost:7700",
                meili_key="master",
                scheduler_interval=15,
                headless=True,
                multi_tenant=True,
                insecure_cookies=True,
            )
            cmd_ui(args)
            mock_run.assert_called_once_with(
                host="0.0.0.0",
                port=9000,
                db_path=":memory:",
                provider="ollama",
                model="mistral",
                meili_url="http://localhost:7700",
                meili_key="master",
                scheduler_interval=15,
                open_window=False,
                multi_tenant=True,
                insecure_cookies=True,
            )
