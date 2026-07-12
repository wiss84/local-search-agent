"""
Regression test for a pre-existing CLI bug found while closing the
"CLI workspace creation isn't wired to Meilisearch key provisioning" gap

cmd_workspace_create/cmd_workspace_delete already referenced
args.multi_tenant/args.meili_url/args.meili_key, but the `workspace create`
and `workspace delete` subparsers never defined those flags -- any
invocation of `local-search workspace create <name> <dir>` (with or
without --multi-tenant) would have raised AttributeError the moment
cmd_workspace_create read args.multi_tenant. Guards against that
regressing silently again.
"""

from __future__ import annotations

from local_search_agent.cli.commands import build_parser


class TestWorkspaceCreateDeleteParserFlags:
    def test_create_parses_without_multi_tenant_flag(self, tmp_path):
        parser = build_parser()
        args = parser.parse_args(
            ["--db", str(tmp_path / "test.db"), "workspace", "create", "myws", str(tmp_path)]
        )
        # These attributes must exist (with sane defaults) even when
        # --multi-tenant wasn't passed -- cmd_workspace_create reads them
        # unconditionally.
        assert args.multi_tenant is False
        assert args.meili_url == "http://localhost:7700"
        assert args.meili_key == "local_search_master_key"

    def test_create_parses_with_multi_tenant_flag(self, tmp_path):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--db",
                str(tmp_path / "test.db"),
                "workspace",
                "create",
                "myws",
                str(tmp_path),
                "--multi-tenant",
                "--meili-url",
                "http://example:7700",
                "--meili-key",
                "custom-key",
            ]
        )
        assert args.multi_tenant is True
        assert args.meili_url == "http://example:7700"
        assert args.meili_key == "custom-key"

    def test_delete_parses_without_multi_tenant_flag(self, tmp_path):
        parser = build_parser()
        args = parser.parse_args(["--db", str(tmp_path / "test.db"), "workspace", "delete", "myws"])
        assert args.multi_tenant is False
        assert args.meili_url == "http://localhost:7700"
        assert args.meili_key == "local_search_master_key"
