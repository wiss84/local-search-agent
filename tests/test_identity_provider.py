"""
Unit tests for multi-tenant RBAC: Identity protocol + APIKeyIdentityProvider.

Covers:
- Identity dataclass defaults
- AuthDB.api_keys CRUD (create / get / revoke / list)
- APIKeyIdentityProvider: create_key / verify_key (valid, wrong secret, unknown
  key_id, revoked, malformed) / revoke_key / list_keys / resolve()
- SearchAgentFramework thin-wrapper methods: grant/revoke/list workspace
  access, get_workspace_role, create/revoke/list API keys
- CLI commands: grant-access / revoke-access / list-access / auth create-key /
  auth revoke-key / auth list-keys
"""

from __future__ import annotations

from io import StringIO
from unittest.mock import patch

import pytest

from local_search_agent.auth.api_key_provider import APIKeyIdentityProvider
from local_search_agent.auth.identity import Identity
from local_search_agent.workspace.auth_db import AuthDB

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def auth_db(tmp_path):
    return AuthDB(db_path=str(tmp_path / "test.db"))


@pytest.fixture
def provider(auth_db):
    return APIKeyIdentityProvider(auth_db)


class _FakeRequest:
    """Minimal request stand-in — resolve() only needs a dict-like .headers."""

    def __init__(self, headers: dict, cookies: dict = None):
        self.headers = headers
        self.cookies = cookies or {}


# ---------------------------------------------------------------------------
# Identity dataclass
# ---------------------------------------------------------------------------


class TestIdentity:
    def test_defaults(self):
        identity = Identity(subject="alice@acme.com")
        assert identity.subject == "alice@acme.com"
        assert identity.display_name == ""
        assert identity.is_superadmin is False

    def test_full_construction(self):
        identity = Identity(subject="bob@acme.com", display_name="Bob", is_superadmin=True)
        assert identity.display_name == "Bob"
        assert identity.is_superadmin is True


# ---------------------------------------------------------------------------
# AuthDB.api_keys CRUD
# ---------------------------------------------------------------------------


class TestAuthDBApiKeys:
    def test_create_and_get(self, auth_db):
        auth_db.create_api_key(
            key_id="abc123", subject="alice@acme.com", key_hash="hashed", created_by="admin"
        )
        row = auth_db.get_api_key("abc123")
        assert row is not None
        assert row["subject"] == "alice@acme.com"
        assert row["revoked_at"] is None

    def test_get_unknown_returns_none(self, auth_db):
        assert auth_db.get_api_key("nonexistent") is None

    def test_revoke_sets_revoked_at(self, auth_db):
        auth_db.create_api_key(
            key_id="abc123", subject="alice@acme.com", key_hash="hashed", created_by="admin"
        )
        revoked = auth_db.revoke_api_key("abc123")
        assert revoked is True
        row = auth_db.get_api_key("abc123")
        assert row["revoked_at"] is not None

    def test_revoke_already_revoked_returns_false(self, auth_db):
        auth_db.create_api_key(
            key_id="abc123", subject="alice@acme.com", key_hash="hashed", created_by="admin"
        )
        auth_db.revoke_api_key("abc123")
        assert auth_db.revoke_api_key("abc123") is False

    def test_revoke_unknown_returns_false(self, auth_db):
        assert auth_db.revoke_api_key("nonexistent") is False

    def test_list_filters_by_subject(self, auth_db):
        auth_db.create_api_key(
            key_id="k1", subject="alice@acme.com", key_hash="h1", created_by="admin"
        )
        auth_db.create_api_key(
            key_id="k2", subject="bob@acme.com", key_hash="h2", created_by="admin"
        )
        rows = auth_db.list_api_keys(subject="alice@acme.com")
        assert len(rows) == 1
        assert rows[0]["key_id"] == "k1"

    def test_list_without_filter_returns_all(self, auth_db):
        auth_db.create_api_key(
            key_id="k1", subject="alice@acme.com", key_hash="h1", created_by="admin"
        )
        auth_db.create_api_key(
            key_id="k2", subject="bob@acme.com", key_hash="h2", created_by="admin"
        )
        assert len(auth_db.list_api_keys()) == 2


# ---------------------------------------------------------------------------
# APIKeyIdentityProvider: create / verify / revoke / list
# ---------------------------------------------------------------------------


class TestAPIKeyIdentityProvider:
    def test_create_key_returns_id_and_raw_key(self, provider):
        key_id, raw_key = provider.create_key(subject="alice@acme.com", created_by="admin")
        assert key_id
        assert raw_key.startswith("lsa_")
        assert key_id in raw_key

    def test_raw_key_never_persisted(self, provider, auth_db):
        key_id, raw_key = provider.create_key(subject="alice@acme.com", created_by="admin")
        row = auth_db.get_api_key(key_id)
        assert raw_key not in row["key_hash"]
        assert row["key_hash"] != raw_key

    def test_verify_valid_key_returns_identity(self, provider):
        _, raw_key = provider.create_key(
            subject="alice@acme.com", created_by="admin", display_name="Alice"
        )
        identity = provider.verify_key(raw_key)
        assert identity is not None
        assert identity.subject == "alice@acme.com"
        assert identity.display_name == "Alice"
        assert identity.is_superadmin is False

    def test_verify_superadmin_flag(self, provider):
        _, raw_key = provider.create_key(
            subject="root@acme.com", created_by="admin", is_superadmin=True
        )
        identity = provider.verify_key(raw_key)
        assert identity.is_superadmin is True

    def test_verify_wrong_secret_returns_none(self, provider):
        key_id, raw_key = provider.create_key(subject="alice@acme.com", created_by="admin")
        tampered = f"lsa_{key_id}_wrongsecretvalue"
        assert provider.verify_key(tampered) is None

    def test_verify_unknown_key_id_returns_none(self, provider):
        assert provider.verify_key("lsa_doesnotexist_somesecret") is None

    def test_verify_malformed_key_returns_none(self, provider):
        assert provider.verify_key("not-a-valid-key-format") is None
        assert provider.verify_key("") is None
        assert provider.verify_key("wrongprefix_abc_def") is None

    def test_verify_revoked_key_returns_none(self, provider):
        key_id, raw_key = provider.create_key(subject="alice@acme.com", created_by="admin")
        provider.revoke_key(key_id)
        assert provider.verify_key(raw_key) is None

    def test_revoke_key_returns_true_when_found(self, provider):
        key_id, _ = provider.create_key(subject="alice@acme.com", created_by="admin")
        assert provider.revoke_key(key_id) is True

    def test_revoke_key_returns_false_when_not_found(self, provider):
        assert provider.revoke_key("nonexistent") is False

    def test_list_keys_excludes_hash(self, provider):
        provider.create_key(subject="alice@acme.com", created_by="admin")
        rows = provider.list_keys()
        assert len(rows) == 1
        assert "key_hash" not in rows[0]

    def test_list_keys_filters_by_subject(self, provider):
        provider.create_key(subject="alice@acme.com", created_by="admin")
        provider.create_key(subject="bob@acme.com", created_by="admin")
        rows = provider.list_keys(subject="alice@acme.com")
        assert len(rows) == 1
        assert rows[0]["subject"] == "alice@acme.com"


# ---------------------------------------------------------------------------
# APIKeyIdentityProvider.resolve() — IdentityProvider protocol
# ---------------------------------------------------------------------------


class TestResolve:
    def test_resolve_valid_bearer_header(self, provider):
        _, raw_key = provider.create_key(subject="alice@acme.com", created_by="admin")
        request = _FakeRequest(headers={"Authorization": f"Bearer {raw_key}"})
        identity = provider.resolve(request)
        assert identity is not None
        assert identity.subject == "alice@acme.com"

    def test_resolve_case_insensitive_bearer(self, provider):
        _, raw_key = provider.create_key(subject="alice@acme.com", created_by="admin")
        request = _FakeRequest(headers={"Authorization": f"bearer {raw_key}"})
        assert provider.resolve(request) is not None

    def test_resolve_missing_header_returns_none(self, provider):
        request = _FakeRequest(headers={})
        assert provider.resolve(request) is None

    def test_resolve_non_bearer_scheme_returns_none(self, provider):
        request = _FakeRequest(headers={"Authorization": "Basic dXNlcjpwYXNz"})
        assert provider.resolve(request) is None

    def test_resolve_invalid_key_returns_none(self, provider):
        request = _FakeRequest(headers={"Authorization": "Bearer lsa_bad_key"})
        assert provider.resolve(request) is None

    def test_resolve_valid_session_cookie(self, provider):
        _, raw_key = provider.create_key(subject="alice@acme.com", created_by="admin")
        token, _ = provider.login(raw_key)
        request = _FakeRequest(headers={}, cookies={"lsa_session": token})
        identity = provider.resolve(request)
        assert identity is not None
        assert identity.subject == "alice@acme.com"

    def test_resolve_invalid_session_cookie_falls_through_to_bearer(self, provider):
        _, raw_key = provider.create_key(subject="alice@acme.com", created_by="admin")
        request = _FakeRequest(
            headers={"Authorization": f"Bearer {raw_key}"},
            cookies={"lsa_session": "bogus-token"},
        )
        identity = provider.resolve(request)
        assert identity is not None


# ---------------------------------------------------------------------------
# SearchAgentFramework thin-wrapper methods
# ---------------------------------------------------------------------------


@pytest.fixture
def framework(tmp_path):
    from local_search_agent.core.config import SearchAgentConfig
    from local_search_agent.core.framework import SearchAgentFramework

    config = SearchAgentConfig(workspace_name="default", db_path=str(tmp_path / "test.db"))
    return SearchAgentFramework(config)


class TestFrameworkWorkspaceAccess:
    def test_grant_and_get_role(self, framework):
        framework.grant_workspace_access(
            workspaces=["finance"], subject="alice@acme.com", role="member", granted_by="admin"
        )
        assert framework.get_workspace_role("alice@acme.com", "finance") == "member"

    def test_grant_multiple_workspaces(self, framework):
        framework.grant_workspace_access(
            workspaces=["finance", "marketing"],
            subject="alice@acme.com",
            role="admin",
            granted_by="admin",
        )
        assert framework.get_workspace_role("alice@acme.com", "finance") == "admin"
        assert framework.get_workspace_role("alice@acme.com", "marketing") == "admin"

    def test_revoke_specific_workspace(self, framework):
        framework.grant_workspace_access(
            workspaces=["finance", "marketing"],
            subject="alice@acme.com",
            role="member",
            granted_by="admin",
        )
        deleted = framework.revoke_workspace_access("alice@acme.com", workspaces=["finance"])
        assert deleted == 1
        assert framework.get_workspace_role("alice@acme.com", "finance") is None
        assert framework.get_workspace_role("alice@acme.com", "marketing") == "member"

    def test_revoke_all(self, framework):
        framework.grant_workspace_access(
            workspaces=["finance", "marketing"],
            subject="alice@acme.com",
            role="member",
            granted_by="admin",
        )
        deleted = framework.revoke_workspace_access("alice@acme.com")
        assert deleted == 2

    def test_list_access(self, framework):
        framework.grant_workspace_access(
            workspaces=["finance"], subject="alice@acme.com", role="admin", granted_by="admin"
        )
        rows = framework.list_workspace_access(workspace="finance")
        assert len(rows) == 1
        assert rows[0]["subject"] == "alice@acme.com"

    def test_invalid_role_raises(self, framework):
        with pytest.raises(ValueError):
            framework.grant_workspace_access(
                workspaces=["finance"], subject="alice@acme.com", role="owner", granted_by="admin"
            )

    def test_grant_invalid_role_raises_via_framework(self, framework):
        with pytest.raises(ValueError):
            framework.grant_workspace_access(
                workspaces=["finance"],
                subject="alice@acme.com",
                role="owner",
                granted_by="admin",
            )


class TestFrameworkApiKeys:
    def test_create_key(self, framework):
        key_id, raw_key = framework.create_api_key(subject="alice@acme.com", created_by="admin")
        assert key_id
        assert raw_key.startswith("lsa_")

    def test_revoke_key(self, framework):
        key_id, _ = framework.create_api_key(subject="alice@acme.com", created_by="admin")
        assert framework.revoke_api_key(key_id) is True

    def test_list_keys(self, framework):
        framework.create_api_key(subject="alice@acme.com", created_by="admin")
        rows = framework.list_api_keys()
        assert len(rows) == 1
        assert "key_hash" not in rows[0]


# ---------------------------------------------------------------------------
# CLI: grant-access / revoke-access / list-access / auth create-key / etc.
# ---------------------------------------------------------------------------


def _run(args: list[str]) -> tuple[int, str, str]:
    """Run the CLI with the given args list. Returns (exit_code, stdout, stderr)."""
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


class TestCLIAccessCommands:
    def test_grant_access_single_workspace(self, tmp_path):
        db = str(tmp_path / "test.db")
        code, out, err = _run(
            [
                "--db",
                db,
                "grant-access",
                "--subject",
                "alice@acme.com",
                "--workspace",
                "finance",
                "--role",
                "member",
                "--granted-by",
                "admin@acme.com",
            ]
        )
        assert code == 0
        assert "alice@acme.com" in out
        assert "finance" in out

    def test_grant_access_multiple_workspaces(self, tmp_path):
        db = str(tmp_path / "test.db")
        code, out, err = _run(
            [
                "--db",
                db,
                "grant-access",
                "--subject",
                "alice@acme.com",
                "--workspace",
                "finance",
                "marketing",
                "--role",
                "admin",
                "--granted-by",
                "admin@acme.com",
            ]
        )
        assert code == 0
        assert "finance" in out
        assert "marketing" in out

    def test_revoke_access_specific_workspace(self, tmp_path):
        db = str(tmp_path / "test.db")
        _run(
            [
                "--db",
                db,
                "grant-access",
                "--subject",
                "alice@acme.com",
                "--workspace",
                "finance",
                "--role",
                "member",
            ]
        )
        code, out, err = _run(
            ["--db", db, "revoke-access", "--subject", "alice@acme.com", "--workspace", "finance"]
        )
        assert code == 0
        assert "finance" in out

    def test_revoke_access_all(self, tmp_path):
        db = str(tmp_path / "test.db")
        _run(
            [
                "--db",
                db,
                "grant-access",
                "--subject",
                "alice@acme.com",
                "--workspace",
                "finance",
                "--role",
                "member",
            ]
        )
        code, out, err = _run(["--db", db, "revoke-access", "--subject", "alice@acme.com"])
        assert code == 0
        assert "all" in out.lower()

    def test_list_access_shows_grant(self, tmp_path):
        db = str(tmp_path / "test.db")
        _run(
            [
                "--db",
                db,
                "grant-access",
                "--subject",
                "alice@acme.com",
                "--workspace",
                "finance",
                "--role",
                "admin",
            ]
        )
        code, out, err = _run(["--db", db, "list-access", "--workspace", "finance"])
        assert code == 0
        assert "alice@acme.com" in out
        assert "admin" in out

    def test_list_access_empty(self, tmp_path):
        db = str(tmp_path / "test.db")
        code, out, err = _run(["--db", db, "list-access", "--workspace", "nonexistent"])
        assert code == 0
        assert "no grants" in out.lower()


class TestCLIAuthKeyCommands:
    def test_create_key_shows_raw_key_once(self, tmp_path):
        db = str(tmp_path / "test.db")
        code, out, err = _run(["--db", db, "auth", "create-key", "--subject", "alice@acme.com"])
        assert code == 0
        assert "lsa_" in out
        assert "not be shown again" in out.lower()

    def test_list_keys_after_create(self, tmp_path):
        db = str(tmp_path / "test.db")
        _run(["--db", db, "auth", "create-key", "--subject", "alice@acme.com"])
        code, out, err = _run(["--db", db, "auth", "list-keys"])
        assert code == 0
        assert "alice@acme.com" in out
        assert "active" in out.lower()

    def test_list_keys_empty(self, tmp_path):
        db = str(tmp_path / "test.db")
        code, out, err = _run(["--db", db, "auth", "list-keys"])
        assert code == 0
        assert "no api keys" in out.lower()

    def test_revoke_key(self, tmp_path):
        db = str(tmp_path / "test.db")
        _, out, _ = _run(["--db", db, "auth", "create-key", "--subject", "alice@acme.com"])
        # Extract key_id from the "key_id=..." confirmation line.
        key_id = out.split("key_id=")[1].split(")")[0]
        code, out2, err = _run(["--db", db, "auth", "revoke-key", key_id])
        assert code == 0
        assert "revoked" in out2.lower()

    def test_revoke_unknown_key(self, tmp_path):
        db = str(tmp_path / "test.db")
        code, out, err = _run(["--db", db, "auth", "revoke-key", "nonexistent"])
        assert code == 0
        assert "no active key" in out.lower()

    def test_auth_help_does_not_crash(self):
        code, out, err = _run(["auth", "--help"])
        assert code == 0
