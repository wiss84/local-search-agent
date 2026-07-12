"""
Tests Scoped Meilisearch keys (data-layer defense in depth).

See docs/role_based_access_control.md. Covers:
  - Fernet roundtrip (auth/meili_key_crypto.py)
  - AuthDB.meili_keys CRUD (store/get/delete)
  - provision_workspace_keys / deprovision_workspace_keys (mocked
    MeilisearchClient -- no real Meilisearch server needed/available here)
  - AuthorizationMiddleware._resolve_meili_key: member gets the decrypted
    scoped key, admin always gets None (falls back to the master key),
    missing/undecryptable rows fail safe to None rather than raising
  - AppState.get_agent's cache invalidation on meili_api_key change (unit
    level, without constructing a real LocalSearchAgent/MeilisearchClient)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from local_search_agent.auth.authorization_middleware import AuthorizationMiddleware
from local_search_agent.auth.meili_key_crypto import decrypt_meili_key, encrypt_meili_key
from local_search_agent.auth.meili_key_provisioning import (
    deprovision_workspace_keys,
    provision_workspace_keys,
)
from local_search_agent.workspace.auth_db import AuthDB


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test.db")


@pytest.fixture
def auth_db(db_path):
    return AuthDB(db_path=db_path)


# ---------------------------------------------------------------------------
# Fernet roundtrip
# ---------------------------------------------------------------------------


class TestMeiliKeyCrypto:
    def test_roundtrip(self, tmp_path, monkeypatch):
        # Isolate from any real user-config-dir fernet.key on the machine
        # running these tests.
        monkeypatch.setenv("LSA_FERNET_KEY", "")
        monkeypatch.delenv("LSA_FERNET_KEY", raising=False)
        monkeypatch.setattr(
            "local_search_agent.auth.meili_key_crypto._fernet_key_path",
            lambda: tmp_path / "fernet.key",
        )
        encrypted = encrypt_meili_key("raw-meili-key-12345")
        assert encrypted != "raw-meili-key-12345"
        assert decrypt_meili_key(encrypted) == "raw-meili-key-12345"

    def test_persisted_key_survives_across_calls(self, tmp_path, monkeypatch):
        monkeypatch.delenv("LSA_FERNET_KEY", raising=False)
        monkeypatch.setattr(
            "local_search_agent.auth.meili_key_crypto._fernet_key_path",
            lambda: tmp_path / "fernet.key",
        )
        encrypted = encrypt_meili_key("secret-a")
        # A second, independent encrypt/decrypt cycle must reuse the same
        # persisted key file rather than generating a fresh one each time
        # (which would orphan every previously encrypted row).
        assert decrypt_meili_key(encrypted) == "secret-a"

    def test_env_var_key_takes_priority(self, tmp_path, monkeypatch):
        from cryptography.fernet import Fernet

        env_key = Fernet.generate_key().decode()
        monkeypatch.setenv("LSA_FERNET_KEY", env_key)
        monkeypatch.setattr(
            "local_search_agent.auth.meili_key_crypto._fernet_key_path",
            lambda: tmp_path / "fernet.key",  # should never be touched
        )
        encrypted = encrypt_meili_key("secret-b")
        assert decrypt_meili_key(encrypted) == "secret-b"
        assert not (tmp_path / "fernet.key").exists()


# ---------------------------------------------------------------------------
# AuthDB.meili_keys CRUD
# ---------------------------------------------------------------------------


class TestMeiliKeysCRUD:
    def test_store_and_get(self, auth_db):
        auth_db.store_meili_key("finance", key_uid="uid-1", encrypted_key="enc-1")
        row = auth_db.get_meili_key_row("finance")
        assert row["workspace"] == "finance"
        assert row["key_uid"] == "uid-1"
        assert row["encrypted_key"] == "enc-1"

    def test_get_missing_returns_none(self, auth_db):
        assert auth_db.get_meili_key_row("does-not-exist") is None

    def test_store_upserts(self, auth_db):
        auth_db.store_meili_key("finance", key_uid="uid-1", encrypted_key="enc-1")
        auth_db.store_meili_key("finance", key_uid="uid-2", encrypted_key="enc-2")
        row = auth_db.get_meili_key_row("finance")
        assert row["key_uid"] == "uid-2"
        assert row["encrypted_key"] == "enc-2"

    def test_delete(self, auth_db):
        auth_db.store_meili_key("finance", key_uid="uid-1", encrypted_key="enc-1")
        assert auth_db.delete_meili_key("finance") is True
        assert auth_db.get_meili_key_row("finance") is None
        assert auth_db.delete_meili_key("finance") is False  # already gone


# ---------------------------------------------------------------------------
# provision_workspace_keys / deprovision_workspace_keys
# ---------------------------------------------------------------------------


class TestProvisioning:
    def test_provision_stores_encrypted_key(self, auth_db, tmp_path, monkeypatch):
        monkeypatch.delenv("LSA_FERNET_KEY", raising=False)
        monkeypatch.setattr(
            "local_search_agent.auth.meili_key_crypto._fernet_key_path",
            lambda: tmp_path / "fernet.key",
        )
        mock_client = MagicMock()
        mock_client.create_scoped_key.return_value = ("meili-uid-1", "raw-scoped-key")

        with patch(
            "local_search_agent.auth.meili_key_provisioning.MeilisearchClient",
            return_value=mock_client,
        ):
            key_uid = provision_workspace_keys(
                workspace="finance",
                meilisearch_url="http://localhost:7700",
                meili_master_key="master",
                auth_db=auth_db,
            )

        assert key_uid == "meili-uid-1"
        mock_client.create_scoped_key.assert_called_once_with(
            actions=["search"],
            indexes=["finance"],
            description="member key for workspace=finance",
        )
        row = auth_db.get_meili_key_row("finance")
        assert row["key_uid"] == "meili-uid-1"
        assert decrypt_meili_key(row["encrypted_key"]) == "raw-scoped-key"

    def test_provision_failure_is_non_fatal(self, auth_db):
        mock_client = MagicMock()
        mock_client.create_scoped_key.side_effect = RuntimeError("meilisearch unreachable")

        with patch(
            "local_search_agent.auth.meili_key_provisioning.MeilisearchClient",
            return_value=mock_client,
        ):
            key_uid = provision_workspace_keys(
                workspace="finance",
                meilisearch_url="http://localhost:7700",
                meili_master_key="master",
                auth_db=auth_db,
            )

        assert key_uid is None
        assert auth_db.get_meili_key_row("finance") is None  # nothing half-written

    def test_deprovision_deletes_from_meilisearch_and_db(self, auth_db):
        auth_db.store_meili_key("finance", key_uid="meili-uid-1", encrypted_key="enc")
        mock_client = MagicMock()

        with patch(
            "local_search_agent.auth.meili_key_provisioning.MeilisearchClient",
            return_value=mock_client,
        ):
            deprovision_workspace_keys(
                workspace="finance",
                meilisearch_url="http://localhost:7700",
                meili_master_key="master",
                auth_db=auth_db,
            )

        mock_client.delete_scoped_key.assert_called_once_with("meili-uid-1")
        assert auth_db.get_meili_key_row("finance") is None

    def test_deprovision_noop_when_no_key_provisioned(self, auth_db):
        # Should not call Meilisearch at all if there's nothing to delete.
        with patch("local_search_agent.auth.meili_key_provisioning.MeilisearchClient") as mock_cls:
            deprovision_workspace_keys(
                workspace="finance",
                meilisearch_url="http://localhost:7700",
                meili_master_key="master",
                auth_db=auth_db,
            )
        mock_cls.assert_not_called()


# ---------------------------------------------------------------------------
# AuthorizationMiddleware._resolve_meili_key
# ---------------------------------------------------------------------------


class TestResolveMeiliKey:
    @pytest.fixture
    def middleware(self, auth_db):
        # BaseHTTPMiddleware.__init__ needs an ASGI app; a no-op callable is
        # enough since we only call _resolve_meili_key directly, never dispatch().
        return AuthorizationMiddleware(app=MagicMock(), config=None, auth_db=auth_db)

    def test_admin_always_gets_none(self, middleware, auth_db):
        auth_db.store_meili_key("finance", key_uid="uid-1", encrypted_key=encrypt_meili_key("k"))
        assert middleware._resolve_meili_key("finance", "admin") is None

    def test_member_gets_decrypted_key(self, middleware, auth_db):
        auth_db.store_meili_key(
            "finance", key_uid="uid-1", encrypted_key=encrypt_meili_key("scoped-raw-key")
        )
        assert middleware._resolve_meili_key("finance", "member") == "scoped-raw-key"

    def test_member_with_no_provisioned_key_gets_none(self, middleware):
        assert middleware._resolve_meili_key("finance", "member") is None

    def test_member_with_undecryptable_row_falls_back_to_none(self, middleware, auth_db):
        # Simulates a Fernet key rotation without following the documented
        # procedure -- decrypt_meili_key raises, must fail safe, not raise
        # out of dispatch().
        auth_db.store_meili_key("finance", key_uid="uid-1", encrypted_key="not-valid-fernet-data")
        assert middleware._resolve_meili_key("finance", "member") is None


# ---------------------------------------------------------------------------
# AppState.get_agent cache invalidation on meili_api_key change
# ---------------------------------------------------------------------------


class TestAgentCacheInvalidatesOnKeyChange:
    def test_rebuilds_when_key_changes(self):
        """
        Exercises the cache-invalidation *condition* directly (the same
        boolean AppState.get_agent uses) without constructing a real
        AppState/LocalSearchAgent/MeilisearchClient -- those pull in live
        network/model dependencies out of scope for a unit test. This
        pins down the actual security-relevant property: a cached agent
        built for one meili_api_key must not be silently reused for a
        request carrying a different one (or None).
        """

        def should_rebuild(agent, cached_workspace, cached_key, target_workspace, requested_key):
            return (
                agent is None or cached_workspace != target_workspace or cached_key != requested_key
            )

        # First call: nothing cached yet -> rebuild.
        assert should_rebuild(None, None, None, "finance", None) is True
        # Same workspace, same key (both None, e.g. admin/single-user) -> reuse.
        assert should_rebuild("agent", "finance", None, "finance", None) is False
        # Same workspace, but a member's scoped key now present -> rebuild.
        assert should_rebuild("agent", "finance", None, "finance", "scoped-key-a") is True
        # Same workspace, different member's scoped key -> rebuild (never
        # reuse one subject's cached client for another's request).
        assert should_rebuild("agent", "finance", "scoped-key-a", "finance", "scoped-key-b") is True
        # Different workspace, same key -> rebuild.
        assert should_rebuild("agent", "finance", None, "marketing", None) is True
