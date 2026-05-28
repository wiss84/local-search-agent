"""
Unit tests for SearchAgentConfig (core/config.py).

Tests cover:
- Default values
- validate() raises on bad provider
- validate() warns (not raises) on missing API key
- validate() raises on missing document_dir
- validate() raises on bad top_k / max_iterations
- api_key resolution priority in __post_init__
- to_dict() / from_dict() round-trip
- api_key excluded from to_dict()
- server_base_url, text_url, docs_url properties
- Single string document_dirs coerced to list
- index_name defaults to workspace_name
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_keys_path(tmp_path):
    keys_file = tmp_path / "keys.json"
    return patch(
        "local_search_agent.core.key_manager._keys_path",
        return_value=keys_file,
    )


def _make_config(tmp_path, **kwargs):
    from local_search_agent.core.config import SearchAgentConfig

    defaults = dict(
        document_dirs=[str(tmp_path)],
        workspace_name="test_ws",
        provider="ollama",
        db_path=str(tmp_path / "test.db"),
    )
    defaults.update(kwargs)
    return SearchAgentConfig(**defaults)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_default_workspace_name(self, tmp_path):
        from local_search_agent.core.config import SearchAgentConfig

        cfg = SearchAgentConfig(document_dirs=[str(tmp_path)], provider="ollama")
        assert cfg.workspace_name == "default"

    def test_default_provider(self, tmp_path):
        from local_search_agent.core.config import SearchAgentConfig

        cfg = SearchAgentConfig(document_dirs=[str(tmp_path)])
        assert cfg.provider == "google"

    def test_default_top_k(self, tmp_path):
        cfg = _make_config(tmp_path)
        assert cfg.top_k == 5

    def test_default_max_iterations(self, tmp_path):
        cfg = _make_config(tmp_path)
        assert cfg.max_iterations == 50

    def test_default_meilisearch_url(self, tmp_path):
        cfg = _make_config(tmp_path)
        assert cfg.meilisearch_url == "http://localhost:7700"

    def test_index_name_defaults_to_workspace_name(self, tmp_path):
        cfg = _make_config(tmp_path, workspace_name="finance")
        assert cfg.index_name == "finance"

    def test_index_name_can_be_overridden(self, tmp_path):
        cfg = _make_config(tmp_path, workspace_name="finance", index_name="finance_v2")
        assert cfg.index_name == "finance_v2"


# ---------------------------------------------------------------------------
# document_dirs coercion
# ---------------------------------------------------------------------------


class TestDocumentDirsCoercion:
    def test_single_string_coerced_to_list(self, tmp_path):
        from local_search_agent.core.config import SearchAgentConfig

        cfg = SearchAgentConfig(
            document_dirs=str(tmp_path),
            provider="ollama",
        )
        assert isinstance(cfg.document_dirs, list)
        assert cfg.document_dirs == [str(tmp_path)]


# ---------------------------------------------------------------------------
# API key resolution in __post_init__
# ---------------------------------------------------------------------------


class TestApiKeyResolution:
    def test_explicit_api_key_used_directly(self, tmp_path):
        with _patch_keys_path(tmp_path):
            cfg = _make_config(tmp_path, provider="google", api_key="explicit_key")
            assert cfg.api_key == "explicit_key"

    def test_saved_key_loaded_when_no_explicit_key(self, tmp_path):
        with _patch_keys_path(tmp_path):
            from local_search_agent.core.key_manager import set_key

            set_key("google", "saved_key_value")
            cfg = _make_config(tmp_path, provider="google")
            assert cfg.api_key == "saved_key_value"

    def test_env_var_used_as_fallback(self, tmp_path):
        with _patch_keys_path(tmp_path):
            with patch.dict(os.environ, {"GOOGLE_API_KEY": "env_key_value"}, clear=False):
                cfg = _make_config(tmp_path, provider="google")
                assert cfg.api_key == "env_key_value"

    def test_ollama_api_key_is_none(self, tmp_path):
        with _patch_keys_path(tmp_path):
            cfg = _make_config(tmp_path, provider="ollama")
            assert cfg.api_key is None


# ---------------------------------------------------------------------------
# validate()
# ---------------------------------------------------------------------------


class TestValidate:
    def test_valid_ollama_config_passes(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.validate()  # should not raise

    def test_unknown_provider_raises(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.provider = "bingus"
        with pytest.raises(ValueError, match="Unknown provider"):
            cfg.validate()

    def test_missing_document_dir_raises(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.document_dirs = ["/this/path/does/not/exist/ever"]
        with pytest.raises(ValueError, match="does not exist"):
            cfg.validate()

    def test_bad_top_k_raises(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.top_k = 0
        with pytest.raises(ValueError, match="top_k"):
            cfg.validate()

    def test_bad_max_iterations_raises(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.max_iterations = 0
        with pytest.raises(ValueError, match="max_iterations"):
            cfg.validate()

    def test_missing_api_key_logs_warning_not_raises(self, tmp_path, caplog):
        with _patch_keys_path(tmp_path):
            import logging

            cfg = _make_config(tmp_path, provider="google", api_key=None)
            # Ensure no Google key in env either
            env = {k: v for k, v in os.environ.items() if "GOOGLE" not in k}
            with patch.dict(os.environ, env, clear=True):
                with caplog.at_level(logging.WARNING):
                    cfg.validate()  # must NOT raise
            assert any("No API key" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


class TestSerialisation:
    def test_to_dict_from_dict_round_trip(self, tmp_path):
        from local_search_agent.core.config import SearchAgentConfig

        cfg = _make_config(tmp_path, workspace_name="finance", top_k=10)
        d = cfg.to_dict()
        restored = SearchAgentConfig.from_dict(d)
        assert restored.workspace_name == "finance"
        assert restored.top_k == 10
        assert restored.provider == "ollama"

    def test_api_key_excluded_from_to_dict(self, tmp_path):
        cfg = _make_config(tmp_path, api_key="super_secret_key", provider="ollama")
        d = cfg.to_dict()
        assert "api_key" not in d

    def test_to_dict_contains_expected_keys(self, tmp_path):
        cfg = _make_config(tmp_path)
        d = cfg.to_dict()
        for key in ("workspace_name", "provider", "model_name", "top_k", "max_iterations"):
            assert key in d


# ---------------------------------------------------------------------------
# URL properties
# ---------------------------------------------------------------------------


class TestUrlProperties:
    def test_server_base_url(self, tmp_path):
        cfg = _make_config(tmp_path, host="127.0.0.1", port=8000)
        assert cfg.server_base_url == "http://127.0.0.1:8000"

    def test_text_url(self, tmp_path):
        cfg = _make_config(tmp_path, host="127.0.0.1", port=8000)
        assert cfg.text_url("abc123") == "http://127.0.0.1:8000/text/abc123"

    def test_docs_url(self, tmp_path):
        cfg = _make_config(tmp_path, host="127.0.0.1", port=8000)
        assert cfg.docs_url("abc123") == "http://127.0.0.1:8000/docs/abc123"
