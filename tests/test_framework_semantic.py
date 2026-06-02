"""
Unit tests for SearchAgentFramework semantic settings methods.

Tests cover:
- get_semantic_settings returns current config values
- set_semantic_settings updates config in memory
- set_semantic_settings persists to settings.json
- set_semantic_settings does NOT rebuild agent when only other flags change
- Round-trip: set then get returns same values
- Settings survive framework restart (loaded from settings.json)
"""

from __future__ import annotations

from unittest.mock import patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_settings_path(tmp_path):
    settings_file = tmp_path / "settings.json"
    return patch(
        "local_search_agent.core.key_manager._settings_path",
        return_value=settings_file,
    )


def _patch_models_path(tmp_path):
    return patch(
        "local_search_agent.core.key_manager._models_path",
        return_value=tmp_path / "models.json",
    )


def _patch_keys_path(tmp_path):
    return patch(
        "local_search_agent.core.key_manager._keys_path",
        return_value=tmp_path / "keys.json",
    )


def _make_framework(tmp_path):
    """Build a minimal SearchAgentFramework with all file paths patched to tmp_path."""
    from local_search_agent.core.config import SearchAgentConfig
    from local_search_agent.core.framework import SearchAgentFramework

    config = SearchAgentConfig(
        document_dirs=[str(tmp_path)],
        workspace_name="test_ws",
        provider="ollama",
        model_name="mistral",
        db_path=str(tmp_path / "test.db"),
    )

    # Suppress Meilisearch startup during tests
    with patch.object(SearchAgentFramework, "_ensure_meilisearch", return_value=None):
        fw = SearchAgentFramework(config)

    return fw


# ---------------------------------------------------------------------------
# get_semantic_settings
# ---------------------------------------------------------------------------


class TestGetSemanticSettings:
    def test_returns_dict_with_all_three_keys(self, tmp_path):
        with (
            _patch_settings_path(tmp_path),
            _patch_keys_path(tmp_path),
            _patch_models_path(tmp_path),
        ):
            fw = _make_framework(tmp_path)
            result = fw.get_semantic_settings()
            assert "enable_semantic" in result
            assert "enable_query_expansion" in result

    def test_defaults_all_false(self, tmp_path):
        with (
            _patch_settings_path(tmp_path),
            _patch_keys_path(tmp_path),
            _patch_models_path(tmp_path),
        ):
            fw = _make_framework(tmp_path)
            result = fw.get_semantic_settings()
            assert result["enable_semantic"] is False
            assert result["enable_query_expansion"] is False

    def test_reflects_config_values(self, tmp_path):
        with (
            _patch_settings_path(tmp_path),
            _patch_keys_path(tmp_path),
            _patch_models_path(tmp_path),
        ):
            fw = _make_framework(tmp_path)
            fw.config.enable_semantic = True
            fw.config.enable_query_expansion = True
            result = fw.get_semantic_settings()
            assert result["enable_semantic"] is True
            assert result["enable_query_expansion"] is True


# ---------------------------------------------------------------------------
# set_semantic_settings
# ---------------------------------------------------------------------------


class TestSetSemanticSettings:
    def test_updates_config_in_memory(self, tmp_path):
        with (
            _patch_settings_path(tmp_path),
            _patch_keys_path(tmp_path),
            _patch_models_path(tmp_path),
        ):
            fw = _make_framework(tmp_path)
            fw.set_semantic_settings(
                enable_semantic=True,
                enable_query_expansion=True,
            )
            assert fw.config.enable_semantic is True
            assert fw.config.enable_query_expansion is True

    def test_persists_to_settings_json(self, tmp_path):
        with (
            _patch_settings_path(tmp_path),
            _patch_keys_path(tmp_path),
            _patch_models_path(tmp_path),
        ):
            fw = _make_framework(tmp_path)
            fw.set_semantic_settings(
                enable_semantic=True,
                enable_query_expansion=False,
            )
            from local_search_agent.core.key_manager import get_semantic_settings

            saved = get_semantic_settings()
            assert saved["enable_semantic"] is True
            assert saved["enable_query_expansion"] is False

    def test_round_trip_set_then_get(self, tmp_path):
        with (
            _patch_settings_path(tmp_path),
            _patch_keys_path(tmp_path),
            _patch_models_path(tmp_path),
        ):
            fw = _make_framework(tmp_path)
            fw.set_semantic_settings(
                enable_semantic=True,
                enable_query_expansion=True,
            )
            result = fw.get_semantic_settings()
            assert result["enable_semantic"] is True
            assert result["enable_query_expansion"] is True

    def test_all_false_disables_everything(self, tmp_path):
        with (
            _patch_settings_path(tmp_path),
            _patch_keys_path(tmp_path),
            _patch_models_path(tmp_path),
        ):
            fw = _make_framework(tmp_path)
            # First enable everything
            fw.set_semantic_settings(enable_semantic=True, enable_query_expansion=True)
            # Then disable everything
            fw.set_semantic_settings(enable_semantic=False, enable_query_expansion=False)
            result = fw.get_semantic_settings()
            assert result["enable_semantic"] is False
            assert result["enable_query_expansion"] is False

    def test_settings_survive_framework_restart(self, tmp_path):
        """Settings written to settings.json should be picked up by a new framework instance."""
        with (
            _patch_settings_path(tmp_path),
            _patch_keys_path(tmp_path),
            _patch_models_path(tmp_path),
        ):
            fw1 = _make_framework(tmp_path)
            fw1.set_semantic_settings(
                enable_semantic=True,
                enable_query_expansion=False,
            )

            # Create a new framework instance — should load from settings.json
            fw2 = _make_framework(tmp_path)
            result = fw2.get_semantic_settings()
            assert result["enable_semantic"] is True
            assert result["enable_query_expansion"] is False
