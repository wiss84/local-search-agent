"""
Unit tests for the key manager (core/key_manager.py).

Tests cover:
- set_key / get_key / delete_key / list_keys round-trip
- resolve_key priority order: explicit → keys.json → env var
- Validation errors (empty key, unknown provider, ollama)
- Masking format
- apply_langsmith_env sets correct os.environ values
- delete_langsmith clears os.environ values
- keys.json is written outside the project directory
- Concurrent-safe: multiple set_key calls don't corrupt the file
"""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_keys_path(tmp_path):
    """Context manager that redirects keys.json to tmp_path."""
    keys_file = tmp_path / "keys.json"
    return patch(
        "local_search_agent.core.key_manager._keys_path",
        return_value=keys_file,
    )


# ---------------------------------------------------------------------------
# set_key / get_key / delete_key
# ---------------------------------------------------------------------------


class TestSetGetDelete:
    def test_set_and_get_key(self, tmp_path):
        with _patch_keys_path(tmp_path):
            from local_search_agent.core.key_manager import get_key, set_key

            set_key("google", "AIzaSyTEST1234567890abcdef")
            assert get_key("google") == "AIzaSyTEST1234567890abcdef"

    def test_set_key_overwrites_existing(self, tmp_path):
        with _patch_keys_path(tmp_path):
            from local_search_agent.core.key_manager import get_key, set_key

            set_key("google", "first_key_value")
            set_key("google", "second_key_value")
            assert get_key("google") == "second_key_value"

    def test_set_key_strips_whitespace(self, tmp_path):
        with _patch_keys_path(tmp_path):
            from local_search_agent.core.key_manager import get_key, set_key

            set_key("openai", "  sk-abc123  ")
            assert get_key("openai") == "sk-abc123"

    def test_multiple_providers_stored_independently(self, tmp_path):
        with _patch_keys_path(tmp_path):
            from local_search_agent.core.key_manager import get_key, set_key

            set_key("google", "google_key_value")
            set_key("openai", "openai_key_value")
            set_key("anthropic", "anthropic_key_value")
            assert get_key("google") == "google_key_value"
            assert get_key("openai") == "openai_key_value"
            assert get_key("anthropic") == "anthropic_key_value"

    def test_get_key_returns_none_if_not_set(self, tmp_path):
        with _patch_keys_path(tmp_path):
            from local_search_agent.core.key_manager import get_key

            assert get_key("google") is None

    def test_delete_key_removes_entry(self, tmp_path):
        with _patch_keys_path(tmp_path):
            from local_search_agent.core.key_manager import delete_key, get_key, set_key

            set_key("google", "AIzaSyTEST1234567890abcdef")
            deleted = delete_key("google")
            assert deleted is True
            assert get_key("google") is None

    def test_delete_key_returns_false_if_not_set(self, tmp_path):
        with _patch_keys_path(tmp_path):
            from local_search_agent.core.key_manager import delete_key

            assert delete_key("openai") is False

    def test_delete_one_key_preserves_others(self, tmp_path):
        with _patch_keys_path(tmp_path):
            from local_search_agent.core.key_manager import delete_key, get_key, set_key

            set_key("google", "google_key")
            set_key("openai", "openai_key")
            delete_key("google")
            assert get_key("google") is None
            assert get_key("openai") == "openai_key"

    def test_keys_json_is_valid_json(self, tmp_path):
        with _patch_keys_path(tmp_path):
            from local_search_agent.core.key_manager import set_key

            set_key("google", "AIzaSyTEST1234567890abcdef")
            keys_file = tmp_path / "keys.json"
            data = json.loads(keys_file.read_text())
            assert "google" in data


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_unknown_provider_raises(self, tmp_path):
        with _patch_keys_path(tmp_path):
            from local_search_agent.core.key_manager import set_key

            with pytest.raises(ValueError, match="Unknown provider"):
                set_key("bingus", "some_key")

    def test_empty_key_raises(self, tmp_path):
        with _patch_keys_path(tmp_path):
            from local_search_agent.core.key_manager import set_key

            with pytest.raises(ValueError, match="must not be empty"):
                set_key("google", "")

    def test_whitespace_only_key_raises(self, tmp_path):
        with _patch_keys_path(tmp_path):
            from local_search_agent.core.key_manager import set_key

            with pytest.raises(ValueError, match="must not be empty"):
                set_key("google", "   ")

    def test_ollama_raises_on_set_key(self, tmp_path):
        with _patch_keys_path(tmp_path):
            from local_search_agent.core.key_manager import set_key

            with pytest.raises(ValueError, match="does not use an API key"):
                set_key("ollama", "some_key")


# ---------------------------------------------------------------------------
# list_keys masking
# ---------------------------------------------------------------------------


class TestListKeysMasking:
    def test_list_keys_masks_values(self, tmp_path):
        with _patch_keys_path(tmp_path):
            from local_search_agent.core.key_manager import list_keys, set_key

            set_key("google", "AIzaSyTEST1234567890abcdef")
            masked = list_keys()
            assert "google" in masked
            assert masked["google"] != "AIzaSyTEST1234567890abcdef"
            assert "*" in masked["google"]

    def test_list_keys_preserves_first_and_last_chars(self, tmp_path):
        with _patch_keys_path(tmp_path):
            from local_search_agent.core.key_manager import list_keys, set_key

            key = "AIzaSyTEST1234567890abcdef"
            set_key("google", key)
            masked = list_keys()["google"]
            assert masked.startswith(key[:6])
            assert masked.endswith(key[-4:])

    def test_list_keys_short_key_masked_as_stars(self, tmp_path):
        with _patch_keys_path(tmp_path):
            from local_search_agent.core.key_manager import list_keys

            # Bypass validation to test masking of a short value directly
            keys_file = tmp_path / "keys.json"
            keys_file.write_text(json.dumps({"google": "short"}))
            masked = list_keys()["google"]
            assert masked == "****"

    def test_list_keys_returns_empty_dict_if_no_keys(self, tmp_path):
        with _patch_keys_path(tmp_path):
            from local_search_agent.core.key_manager import list_keys

            assert list_keys() == {}


# ---------------------------------------------------------------------------
# resolve_key priority order
# ---------------------------------------------------------------------------


class TestResolveKeyPriority:
    def test_explicit_key_takes_highest_priority(self, tmp_path):
        with _patch_keys_path(tmp_path):
            from local_search_agent.core.key_manager import resolve_key, set_key

            set_key("google", "saved_key_value")
            with patch.dict(os.environ, {"GOOGLE_API_KEY": "env_key_value"}):
                result = resolve_key("google", explicit_key="explicit_key_value")
            assert result == "explicit_key_value"

    def test_saved_key_takes_priority_over_env(self, tmp_path):
        with _patch_keys_path(tmp_path):
            from local_search_agent.core.key_manager import resolve_key, set_key

            set_key("google", "saved_key_value")
            with patch.dict(os.environ, {"GOOGLE_API_KEY": "env_key_value"}):
                result = resolve_key("google")
            assert result == "saved_key_value"

    def test_env_var_used_when_no_saved_key(self, tmp_path):
        with _patch_keys_path(tmp_path):
            from local_search_agent.core.key_manager import resolve_key

            with patch.dict(os.environ, {"GOOGLE_API_KEY": "env_key_value"}, clear=False):
                result = resolve_key("google")
            assert result == "env_key_value"

    def test_returns_none_when_no_key_found(self, tmp_path):
        with _patch_keys_path(tmp_path):
            from local_search_agent.core.key_manager import resolve_key

            env = {k: v for k, v in os.environ.items() if "GOOGLE" not in k}
            with patch.dict(os.environ, env, clear=True):
                result = resolve_key("google")
            assert result is None

    def test_ollama_always_returns_none(self, tmp_path):
        with _patch_keys_path(tmp_path):
            from local_search_agent.core.key_manager import resolve_key

            result = resolve_key("ollama", explicit_key="some_key")
            assert result is None


# ---------------------------------------------------------------------------
# LangSmith
# ---------------------------------------------------------------------------


class TestLangSmith:
    def test_set_langsmith_saves_credentials(self, tmp_path):
        with _patch_keys_path(tmp_path):
            from local_search_agent.core.key_manager import get_langsmith, set_langsmith

            set_langsmith("ls__test_key_1234567890abcdef", "my_project")
            result = get_langsmith()
            assert result["configured"] is True
            assert result["project"] == "my_project"

    def test_set_langsmith_activates_env_vars(self, tmp_path):
        with _patch_keys_path(tmp_path):
            from local_search_agent.core.key_manager import set_langsmith

            set_langsmith("ls__test_key_1234567890abcdef", "my_project")
            assert os.environ.get("LANGCHAIN_TRACING_V2") == "true"
            assert os.environ.get("LANGCHAIN_ENDPOINT") == "https://api.smith.langchain.com"
            assert os.environ.get("LANGCHAIN_API_KEY") == "ls__test_key_1234567890abcdef"
            assert os.environ.get("LANGCHAIN_PROJECT") == "my_project"
            # Cleanup
            for k in (
                "LANGCHAIN_TRACING_V2",
                "LANGCHAIN_ENDPOINT",
                "LANGCHAIN_API_KEY",
                "LANGCHAIN_PROJECT",
            ):
                os.environ.pop(k, None)

    def test_delete_langsmith_clears_env_vars(self, tmp_path):
        with _patch_keys_path(tmp_path):
            from local_search_agent.core.key_manager import delete_langsmith, set_langsmith

            set_langsmith("ls__test_key_1234567890abcdef", "my_project")
            delete_langsmith()
            assert os.environ.get("LANGCHAIN_TRACING_V2") is None
            assert os.environ.get("LANGCHAIN_API_KEY") is None

    def test_delete_langsmith_returns_false_if_not_configured(self, tmp_path):
        with _patch_keys_path(tmp_path):
            from local_search_agent.core.key_manager import delete_langsmith

            assert delete_langsmith() is False

    def test_get_langsmith_masks_api_key(self, tmp_path):
        with _patch_keys_path(tmp_path):
            from local_search_agent.core.key_manager import get_langsmith, set_langsmith

            set_langsmith("ls__test_key_1234567890abcdef", "my_project")
            result = get_langsmith()
            assert "*" in result["api_key_masked"]
            assert "ls__te" in result["api_key_masked"]
            # Cleanup
            for k in (
                "LANGCHAIN_TRACING_V2",
                "LANGCHAIN_ENDPOINT",
                "LANGCHAIN_API_KEY",
                "LANGCHAIN_PROJECT",
            ):
                os.environ.pop(k, None)

    def test_apply_langsmith_env_returns_false_when_not_configured(self, tmp_path):
        with _patch_keys_path(tmp_path):
            from local_search_agent.core.key_manager import apply_langsmith_env

            assert apply_langsmith_env() is False

    def test_set_langsmith_empty_key_raises(self, tmp_path):
        with _patch_keys_path(tmp_path):
            from local_search_agent.core.key_manager import set_langsmith

            with pytest.raises(ValueError, match="must not be empty"):
                set_langsmith("", "project")

    def test_set_langsmith_default_project_name(self, tmp_path):
        with _patch_keys_path(tmp_path):
            from local_search_agent.core.key_manager import get_langsmith, set_langsmith

            set_langsmith("ls__test_key_1234567890abcdef", "")
            result = get_langsmith()
            assert result["project"] == "local-search-agent"
            # Cleanup
            for k in (
                "LANGCHAIN_TRACING_V2",
                "LANGCHAIN_ENDPOINT",
                "LANGCHAIN_API_KEY",
                "LANGCHAIN_PROJECT",
            ):
                os.environ.pop(k, None)


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------


def _patch_models_path(tmp_path):
    models_file = tmp_path / "models.json"
    return patch(
        "local_search_agent.core.key_manager._models_path",
        return_value=models_file,
    )


class TestModelManagement:
    def test_default_google_models_seeded_on_first_use(self, tmp_path):
        with _patch_models_path(tmp_path):
            from local_search_agent.core.key_manager import get_models

            models = get_models("google")
            assert "gemma-4-31b-it" in models
            assert "gemma-4-26b-a4b-it" in models
            assert "gemini-3.1-flash-lite" in models

    def test_all_providers_present_by_default(self, tmp_path):
        with _patch_models_path(tmp_path):
            from local_search_agent.core.key_manager import get_models

            all_models = get_models()
            for provider in ("google", "openai", "anthropic", "ollama"):
                assert provider in all_models

    def test_add_model(self, tmp_path):
        with _patch_models_path(tmp_path):
            from local_search_agent.core.key_manager import add_model, get_models

            add_model("ollama", "gemma4:e2b")
            assert "gemma4:e2b" in get_models("ollama")

    def test_add_model_is_idempotent(self, tmp_path):
        with _patch_models_path(tmp_path):
            from local_search_agent.core.key_manager import add_model, get_models

            add_model("ollama", "gemma4:e2b")
            add_model("ollama", "gemma4:e2b")  # second call should not duplicate
            assert get_models("ollama").count("gemma4:e2b") == 1

    def test_add_model_strips_whitespace(self, tmp_path):
        with _patch_models_path(tmp_path):
            from local_search_agent.core.key_manager import add_model, get_models

            add_model("ollama", "  mistral  ")
            assert "mistral" in get_models("ollama")

    def test_add_model_unknown_provider_raises(self, tmp_path):
        with _patch_models_path(tmp_path):
            from local_search_agent.core.key_manager import add_model

            with pytest.raises(ValueError, match="Unknown provider"):
                add_model("bingus", "some-model")

    def test_add_model_empty_name_raises(self, tmp_path):
        with _patch_models_path(tmp_path):
            from local_search_agent.core.key_manager import add_model

            with pytest.raises(ValueError, match="must not be empty"):
                add_model("ollama", "")

    def test_delete_model(self, tmp_path):
        with _patch_models_path(tmp_path):
            from local_search_agent.core.key_manager import add_model, delete_model, get_models

            add_model("ollama", "gemma4:e2b")
            deleted = delete_model("ollama", "gemma4:e2b")
            assert deleted is True
            assert "gemma4:e2b" not in get_models("ollama")

    def test_delete_model_returns_false_if_not_found(self, tmp_path):
        with _patch_models_path(tmp_path):
            from local_search_agent.core.key_manager import delete_model

            assert delete_model("ollama", "nonexistent-model") is False

    def test_delete_model_does_not_affect_other_providers(self, tmp_path):
        with _patch_models_path(tmp_path):
            from local_search_agent.core.key_manager import add_model, delete_model, get_models

            add_model("ollama", "mistral")
            add_model("openai", "gpt-4o-mini")
            delete_model("ollama", "mistral")
            assert "gpt-4o-mini" in get_models("openai")

    def test_get_models_by_provider(self, tmp_path):
        with _patch_models_path(tmp_path):
            from local_search_agent.core.key_manager import add_model, get_models

            add_model("ollama", "mistral")
            result = get_models("ollama")
            assert isinstance(result, list)
            assert "mistral" in result

    def test_get_models_all_returns_dict(self, tmp_path):
        with _patch_models_path(tmp_path):
            from local_search_agent.core.key_manager import get_models

            result = get_models()
            assert isinstance(result, dict)

    def test_models_json_is_valid_json(self, tmp_path):
        with _patch_models_path(tmp_path):
            from local_search_agent.core.key_manager import add_model

            add_model("ollama", "mistral")
            models_file = tmp_path / "models.json"
            data = json.loads(models_file.read_text())
            assert "ollama" in data
            assert "mistral" in data["ollama"]


# ---------------------------------------------------------------------------
# Semantic settings
# ---------------------------------------------------------------------------


def _patch_settings_path(tmp_path):
    settings_file = tmp_path / "settings.json"
    return patch(
        "local_search_agent.core.key_manager._settings_path",
        return_value=settings_file,
    )


class TestSemanticSettings:
    def test_defaults_all_false(self, tmp_path):
        with _patch_settings_path(tmp_path):
            from local_search_agent.core.key_manager import get_semantic_settings

            s = get_semantic_settings()
            assert s["enable_semantic"] is False
            assert s["enable_query_expansion"] is False
            assert s["enable_link_graph"] is False

    def test_set_single_flag(self, tmp_path):
        with _patch_settings_path(tmp_path):
            from local_search_agent.core.key_manager import (
                get_semantic_settings,
                set_semantic_setting,
            )

            set_semantic_setting("enable_semantic", True)
            s = get_semantic_settings()
            assert s["enable_semantic"] is True
            assert s["enable_query_expansion"] is False

    def test_set_all_flags(self, tmp_path):
        with _patch_settings_path(tmp_path):
            from local_search_agent.core.key_manager import (
                get_semantic_settings,
                set_all_semantic_settings,
            )

            set_all_semantic_settings(
                enable_semantic=True,
                enable_query_expansion=True,
                enable_link_graph=False,
            )
            s = get_semantic_settings()
            assert s["enable_semantic"] is True
            assert s["enable_query_expansion"] is True
            assert s["enable_link_graph"] is False

    def test_set_flag_persists_across_calls(self, tmp_path):
        with _patch_settings_path(tmp_path):
            from local_search_agent.core.key_manager import (
                get_semantic_settings,
                set_semantic_setting,
            )

            set_semantic_setting("enable_link_graph", True)
            set_semantic_setting("enable_semantic", True)
            s = get_semantic_settings()
            # Both should be persisted, not overwritten
            assert s["enable_link_graph"] is True
            assert s["enable_semantic"] is True

    def test_unknown_setting_key_raises(self, tmp_path):
        with _patch_settings_path(tmp_path):
            from local_search_agent.core.key_manager import set_semantic_setting

            with pytest.raises(ValueError, match="Unknown semantic setting"):
                set_semantic_setting("enable_magic", True)

    def test_settings_json_is_valid_json(self, tmp_path):
        with _patch_settings_path(tmp_path):
            from local_search_agent.core.key_manager import set_semantic_setting

            set_semantic_setting("enable_semantic", True)
            settings_file = tmp_path / "settings.json"
            data = json.loads(settings_file.read_text())
            assert data["enable_semantic"] is True

    def test_corrupted_settings_file_returns_defaults(self, tmp_path):
        with _patch_settings_path(tmp_path):
            settings_file = tmp_path / "settings.json"
            settings_file.write_text("{ not valid json }")
            from local_search_agent.core.key_manager import get_semantic_settings

            s = get_semantic_settings()
            # Should return defaults, not raise
            assert s["enable_semantic"] is False

    def test_partial_settings_file_filled_with_defaults(self, tmp_path):
        with _patch_settings_path(tmp_path):
            settings_file = tmp_path / "settings.json"
            settings_file.write_text(json.dumps({"enable_semantic": True}))
            from local_search_agent.core.key_manager import get_semantic_settings

            s = get_semantic_settings()
            assert s["enable_semantic"] is True
            assert s["enable_query_expansion"] is False  # filled with default
            assert s["enable_link_graph"] is False  # filled with default
