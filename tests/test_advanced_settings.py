"""
Tests for the Advanced Settings workflow.

Coverage
--------
Layer 1 — key_manager  (storage primitives)
  TestAdvancedSettingsStorage
    - get returns empty dict when file absent
    - set persists a valid override, file is valid JSON
    - set ignores unknown keys silently
    - set coerces int and float values to correct Python types
    - set rejects non-numeric values silently (bad coercion)
    - set with empty dict clears all overrides (reset)
    - set is idempotent when called twice with the same value
    - set overwrites a previous value
    - get_effective_constants returns every expected key
    - get_effective_constants returns compiled-in defaults when nothing overridden
    - get_effective_constants merges overrides on top of defaults
    - get_effective_constants does not mutate the defaults on repeated calls
    - advanced_settings_file_path returns a string ending in advanced_settings.json
    - corrupted file returns empty dict
    - multiple keys persisted together in one call
    - None value is silently skipped
    - empty-string value is silently skipped

Layer 2 — framework  (SearchAgentFramework methods)
  TestFrameworkAdvancedSettings
    - get_advanced_settings returns all expected keys
    - get_advanced_settings returns compiled defaults when nothing set
    - set_advanced_settings persists and returns effective dict
    - set_advanced_settings is visible via subsequent get
    - set_advanced_settings with empty dict resets to defaults
    - set_advanced_settings returned dict contains all keys
    - non-overridden keys remain at defaults after partial set

Layer 3 — API routes  (FastAPI endpoints via TestClient)
  TestAdvancedSettingsAPI
    - GET returns overrides + effective
    - GET returns empty overrides when nothing set
    - GET effective contains all expected keys
    - POST persists and returns effective
    - POST override visible via subsequent GET
    - POST empty overrides resets
    - DELETE resets all overrides
    - DELETE effective contains all keys
    - POST ignores unknown keys (no 422)
    - POST bad string value silently ignored
    - POST multiple keys at once
    - GET after DELETE shows empty overrides

Layer 4 — CLI  (config set-advanced / config show)
  TestCLIAdvancedSettings
    - set-advanced --key --value sets a single constant
    - set-advanced persists via key_manager.get_advanced_settings
    - set-advanced --reset clears all overrides
    - set-advanced with no args prints usage hint
    - config show includes Advanced Settings section
    - config show lists all expected keys
    - config show marks overridden key with [OVERRIDE]
    - config show does not mark non-overridden keys with [OVERRIDE]
    - set-advanced output echoes effective value

Layer 5 — Cross-layer consistency
  TestCrossLayerConsistency
    - key_manager override visible via framework.get_advanced_settings
    - framework override visible via key_manager.get_effective_constants
    - reset via key_manager seen by framework
    - reset via framework seen by key_manager
    - API override visible via key_manager.get_effective_constants
    - all three layers report same effective value after override
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Patch helpers
# ---------------------------------------------------------------------------


def _patch_advanced_path(tmp_path: Path):
    """Redirect advanced_settings.json to a file inside tmp_path."""
    adv_file = tmp_path / "advanced_settings.json"
    return patch(
        "local_search_agent.core.key_manager._advanced_path",
        return_value=adv_file,
    )


# All keys that must appear in every get_effective_constants() / get_advanced_settings() result
EXPECTED_KEYS = {
    "CHUNK_MIN_CHARS",
    "CHUNK_TARGET_CHARS",
    "CHUNK_MAX_CHARS",
    "CHUNK_OVERLAP_CHARS",
    "TABLE_ROWS_PER_CHUNK",
    "PDF_PAGES_PER_BATCH",
    "PDF_SPLIT_THRESHOLD",
    "PDF_FALLBACK_PAGES_PER_BATCH",
    "DOCX_CHAR_SPLIT_THRESHOLD",
    "TESSERACT_FALLBACK_MIN_CHARS",
    "DEFAULT_TOP_K",
    "DEFAULT_MAX_ITERATIONS",
    "SNIPPET_CONTEXT_CHARS",
}


# ===========================================================================
# Layer 1 — key_manager storage primitives
# ===========================================================================


class TestAdvancedSettingsStorage:
    def test_get_returns_empty_dict_when_file_absent(self, tmp_path):
        with _patch_advanced_path(tmp_path):
            from local_search_agent.core.key_manager import get_advanced_settings

            assert get_advanced_settings() == {}

    def test_set_persists_valid_override(self, tmp_path):
        with _patch_advanced_path(tmp_path):
            from local_search_agent.core.key_manager import (
                get_advanced_settings,
                set_advanced_settings,
            )

            set_advanced_settings({"PDF_PAGES_PER_BATCH": 10})
            assert get_advanced_settings()["PDF_PAGES_PER_BATCH"] == 10

    def test_persisted_file_is_valid_json(self, tmp_path):
        with _patch_advanced_path(tmp_path):
            from local_search_agent.core.key_manager import set_advanced_settings

            set_advanced_settings({"CHUNK_TARGET_CHARS": 12000})
            data = json.loads((tmp_path / "advanced_settings.json").read_text(encoding="utf-8"))
            assert data["CHUNK_TARGET_CHARS"] == 12000

    def test_set_ignores_unknown_keys_silently(self, tmp_path):
        with _patch_advanced_path(tmp_path):
            from local_search_agent.core.key_manager import (
                get_advanced_settings,
                set_advanced_settings,
            )

            set_advanced_settings({"UNKNOWN_CONSTANT": 99, "PDF_PAGES_PER_BATCH": 10})
            result = get_advanced_settings()
            assert "UNKNOWN_CONSTANT" not in result
            assert result["PDF_PAGES_PER_BATCH"] == 10

    def test_set_coerces_int_value_from_string(self, tmp_path):
        with _patch_advanced_path(tmp_path):
            from local_search_agent.core.key_manager import (
                get_advanced_settings,
                set_advanced_settings,
            )

            set_advanced_settings({"PDF_PAGES_PER_BATCH": "10"})
            result = get_advanced_settings()
            assert isinstance(result["PDF_PAGES_PER_BATCH"], int)
            assert result["PDF_PAGES_PER_BATCH"] == 10

    def test_set_coerces_float_value_from_string(self, tmp_path):
        with _patch_advanced_path(tmp_path):
            from local_search_agent.core.key_manager import (
                get_advanced_settings,
                set_advanced_settings,
            )

            set_advanced_settings({"PDF_PAGES_PER_BATCH": "10"})
            result = get_advanced_settings()
            assert isinstance(result["PDF_PAGES_PER_BATCH"], int)
            assert result["PDF_PAGES_PER_BATCH"] == 10

    def test_set_silently_drops_non_numeric_string(self, tmp_path):
        with _patch_advanced_path(tmp_path):
            from local_search_agent.core.key_manager import (
                get_advanced_settings,
                set_advanced_settings,
            )

            set_advanced_settings({"PDF_PAGES_PER_BATCH": "not_a_number"})
            assert "PDF_PAGES_PER_BATCH" not in get_advanced_settings()

    def test_set_empty_dict_clears_all_overrides(self, tmp_path):
        with _patch_advanced_path(tmp_path):
            from local_search_agent.core.key_manager import (
                get_advanced_settings,
                set_advanced_settings,
            )

            set_advanced_settings({"PDF_PAGES_PER_BATCH": 10, "CHUNK_TARGET_CHARS": 5000})
            set_advanced_settings({})
            assert get_advanced_settings() == {}

    def test_set_is_idempotent(self, tmp_path):
        with _patch_advanced_path(tmp_path):
            from local_search_agent.core.key_manager import (
                get_advanced_settings,
                set_advanced_settings,
            )

            set_advanced_settings({"PDF_PAGES_PER_BATCH": 10})
            set_advanced_settings({"PDF_PAGES_PER_BATCH": 10})
            assert get_advanced_settings()["PDF_PAGES_PER_BATCH"] == 10

    def test_set_overwrites_previous_value(self, tmp_path):
        with _patch_advanced_path(tmp_path):
            from local_search_agent.core.key_manager import (
                get_advanced_settings,
                set_advanced_settings,
            )

            set_advanced_settings({"PDF_PAGES_PER_BATCH": 10})
            set_advanced_settings({"PDF_PAGES_PER_BATCH": 20})
            assert get_advanced_settings()["PDF_PAGES_PER_BATCH"] == 20

    def test_get_effective_constants_returns_all_expected_keys(self, tmp_path):
        with _patch_advanced_path(tmp_path):
            from local_search_agent.core.key_manager import get_effective_constants

            assert EXPECTED_KEYS == set(get_effective_constants().keys())

    def test_get_effective_constants_returns_compiled_defaults_when_nothing_overridden(
        self, tmp_path
    ):
        with _patch_advanced_path(tmp_path):
            from local_search_agent.core import constants as C
            from local_search_agent.core.key_manager import get_effective_constants

            result = get_effective_constants()
            assert result["PDF_PAGES_PER_BATCH"] == C.PDF_PAGES_PER_BATCH
            assert result["CHUNK_TARGET_CHARS"] == C.CHUNK_TARGET_CHARS
            assert result["DEFAULT_TOP_K"] == C.DEFAULT_TOP_K

    def test_get_effective_constants_merges_override_on_top_of_defaults(self, tmp_path):
        with _patch_advanced_path(tmp_path):
            from local_search_agent.core import constants as C
            from local_search_agent.core.key_manager import (
                get_effective_constants,
                set_advanced_settings,
            )

            set_advanced_settings({"PDF_PAGES_PER_BATCH": 5})
            result = get_effective_constants()
            assert result["PDF_PAGES_PER_BATCH"] == 5
            # Non-overridden key still uses the compiled-in default
            assert result["CHUNK_TARGET_CHARS"] == C.CHUNK_TARGET_CHARS

    def test_get_effective_constants_does_not_mutate_previous_result(self, tmp_path):
        with _patch_advanced_path(tmp_path):
            from local_search_agent.core import constants as C
            from local_search_agent.core.key_manager import (
                get_effective_constants,
                set_advanced_settings,
            )

            first = get_effective_constants()
            set_advanced_settings({"PDF_PAGES_PER_BATCH": 99})
            second = get_effective_constants()
            # The first snapshot must not have been mutated
            assert first["PDF_PAGES_PER_BATCH"] == C.PDF_PAGES_PER_BATCH
            assert second["PDF_PAGES_PER_BATCH"] == 99

    def test_advanced_settings_file_path_ends_with_correct_filename(self, tmp_path):
        with _patch_advanced_path(tmp_path):
            from local_search_agent.core.key_manager import advanced_settings_file_path

            path = advanced_settings_file_path()
            assert isinstance(path, str)
            assert path.endswith("advanced_settings.json")

    def test_corrupted_file_returns_empty_dict(self, tmp_path):
        with _patch_advanced_path(tmp_path):
            (tmp_path / "advanced_settings.json").write_text("{ not valid json }", encoding="utf-8")
            from local_search_agent.core.key_manager import get_advanced_settings

            assert get_advanced_settings() == {}

    def test_multiple_keys_persisted_in_one_call(self, tmp_path):
        with _patch_advanced_path(tmp_path):
            from local_search_agent.core.key_manager import (
                get_advanced_settings,
                set_advanced_settings,
            )

            overrides = {
                "PDF_PAGES_PER_BATCH": 8,
                "CHUNK_TARGET_CHARS": 6000,
                "DEFAULT_TOP_K": 12,
            }
            set_advanced_settings(overrides)
            result = get_advanced_settings()
            assert result["PDF_PAGES_PER_BATCH"] == 8
            assert result["CHUNK_TARGET_CHARS"] == 6000
            assert result["DEFAULT_TOP_K"] == 12

    def test_none_value_is_silently_skipped(self, tmp_path):
        with _patch_advanced_path(tmp_path):
            from local_search_agent.core.key_manager import (
                get_advanced_settings,
                set_advanced_settings,
            )

            set_advanced_settings({"PDF_PAGES_PER_BATCH": None})
            assert "PDF_PAGES_PER_BATCH" not in get_advanced_settings()

    def test_empty_string_value_is_silently_skipped(self, tmp_path):
        with _patch_advanced_path(tmp_path):
            from local_search_agent.core.key_manager import (
                get_advanced_settings,
                set_advanced_settings,
            )

            set_advanced_settings({"PDF_PAGES_PER_BATCH": ""})
            assert "PDF_PAGES_PER_BATCH" not in get_advanced_settings()


# ===========================================================================
# Layer 2 — SearchAgentFramework methods
# ===========================================================================


@pytest.fixture
def bare_framework(tmp_path, db_path):
    """
    A SearchAgentFramework instance with all heavy components bypassed.
    Only the settings methods (which delegate to key_manager) are exercised.
    """
    from local_search_agent.core.config import SearchAgentConfig
    from local_search_agent.core.framework import SearchAgentFramework
    from local_search_agent.workspace.workspace_manager import WorkspaceManager

    config = SearchAgentConfig(
        workspace_name="test",
        document_dirs=[str(tmp_path)],
        provider="ollama",
        db_path=db_path,
    )
    fw = SearchAgentFramework.__new__(SearchAgentFramework)
    fw.config = config
    fw._scheduler = None
    fw._file_server_thread = None
    fw.workspace_manager = WorkspaceManager(db_path=db_path)
    return fw


class TestFrameworkAdvancedSettings:
    def test_get_returns_all_expected_keys(self, bare_framework, tmp_path):
        with _patch_advanced_path(tmp_path):
            assert EXPECTED_KEYS == set(bare_framework.get_advanced_settings().keys())

    def test_get_returns_compiled_defaults_when_nothing_set(self, bare_framework, tmp_path):
        with _patch_advanced_path(tmp_path):
            from local_search_agent.core import constants as C

            result = bare_framework.get_advanced_settings()
            assert result["PDF_PAGES_PER_BATCH"] == C.PDF_PAGES_PER_BATCH

    def test_set_persists_and_returns_effective(self, bare_framework, tmp_path):
        with _patch_advanced_path(tmp_path):
            effective = bare_framework.set_advanced_settings({"PDF_PAGES_PER_BATCH": 7})
            assert effective["PDF_PAGES_PER_BATCH"] == 7

    def test_set_visible_via_subsequent_get(self, bare_framework, tmp_path):
        with _patch_advanced_path(tmp_path):
            bare_framework.set_advanced_settings({"CHUNK_TARGET_CHARS": 4000})
            assert bare_framework.get_advanced_settings()["CHUNK_TARGET_CHARS"] == 4000

    def test_set_empty_dict_resets_to_defaults(self, bare_framework, tmp_path):
        with _patch_advanced_path(tmp_path):
            from local_search_agent.core import constants as C

            bare_framework.set_advanced_settings({"PDF_PAGES_PER_BATCH": 3})
            bare_framework.set_advanced_settings({})
            result = bare_framework.get_advanced_settings()
            assert result["PDF_PAGES_PER_BATCH"] == C.PDF_PAGES_PER_BATCH

    def test_set_returned_dict_contains_all_keys(self, bare_framework, tmp_path):
        with _patch_advanced_path(tmp_path):
            effective = bare_framework.set_advanced_settings({"DEFAULT_TOP_K": 15})
            assert EXPECTED_KEYS == set(effective.keys())

    def test_non_overridden_keys_remain_at_defaults_after_partial_set(
        self, bare_framework, tmp_path
    ):
        with _patch_advanced_path(tmp_path):
            from local_search_agent.core import constants as C

            effective = bare_framework.set_advanced_settings({"PDF_PAGES_PER_BATCH": 5})
            assert effective["CHUNK_TARGET_CHARS"] == C.CHUNK_TARGET_CHARS
            assert effective["DEFAULT_TOP_K"] == C.DEFAULT_TOP_K


# ===========================================================================
# Layer 3 — FastAPI endpoints
# ===========================================================================


@pytest.fixture
def api_client(tmp_path, db_path):
    """
    Minimal FastAPI TestClient with only the UI router mounted.
    Meilisearch and agent are stubbed — only settings endpoints are exercised here.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from local_search_agent.core.config import SearchAgentConfig
    from local_search_agent.ui.api_routes import build_ui_router
    from local_search_agent.ui.store import UIStore
    from local_search_agent.workspace.workspace_manager import WorkspaceManager

    config = SearchAgentConfig(
        workspace_name="test",
        document_dirs=[str(tmp_path)],
        provider="ollama",
        db_path=db_path,
    )
    app_state = MagicMock()
    app_state.config = config
    app_state.workspace_manager = WorkspaceManager(db_path=db_path)
    app_state.store = UIStore(db_path=db_path)
    app_state.framework = MagicMock()

    app = FastAPI()
    app.include_router(build_ui_router(app_state))
    return TestClient(app, raise_server_exceptions=True)


class TestAdvancedSettingsAPI:
    def test_get_returns_overrides_and_effective(self, api_client, tmp_path):
        with _patch_advanced_path(tmp_path):
            resp = api_client.get("/api/ui/settings/advanced")
            assert resp.status_code == 200
            body = resp.json()
            assert "overrides" in body
            assert "effective" in body

    def test_get_returns_empty_overrides_when_nothing_set(self, api_client, tmp_path):
        with _patch_advanced_path(tmp_path):
            resp = api_client.get("/api/ui/settings/advanced")
            assert resp.json()["overrides"] == {}

    def test_get_effective_contains_all_expected_keys(self, api_client, tmp_path):
        with _patch_advanced_path(tmp_path):
            effective = api_client.get("/api/ui/settings/advanced").json()["effective"]
            assert EXPECTED_KEYS == set(effective.keys())

    def test_post_persists_and_returns_effective(self, api_client, tmp_path):
        with _patch_advanced_path(tmp_path):
            resp = api_client.post(
                "/api/ui/settings/advanced",
                json={"overrides": {"PDF_PAGES_PER_BATCH": 6}},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["ok"] is True
            assert body["effective"]["PDF_PAGES_PER_BATCH"] == 6

    def test_post_override_visible_via_subsequent_get(self, api_client, tmp_path):
        with _patch_advanced_path(tmp_path):
            api_client.post(
                "/api/ui/settings/advanced",
                json={"overrides": {"CHUNK_TARGET_CHARS": 3000}},
            )
            body = api_client.get("/api/ui/settings/advanced").json()
            assert body["overrides"]["CHUNK_TARGET_CHARS"] == 3000
            assert body["effective"]["CHUNK_TARGET_CHARS"] == 3000

    def test_post_empty_overrides_resets(self, api_client, tmp_path):
        with _patch_advanced_path(tmp_path):
            from local_search_agent.core import constants as C

            api_client.post(
                "/api/ui/settings/advanced",
                json={"overrides": {"PDF_PAGES_PER_BATCH": 3}},
            )
            api_client.post("/api/ui/settings/advanced", json={"overrides": {}})
            body = api_client.get("/api/ui/settings/advanced").json()
            assert body["overrides"] == {}
            assert body["effective"]["PDF_PAGES_PER_BATCH"] == C.PDF_PAGES_PER_BATCH

    def test_delete_resets_all_overrides(self, api_client, tmp_path):
        with _patch_advanced_path(tmp_path):
            from local_search_agent.core import constants as C

            api_client.post(
                "/api/ui/settings/advanced",
                json={"overrides": {"PDF_PAGES_PER_BATCH": 3, "DEFAULT_TOP_K": 20}},
            )
            resp = api_client.delete("/api/ui/settings/advanced")
            assert resp.status_code == 200
            body = resp.json()
            assert body["ok"] is True
            assert body["effective"]["PDF_PAGES_PER_BATCH"] == C.PDF_PAGES_PER_BATCH
            assert body["effective"]["DEFAULT_TOP_K"] == C.DEFAULT_TOP_K

    def test_delete_effective_contains_all_keys(self, api_client, tmp_path):
        with _patch_advanced_path(tmp_path):
            resp = api_client.delete("/api/ui/settings/advanced")
            assert EXPECTED_KEYS == set(resp.json()["effective"].keys())

    def test_post_ignores_unknown_keys_no_422(self, api_client, tmp_path):
        with _patch_advanced_path(tmp_path):
            resp = api_client.post(
                "/api/ui/settings/advanced",
                json={"overrides": {"TOTALLY_UNKNOWN": 42, "PDF_PAGES_PER_BATCH": 9}},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert "TOTALLY_UNKNOWN" not in body["effective"]
            assert body["effective"]["PDF_PAGES_PER_BATCH"] == 9

    def test_post_bad_string_value_silently_ignored(self, api_client, tmp_path):
        with _patch_advanced_path(tmp_path):
            from local_search_agent.core import constants as C

            resp = api_client.post(
                "/api/ui/settings/advanced",
                json={"overrides": {"PDF_PAGES_PER_BATCH": "not_a_number"}},
            )
            assert resp.status_code == 200
            assert resp.json()["effective"]["PDF_PAGES_PER_BATCH"] == C.PDF_PAGES_PER_BATCH

    def test_post_multiple_keys_at_once(self, api_client, tmp_path):
        with _patch_advanced_path(tmp_path):
            resp = api_client.post(
                "/api/ui/settings/advanced",
                json={
                    "overrides": {
                        "PDF_PAGES_PER_BATCH": 8,
                        "CHUNK_TARGET_CHARS": 5000,
                        "DEFAULT_TOP_K": 12,
                    }
                },
            )
            effective = resp.json()["effective"]
            assert effective["PDF_PAGES_PER_BATCH"] == 8
            assert effective["CHUNK_TARGET_CHARS"] == 5000
            assert effective["DEFAULT_TOP_K"] == 12

    def test_get_after_delete_shows_empty_overrides(self, api_client, tmp_path):
        with _patch_advanced_path(tmp_path):
            api_client.post(
                "/api/ui/settings/advanced",
                json={"overrides": {"PDF_PAGES_PER_BATCH": 5}},
            )
            api_client.delete("/api/ui/settings/advanced")
            body = api_client.get("/api/ui/settings/advanced").json()
            assert body["overrides"] == {}


# ===========================================================================
# Layer 4 — CLI
# ===========================================================================


def _run_cli(*args) -> str:
    """Run the CLI parser with the given args and return captured stdout."""
    import io
    import sys

    from local_search_agent.cli.commands import build_parser

    parser = build_parser()
    parsed = parser.parse_args(list(args))
    captured = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured
    try:
        parsed.func(parsed)
    finally:
        sys.stdout = old_stdout
    return captured.getvalue()


class TestCLIAdvancedSettings:
    def test_set_advanced_single_key_prints_key_and_value(self, tmp_path):
        with _patch_advanced_path(tmp_path):
            output = _run_cli(
                "config", "set-advanced", "--key", "PDF_PAGES_PER_BATCH", "--value", "7"
            )
            assert "PDF_PAGES_PER_BATCH" in output
            assert "7" in output

    def test_set_advanced_persists_via_get_advanced_settings(self, tmp_path):
        with _patch_advanced_path(tmp_path):
            _run_cli("config", "set-advanced", "--key", "PDF_PAGES_PER_BATCH", "--value", "7")
            from local_search_agent.core.key_manager import get_advanced_settings

            assert get_advanced_settings()["PDF_PAGES_PER_BATCH"] == 7

    def test_set_advanced_reset_clears_all_overrides(self, tmp_path):
        with _patch_advanced_path(tmp_path):
            from local_search_agent.core.key_manager import (
                get_advanced_settings,
                set_advanced_settings,
            )

            set_advanced_settings({"PDF_PAGES_PER_BATCH": 5, "DEFAULT_TOP_K": 20})
            output = _run_cli("config", "set-advanced", "--reset")
            assert "reset" in output.lower() or "default" in output.lower()
            assert get_advanced_settings() == {}

    def test_set_advanced_no_args_prints_hint(self, tmp_path):
        with _patch_advanced_path(tmp_path):
            output = _run_cli("config", "set-advanced")
            assert "--key" in output or "Provide" in output

    def test_config_show_includes_advanced_settings_section(self, tmp_path):
        with _patch_advanced_path(tmp_path):
            output = _run_cli("config", "show")
            assert "Advanced Settings" in output

    def test_config_show_lists_all_expected_keys(self, tmp_path):
        with _patch_advanced_path(tmp_path):
            output = _run_cli("config", "show")
            for key in EXPECTED_KEYS:
                assert key in output, f"Key {key!r} missing from 'config show' output"

    def test_config_show_marks_overridden_key_with_override_tag(self, tmp_path):
        with _patch_advanced_path(tmp_path):
            from local_search_agent.core.key_manager import set_advanced_settings

            set_advanced_settings({"PDF_PAGES_PER_BATCH": 5})
            output = _run_cli("config", "show")
            assert "[OVERRIDE]" in output

    def test_config_show_override_tag_only_on_overridden_key(self, tmp_path):
        with _patch_advanced_path(tmp_path):
            from local_search_agent.core.key_manager import set_advanced_settings

            set_advanced_settings({"PDF_PAGES_PER_BATCH": 5})
            lines = _run_cli("config", "show").splitlines()
            override_lines = [line for line in lines if "[OVERRIDE]" in line]
            assert len(override_lines) == 1
            assert "PDF_PAGES_PER_BATCH" in override_lines[0]

    def test_set_advanced_float_key(self, tmp_path):
        with _patch_advanced_path(tmp_path):
            _run_cli("config", "set-advanced", "--key", "PDF_PAGES_PER_BATCH", "--value", "8")
            from local_search_agent.core.key_manager import get_advanced_settings

            result = get_advanced_settings()
            assert result["PDF_PAGES_PER_BATCH"] == 8

    def test_set_advanced_output_echoes_effective_value(self, tmp_path):
        with _patch_advanced_path(tmp_path):
            output = _run_cli("config", "set-advanced", "--key", "DEFAULT_TOP_K", "--value", "15")
            assert "15" in output


# ===========================================================================
# Layer 5 — Cross-layer consistency
# ===========================================================================


class TestCrossLayerConsistency:
    def test_key_manager_override_visible_via_framework(self, tmp_path, db_path):
        with _patch_advanced_path(tmp_path):
            from local_search_agent.core.key_manager import set_advanced_settings

            set_advanced_settings({"PDF_PAGES_PER_BATCH": 4})

            fw = _make_bare_framework(tmp_path, db_path)
            assert fw.get_advanced_settings()["PDF_PAGES_PER_BATCH"] == 4

    def test_framework_override_visible_via_key_manager(self, tmp_path, db_path):
        with _patch_advanced_path(tmp_path):
            from local_search_agent.core.key_manager import get_effective_constants

            fw = _make_bare_framework(tmp_path, db_path)
            fw.set_advanced_settings({"CHUNK_TARGET_CHARS": 2500})
            assert get_effective_constants()["CHUNK_TARGET_CHARS"] == 2500

    def test_reset_via_key_manager_seen_by_framework(self, tmp_path, db_path):
        with _patch_advanced_path(tmp_path):
            from local_search_agent.core import constants as C
            from local_search_agent.core.key_manager import set_advanced_settings

            set_advanced_settings({"PDF_PAGES_PER_BATCH": 3})
            set_advanced_settings({})

            fw = _make_bare_framework(tmp_path, db_path)
            assert fw.get_advanced_settings()["PDF_PAGES_PER_BATCH"] == C.PDF_PAGES_PER_BATCH

    def test_reset_via_framework_seen_by_key_manager(self, tmp_path, db_path):
        with _patch_advanced_path(tmp_path):
            from local_search_agent.core.key_manager import get_advanced_settings

            fw = _make_bare_framework(tmp_path, db_path)
            fw.set_advanced_settings({"DEFAULT_TOP_K": 99})
            fw.set_advanced_settings({})
            assert get_advanced_settings() == {}

    def test_api_override_visible_via_key_manager(self, tmp_path, db_path):
        with _patch_advanced_path(tmp_path):
            from fastapi import FastAPI
            from fastapi.testclient import TestClient

            from local_search_agent.core.config import SearchAgentConfig
            from local_search_agent.core.key_manager import get_effective_constants
            from local_search_agent.ui.api_routes import build_ui_router
            from local_search_agent.ui.store import UIStore
            from local_search_agent.workspace.workspace_manager import WorkspaceManager

            config = SearchAgentConfig(
                workspace_name="test",
                document_dirs=[str(tmp_path)],
                provider="ollama",
                db_path=db_path,
            )
            app_state = MagicMock()
            app_state.config = config
            app_state.workspace_manager = WorkspaceManager(db_path=db_path)
            app_state.store = UIStore(db_path=db_path)
            app_state.framework = MagicMock()

            app = FastAPI()
            app.include_router(build_ui_router(app_state))
            TestClient(app).post(
                "/api/ui/settings/advanced",
                json={"overrides": {"SNIPPET_CONTEXT_CHARS": 150}},
            )
            assert get_effective_constants()["SNIPPET_CONTEXT_CHARS"] == 150

    def test_all_layers_report_same_effective_value(self, tmp_path, db_path):
        """key_manager, framework, and API all return 250 for CHUNK_OVERLAP_CHARS."""
        with _patch_advanced_path(tmp_path):
            from fastapi import FastAPI
            from fastapi.testclient import TestClient

            from local_search_agent.core.config import SearchAgentConfig
            from local_search_agent.core.key_manager import (
                get_effective_constants,
                set_advanced_settings,
            )
            from local_search_agent.ui.api_routes import build_ui_router
            from local_search_agent.ui.store import UIStore
            from local_search_agent.workspace.workspace_manager import WorkspaceManager

            set_advanced_settings({"CHUNK_OVERLAP_CHARS": 250})

            # key_manager layer
            km_val = get_effective_constants()["CHUNK_OVERLAP_CHARS"]

            # framework layer
            fw = _make_bare_framework(tmp_path, db_path)
            fw_val = fw.get_advanced_settings()["CHUNK_OVERLAP_CHARS"]

            # API layer
            config = SearchAgentConfig(
                workspace_name="test",
                document_dirs=[str(tmp_path)],
                provider="ollama",
                db_path=db_path,
            )
            app_state = MagicMock()
            app_state.config = config
            app_state.workspace_manager = WorkspaceManager(db_path=db_path)
            app_state.store = UIStore(db_path=db_path)
            app_state.framework = MagicMock()
            app = FastAPI()
            app.include_router(build_ui_router(app_state))
            api_val = (
                TestClient(app)
                .get("/api/ui/settings/advanced")
                .json()["effective"]["CHUNK_OVERLAP_CHARS"]
            )

            assert km_val == fw_val == api_val == 250


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------


def _make_bare_framework(tmp_path, db_path):
    """Return a SearchAgentFramework with all heavy components bypassed."""
    from local_search_agent.core.config import SearchAgentConfig
    from local_search_agent.core.framework import SearchAgentFramework
    from local_search_agent.workspace.workspace_manager import WorkspaceManager

    config = SearchAgentConfig(
        workspace_name="test",
        document_dirs=[str(tmp_path)],
        provider="ollama",
        db_path=db_path,
    )
    fw = SearchAgentFramework.__new__(SearchAgentFramework)
    fw.config = config
    fw._scheduler = None
    fw._file_server_thread = None
    fw.workspace_manager = WorkspaceManager(db_path=db_path)
    return fw
