"""
Unit tests for MeilisearchManager and related utilities in
local_search_agent/core/meilisearch_manager.py.

All network calls, subprocess calls, and file system operations are mocked.
"""

from __future__ import annotations

import signal
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from local_search_agent.core.meilisearch_manager import (
    MEILI_BINARY_VERSION,
    MeilisearchManager,
    _checksum_url,
    _detect_asset,
    _fetch_checksums,
    _sha256,
    run_setup,
)

# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


class TestMeilisearchManagerProperties:
    """Tests for binary_path, cache_dir, and version properties."""

    @patch("local_search_agent.core.meilisearch_manager._binary_path")
    def test_binary_path_returns_versioned_path(self, mock_binary_path):
        expected = Path("/fake/cache/v1.13.3/meilisearch")
        mock_binary_path.return_value = expected
        manager = MeilisearchManager()
        assert manager.binary_path == expected
        mock_binary_path.assert_called_once()

    @patch("local_search_agent.core.meilisearch_manager._logs_dir")
    @patch("local_search_agent.core.meilisearch_manager._cache_dir")
    def test_cache_dir_returns_user_cache_dir(self, mock_cache_dir, mock_logs_dir):
        expected = Path("/fake/cache")
        mock_cache_dir.return_value = expected
        mock_logs_dir.return_value = Path("/fake/cache/logs")
        manager = MeilisearchManager()
        assert manager.cache_dir == expected
        mock_cache_dir.assert_called()

    def test_version_returns_constant(self):
        manager = MeilisearchManager()
        assert manager.version == MEILI_BINARY_VERSION
        assert manager.version == "v1.13.3"


# ---------------------------------------------------------------------------
# ensure_binary tests
# ---------------------------------------------------------------------------


class TestEnsureBinary:
    """Tests for MeilisearchManager.ensure_binary."""

    def test_ensure_binary_returns_existing(self, tmp_path):
        binary = tmp_path / "meilisearch"
        binary.write_bytes(b"fake binary")

        with (
            patch("local_search_agent.core.meilisearch_manager._binary_path", return_value=binary),
            patch("local_search_agent.core.meilisearch_manager._cache_dir", return_value=tmp_path),
            patch("local_search_agent.core.meilisearch_manager._logs_dir", return_value=tmp_path),
            patch("local_search_agent.core.meilisearch_manager.FileLock"),
        ):
            manager = MeilisearchManager()
            result = manager.ensure_binary()
            assert result == binary

    @patch("local_search_agent.core.meilisearch_manager._download_binary")
    @patch("local_search_agent.core.meilisearch_manager._lock_path")
    @patch("local_search_agent.core.meilisearch_manager.FileLock")
    @patch("local_search_agent.core.meilisearch_manager._binary_path")
    def test_ensure_binary_downloads_when_missing(
        self, mock_binary_path, mock_lock_cls, mock_lock_path, mock_download, tmp_path
    ):
        dest = tmp_path / "meilisearch"
        mock_binary_path.return_value = dest

        mock_lock = MagicMock()
        mock_lock_cls.return_value = mock_lock
        mock_lock_path.return_value = tmp_path / "download.lock"

        with (
            patch("local_search_agent.core.meilisearch_manager._cache_dir", return_value=tmp_path),
            patch("local_search_agent.core.meilisearch_manager._logs_dir", return_value=tmp_path),
        ):
            manager = MeilisearchManager()
            result = manager.ensure_binary()
            mock_download.assert_called_once_with(dest)
            assert result == dest

    def test_ensure_binary_raises_when_auto_download_false(self, tmp_path):
        binary = tmp_path / "meilisearch"

        with (
            patch("local_search_agent.core.meilisearch_manager._binary_path", return_value=binary),
            patch("local_search_agent.core.meilisearch_manager._cache_dir", return_value=tmp_path),
            patch("local_search_agent.core.meilisearch_manager._logs_dir", return_value=tmp_path),
        ):
            manager = MeilisearchManager(auto_download=False)
            with pytest.raises(FileNotFoundError, match="Meilisearch binary not found"):
                manager.ensure_binary()


# ---------------------------------------------------------------------------
# is_running tests
# ---------------------------------------------------------------------------


class TestIsRunning:
    """Tests for MeilisearchManager.is_running."""

    @patch("local_search_agent.core.meilisearch_manager._is_http_healthy", return_value=True)
    def test_is_running_true_when_healthy(self, mock_healthy):
        manager = MeilisearchManager()
        assert manager.is_running() is True

    @patch("local_search_agent.core.meilisearch_manager._is_http_healthy", return_value=True)
    def test_is_running_false_when_process_died(self, mock_healthy):
        manager = MeilisearchManager()
        mock_process = MagicMock()
        mock_process.poll.return_value = 1
        mock_process.returncode = 1
        manager._process = mock_process

        assert manager.is_running() is False
        assert manager._process is None

    @patch("local_search_agent.core.meilisearch_manager._is_http_healthy", return_value=False)
    def test_is_running_false_when_unhealthy(self, mock_healthy):
        manager = MeilisearchManager()
        assert manager.is_running() is False


# ---------------------------------------------------------------------------
# start tests
# ---------------------------------------------------------------------------


class TestStart:
    """Tests for MeilisearchManager.start."""

    @patch.object(MeilisearchManager, "is_running", return_value=True)
    @patch("local_search_agent.core.meilisearch_manager.subprocess.Popen")
    def test_start_skips_when_already_running(self, mock_popen, mock_is_running):
        manager = MeilisearchManager()
        manager.start()
        mock_popen.assert_not_called()

    @patch("local_search_agent.core.meilisearch_manager.subprocess.Popen")
    @patch("local_search_agent.core.meilisearch_manager._wait_healthy", return_value=True)
    @patch.object(MeilisearchManager, "ensure_binary", return_value=Path("/fake/meilisearch"))
    @patch.object(MeilisearchManager, "is_running", return_value=False)
    def test_start_launches_process(
        self, mock_is_running, mock_ensure_binary, mock_wait_healthy, mock_popen, tmp_path
    ):
        manager = MeilisearchManager()
        manager._stdout_log = MagicMock()
        manager._stderr_log = MagicMock()

        mock_process = MagicMock()
        mock_process.pid = 1234
        mock_popen.return_value = mock_process

        manager.start()

        args, kwargs = mock_popen.call_args
        cmd = args[0]

        assert cmd[0] == str(Path("/fake/meilisearch"))
        assert "--db-path" in cmd
        assert "--master-key" in cmd
        assert "local_search_master_key" in cmd
        assert "--http-addr" in cmd
        assert "127.0.0.1:7700" in cmd
        assert "--env" in cmd
        assert "development" in cmd
        assert "--no-analytics" in cmd

        assert kwargs["stdout"] == manager._stdout_log.open.return_value
        assert kwargs["stderr"] == manager._stderr_log.open.return_value

        assert manager._process == mock_process

    @patch("local_search_agent.core.meilisearch_manager._wait_healthy", return_value=False)
    @patch("local_search_agent.core.meilisearch_manager.subprocess.Popen")
    @patch.object(MeilisearchManager, "ensure_binary", return_value=Path("/fake/meilisearch"))
    @patch.object(MeilisearchManager, "is_running", return_value=False)
    def test_start_raises_on_health_timeout(
        self, mock_is_running, mock_ensure_binary, mock_popen, mock_wait_healthy
    ):
        manager = MeilisearchManager()
        manager._stdout_log = MagicMock()
        manager._stderr_log = MagicMock()

        mock_process = MagicMock()
        mock_process.pid = 1234
        mock_popen.return_value = mock_process

        with patch.object(manager, "stop") as mock_stop:
            with pytest.raises(RuntimeError, match="failed health check"):
                manager.start()
            mock_stop.assert_called_once()


# ---------------------------------------------------------------------------
# stop tests
# ---------------------------------------------------------------------------


class TestStop:
    """Tests for MeilisearchManager.stop."""

    def test_stop_noop_when_no_process(self):
        manager = MeilisearchManager()
        manager._process = None
        manager.stop()

    def test_stop_terminates_running_process(self):
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_process.pid = 1234
        manager = MeilisearchManager()
        manager._process = mock_process
        manager.stop()
        if sys.platform == "win32":
            mock_process.terminate.assert_called_once()
        else:
            mock_process.send_signal.assert_called_once_with(signal.SIGTERM)
        mock_process.wait.assert_called_once_with(timeout=10)
        assert manager._process is None

    def test_stop_force_kills_on_timeout(self):
        mock_process = MagicMock()
        mock_process.poll.return_value = None

        call_count = [0]

        def wait_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1 and kwargs.get("timeout") == 10:
                raise subprocess.TimeoutExpired("meilisearch", 10)
            return None

        mock_process.wait.side_effect = wait_side_effect
        manager = MeilisearchManager()
        manager._process = mock_process
        manager.stop()
        mock_process.kill.assert_called_once()
        mock_process.wait.assert_called()
        assert manager._process is None


# ---------------------------------------------------------------------------
# cleanup_old_versions tests
# ---------------------------------------------------------------------------


class TestCleanupOldVersions:
    """Tests for MeilisearchManager.cleanup_old_versions."""

    def test_cleanup_old_versions_keeps_current(self, tmp_path):
        cache = tmp_path / "cache"
        current = cache / "v1.13.3"
        old = cache / "v1.13.2"
        logs = cache / "logs"

        for d in [current, old, logs]:
            d.mkdir(parents=True)

        with (
            patch("local_search_agent.core.meilisearch_manager._cache_dir", return_value=cache),
            patch("local_search_agent.core.meilisearch_manager._logs_dir", return_value=logs),
            patch("local_search_agent.core.meilisearch_manager.shutil.rmtree") as mock_rmtree,
        ):
            manager = MeilisearchManager()
            manager.cleanup_old_versions()
            mock_rmtree.assert_called_once_with(old, ignore_errors=True)

    def test_cleanup_skips_logs_dir(self, tmp_path):
        cache = tmp_path / "cache"
        current = cache / "v1.13.3"
        old = cache / "v1.13.2"
        logs = cache / "logs"

        for d in [current, old, logs]:
            d.mkdir(parents=True)

        with (
            patch("local_search_agent.core.meilisearch_manager._cache_dir", return_value=cache),
            patch("local_search_agent.core.meilisearch_manager._logs_dir", return_value=logs),
            patch("local_search_agent.core.meilisearch_manager.shutil.rmtree") as mock_rmtree,
        ):
            manager = MeilisearchManager()
            manager.cleanup_old_versions()
            mock_rmtree.assert_called_once_with(old, ignore_errors=True)


# ---------------------------------------------------------------------------
# Platform / URL / checksum utility tests
# ---------------------------------------------------------------------------


class TestPlatformUtilities:
    """Tests for platform detection, URL, checksum, and SHA256 utilities."""

    @patch("local_search_agent.core.meilisearch_manager.platform.system", return_value="sunos")
    @patch("local_search_agent.core.meilisearch_manager.platform.machine", return_value="sparc")
    def test_detect_asset_raises_on_unsupported_platform(self, mock_machine, mock_system):
        with pytest.raises(RuntimeError, match="Unsupported platform"):
            _detect_asset()

    def test_checksum_url_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="Meilisearch does not provide"):
            _checksum_url()

    def test_sha256_returns_hex_digest(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello world")
        result = _sha256(f)
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    @patch(
        "local_search_agent.core.meilisearch_manager._checksum_url",
        return_value="https://example.com/checksums.txt",
    )
    def test_fetch_checksums_parses_valid_checksums(self, mock_url):
        fake_content = "abc123  meilisearch-linux-amd64\ndef456  meilisearch-windows-amd64.exe\n\n"
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.return_value = fake_content.encode()

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = _fetch_checksums()

        assert result == {
            "meilisearch-linux-amd64": "abc123",
            "meilisearch-windows-amd64.exe": "def456",
        }


# ---------------------------------------------------------------------------
# run_setup tests
# ---------------------------------------------------------------------------


class TestRunSetup:
    """Tests for the run_setup CLI helper."""

    @patch(
        "local_search_agent.core.meilisearch_manager._cache_dir", return_value=Path("/fake/cache")
    )
    @patch("local_search_agent.core.meilisearch_manager.platform.machine", return_value="x86_64")
    @patch("local_search_agent.core.meilisearch_manager.platform.system", return_value="Linux")
    @patch("local_search_agent.core.meilisearch_manager.MeilisearchManager")
    def test_run_setup_prints_info_when_force(
        self, mock_manager_cls, mock_system, mock_machine, mock_cache
    ):
        mock_manager = MagicMock()
        mock_manager.binary_path = Path("/fake/cache/v1.13.3/meilisearch")
        mock_manager.ensure_binary.return_value = Path("/fake/cache/v1.13.3/meilisearch")
        mock_manager_cls.return_value = mock_manager

        with patch("builtins.print") as mock_print:
            run_setup(force=True)

        printed = " ".join(str(call.args[0]) for call in mock_print.call_args_list if call.args)

        assert "Linux" in printed
        assert "x86_64" in printed
        assert "v1.13.3" in printed
        mock_manager.ensure_binary.assert_called_once()

    @patch("local_search_agent.core.meilisearch_manager.MeilisearchManager")
    def test_run_setup_skips_when_exists_and_not_force(self, mock_manager_cls):
        mock_manager = MagicMock()
        mock_manager.binary_path.exists.return_value = True
        mock_manager_cls.return_value = mock_manager

        with patch("builtins.print") as mock_print:
            run_setup(force=False)

        mock_manager.ensure_binary.assert_not_called()
        assert any(
            "already installed" in str(call.args[0])
            for call in mock_print.call_args_list
            if call.args
        )
