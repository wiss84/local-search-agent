"""
Unit tests for server/static_mounts.py.

Covers:
- ValueError when directory does not exist
- Successful mount of an existing directory
- Route name defaults to directory basename
- Custom name override
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from local_search_agent.server.static_mounts import mount_directory


class TestMountDirectory:
    def test_raises_value_error_for_missing_directory(self, tmp_path):
        app = FastAPI()
        missing = str(tmp_path / "does_not_exist")
        with pytest.raises(ValueError, match="not a directory"):
            mount_directory(app, missing, "/static")

    def test_raises_value_error_for_file_path(self, tmp_path):
        app = FastAPI()
        f = tmp_path / "file.txt"
        f.write_text("hello")
        with pytest.raises(ValueError, match="not a directory"):
            mount_directory(app, str(f), "/static")

    def test_mounts_existing_directory(self, tmp_path):
        app = FastAPI()
        mount_directory(app, str(tmp_path), "/static")
        routes = [r.path for r in app.routes]
        assert "/static" in routes

    def test_default_route_name_is_basename(self, tmp_path):
        app = FastAPI()
        mount_directory(app, str(tmp_path), "/static")
        route = next(r for r in app.routes if getattr(r, "path", None) == "/static")
        assert route.name == tmp_path.name

    def test_custom_name_override(self, tmp_path):
        app = FastAPI()
        mount_directory(app, str(tmp_path), "/static", name="custom")
        route = next(r for r in app.routes if getattr(r, "path", None) == "/static")
        assert route.name == "custom"

    def test_mount_uses_static_files(self, tmp_path):
        app = FastAPI()
        mount_directory(app, str(tmp_path), "/static")
        route = next(r for r in app.routes if getattr(r, "path", None) == "/static")
        assert isinstance(route.app, StaticFiles)
        assert route.app.directory == str(tmp_path)
