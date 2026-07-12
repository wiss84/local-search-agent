"""
Regression tests for the export-chat dual delivery mode (see
upcoming_features/09-export-chat-browser-download-redesign.md):

- `folder` given in the request body -> unchanged legacy behavior (write
  to that path on disk, return {"ok": true, "path": ...}).
- `folder` absent -> return the file bytes directly with
  Content-Disposition: attachment, no disk write at all.

Covers all three export routes: /export-chat (Markdown), /export-chat-docx
(Word), /export-table-xlsx (Excel).
"""

from __future__ import annotations

import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from local_search_agent.ui.api_routes import build_ui_router


class _FakeConfig:
    identity_provider = None


class _FakeAppState:
    def __init__(self):
        self.config = _FakeConfig()
        self.workspace_manager = None  # only touched for citation-link resolution


@pytest.fixture
def client(tmp_path, monkeypatch):
    # os.startfile only exists on Windows -- patch it in unconditionally so
    # the "folder given" branch's open-with-OS-default-app step is a no-op
    # and safe to run on any platform / CI.
    monkeypatch.setattr(os, "startfile", lambda path: None, raising=False)
    app = FastAPI()
    app.include_router(build_ui_router(_FakeAppState()))
    return TestClient(app)


class TestExportChatMarkdownDualMode:
    def test_folder_given_writes_to_disk_and_returns_json(self, client, tmp_path):
        resp = client.post(
            "/api/ui/export-chat",
            json={"folder": str(tmp_path), "filename": "chat.md", "content": "hello world"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert os.path.isfile(body["path"])
        with open(body["path"], encoding="utf-8") as f:
            assert f.read() == "hello world"

    def test_no_folder_returns_bytes_with_content_disposition(self, client, tmp_path):
        resp = client.post(
            "/api/ui/export-chat",
            json={"filename": "chat.md", "content": "hello world"},
        )
        assert resp.status_code == 200
        assert "attachment" in resp.headers["content-disposition"]
        assert 'filename="chat.md"' in resp.headers["content-disposition"]
        assert resp.content == b"hello world"
        # No stray files should have been written anywhere.
        assert list(tmp_path.iterdir()) == []

    def test_invalid_folder_rejected(self, client):
        resp = client.post(
            "/api/ui/export-chat",
            json={"folder": "/definitely/not/a/real/path", "filename": "x.md", "content": "x"},
        )
        assert resp.status_code == 400


class TestExportChatDocxDualMode:
    def test_folder_given_writes_to_disk_and_returns_json(self, client, tmp_path):
        resp = client.post(
            "/api/ui/export-chat-docx",
            json={
                "folder": str(tmp_path),
                "filename": "chat.docx",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert os.path.isfile(body["path"])
        assert os.path.getsize(body["path"]) > 0

    def test_no_folder_returns_docx_bytes(self, client, tmp_path):
        resp = client.post(
            "/api/ui/export-chat-docx",
            json={
                "filename": "chat.docx",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200
        assert "attachment" in resp.headers["content-disposition"]
        assert resp.headers["content-type"].startswith(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        assert len(resp.content) > 0
        assert list(tmp_path.iterdir()) == []


class TestExportTableXlsxDualMode:
    def test_folder_given_writes_to_disk_and_returns_json(self, client, tmp_path):
        resp = client.post(
            "/api/ui/export-table-xlsx",
            json={
                "folder": str(tmp_path),
                "filename": "table.xlsx",
                "headers": ["Name", "Value"],
                "rows": [["a", "1"], ["b", "2"]],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert os.path.isfile(body["path"])

    def test_no_folder_returns_xlsx_bytes(self, client, tmp_path):
        resp = client.post(
            "/api/ui/export-table-xlsx",
            json={
                "filename": "table.xlsx",
                "headers": ["Name", "Value"],
                "rows": [["a", "1"]],
            },
        )
        assert resp.status_code == 200
        assert "attachment" in resp.headers["content-disposition"]
        assert resp.headers["content-type"].startswith(
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        assert len(resp.content) > 0
        assert list(tmp_path.iterdir()) == []

    def test_empty_headers_rejected_regardless_of_mode(self, client):
        resp = client.post(
            "/api/ui/export-table-xlsx",
            json={"filename": "table.xlsx", "headers": [], "rows": []},
        )
        assert resp.status_code == 400
