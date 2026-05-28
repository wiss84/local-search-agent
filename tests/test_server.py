"""
Unit tests for the FastAPI file server (server/fastapi_app.py).

Uses FastAPI's TestClient — no real uvicorn process needed.
The metadata_db parameter is intentionally omitted in the main client fixture,
which is the realistic case for a minimal deployment. History endpoints
correctly return 503 when MetadataDB is absent.

Covers:
- GET /health
- GET /workspaces
- GET /workspaces/{name}/docs
- GET /workspaces/{name}/history  (503 when MetadataDB absent)
- GET /text/{doc_id}              (hit, miss, correct content-type)
- GET /docs/{doc_id}              (hit, miss, file-deleted → 410)
- GET /help/*                     (docs endpoint present or 404, never 500)

Design notes:
- doc_ids are derived via DocumentNode.make_doc_id(abs_path), so we look
  them up from wm.list_documents() rather than hard-coding them.
- All file paths are resolved to absolute paths via DocumentNode.from_file(),
  which calls os.path.abspath() internally.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from local_search_agent.core.config import SearchAgentConfig
from local_search_agent.core.document_node import DocumentNode
from local_search_agent.server.fastapi_app import build_app
from local_search_agent.workspace.workspace_manager import WorkspaceManager

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_workspace(tmp_path):
    """A temp dir with two real files (txt + md)."""
    txt = tmp_path / "report.txt"
    txt.write_text("Q3 2024 AWS spend on Project Alpha was $1.2M.", encoding="utf-8")

    md = tmp_path / "handbook.md"
    md.write_text("# Employee Handbook\n\nWelcome to the company.", encoding="utf-8")

    return tmp_path, txt, md


@pytest.fixture
def client(tmp_path, tmp_workspace):
    """
    TestClient backed by a real WorkspaceManager (real SQLite in tmp_path)
    with two documents pre-registered. metadata_db is NOT passed — this is
    the standard minimal deployment.

    Returns (TestClient, WorkspaceManager, txt_path, md_path).
    """
    ws_dir, txt_file, md_file = tmp_workspace

    config = SearchAgentConfig(
        document_dirs=[str(ws_dir)],
        workspace_name="test_ws",
        db_path=str(tmp_path / "test.db"),
        provider="ollama",
    )
    wm = WorkspaceManager(db_path=config.db_path)
    wm.create_workspace(name="test_ws", document_dir=str(ws_dir))

    for f in [txt_file, md_file]:
        node = DocumentNode.from_file(
            source_path=str(f),
            text=f.read_text(encoding="utf-8"),
            workspace="test_ws",
        )
        wm.register_document(node)

    app = build_app(config=config, workspace_manager=wm)
    return TestClient(app), wm, txt_file, md_file


def _doc_id(wm: WorkspaceManager, filename: str) -> str:
    """Look up a doc_id by matching filename stem against registered documents."""
    docs = wm.list_documents("test_ws")
    assert docs is not None
    for d in docs:
        if filename in d["source_path"]:
            return d["doc_id"]
    raise ValueError(f"No registered document matching {filename!r}")


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_200(self, client):
        tc, *_ = client
        assert tc.get("/health").status_code == 200

    def test_status_is_ok(self, client):
        tc, *_ = client
        assert tc.get("/health").json()["status"] == "ok"

    def test_version_key_present(self, client):
        tc, *_ = client
        assert "version" in tc.get("/health").json()

    def test_version_is_semver_string(self, client):
        tc, *_ = client
        version = tc.get("/health").json()["version"]
        parts = version.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)


# ---------------------------------------------------------------------------
# GET /workspaces
# ---------------------------------------------------------------------------

class TestWorkspaces:
    def test_returns_200(self, client):
        tc, *_ = client
        assert tc.get("/workspaces").status_code == 200

    def test_lists_registered_workspace(self, client):
        tc, *_ = client
        data = tc.get("/workspaces").json()
        names = [ws["name"] for ws in data["workspaces"]]
        assert "test_ws" in names

    def test_response_has_workspaces_key(self, client):
        tc, *_ = client
        data = tc.get("/workspaces").json()
        assert "workspaces" in data
        assert isinstance(data["workspaces"], list)


# ---------------------------------------------------------------------------
# GET /workspaces/{name}/docs
# ---------------------------------------------------------------------------

class TestWorkspaceDocs:
    def test_returns_200_for_known_workspace(self, client):
        tc, *_ = client
        assert tc.get("/workspaces/test_ws/docs").status_code == 200

    def test_lists_two_documents(self, client):
        tc, *_ = client
        docs = tc.get("/workspaces/test_ws/docs").json()["documents"]
        assert len(docs) == 2

    def test_document_fields_present(self, client):
        tc, *_ = client
        docs = tc.get("/workspaces/test_ws/docs").json()["documents"]
        required_fields = {"doc_id", "title", "file_type", "source_path"}
        for doc in docs:
            assert required_fields <= doc.keys(), \
                f"Missing fields in document: {required_fields - doc.keys()}"

    def test_unknown_workspace_returns_404(self, client):
        tc, *_ = client
        assert tc.get("/workspaces/does_not_exist/docs").status_code == 404

    def test_both_file_types_present(self, client):
        tc, *_ = client
        docs = tc.get("/workspaces/test_ws/docs").json()["documents"]
        file_types = {d["file_type"] for d in docs}
        assert "txt" in file_types
        assert "md" in file_types


# ---------------------------------------------------------------------------
# GET /text/{doc_id}
# ---------------------------------------------------------------------------

class TestTextEndpoint:
    def test_returns_200_for_known_doc(self, client):
        tc, wm, *_ = client
        doc_id = _doc_id(wm, "report.txt")
        assert tc.get(f"/text/{doc_id}").status_code == 200

    def test_returns_correct_text_content(self, client):
        tc, wm, *_ = client
        doc_id = _doc_id(wm, "report.txt")
        resp = tc.get(f"/text/{doc_id}")
        assert "AWS" in resp.text
        assert "$1.2M" in resp.text

    def test_content_type_is_plain_text(self, client):
        tc, wm, *_ = client
        doc_id = _doc_id(wm, "report.txt")
        resp = tc.get(f"/text/{doc_id}")
        assert "text/plain" in resp.headers["content-type"]

    def test_unknown_doc_id_returns_404(self, client):
        tc, *_ = client
        assert tc.get("/text/000000000000dead").status_code == 404

    def test_md_file_text_served_correctly(self, client):
        tc, wm, _, md_file = client
        doc_id = _doc_id(wm, "handbook.md")
        resp = tc.get(f"/text/{doc_id}")
        assert resp.status_code == 200
        assert "Employee Handbook" in resp.text

    def test_txt_and_md_return_different_content(self, client):
        tc, wm, _, _ = client
        txt_id = _doc_id(wm, "report.txt")
        md_id = _doc_id(wm, "handbook.md")
        assert tc.get(f"/text/{txt_id}").text != tc.get(f"/text/{md_id}").text


# ---------------------------------------------------------------------------
# GET /docs/{doc_id}
# ---------------------------------------------------------------------------

class TestDocsEndpoint:
    def test_returns_200_for_known_doc(self, client):
        tc, wm, *_ = client
        doc_id = _doc_id(wm, "report.txt")
        assert tc.get(f"/docs/{doc_id}").status_code == 200

    def test_returns_original_file_bytes(self, client):
        tc, wm, txt_file, _ = client
        doc_id = _doc_id(wm, "report.txt")
        resp = tc.get(f"/docs/{doc_id}")
        assert b"AWS" in resp.content

    def test_content_disposition_contains_filename(self, client):
        tc, wm, *_ = client
        doc_id = _doc_id(wm, "report.txt")
        resp = tc.get(f"/docs/{doc_id}")
        assert "report.txt" in resp.headers.get("content-disposition", "")

    def test_unknown_doc_id_returns_404(self, client):
        tc, *_ = client
        assert tc.get("/docs/000000000000dead").status_code == 404

    def test_deleted_file_returns_410(self, client):
        """
        If the source file is deleted after registration, the server must
        return 410 Gone (not 404 or 500) because the document record still
        exists but the underlying file is missing.
        """
        tc, wm, txt_file, _ = client
        doc_id = _doc_id(wm, "report.txt")
        os.remove(str(txt_file))
        assert tc.get(f"/docs/{doc_id}").status_code == 410

    def test_md_file_served(self, client):
        tc, wm, _, md_file = client
        doc_id = _doc_id(wm, "handbook.md")
        resp = tc.get(f"/docs/{doc_id}")
        assert resp.status_code == 200
        assert b"Employee Handbook" in resp.content


# ---------------------------------------------------------------------------
# GET /workspaces/{name}/history  (MetadataDB absent → 503)
# ---------------------------------------------------------------------------

class TestWorkspaceHistory:
    def test_known_workspace_returns_503_without_metadata_db(self, client):
        """
        metadata_db is not passed to build_app in this fixture.
        The endpoint must return 503 (not 500) with a clear detail message.
        """
        tc, *_ = client
        resp = tc.get("/workspaces/test_ws/history")
        assert resp.status_code == 503

    def test_503_response_has_detail_message(self, client):
        tc, *_ = client
        resp = tc.get("/workspaces/test_ws/history")
        assert "detail" in resp.json()
        assert resp.json()["detail"]  # non-empty

    def test_unknown_workspace_also_returns_503_without_metadata_db(self, client):
        """
        Without MetadataDB the 503 gate fires before the workspace-existence
        check, so unknown workspaces also get 503 (not 404).
        """
        tc, *_ = client
        assert tc.get("/workspaces/does_not_exist/history").status_code == 503


# ---------------------------------------------------------------------------
# GET /help/* — static documentation endpoint
# ---------------------------------------------------------------------------

class TestHelpEndpoint:
    def test_nonexistent_file_returns_404_not_500(self, client):
        tc, *_ = client
        resp = tc.get("/help/this-file-does-not-exist.md")
        assert resp.status_code in (404, 405)  # 405 if /help not mounted at all

    def test_getting_started_returns_200_or_404(self, client):
        """
        /help is mounted only when the docs/ directory exists.
        In CI (docs/ present) we expect 200; if absent we expect 404, never 500.
        """
        tc, *_ = client
        resp = tc.get("/help/getting-started.md")
        assert resp.status_code in (200, 404, 405)
