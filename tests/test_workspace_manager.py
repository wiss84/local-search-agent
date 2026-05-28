"""
Unit tests for WorkspaceManager (SQLite + in-memory cache).

Covers:
- create / list / get / delete workspace
- register_document / get_document (cache hit + cold path)
- list_documents
- document_needs_reindex delta logic
"""

from __future__ import annotations

import pytest

from local_search_agent.core.document_node import DocumentNode
from local_search_agent.workspace.workspace_manager import WorkspaceManager

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def wm(tmp_path):
    """Fresh WorkspaceManager backed by a temp SQLite file."""
    return WorkspaceManager(db_path=str(tmp_path / "test.db"))


@pytest.fixture
def sample_node(tmp_path):
    """A DocumentNode backed by a real temp file."""
    f = tmp_path / "sample.txt"
    f.write_text("Hello world. This is a test document.", encoding="utf-8")
    return DocumentNode.from_file(str(f), text="Hello world.", workspace="ws1")


# ---------------------------------------------------------------------------
# Workspace CRUD
# ---------------------------------------------------------------------------

class TestWorkspaceCRUD:
    def test_create_and_list(self, wm):
        wm.create_workspace("alpha", "/data/alpha")
        ws = wm.list_workspaces()
        assert any(w["name"] == "alpha" for w in ws)

    def test_get_existing(self, wm):
        wm.create_workspace("beta", "/data/beta")
        ws = wm.get_workspace("beta")
        assert ws is not None
        assert ws["document_dir"] == "/data/beta"

    def test_get_nonexistent_returns_none(self, wm):
        assert wm.get_workspace("nope") is None

    def test_create_is_idempotent_updates_dir(self, wm):
        wm.create_workspace("gamma", "/data/old")
        wm.create_workspace("gamma", "/data/new")
        ws = wm.get_workspace("gamma")
        assert ws["document_dir"] == "/data/new"

    def test_delete_workspace(self, wm):
        wm.create_workspace("delta", "/data/delta")
        wm.delete_workspace("delta")
        assert wm.get_workspace("delta") is None

    def test_delete_evicts_documents_from_cache(self, wm, sample_node):
        wm.create_workspace("ws1", "/data/ws1")
        wm.register_document(sample_node)
        wm.delete_workspace("ws1")
        assert wm.get_document(sample_node.doc_id) is None


# ---------------------------------------------------------------------------
# Document CRUD
# ---------------------------------------------------------------------------

class TestDocumentCRUD:
    def test_register_and_get_from_cache(self, wm, sample_node):
        wm.create_workspace("ws1", "/data/ws1")
        wm.register_document(sample_node)
        retrieved = wm.get_document(sample_node.doc_id)
        assert retrieved is not None
        assert retrieved.doc_id == sample_node.doc_id
        assert retrieved.text == sample_node.text

    def test_get_unknown_doc_returns_none(self, wm):
        assert wm.get_document("000000000000dead") is None

    def test_register_is_upsert(self, wm, sample_node):
        wm.create_workspace("ws1", "/data/ws1")
        wm.register_document(sample_node)
        # Update title and re-register
        sample_node.title = "Updated Title"
        wm.register_document(sample_node)
        retrieved = wm.get_document(sample_node.doc_id)
        assert retrieved.title == "Updated Title"

    def test_list_documents_returns_all(self, wm, tmp_path):
        wm.create_workspace("ws1", "/data/ws1")
        for i in range(3):
            f = tmp_path / f"doc{i}.txt"
            f.write_text(f"Content {i}", encoding="utf-8")
            node = DocumentNode.from_file(str(f), text=f"Content {i}", workspace="ws1")
            wm.register_document(node)
        docs = wm.list_documents("ws1")
        assert len(docs) == 3

    def test_list_documents_unknown_workspace_returns_none(self, wm):
        assert wm.list_documents("no_such_ws") is None


# ---------------------------------------------------------------------------
# Delta logic
# ---------------------------------------------------------------------------

class TestDeltaLogic:
    def test_unindexed_file_needs_reindex(self, wm, sample_node):
        assert wm.document_needs_reindex(sample_node.source_path, sample_node.modified_at) is True

    def test_unchanged_file_does_not_need_reindex(self, wm, sample_node):
        wm.create_workspace("ws1", "/data/ws1")
        wm.register_document(sample_node)
        assert wm.document_needs_reindex(sample_node.source_path, sample_node.modified_at) is False

    def test_changed_modified_at_triggers_reindex(self, wm, sample_node):
        wm.create_workspace("ws1", "/data/ws1")
        wm.register_document(sample_node)
        assert wm.document_needs_reindex(sample_node.source_path, "2099-01-01T00:00:00") is True
