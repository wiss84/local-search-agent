"""
Unit tests for DocumentNode.

Covers:
- make_doc_id stability (same path → same id)
- from_file() factory
- to_dict / from_dict round-trip
- snippet() extraction logic
"""

from __future__ import annotations

import os

import pytest

from local_search_agent.core.document_node import DocumentNode

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_txt(tmp_path):
    """Create a real temp file so from_file() can stat it."""
    p = tmp_path / "quarterly_report.txt"
    p.write_text(
        "This is the quarterly report for Q3 2024. "
        "AWS spend on Project Alpha was $1.2M. "
        "Employee morale surveys showed improvement. "
        "Turnover rate dropped to 5% this quarter.",
        encoding="utf-8",
    )
    return str(p)


# ---------------------------------------------------------------------------
# make_doc_id
# ---------------------------------------------------------------------------

class TestMakeDocId:
    def test_same_path_produces_same_id(self):
        path = "C:/shares/finance/report.pdf"
        assert DocumentNode.make_doc_id(path) == DocumentNode.make_doc_id(path)

    def test_different_paths_produce_different_ids(self):
        a = DocumentNode.make_doc_id("C:/shares/finance/report.pdf")
        b = DocumentNode.make_doc_id("C:/shares/hr/handbook.pdf")
        assert a != b

    def test_id_is_16_chars(self):
        doc_id = DocumentNode.make_doc_id("any/path/here.docx")
        assert len(doc_id) == 16

    def test_id_is_url_safe(self):
        doc_id = DocumentNode.make_doc_id("C:/some path with spaces/file.pdf")
        assert all(c in "0123456789abcdef" for c in doc_id)


# ---------------------------------------------------------------------------
# from_file()
# ---------------------------------------------------------------------------

class TestFromFile:
    def test_basic_construction(self, sample_txt):
        node = DocumentNode.from_file(
            source_path=sample_txt,
            text="cleaned text",
            workspace="finance",
        )
        assert node.file_type == "txt"
        assert node.workspace == "finance"
        assert node.text == "cleaned text"
        assert node.title == "quarterly_report"
        assert len(node.doc_id) == 16

    def test_title_override(self, sample_txt):
        node = DocumentNode.from_file(sample_txt, "text", "ws", title="Custom Title")
        assert node.title == "Custom Title"

    def test_concepts_and_synonyms(self, sample_txt):
        node = DocumentNode.from_file(
            sample_txt, "text", "ws",
            concepts=["finance", "cloud"],
            synonyms=["AWS", "Amazon Web Services"],
        )
        assert "finance" in node.concepts
        assert "AWS" in node.synonyms

    def test_modified_at_is_set(self, sample_txt):
        node = DocumentNode.from_file(sample_txt, "text", "ws")
        assert node.modified_at  # non-empty

    def test_source_path_is_absolute(self, sample_txt):
        node = DocumentNode.from_file(sample_txt, "text", "ws")
        assert os.path.isabs(node.source_path)


# ---------------------------------------------------------------------------
# Serialisation round-trip
# ---------------------------------------------------------------------------

class TestSerialisation:
    def test_to_dict_from_dict_roundtrip(self, sample_txt):
        node = DocumentNode.from_file(sample_txt, "some text", "ws")
        d = node.to_dict()
        restored = DocumentNode.from_dict(d)
        assert restored.doc_id == node.doc_id
        assert restored.title == node.title
        assert restored.text == node.text
        assert restored.file_type == node.file_type
        assert restored.workspace == node.workspace
        assert restored.concepts == node.concepts
        assert restored.synonyms == node.synonyms

    def test_to_dict_has_all_fields(self, sample_txt):
        node = DocumentNode.from_file(sample_txt, "text", "ws")
        d = node.to_dict()
        expected_keys = {
            "doc_id", "title", "text", "file_type", "source_path",
            "folder_path", "workspace", "modified_at", "indexed_at",
            "concepts", "synonyms",
        }
        assert expected_keys == set(d.keys())


# ---------------------------------------------------------------------------
# snippet()
# ---------------------------------------------------------------------------

class TestSnippet:
    TEXT = (
        "The company was founded in 2001. "
        "AWS spend on Project Alpha reached $1.2M in Q3 2024. "
        "Employee turnover rate improved significantly. "
        "The board approved the new budget for 2025."
    )

    def _node(self):
        return DocumentNode(
            doc_id="test0001",
            title="Test",
            text=self.TEXT,
            file_type="txt",
            source_path="/tmp/test.txt",
            folder_path="/tmp",
            workspace="test",
            modified_at="2024-01-01T00:00:00",
        )

    def test_snippet_contains_keyword(self):
        node = self._node()
        snippet = node.snippet("turnover")
        assert "turnover" in snippet.lower()

    def test_snippet_respects_context_length(self):
        node = self._node()
        snippet = node.snippet("AWS", context_chars=100)
        # Strip ellipsis markers before measuring
        clean = snippet.replace("…", "")
        assert len(clean) <= 110  # small tolerance for word boundaries

    def test_snippet_fallback_on_no_match(self):
        node = self._node()
        snippet = node.snippet("zzznomatch")
        # Should return start of text
        assert snippet.startswith(self.TEXT[:20])

    def test_snippet_adds_ellipsis_when_truncated(self):
        node = self._node()
        # Use a keyword in the middle to force truncation on both sides
        snippet = node.snippet("turnover", context_chars=50)
        assert "…" in snippet
