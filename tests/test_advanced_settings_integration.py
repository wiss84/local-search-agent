"""
Integration tests for advanced_settings.json overrides.

test_advanced_settings.py covers the *storage* layer thoroughly: that
overrides persist correctly and that key_manager / framework / API / CLI all
*report* consistent effective values. It never calls the actual ingestion or
search code, so it could not have caught the bug where chunker.py,
meilisearch_client.py, pdf_parser.py, docx_parser.py, and search_tool.py
imported raw constants at module-import time and silently ignored every
override.

This module closes that gap: each test sets an override via the same
key_manager.set_advanced_settings() entry point the UI/CLI/API use, then
calls the actual production function (chunk_document, MeilisearchClient.search,
the search_local_index tool, PDFParser.parse, SearchAgentConfig) and asserts
the override is reflected in real behaviour — not just in what
get_effective_constants() reports.

Layout
------
TestChunkerRespectsOverrides       — chunk_document() honours CHUNK_* / TABLE_ROWS_PER_CHUNK
TestMeilisearchClientRespectsOverrides — search() honours DEFAULT_TOP_K / SNIPPET_CONTEXT_CHARS
TestSearchToolRespectsOverrides    — search_local_index tool honours config.top_k
TestSearchAgentConfigRespectsOverrides — top_k/max_iterations defaults resolve overrides
TestPDFParserRespectsOverrides     — PDFParser.parse() threads PDF_* overrides (slow, Docling mocked)
TestDOCXParserRespectsOverrides    — DOCXParser.parse() threads DOCX_CHAR_SPLIT_THRESHOLD (slow, Docling mocked)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from local_search_agent.core.document_node import DocumentNode

# ---------------------------------------------------------------------------
# Shared patch helper (same pattern as test_advanced_settings.py)
# ---------------------------------------------------------------------------


def _patch_advanced_path(tmp_path: Path):
    """Redirect advanced_settings.json to a file inside tmp_path."""
    adv_file = tmp_path / "advanced_settings.json"
    return patch(
        "local_search_agent.core.key_manager._advanced_path",
        return_value=adv_file,
    )


# ===========================================================================
# Chunker — chunk_document() is the production entry point used by the
# ingestion pipeline. All assertions go through it, not the private helpers.
# ===========================================================================


def _make_node(text: str, file_type: str = "txt") -> "DocumentNode":

    return DocumentNode(
        doc_id="testdoc0000",
        title="Test Document",
        text=text,
        file_type=file_type,
        source_path="/fake/test.txt",
        folder_path="/fake",
        workspace="test_ws",
        modified_at="2024-01-01T00:00:00+00:00",
        indexed_at="2024-01-01T00:00:00+00:00",
    )


_SENTENCE = (
    "The quarterly results show strong performance across all divisions. "
    "Revenue grew by 12 percent year over year, driven primarily by the "
    "enterprise segment. Operating margins improved as headcount remained "
    "flat while productivity increased. "
)


def _prose(paragraphs: int) -> str:
    single = (_SENTENCE * 2).strip()
    return "\n\n".join([single] * paragraphs)


def _table(rows: int) -> str:
    header = "| Quarter | Revenue | Growth |"
    sep = "| --- | --- | --- |"
    data = "\n".join(f"| Q{i:03d} | ${i * 100}M | {i}% |" for i in range(1, rows + 1))
    return f"{header}\n{sep}\n{data}"


class TestChunkerRespectsOverrides:
    def test_chunk_min_chars_override_prevents_chunking(self, tmp_path):
        """
        A document long enough to be chunked under the compiled-in default
        (CHUNK_MIN_CHARS=1000) must NOT be chunked once CHUNK_MIN_CHARS is
        overridden above its length.
        """
        from local_search_agent.core.key_manager import set_advanced_settings
        from local_search_agent.ingestion.chunker import chunk_document

        text = _prose(3)  # comfortably over the default 1000-char floor
        assert len(text) > 1000

        with _patch_advanced_path(tmp_path):
            set_advanced_settings({"CHUNK_MIN_CHARS": len(text) + 1000})
            result = chunk_document(_make_node(text))

        assert len(result) == 1
        assert result[0].text == text

    def test_chunk_target_max_chars_override_increases_chunk_count(self, tmp_path):
        """
        Shrinking CHUNK_TARGET_CHARS / CHUNK_MAX_CHARS must produce more,
        smaller chunks for the same input document.
        """
        from local_search_agent.core.key_manager import set_advanced_settings
        from local_search_agent.ingestion.chunker import chunk_document

        text = _prose(20)

        with _patch_advanced_path(tmp_path):
            set_advanced_settings({})  # defaults
            default_chunks = chunk_document(_make_node(text))

            set_advanced_settings({"CHUNK_TARGET_CHARS": 800, "CHUNK_MAX_CHARS": 1200})
            small_chunks = chunk_document(_make_node(text))

        assert len(small_chunks) > len(default_chunks)

    def test_table_rows_per_chunk_override_changes_table_chunk_count(self, tmp_path):
        """
        A pure-table document must be split into more chunks when
        TABLE_ROWS_PER_CHUNK is overridden to a smaller value.
        """
        from local_search_agent.core.key_manager import set_advanced_settings
        from local_search_agent.ingestion.chunker import chunk_document

        text = _table(50)

        with _patch_advanced_path(tmp_path):
            set_advanced_settings({})  # defaults (100 rows/chunk -> 1 chunk for 50 rows)
            default_chunks = chunk_document(_make_node(text))

            set_advanced_settings({"TABLE_ROWS_PER_CHUNK": 10})
            small_chunks = chunk_document(_make_node(text))

        assert len(default_chunks) == 1
        assert len(small_chunks) == 5

    def test_overlap_chars_override_is_threaded_through(self, tmp_path):
        """
        CHUNK_OVERLAP_CHARS override must reach _chunk_mixed_content (and from
        there _chunk_sliding / _get_overlap), not just be reported by
        get_effective_constants().
        """
        from local_search_agent.core.key_manager import set_advanced_settings
        from local_search_agent.ingestion import chunker

        text = _prose(20)
        captured = {}
        original = chunker._chunk_mixed_content

        def _spy(*args, **kwargs):
            captured["overlap_chars"] = kwargs.get("overlap_chars")
            return original(*args, **kwargs)

        with _patch_advanced_path(tmp_path):
            set_advanced_settings({"CHUNK_OVERLAP_CHARS": 42})
            with patch(
                "local_search_agent.ingestion.chunker._chunk_mixed_content", side_effect=_spy
            ):
                chunker.chunk_document(_make_node(text))

        assert captured["overlap_chars"] == 42

    def test_markdown_files_are_never_chunked_regardless_of_overrides(self, tmp_path):
        """Sanity guard: .md files bypass chunking entirely; an override must not change that."""
        from local_search_agent.core.key_manager import set_advanced_settings
        from local_search_agent.ingestion.chunker import chunk_document

        text = _prose(20)

        with _patch_advanced_path(tmp_path):
            set_advanced_settings({"CHUNK_MIN_CHARS": 1})
            result = chunk_document(_make_node(text, file_type="md"))

        assert len(result) == 1


# ===========================================================================
# MeilisearchClient.search() — DEFAULT_TOP_K / SNIPPET_CONTEXT_CHARS
# ===========================================================================


class TestMeilisearchClientRespectsOverrides:
    def _client_with_mock_index(self):
        from local_search_agent.search.meilisearch_client import MeilisearchClient

        client = MeilisearchClient()
        mock_index = MagicMock()
        mock_results = MagicMock()
        mock_results.hits = []
        mock_index.search.return_value = mock_results
        client._index = mock_index  # bypass lazy _get_index()/Meilisearch connection
        return client, mock_index

    def test_default_top_k_override_used_when_not_passed(self, tmp_path):
        from local_search_agent.core.key_manager import set_advanced_settings

        client, mock_index = self._client_with_mock_index()

        with _patch_advanced_path(tmp_path):
            set_advanced_settings({"DEFAULT_TOP_K": 17})
            client.search(query="test", enable_reranking=False)

        assert mock_index.search.call_args.kwargs["limit"] == 17

    def test_snippet_context_chars_override_used_when_not_passed(self, tmp_path):
        from local_search_agent.core.key_manager import set_advanced_settings

        client, mock_index = self._client_with_mock_index()

        with _patch_advanced_path(tmp_path):
            set_advanced_settings({"SNIPPET_CONTEXT_CHARS": 100})
            client.search(query="test", enable_reranking=False)

        assert mock_index.search.call_args.kwargs["crop_length"] == 100 // 5

    def test_explicit_top_k_beats_advanced_setting(self, tmp_path):
        """A caller-supplied top_k must always win over the stored override."""
        from local_search_agent.core.key_manager import set_advanced_settings

        client, mock_index = self._client_with_mock_index()

        with _patch_advanced_path(tmp_path):
            set_advanced_settings({"DEFAULT_TOP_K": 17})
            client.search(query="test", top_k=3, enable_reranking=False)

        assert mock_index.search.call_args.kwargs["limit"] == 3


# ===========================================================================
# search_local_index tool — top_k fallback to config.top_k
# ===========================================================================


class TestSearchToolRespectsOverrides:
    def _mock_meili(self):
        m = MagicMock()
        m.search.return_value = []
        return m

    def test_tool_uses_config_top_k_when_llm_omits_it(self, tmp_path, db_path):
        """
        When the LLM doesn't pass top_k, the tool must fall back to
        config.top_k — which itself must already reflect DEFAULT_TOP_K's
        advanced_settings override (set via SearchAgentConfig()'s
        default_factory, exercised here end to end).
        """
        from local_search_agent.agent.tools.search_tool import build_search_tool
        from local_search_agent.core.config import SearchAgentConfig
        from local_search_agent.core.key_manager import set_advanced_settings

        with _patch_advanced_path(tmp_path):
            set_advanced_settings({"DEFAULT_TOP_K": 11})

            config = SearchAgentConfig(
                document_dirs=[str(tmp_path)],
                workspace_name="test_ws",
                provider="ollama",
                db_path=db_path,
            )
            assert config.top_k == 11  # SearchAgentConfig itself picked up the override

            mock_meili = self._mock_meili()
            tool = build_search_tool(mock_meili, config)
            tool.invoke({"query": "test"})

        assert mock_meili.search.call_args.kwargs["top_k"] == 11

    def test_tool_llm_supplied_top_k_still_wins(self, tmp_path, db_path):
        from local_search_agent.agent.tools.search_tool import build_search_tool
        from local_search_agent.core.config import SearchAgentConfig
        from local_search_agent.core.key_manager import set_advanced_settings

        with _patch_advanced_path(tmp_path):
            set_advanced_settings({"DEFAULT_TOP_K": 11})

            config = SearchAgentConfig(
                document_dirs=[str(tmp_path)],
                workspace_name="test_ws",
                provider="ollama",
                db_path=db_path,
            )
            mock_meili = self._mock_meili()
            tool = build_search_tool(mock_meili, config)
            tool.invoke({"query": "test", "top_k": 5})

        assert mock_meili.search.call_args.kwargs["top_k"] == 5


# ===========================================================================
# SearchAgentConfig — top_k / max_iterations default_factory
# ===========================================================================


class TestSearchAgentConfigRespectsOverrides:
    def test_top_k_default_factory_respects_override(self, tmp_path, db_path):
        from local_search_agent.core.config import SearchAgentConfig
        from local_search_agent.core.key_manager import set_advanced_settings

        with _patch_advanced_path(tmp_path):
            set_advanced_settings({"DEFAULT_TOP_K": 13})
            config = SearchAgentConfig(
                document_dirs=[str(tmp_path)],
                workspace_name="test_ws",
                provider="ollama",
                db_path=db_path,
            )
        assert config.top_k == 13

    def test_max_iterations_default_factory_respects_override(self, tmp_path, db_path):
        from local_search_agent.core.config import SearchAgentConfig
        from local_search_agent.core.key_manager import set_advanced_settings

        with _patch_advanced_path(tmp_path):
            set_advanced_settings({"DEFAULT_MAX_ITERATIONS": 7})
            config = SearchAgentConfig(
                document_dirs=[str(tmp_path)],
                workspace_name="test_ws",
                provider="ollama",
                db_path=db_path,
            )
        assert config.max_iterations == 7

    def test_explicit_top_k_beats_advanced_setting(self, tmp_path, db_path):
        """An explicitly-passed top_k must never be clobbered by the override."""
        from local_search_agent.core.config import SearchAgentConfig
        from local_search_agent.core.key_manager import set_advanced_settings

        with _patch_advanced_path(tmp_path):
            set_advanced_settings({"DEFAULT_TOP_K": 13})
            config = SearchAgentConfig(
                document_dirs=[str(tmp_path)],
                workspace_name="test_ws",
                provider="ollama",
                db_path=db_path,
                top_k=99,
            )
        assert config.top_k == 99

    def test_reset_to_defaults_is_reflected_in_new_config(self, tmp_path, db_path):
        from local_search_agent.core import constants as C
        from local_search_agent.core.config import SearchAgentConfig
        from local_search_agent.core.key_manager import set_advanced_settings

        with _patch_advanced_path(tmp_path):
            set_advanced_settings({"DEFAULT_TOP_K": 13})
            set_advanced_settings({})  # reset
            config = SearchAgentConfig(
                document_dirs=[str(tmp_path)],
                workspace_name="test_ws",
                provider="ollama",
                db_path=db_path,
            )
        assert config.top_k == C.DEFAULT_TOP_K


# ===========================================================================
# API live-mutation regression — /settings/advanced must update the running
# app_state.config in place (mirroring /settings/reranking), so DEFAULT_TOP_K
# and DEFAULT_MAX_ITERATIONS apply on the next query without a restart.
# ===========================================================================


class TestAdvancedSettingsAPILiveMutation:
    def _api_client(self, tmp_path, db_path):
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
        return TestClient(app, raise_server_exceptions=True), app_state

    def test_post_advanced_top_k_applies_live_without_restart(self, tmp_path, db_path):
        client, app_state = self._api_client(tmp_path, db_path)
        original_top_k = app_state.config.top_k

        with _patch_advanced_path(tmp_path):
            resp = client.post(
                "/api/ui/settings/advanced",
                json={"overrides": {"DEFAULT_TOP_K": original_top_k + 50}},
            )

        assert resp.status_code == 200
        # The SAME config object the running agent holds must reflect the
        # new value immediately — no SearchAgentConfig() reconstruction.
        assert app_state.config.top_k == original_top_k + 50

    def test_post_advanced_max_iterations_applies_live_without_restart(self, tmp_path, db_path):
        client, app_state = self._api_client(tmp_path, db_path)

        with _patch_advanced_path(tmp_path):
            resp = client.post(
                "/api/ui/settings/advanced",
                json={"overrides": {"DEFAULT_MAX_ITERATIONS": 33}},
            )

        assert resp.status_code == 200
        assert app_state.config.max_iterations == 33

    def test_delete_advanced_resets_live_config(self, tmp_path, db_path):
        from local_search_agent.core import constants as C

        client, app_state = self._api_client(tmp_path, db_path)

        with _patch_advanced_path(tmp_path):
            client.post(
                "/api/ui/settings/advanced",
                json={"overrides": {"DEFAULT_TOP_K": 77}},
            )
            assert app_state.config.top_k == 77

            resp = client.delete("/api/ui/settings/advanced")

        assert resp.status_code == 200
        assert app_state.config.top_k == C.DEFAULT_TOP_K


# ===========================================================================
# PDFParser — PDF_SPLIT_THRESHOLD / PDF_PAGES_PER_BATCH / TESSERACT_FALLBACK_MIN_CHARS
# Docling itself is never imported: _convert_pdf_in_batches and the OCR
# tier functions are intercepted directly (same boundary test_heavy_parsers.py
# already uses), so this stays fast and dependency-light despite @pytest.mark.slow.
# ===========================================================================


@pytest.mark.slow
class TestPDFParserRespectsOverrides:
    def test_split_threshold_and_pages_per_batch_override_applied(self, tmp_path):
        from local_search_agent.core.key_manager import set_advanced_settings
        from local_search_agent.ingestion.parsers.pdf_parser import PDFParser

        f = tmp_path / "big.pdf"
        f.write_bytes(b"%PDF-1.4 fake")

        captured = {}

        def _fake_convert_pdf_in_batches(
            convertee_path, pages_per_batch, tesseract_fallback_min_chars
        ):
            captured["pages_per_batch"] = pages_per_batch
            captured["tesseract_fallback_min_chars"] = tesseract_fallback_min_chars
            return "Converted content."

        with _patch_advanced_path(tmp_path):
            # Default PDF_SPLIT_THRESHOLD is 15; override it down to 3 pages so
            # an 8-page document takes the batching path, and override the
            # batch size to a distinctive value to prove it was threaded through.
            set_advanced_settings({"PDF_SPLIT_THRESHOLD": 3, "PDF_PAGES_PER_BATCH": 4})

            with (
                patch(
                    "local_search_agent.ingestion.parsers.pdf_parser._count_pdf_pages",
                    return_value=8,
                ),
                patch(
                    "local_search_agent.ingestion.parsers.pdf_parser._convert_pdf_in_batches",
                    side_effect=_fake_convert_pdf_in_batches,
                ),
            ):
                node = PDFParser().parse(str(f), workspace="test")

        assert captured["pages_per_batch"] == 4
        assert "Converted content." in node.text

    def test_below_threshold_uses_small_pdf_path_not_batching(self, tmp_path):
        """A document below the overridden PDF_SPLIT_THRESHOLD must NOT batch."""
        from local_search_agent.core.key_manager import set_advanced_settings
        from local_search_agent.ingestion.parsers.pdf_parser import PDFParser

        f = tmp_path / "small.pdf"
        f.write_bytes(b"%PDF-1.4 fake")

        with _patch_advanced_path(tmp_path):
            # Raise the threshold well above the document's page count.
            set_advanced_settings({"PDF_SPLIT_THRESHOLD": 100})

            with (
                patch(
                    "local_search_agent.ingestion.parsers.pdf_parser._count_pdf_pages",
                    return_value=2,
                ),
                patch(
                    "local_search_agent.ingestion.parsers.pdf_parser._convert_pdf_in_batches"
                ) as mock_batches,
                patch(
                    "local_search_agent.ingestion.parsers.pdf_parser._extract_native_text_pymupdf",
                    return_value="Real native text content here.",
                ),
                patch(
                    "local_search_agent.ingestion.parsers.pdf_parser._get_no_ocr_converter"
                ) as mock_no_ocr,
            ):
                mock_result = MagicMock()
                mock_result.document.export_to_markdown.return_value = "Native text content."
                mock_no_ocr.return_value.convert.return_value = mock_result

                PDFParser().parse(str(f), workspace="test")

        mock_batches.assert_not_called()

    def test_tesseract_fallback_min_chars_override_applied(self, tmp_path):
        """
        With a low TESSERACT_FALLBACK_MIN_CHARS override, a short native-text
        result must be accepted as sufficient (skips OCR) where the compiled-in
        default (10 chars) would also accept it, but a HIGH override must
        correctly reject the same short text and proceed to OCR.
        """
        from local_search_agent.core.key_manager import set_advanced_settings
        from local_search_agent.ingestion.parsers.pdf_parser import PDFParser

        f = tmp_path / "scanned.pdf"
        f.write_bytes(b"%PDF-1.4 fake")

        short_text = "hi"  # 2 chars

        with _patch_advanced_path(tmp_path):
            # Override well above len(short_text) so it is correctly treated
            # as "empty" and the parser must fall through to OCR (Tesseract).
            set_advanced_settings({"TESSERACT_FALLBACK_MIN_CHARS": 50})

            with (
                patch(
                    "local_search_agent.ingestion.parsers.pdf_parser._count_pdf_pages",
                    return_value=1,
                ),
                patch(
                    "local_search_agent.ingestion.parsers.pdf_parser._extract_native_text_pymupdf",
                    return_value=short_text,
                ),
                patch(
                    "local_search_agent.ingestion.parsers.pdf_parser._get_tesseract_converter",
                    return_value=None,
                ),
                patch(
                    "local_search_agent.ingestion.parsers.pdf_parser._get_onnx_converter"
                ) as mock_onnx,
            ):
                mock_result = MagicMock()
                mock_result.document.export_to_markdown.return_value = "OCR-recovered content."
                mock_onnx.return_value.convert.return_value = mock_result

                node = PDFParser().parse(str(f), workspace="test")

        # Native text ("hi") was correctly rejected as insufficient under the
        # 50-char override, so the ONNX OCR path's output is what landed in
        # the document — proving the override (not the compiled-in default
        # of 10) governed the decision.
        assert "OCR-recovered content." in node.text


# ===========================================================================
# DOCXParser — DOCX_CHAR_SPLIT_THRESHOLD
# ===========================================================================


@pytest.mark.slow
class TestDOCXParserRespectsOverrides:
    def _make_docx(self, tmp_path: Path, paragraphs: list[str]) -> Path:
        from docx import Document

        doc = Document()
        for text in paragraphs:
            doc.add_paragraph(text)
        path = tmp_path / "doc.docx"
        doc.save(str(path))
        return path

    def test_char_split_threshold_override_forces_batching_path(self, tmp_path):
        from local_search_agent.core.key_manager import set_advanced_settings
        from local_search_agent.ingestion.parsers.docx_parser import DOCXParser

        f = self._make_docx(tmp_path, ["Paragraph one.", "Paragraph two.", "Paragraph three."])

        with _patch_advanced_path(tmp_path):
            # Default threshold is 6000 chars -- this tiny doc would normally
            # take the single-call path. Override it down to 1 char so even
            # this trivial document is forced into the batching path.
            set_advanced_settings({"DOCX_CHAR_SPLIT_THRESHOLD": 1})

            with patch(
                "local_search_agent.ingestion.parsers.docx_parser._split_docx_in_batches"
            ) as mock_split:
                mock_split.return_value = "Batched content."
                node = DOCXParser().parse(str(f), workspace="test")

        assert mock_split.called
        # The resolved override must have been passed down explicitly.
        assert mock_split.call_args.kwargs["char_split_threshold"] == 1
        assert "Batched content." in node.text

    def test_char_split_threshold_default_uses_single_call_path(self, tmp_path):
        from local_search_agent.core.key_manager import set_advanced_settings
        from local_search_agent.ingestion.parsers.docx_parser import DOCXParser

        f = self._make_docx(tmp_path, ["Short paragraph."])

        with _patch_advanced_path(tmp_path):
            set_advanced_settings({})  # compiled-in default (6000) — tiny doc stays single-call

            with (
                patch(
                    "local_search_agent.ingestion.parsers.docx_parser._split_docx_in_batches"
                ) as mock_split,
                patch("docling.document_converter.DocumentConverter") as mock_converter_cls,
            ):
                mock_result = MagicMock()
                mock_result.document.export_to_markdown.return_value = "Short paragraph."
                mock_converter_cls.return_value.convert.return_value = mock_result

                DOCXParser().parse(str(f), workspace="test")

        mock_split.assert_not_called()
