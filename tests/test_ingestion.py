"""
Unit tests for the ingestion pipeline (ingestion/pipeline.py).

Tests the orchestration logic: file discovery, delta logic, parser dispatch,
batch flushing, and error handling. Uses mocks for parsers and Meilisearch
so the tests have zero external dependencies and never hit the network.

Design notes:
- SUPPORTED_EXTENSIONS drives which files the pipeline walks, so .zip is
  always excluded. Of the files created by tmp_workspace: report.txt,
  handbook.md, data.xlsx, nested.txt are eligible; ignored.zip is not.
- mock_parser.parse is always called with keyword args
  (source_path=..., workspace=...) matching BaseParser.parse's signature,
  so we access call args via .call_args.kwargs.
- IngestStats.indexed counts chunks successfully registered in Meilisearch.
  IngestStats.files_indexed counts source files whose nodes entered the batch
  (incremented before flush, so it reflects parse success, not Meili success).
  IngestStats.failed counts both parse failures and flush failures.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from local_search_agent.core.config import SearchAgentConfig
from local_search_agent.core.document_node import DocumentNode
from local_search_agent.ingestion.parser import ParserError
from local_search_agent.ingestion.pipeline import IngestionPipeline, IngestStats
from local_search_agent.workspace.workspace_manager import WorkspaceManager

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_workspace(tmp_path):
    """
    Create a temp dir with files of different types.

    Eligible (in SUPPORTED_EXTENSIONS):   report.txt, handbook.md, data.xlsx, nested.txt
    Ineligible (not in SUPPORTED_EXTENSIONS): ignored.zip
    """
    (tmp_path / "report.txt").write_text("Q3 AWS spend was $1.2M.", encoding="utf-8")
    (tmp_path / "handbook.md").write_text("# Handbook\n\nWelcome.", encoding="utf-8")
    (tmp_path / "data.xlsx").write_bytes(b"fake xlsx content")
    (tmp_path / "ignored.zip").write_bytes(b"not supported")
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "nested.txt").write_text("Nested document.", encoding="utf-8")
    return tmp_path


@pytest.fixture
def config(tmp_path, tmp_workspace):
    return SearchAgentConfig(
        document_dirs=[str(tmp_workspace)],
        workspace_name="test_ws",
        db_path=str(tmp_path / "test.db"),
        provider="ollama",
    )


@pytest.fixture
def wm(config):
    wm = WorkspaceManager(db_path=config.db_path)
    wm.create_workspace(name="test_ws", document_dir=config.document_dirs[0])
    return wm


@pytest.fixture
def mock_meili():
    m = MagicMock()
    m.index_documents = MagicMock(return_value=None)
    return m


def _make_node(source_path: str, workspace: str = "test_ws") -> DocumentNode:
    """Return a real DocumentNode from a real file."""
    return DocumentNode.from_file(
        source_path=source_path,
        text="Test content.",
        workspace=workspace,
    )


def _txt_parser(extra_extensions: frozenset[str] = frozenset()) -> MagicMock:
    """
    Return a mock parser that handles .txt (and optionally extra extensions).
    parse() is called with keyword args matching BaseParser.parse(source_path, workspace).
    """
    extensions = frozenset({".txt"}) | extra_extensions
    mock = MagicMock()
    mock.can_parse = lambda p: any(p.endswith(ext) for ext in extensions)
    mock.parse = MagicMock(
        side_effect=lambda source_path, workspace, title=None: _make_node(source_path)
    )
    return mock


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

class TestFileDiscovery:
    def test_total_count_excludes_zip(self, config, wm, mock_meili, tmp_workspace):
        """
        stats.total == eligible files found by _walk.
        .zip is not in SUPPORTED_EXTENSIONS, so it's excluded.
        Expected: report.txt, handbook.md, data.xlsx, nested.txt → total == 4.
        """
        pipeline = IngestionPipeline(
            config=config, workspace_manager=wm,
            meili_client=mock_meili, parsers=[_txt_parser()],
        )
        stats = pipeline.run(force=True)
        assert stats.total == 4

    def test_skips_hidden_files(self, config, wm, mock_meili, tmp_workspace):
        (tmp_workspace / ".hidden.txt").write_text("hidden", encoding="utf-8")

        pipeline = IngestionPipeline(
            config=config, workspace_manager=wm,
            meili_client=mock_meili, parsers=[_txt_parser()],
        )
        pipeline.run(force=True)

        parsed_paths = [
            c.kwargs["source_path"] if c.kwargs else c.args[0]
            for c in pipeline._parsers[0].parse.call_args_list
        ]
        assert not any(".hidden.txt" in p for p in parsed_paths)

    def test_skips_hidden_directories(self, config, wm, mock_meili, tmp_workspace):
        hidden_dir = tmp_workspace / ".hidden_dir"
        hidden_dir.mkdir()
        (hidden_dir / "secret.txt").write_text("secret", encoding="utf-8")

        pipeline = IngestionPipeline(
            config=config, workspace_manager=wm,
            meili_client=mock_meili, parsers=[_txt_parser()],
        )
        stats = pipeline.run(force=True)
        assert stats.total == 4

    def test_walks_subdirectories(self, config, wm, mock_meili, tmp_workspace):
        """Files in subdirectories must be discovered."""
        pipeline = IngestionPipeline(
            config=config, workspace_manager=wm,
            meili_client=mock_meili, parsers=[_txt_parser()],
        )
        pipeline.run(force=True)

        parsed_paths = [
            c.kwargs["source_path"] if c.kwargs else c.args[0]
            for c in pipeline._parsers[0].parse.call_args_list
        ]
        assert any("nested.txt" in p for p in parsed_paths)

    def test_no_parser_increments_failed(self, config, wm, mock_meili, tmp_workspace):
        """Files with no matching parser increment stats.failed."""
        pipeline = IngestionPipeline(
            config=config, workspace_manager=wm,
            meili_client=mock_meili, parsers=[_txt_parser()],
        )
        stats = pipeline.run(force=True)
        # .md (1) + .xlsx (1) have no parser → failed >= 2
        assert stats.failed >= 2

    def test_continues_after_parse_error(self, config, wm, mock_meili, tmp_workspace):
        """A ParserError on one file must not abort the rest of the pipeline."""
        call_count = 0

        def flaky_parse(source_path, workspace, title=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ParserError(source_path, "Simulated failure on first file")
            return _make_node(source_path)

        mock_parser = MagicMock()
        mock_parser.can_parse = lambda p: p.endswith(".txt")
        mock_parser.parse = flaky_parse

        pipeline = IngestionPipeline(
            config=config, workspace_manager=wm,
            meili_client=mock_meili, parsers=[mock_parser],
        )
        stats = pipeline.run(force=True)

        assert stats.failed >= 1
        assert stats.files_indexed >= 1


# ---------------------------------------------------------------------------
# Delta / incremental indexing
# ---------------------------------------------------------------------------

class TestDeltaLogic:
    def test_unchanged_files_are_skipped(self, config, wm, mock_meili, tmp_workspace):
        """Running the pipeline twice: files indexed on first run, skipped on second."""
        parser = _txt_parser(extra_extensions=frozenset({".md", ".xlsx"}))
        pipeline = IngestionPipeline(
            config=config, workspace_manager=wm,
            meili_client=mock_meili, parsers=[parser],
        )

        first = pipeline.run(force=False)
        second = pipeline.run(force=False)

        assert second.skipped == first.files_indexed
        assert second.files_indexed == 0

    def test_force_reindexes_all(self, config, wm, mock_meili, tmp_workspace):
        """force=True must re-index even files that haven't changed."""
        parser = _txt_parser()
        pipeline = IngestionPipeline(
            config=config, workspace_manager=wm,
            meili_client=mock_meili, parsers=[parser],
        )

        pipeline.run(force=True)
        initial_call_count = parser.parse.call_count

        pipeline.run(force=True)
        assert parser.parse.call_count == initial_call_count * 2


# ---------------------------------------------------------------------------
# Batch logic
# ---------------------------------------------------------------------------

class TestBatchLogic:
    def test_batch_size_one_calls_meili_per_file(self, config, wm, mock_meili, tmp_workspace):
        """With batch_size=1 each document triggers a separate Meilisearch call."""
        parser = _txt_parser()
        pipeline = IngestionPipeline(
            config=config, workspace_manager=wm,
            meili_client=mock_meili, parsers=[parser],
            batch_size=1,
        )
        stats = pipeline.run(force=True)
        assert mock_meili.index_documents.call_count == stats.files_indexed

    def test_large_batch_single_meili_call(self, config, wm, mock_meili, tmp_workspace):
        """With batch_size > total files, all docs are flushed in one call."""
        parser = _txt_parser()
        pipeline = IngestionPipeline(
            config=config, workspace_manager=wm,
            meili_client=mock_meili, parsers=[parser],
            batch_size=1000,
        )
        stats = pipeline.run(force=True)
        assert mock_meili.index_documents.call_count == 1
        assert stats.files_indexed == 2

    def test_meili_failure_increments_failed(self, config, wm, mock_meili, tmp_workspace):
        """
        When Meilisearch raises, _flush_batch adds the batch size to stats.failed.
        Note: files_indexed is incremented before flushing (parse success), so it
        will be > 0 even when Meilisearch fails. What must be true is:
        - stats.failed > 0 (flush failure recorded)
        - stats.indexed == 0 (nothing was successfully registered in Meilisearch)
        - errors list is populated
        """
        mock_meili.index_documents.side_effect = RuntimeError("Meilisearch unavailable")
        parser = _txt_parser()
        pipeline = IngestionPipeline(
            config=config, workspace_manager=wm,
            meili_client=mock_meili, parsers=[parser],
        )
        stats = pipeline.run(force=True)

        assert stats.failed > 0         # flush failure counted
        assert stats.indexed == 0       # nothing made it into Meilisearch
        assert len(stats.errors) > 0    # error messages recorded


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------

class TestProgressCallback:
    def test_callback_called_for_each_file(self, config, wm, mock_meili, tmp_workspace):
        """progress_callback must be called at least once per eligible file."""
        events = []
        pipeline = IngestionPipeline(
            config=config, workspace_manager=wm,
            meili_client=mock_meili, parsers=[_txt_parser()],
        )
        pipeline.run(force=True, progress_callback=lambda *args: events.append(args))

        assert len(events) >= 3
        file_args = [e[4] for e in events]
        assert "__done__" in file_args


# ---------------------------------------------------------------------------
# IngestStats
# ---------------------------------------------------------------------------

class TestIngestStats:
    def test_str_contains_all_key_counts(self):
        s = IngestStats(total=10, indexed=8, skipped=2, failed=0, duration_s=1.5)
        text = str(s)
        assert "10" in text
        assert "8" in text
        assert "1.5" in text

    def test_default_stats_are_zero(self):
        s = IngestStats()
        assert s.total == 0
        assert s.indexed == 0
        assert s.files_indexed == 0
        assert s.skipped == 0
        assert s.failed == 0
        assert s.errors == []

    def test_errors_list_populated_on_failure(self, config, wm, mock_meili, tmp_workspace):
        mock_parser = MagicMock()
        mock_parser.can_parse = lambda p: p.endswith(".txt")
        mock_parser.parse = MagicMock(
            side_effect=ParserError("/some/file.txt", "Boom")
        )
        pipeline = IngestionPipeline(
            config=config, workspace_manager=wm,
            meili_client=mock_meili, parsers=[mock_parser],
        )
        stats = pipeline.run(force=True)

        assert len(stats.errors) >= 1
        assert any("Boom" in e or "file.txt" in e for e in stats.errors)
