"""
conftest.py — shared pytest fixtures and configuration.

Available to every test module in this directory without importing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_search_agent.core.config import SearchAgentConfig
from local_search_agent.core.document_node import DocumentNode
from local_search_agent.workspace.metadata_db import MetadataDB
from local_search_agent.workspace.workspace_manager import WorkspaceManager

# ---------------------------------------------------------------------------
# Pytest markers
# ---------------------------------------------------------------------------
# These are registered here so pytest doesn't warn about unknown marks.
# Usage:
#   @pytest.mark.unit         — pure logic, no I/O
#   @pytest.mark.integration  — hits real filesystem / SQLite
#   @pytest.mark.slow         — skipped in fast CI pass


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: fast, pure-logic tests (no I/O)")
    config.addinivalue_line("markers", "integration: tests that use filesystem or SQLite")
    config.addinivalue_line("markers", "slow: tests skipped in fast CI (--fast flag)")


def pytest_addoption(parser):
    parser.addoption(
        "--fast", action="store_true", default=False, help="Skip tests marked @pytest.mark.slow"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--fast"):
        skip_slow = pytest.mark.skip(reason="Skipped in --fast mode")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


# ---------------------------------------------------------------------------
# Core fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Path to the tests/fixtures/ directory containing sample files."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def db_path(tmp_path) -> str:
    """Path to an ephemeral test SQLite database."""
    return str(tmp_path / "test.db")


@pytest.fixture
def base_config(tmp_path, db_path) -> SearchAgentConfig:
    """Minimal SearchAgentConfig using Ollama (no API key) for unit tests."""
    return SearchAgentConfig(
        document_dirs=[str(tmp_path)],
        workspace_name="test_ws",
        provider="ollama",
        model_name="mistral",
        db_path=db_path,
        max_iterations=5,
        top_k=3,
    )


@pytest.fixture
def wm(db_path) -> WorkspaceManager:
    """WorkspaceManager backed by an ephemeral SQLite database."""
    return WorkspaceManager(db_path=db_path)


@pytest.fixture
def mdb(db_path) -> MetadataDB:
    """MetadataDB backed by an ephemeral SQLite database."""
    return MetadataDB(db_path=db_path)


@pytest.fixture
def workspace(wm, tmp_path) -> str:
    """Register a default workspace and return its name."""
    wm.create_workspace("test_ws", str(tmp_path))
    return "test_ws"


@pytest.fixture
def sample_txt_file(tmp_path) -> Path:
    """A real .txt file with realistic content."""
    f = tmp_path / "quarterly_report.txt"
    f.write_text(
        "Q3 2024 Financial Report\n\n"
        "AWS spend on Project Alpha reached $1.2M this quarter.\n"
        "Employee morale surveys showed improvement.\n"
        "Turnover rate dropped to 5%.\n"
        "The board approved a new infrastructure budget for 2025.\n",
        encoding="utf-8",
    )
    return f


@pytest.fixture
def sample_md_file(tmp_path) -> Path:
    """A real .md file with headings and tables."""
    f = tmp_path / "handbook.md"
    f.write_text(
        "# Employee Handbook\n\n"
        "## Remote Work Policy\n\n"
        "Employees may work remotely up to 3 days per week.\n\n"
        "## Leave Policy\n\n"
        "| Type | Days |\n"
        "| --- | --- |\n"
        "| Annual | 25 |\n"
        "| Sick | 10 |\n"
        "| Parental | 90 |\n",
        encoding="utf-8",
    )
    return f


@pytest.fixture
def sample_node(sample_txt_file) -> DocumentNode:
    """A DocumentNode built from the sample .txt file."""
    return DocumentNode.from_file(
        source_path=str(sample_txt_file),
        text=sample_txt_file.read_text(encoding="utf-8"),
        workspace="test_ws",
    )


@pytest.fixture
def registered_node(wm, workspace, sample_node) -> DocumentNode:
    """A DocumentNode that has been registered in the WorkspaceManager."""
    wm.register_document(sample_node)
    return sample_node


# ---------------------------------------------------------------------------
# Settings isolation fixture
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """
    Automatically mock settings and advanced_settings paths for all tests.
    This ensures tests use isolated, ephemeral settings instead of loading
    from the actual user config directory.
    """
    from local_search_agent.core import key_manager

    # Mock settings path (used by semantic, watch-mode, and reranking settings)
    monkeypatch.setattr(key_manager, "_settings_path", lambda: tmp_path / "settings.json")
    # Mock advanced settings path (used by get_effective_constants)
    monkeypatch.setattr(key_manager, "_advanced_path", lambda: tmp_path / "advanced_settings.json")


# ---------------------------------------------------------------------------
# Mock fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_meili_client():
    """
    A fully-mocked MeilisearchClient that returns one realistic search hit.
    Override .search.return_value or .index_documents in individual tests as needed.
    """
    from unittest.mock import MagicMock

    m = MagicMock()
    m.is_healthy.return_value = True
    m.search.return_value = [
        {
            "doc_id": "abc123def456abcd",
            "title": "Finance Report Q3",
            "file_type": "pdf",
            "workspace": "test_ws",
            "source_path": "/shares/finance/q3.pdf",
            "folder_path": "/shares/finance",
            "modified_at": "2024-09-30T10:00:00+02:00",
            "concepts": ["finance", "AWS"],
            "synonyms": ["Amazon Web Services"],
            "snippet": "AWS spend on Project Alpha reached $1.2M in Q3 2024.",
        }
    ]
    m.index_documents.return_value = None
    m.delete_document.return_value = None
    m.delete_index.return_value = None
    m.get_index_stats.return_value = {"number_of_documents": 1, "is_indexing": False}
    return m


@pytest.fixture
def mock_llm():
    """
    A MagicMock LLM that returns a default concept JSON.
    Override .invoke.return_value.content in individual tests as needed.
    """
    from unittest.mock import MagicMock

    llm = MagicMock()
    response = MagicMock()
    response.content = (
        '{"concepts": ["cloud costs", "AWS"], '
        '"synonyms": ["Amazon Web Services"], '
        '"entities": ["Project Alpha"], '
        '"summary": "Q3 finance report."}'
    )
    llm.invoke.return_value = response
    return llm
