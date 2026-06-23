"""
Unit tests for the search layer (search/meilisearch_client.py + query_builder.py).

MeilisearchClient tests use a mock SDK — no live Meilisearch needed.
QueryBuilder tests are pure logic — no mocking needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from local_search_agent.core.document_node import DocumentNode
from local_search_agent.search.query_builder import QueryBuilder

# ---------------------------------------------------------------------------
# QueryBuilder tests — pure logic, no mocking
# ---------------------------------------------------------------------------


class TestQueryBuilder:
    def test_no_filters_returns_none(self):
        assert QueryBuilder().build() is None

    def test_workspace_filter(self):
        expr = QueryBuilder(workspace="finance").build()
        assert 'workspace = "finance"' in expr

    def test_single_file_type(self):
        expr = QueryBuilder(file_type="pdf").build()
        assert 'file_type = "pdf"' in expr

    def test_multiple_file_types_or_combined(self):
        expr = QueryBuilder(file_type=["pdf", "docx"]).build()
        assert 'file_type = "pdf"' in expr
        assert 'file_type = "docx"' in expr
        assert " OR " in expr

    def test_modified_after(self):
        expr = QueryBuilder(modified_after="2024-01-01T00:00:00").build()
        assert 'modified_at > "2024-01-01T00:00:00"' in expr

    def test_modified_before(self):
        expr = QueryBuilder(modified_before="2025-01-01T00:00:00").build()
        assert 'modified_at < "2025-01-01T00:00:00"' in expr

    def test_combined_filters_use_and(self):
        expr = QueryBuilder(workspace="finance", file_type="pdf").build()
        assert " AND " in expr
        assert 'workspace = "finance"' in expr
        assert 'file_type = "pdf"' in expr

    def test_folder_path_filter(self):
        expr = QueryBuilder(folder_path="/shares/finance").build()
        assert 'folder_path = "/shares/finance"' in expr

    def test_raw_filter_appended(self):
        expr = QueryBuilder(workspace="hr", raw='concepts = "turnover"').build()
        assert 'concepts = "turnover"' in expr
        assert 'workspace = "hr"' in expr

    def test_repr_shows_filter(self):
        qb = QueryBuilder(workspace="it")
        assert "it" in repr(qb)


# ---------------------------------------------------------------------------
# MeilisearchClient tests — mock the SDK
# ---------------------------------------------------------------------------


class TestMeilisearchClient:
    """
    Tests for MeilisearchClient using a fully mocked meilisearch_python_sdk.
    The mock prevents any real network calls.
    """

    def _make_client(self, mock_sdk_client):
        """Helper: create a MeilisearchClient with pre-injected mock."""
        from local_search_agent.search.meilisearch_client import MeilisearchClient

        client = MeilisearchClient(
            url="http://localhost:7700",
            api_key="test_key",
            index_name="test_index",
        )
        # Inject mock directly (bypass lazy init)
        client._client = mock_sdk_client
        return client

    def _make_mock_sdk(self):
        """Create a mock that mimics the meilisearch_python_sdk.Client interface."""
        mock_client = MagicMock()

        # Mock task polling
        mock_task = MagicMock()
        mock_task.task_uid = 1
        mock_task.status = "succeeded"
        mock_client.get_task.return_value = mock_task
        mock_client.create_index.return_value = mock_task
        mock_client.delete_index.return_value = mock_task

        # Mock index
        mock_index = MagicMock()
        mock_index.add_documents.return_value = mock_task
        mock_index.delete_document.return_value = mock_task
        mock_index.update_searchable_attributes.return_value = mock_task
        mock_index.update_filterable_attributes.return_value = mock_task
        mock_index.update_settings.return_value = mock_task

        mock_client.get_index.return_value = mock_index

        # Mock health
        mock_health = MagicMock()
        mock_health.status = "available"
        mock_client.health.return_value = mock_health

        return mock_client, mock_index

    def _make_node(self, tmp_path, name="report.txt"):
        f = tmp_path / name
        f.write_text("Test content.", encoding="utf-8")
        return DocumentNode.from_file(str(f), text="Test content.", workspace="test_ws")

    def test_is_healthy_true(self, tmp_path):
        mock_client, _ = self._make_mock_sdk()
        client = self._make_client(mock_client)
        assert client.is_healthy() is True

    def test_is_healthy_false_on_exception(self, tmp_path):
        mock_client, _ = self._make_mock_sdk()
        mock_client.health.side_effect = ConnectionError("down")
        client = self._make_client(mock_client)
        assert client.is_healthy() is False

    def test_index_documents_calls_add_documents(self, tmp_path):
        mock_client, mock_index = self._make_mock_sdk()
        client = self._make_client(mock_client)
        client._index = mock_index  # skip _get_index lazy init

        nodes = [self._make_node(tmp_path, f"doc{i}.txt") for i in range(3)]
        client.index_documents(nodes)

        mock_index.add_documents.assert_called_once()
        docs_passed = mock_index.add_documents.call_args[0][0]
        assert len(docs_passed) == 3

    def test_index_documents_empty_list_is_noop(self, tmp_path):
        mock_client, mock_index = self._make_mock_sdk()
        client = self._make_client(mock_client)
        client._index = mock_index

        client.index_documents([])
        mock_index.add_documents.assert_not_called()

    def test_search_returns_list_of_dicts(self, tmp_path):
        mock_client, mock_index = self._make_mock_sdk()

        # Mock search results
        mock_hit = {
            "doc_id": "abc123",
            "title": "Finance Report",
            "file_type": "pdf",
            "workspace": "finance",
            "source_path": "/shares/finance/report.pdf",
            "modified_at": "2024-09-30T10:00:00+02:00",
            "concepts": ["finance", "AWS"],
            "_formatted": {"text": "AWS spend was $1.2M"},
        }
        mock_results = MagicMock()
        mock_results.hits = [mock_hit]
        mock_index.search.return_value = mock_results

        client = self._make_client(mock_client)
        client._index = mock_index

        # Re-ranking is unrelated to this test and would otherwise trigger a
        # real flashrank model load; disable it explicitly.
        results = client.search("AWS spend", enable_reranking=False)

        assert len(results) == 1
        assert results[0]["doc_id"] == "abc123"
        assert results[0]["title"] == "Finance Report"
        assert "AWS" in results[0]["snippet"]

    def test_search_returns_empty_on_exception(self, tmp_path):
        mock_client, mock_index = self._make_mock_sdk()
        mock_index.search.side_effect = RuntimeError("search failed")

        client = self._make_client(mock_client)
        client._index = mock_index

        results = client.search("anything")
        assert results == []

    def test_search_strips_highlight_markers(self, tmp_path):
        mock_client, mock_index = self._make_mock_sdk()

        mock_hit = {
            "doc_id": "xyz",
            "title": "Doc",
            "file_type": "txt",
            "workspace": "ws",
            "source_path": "/doc.txt",
            "modified_at": "2025-01-01T00:00:00+00:00",
            "concepts": [],
            "_formatted": {"text": "The <em>AWS</em> spend was high"},
        }
        mock_results = MagicMock()
        mock_results.hits = [mock_hit]
        mock_index.search.return_value = mock_results

        client = self._make_client(mock_client)
        client._index = mock_index

        results = client.search("AWS", enable_reranking=False)
        assert "<em>" not in results[0]["snippet"]
        assert "AWS" in results[0]["snippet"]

    def test_delete_document_calls_sdk(self, tmp_path):
        mock_client, mock_index = self._make_mock_sdk()
        client = self._make_client(mock_client)
        client._index = mock_index

        client.delete_document("abc123")
        mock_index.delete_document.assert_called_once_with("abc123")

    def test_get_index_stats_returns_dict(self, tmp_path):
        mock_client, mock_index = self._make_mock_sdk()
        mock_stats = MagicMock()
        mock_stats.number_of_documents = 42
        mock_stats.is_indexing = False
        mock_stats.field_distribution = {"title": 42}
        mock_index.get_stats.return_value = mock_stats

        client = self._make_client(mock_client)
        client._index = mock_index

        stats = client.get_index_stats()
        assert stats["number_of_documents"] == 42
        assert stats["is_indexing"] is False


# ---------------------------------------------------------------------------
# MeilisearchClient re-ranking integration tests
# ---------------------------------------------------------------------------


class TestMeilisearchClientReranking:
    """
    Tests for the re-ranking integration in MeilisearchClient.search().
    The Reranker itself is mocked here — see test_reranker.py for
    Reranker-specific unit tests (including a real flashrank smoke test).
    """

    def _make_client_with_hits(self, hits: list[dict]):
        from local_search_agent.search.meilisearch_client import MeilisearchClient

        client = MeilisearchClient(
            url="http://localhost:7700", api_key="test_key", index_name="test_index"
        )
        mock_client = MagicMock()
        mock_index = MagicMock()
        mock_results = MagicMock()
        mock_results.hits = hits
        mock_index.search.return_value = mock_results
        client._client = mock_client
        client._index = mock_index
        return client, mock_index

    def _hit(self, doc_id, text="some content"):
        return {
            "doc_id": doc_id,
            "title": f"Doc {doc_id}",
            "file_type": "txt",
            "workspace": "ws",
            "source_path": f"/{doc_id}.txt",
            "modified_at": "2025-01-01T00:00:00+00:00",
            "concepts": [],
            "_formatted": {"text": text},
        }

    def test_reranking_disabled_uses_meili_limit_directly(self):
        client, mock_index = self._make_client_with_hits([self._hit("a")])
        client.search("query", top_k=5, enable_reranking=False)
        _, kwargs = mock_index.search.call_args
        assert kwargs["limit"] == 5

    def test_reranking_enabled_widens_meili_limit(self):
        client, mock_index = self._make_client_with_hits([self._hit("a")])
        client.search("query", top_k=5, enable_reranking=True, rerank_candidate_multiplier=4)
        _, kwargs = mock_index.search.call_args
        assert kwargs["limit"] == 20  # 5 * 4

    def test_reranking_limit_capped_at_1000(self):
        client, mock_index = self._make_client_with_hits([self._hit("a")])
        client.search("query", top_k=500, enable_reranking=True, rerank_candidate_multiplier=10)
        _, kwargs = mock_index.search.call_args
        assert kwargs["limit"] == 1000

    def test_reranker_called_with_candidates_and_top_k(self):
        hits = [self._hit("a"), self._hit("b"), self._hit("c")]
        client, mock_index = self._make_client_with_hits(hits)

        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = hits[:2]
        client._reranker = mock_reranker

        results = client.search("query", top_k=2, enable_reranking=True)

        mock_reranker.rerank.assert_called_once()
        call_kwargs = mock_reranker.rerank.call_args.kwargs
        assert call_kwargs["query"] == "query"
        assert len(call_kwargs["candidates"]) == 3
        assert call_kwargs["top_k"] == 2
        assert len(results) == 2

    def test_reranker_not_called_when_disabled(self):
        hits = [self._hit("a")]
        client, mock_index = self._make_client_with_hits(hits)

        mock_reranker = MagicMock()
        client._reranker = mock_reranker

        client.search("query", enable_reranking=False)
        mock_reranker.rerank.assert_not_called()

    def test_reranker_not_called_on_empty_hits(self):
        client, mock_index = self._make_client_with_hits([])

        mock_reranker = MagicMock()
        client._reranker = mock_reranker

        results = client.search("query", enable_reranking=True)
        mock_reranker.rerank.assert_not_called()
        assert results == []
