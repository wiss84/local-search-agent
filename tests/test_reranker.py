"""
Unit tests for the re-ranking layer (search/reranker.py).

flashrank itself is mocked throughout — these tests verify Reranker's own
logic (guard conditions, candidate-pool handling, graceful fallback on
failure) without requiring a real model download in CI.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _make_candidates(n, with_snippets=True):
    return [
        {
            "doc_id": f"doc{i}",
            "title": f"Title {i}",
            "snippet": f"some content about topic {i}" if with_snippets else "",
        }
        for i in range(n)
    ]


class TestRerankerGuardConditions:
    """Tests for early-return / no-op paths that don't touch flashrank at all."""

    def test_empty_candidates_returns_empty_list(self):
        from local_search_agent.search.reranker import Reranker

        reranker = Reranker()
        result = reranker.rerank(query="test", candidates=[], top_k=5)
        assert result == []

    def test_candidates_fewer_than_top_k_returned_unchanged(self):
        from local_search_agent.search.reranker import Reranker

        reranker = Reranker()
        candidates = _make_candidates(3)
        result = reranker.rerank(query="test", candidates=candidates, top_k=5)
        # Should return the original list untouched (no reranker model load)
        assert result == candidates

    def test_candidates_equal_to_top_k_returned_unchanged(self):
        from local_search_agent.search.reranker import Reranker

        reranker = Reranker()
        candidates = _make_candidates(5)
        result = reranker.rerank(query="test", candidates=candidates, top_k=5)
        assert result == candidates

    def test_guard_path_never_loads_model(self):
        """Confirm the cheap guard path never calls _get_ranker (i.e. never downloads)."""
        from local_search_agent.search.reranker import Reranker

        reranker = Reranker()
        with patch.object(reranker, "_get_ranker") as mock_get_ranker:
            reranker.rerank(query="test", candidates=_make_candidates(2), top_k=5)
            mock_get_ranker.assert_not_called()


class TestRerankerWithMockedFlashrank:
    """Tests for the actual re-ranking path, with flashrank mocked out."""

    def test_rerank_calls_ranker_and_returns_top_k(self):
        from local_search_agent.search.reranker import Reranker

        reranker = Reranker()
        candidates = _make_candidates(10)

        mock_ranker = MagicMock()
        # flashrank.Ranker.rerank returns list of dicts with id/score/meta
        mock_ranker.rerank.return_value = [
            {"id": i, "score": 1.0 - (i * 0.1), "meta": candidates[i]} for i in range(10)
        ]

        with patch.object(reranker, "_get_ranker", return_value=mock_ranker):
            with patch("flashrank.RerankRequest"):
                result = reranker.rerank(query="topic 3", candidates=candidates, top_k=3)

        assert len(result) == 3
        # Highest score (id=0) should be first
        assert result[0]["doc_id"] == "doc0"
        assert "rerank_score" in result[0]

    def test_rerank_sorts_by_score_descending(self):
        from local_search_agent.search.reranker import Reranker

        reranker = Reranker()
        candidates = _make_candidates(6)

        mock_ranker = MagicMock()
        # Deliberately return out-of-order scores to verify explicit sort
        mock_ranker.rerank.return_value = [
            {"id": 0, "score": 0.2, "meta": candidates[0]},
            {"id": 1, "score": 0.9, "meta": candidates[1]},
            {"id": 2, "score": 0.5, "meta": candidates[2]},
            {"id": 3, "score": 0.1, "meta": candidates[3]},
            {"id": 4, "score": 0.7, "meta": candidates[4]},
            {"id": 5, "score": 0.3, "meta": candidates[5]},
        ]

        with patch.object(reranker, "_get_ranker", return_value=mock_ranker):
            with patch("flashrank.RerankRequest"):
                result = reranker.rerank(query="q", candidates=candidates, top_k=3)

        assert [r["doc_id"] for r in result] == ["doc1", "doc4", "doc2"]

    def test_rerank_falls_back_to_bm25_order_on_exception(self):
        from local_search_agent.search.reranker import Reranker

        reranker = Reranker()
        candidates = _make_candidates(8)

        with patch.object(reranker, "_get_ranker", side_effect=RuntimeError("model load failed")):
            result = reranker.rerank(query="q", candidates=candidates, top_k=4)

        # Falls back to first top_k of original (BM25) order, doesn't raise
        assert result == candidates[:4]

    def test_rerank_uses_title_fallback_when_snippet_missing(self):
        from local_search_agent.search.reranker import Reranker

        reranker = Reranker()
        candidates = _make_candidates(6, with_snippets=False)

        mock_ranker = MagicMock()
        mock_ranker.rerank.return_value = [
            {"id": i, "score": 1.0, "meta": candidates[i]} for i in range(6)
        ]

        captured_passages = {}

        def _capture_request(query, passages):
            captured_passages["value"] = passages
            return MagicMock()

        with patch.object(reranker, "_get_ranker", return_value=mock_ranker):
            with patch("flashrank.RerankRequest", side_effect=_capture_request):
                reranker.rerank(query="q", candidates=candidates, top_k=3)

        # Passage text should fall back to title since snippet is empty
        assert captured_passages["value"][0]["text"] == "Title 0"


class TestRerankerModelCaching:
    def test_cache_dir_defaults_to_user_config_dir(self):
        from local_search_agent.search.reranker import _model_cache_dir

        cache_dir = _model_cache_dir()
        assert "flashrank" in cache_dir
        assert "local-search-agent" in cache_dir

    def test_explicit_cache_dir_is_respected(self, tmp_path):
        from local_search_agent.search.reranker import Reranker

        custom_dir = str(tmp_path / "my_custom_cache")
        reranker = Reranker(cache_dir=custom_dir)
        assert reranker._cache_dir == custom_dir

    def test_get_ranker_raises_clear_error_if_flashrank_missing(self):
        from local_search_agent.search.reranker import Reranker

        reranker = Reranker()
        with patch.dict("sys.modules", {"flashrank": None}):
            with pytest.raises(ImportError, match="flashrank"):
                reranker._get_ranker()

    def test_get_ranker_only_loads_once(self):
        from local_search_agent.search.reranker import Reranker

        reranker = Reranker()
        mock_ranker_instance = MagicMock()

        with patch("flashrank.Ranker", return_value=mock_ranker_instance) as MockRankerClass:
            r1 = reranker._get_ranker()
            r2 = reranker._get_ranker()

        assert r1 is r2
        MockRankerClass.assert_called_once()
