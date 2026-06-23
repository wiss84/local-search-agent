"""
Re-ranking layer for the Local Search Agent framework.

Wraps `flashrank` (CPU-only cross-encoder re-ranker) to improve the
relevance ordering of BM25 candidates returned by Meilisearch.

Why re-ranking helps
--------------------
BM25 scores documents by term frequency and rarity — it has no understanding
of *meaning*. A cross-encoder re-ranker (flashrank) takes each query+chunk
pair together and scores them on a learned semantic basis. This catches:
  - Synonym / paraphrase mismatches (query says "fail", doc says "exception")
  - BM25 over-ranking short chunks that happen to contain rare query terms
    but are contextually thin

Flow
----
1. MeilisearchClient.search() fetches top_k * rerank_candidate_multiplier
   candidates from Meilisearch (wider BM25 pool).
2. Reranker.rerank() scores all candidates against the query.
3. Results are sorted by cross-encoder score and truncated to top_k.

Model caching
-------------
flashrank downloads its model on first use. The model is cached to
`<user_config_dir>/local-search-agent/models/flashrank` so it only
downloads once and survives across restarts.

Default model: ms-marco-TinyBERT-L-2-v2 (~17MB, CPU-only, fast).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Model used for re-ranking. TinyBERT-L-2-v2 is the best speed/quality
# tradeoff for a local, CPU-only setup.
DEFAULT_RERANK_MODEL = "ms-marco-TinyBERT-L-2-v2"


def _model_cache_dir() -> str:
    """Return the platform-appropriate path for the flashrank model cache."""
    try:
        from platformdirs import user_config_dir

        base = Path(user_config_dir("local-search-agent")) / "models" / "flashrank"
    except ImportError:
        # platformdirs not available — fall back to current working dir
        base = Path.cwd() / ".local-search-agent" / "models" / "flashrank"
    base.mkdir(parents=True, exist_ok=True)
    return str(base)


class Reranker:
    """
    Lazy-initialised wrapper around flashrank.Ranker.

    The underlying model is downloaded on the first call to rerank() and
    cached to the user config dir. Subsequent calls load from disk only.

    Parameters
    ----------
    model_name  : flashrank model name (default: ms-marco-TinyBERT-L-2-v2).
    cache_dir   : Directory to cache model files. Defaults to
                  <user_config_dir>/local-search-agent/models/flashrank.
    max_length  : Max token length per passage passed to the cross-encoder.
                  512 is the model's native limit; shorter values are faster.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANK_MODEL,
        cache_dir: Optional[str] = None,
        max_length: int = 512,
    ):
        self._model_name = model_name
        self._cache_dir = cache_dir or _model_cache_dir()
        self._max_length = max_length
        self._ranker = None  # Lazy init

    def _get_ranker(self):
        """Lazily load the flashrank Ranker (downloads model on first use)."""
        if self._ranker is None:
            try:
                from flashrank import Ranker
            except ImportError as e:
                raise ImportError(
                    "flashrank is not installed. Run: pip install 'flashrank>=0.2.10'"
                ) from e
            logger.info(
                "Loading re-ranking model %r from cache dir %r ...",
                self._model_name,
                self._cache_dir,
            )
            self._ranker = Ranker(
                model_name=self._model_name,
                cache_dir=self._cache_dir,
                max_length=self._max_length,
            )
            logger.info("Re-ranking model loaded.")
        return self._ranker

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        """
        Re-rank BM25 candidates with a cross-encoder and return the top_k results.

        Parameters
        ----------
        query      : The original user query string.
        candidates : List of result dicts from MeilisearchClient.search()
                     (each must have at least 'snippet' and 'doc_id').
        top_k      : Number of results to return after re-ranking.

        Returns
        -------
        List of result dicts (subset of candidates), sorted by cross-encoder
        relevance score descending, length <= top_k.
        """
        if not candidates:
            return []

        if len(candidates) <= top_k:
            # Nothing to gain from re-ranking fewer candidates than top_k
            return candidates

        try:
            from flashrank import RerankRequest

            ranker = self._get_ranker()

            # flashrank expects passages as list of {"id": ..., "text": ...}
            # Use the snippet as the passage text; fall back to title if empty.
            passages = [
                {
                    "id": i,
                    "text": c.get("snippet") or c.get("title") or "",
                    "meta": c,  # carry the full result dict through
                }
                for i, c in enumerate(candidates)
            ]

            request = RerankRequest(query=query, passages=passages)
            reranked = ranker.rerank(request)

            # reranked is a list of dicts: {"id": ..., "score": ..., "text": ..., "meta": ...}
            # Sort by score descending (flashrank may already return them sorted, but be explicit)
            reranked.sort(key=lambda x: x.get("score", 0.0), reverse=True)

            # Attach the re-ranker score to each result dict and truncate to top_k
            results = []
            for entry in reranked[:top_k]:
                result = dict(entry["meta"])
                result["rerank_score"] = entry.get("score", 0.0)
                results.append(result)

            logger.debug(
                "Re-ranked %d candidates → %d results for query %r.",
                len(candidates),
                len(results),
                query,
            )
            return results

        except Exception as e:
            # Re-ranking is non-critical: if it fails, fall back to BM25 ordering
            logger.warning("Re-ranking failed (falling back to BM25 order): %s", e)
            return candidates[:top_k]
