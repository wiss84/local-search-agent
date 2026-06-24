"""
search_local_index — LangChain tool for the agent loop.

Calls MeilisearchClient.search() and formats results as a structured
string. Optionally applies query expansion (Phase 5 Option C) when
enable_query_expansion=True in config.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Maps the human-friendly preset the LLM passes to a timedelta
_RECENCY_DELTAS: dict[str, Optional[timedelta]] = {
    "1d": timedelta(days=1),
    "3d": timedelta(days=3),
    "7d": timedelta(days=7),
    "1m": timedelta(days=30),
    "6m": timedelta(days=182),
    "1y": timedelta(days=365),
    "all": None,  # no date filter
}


def _recency_to_iso(preset: Optional[str]) -> Optional[str]:
    """
    Convert a recency preset string (e.g. "7d") to an ISO-8601 cutoff timestamp.
    Returns None when the preset is "all" or unrecognised (meaning: no filter).
    """
    if not preset or preset.strip().lower() in ("all", ""):
        return None
    delta = _RECENCY_DELTAS.get(preset.strip().lower())
    if delta is None:
        return None
    cutoff = datetime.now(timezone.utc) - delta
    return cutoff.strftime("%Y-%m-%dT%H:%M:%S")


def build_search_tool(meili_client, config):
    """
    Factory that creates the search_local_index tool.

    Parameters
    ----------
    meili_client : MeilisearchClient instance
    config       : SearchAgentConfig instance
    """
    from local_search_agent.search.query_builder import QueryBuilder

    @tool
    def search_local_index(
        query: str,
        file_type: Optional[str] = None,
        top_k: Optional[int] = None,
        date_filter: Optional[str] = None,
    ) -> str:
        """
        Search the local document index for relevant documents.

        Use this tool first before answering any question.
        Returns a list of matching documents with short snippets and summaries.
        If a snippet looks relevant but incomplete, use fetch_local_url to read the full document.

        Args:
            query: Short keyword query (3-6 words). Focus on key terms, not full sentences.
            file_type: Optional filter. One of: "pdf", "docx", "html", "xlsx", "txt", "md".
            top_k: Number of results to return (defaults to the configured top_k, max 20).
            date_filter: Optional recency filter. One of: "1d" (last 24h), "3d" (last 3 days),
                "7d" (last 7 days), "1m" (last month), "6m" (last 6 months),
                "1y" (last year), "all" (no filter, default).
                Use when the user asks for recent documents or mentions a time period.
        """
        if top_k is None:
            top_k = config.top_k
        top_k = min(max(1, top_k), 20)

        # Phase 5 Option C: query expansion
        effective_query = query
        if config.enable_query_expansion:
            try:
                from local_search_agent.semantic.query_expander import QueryExpander

                expander = QueryExpander(llm=None)  # index-based expansion (no LLM cost)
                effective_query = expander.expand(
                    query=query,
                    meili_client=meili_client,
                    workspace=config.workspace_name,
                )
                if effective_query != query:
                    logger.debug("Query expanded: %r → %r", query, effective_query)
            except Exception as e:
                logger.warning("Query expansion failed, using original query: %s", e)
                effective_query = query

        filter_expr = QueryBuilder(
            workspace=config.workspace_name,
            file_type=file_type if file_type else None,
            modified_after=_recency_to_iso(date_filter),
        ).build()

        try:
            results = meili_client.search(
                query=effective_query,
                top_k=top_k,
                filter_expr=filter_expr,
                enable_reranking=config.enable_reranking,
                rerank_candidate_multiplier=config.rerank_candidate_multiplier,
            )
        except Exception as e:
            logger.error("search_local_index failed: %s", e)
            return f"Search failed: {e}"

        if not results:
            return (
                f"No documents found for query: {query!r}\n"
                "Try different keywords or a broader query."
            )

        lines = [f"Found {len(results)} result(s) for query: {query!r}\n"]

        for i, r in enumerate(results, 1):
            doc_id = r["doc_id"]
            docs_url = config.docs_url(doc_id)
            summary = r.get("summary", "")

            lines.append(
                f"[{i}] {r['title']} ({r['file_type'].upper()})\n"
                f"    doc_id   : {doc_id}\n"
                + (f"    summary  : {summary}\n" if summary else "")
                + f"    snippet  : {r['snippet'] or '(no snippet available)'}\n"
                f"    docs_url : {docs_url}\n"
                f"    modified : {r['modified_at']}\n"
            )

        return "\n".join(lines)

    return search_local_index
