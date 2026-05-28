"""
Option C: Query Expansion Loop — query-time semantic expansion.

Responsibility
--------------
Before calling Meilisearch, expand the user's query by:
1. Extracting key terms from the query
2. Looking up synonyms/concepts stored in the index for those terms
3. Building an expanded query that includes alternative phrasings

This makes BM25 match documents even when the user's exact words differ
from the indexed text — e.g. "morale" matches docs indexed with
"employee happiness", "job satisfaction", "turnover rate".

Two expansion strategies
-------------------------
A. LLM-based (rich, flexible):
   Calls the LLM to generate synonyms and alternative phrasings for the query.
   One cheap LLM call (small prompt, 1-2 sentences output).
   Used when an LLM is available.

B. Index-based (fast, no LLM):
   Looks up the `synonyms` field of the top-k search results and extracts
   any terms that overlap with the query. Then re-searches with those terms.
   Zero LLM cost. Works even with Ollama or no API key.

Design
------
- QueryExpander is stateless and thread-safe.
- Expansion is opt-in: the agent can call expand_query() before search_local_index.
- Falls back to original query if expansion fails.
- Expanded query never replaces the original — it appends terms:
  "morale" → "morale employee happiness job satisfaction turnover"
  This ensures original exact matches still rank highest.

Usage
-----
    from local_search_agent.semantic.query_expander import QueryExpander

    expander = QueryExpander(llm=llm)   # or QueryExpander() for index-based only
    expanded = expander.expand(query="morale", workspace="hr")
    results = meili_client.search(expanded, ...)
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

_EXPAND_PROMPT = """\
Generate alternative search terms for the query below to improve document retrieval.

Query: "{query}"

Respond with ONLY a JSON array of 5-10 alternative terms/phrases. No explanation.
Include: synonyms, abbreviations, related concepts, and common alternative phrasings.

Example for "AWS spend":
["Amazon Web Services cost", "cloud infrastructure budget", "EC2 charges", "cloud spend", "infrastructure cost"]

JSON array:"""


class QueryExpander:
    """
    Expand a user query with synonyms and related terms before searching.

    Parameters
    ----------
    llm : Optional LangChain BaseChatModel. If None, uses index-based expansion only.
    """

    def __init__(self, llm=None):
        self._llm = llm

    def expand(
        self,
        query: str,
        meili_client=None,
        workspace: Optional[str] = None,
        top_k: int = 3,
    ) -> str:
        """
        Expand a query with synonyms and alternative phrasings.

        Strategy:
        1. If LLM available → LLM-based expansion (richer)
        2. Else if meili_client available → index-based expansion (fast, free)
        3. Else → return original query unchanged

        Parameters
        ----------
        query        : Original user query string.
        meili_client : Optional MeilisearchClient for index-based expansion.
        workspace    : Workspace name for index-based lookup.
        top_k        : Number of documents to sample for index-based expansion.

        Returns
        -------
        Expanded query string (original terms + additional terms).
        Always returns a valid string — falls back to original on any error.
        """
        if not query.strip():
            return query

        try:
            if self._llm is not None:
                return self._llm_expand(query)
            elif meili_client is not None:
                return self._index_expand(query, meili_client, workspace, top_k)
            else:
                return query
        except Exception as e:
            logger.warning("QueryExpander failed for %r: %s. Using original query.", query, e)
            return query

    def _llm_expand(self, query: str) -> str:
        """Use the LLM to generate alternative search terms."""
        import json

        from langchain_core.messages import HumanMessage

        prompt = _EXPAND_PROMPT.format(query=query)

        try:
            response = self._llm.invoke([HumanMessage(content=prompt)])
            raw = response.content if isinstance(response.content, str) else str(response.content)

            # Strip markdown fences
            clean = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
            terms = json.loads(clean)

            if isinstance(terms, list):
                extra = " ".join(
                    str(t).strip() for t in terms
                    if t and str(t).strip().lower() not in query.lower()
                )
                expanded = f"{query} {extra}".strip()
                logger.debug("LLM query expansion: %r → %r", query, expanded)
                return expanded
        except Exception as e:
            logger.warning("LLM query expansion failed: %s", e)

        return query

    def _index_expand(
        self,
        query: str,
        meili_client,
        workspace: Optional[str],
        top_k: int,
    ) -> str:
        """
        Index-based expansion: sample synonyms from top search results.

        Searches with the original query, collects synonyms from matched
        documents, and appends any synonym terms that are not already in
        the query.
        """
        from local_search_agent.search.query_builder import QueryBuilder

        filter_expr = QueryBuilder(workspace=workspace).build() if workspace else None

        try:
            results = meili_client.search(
                query=query,
                top_k=top_k,
                filter_expr=filter_expr,
            )
        except Exception as e:
            logger.warning("Index-based expansion search failed: %s", e)
            return query

        # Collect synonyms from matched documents
        all_synonyms: list[str] = []
        for r in results:
            all_synonyms.extend(r.get("concepts", []))

        if not all_synonyms:
            return query

        # Filter: only add terms not already in the query
        query_lower = query.lower()
        new_terms = [
            s for s in dict.fromkeys(all_synonyms)   # deduplicate
            if s.lower() not in query_lower and len(s) > 2
        ][:8]   # cap additions

        if not new_terms:
            return query

        expanded = f"{query} {' '.join(new_terms)}".strip()
        logger.debug("Index query expansion: %r → %r", query, expanded)
        return expanded
