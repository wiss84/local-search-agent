"""
get_related_docs — LangChain tool for the agent loop (Phase 5).

Uses the LinkGraph SQLite store to find documents related to a given doc_id
by shared concepts (same_topic), citations (references), or folder proximity.

Falls back gracefully if the link graph is not enabled in config.
"""

from __future__ import annotations

import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def build_graph_tool(config, workspace_manager=None):
    """
    Factory for the get_related_docs tool.

    Parameters
    ----------
    config           : SearchAgentConfig
    workspace_manager: WorkspaceManager for title lookups (optional)
    """

    @tool
    def get_related_docs(doc_id: str, max_results: int = 5) -> str:
        """
        Find documents related to a given document by shared concepts or citations.

        Use this after fetching a document when you want to explore related content
        without running another keyword search.

        Args:
            doc_id: The document ID to find related documents for.
            max_results: Maximum number of related documents to return (default 5).
        """
        if not config.enable_link_graph:
            return (
                "get_related_docs is not available: enable_link_graph=False in config. "
                "Set enable_link_graph=True in SearchAgentConfig to use this tool. "
                "Use search_local_index with different keywords to find related documents."
            )

        try:
            from local_search_agent.semantic.link_graph import LinkGraph
        except ImportError:
            return "ERROR: semantic module not available. Install with: pip install -e '.[agent]'"

        try:
            graph = LinkGraph(db_path=config.db_path)
            related = graph.get_related(doc_id, limit=max_results)
        except Exception as e:
            logger.error("get_related_docs LinkGraph query failed: %s", e)
            return f"ERROR: Link graph query failed: {e}"

        if not related:
            return (
                f"No related documents found for doc_id={doc_id!r}. "
                "The link graph may not have been built yet — re-run ingestion with "
                "enable_link_graph=True to build cross-document relationships."
            )

        lines = [f"Found {len(related)} related document(s) for doc_id={doc_id!r}:\n"]
        for i, link in enumerate(related, 1):
            target_id = link["target_doc_id"]
            text_url = config.text_url(target_id)
            docs_url = config.docs_url(target_id)

            # Look up title if workspace_manager provided
            title = target_id
            if workspace_manager is not None:
                node = workspace_manager.get_document(target_id)
                if node:
                    title = node.title

            lines.append(
                f"[{i}] {title}\n"
                f"    doc_id      : {target_id}\n"
                f"    relation    : {link['relation_type']}\n"
                f"    weight      : {link['weight']:.2f}\n"
                f"    text_url    : {text_url}\n"
                f"    docs_url    : {docs_url}\n"
            )

        return "\n".join(lines)

    return get_related_docs
