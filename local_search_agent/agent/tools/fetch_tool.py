"""
fetch_local_url — LangChain tool for the agent loop.

Retrieves the full pre-cleaned Markdown text of a document from the
FastAPI file server's /text/{doc_id} endpoint.

The agent uses this after search_local_index when a snippet is relevant
but too short to answer the question fully.

Tool schema (what the LLM sees via bind_tools):
    fetch_local_url(doc_id: str) -> str

Returns the full document text, or an error message if the fetch fails.

Design note
-----------
We use HTTP (via httpx) rather than calling WorkspaceManager directly.
This keeps the agent decoupled from the server internals — the same tool
works whether the agent and server run in the same process or separately.
"""

from __future__ import annotations

import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Maximum characters to return to the agent per fetch.
# Prevents enormous documents from flooding the context window.
# The agent can search within the returned text to find what it needs.
_MAX_FETCH_CHARS = 12_000


def build_fetch_tool(config):
    """
    Factory that creates the fetch_local_url tool with injected config.

    Parameters
    ----------
    config : SearchAgentConfig (for server base URL)

    Returns
    -------
    A LangChain @tool decorated function ready for bind_tools().
    """

    @tool
    def fetch_local_url(doc_id: str) -> str:
        """
        Fetch the full text of a document from the local file server.

        Use this after search_local_index when you need more context than the snippet provides.
        Pass the doc_id from a search result to retrieve the complete document text.

        Args:
            doc_id: The document ID from a search_local_index result.
        """
        try:
            import httpx
        except ImportError:
            return "ERROR: httpx is not installed. Run: pip install 'httpx>=0.28.1'"

        url = config.text_url(doc_id)
        logger.info("Fetching document: %s", url)

        try:
            response = httpx.get(url, timeout=30.0)
        except httpx.ConnectError:
            return (
                f"ERROR: Could not connect to the file server at {config.server_base_url}. "
                "Is the server running? Call framework.start_file_server() first."
            )
        except httpx.TimeoutException:
            return f"ERROR: Request timed out fetching {url}."
        except Exception as e:
            logger.error("fetch_local_url failed for %s: %s", doc_id, e)
            return f"ERROR: Unexpected error fetching document {doc_id!r}: {e}"

        if response.status_code == 404:
            return (
                f"ERROR: Document {doc_id!r} not found. "
                "It may have been removed or not yet indexed."
            )
        if response.status_code == 410:
            return (
                f"ERROR: Document {doc_id!r} was indexed but the source file no longer exists. "
                "Re-run ingest_and_index() to refresh."
            )
        if response.status_code != 200:
            return f"ERROR: Server returned HTTP {response.status_code} for document {doc_id!r}."

        text = response.text

        # Truncate if too large for context window
        if len(text) > _MAX_FETCH_CHARS:
            logger.info(
                "Document %r truncated from %d to %d chars for context window.",
                doc_id,
                len(text),
                _MAX_FETCH_CHARS,
            )
            text = (
                text[:_MAX_FETCH_CHARS]
                + f"\n\n[... Document truncated at {_MAX_FETCH_CHARS} characters. "
                "Use a more specific search query to find the relevant section ...]"
            )

        return text

    return fetch_local_url
