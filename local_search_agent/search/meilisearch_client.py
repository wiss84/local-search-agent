"""
Meilisearch client wrapper for the Local Search Agent framework.

Uses meilisearch-python-sdk (async-capable, actively maintained).
This module exposes a synchronous interface for Phase 2 ingestion
and a clean search method for Phase 3 agent tools.

The client handles:
- Index creation and settings configuration on first use
- Batch document indexing with task polling
- Search with filter, snippet extraction, and top-k
- Index health checks

Install: pip install "meilisearch-python-sdk>=7.1.5"
"""

from __future__ import annotations

import logging
import time
from typing import Optional

try:
    from meilisearch_python_sdk.models.settings import MeilisearchSettings, Pagination
except ImportError:
    MeilisearchSettings = None  # type: ignore[assignment,misc]
    Pagination = None  # type: ignore[assignment]


from local_search_agent.core.constants import (
    DEFAULT_MEILI_MASTER_KEY,
    DEFAULT_MEILI_URL,
    DEFAULT_TOP_K,
    FIELD_DOC_ID,
    FILTERABLE_ATTRIBUTES,
    SEARCHABLE_ATTRIBUTES,
    SNIPPET_CONTEXT_CHARS,
)
from local_search_agent.core.document_node import DocumentNode

logger = logging.getLogger(__name__)

# Meilisearch task polling
_POLL_INTERVAL_S = 0.5
_POLL_TIMEOUT_S = 120


class MeilisearchClient:
    """
    Synchronous wrapper around the Meilisearch Python SDK.

    Initialises the target index with correct settings on first use,
    then exposes index_documents() and search() for the pipeline and agent.

    Parameters
    ----------
    url        : Meilisearch server URL (default: http://localhost:7700)
    api_key    : Meilisearch master key
    index_name : Name of the Meilisearch index to use
    """

    def __init__(
        self,
        url: str = DEFAULT_MEILI_URL,
        api_key: str = DEFAULT_MEILI_MASTER_KEY,
        index_name: str = "documents",
    ):
        self._url = url
        self._api_key = api_key
        self._index_name = index_name
        self._client = None      # Lazy init
        self._index = None       # Lazy init

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _get_client(self):
        """Lazily create and return the meilisearch Client instance."""
        if self._client is None:
            try:
                from meilisearch_python_sdk import Client
            except ImportError as e:
                raise ImportError(
                    "meilisearch-python-sdk is not installed. "
                    "Run: pip install 'meilisearch-python-sdk>=7.1.5'"
                ) from e
            self._client = Client(url=self._url, api_key=self._api_key)
        return self._client

    def _get_index(self):
        """
        Return the Meilisearch index, creating it with correct settings if needed.
        """
        if self._index is None:
            client = self._get_client()
            # Create index if it doesn't exist (primary key = doc_id)
            try:
                self._index = client.get_index(self._index_name)
            except Exception:
                logger.info("Creating Meilisearch index %r ...", self._index_name)
                # In SDK v7+, create_index() blocks until done and returns the Index directly.
                self._index = client.create_index(self._index_name, primary_key=FIELD_DOC_ID)
            self._configure_index_settings()
        return self._index

    def _configure_index_settings(self) -> None:
        """
        Apply searchable and filterable attribute settings to the index.
        Only needs to run once; subsequent calls are no-ops if settings match.
        """
        index = self._index
        try:
            client = self._get_client()
            task = index.update_searchable_attributes(SEARCHABLE_ATTRIBUTES)
            if task is not None:
                self._wait_for_task(client, self._get_task_uid(task))

            task = index.update_filterable_attributes(FILTERABLE_ATTRIBUTES)
            if task is not None:
                self._wait_for_task(client, self._get_task_uid(task))

            # Enable cropping for snippet extraction
            if MeilisearchSettings is not None and Pagination is not None:
                settings_body = MeilisearchSettings(pagination=Pagination(max_total_hits=1000))
            else:
                settings_body = {"pagination": {"maxTotalHits": 1000}}  # fallback for older SDK
            task = index.update_settings(settings_body)
            if task is not None:
                self._wait_for_task(client, self._get_task_uid(task))

            logger.info("Meilisearch index %r configured.", self._index_name)
        except Exception as e:
            logger.warning("Failed to configure index settings: %s", e)

    # ------------------------------------------------------------------
    # Task polling helper
    # ------------------------------------------------------------------

    def _wait_for_task(self, client, task_uid: int, timeout: float = _POLL_TIMEOUT_S) -> None:
        """
        Poll until a Meilisearch task completes or timeout expires.

        Raises RuntimeError if the task fails or times out.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            task = client.get_task(task_uid)
            # meilisearch-python-sdk returns a Task model object with .status and .uid
            # (not a dict, and not .task_uid — that's only on TaskInfo returned by mutations)
            if isinstance(task, dict):
                status = task.get("status", "")
                task_error = task.get("error", "unknown error")
            else:
                status = task.status
                task_error = getattr(task, "error", "unknown error")
            if status == "succeeded":
                return
            if status in ("failed", "canceled"):
                raise RuntimeError(
                    f"Meilisearch task {task_uid} {status}: "
                    f"{task_error}"
                )
            time.sleep(_POLL_INTERVAL_S)
        raise TimeoutError(f"Meilisearch task {task_uid} did not complete within {timeout}s")

    def _get_task_uid(self, task) -> int:
        """
        Extract the task uid from a TaskInfo object returned by any mutation
        (add_documents, update_settings, create_index, etc.).

        meilisearch-python-sdk returns a TaskInfo model with .task_uid.
        Falls back to dict .get('task_uid') for forward-compatibility.
        """
        if isinstance(task, dict):
            uid = task.get("task_uid") or task.get("uid")
        else:
            uid = getattr(task, "task_uid", None) or getattr(task, "uid", None)
        if uid is None:
            raise RuntimeError(f"Could not extract task uid from: {task!r}")
        return uid

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_documents(self, nodes: list[DocumentNode]) -> None:
        """
        Index a batch of DocumentNodes into Meilisearch.

        The `text` field is included in the indexed document so it can be
        returned in search results and used for snippet extraction.

        Parameters
        ----------
        nodes : List of DocumentNode objects to index.

        Raises
        ------
        RuntimeError if Meilisearch task fails.
        """
        if not nodes:
            return

        index = self._get_index()
        client = self._get_client()

        docs = [node.to_dict() for node in nodes]

        try:
            task = index.add_documents(docs, primary_key=FIELD_DOC_ID)
            self._wait_for_task(client, self._get_task_uid(task))
            logger.info("Indexed %d documents into %r", len(nodes), self._index_name)
        except Exception as e:
            raise RuntimeError(f"Failed to index documents: {e}") from e

    def delete_document(self, doc_id: str) -> None:
        """Remove a single document from the index by doc_id."""
        index = self._get_index()
        client = self._get_client()
        task = index.delete_document(doc_id)
        self._wait_for_task(client, self._get_task_uid(task))
        logger.debug("Deleted document %r from index.", doc_id)

    def delete_index(self) -> None:
        """Delete the entire index. Used when a workspace is removed."""
        try:
            index = self._get_index()
            client = self._get_client()
            # SDK v7+: delete() is on the Index object, not the Client.
            task = index.delete()
            self._wait_for_task(client, self._get_task_uid(task))
            self._index = None
            logger.info("Meilisearch index %r deleted.", self._index_name)
        except Exception as e:
            logger.warning("Could not delete index %r: %s", self._index_name, e)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        filter_expr: Optional[str] = None,
        snippet_chars: int = SNIPPET_CONTEXT_CHARS,
    ) -> list[dict]:
        """
        Search the index and return top-k results with snippets.

        Parameters
        ----------
        query        : User query string.
        top_k        : Maximum number of results to return.
        filter_expr  : Optional Meilisearch filter expression string,
                       e.g. 'file_type = "pdf" AND workspace = "finance"'.
                       Built by QueryBuilder.
        snippet_chars: Approximate length of the context snippet (chars).

        Returns
        -------
        List of dicts, each containing:
            doc_id, title, file_type, workspace, source_path,
            snippet (short context text), score (not available in Meilisearch CE,
            will be 0.0), modified_at
        """
        index = self._get_index()

        try:
            results = index.search(
                query,
                limit=top_k,
                attributes_to_crop=["text"],
                crop_length=snippet_chars // 5,
                attributes_to_retrieve=[
                    "doc_id", "title", "file_type", "workspace",
                    "source_path", "modified_at", "concepts",
                ],
                attributes_to_highlight=[],
                filter=filter_expr,
            )
        except Exception as e:
            logger.error("Meilisearch search failed: %s", e)
            return []

        hits = []
        for hit in results.hits:
            # Extract snippet from _formatted if available, fallback to empty
            snippet = ""
            formatted = hit.get("_formatted", {})
            if "text" in formatted:
                snippet = formatted["text"]
                # Strip Meilisearch highlight markers if present
                snippet = snippet.replace("<em>", "").replace("</em>", "")

            hits.append({
                "doc_id": hit.get("doc_id", ""),
                "title": hit.get("title", ""),
                "file_type": hit.get("file_type", ""),
                "workspace": hit.get("workspace", ""),
                "source_path": hit.get("source_path", ""),
                "modified_at": hit.get("modified_at", ""),
                "concepts": hit.get("concepts", []),
                "snippet": snippet,
                "score": 0.0,  # Meilisearch CE does not expose BM25 scores
            })

        return hits

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def is_healthy(self) -> bool:
        """Return True if the Meilisearch server is reachable and healthy."""
        try:
            client = self._get_client()
            health = client.health()
            return getattr(health, "status", None) == "available"
        except Exception:
            return False

    def get_index_stats(self) -> dict:
        """Return basic stats about the index (document count, field distribution)."""
        try:
            index = self._get_index()
            stats = index.get_stats()
            return {
                "number_of_documents": stats.number_of_documents,
                "is_indexing": stats.is_indexing,
                "field_distribution": stats.field_distribution,
            }
        except Exception as e:
            return {"error": str(e)}
