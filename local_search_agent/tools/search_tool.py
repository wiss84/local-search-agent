"""
LocalSearchTool — opaque search tool for integration with external AI agents.

Wraps a full SearchAgentFramework instance behind a single .run(query) call.
The caller gets a clean answer and a list of source titles; all internal
details (Meilisearch, agent loop, document fetching) are hidden.

Typical usage
-------------
```python
from local_search_agent import SearchAgentConfig
from local_search_agent.tools import LocalSearchTool

skill_tool = LocalSearchTool(SearchAgentConfig(
    document_dirs=["C:/skills"],
    workspace_name="skills",
    provider="google",
    model_name="gemini-3.1-flash-lite",
))

# Use directly
result = skill_tool.run("how do I handle rate limits in Python?")
print(result.answer)
print(result.sources)   # ["rate_limit_handler", "retry_patterns"]

# Use inside a LangChain agent
from langchain.tools import tool

@tool
def skill_search(query: str) -> str:
    \"\"\"Search the skills knowledge base for coding patterns and techniques.\"\"\"
    return skill_tool.run(query).answer
```
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

from local_search_agent.core.config import SearchAgentConfig

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """
    The return value of LocalSearchTool.run().

    Attributes
    ----------
    answer  : Clean prose answer synthesised by the internal agent.
    sources : Titles of the documents the answer was drawn from
              (e.g. ["handover_notes", "project_plan"]).
              Empty list if no sources could be identified.
    """

    answer: str
    sources: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        """Return just the answer string so the result works as a plain string."""
        return self.answer


class LocalSearchTool:
    """
    A self-contained, opaque search tool that wraps SearchAgentFramework.

    Each instance manages its own indexed workspace. Create one instance
    per directory / knowledge domain you want to expose as a tool.

    Parameters
    ----------
    config : SearchAgentConfig pointing at the directory to index and the
             LLM to use for retrieval synthesis.  Use a cheap/fast model
             here (e.g. ``gemini-3.1-flash-lite``) and reserve your
             application's primary model for higher-level reasoning.

    Notes
    -----
    - The framework and index are initialised lazily on the first call to
      :meth:`run` so that creating the tool object is cheap.
    - If the workspace has never been ingested, :meth:`run` will ingest it
      automatically before answering.
    - Thread-safe: a single instance can be shared across threads.
    """

    def __init__(self, config: SearchAgentConfig, return_raw: bool = False) -> None:
        self._config = config
        self._return_raw = return_raw
        self._framework = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_framework(self):
        """Lazily initialise the framework (not thread-safe — call inside lock)."""
        if self._framework is None:
            from local_search_agent.core.framework import SearchAgentFramework

            self._framework = SearchAgentFramework(self._config)
        return self._framework

    def _ensure_ready(self) -> None:
        """Ensure the framework is initialised. Ingestion is the user's responsibility."""
        self._get_framework()

    def _resolve_titles(self, raw_sources) -> list[str]:
        """
        Convert sources into a deduplicated list of human-readable document titles.
        Accepts either:
        - a list of dicts with a 'doc_id' key (from framework.query())
        - a set/list of doc_id strings (from state['sources_seen'])
        """
        wm = self._get_framework()._workspace_manager
        titles: list[str] = []
        seen: set[str] = set()

        for source in raw_sources:
            doc_id = source if isinstance(source, str) else source.get("doc_id", "")
            if not doc_id:
                continue
            node = wm.get_document(doc_id)
            if node and node.title not in seen:
                titles.append(node.title)
                seen.add(node.title)

        return titles

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _extract_fetch_content(self, state: dict) -> str | None:
        """
        Walk the state messages in reverse and return the content of the last
        ToolMessage produced by fetch_local_url, bypassing the LLM response.
        Returns None if no fetch tool result is found.
        """
        from langchain_core.messages import AIMessage, ToolMessage

        messages = state.get("messages", [])
        # Build a map of tool_call_id -> tool_name from AIMessages
        call_id_to_tool: dict[str, str] = {}
        for msg in messages:
            if isinstance(msg, AIMessage):
                for tc in msg.tool_calls or []:
                    call_id_to_tool[tc["id"]] = tc["name"]

        # Walk in reverse to find the last fetch_local_url result
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                tool_name = call_id_to_tool.get(msg.tool_call_id, "")
                if tool_name == "fetch_local_url":
                    return str(msg.content)
        return None

    def run(self, query: str) -> ToolResult:
        """
        Search the indexed workspace and return a synthesised answer.

        Starts the file server before querying and stops it when done.

        Parameters
        ----------
        query : Natural language question or search query.

        Returns
        -------
        ToolResult with ``answer`` (str) and ``sources`` (list[str]).
        """
        self._ensure_ready()
        framework = self._get_framework()

        with self._lock:
            if self._return_raw:
                state = framework.query_raw_state(query)
            else:
                raw = framework.query(query)

        if self._return_raw:
            answer = self._extract_fetch_content(state)
            if not answer:
                from local_search_agent.agent.agent import LocalSearchAgent

                agent = LocalSearchAgent(
                    config=self._config,
                    meili_client=framework._get_meili_client(),
                )
                built = agent._build_response(state, query)
                answer = built.get("answer", "I could not find relevant information.")
            sources = self._resolve_titles(state.get("sources_seen", set()) or [])
        else:
            answer = raw.get("answer", "")
            sources = self._resolve_titles(raw.get("sources", []))

        logger.debug(
            "LocalSearchTool.run(%r): answer_len=%d, sources=%s",
            query,
            len(answer),
            sources,
        )

        return ToolResult(answer=answer, sources=sources)
