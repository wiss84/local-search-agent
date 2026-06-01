"""
LangGraph agent loop for the Local Search Agent framework.

Architecture: ReAct-style StateGraph
  START → call_llm → route → call_tools → call_llm → ... → END
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from local_search_agent.agent.prompts import MAX_ITERATIONS_NOTICE, build_system_prompt
from local_search_agent.agent.provider_factory import build_llm
from local_search_agent.agent.rate_limit_handler import RateLimitHandler
from local_search_agent.agent.tools.fetch_tool import build_fetch_tool
from local_search_agent.agent.tools.graph_tool import build_graph_tool
from local_search_agent.agent.tools.search_tool import build_search_tool
from local_search_agent.core.config import SearchAgentConfig
from local_search_agent.core.constants import LANGGRAPH_RECURSION_LIMIT

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    iterations: int
    sources_seen: set[str]
    truncated: bool


class LocalSearchAgent:
    """
    LangGraph-powered research agent that answers questions from indexed documents.

    Parameters
    ----------
    config           : SearchAgentConfig
    meili_client     : MeilisearchClient instance
    workspace_manager: Optional WorkspaceManager — enables title lookups in get_related_docs
    """

    def __init__(
        self,
        config: SearchAgentConfig,
        meili_client,
        workspace_manager=None,
    ):
        self._config = config
        self._meili = meili_client
        self._workspace_manager = workspace_manager
        self._lock = threading.RLock()  # RLock: _get_graph calls _get_tools from same thread
        self._graph = None
        self._tools = None
        self._tool_map = None

    def _get_tools(self) -> list:
        with self._lock:
            if self._tools is None:
                self._tools = [
                    build_search_tool(self._meili, self._config),
                    build_fetch_tool(self._config),
                    build_graph_tool(self._config, self._workspace_manager),
                ]
                self._tool_map = {t.name: t for t in self._tools}
        return self._tools

    def _get_graph(self):
        with self._lock:
            if self._graph is None:
                self._graph = self._build_graph_unlocked()
        return self._graph

    def _build_graph_unlocked(self):
        """Build the LangGraph graph. Must be called with self._lock held."""
        tools = self._get_tools()  # safe: RLock allows re-entry from same thread
        llm = build_llm(self._config)
        llm_with_tools = llm.bind_tools(tools)
        config = self._config

        rate_limiter = RateLimitHandler(
            provider=config.provider,
            model_name=config.model_name,
            max_retries=config.max_retries,
        )

        def call_llm(state: AgentState) -> dict:
            logger.debug(
                "call_llm: %d messages, iteration %d", len(state["messages"]), state["iterations"]
            )
            response = rate_limiter.call_with_retry(
                llm_with_tools.invoke,
                state["messages"],
            )
            return {"messages": [response]}

        def call_tools(state: AgentState) -> dict:
            last_message = state["messages"][-1]
            if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
                return {"iterations": state["iterations"]}

            tool_messages: list[ToolMessage] = []
            seen = set(state.get("sources_seen", set()))

            for tc in last_message.tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]
                tool_id = tc["id"]

                tool_fn = self._tool_map.get(tool_name)
                if tool_fn is None:
                    result = f"ERROR: Unknown tool {tool_name!r}."
                else:
                    try:
                        result = tool_fn.invoke(tool_args)
                    except Exception as e:
                        logger.error("Tool %r raised: %s", tool_name, e)
                        result = f"ERROR: Tool {tool_name!r} failed: {e}"

                if tool_name == "fetch_local_url":
                    doc_id = tool_args.get("doc_id", "")
                    if doc_id:
                        seen.add(doc_id)

                tool_messages.append(ToolMessage(content=str(result), tool_call_id=tool_id))

            return {
                "messages": tool_messages,
                "iterations": state["iterations"] + 1,
                "sources_seen": seen,
            }

        def route(state: AgentState) -> str:
            if state["iterations"] >= config.max_iterations:
                logger.warning("Max iterations (%d) reached.", config.max_iterations)
                return END
            last = state["messages"][-1]
            if isinstance(last, AIMessage) and last.tool_calls:
                return "call_tools"
            return END

        graph = StateGraph(AgentState)
        graph.add_node("call_llm", call_llm)
        graph.add_node("call_tools", call_tools)
        graph.add_edge(START, "call_llm")
        graph.add_conditional_edges("call_llm", route, {"call_tools": "call_tools", END: END})
        graph.add_edge("call_tools", "call_llm")
        return graph.compile()

    def query_raw_state(
        self,
        question: str,
        workspace: Optional[str] = None,
    ) -> dict:
        """Run the agent and return the raw LangGraph state."""
        effective_workspace = workspace or self._config.workspace_name
        system_msg = SystemMessage(
            content=build_system_prompt(
                workspace_name=effective_workspace,
                document_dirs=self._config.document_dirs,
            )
        )
        initial_state: AgentState = {
            "messages": [system_msg, HumanMessage(content=question)],
            "iterations": 0,
            "sources_seen": set(),
            "truncated": False,
        }
        graph = self._get_graph()
        return graph.invoke(
            initial_state,
            config={"recursion_limit": LANGGRAPH_RECURSION_LIMIT},
        )

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        workspace: Optional[str] = None,
    ) -> dict:
        """Ask the agent a question and return a structured response."""
        effective_workspace = workspace or self._config.workspace_name

        system_msg = SystemMessage(
            content=build_system_prompt(
                workspace_name=effective_workspace,
                document_dirs=self._config.document_dirs,
            )
        )

        initial_state: AgentState = {
            "messages": [system_msg, HumanMessage(content=question)],
            "iterations": 0,
            "sources_seen": set(),
            "truncated": False,
        }

        logger.info("Agent query: %r (workspace=%r)", question, effective_workspace)
        graph = self._get_graph()
        final_state = graph.invoke(
            initial_state,
            config={"recursion_limit": LANGGRAPH_RECURSION_LIMIT},
        )
        return self._build_response(final_state, question)

    def stream(self, question: str, workspace: Optional[str] = None):
        """
        Stream the agent execution step by step using graph.stream().

        Uses stream_mode="updates" so each yielded step is
        {node_name: state_update} — only the delta, not the full state.
        The last step that contains a call_llm update is used to build
        the final response via _build_response().

        Yields dicts with one of these shapes:
            {"type": "thinking",  "text": str}
            {"type": "tool_start", "tool": str, "input": dict, "call_id": str}
            {"type": "tool_end",   "tool": str, "output": str,
                                   "duration_ms": int, "call_id": str}
            {"type": "text",       "text": str}
            {"type": "done",       "token_input": int, "token_output": int,
                                   "iterations_used": int, "truncated": bool,
                                   "sources": list}
        """
        import time as _time
        import uuid as _uuid

        effective_workspace = workspace or self._config.workspace_name
        system_msg = SystemMessage(
            content=build_system_prompt(
                workspace_name=effective_workspace,
                document_dirs=self._config.document_dirs,
            )
        )
        initial_state: AgentState = {
            "messages": [system_msg, HumanMessage(content=question)],
            "iterations": 0,
            "sources_seen": set(),
            "truncated": False,
        }

        logger.info("Agent stream: %r (workspace=%r)", question, effective_workspace)
        graph = self._get_graph()

        # Per-call_id timing so tool_end can report duration_ms
        _tool_start_times: dict[str, float] = {}
        # Map call_id -> tool_name (stored at tool_start, used at tool_end)
        _tool_names: dict[str, str] = {}

        token_input = 0
        token_output = 0

        # We accumulate all messages ourselves so _build_response works at the end
        # without a second graph.invoke() call.
        accumulated_messages: list[BaseMessage] = list(initial_state["messages"])
        accumulated_sources_seen: set[str] = set()
        accumulated_iterations: int = 0

        for step in graph.stream(
            initial_state,
            config={"recursion_limit": LANGGRAPH_RECURSION_LIMIT},
            stream_mode="updates",
        ):
            # step is {node_name: state_update_dict}
            for node_name, update in step.items():
                new_messages = update.get("messages", [])
                accumulated_messages.extend(new_messages)
                if update.get("sources_seen"):
                    accumulated_sources_seen.update(update["sources_seen"])
                if update.get("iterations"):
                    accumulated_iterations = update["iterations"]

                if node_name == "call_llm":
                    for msg in new_messages:
                        if not isinstance(msg, AIMessage):
                            continue

                        # Accumulate token usage
                        usage = getattr(msg, "usage_metadata", None)
                        if usage and isinstance(usage, dict):
                            token_input += usage.get("input_tokens", 0) or 0
                            token_output += usage.get("output_tokens", 0) or 0
                            yield {
                                "type": "token_update",
                                "token_input": token_input,
                                "token_output": token_output,
                            }

                        # Extract thinking and text from content blocks
                        thinking_text = ""
                        content = msg.content
                        if isinstance(content, list):
                            for block in content:
                                if isinstance(block, dict) and block.get("type") == "thinking":
                                    thinking_text += block.get("thinking", "")
                        if thinking_text:
                            yield {"type": "thinking", "text": thinking_text}

                        # Emit tool_start for every tool call in this LLM step
                        for tc in msg.tool_calls or []:
                            call_id = tc.get("id") or _uuid.uuid4().hex[:8]
                            _tool_start_times[call_id] = _time.monotonic()
                            _tool_names[call_id] = tc["name"]
                            yield {
                                "type": "tool_start",
                                "tool": tc["name"],
                                "input": tc["args"],
                                "call_id": call_id,
                            }

                elif node_name == "call_tools":
                    for msg in new_messages:
                        if not isinstance(msg, ToolMessage):
                            continue
                        call_id = msg.tool_call_id or ""
                        t0 = _tool_start_times.pop(call_id, None)
                        duration_ms = int((_time.monotonic() - t0) * 1000) if t0 else 0
                        tool_name = _tool_names.pop(call_id, "unknown")
                        yield {
                            "type": "tool_end",
                            "tool": tool_name,
                            "output": str(msg.content),
                            "duration_ms": duration_ms,
                            "call_id": call_id,
                        }

        # Build the final response from the accumulated state — no second invoke needed
        final_state: AgentState = {
            "messages": accumulated_messages,
            "iterations": accumulated_iterations,
            "sources_seen": accumulated_sources_seen,
            "truncated": accumulated_iterations >= self._config.max_iterations,
        }
        response = self._build_response(final_state, question)

        # Fall back to char-based estimate when provider didn't return token counts
        if token_input == 0 and token_output == 0:
            token_input = response.get("token_input", 0)
            token_output = response.get("token_output", 0)

        yield {"type": "text", "text": response["answer"]}
        yield {
            "type": "done",
            "token_input": token_input,
            "token_output": token_output,
            "iterations_used": response.get("iterations_used", 0),
            "truncated": response.get("truncated", False),
            "sources": response.get("sources", []),
        }

    def _build_response(self, state: AgentState, question: str) -> dict:
        iterations_used = state.get("iterations", 0)
        truncated = iterations_used >= self._config.max_iterations

        answer = ""
        thinking = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                answer, thinking = self._extract_response_and_thinking(msg)
                break

        if not answer:
            answer = "I could not find relevant information in the indexed documents."

        if truncated:
            answer = (
                MAX_ITERATIONS_NOTICE.format(max_iterations=self._config.max_iterations)
                + "\n\n"
                + answer
            )

        sources = self._extract_sources(state)

        # Accumulate token usage from all AIMessage usage_metadata across the run
        token_input = 0
        token_output = 0
        for msg in state["messages"]:
            if isinstance(msg, AIMessage):
                usage = getattr(msg, "usage_metadata", None)
                if usage and isinstance(usage, dict):
                    token_input += usage.get("input_tokens", 0) or 0
                    token_output += usage.get("output_tokens", 0) or 0
        # Fallback: rough character-based estimate when provider doesn't return usage
        if token_input == 0 and token_output == 0:
            total_input_chars = sum(
                len(getattr(m, "content", "") or "")
                for m in state["messages"]
                if not isinstance(m, AIMessage)
            )
            token_input = max(1, total_input_chars // 4)
            token_output = max(1, len(answer) // 4)

        logger.info(
            "Query complete: iterations=%d, truncated=%s, sources=%d, tokens=%d+%d",
            iterations_used,
            truncated,
            len(sources),
            token_input,
            token_output,
        )

        return {
            "answer": answer,
            "thinking": thinking,
            "sources": sources,
            "iterations_used": iterations_used,
            "truncated": truncated,
            "token_input": token_input,
            "token_output": token_output,
        }

    # ------------------------------------------------------------------
    # Response content helpers
    # ------------------------------------------------------------------

    def _extract_response_content(self, response) -> str:
        """Extract the text response only, discarding thinking blocks.
        Use _extract_response_and_thinking() when thinking content is needed."""
        text, _ = self._extract_response_and_thinking(response)
        return text

    def _extract_response_and_thinking(self, response) -> tuple[str, str]:
        """
        Extract both the text response and thinking content from an LLM response.

        Returns
        -------
        (response_text, thinking_text)
            thinking_text is an empty string if no thinking blocks were present.
        """
        content = response.content
        if isinstance(content, str):
            return content, ""
        if isinstance(content, list):
            text_parts = []
            thinking_parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "thinking":
                        thinking_parts.append(block.get("thinking", ""))
                    else:
                        text_parts.append(block.get("text", ""))
            return (
                "".join(text_parts) if text_parts else str(content),
                "\n\n".join(t for t in thinking_parts if t),
            )
        return str(content), ""

    def _extract_sources(self, state: AgentState) -> list[dict]:
        doc_ids: set[str] = set(state.get("sources_seen", set()))
        pattern = re.compile(r"/docs/([a-f0-9]{16})")
        for msg in state["messages"]:
            if isinstance(msg, AIMessage) and msg.content:
                text_content, _ = self._extract_response_and_thinking(msg)
                for match in pattern.finditer(text_content):
                    doc_ids.add(match.group(1))
        return [
            {
                "doc_id": doc_id,
                "text_url": self._config.text_url(doc_id),
                "docs_url": self._config.docs_url(doc_id),
            }
            for doc_id in sorted(doc_ids)
        ]
