"""
Unit tests for the Phase 3 agent layer.

Tests cover:
- Tool output formatting (search_tool, fetch_tool)
- Agent response extraction
- Max-iterations guard
- Source extraction from agent messages
- Provider factory (import error handling)
- QueryBuilder (already in test_search.py — not repeated here)

All LLM calls and HTTP calls are mocked. No live services needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from local_search_agent.agent.agent import AgentState, LocalSearchAgent
from local_search_agent.agent.prompts import MAX_ITERATIONS_NOTICE, build_system_prompt
from local_search_agent.core.config import SearchAgentConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config(tmp_path):
    return SearchAgentConfig(
        document_dirs=[str(tmp_path)],
        workspace_name="test_ws",
        provider="ollama",
        model_name="mistral",
        db_path=str(tmp_path / "test.db"),
        max_iterations=5,
        top_k=3,
    )


@pytest.fixture
def mock_meili():
    m = MagicMock()
    m.search.return_value = [
        {
            "doc_id": "abc123def456abcd",
            "title": "Finance Report Q3",
            "file_type": "pdf",
            "workspace": "test_ws",
            "source_path": "/shares/finance/q3.pdf",
            "modified_at": "2024-09-30T10:00:00+02:00",
            "concepts": ["finance", "AWS"],
            "snippet": "AWS spend on Project Alpha reached $1.2M in Q3 2024.",
        }
    ]
    return m


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    def test_contains_workspace_name(self):
        prompt = build_system_prompt("finance", ["/shares/finance"])
        assert "finance" in prompt

    def test_contains_document_dirs(self):
        prompt = build_system_prompt("hr", ["/shares/hr", "/shares/policies"])
        assert "/shares/hr" in prompt
        assert "/shares/policies" in prompt

    def test_contains_workflow_instructions(self):
        prompt = build_system_prompt("it", [])
        assert "never answer from memory" in prompt.lower() or "search" in prompt.lower()

    def test_contains_citation_instructions(self):
        prompt = build_system_prompt("legal", [])
        assert "cite" in prompt.lower() or "source" in prompt.lower()

    def test_no_document_dirs_handled_gracefully(self):
        prompt = build_system_prompt("ws", [])
        assert "unspecified" in prompt


# ---------------------------------------------------------------------------
# Search tool output
# ---------------------------------------------------------------------------


class TestSearchTool:
    def test_formats_results_correctly(self, config, mock_meili):
        from local_search_agent.agent.tools.search_tool import build_search_tool

        tool = build_search_tool(mock_meili, config)
        result = tool.invoke({"query": "AWS spend Q3 2024"})

        assert "Finance Report Q3" in result
        assert "abc123def456abcd" in result
        assert "$1.2M" in result
        assert "text_url" in result
        assert "docs_url" in result

    def test_no_results_returns_helpful_message(self, config, mock_meili):
        from local_search_agent.agent.tools.search_tool import build_search_tool

        mock_meili.search.return_value = []
        tool = build_search_tool(mock_meili, config)
        result = tool.invoke({"query": "something obscure"})

        assert "No documents found" in result
        assert "something obscure" in result

    def test_search_failure_returns_error_string(self, config, mock_meili):
        from local_search_agent.agent.tools.search_tool import build_search_tool

        mock_meili.search.side_effect = RuntimeError("Meilisearch down")
        tool = build_search_tool(mock_meili, config)
        result = tool.invoke({"query": "test"})

        assert "Search failed" in result
        assert "Meilisearch down" in result

    def test_top_k_capped_at_20(self, config, mock_meili):
        from local_search_agent.agent.tools.search_tool import build_search_tool

        tool = build_search_tool(mock_meili, config)
        tool.invoke({"query": "test", "top_k": 999})

        call_kwargs = mock_meili.search.call_args
        assert call_kwargs.kwargs["top_k"] == 20

    def test_file_type_filter_applied(self, config, mock_meili):
        from local_search_agent.agent.tools.search_tool import build_search_tool

        tool = build_search_tool(mock_meili, config)
        tool.invoke({"query": "policy", "file_type": "pdf"})

        filter_expr = mock_meili.search.call_args.kwargs.get("filter_expr", "")
        assert "pdf" in (filter_expr or "")


# ---------------------------------------------------------------------------
# Fetch tool output
# ---------------------------------------------------------------------------


class TestFetchTool:
    def test_returns_document_text_on_success(self, config):
        from local_search_agent.agent.tools.fetch_tool import build_fetch_tool

        tool = build_fetch_tool(config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "# Finance Report\n\nAWS spend was $1.2M."

        with patch("httpx.get", return_value=mock_response):
            result = tool.invoke({"doc_id": "abc123def456abcd"})

        assert "Finance Report" in result
        assert "$1.2M" in result

    def test_returns_error_on_404(self, config):
        from local_search_agent.agent.tools.fetch_tool import build_fetch_tool

        tool = build_fetch_tool(config)

        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("httpx.get", return_value=mock_response):
            result = tool.invoke({"doc_id": "deadbeef00000000"})

        assert "ERROR" in result
        assert "not found" in result.lower()

    def test_returns_error_on_410(self, config):
        from local_search_agent.agent.tools.fetch_tool import build_fetch_tool

        tool = build_fetch_tool(config)

        mock_response = MagicMock()
        mock_response.status_code = 410

        with patch("httpx.get", return_value=mock_response):
            result = tool.invoke({"doc_id": "deadbeef00000000"})

        assert "ERROR" in result
        assert "no longer exists" in result.lower()

    def test_returns_error_on_connection_failure(self, config):
        import httpx

        from local_search_agent.agent.tools.fetch_tool import build_fetch_tool

        tool = build_fetch_tool(config)

        with patch("httpx.get", side_effect=httpx.ConnectError("refused")):
            result = tool.invoke({"doc_id": "abc123def456abcd"})

        assert "ERROR" in result
        assert "connect" in result.lower()

    def test_truncates_large_documents(self, config):
        from local_search_agent.agent.tools.fetch_tool import _MAX_FETCH_CHARS, build_fetch_tool

        tool = build_fetch_tool(config)

        large_text = "A" * (_MAX_FETCH_CHARS + 5000)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = large_text

        with patch("httpx.get", return_value=mock_response):
            result = tool.invoke({"doc_id": "abc123def456abcd"})

        assert len(result) < len(large_text)
        assert "truncated" in result.lower()


# ---------------------------------------------------------------------------
# Agent response extraction
# ---------------------------------------------------------------------------


class TestAgentResponseExtraction:
    def _make_agent(self, config, mock_meili):
        agent = LocalSearchAgent(config=config, meili_client=mock_meili)
        return agent

    def test_extracts_last_ai_message_as_answer(self, config, mock_meili):
        agent = self._make_agent(config, mock_meili)
        state: AgentState = {
            "messages": [
                HumanMessage(content="What was AWS spend?"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "search_local_index", "args": {"query": "AWS spend"}, "id": "tc1"}
                    ],
                ),
                ToolMessage(content="Found 1 result...", tool_call_id="tc1"),
                AIMessage(
                    content="AWS spend on Project Alpha was $1.2M in Q3 2024. "
                    "Source: http://localhost:8000/docs/abc123def456abcd"
                ),
            ],
            "iterations": 1,
            "sources_seen": {"abc123def456abcd"},
            "truncated": False,
        }
        response = agent._build_response(state, "What was AWS spend?")

        assert "$1.2M" in response["answer"]
        assert response["iterations_used"] == 1
        assert response["truncated"] is False

    def test_empty_messages_returns_fallback(self, config, mock_meili):
        agent = self._make_agent(config, mock_meili)
        state: AgentState = {
            "messages": [HumanMessage(content="test")],
            "iterations": 0,
            "sources_seen": set(),
            "truncated": False,
        }
        response = agent._build_response(state, "test")
        assert "could not find" in response["answer"].lower()

    def test_truncated_flag_set_when_iterations_maxed(self, config, mock_meili):
        agent = self._make_agent(config, mock_meili)
        state: AgentState = {
            "messages": [
                HumanMessage(content="test"),
                AIMessage(content="Partial answer based on limited search."),
            ],
            "iterations": config.max_iterations,  # at the limit
            "sources_seen": set(),
            "truncated": False,
        }
        response = agent._build_response(state, "test")
        assert response["truncated"] is True
        assert MAX_ITERATIONS_NOTICE.split("\n")[1][:20] in response["answer"]

    def test_source_extraction_from_inline_citations(self, config, mock_meili):
        agent = self._make_agent(config, mock_meili)
        state: AgentState = {
            "messages": [
                HumanMessage(content="test"),
                AIMessage(
                    content=(
                        "The AWS spend was $1.2M. "
                        "Source: http://localhost:8000/docs/abc123def456abcd\n"
                        "See also: http://localhost:8000/docs/ffff000011112222"
                    )
                ),
            ],
            "iterations": 1,
            "sources_seen": set(),
            "truncated": False,
        }
        response = agent._build_response(state, "test")
        doc_ids = {s["doc_id"] for s in response["sources"]}
        assert "abc123def456abcd" in doc_ids
        assert "ffff000011112222" in doc_ids

    def test_sources_seen_included_in_output(self, config, mock_meili):
        agent = self._make_agent(config, mock_meili)
        state: AgentState = {
            "messages": [
                HumanMessage(content="test"),
                AIMessage(content="The spend was $1.2M."),
            ],
            "iterations": 1,
            "sources_seen": {"abc123def456abcd"},
            "truncated": False,
        }
        response = agent._build_response(state, "test")
        doc_ids = {s["doc_id"] for s in response["sources"]}
        assert "abc123def456abcd" in doc_ids


# ---------------------------------------------------------------------------
# Provider factory — import error handling
# ---------------------------------------------------------------------------


class TestProviderFactory:
    def test_unknown_provider_raises_value_error(self, config):
        from local_search_agent.agent.provider_factory import build_llm

        config.provider = "unknown_provider"
        with pytest.raises(ValueError, match="Unknown provider"):
            build_llm(config)

    def test_google_missing_api_key_raises(self, config):
        from local_search_agent.agent.provider_factory import _build_google

        config.api_key = None
        with pytest.raises(ValueError, match="api_key is required"):
            _build_google(config)

    def test_openai_missing_api_key_raises(self, config):
        from local_search_agent.agent.provider_factory import _build_openai

        config.provider = "openai"
        config.api_key = None
        with pytest.raises(ValueError, match="api_key is required"):
            _build_openai(config)
