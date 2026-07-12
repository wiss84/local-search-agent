from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, ToolMessage

from local_search_agent.core.config import SearchAgentConfig
from local_search_agent.tools.search_tool import LocalSearchTool, ToolResult

# ---------------------------------------------------------------------------
# ToolResult tests
# ---------------------------------------------------------------------------


class TestToolResult:
    def test_str_returns_answer(self):
        result = ToolResult(answer="The answer is 42.")
        assert str(result) == "The answer is 42."

    def test_default_sources_empty(self):
        result = ToolResult(answer="Some answer")
        assert result.sources == []

    def test_with_sources(self):
        result = ToolResult(answer="Some answer", sources=["doc_a", "doc_b"])
        assert result.sources == ["doc_a", "doc_b"]


# ---------------------------------------------------------------------------
# LocalSearchTool tests
# ---------------------------------------------------------------------------


class TestLocalSearchTool:
    def _make_tool(self, return_raw=False):
        config = SearchAgentConfig(
            document_dirs=["/tmp/test_docs"],
            workspace_name="test_ws",
        )
        return LocalSearchTool(config, return_raw=return_raw)

    def test_init_stores_config_and_framework_none(self):
        tool = self._make_tool()
        assert tool._config.workspace_name == "test_ws"
        assert tool._framework is None
        assert tool._return_raw is False

    def test_get_framework_lazy_init(self):
        tool = self._make_tool()
        mock_framework = MagicMock()
        with patch(
            "local_search_agent.core.framework.SearchAgentFramework",
            return_value=mock_framework,
        ) as mock_ctor:
            framework = tool._get_framework()
            mock_ctor.assert_called_once_with(tool._config)
            assert framework is mock_framework

    def test_get_framework_returns_cached(self):
        tool = self._make_tool()
        mock_framework = MagicMock()
        with patch(
            "local_search_agent.core.framework.SearchAgentFramework",
            return_value=mock_framework,
        ):
            first = tool._get_framework()
            second = tool._get_framework()
            assert first is second
            assert tool._framework is mock_framework

    def test_ensure_ready_initializes_framework(self):
        tool = self._make_tool()
        mock_framework = MagicMock()
        with patch(
            "local_search_agent.core.framework.SearchAgentFramework",
            return_value=mock_framework,
        ) as mock_ctor:
            tool._ensure_ready()
            mock_ctor.assert_called_once_with(tool._config)

    def test_resolve_titles_with_dict_sources(self):
        tool = self._make_tool()
        mock_framework = MagicMock()
        mock_wm = MagicMock()
        node_a = MagicMock()
        node_a.title = "Alpha Doc"
        node_b = MagicMock()
        node_b.title = "Beta Doc"
        mock_wm.get_document.side_effect = lambda doc_id: {
            "d1": node_a,
            "d2": node_b,
        }.get(doc_id)
        mock_framework._workspace_manager = mock_wm
        tool._framework = mock_framework

        raw_sources = [{"doc_id": "d1"}, {"doc_id": "d2"}]
        titles = tool._resolve_titles(raw_sources)
        assert titles == ["Alpha Doc", "Beta Doc"]

    def test_resolve_titles_with_string_sources(self):
        tool = self._make_tool()
        mock_framework = MagicMock()
        node = MagicMock()
        node.title = "String Source Doc"
        mock_wm = MagicMock()
        mock_wm.get_document.return_value = node
        mock_framework._workspace_manager = mock_wm
        tool._framework = mock_framework

        raw_sources = {"d1", "d2"}
        titles = tool._resolve_titles(raw_sources)
        assert titles == ["String Source Doc"]

    def test_resolve_titles_skips_missing_docs(self):
        tool = self._make_tool()
        mock_framework = MagicMock()
        mock_wm = MagicMock()
        mock_wm.get_document.return_value = None
        mock_framework._workspace_manager = mock_wm
        tool._framework = mock_framework

        raw_sources = [{"doc_id": "missing"}]
        titles = tool._resolve_titles(raw_sources)
        assert titles == []

    def test_resolve_titles_deduplicates_by_title(self):
        tool = self._make_tool()
        mock_framework = MagicMock()
        node = MagicMock()
        node.title = "Same Title"
        mock_wm = MagicMock()
        mock_wm.get_document.return_value = node
        mock_framework._workspace_manager = mock_wm
        tool._framework = mock_framework

        raw_sources = [{"doc_id": "d1"}, {"doc_id": "d2"}]
        titles = tool._resolve_titles(raw_sources)
        assert titles == ["Same Title"]

    def test_extract_fetch_content_returns_last_tool_message(self):
        tool = self._make_tool(return_raw=True)
        fetch_tool_call = {
            "id": "call_123",
            "name": "fetch_local_url",
            "args": {"url": "file:///tmp/doc.txt"},
        }
        state = {
            "messages": [
                AIMessage(content="Let me fetch that.", tool_calls=[fetch_tool_call]),
                ToolMessage(content="fetched text", tool_call_id="call_123"),
            ]
        }
        result = tool._extract_fetch_content(state)
        assert result == "fetched text"

    def test_extract_fetch_content_returns_none_when_no_fetch(self):
        tool = self._make_tool(return_raw=True)
        other_tool_call = {
            "id": "call_1",
            "name": "other_tool",
            "args": {},
        }
        state = {
            "messages": [
                AIMessage(content="Something.", tool_calls=[other_tool_call]),
                ToolMessage(content="result", tool_call_id="call_1"),
            ]
        }
        result = tool._extract_fetch_content(state)
        assert result is None

    def test_extract_fetch_content_returns_none_when_no_tool_messages(self):
        tool = self._make_tool(return_raw=True)
        state = {
            "messages": [
                AIMessage(content="No tool calls here."),
            ]
        }
        result = tool._extract_fetch_content(state)
        assert result is None

    def test_run_without_raw_calls_query(self):
        tool = self._make_tool(return_raw=False)
        mock_framework = MagicMock()
        mock_framework.query.return_value = {
            "answer": "The capital of France is Paris.",
            "sources": [{"doc_id": "d1"}, {"doc_id": "d2"}],
        }
        tool._framework = mock_framework
        with patch.object(tool, "_resolve_titles", return_value=["France Doc"]) as mock_resolve:
            result = tool.run("What is the capital of France?")
            mock_framework.query.assert_called_once_with("What is the capital of France?")
            mock_resolve.assert_called_once()
            assert result.answer == "The capital of France is Paris."
            assert result.sources == ["France Doc"]

    def test_run_with_raw_calls_query_raw_state(self):
        tool = self._make_tool(return_raw=True)
        mock_framework = MagicMock()
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_fetch",
                            "name": "fetch_local_url",
                            "args": {"url": "file:///tmp/doc.txt"},
                        }
                    ],
                ),
                ToolMessage(content="fetched content", tool_call_id="call_fetch"),
            ],
            "sources_seen": {"d1"},
        }
        mock_framework.query_raw_state.return_value = state
        tool._framework = mock_framework
        with patch.object(tool, "_resolve_titles", return_value=["Fetched Doc"]):
            result = tool.run("Find info in doc.txt")
            mock_framework.query_raw_state.assert_called_once_with("Find info in doc.txt")
            assert result.answer == "fetched content"
            assert result.sources == ["Fetched Doc"]

    def test_run_raw_falls_back_to_build_response_when_no_fetch_content(self):
        tool = self._make_tool(return_raw=True)
        mock_framework = MagicMock()
        state = {
            "messages": [
                AIMessage(content="Let me think.", tool_calls=[]),
            ],
            "sources_seen": set(),
        }
        mock_framework.query_raw_state.return_value = state
        tool._framework = mock_framework

        mock_agent = MagicMock()
        mock_agent._build_response.return_value = {
            "answer": "Fallback answer from agent.",
            "sources": [],
        }
        with (
            patch(
                "local_search_agent.agent.agent.LocalSearchAgent",
                return_value=mock_agent,
            ),
            patch.object(tool, "_resolve_titles", return_value=[]),
        ):
            result = tool.run("Query with no fetch")
            mock_agent._build_response.assert_called_once_with(state, "Query with no fetch")
            assert result.answer == "Fallback answer from agent."

    def test_run_resolves_titles_from_sources_seen(self):
        tool = self._make_tool(return_raw=False)
        mock_framework = MagicMock()
        mock_framework.query.return_value = {
            "answer": "Answer text.",
            "sources_seen": [{"doc_id": "d1"}],
        }
        mock_framework._get_meili_client.return_value = MagicMock()
        tool._framework = mock_framework

        mock_wm = MagicMock()
        node = MagicMock()
        node.title = "Seen Doc"
        mock_wm.get_document.return_value = node
        mock_framework._workspace_manager = mock_wm

        with patch.object(tool, "_resolve_titles", return_value=["Seen Doc"]) as mock_resolve:
            result = tool.run("Test query")
            mock_resolve.assert_called_once()
            assert result.sources == ["Seen Doc"]
            assert result.answer == "Answer text."
