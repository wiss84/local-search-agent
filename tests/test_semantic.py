"""
Unit tests for Phase 5: semantic layer, link graph, access control.

All LLM calls are mocked. No live services needed.

Covers:
- ConceptCompiler: JSON parsing, malformed output fallback, truncation
- StructuralParser: headings, definitions, tables, references
- QueryExpander: LLM-based and index-based expansion
- LinkGraph: add/query links, batch insert, same_topic auto-build
- SemanticEnricher: end-to-end enrichment on DocumentNode
- AccessControlMiddleware: 401 on missing header, 403 on denied, 200 on allowed
"""

from __future__ import annotations

from unittest.mock import MagicMock

from local_search_agent.core.document_node import DocumentNode
from local_search_agent.semantic.concept_compiler import ConceptCompiler, ConceptMetadata
from local_search_agent.semantic.enricher import SemanticEnricher
from local_search_agent.semantic.link_graph import RELATION_SAME_TOPIC, LinkGraph
from local_search_agent.semantic.query_expander import QueryExpander
from local_search_agent.semantic.structural_parser import StructuralParser

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_node(
    tmp_path, name="report.txt", text="Default content.", workspace="ws"
) -> DocumentNode:
    f = tmp_path / name
    f.write_text(text, encoding="utf-8")
    return DocumentNode.from_file(str(f), text=text, workspace=workspace)


def _mock_llm(response_text: str):
    llm = MagicMock()
    response = MagicMock()
    response.content = response_text
    llm.invoke.return_value = response
    return llm


# ---------------------------------------------------------------------------
# ConceptCompiler
# ---------------------------------------------------------------------------


class TestConceptCompiler:
    def test_parses_valid_json(self, tmp_path):
        llm = _mock_llm("""{
            "concepts": ["cloud costs", "AWS", "finance"],
            "synonyms": ["Amazon Web Services", "infrastructure budget"],
            "entities": ["Project Alpha", "Finance Division"],
            "summary": "Q3 finance report for Project Alpha."
        }""")
        compiler = ConceptCompiler(llm=llm)
        node = _make_node(tmp_path, text="AWS spend on Project Alpha was $1.2M in Q3 2024.")
        meta = compiler.compile(node)

        assert "cloud costs" in meta.concepts
        assert "Amazon Web Services" in meta.synonyms
        assert "Project Alpha" in meta.entities
        assert "Q3 finance" in meta.summary

    def test_strips_markdown_fences(self, tmp_path):
        llm = _mock_llm(
            '```json\n{"concepts": ["test"], "synonyms": [], "entities": [], "summary": "ok"}\n```'
        )
        compiler = ConceptCompiler(llm=llm)
        node = _make_node(tmp_path)
        meta = compiler.compile(node)
        assert "test" in meta.concepts

    def test_fallback_on_invalid_json(self, tmp_path):
        llm = _mock_llm("This is not JSON at all.")
        compiler = ConceptCompiler(llm=llm)
        node = _make_node(tmp_path)
        meta = compiler.compile(node)
        assert meta.concepts == []
        assert meta.synonyms == []

    def test_fallback_on_llm_exception(self, tmp_path):
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("LLM down")
        compiler = ConceptCompiler(llm=llm)
        node = _make_node(tmp_path)
        meta = compiler.compile(node)
        assert isinstance(meta, ConceptMetadata)
        assert meta.concepts == []

    def test_truncates_long_document(self, tmp_path):
        llm = _mock_llm('{"concepts": [], "synonyms": [], "entities": [], "summary": ""}')
        compiler = ConceptCompiler(llm=llm)
        node = _make_node(tmp_path, text="A" * 10000)
        compiler.compile(node)
        call_args = llm.invoke.call_args[0][0]
        prompt_text = call_args[0].content
        assert len(prompt_text) < 10000 + 500  # truncated + prompt overhead


# ---------------------------------------------------------------------------
# StructuralParser
# ---------------------------------------------------------------------------


class TestStructuralParser:
    def test_extracts_headings(self, tmp_path):
        text = "# Executive Summary\n\n## Financial Overview\n\nSome content.\n"
        node = _make_node(tmp_path, text=text)
        parser = StructuralParser()
        meta = parser.parse(node)
        assert "Executive Summary" in meta.sections
        assert "Financial Overview" in meta.sections

    def test_extracts_bold_definitions(self, tmp_path):
        text = "**AWS**: Amazon Web Services cloud platform.\n"
        node = _make_node(tmp_path, text=text)
        parser = StructuralParser()
        meta = parser.parse(node)
        assert any("AWS" in d for d in meta.definitions)

    def test_extracts_table_key_values(self, tmp_path):
        text = "| Metric | Value |\n| --- | --- |\n| Total Spend | $1.2M |\n| Headcount | 42 |\n"
        node = _make_node(tmp_path, text=text)
        parser = StructuralParser()
        meta = parser.parse(node)
        assert any("Total Spend" in kv for kv in meta.key_values)

    def test_extracts_references(self, tmp_path):
        text = "See Project Alpha Budget for details. Refer to Finance Policy 2024.\n"
        node = _make_node(tmp_path, text=text)
        parser = StructuralParser()
        meta = parser.parse(node)
        assert any("Project Alpha" in r for r in meta.references)

    def test_deduplicates_sections(self, tmp_path):
        text = "# Summary\n\nContent.\n\n# Summary\n\nMore content.\n"
        node = _make_node(tmp_path, text=text)
        parser = StructuralParser()
        meta = parser.parse(node)
        assert meta.sections.count("Summary") == 1

    def test_to_searchable_text(self, tmp_path):
        from local_search_agent.semantic.structural_parser import StructuralMetadata

        meta = StructuralMetadata(
            sections=["Executive Summary"],
            definitions=["AWS: Amazon Web Services"],
            key_values=["Total Spend: $1.2M"],
        )
        parser = StructuralParser()
        text = parser.to_searchable_text(meta)
        assert "Executive Summary" in text
        assert "AWS" in text
        assert "Total Spend" in text

    def test_graceful_on_empty_text(self, tmp_path):
        node = _make_node(tmp_path, text="")
        parser = StructuralParser()
        meta = parser.parse(node)
        assert meta.sections == []
        assert meta.definitions == []


# ---------------------------------------------------------------------------
# QueryExpander
# ---------------------------------------------------------------------------


class TestQueryExpander:
    def test_llm_expansion_appends_terms(self):
        llm = _mock_llm('["Amazon Web Services", "cloud spend", "infra budget"]')
        expander = QueryExpander(llm=llm)
        expanded = expander.expand("AWS costs")
        assert "AWS costs" in expanded
        assert "Amazon Web Services" in expanded or "cloud spend" in expanded

    def test_llm_expansion_fallback_on_bad_json(self):
        llm = _mock_llm("not json")
        expander = QueryExpander(llm=llm)
        result = expander.expand("turnover rate")
        assert result == "turnover rate"

    def test_llm_expansion_fallback_on_exception(self):
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("LLM down")
        expander = QueryExpander(llm=llm)
        result = expander.expand("morale")
        assert result == "morale"

    def test_no_llm_returns_original(self):
        expander = QueryExpander(llm=None)
        result = expander.expand("test query")
        assert result == "test query"

    def test_index_based_expansion_appends_concepts(self):
        mock_meili = MagicMock()
        mock_meili.search.return_value = [
            {
                "doc_id": "abc",
                "concepts": ["employee satisfaction", "job retention"],
                "title": "HR Report",
                "file_type": "pdf",
                "workspace": "hr",
                "source_path": "/hr.pdf",
                "modified_at": "2025-01-01",
                "snippet": "",
            },
        ]
        expander = QueryExpander(llm=None)
        expanded = expander.expand("morale", meili_client=mock_meili, workspace="hr")
        assert "morale" in expanded
        assert "employee satisfaction" in expanded or "job retention" in expanded

    def test_empty_query_unchanged(self):
        expander = QueryExpander()
        assert expander.expand("") == ""
        assert expander.expand("   ") == "   "


# ---------------------------------------------------------------------------
# LinkGraph
# ---------------------------------------------------------------------------


class TestLinkGraph:
    def test_add_and_get_link(self, tmp_path):
        graph = LinkGraph(db_path=str(tmp_path / "test.db"))
        graph.add_link("doc_a", "doc_b", "references", weight=0.9)
        related = graph.get_related("doc_a")
        assert len(related) == 1
        assert related[0]["target_doc_id"] == "doc_b"
        assert related[0]["relation_type"] == "references"
        assert abs(related[0]["weight"] - 0.9) < 0.01

    def test_self_link_ignored(self, tmp_path):
        graph = LinkGraph(db_path=str(tmp_path / "test.db"))
        graph.add_link("doc_a", "doc_a", "references")
        assert graph.get_related("doc_a") == []

    def test_duplicate_link_ignored(self, tmp_path):
        graph = LinkGraph(db_path=str(tmp_path / "test.db"))
        graph.add_link("doc_a", "doc_b", "references")
        graph.add_link("doc_a", "doc_b", "references")
        assert graph.get_link_count("doc_a") == 1

    def test_weight_clamped_to_0_1(self, tmp_path):
        graph = LinkGraph(db_path=str(tmp_path / "test.db"))
        graph.add_link("doc_a", "doc_b", weight=5.0)
        related = graph.get_related("doc_a")
        assert related[0]["weight"] <= 1.0

    def test_filter_by_relation_type(self, tmp_path):
        graph = LinkGraph(db_path=str(tmp_path / "test.db"))
        graph.add_link("doc_a", "doc_b", "references")
        graph.add_link("doc_a", "doc_c", "same_topic")
        refs = graph.get_related("doc_a", relation_type="references")
        assert len(refs) == 1
        assert refs[0]["target_doc_id"] == "doc_b"

    def test_min_weight_filter(self, tmp_path):
        graph = LinkGraph(db_path=str(tmp_path / "test.db"))
        graph.add_link("doc_a", "doc_b", weight=0.3)
        graph.add_link("doc_a", "doc_c", weight=0.8)
        high_weight = graph.get_related("doc_a", min_weight=0.5)
        assert len(high_weight) == 1
        assert high_weight[0]["target_doc_id"] == "doc_c"

    def test_delete_links_for_doc(self, tmp_path):
        graph = LinkGraph(db_path=str(tmp_path / "test.db"))
        graph.add_link("doc_a", "doc_b")
        graph.add_link("doc_a", "doc_c")
        graph.delete_links_for_doc("doc_a")
        assert graph.get_link_count("doc_a") == 0

    def test_build_same_topic_links(self, tmp_path):
        graph = LinkGraph(db_path=str(tmp_path / "test.db"))

        # Create nodes with shared concepts
        node_a = MagicMock()
        node_a.doc_id = "doc_a"
        node_a.concepts = ["cloud costs", "AWS", "Q3 finance", "infra"]

        node_b = MagicMock()
        node_b.doc_id = "doc_b"
        node_b.concepts = ["cloud costs", "AWS", "Q3 finance", "budget"]

        node_c = MagicMock()
        node_c.doc_id = "doc_c"
        node_c.concepts = ["hr policy", "recruitment"]  # no overlap

        pairs = graph.build_same_topic_links([node_a, node_b, node_c], min_shared_concepts=3)
        assert pairs == 1  # only a↔b share 3 concepts

        related_a = graph.get_related("doc_a", relation_type=RELATION_SAME_TOPIC)
        assert any(r["target_doc_id"] == "doc_b" for r in related_a)


# ---------------------------------------------------------------------------
# SemanticEnricher
# ---------------------------------------------------------------------------


class TestSemanticEnricher:
    def test_enriches_concepts_and_synonyms(self, tmp_path):
        llm = _mock_llm(
            '{"concepts": ["cloud costs"], "synonyms": ["AWS spend"], "entities": ["Project Alpha"], "summary": "Finance report."}'
        )
        enricher = SemanticEnricher(llm=llm, enable_structural=True)
        node = _make_node(tmp_path, text="# Finance\n\nAWS spend on Project Alpha was $1.2M.")
        enricher.enrich(node)

        assert "cloud costs" in node.concepts
        assert "Project Alpha" in node.concepts
        assert "AWS spend" in node.synonyms

    def test_structural_adds_section_headings(self, tmp_path):
        enricher = SemanticEnricher(llm=None, enable_structural=True)
        node = _make_node(tmp_path, text="# Executive Summary\n\nContent here.")
        enricher.enrich(node)
        assert "Executive Summary" in node.synonyms

    def test_no_llm_still_runs_structural(self, tmp_path):
        enricher = SemanticEnricher(llm=None, enable_structural=True)
        node = _make_node(tmp_path, text="# HR Policy\n\n**Turnover**: employee departure rate.")
        enricher.enrich(node)
        assert "HR Policy" in node.synonyms

    def test_enrich_batch_processes_all(self, tmp_path):
        enricher = SemanticEnricher(llm=None, enable_structural=True)
        nodes = [_make_node(tmp_path, name=f"doc{i}.txt", text=f"# Section {i}") for i in range(5)]
        enricher.enrich_batch(nodes)
        for i, node in enumerate(nodes):
            assert f"Section {i}" in node.synonyms

    def test_link_graph_built_for_shared_concepts(self, tmp_path):
        llm = _mock_llm(
            '{"concepts": ["cloud", "AWS", "infra", "budget"], "synonyms": [], "entities": [], "summary": ""}'
        )
        enricher = SemanticEnricher(
            llm=llm,
            enable_structural=False,
            enable_link_graph=True,
            db_path=str(tmp_path / "test.db"),
            min_shared_concepts=3,
        )
        nodes = [
            _make_node(tmp_path, name=f"doc{i}.txt", text=f"AWS cloud infra budget doc {i}")
            for i in range(2)
        ]
        # Pre-set concepts (normally set by compiler)
        for node in nodes:
            node.concepts = ["cloud", "AWS", "infra", "budget"]

        enricher.enrich_batch(nodes)

        from local_search_agent.semantic.link_graph import LinkGraph

        graph = LinkGraph(db_path=str(tmp_path / "test.db"))
        related = graph.get_related(nodes[0].doc_id)
        assert any(r["target_doc_id"] == nodes[1].doc_id for r in related)


# ---------------------------------------------------------------------------
# AccessControlMiddleware
# ---------------------------------------------------------------------------


class TestAccessControlMiddleware:
    def _make_app(self, tmp_path, enable_ac: bool = True):

        from fastapi.testclient import TestClient

        from local_search_agent.core.config import SearchAgentConfig
        from local_search_agent.server.fastapi_app import build_app
        from local_search_agent.workspace.workspace_manager import WorkspaceManager

        db_path = str(tmp_path / "test.db")
        config = SearchAgentConfig(
            workspace_name="ws",
            provider="ollama",
            db_path=db_path,
            enable_access_control=enable_ac,
        )
        wm = WorkspaceManager(db_path=db_path)

        # Register a test document
        txt = tmp_path / "report.txt"
        txt.write_text("AWS spend $1.2M", encoding="utf-8")
        from local_search_agent.core.document_node import DocumentNode

        node = DocumentNode.from_file(str(txt), text="AWS spend $1.2M", workspace="ws")
        wm.create_workspace("ws", str(tmp_path))
        wm.register_document(node)

        app = build_app(config=config, workspace_manager=wm)
        return TestClient(app), node.doc_id

    def test_no_ac_serves_document(self, tmp_path):
        client, doc_id = self._make_app(tmp_path, enable_ac=False)
        resp = client.get(f"/text/{doc_id}")
        assert resp.status_code == 200

    def test_ac_enabled_no_header_returns_401(self, tmp_path):
        import os

        os.environ.pop("LSA_ACCESS_CONTROL_BYPASS", None)
        client, doc_id = self._make_app(tmp_path, enable_ac=True)
        resp = client.get(f"/text/{doc_id}")
        assert resp.status_code == 401

    def test_ac_bypass_env_allows_request(self, tmp_path):
        import os

        os.environ["LSA_ACCESS_CONTROL_BYPASS"] = "1"
        try:
            client, doc_id = self._make_app(tmp_path, enable_ac=True)
            resp = client.get(f"/text/{doc_id}")
            assert resp.status_code == 200
        finally:
            os.environ.pop("LSA_ACCESS_CONTROL_BYPASS", None)

    def test_health_endpoint_not_protected(self, tmp_path):
        import os

        os.environ.pop("LSA_ACCESS_CONTROL_BYPASS", None)
        client, _ = self._make_app(tmp_path, enable_ac=True)
        resp = client.get("/health")
        assert resp.status_code == 200
