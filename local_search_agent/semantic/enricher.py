"""
SemanticEnricher: orchestrates all three semantic search options at ingest time.

Runs Option A (ConceptCompiler) + Option B (StructuralParser) on each document,
populates DocumentNode.concepts and DocumentNode.synonyms, and optionally
builds the link graph (Option C is query-time only, used in the agent tool).

Usage
-----
    from local_search_agent.semantic.enricher import SemanticEnricher

    enricher = SemanticEnricher(
        llm=llm,                    # for Option A (concept compiler)
        enable_link_graph=True,     # for cross-document links
        db_path="local_search_agent.db",
    )
    enriched_nodes = enricher.enrich_batch(nodes)
    # nodes now have concepts, synonyms populated
    # link graph updated if enable_link_graph=True
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SemanticEnricher:
    """
    Orchestrates Option A (ConceptCompiler) + Option B (StructuralParser)
    at ingest time, and builds same_topic links via LinkGraph.

    Parameters
    ----------
    llm               : LangChain BaseChatModel for Option A. If None, skips Option A.
    enable_structural : Run Option B (StructuralParser). Default True.
    enable_link_graph : Build same_topic links via LinkGraph. Default False.
    db_path           : SQLite path (needed only if enable_link_graph=True).
    min_shared_concepts: Minimum shared concepts to create a same_topic link.
    """

    def __init__(
        self,
        llm=None,
        enable_structural: bool = True,
        enable_link_graph: bool = False,
        db_path: Optional[str] = None,
        min_shared_concepts: int = 3,
    ):
        self._llm = llm
        self._enable_structural = enable_structural
        self._enable_link_graph = enable_link_graph
        self._db_path = db_path
        self._min_shared_concepts = min_shared_concepts

        # Lazy-init components
        self._concept_compiler = None
        self._structural_parser = None
        self._link_graph = None

    def _get_concept_compiler(self):
        if self._concept_compiler is None and self._llm is not None:
            from local_search_agent.semantic.concept_compiler import ConceptCompiler

            self._concept_compiler = ConceptCompiler(llm=self._llm)
        return self._concept_compiler

    def _get_structural_parser(self):
        if self._structural_parser is None:
            from local_search_agent.semantic.structural_parser import StructuralParser

            self._structural_parser = StructuralParser()
        return self._structural_parser

    def _get_link_graph(self):
        if self._link_graph is None and self._db_path:
            from local_search_agent.semantic.link_graph import LinkGraph

            self._link_graph = LinkGraph(db_path=self._db_path)
        return self._link_graph

    def enrich(self, node) -> None:
        """
        Enrich a single DocumentNode in-place with semantic metadata.

        Option A: LLM concept extraction → node.concepts + node.synonyms
        Option B: Structural parsing → sections/definitions appended to node.synonyms
        """
        # Option A: Concept Compiler
        compiler = self._get_concept_compiler()
        if compiler is not None:
            meta = compiler.compile(node)
            # Merge: combine AI concepts with entities into node.concepts
            # Combine AI synonyms into node.synonyms
            node.concepts = list(dict.fromkeys(node.concepts + meta.concepts + meta.entities))
            node.synonyms = list(dict.fromkeys(node.synonyms + meta.synonyms))
            logger.debug(
                "ConceptCompiler enriched %r: %d concepts, %d synonyms",
                node.title,
                len(node.concepts),
                len(node.synonyms),
            )

        # Option B: Structural Parser
        if self._enable_structural:
            parser = self._get_structural_parser()
            struct_meta = parser.parse(node)
            # Add section headings and key_values as searchable synonyms
            structural_terms = (
                struct_meta.sections
                + [kv.split(":")[0].strip() for kv in struct_meta.key_values]
                + struct_meta.definitions
            )
            node.synonyms = list(dict.fromkeys(node.synonyms + structural_terms))
            logger.debug(
                "StructuralParser enriched %r: %d sections, %d definitions",
                node.title,
                len(struct_meta.sections),
                len(struct_meta.definitions),
            )

    def enrich_batch(self, nodes: list) -> list:
        """
        Enrich a list of DocumentNodes in-place.

        After enriching all nodes, optionally builds same_topic links
        in the link graph (requires enable_link_graph=True and db_path).

        Parameters
        ----------
        nodes : List of DocumentNode objects (modified in-place).

        Returns
        -------
        The same list (modified in-place) for convenience.
        """
        for node in nodes:
            self.enrich(node)

        # Build same_topic links across the batch
        if self._enable_link_graph and nodes:
            graph = self._get_link_graph()
            if graph is not None:
                linked_pairs = graph.build_same_topic_links(
                    nodes,
                    min_shared_concepts=self._min_shared_concepts,
                )
                logger.info(
                    "LinkGraph: built same_topic links for %d document pairs.", linked_pairs
                )

        return nodes
