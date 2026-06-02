"""
SemanticEnricher: orchestrates all three semantic search options at ingest time.

Runs Option A (ConceptCompiler) + Option B (StructuralParser) on each document,
populates DocumentNode.concepts and DocumentNode.synonyms.

Usage
-----
    from local_search_agent.semantic.enricher import SemanticEnricher

    enricher = SemanticEnricher(llm=llm)
    enriched_nodes = enricher.enrich_batch(nodes)
    # nodes now have concepts, synonyms populated
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class SemanticEnricher:
    """
    Orchestrates Option A (ConceptCompiler) + Option B (StructuralParser)
    at ingest time.

    Parameters
    ----------
    llm               : LangChain BaseChatModel for Option A. If None, skips Option A.
    enable_structural : Run Option B (StructuralParser). Default True.
    """

    def __init__(
        self,
        llm=None,
        enable_structural: bool = True,
    ):
        self._llm = llm
        self._enable_structural = enable_structural

        # Lazy-init components
        self._concept_compiler = None
        self._structural_parser = None

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

        Parameters
        ----------
        nodes : List of DocumentNode objects (modified in-place).

        Returns
        -------
        The same list (modified in-place) for convenience.
        """
        for node in nodes:
            self.enrich(node)

        return nodes
