"""
Option A: AI-Driven Concept Compiler — ingest-time semantic metadata generator.

Responsibility
--------------
Given the clean Markdown text of a document, call the configured LLM once
to extract:
  - concepts  : 5-15 broad topic tags (e.g. ["cloud costs", "AWS", "Q3 finance"])
  - synonyms  : 10-30 alternative terms/phrases for the key concepts
                (e.g. ["Amazon Web Services", "cloud spend", "infra budget"])
  - entities  : Named entities: people, products, projects, departments
                (e.g. ["Project Alpha", "Finance Division", "Sarah Chen"])
  - summary   : 2-3 sentence plain-text summary of the document

These are stored as Meilisearch searchable/filterable attributes on the
DocumentNode, making BM25 matches significantly richer without any vectors.

Design
------
- One LLM call per document at ingest time (not per query — cheap).
- LLM prompted to respond ONLY in JSON (no preamble, no markdown fences).
- Falls back gracefully: if the LLM call fails or returns malformed JSON,
  the document is indexed without semantic metadata (pure BM25 still works).
- ConceptCompiler is stateless and thread-safe.

Usage
-----
    from local_search_agent.semantic.concept_compiler import ConceptCompiler

    compiler = ConceptCompiler(llm=llm)
    metadata = compiler.compile(node)
    node.concepts = metadata.concepts + metadata.entities
    node.synonyms = metadata.synonyms
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_MAX_TEXT_CHARS = 3000

_CONCEPT_PROMPT = """\
You are a precise metadata extraction assistant. Analyze the document excerpt and extract structured semantic metadata.

Respond ONLY with a valid JSON object. No preamble, no explanation, no markdown fences.

Extract:
- "concepts": array of 5-15 broad topic tags describing what this document is about
- "synonyms": array of 10-30 alternative terms, abbreviations, and related phrases for the key concepts
- "entities": array of named entities (people, projects, products, departments, companies)
- "summary": a 2-3 sentence plain-text summary

Rules:
- concepts: lowercase short phrases, 2-4 words max each
- synonyms: include abbreviations, acronyms, common alternative names
- entities: proper nouns exactly as they appear in the document
- summary: factual and concise

Document title: {title}

Document excerpt:
{text}

JSON response:"""


@dataclass
class ConceptMetadata:
    """Semantic metadata extracted from a document by the concept compiler."""
    concepts: list[str] = field(default_factory=list)
    synonyms: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    summary: str = ""


class ConceptCompiler:
    """
    AI-driven concept compiler for ingest-time semantic metadata generation.

    One LLM call per document. Results stored on DocumentNode.concepts and
    DocumentNode.synonyms, which are Meilisearch searchable attributes.

    Parameters
    ----------
    llm : A LangChain BaseChatModel instance.
    """

    def __init__(self, llm):
        self._llm = llm

    def compile(self, node) -> ConceptMetadata:
        """
        Extract semantic metadata from a DocumentNode.

        Always returns a ConceptMetadata object — falls back to empty on failure.
        """
        from langchain_core.messages import HumanMessage

        excerpt = node.text[:_MAX_TEXT_CHARS]
        if len(node.text) > _MAX_TEXT_CHARS:
            excerpt += "\n[... document truncated for metadata extraction ...]"

        prompt = _CONCEPT_PROMPT.format(title=node.title, text=excerpt)

        try:
            response = self._llm.invoke([HumanMessage(content=prompt)])
            raw = response.content if isinstance(response.content, str) else str(response.content)
            return self._parse_response(raw, node.title)
        except Exception as e:
            logger.warning(
                "ConceptCompiler LLM call failed for %r: %s. "
                "Indexing without semantic metadata.",
                node.title, e,
            )
            return ConceptMetadata()

    def _parse_response(self, raw: str, title: str) -> ConceptMetadata:
        """Parse LLM JSON response, stripping markdown fences if present."""
        clean = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()

        try:
            data = json.loads(clean)
        except json.JSONDecodeError as e:
            logger.warning(
                "ConceptCompiler: failed to parse JSON for %r: %s", title, e
            )
            return ConceptMetadata()

        def _str_list(val) -> list[str]:
            if isinstance(val, list):
                return [str(v).strip() for v in val if v]
            return []

        return ConceptMetadata(
            concepts=_str_list(data.get("concepts", [])),
            synonyms=_str_list(data.get("synonyms", [])),
            entities=_str_list(data.get("entities", [])),
            summary=str(data.get("summary", "")).strip(),
        )
