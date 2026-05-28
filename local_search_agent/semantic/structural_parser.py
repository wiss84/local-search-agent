"""
Option B: Document AST / Structural Parser — ingest-time structural metadata.

Responsibility
--------------
Parse the structural skeleton of a Markdown document to extract:
  - sections     : Heading hierarchy (h1 → h2 → h3) with their text
  - definitions  : Lines matching "Term: definition" or bold-term patterns
  - references   : Cross-document references (e.g. "see [Document Name]" patterns)
  - key_values   : Table rows that look like label:value pairs (e.g. budget tables)

These are stored as structured metadata attributes on DocumentNode and fed
into Meilisearch as searchable text, making structural queries like
"what is the definition of X" or "find documents that reference Y" more accurate.

Design
------
- Pure Python regex/line-scan — no LLM call needed.
- Works on the clean Markdown produced by the ingestion cleaner.
- Graceful: returns empty StructuralMetadata on any parse error.
- Stateless and thread-safe.

Usage
-----
    from local_search_agent.semantic.structural_parser import StructuralParser

    parser = StructuralParser()
    meta = parser.parse(node)
    # meta.sections, meta.definitions, meta.references, meta.key_values
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class StructuralMetadata:
    """Structural metadata extracted from a document's Markdown AST."""
    sections: list[str] = field(default_factory=list)
    """Heading text lines, flattened (e.g. ["Executive Summary", "Financial Overview"])"""

    definitions: list[str] = field(default_factory=list)
    """Detected definition strings (e.g. ["AWS: Amazon Web Services"])"""

    references: list[str] = field(default_factory=list)
    """Detected cross-document references (e.g. ["Project Alpha Budget"])"""

    key_values: list[str] = field(default_factory=list)
    """Label:value pairs from tables (e.g. ["Total Spend: $1.2M"])"""


# Regex patterns
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
_DEFINITION_RE = re.compile(
    r"^(?:\*\*(.+?)\*\*|__(.+?)__)[\s:–\-]+(.+)$"   # **Term**: definition
)
_INLINE_DEF_RE = re.compile(r"^([A-Z][A-Za-z\s]{2,30}):\s+(.{10,})$")  # Term: definition
_SEE_REF_RE = re.compile(r"(?:see|refer to|reference|per)\s+[\"'\[]?([A-Z][^\"\'\].,\n]{3,60})[\"'\]]?", re.IGNORECASE)
_TABLE_ROW_RE = re.compile(r"^\|([^|]+)\|([^|]+)\|")


class StructuralParser:
    """
    Parse the structural skeleton of a Markdown document.

    No LLM required. Pure regex/line-scan on clean Markdown text.
    """

    def parse(self, node) -> StructuralMetadata:
        """
        Extract structural metadata from a DocumentNode.

        Parameters
        ----------
        node : DocumentNode with populated text field.

        Returns
        -------
        StructuralMetadata — always valid, empty on error.
        """
        try:
            return self._parse_text(node.text)
        except Exception as e:
            logger.warning("StructuralParser failed for %r: %s", node.title, e)
            return StructuralMetadata()

    def _parse_text(self, text: str) -> StructuralMetadata:
        meta = StructuralMetadata()
        lines = text.split("\n")

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Headings
            h_match = _HEADING_RE.match(stripped)
            if h_match:
                meta.sections.append(h_match.group(2).strip())
                continue

            # Bold-term definitions: **Term**: description
            d_match = _DEFINITION_RE.match(stripped)
            if d_match:
                term = (d_match.group(1) or d_match.group(2) or "").strip()
                defn = d_match.group(3).strip()
                if term and defn:
                    meta.definitions.append(f"{term}: {defn}")
                continue

            # Inline definitions: "Term: description" (capital letter start)
            id_match = _INLINE_DEF_RE.match(stripped)
            if id_match:
                meta.definitions.append(f"{id_match.group(1).strip()}: {id_match.group(2).strip()}")
                continue

            # Table rows — extract label:value pairs from 2-column tables
            t_match = _TABLE_ROW_RE.match(stripped)
            if t_match:
                label = t_match.group(1).strip().strip("*_")
                value = t_match.group(2).strip().strip("*_")
                # Skip separator rows and header rows
                if label and value and not re.match(r"^[-:]+$", label) and label.lower() != "metric":
                    meta.key_values.append(f"{label}: {value}")
                continue

            # Cross-document references
            for ref_match in _SEE_REF_RE.finditer(stripped):
                ref = ref_match.group(1).strip()
                if ref and len(ref) > 3:
                    meta.references.append(ref)

        # Deduplicate while preserving order
        meta.sections = list(dict.fromkeys(meta.sections))
        meta.definitions = list(dict.fromkeys(meta.definitions))
        meta.references = list(dict.fromkeys(meta.references))
        meta.key_values = list(dict.fromkeys(meta.key_values))

        return meta

    def to_searchable_text(self, meta: StructuralMetadata) -> str:
        """
        Flatten StructuralMetadata into a single searchable text string.

        This is appended to the DocumentNode's concepts/synonyms fields
        so Meilisearch can match structural content via BM25.
        """
        parts: list[str] = []
        if meta.sections:
            parts.append("Sections: " + ", ".join(meta.sections))
        if meta.definitions:
            parts.extend(meta.definitions)
        if meta.key_values:
            parts.extend(meta.key_values)
        if meta.references:
            parts.append("References: " + ", ".join(meta.references))
        return "\n".join(parts)
