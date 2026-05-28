"""
XML parser for the Local Search Agent ingestion pipeline.

Uses Python's stdlib xml.etree.ElementTree — no third-party dependency.

Strategy:
- The root element name becomes the document title (if not overridden)
- Each direct child of root becomes a ## section
- Element text content and attributes are rendered as key: value pairs
- Deeply nested elements are flattened with dot-notation paths
- CDATA and tail text are preserved
- Malformed XML falls back to raw text extraction via regex
"""

from __future__ import annotations

import logging
import os
import re
import xml.etree.ElementTree as ET
from typing import Optional

from local_search_agent.core.document_node import DocumentNode
from local_search_agent.ingestion.cleaner import clean
from local_search_agent.ingestion.parser import BaseParser, ParserError

logger = logging.getLogger(__name__)

# Strip namespace URIs like {http://...}tagname → tagname
_NS_RE = re.compile(r"\{[^}]+\}")


def _strip_ns(tag: str) -> str:
    return _NS_RE.sub("", tag)


def _element_to_markdown(el: ET.Element, depth: int = 0) -> str:
    """Recursively render an XML element as Markdown text."""
    indent = "  " * depth
    # tag = _strip_ns(el.tag)
    parts: list[str] = []

    # Attributes
    for attr_key, attr_val in el.attrib.items():
        parts.append(f"{indent}**{_strip_ns(attr_key)}**: {attr_val.strip()}")

    # Element text content
    if el.text and el.text.strip():
        parts.append(f"{indent}{el.text.strip()}")

    # Children
    for child in el:
        child_tag = _strip_ns(child.tag)
        child_md = _element_to_markdown(child, depth + 1)
        if child_md.strip():
            parts.append(f"{indent}**{child_tag}**:\n{child_md}")
        # Tail text (text after closing tag, before next sibling)
        if child.tail and child.tail.strip():
            parts.append(f"{indent}{child.tail.strip()}")

    return "\n".join(parts)


def _xml_to_markdown(root: ET.Element) -> str:
    """Convert an XML ElementTree root to Markdown sections."""
    sections: list[str] = []

    # If root has direct children, each becomes a section
    children = list(root)
    if children:
        for child in children:
            child_tag = _strip_ns(child.tag)
            body = _element_to_markdown(child, depth=0)
            if body.strip():
                sections.append(f"## {child_tag}\n\n{body}")
    else:
        # Root itself is the only content
        body = _element_to_markdown(root, depth=0)
        root_tag = _strip_ns(root.tag)
        if body.strip():
            sections.append(f"## {root_tag}\n\n{body}")

    return "\n\n".join(sections)


def _fallback_text_extract(source_path: str) -> str:
    """Strip XML tags and return raw text — used when parsing fails."""
    with open(source_path, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()
    text = re.sub(r"<[^>]+>", " ", raw)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


class XMLParser(BaseParser):
    """Parse XML files into a structured Markdown DocumentNode."""

    @property
    def supported_extensions(self) -> frozenset[str]:
        return frozenset({".xml"})

    def parse(
        self,
        source_path: str,
        workspace: str,
        title: Optional[str] = None,
    ) -> DocumentNode:
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"XML file not found: {source_path!r}")

        logger.info("Parsing XML: %s", source_path)

        try:
            tree = ET.parse(source_path)
            root = tree.getroot()
        except ET.ParseError as e:
            logger.warning(
                "XML parse error for %s, falling back to text extraction: %s", source_path, e
            )
            try:
                raw_text = _fallback_text_extract(source_path)
            except Exception as fe:
                raise ParserError(
                    source_path, f"XML parsing and fallback both failed: {fe}", original=e
                )
            cleaned_text = clean(raw_text)
            return DocumentNode.from_file(
                source_path=source_path,
                text=cleaned_text,
                workspace=workspace,
                title=title,
            )
        except Exception as e:
            raise ParserError(source_path, f"XML read failed: {e}", original=e)

        # Use root tag as title if not provided
        if title is None:
            title = _strip_ns(root.tag)

        try:
            raw_text = _xml_to_markdown(root)
        except Exception as e:
            raise ParserError(source_path, f"XML to Markdown conversion failed: {e}", original=e)

        if not raw_text.strip():
            raise ParserError(source_path, "XML produced empty output.")

        cleaned_text = clean(raw_text)

        return DocumentNode.from_file(
            source_path=source_path,
            text=cleaned_text,
            workspace=workspace,
            title=title,
        )
