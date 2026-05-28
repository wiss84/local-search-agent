"""
HTML parser for the Local Search Agent ingestion pipeline.

Uses BeautifulSoup4 + lxml for HTML wiki and intranet page extraction.

Handles:
- Confluence wiki pages
- SharePoint HTML exports
- Internal documentation sites
- Any well-structured HTML with main content areas

Strategy:
- Extract the main content area (heuristic: largest <article>, <main>,
  or <div class="content|wiki-content|page-content"> element)
- Convert heading tags → Markdown headings
- Convert tables → Markdown tables
- Strip nav bars, sidebars, script/style blocks
- Preserve code blocks as fenced Markdown

Install: pip install "beautifulsoup4>=4.12.0" "lxml>=5.2.0"
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from local_search_agent.core.document_node import DocumentNode
from local_search_agent.ingestion.cleaner import clean
from local_search_agent.ingestion.parser import BaseParser, ParserError

logger = logging.getLogger(__name__)

# CSS classes / ids that indicate main content areas (Confluence, SharePoint, etc.)
_CONTENT_SELECTORS = [
    "article",
    "main",
    '[role="main"]',
    ".wiki-content",
    ".page-content",
    ".content",
    ".main-content",
    "#content",
    "#main",
    ".documentation",
    ".doc-content",
]

# Tags to unconditionally strip (noise)
_STRIP_TAGS = {
    "script",
    "style",
    "noscript",
    "nav",
    "header",
    "footer",
    "aside",
    "form",
    "button",
    "input",
    "select",
    "textarea",
    "iframe",
    "object",
    "embed",
    "svg",
}

# Heading level map
_HEADING_MAP = {"h1": "#", "h2": "##", "h3": "###", "h4": "####", "h5": "#####", "h6": "######"}


def _table_to_markdown(table_tag) -> str:
    """Convert a BeautifulSoup <table> element to a Markdown table string."""
    rows = table_tag.find_all("tr")
    if not rows:
        return ""

    md_rows: list[list[str]] = []
    for tr in rows:
        cells = [
            cell.get_text(separator=" ", strip=True).replace("|", "\\|")
            for cell in tr.find_all(["td", "th"])
        ]
        md_rows.append(cells)

    if not md_rows:
        return ""

    # Pad all rows to same column count
    max_cols = max(len(r) for r in md_rows)
    padded = [r + [""] * (max_cols - len(r)) for r in md_rows]

    header = padded[0]
    separator = ["---"] * max_cols
    body = padded[1:]

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def _element_to_markdown(element) -> str:
    """
    Recursively convert a BeautifulSoup element tree to Markdown text.
    Handles headings, paragraphs, lists, code blocks, tables, and inline formatting.
    """
    from bs4 import NavigableString, Tag

    if isinstance(element, NavigableString):
        return str(element)

    if not isinstance(element, Tag):
        return ""

    tag = element.name.lower() if element.name else ""

    # Strip noise tags entirely
    if tag in _STRIP_TAGS:
        return ""

    # Headings
    if tag in _HEADING_MAP:
        inner = element.get_text(separator=" ", strip=True)
        return f"\n{_HEADING_MAP[tag]} {inner}\n"

    # Paragraphs
    if tag == "p":
        inner = "".join(_element_to_markdown(c) for c in element.children).strip()
        return f"\n{inner}\n" if inner else ""

    # Code blocks
    if tag == "pre":
        code_tag = element.find("code")
        code_text = code_tag.get_text() if code_tag else element.get_text()
        lang = ""
        if code_tag and code_tag.get("class"):
            classes = code_tag.get("class", [])
            for cls in classes:
                if cls.startswith("language-"):
                    lang = cls[len("language-") :]
                    break
        return f"\n```{lang}\n{code_text.strip()}\n```\n"

    if tag == "code":
        return f"`{element.get_text()}`"

    # Tables
    if tag == "table":
        return f"\n{_table_to_markdown(element)}\n"

    # Lists
    if tag in ("ul", "ol"):
        items = []
        for i, li in enumerate(element.find_all("li", recursive=False)):
            prefix = f"{i + 1}." if tag == "ol" else "-"
            inner = "".join(_element_to_markdown(c) for c in li.children).strip()
            items.append(f"{prefix} {inner}")
        return "\n" + "\n".join(items) + "\n"

    # Inline bold/italic
    if tag in ("strong", "b"):
        inner = "".join(_element_to_markdown(c) for c in element.children)
        return f"**{inner.strip()}**"

    if tag in ("em", "i"):
        inner = "".join(_element_to_markdown(c) for c in element.children)
        return f"_{inner.strip()}_"

    # Anchor: keep text, discard href (internal links not useful post-extraction)
    if tag == "a":
        return "".join(_element_to_markdown(c) for c in element.children)

    # Line break
    if tag == "br":
        return "\n"

    # Horizontal rule
    if tag == "hr":
        return "\n---\n"

    # Default: recurse into children
    return "".join(_element_to_markdown(c) for c in element.children)


class HTMLParser(BaseParser):
    """
    Parse HTML wiki/intranet pages using BeautifulSoup4 + lxml.

    Extracts the main content area, converts to Markdown, then cleans.
    Falls back to full body if no recognised content selector matches.
    """

    @property
    def supported_extensions(self) -> frozenset[str]:
        return frozenset({".html", ".htm"})

    def parse(
        self,
        source_path: str,
        workspace: str,
        title: Optional[str] = None,
    ) -> DocumentNode:
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"HTML file not found: {source_path!r}")

        try:
            from bs4 import BeautifulSoup
        except ImportError as e:
            raise ParserError(
                source_path,
                "BeautifulSoup4 is not installed. Run: pip install 'beautifulsoup4>=4.12.0' 'lxml>=5.2.0'",
                original=e,
            )

        logger.info("Parsing HTML: %s", source_path)

        try:
            with open(source_path, "r", encoding="utf-8", errors="replace") as f:
                html = f.read()

            soup = BeautifulSoup(html, "lxml")

            # Extract page title from <title> tag if not overridden
            if title is None:
                title_tag = soup.find("title")
                if title_tag:
                    title = title_tag.get_text(strip=True) or None

            # Find the main content container
            content = None
            for selector in _CONTENT_SELECTORS:
                content = soup.select_one(selector)
                if content:
                    break

            # Fallback to <body>
            if content is None:
                content = soup.find("body") or soup

            # Remove noise elements in-place before conversion
            for tag in _STRIP_TAGS:
                for el in content.find_all(tag):
                    el.decompose()

            raw_markdown = _element_to_markdown(content)

        except Exception as e:
            raise ParserError(source_path, f"HTML parsing failed: {e}", original=e)

        cleaned_text = clean(raw_markdown)

        return DocumentNode.from_file(
            source_path=source_path,
            text=cleaned_text,
            workspace=workspace,
            title=title,
        )
