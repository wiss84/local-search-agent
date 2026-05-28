"""
JSON parser for the Local Search Agent ingestion pipeline.

Uses Python's stdlib json module — no third-party dependency.

Strategy:
- Top-level keys become ## headings
- Nested objects are rendered as indented key: value pairs
- Arrays are rendered as Markdown lists
- Deeply nested structures are pretty-printed with indentation
- Non-object roots (arrays, scalars) are handled gracefully
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

from local_search_agent.core.document_node import DocumentNode
from local_search_agent.ingestion.cleaner import clean
from local_search_agent.ingestion.parser import BaseParser, ParserError

logger = logging.getLogger(__name__)

# Maximum recursion depth before falling back to json.dumps pretty-print
_MAX_DEPTH = 6


def _value_to_markdown(value: Any, depth: int = 0) -> str:
    """Recursively convert a JSON value to readable Markdown text."""
    indent = "  " * depth

    if value is None:
        return "_(empty)_"

    if isinstance(value, bool):
        return "Yes" if value else "No"

    if isinstance(value, (int, float)):
        return str(value)

    if isinstance(value, str):
        return value.strip() or "_(empty)_"

    if isinstance(value, list):
        if not value:
            return "_(empty list)_"
        # If all items are scalars, render as a compact list
        if all(isinstance(v, (str, int, float, bool, type(None))) for v in value):
            items = [f"{indent}- {_value_to_markdown(v)}" for v in value]
            return "\n".join(items)
        # Mixed / nested list
        parts = []
        for i, item in enumerate(value):
            parts.append(f"{indent}- **Item {i + 1}**\n{_value_to_markdown(item, depth + 1)}")
        return "\n".join(parts)

    if isinstance(value, dict):
        if depth >= _MAX_DEPTH:
            return f"```json\n{json.dumps(value, indent=2, ensure_ascii=False)}\n```"
        parts = []
        for k, v in value.items():
            rendered = _value_to_markdown(v, depth + 1)
            if isinstance(v, (dict, list)):
                parts.append(f"{indent}**{k}**:\n{rendered}")
            else:
                parts.append(f"{indent}**{k}**: {rendered}")
        return "\n".join(parts)

    # Fallback for unexpected types
    return str(value)


def _json_to_markdown(data: Any) -> str:
    """Convert a parsed JSON document to Markdown text."""
    if isinstance(data, dict):
        sections = []
        for key, value in data.items():
            heading = f"## {key}"
            body = _value_to_markdown(value, depth=0)
            sections.append(f"{heading}\n\n{body}")
        return "\n\n".join(sections)

    if isinstance(data, list):
        lines = ["## Items\n"]
        for i, item in enumerate(data):
            lines.append(f"### Item {i + 1}\n\n{_value_to_markdown(item)}")
        return "\n\n".join(lines)

    # Scalar root (unusual but valid JSON)
    return str(data)


class JSONParser(BaseParser):
    """Parse JSON files into a structured Markdown DocumentNode."""

    @property
    def supported_extensions(self) -> frozenset[str]:
        return frozenset({".json"})

    def parse(
        self,
        source_path: str,
        workspace: str,
        title: Optional[str] = None,
    ) -> DocumentNode:
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"JSON file not found: {source_path!r}")

        logger.info("Parsing JSON: %s", source_path)

        try:
            with open(source_path, "r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ParserError(source_path, f"Invalid JSON: {e}", original=e)
        except Exception as e:
            raise ParserError(source_path, f"JSON read failed: {e}", original=e)

        try:
            raw_text = _json_to_markdown(data)
        except Exception as e:
            raise ParserError(source_path, f"JSON to Markdown conversion failed: {e}", original=e)

        if not raw_text.strip():
            raise ParserError(source_path, "JSON produced empty output.")

        cleaned_text = clean(raw_text)

        return DocumentNode.from_file(
            source_path=source_path,
            text=cleaned_text,
            workspace=workspace,
            title=title,
        )
