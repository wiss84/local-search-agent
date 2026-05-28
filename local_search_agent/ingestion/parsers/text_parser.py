"""
Plain text and Markdown fallback parser.

Handles .txt and .md files directly — no third-party library needed.
For .md files the content is already Markdown; we just clean and wrap it.
For .txt files we preserve the content as-is (no conversion needed).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from local_search_agent.core.document_node import DocumentNode
from local_search_agent.ingestion.cleaner import clean
from local_search_agent.ingestion.parser import BaseParser, ParserError

logger = logging.getLogger(__name__)


class TextParser(BaseParser):
    """Parse plain text (.txt) and Markdown (.md) files."""

    @property
    def supported_extensions(self) -> frozenset[str]:
        return frozenset({".txt", ".md"})

    def parse(
        self,
        source_path: str,
        workspace: str,
        title: Optional[str] = None,
    ) -> DocumentNode:
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"Text file not found: {source_path!r}")

        logger.info("Parsing text file: %s", source_path)

        try:
            with open(source_path, "r", encoding="utf-8", errors="replace") as f:
                raw_text = f.read()
        except Exception as e:
            raise ParserError(source_path, f"Could not read file: {e}", original=e)

        cleaned_text = clean(raw_text)

        return DocumentNode.from_file(
            source_path=source_path,
            text=cleaned_text,
            workspace=workspace,
            title=title,
        )
