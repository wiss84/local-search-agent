"""
CSV parser for the Local Search Agent ingestion pipeline.

Uses Python's stdlib csv module — no third-party dependency.

Strategy:
- First row is treated as the header
- All rows are rendered as a Markdown table
- Empty rows are skipped
- Values containing '|' are escaped
- Encoding is detected via UTF-8 with BOM fallback to latin-1
"""

from __future__ import annotations

import csv
import logging
import os
from typing import Optional

from local_search_agent.core.document_node import DocumentNode
from local_search_agent.ingestion.cleaner import clean
from local_search_agent.ingestion.parser import BaseParser, ParserError

logger = logging.getLogger(__name__)


def _read_csv(source_path: str) -> list[list[str]]:
    """
    Read a CSV file and return rows as lists of strings.
    Tries UTF-8-sig first (handles BOM), falls back to latin-1.
    """
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            with open(source_path, "r", encoding=encoding, newline="") as f:
                reader = csv.reader(f)
                rows = [row for row in reader if any(cell.strip() for cell in row)]
            return rows
        except UnicodeDecodeError:
            continue
    raise ParserError(source_path, "Could not decode CSV with UTF-8 or latin-1 encoding.")


def _rows_to_markdown(rows: list[list[str]]) -> str:
    """Convert a list of string rows to a Markdown table."""
    if not rows:
        return ""

    # Normalise all rows to the same column count
    max_cols = max(len(r) for r in rows)
    padded = [r + [""] * (max_cols - len(r)) for r in rows]

    header = [cell.strip().replace("|", "\\|") for cell in padded[0]]
    separator = ["---"] * max_cols
    body = [[cell.strip().replace("|", "\\|") for cell in row] for row in padded[1:]]

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


class CSVParser(BaseParser):
    """Parse CSV files into a Markdown table DocumentNode."""

    @property
    def supported_extensions(self) -> frozenset[str]:
        return frozenset({".csv"})

    def parse(
        self,
        source_path: str,
        workspace: str,
        title: Optional[str] = None,
    ) -> DocumentNode:
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"CSV file not found: {source_path!r}")

        logger.info("Parsing CSV: %s", source_path)

        try:
            rows = _read_csv(source_path)
        except ParserError:
            raise
        except Exception as e:
            raise ParserError(source_path, f"CSV read failed: {e}", original=e)

        if not rows:
            raise ParserError(source_path, "CSV file contains no data.")

        raw_text = _rows_to_markdown(rows)
        if not raw_text:
            raise ParserError(source_path, "CSV produced empty Markdown output.")

        cleaned_text = clean(raw_text)

        return DocumentNode.from_file(
            source_path=source_path,
            text=cleaned_text,
            workspace=workspace,
            title=title,
        )
