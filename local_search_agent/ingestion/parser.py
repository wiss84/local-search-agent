"""
Base parser interface for the Local Search Agent ingestion pipeline.

Every file-type parser inherits from BaseParser and implements parse().
The pipeline calls parse() and receives a DocumentNode with clean Markdown text.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional

from local_search_agent.core.document_node import DocumentNode


class BaseParser(ABC):
    """
    Abstract base class for all document parsers.

    Subclasses implement:
    - supported_extensions  : frozenset of lowercase extensions (e.g. {".pdf"})
    - parse()               : returns a DocumentNode with clean Markdown text
    """

    @property
    @abstractmethod
    def supported_extensions(self) -> frozenset[str]:
        """Return the set of file extensions this parser handles."""
        ...

    def can_parse(self, path: str) -> bool:
        """Return True if this parser handles the given file path."""
        ext = os.path.splitext(path)[1].lower()
        return ext in self.supported_extensions

    @abstractmethod
    def parse(
        self,
        source_path: str,
        workspace: str,
        title: Optional[str] = None,
    ) -> DocumentNode:
        """
        Parse a file and return a DocumentNode with clean Markdown text.

        Parameters
        ----------
        source_path : Absolute path to the source file.
        workspace   : Logical workspace name for the resulting DocumentNode.
        title       : Optional title override. Defaults to filename stem.

        Returns
        -------
        DocumentNode with populated text, metadata, and doc_id.

        Raises
        ------
        FileNotFoundError   : If source_path does not exist.
        ParserError         : If parsing fails for any reason.
        """
        ...


class ParserError(Exception):
    """Raised when a parser cannot process a document."""

    def __init__(self, path: str, reason: str, original: Optional[Exception] = None):
        self.path = path
        self.reason = reason
        self.original = original
        super().__init__(f"Parser failed for {path!r}: {reason}")
