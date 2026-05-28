"""local_search_agent.ingestion — public re-exports."""

from local_search_agent.ingestion.cleaner import clean
from local_search_agent.ingestion.parser import BaseParser, ParserError
from local_search_agent.ingestion.parsers import (
    DOCXParser,
    HTMLParser,
    PDFParser,
    TextParser,
    XLSXParser,
)
from local_search_agent.ingestion.pipeline import IngestionPipeline, IngestStats

__all__ = [
    "BaseParser",
    "ParserError",
    "clean",
    "IngestionPipeline",
    "IngestStats",
    "PDFParser",
    "DOCXParser",
    "HTMLParser",
    "XLSXParser",
    "TextParser",
]
