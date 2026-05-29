"""local_search_agent.ingestion.parsers — package init."""

from local_search_agent.ingestion.parsers.csv_parser import CSVParser
from local_search_agent.ingestion.parsers.docx_parser import DOCXParser
from local_search_agent.ingestion.parsers.eml_parser import EMLParser
from local_search_agent.ingestion.parsers.html_parser import HTMLParser
from local_search_agent.ingestion.parsers.json_parser import JSONParser
from local_search_agent.ingestion.parsers.pdf_parser import PDFParser
from local_search_agent.ingestion.parsers.pptx_parser import PPTXParser
from local_search_agent.ingestion.parsers.text_parser import TextParser
from local_search_agent.ingestion.parsers.xlsx_parser import XLSXParser
from local_search_agent.ingestion.parsers.xml_parser import XMLParser

__all__ = [
    "PDFParser",
    "DOCXParser",
    "HTMLParser",
    "PPTXParser",
    "XLSXParser",
    "TextParser",
    "CSVParser",
    "JSONParser",
    "XMLParser",
    "EMLParser",
]
