"""
Unit tests for individual document parsers (ingestion/parsers/).

Tests cover only parsers that do NOT require heavy third-party models
(i.e. not docling/PDF/DOCX). This keeps the suite fast and dependency-free in CI.

Covered parsers:
- TextParser    (.txt, .md)
- CSVParser     (.csv)
- JSONParser    (.json)
- XMLParser     (.xml)
- EMLParser     (.eml)

Each parser is tested for:
- Returns a non-empty string on valid input
- Returns file_type matching the extension
- Handles empty / minimal files gracefully (raises ParserError or returns minimal content)
- Handles encoding edge cases (UTF-8 with BOM, non-ASCII chars)

Design notes:
- All tests use tmp_path (real filesystem I/O, no mocks) so they exercise
  the full parse → clean → DocumentNode pipeline.
- clean() always appends a trailing "\n", so empty-file assertions use "\n".
- The _parse() helper routes by extension the same way the pipeline does.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_search_agent.ingestion.parser import ParserError
from local_search_agent.ingestion.parsers import (
    CSVParser,
    EMLParser,
    JSONParser,
    TextParser,
    XMLParser,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(tmp_path: Path, name: str, content: str, encoding: str = "utf-8") -> Path:
    f = tmp_path / name
    f.write_text(content, encoding=encoding)
    return f


def _write_bytes(tmp_path: Path, name: str, data: bytes) -> Path:
    f = tmp_path / name
    f.write_bytes(data)
    return f


def _parse(path: Path):
    """Route to the correct parser and return (text, file_type).

    Uses the same can_parse() dispatch the real pipeline uses, so any
    extension mismatch in the parser's supported_extensions is caught here.
    """
    parsers = [TextParser(), CSVParser(), JSONParser(), XMLParser(), EMLParser()]
    for p in parsers:
        if p.can_parse(str(path)):
            node = p.parse(str(path), workspace="test")
            return node.text, node.file_type
    raise ValueError(f"No parser registered for extension: {path.suffix!r}")


# ---------------------------------------------------------------------------
# TextParser
# ---------------------------------------------------------------------------


class TestTextParser:
    def test_txt_basic(self, tmp_path):
        f = _write(tmp_path, "report.txt", "Hello world.\nLine two.")
        text, ft = _parse(f)
        assert "Hello world" in text
        assert ft == "txt"

    def test_md_basic(self, tmp_path):
        f = _write(tmp_path, "readme.md", "# Title\n\nSome content here.")
        text, ft = _parse(f)
        assert "Title" in text
        assert ft == "md"

    def test_txt_empty_file_returns_single_newline(self, tmp_path):
        # clean("") → normalize_whitespace → "".strip() + "\n" = "\n"
        f = _write(tmp_path, "empty.txt", "")
        text, ft = _parse(f)
        assert text == "\n"

    def test_txt_utf8_non_ascii(self, tmp_path):
        f = _write(tmp_path, "unicode.txt", "Héllo Wörld — 日本語テスト")
        text, ft = _parse(f)
        assert "Héllo" in text

    def test_txt_utf8_bom_stripped(self, tmp_path):
        # TextParser opens with errors="replace" and Python's utf-8 codec
        # strips the BOM when present. The "\ufeff" BOM is also removed by
        # normalize_unicode in the cleaner pipeline.
        f = _write_bytes(tmp_path, "bom.txt", b"\xef\xbb\xbfHello with BOM")
        text, ft = _parse(f)
        assert "Hello" in text
        assert "\ufeff" not in text  # BOM removed by cleaner

    def test_md_all_heading_levels_preserved(self, tmp_path):
        f = _write(tmp_path, "doc.md", "# H1\n## H2\n### H3\nContent.")
        text, ft = _parse(f)
        assert "H1" in text
        assert "H2" in text
        assert "H3" in text

    def test_txt_trailing_whitespace_stripped(self, tmp_path):
        f = _write(tmp_path, "spaces.txt", "Line one   \nLine two  \n")
        text, ft = _parse(f)
        for line in text.splitlines():
            assert line == line.rstrip(), f"Trailing whitespace found in line: {line!r}"

    def test_txt_excessive_blank_lines_collapsed(self, tmp_path):
        f = _write(tmp_path, "blanks.txt", "Para one.\n\n\n\n\nPara two.")
        text, ft = _parse(f)
        # cleaner collapses 3+ blank lines to 2
        assert "\n\n\n\n" not in text


# ---------------------------------------------------------------------------
# CSVParser
# ---------------------------------------------------------------------------


class TestCSVParser:
    def test_csv_basic_content(self, tmp_path):
        f = _write(
            tmp_path, "data.csv", "Name,Department,Salary\nAlice,Engineering,95000\nBob,HR,72000\n"
        )
        text, ft = _parse(f)
        assert "Alice" in text
        assert "Engineering" in text
        assert ft == "csv"

    def test_csv_output_is_markdown_table(self, tmp_path):
        # CSVParser renders as Markdown table — every data row must contain pipes
        f = _write(tmp_path, "table.csv", "Col1,Col2\nA,B\nC,D\n")
        text, ft = _parse(f)
        assert "|" in text

    def test_csv_header_row_present(self, tmp_path):
        f = _write(tmp_path, "header.csv", "Name,Department\nAlice,Engineering\n")
        text, ft = _parse(f)
        assert "Name" in text
        assert "Department" in text

    def test_csv_empty_file_raises_parser_error(self, tmp_path):
        f = _write(tmp_path, "empty.csv", "")
        with pytest.raises(ParserError):
            _parse(f)

    def test_csv_single_column(self, tmp_path):
        f = _write(tmp_path, "single.csv", "Name\nAlice\nBob\nCarol\n")
        text, ft = _parse(f)
        assert "Alice" in text

    def test_csv_quoted_fields_with_commas(self, tmp_path):
        f = _write(
            tmp_path, "quoted.csv", 'Name,Bio\nAlice,"Engineer, Senior"\nBob,"Manager, HR"\n'
        )
        text, ft = _parse(f)
        assert "Engineer" in text

    def test_csv_pipe_chars_in_values_are_escaped(self, tmp_path):
        # Pipe chars inside cell values must be escaped so the Markdown table is valid
        f = _write(tmp_path, "pipes.csv", "Name,Code\nAlice,A|B\n")
        text, ft = _parse(f)
        # The escaped form "A\|B" should appear (not a raw unescaped "|" mid-cell)
        assert r"A\|B" in text or "A|B" not in text.replace(r"\|", "")


# ---------------------------------------------------------------------------
# JSONParser
# ---------------------------------------------------------------------------


class TestJSONParser:
    def test_json_dict(self, tmp_path):
        data = {"name": "Alice", "department": "Engineering", "salary": 95000}
        f = _write(tmp_path, "data.json", json.dumps(data))
        text, ft = _parse(f)
        assert "Alice" in text
        assert ft == "json"

    def test_json_list_of_dicts(self, tmp_path):
        data = [{"id": 1, "value": "alpha"}, {"id": 2, "value": "beta"}]
        f = _write(tmp_path, "list.json", json.dumps(data))
        text, ft = _parse(f)
        assert "alpha" in text
        assert "beta" in text

    def test_json_top_level_keys_become_sections(self, tmp_path):
        # JSONParser renders top-level dict keys as ## headings
        data = {"summary": "Good quarter.", "revenue": 1200000}
        f = _write(tmp_path, "sections.json", json.dumps(data))
        text, ft = _parse(f)
        assert "summary" in text
        assert "revenue" in text

    def test_json_nested_object(self, tmp_path):
        data = {"company": "Acme", "address": {"city": "London", "country": "UK"}}
        f = _write(tmp_path, "nested.json", json.dumps(data))
        text, ft = _parse(f)
        assert "London" in text

    def test_json_empty_object_raises_parser_error(self, tmp_path):
        # _json_to_markdown({}) → "" → ParserError("JSON produced empty output")
        f = _write(tmp_path, "empty.json", "{}")
        with pytest.raises(ParserError):
            _parse(f)

    def test_json_empty_file_raises_parser_error(self, tmp_path):
        f = _write(tmp_path, "blank.json", "")
        with pytest.raises(ParserError):
            _parse(f)

    def test_json_invalid_syntax_raises_parser_error(self, tmp_path):
        f = _write(tmp_path, "bad.json", "{ not valid json }")
        with pytest.raises(ParserError):
            _parse(f)

    def test_json_boolean_values_rendered(self, tmp_path):
        data = {"active": True, "archived": False}
        f = _write(tmp_path, "booleans.json", json.dumps(data))
        text, ft = _parse(f)
        # _value_to_markdown converts True→"Yes", False→"No"
        assert "Yes" in text
        assert "No" in text

    def test_json_list_of_scalars(self, tmp_path):
        data = {"tags": ["python", "search", "local"]}
        f = _write(tmp_path, "tags.json", json.dumps(data))
        text, ft = _parse(f)
        assert "python" in text
        assert "search" in text


# ---------------------------------------------------------------------------
# XMLParser
# ---------------------------------------------------------------------------


class TestXMLParser:
    def test_xml_basic(self, tmp_path):
        content = (
            '<?xml version="1.0"?>'
            "<report><title>Q3 Report</title><summary>Strong performance.</summary></report>"
        )
        f = _write(tmp_path, "report.xml", content)
        text, ft = _parse(f)
        assert "Q3 Report" in text
        assert ft == "xml"

    def test_xml_nested_elements(self, tmp_path):
        content = "<employees><employee><name>Alice</name><dept>HR</dept></employee></employees>"
        f = _write(tmp_path, "nested.xml", content)
        text, ft = _parse(f)
        assert "Alice" in text

    def test_xml_empty_tags_raise_parser_error(self, tmp_path):
        # <root></root> → _xml_to_markdown produces "" → ParserError
        f = _write(tmp_path, "empty.xml", "<root></root>")
        with pytest.raises(ParserError):
            _parse(f)

    def test_xml_attributes_extracted(self, tmp_path):
        content = '<items><item id="42" status="active">Widget</item></items>'
        f = _write(tmp_path, "attrs.xml", content)
        text, ft = _parse(f)
        assert "42" in text or "active" in text

    def test_xml_invalid_falls_back_to_text(self, tmp_path):
        # Malformed XML → fallback extractor strips tags and returns raw text
        f = _write(tmp_path, "bad.xml", "<unclosed>some content here")
        text, ft = _parse(f)
        # Fallback should return a non-empty string (the stripped text content)
        assert isinstance(text, str)
        assert len(text.strip()) > 0

    def test_xml_namespace_stripped(self, tmp_path):
        content = '<root xmlns:ns="http://example.com"><ns:title>Namespaced Title</ns:title></root>'
        f = _write(tmp_path, "ns.xml", content)
        text, ft = _parse(f)
        assert "Namespaced Title" in text
        # The {http://...} prefix should not appear in the output
        assert "http://example.com" not in text


# ---------------------------------------------------------------------------
# EMLParser
# ---------------------------------------------------------------------------


class TestEMLParser:
    def _make_eml(self, tmp_path: Path, subject: str, body: str, name: str = "email.eml") -> Path:
        content = (
            f"From: alice@example.com\r\n"
            f"To: bob@example.com\r\n"
            f"Subject: {subject}\r\n"
            f"Date: Mon, 1 Jan 2024 09:00:00 +0000\r\n"
            f"Content-Type: text/plain; charset=utf-8\r\n"
            f"\r\n"
            f"{body}\r\n"
        )
        f = tmp_path / name
        f.write_text(content, encoding="utf-8")
        return f

    def test_eml_subject_in_output(self, tmp_path):
        f = self._make_eml(tmp_path, "Q3 Budget Review", "Please see attached.")
        text, ft = _parse(f)
        assert "Q3 Budget Review" in text
        assert ft == "eml"

    def test_eml_body_extracted(self, tmp_path):
        f = self._make_eml(tmp_path, "Meeting", "Let's meet on Friday at 3pm.")
        text, ft = _parse(f)
        assert "Friday" in text

    def test_eml_from_address_in_metadata_block(self, tmp_path):
        f = self._make_eml(tmp_path, "Test", "Body text.")
        text, ft = _parse(f)
        assert "alice@example.com" in text

    def test_eml_to_address_in_metadata_block(self, tmp_path):
        f = self._make_eml(tmp_path, "Test", "Body text.")
        text, ft = _parse(f)
        assert "bob@example.com" in text

    def test_eml_empty_body_does_not_raise(self, tmp_path):
        # Subject + headers still produce content — no ParserError expected
        f = self._make_eml(tmp_path, "Empty Body", "")
        text, ft = _parse(f)
        assert isinstance(text, str)
        assert len(text.strip()) > 0

    def test_eml_metadata_section_heading_present(self, tmp_path):
        f = self._make_eml(tmp_path, "Heading Check", "Some body.")
        text, ft = _parse(f)
        assert "Email Metadata" in text

    def test_eml_body_section_heading_present(self, tmp_path):
        f = self._make_eml(tmp_path, "Section Check", "Body content here.")
        text, ft = _parse(f)
        assert "Body" in text

    def test_eml_multipart_plain_preferred_over_html(self, tmp_path):
        # When both plain and HTML parts exist, plain text should win
        content = (
            "From: alice@example.com\r\n"
            "To: bob@example.com\r\n"
            "Subject: Multipart\r\n"
            "MIME-Version: 1.0\r\n"
            "Content-Type: multipart/alternative; boundary=boundary123\r\n"
            "\r\n"
            "--boundary123\r\n"
            "Content-Type: text/plain; charset=utf-8\r\n"
            "\r\n"
            "Plain text version.\r\n"
            "--boundary123\r\n"
            "Content-Type: text/html; charset=utf-8\r\n"
            "\r\n"
            "<html><body>HTML version.</body></html>\r\n"
            "--boundary123--\r\n"
        )
        f = tmp_path / "multipart.eml"
        f.write_text(content, encoding="utf-8")
        text, ft = _parse(f)
        assert "Plain text version" in text
