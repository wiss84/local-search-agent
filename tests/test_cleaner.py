"""
Unit tests for the text cleaning pipeline (ingestion/cleaner.py).

All tests are pure string operations — no filesystem or network needed.
"""

from __future__ import annotations

from local_search_agent.ingestion.cleaner import (
    clean,
    fix_broken_words,
    normalize_unicode,
    normalize_whitespace,
    remove_control_characters,
    remove_watermarks,
    strip_page_numbers,
)

# ---------------------------------------------------------------------------
# Individual step tests
# ---------------------------------------------------------------------------

class TestNormalizeUnicode:
    def test_smart_quotes_replaced(self):
        assert normalize_unicode("\u201cHello\u201d") == '"Hello"'
        assert normalize_unicode("\u2018it\u2019s") == "'it's"

    def test_em_dash_replaced(self):
        assert normalize_unicode("foo\u2014bar") == "foo-bar"

    def test_non_breaking_space_replaced(self):
        assert normalize_unicode("a\u00a0b") == "a b"

    def test_zero_width_removed(self):
        assert normalize_unicode("a\u200bb") == "ab"
        assert normalize_unicode("\ufeffstart") == "start"

    def test_ellipsis_replaced(self):
        assert normalize_unicode("wait\u2026") == "wait..."

    def test_clean_text_unchanged(self):
        text = "Hello world. Normal text."
        assert normalize_unicode(text) == text


class TestRemoveWatermarks:
    def test_removes_confidential_line(self):
        text = "Header\nCONFIDENTIAL\nContent here."
        result = remove_watermarks(text)
        assert "CONFIDENTIAL" not in result
        assert "Content here." in result

    def test_removes_draft_line(self):
        text = "DRAFT\nActual content."
        result = remove_watermarks(text)
        assert "DRAFT" not in result

    def test_removes_internal_use_only(self):
        text = "INTERNAL USE ONLY\nContent."
        result = remove_watermarks(text)
        assert "INTERNAL USE ONLY" not in result

    def test_does_not_remove_inline_confidential(self):
        # "CONFIDENTIAL" embedded in a sentence should NOT be removed
        text = "This document is CONFIDENTIAL and private."
        result = remove_watermarks(text)
        assert "CONFIDENTIAL" in result


class TestStripPageNumbers:
    def test_page_n_of_m(self):
        text = "Content\nPage 3 of 12\nMore content"
        result = strip_page_numbers(text)
        assert "Page 3 of 12" not in result
        assert "Content" in result

    def test_page_n_alone(self):
        text = "Content\nPage 5\nMore content"
        result = strip_page_numbers(text)
        assert "Page 5" not in result

    def test_dash_number_dash(self):
        text = "Content\n- 7 -\nMore"
        result = strip_page_numbers(text)
        assert "- 7 -" not in result

    def test_preserves_normal_content(self):
        text = "The project started on page 1 of the report."
        result = strip_page_numbers(text)
        # This is inline text, not a standalone page number line — should be preserved
        assert "project started" in result



class TestFixBrokenWords:
    def test_rejoins_hyphenated_linebreak(self):
        text = "implemen-\ntation of the"
        result = fix_broken_words(text)
        assert "implementation" in result

    def test_preserves_list_hyphens(self):
        # List items starting with "- " should not be joined
        text = "- First item\n- Second item"
        result = fix_broken_words(text)
        assert "- First item" in result

    def test_no_change_to_normal_text(self):
        text = "This is normal text without any breaks."
        assert fix_broken_words(text) == text


class TestNormalizeWhitespace:
    def test_collapses_excessive_blank_lines(self):
        text = "Para 1\n\n\n\n\nPara 2"
        result = normalize_whitespace(text)
        assert "\n\n\n" not in result
        assert "Para 1" in result
        assert "Para 2" in result

    def test_strips_trailing_spaces(self):
        text = "Line one   \nLine two  \n"
        result = normalize_whitespace(text)
        assert "   " not in result

    def test_ends_with_single_newline(self):
        text = "Content here"
        result = normalize_whitespace(text)
        assert result.endswith("\n")
        assert not result.endswith("\n\n")


class TestRemoveControlCharacters:
    def test_removes_null_bytes(self):
        assert remove_control_characters("a\x00b") == "ab"

    def test_removes_bell_char(self):
        assert remove_control_characters("a\x07b") == "ab"

    def test_preserves_newlines_and_tabs(self):
        text = "line1\nline2\ttabbed"
        assert remove_control_characters(text) == text


# ---------------------------------------------------------------------------
# Full pipeline test
# ---------------------------------------------------------------------------

class TestCleanPipeline:
    def test_full_pipeline_on_realistic_pdf_extract(self):
        """Simulate messy text that would come from a PDF parser."""
        raw = (
            "Acme Corp \u2014 Finance Division\n"
            "CONFIDENTIAL\n"
            "## Q3 2024 Financial Report\n\n"
            "AWS spend on Pro-\nject Alpha reached $1.2M.\n\n"
            "Page 3 of 12\n\n"
            "Acme Corp \u2014 Finance Division\n"
            "Employee morale sur\u00adveys showed improvement.\n\n"
            "| Metric | Q2 | Q3 |\n"
            "| --- | --- | --- |\n"
            "| Revenue | $4M | $5M |\n\n"
            "Acme Corp \u2014 Finance Division\n"
            "Conclusion: strong quarter.\n"
        )
        result = clean(raw)

        assert "CONFIDENTIAL" not in result
        assert "Page 3 of 12" not in result
        assert "Project Alpha" in result         # hyphen rejoined
        assert "$1.2M" in result
        assert "Revenue" in result               # table preserved
        assert "strong quarter" in result
        assert result.endswith("\n")
