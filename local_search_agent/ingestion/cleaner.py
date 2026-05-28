"""
Text cleaning pipeline for the Local Search Agent ingestion framework.

Responsibilities
----------------
- Strip common header/footer boilerplate (page numbers, document titles repeated on every page)
- Normalize whitespace (collapse blank lines, strip trailing spaces)
- Remove watermarks and common noise patterns
- Normalize Unicode (smart quotes, em-dashes, zero-width chars)
- Preserve Markdown structure produced by Docling/BeautifulSoup parsers
- Tables are already Markdown by the time they reach the cleaner (parsers handle conversion)

Design
------
The cleaner is a pure function pipeline: each step takes a string and returns a string.
Steps are composable and individually testable.
`clean()` is the public entry point that runs all steps in order.
"""

from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# Individual cleaning steps
# ---------------------------------------------------------------------------


def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode to NFC form and replace common typographic substitutions.

    Replaces:
    - Smart quotes (left/right single/double) → straight ASCII quotes
    - Em-dash / en-dash → ASCII hyphen-minus
    - Non-breaking space → regular space
    - Zero-width characters (ZWJ, ZWNJ, BOM, soft hyphen) → empty string
    - Horizontal ellipsis → three dots
    """
    text = unicodedata.normalize("NFC", text)
    replacements = {
        "\u2018": "'",  # left single quotation mark
        "\u2019": "'",  # right single quotation mark
        "\u201c": '"',  # left double quotation mark
        "\u201d": '"',  # right double quotation mark
        "\u2014": "-",  # em dash
        "\u2013": "-",  # en dash
        "\u00a0": " ",  # non-breaking space
        "\u200b": "",  # zero-width space
        "\u200c": "",  # zero-width non-joiner
        "\u200d": "",  # zero-width joiner
        "\ufeff": "",  # BOM
        "\u00ad": "",  # soft hyphen
        "\u2026": "...",  # horizontal ellipsis
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def remove_watermarks(text: str) -> str:
    """
    Remove common watermark strings that appear repeatedly across pages.

    Patterns targeted:
    - "CONFIDENTIAL", "DRAFT", "INTERNAL USE ONLY" as standalone lines
    - "PROPRIETARY", "DO NOT DISTRIBUTE" as standalone lines
    """
    watermark_patterns = [
        r"(?im)^[ \t]*(CONFIDENTIAL|DRAFT|INTERNAL[ \t]+USE[ \t]+ONLY|PROPRIETARY|DO[ \t]+NOT[ \t]+DISTRIBUTE|FOR[ \t]+INTERNAL[ \t]+USE)[ \t]*$\n?",
    ]
    for pat in watermark_patterns:
        text = re.sub(pat, "", text)
    return text


def strip_page_numbers(text: str) -> str:
    """
    Remove standalone page number lines.

    Matches lines that are just:
    - "Page 3 of 12" / "Page 3"
    - "- 3 -" / "3" (lone digit lines)
    - "3 | Company Report"
    """
    patterns = [
        r"(?im)^[ \t]*[Pp]age\s+\d+(\s+of\s+\d+)?[ \t]*$\n?",
        r"(?im)^[ \t]*-\s*\d+\s*-[ \t]*$\n?",
        r"(?im)^[ \t]*\d+\s*\|[^\n]*$\n?",
    ]
    for pat in patterns:
        text = re.sub(pat, "", text)
    return text


def normalize_whitespace(text: str) -> str:
    """
    Collapse excessive blank lines and strip trailing whitespace per line.

    - More than 2 consecutive blank lines → 2 blank lines
    - Trailing spaces/tabs on each line → stripped
    - Ensure single trailing newline at end of document
    """
    # Strip trailing whitespace per line
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    # Collapse 3+ consecutive blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def fix_broken_words(text: str) -> str:
    """
    Re-join hyphenated line-breaks produced by PDF column layout extraction.

    Pattern: "hyphen-\\nnewword" → "hyphennewword"
    Only applies when the hyphen is at end of line (not a list dash).
    """
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def remove_control_characters(text: str) -> str:
    """Remove non-printable ASCII control characters (except newline and tab)."""
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

_PIPELINE = [
    remove_control_characters,
    normalize_unicode,
    remove_watermarks,
    strip_page_numbers,
    fix_broken_words,
    normalize_whitespace,
]


def clean(text: str) -> str:
    """
    Run the full cleaning pipeline on raw extracted text.

    Steps (in order):
    1. remove_control_characters  — strip non-printable bytes
    2. normalize_unicode           — NFC + typographic substitutions
    3. remove_watermarks           — strip CONFIDENTIAL / DRAFT lines
    4. strip_page_numbers          — strip "Page 3 of 12" lines
    5. fix_broken_words            — rejoin PDF hyphenated line breaks
    6. normalize_whitespace        — collapse blank lines, strip trailing spaces

    Parameters
    ----------
    text : Raw text string from any parser.

    Returns
    -------
    Cleaned text string, always ending with a single newline.
    """
    for step in _PIPELINE:
        text = step(text)
    return text
