"""
PDF parser for the Local Search Agent ingestion pipeline.

Uses Docling (IBM) for high-quality PDF extraction:
- Preserves document structure (headings, sections)
- Converts tables to Markdown
- Handles multi-column layouts
- Strips embedded images (text only)

Large-file protection: PDFs whose page count meets or exceeds
PDF_SPLIT_THRESHOLD are split into batches of PDF_PAGES_PER_BATCH pages.
Each batch is converted independently via a temporary file so that docling's
peak memory is bounded to one batch at a time, preventing std::bad_alloc
crashes on multi-hundred-page documents.

Dependencies for splitting (checked in priority order):
1. PyMuPDF -- preferred, preserves page labels / metadata
2. pypdf    -- fallback, pure-Python, always available if the dep is installed

Install: pip install "docling>=2.0.0" "PyMuPDF>=1.25.0" "pypdf>=5.0.0"
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Optional

from local_search_agent.core.constants import (
    PDF_PAGES_PER_BATCH,
    PDF_SPLIT_THRESHOLD,
)
from local_search_agent.core.document_node import DocumentNode
from local_search_agent.ingestion.cleaner import clean
from local_search_agent.ingestion.parser import BaseParser, ParserError

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Docling singleton                                                             #
# --------------------------------------------------------------------------- #

# DocumentConverter is expensive to initialise — it downloads and loads
# layout + OCR model weights (~4s on first call, ~0.5s on subsequent calls
# if cached).  We keep one instance alive for the duration of the process
# so that every PDF in a workspace shares the same loaded models.
#
# Thread-safety: Docling's converter is not documented as thread-safe, but
# ingestion always runs in a single background thread so this is safe.

_CONVERTER = None


def _get_converter():
    """Return the module-level DocumentConverter singleton, creating it if needed."""
    global _CONVERTER
    if _CONVERTER is None:
        try:
            from docling.document_converter import DocumentConverter  # noqa: PLC0415
            _CONVERTER = DocumentConverter()
            logger.info("Docling DocumentConverter initialised (singleton).")
        except ImportError as e:
            raise ImportError(
                "Docling is not installed. Run: pip install 'docling>=2.0.0'"
            ) from e
    return _CONVERTER


# --------------------------------------------------------------------------- #
# Internal helpers                                                              #
# --------------------------------------------------------------------------- #


def _count_pdf_pages(path: str) -> Optional[int]:
    """
    Return the total page count of a PDF without loading it into memory.

    Tries PyMuPDF first (faster, more reliable page label support), then
    falls back to pypdf. Returns ``None`` if neither library is available.
    """
    # --- PyMuPDF ---
    try:
        import pymupdf  # noqa: PLC0415

        doc = pymupdf.open(path)
        try:
            return len(doc)
        finally:
            doc.close()
    except ImportError:
        pass
    except Exception as e:
        logger.debug("PyMuPDF page-count failed for %r: %s", path, e)

    # --- pypdf ---
    try:
        import warnings

        from pypdf import PdfReader  # noqa: PLC0415

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")   # suppress PdfReadWarning noise
            reader = PdfReader(path, strict=False)
            return len(reader.pages)
    except ImportError:
        pass
    except Exception as e:
        logger.debug("pypdf page-count failed for %r: %s", path, e)

    return None


def _split_pdf_batch_pymupdf(
    source_path: str, start: int, end: int
) -> Optional[str]:
    """
    Split pages [start, end) from *source_path* into a temporary PDF file.

    Returns the temporary file path, or ``None`` on failure.
    The caller is responsible for deleting the file when done.
    """
    try:
        import pymupdf  # noqa: PLC0415

        src = pymupdf.open(source_path)
        try:
            dst = pymupdf.open()
            # insert_pdf supports page labels / metadata by default
            dst.insert_pdf(src, from_page=start, to_page=end - 1)
            tf = tempfile.NamedTemporaryFile(
                suffix=".pdf", delete=False, prefix="lsa_batch_"
            )
            dst.save(tf.name)
            dst.close()
            tf.close()
            return tf.name
        finally:
            src.close()
    except ImportError:
        return None
    except Exception as e:
        logger.debug("PyMuPDF batch split [%d:%d] failed: %s", start, end, e)
        return None


def _split_pdf_batch_pypdf(
    source_path: str, start: int, end: int
) -> Optional[str]:
    """
    Split pages [start, end) from *source_path* into a temporary PDF file
    using pypdf.

    Returns the temporary file path, or ``None`` on failure.
    """
    try:
        from pypdf import PdfReader, PdfWriter  # noqa: PLC0415

        reader = PdfReader(source_path)
        writer = PdfWriter()

        for i in range(start, min(end, len(reader.pages))):
            writer.add_page(reader.pages[i])

        tf = tempfile.NamedTemporaryFile(
            suffix=".pdf", delete=False, prefix="lsa_batch_"
        )
        with open(tf.name, "wb") as f:
            writer.write(f)
        tf.close()
        return tf.name
    except ImportError:
        return None
    except Exception as e:
        logger.debug("pypdf batch split [%d:%d] failed: %s", start, end, e)
        return None


def _split_batch(
    source_path: str,
    start: int,
    end: int,
    *,
    prefer: str = "pymupdf",
) -> Optional[str]:
    """
    Dispatch to the preferred PDF splitter, falling back if unavailable.

    Parameters
    ----------
    source_path: Path to the source PDF.
    start, end  : Zero-based, half-open page range [start, end).
    prefer      : ``"pymupdf"`` or ``"pypdf"``.
    """
    if prefer == "pymupdf":
        path = _split_pdf_batch_pymupdf(source_path, start, end)
        if path is not None:
            return path
        return _split_pdf_batch_pypdf(source_path, start, end)
    else:
        path = _split_pdf_batch_pypdf(source_path, start, end)
        if path is not None:
            return path
        return _split_pdf_batch_pymupdf(source_path, start, end)


def _detect_splitting_lib() -> str:
    """Return the name of the first available PDF splitting library."""
    try:
        import pymupdf  # noqa: F401, PLC0415
        return "pymupdf"
    except ImportError:
        pass
    try:
        from pypdf import PdfReader  # noqa: F401, PLC0415
        return "pypdf"
    except ImportError:
        pass
    return ""


def _convert_pdf_in_batches(
    convertee_path: str,
    converter,
    pages_per_batch: int = PDF_PAGES_PER_BATCH,
) -> str:
    """
    Split a large PDF into page-range batches, convert each independently, and
    concatenate the resulting Markdown.

    Each batch is a temporary file that is deleted after conversion.  If a
    single batch fails with a memory error, its pages are skipped and a
    warning is appended to the output so the rest of the document is not lost.
    """
    total_pages = _count_pdf_pages(convertee_path)

    if total_pages is None:
        logger.warning(
            "No PDF splitting library available — falling back to single-call "
            "conversion for %r.", convertee_path
        )
        return _convert_single(converter, convertee_path)

    logger.info(
        "Large PDF (%d pages) — processing in batches of %d pages",
        total_pages,
        pages_per_batch,
    )

    accumulated: list[str] = []
    warn_parts: list[str] = []

    for start in range(0, total_pages, pages_per_batch):
        end = min(start + pages_per_batch, total_pages)
        tmp_path = _split_batch(convertee_path, start, end)
        if tmp_path is None:
            msg = f"[pages {start + 1}-{end}] skipped: could not create batch"
            warn_parts.append(msg)
            logger.warning("%s", msg)
            continue

        try:
            result = converter.convert(tmp_path)
            batch_md = result.document.export_to_markdown()
            accumulated.append(batch_md)
            logger.debug(
                "Converted pages %d-%d / %d",
                start + 1,
                end,
                total_pages,
            )
        except MemoryError as e:
            msg = (
                f"[pages {start + 1}-{end}] skipped: out of memory ({e!r}). "
                "Consider reducing PDF_PAGES_PER_BATCH."
            )
            warn_parts.append(msg)
            logger.warning("%s", msg)
        except Exception as e:
            msg = (
                f"[pages {start + 1}-{end}] skipped: conversion error: {e!r}"
            )
            warn_parts.append(msg)
            logger.warning("%s", msg)
        finally:
            os.unlink(tmp_path)

    if not accumulated:
        raise ParserError(
            convertee_path,
            "All PDF batches failed. The document may be empty or corrupted.",
        )

    combined = "\n\n".join(accumulated)
    if warn_parts:
        combined += "\n\n_" + "  ".join(warn_parts) + "_"

    return combined


def _convert_single(converter, path: str) -> str:
    """Convert the entire PDF in one docling call; thin wrapper for reuse."""
    result = converter.convert(path)
    return result.document.export_to_markdown()


# --------------------------------------------------------------------------- #
# Parser                                                                       #
# --------------------------------------------------------------------------- #


class PDFParser(BaseParser):
    """
    Parse PDF files using Docling.

    Docling handles:
    - Text extraction with layout awareness
    - Table detection and Markdown conversion
    - Multi-column PDF reflow
    - Header/footer detection (supplemented by our cleaner)

    Large files are transparently split into page batches so that docling's
    peak memory footprint stays bounded regardless of total page count.
    """

    @property
    def supported_extensions(self) -> frozenset[str]:
        return frozenset({".pdf"})

    def parse(
        self,
        source_path: str,
        workspace: str,
        title: Optional[str] = None,
    ) -> DocumentNode:
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"PDF file not found: {source_path!r}")

        logger.info("Parsing PDF: %s", source_path)

        try:
            converter = _get_converter()

            total_pages = _count_pdf_pages(source_path)

            if total_pages is not None and total_pages >= PDF_SPLIT_THRESHOLD:
                raw_markdown = _convert_pdf_in_batches(
                    source_path, converter, pages_per_batch=PDF_PAGES_PER_BATCH
                )
            else:
                result = converter.convert(source_path)
                raw_markdown = result.document.export_to_markdown()

            logger.debug(
                "Docling raw output for %r: %d chars, %d lines",
                os.path.basename(source_path),
                len(raw_markdown),
                raw_markdown.count("\n"),
            )

        except ParserError:
            raise
        except Exception as e:
            raise ParserError(
                source_path, f"Docling conversion failed: {e}", original=e
            )

        cleaned_text = clean(raw_markdown)

        return DocumentNode.from_file(
            source_path=source_path,
            text=cleaned_text,
            workspace=workspace,
            title=title,
        )
