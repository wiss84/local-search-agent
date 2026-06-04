"""
PDF parser for the Local Search Agent ingestion pipeline.

Uses Docling (IBM) for high-quality PDF extraction:
- Preserves document structure (headings, sections)
- Converts tables to Markdown
- Handles multi-column layouts
- Strips embedded images (text only)

OCR strategy (tiered, fastest-first per batch):
  1. PyMuPDF native text extraction — instant, zero OCR cost for digital PDFs
  2. If native text is empty/minimal → batch is scanned → try Tesseract (~1 sec/page)
  3. If Tesseract is unavailable or returns empty → fall back to RapidOCR + ONNXRuntime

RapidOCR + ONNXRuntime is the last resort, not the first attempt. This means:
  - Clean digital PDFs: never touch OCR at all
  - Scanned pages: go straight to Tesseract (fast)
  - Tesseract absent/failed: fall back to RapidOCR ONNX (slower but accurate)

Large-file protection: PDFs whose page count meets or exceeds
PDF_SPLIT_THRESHOLD are split into batches of PDF_PAGES_PER_BATCH pages.
Each batch is converted independently via a temporary file so that docling's
peak memory is bounded to one batch at a time, preventing std::bad_alloc
crashes on multi-hundred-page documents.

Dependencies for splitting (checked in priority order):
1. PyMuPDF -- preferred, preserves page labels / metadata
2. pypdf    -- fallback, pure-Python, always available if the dep is installed

Install: pip install "docling>=2.0.0" "PyMuPDF>=1.25.0" "pypdf>=5.0.0"
         pip install "rapidocr_onnxruntime"   # ONNXRuntime OCR backend
         pip install "pytesseract"             # optional Tesseract wrapper
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from typing import Optional

from local_search_agent.core.constants import (
    PDF_PAGES_PER_BATCH,
    PDF_SPLIT_THRESHOLD,
    TESSERACT_FALLBACK_MIN_CHARS,
)
from local_search_agent.core.document_node import DocumentNode
from local_search_agent.ingestion.cleaner import clean
from local_search_agent.ingestion.parser import BaseParser, ParserError

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Docling converter singletons                                                  #
# --------------------------------------------------------------------------- #
#
# Three singletons, all lazy-initialised on first use:
#   _CONVERTER_NO_OCR    — OCR disabled, layout/markdown only (digital PDFs)
#   _CONVERTER_TESSERACT — Tesseract CLI OCR (fast, scanned pages)
#   _CONVERTER_ONNX      — RapidOCR + ONNXRuntime (last resort, accurate)
#
# DocumentConverter is expensive to build (~4 s first call, ~0.5 s cached).
# Keeping singletons prevents pipeline reinitialisation on every batch.

_CONVERTER_NO_OCR = None
_CONVERTER_ONNX = None
_CONVERTER_TESSERACT = None


def _build_no_ocr_converter():
    """Build a DocumentConverter with OCR fully disabled (digital PDF fast path)."""
    from docling.datamodel.pipeline_options import PdfPipelineOptions  # noqa: PLC0415
    from docling.document_converter import (  # noqa: PLC0415
        DocumentConverter,
        InputFormat,
        PdfFormatOption,
    )

    opts = PdfPipelineOptions()
    opts.do_ocr = False
    logger.info("Docling no-OCR converter initialised (singleton).")
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )


def _build_onnx_converter():
    """
    Build a DocumentConverter that uses RapidOCR with the ONNXRuntime backend.

    Requires: pip install rapidocr_onnxruntime
    Falls back to default converter if rapidocr_onnxruntime is not installed.
    """
    from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
    from docling.document_converter import DocumentConverter, InputFormat, PdfFormatOption

    try:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.ocr_options = RapidOcrOptions(backend="onnxruntime")
        logger.info("Docling OCR: RapidOCR with ONNXRuntime backend.")
    except Exception as e:
        logger.warning(
            "Could not configure RapidOCR ONNXRuntime backend (%s). "
            "Falling back to default docling OCR settings.",
            e,
        )
        return DocumentConverter()

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )


def _build_tesseract_converter(tesseract_cmd: str):
    """
    Build a DocumentConverter that uses Tesseract CLI as the OCR engine.

    Requires: tesseract binary on PATH or managed by TesseractManager.
    """
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
    from docling.document_converter import DocumentConverter, InputFormat, PdfFormatOption

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.ocr_options = TesseractCliOcrOptions(
        tesseract_cmd=tesseract_cmd,
    )
    logger.info("Docling OCR fallback: Tesseract CLI at %s.", tesseract_cmd)

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )


def _get_no_ocr_converter():
    """Return the no-OCR converter singleton, creating it if needed."""
    global _CONVERTER_NO_OCR
    if _CONVERTER_NO_OCR is None:
        try:
            _CONVERTER_NO_OCR = _build_no_ocr_converter()
        except ImportError as e:
            raise ImportError("Docling is not installed. Run: pip install 'docling>=2.0.0'") from e
    return _CONVERTER_NO_OCR


def _get_onnx_converter():
    """Return the ONNX converter singleton, creating it if needed."""
    global _CONVERTER_ONNX
    if _CONVERTER_ONNX is None:
        try:
            _CONVERTER_ONNX = _build_onnx_converter()
            # Models are now cached locally — suppress HuggingFace revision
            # check HTTP calls on every subsequent converter build.
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            logger.info("Docling ONNX converter initialised (singleton). HF offline mode enabled.")
        except ImportError as e:
            raise ImportError("Docling is not installed. Run: pip install 'docling>=2.0.0'") from e
    return _CONVERTER_ONNX


def _get_tesseract_converter() -> Optional[object]:
    """
    Return the Tesseract converter singleton, or None if Tesseract is unavailable.
    Creates the singleton on first call.
    """
    global _CONVERTER_TESSERACT
    if _CONVERTER_TESSERACT is not None:
        return _CONVERTER_TESSERACT

    try:
        from local_search_agent.core.tesseract_manager import get_tesseract_manager  # noqa: PLC0415

        manager = get_tesseract_manager()
        cmd = manager.ensure()
        if cmd is None:
            logger.debug("Tesseract not available — skipping Tesseract converter init.")
            return None

        _CONVERTER_TESSERACT = _build_tesseract_converter(cmd)
        logger.info("Docling Tesseract converter initialised (singleton).")
        return _CONVERTER_TESSERACT

    except Exception as e:
        logger.warning("Could not initialise Tesseract converter: %s", e)
        return None


# --------------------------------------------------------------------------- #
# Internal helpers                                                              #
# --------------------------------------------------------------------------- #


def _count_pdf_pages(path: str) -> Optional[int]:
    """
    Return the total page count of a PDF without loading it into memory.

    Tries PyMuPDF first (faster, more reliable page label support), then
    falls back to pypdf. Returns ``None`` if neither library is available.
    """
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

    try:
        import warnings

        from pypdf import PdfReader  # noqa: PLC0415

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            reader = PdfReader(path, strict=False)
            return len(reader.pages)
    except ImportError:
        pass
    except Exception as e:
        logger.debug("pypdf page-count failed for %r: %s", path, e)

    return None


def _split_pdf_batch_pymupdf(source_path: str, start: int, end: int) -> Optional[str]:
    """
    Split pages [start, end) from *source_path* into a temporary PDF file.
    Returns the temporary file path, or None on failure.
    Caller is responsible for deleting the file when done.
    """
    try:
        import pymupdf  # noqa: PLC0415

        src = pymupdf.open(source_path)
        try:
            dst = pymupdf.open()
            dst.insert_pdf(src, from_page=start, to_page=end - 1)
            tf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, prefix="lsa_batch_")
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


def _split_pdf_batch_pypdf(source_path: str, start: int, end: int) -> Optional[str]:
    """
    Split pages [start, end) from *source_path* into a temporary PDF file
    using pypdf. Returns the temporary file path, or None on failure.
    """
    try:
        from pypdf import PdfReader, PdfWriter  # noqa: PLC0415

        reader = PdfReader(source_path)
        writer = PdfWriter()
        for i in range(start, min(end, len(reader.pages))):
            writer.add_page(reader.pages[i])

        tf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, prefix="lsa_batch_")
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
    """Dispatch to the preferred PDF splitter, falling back if unavailable."""
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


def _convert_single(converter, path: str) -> str:
    """Convert the entire PDF in one docling call."""
    result = converter.convert(path)
    return result.document.export_to_markdown()


def _extract_native_text_pymupdf(path: str) -> str:
    """
    Extract the embedded text layer from a PDF using PyMuPDF.

    Instant — no OCR, no ML models. Returns empty string if the PDF
    has no text layer (i.e. it is a scanned/image-only document).
    """
    try:
        import pymupdf  # noqa: PLC0415

        doc = pymupdf.open(path)
        try:
            return "".join(page.get_text() for page in doc)
        finally:
            doc.close()
    except ImportError:
        pass
    except Exception as e:
        logger.debug("PyMuPDF native text extraction failed for %r: %s", path, e)
    return ""


def _is_empty_result(text: str) -> bool:
    """
    Return True if extracted text is effectively empty.

    Used both for the native text check (decide if OCR is needed at all)
    and for the Tesseract output check (decide if ONNX fallback is needed).
    """
    return len(text.strip()) < TESSERACT_FALLBACK_MIN_CHARS


# --------------------------------------------------------------------------- #
# Core conversion logic                                                         #
# --------------------------------------------------------------------------- #


def _convert_batch_with_fallback(tmp_path: str, start: int, end: int) -> tuple[str, str]:
    """
    Convert a single page-range batch using the tiered OCR strategy.

    Strategy (fastest-first):
      1. PyMuPDF native text — instant, zero OCR cost
         Sufficient text found → run docling with OCR disabled for layout/markdown
      2. Tesseract CLI — fast (~1 sec/page), for scanned batches
         Tesseract available and returns sufficient text → return
      3. RapidOCR + ONNXRuntime — slow but accurate, true last resort

    Returns
    -------
    (markdown_text, engine_used)  where engine_used is: 'native' | 'tesseract' | 'onnx'
    """
    # --- Step 1: Native text extraction (instant, no OCR) ---
    native_text = _extract_native_text_pymupdf(tmp_path)
    if not _is_empty_result(native_text):
        logger.debug("Pages %d-%d: native text sufficient, skipping OCR.", start + 1, end)
        try:
            converter = _get_no_ocr_converter()
            result = converter.convert(tmp_path)
            return result.document.export_to_markdown(), "native"
        except Exception as e:
            logger.debug("Native docling pass failed (%s) — proceeding to OCR.", e)

    # --- Step 2: Tesseract (fast, ~1 sec/page) ---
    tess_converter = _get_tesseract_converter()
    if tess_converter is not None:
        try:
            logger.info("Pages %d-%d: scanned batch — trying Tesseract.", start + 1, end)
            result = tess_converter.convert(tmp_path)
            tess_text = result.document.export_to_markdown()
            if not _is_empty_result(tess_text):
                return tess_text, "tesseract"
            logger.debug(
                "Pages %d-%d: Tesseract returned minimal text — falling back to ONNX.",
                start + 1,
                end,
            )
        except Exception as e:
            logger.warning(
                "Pages %d-%d: Tesseract failed (%s) — falling back to ONNX.",
                start + 1,
                end,
                e,
            )
    else:
        logger.debug(
            "Pages %d-%d: scanned batch, Tesseract unavailable — using ONNX.",
            start + 1,
            end,
        )

    # --- Step 3: RapidOCR + ONNXRuntime (last resort) ---
    onnx_converter = _get_onnx_converter()
    try:
        result = onnx_converter.convert(tmp_path)
        return result.document.export_to_markdown(), "onnx"
    except Exception as e:
        logger.warning("ONNX converter failed for pages %d-%d: %s", start + 1, end, e)
        raise


def _convert_pdf_in_batches(
    convertee_path: str,
    pages_per_batch: int = PDF_PAGES_PER_BATCH,
) -> str:
    """
    Split a large PDF into page-range batches, convert each with the tiered
    OCR strategy, and concatenate the resulting Markdown.

    Each batch is a temporary file deleted after conversion. If a single batch
    fails with a memory error, its pages are skipped and a warning is appended
    to the output so the rest of the document is not lost.
    """
    total_pages = _count_pdf_pages(convertee_path)

    if total_pages is None:
        logger.warning(
            "No PDF splitting library available — falling back to single-call conversion for %r.",
            convertee_path,
        )
        onnx_converter = _get_onnx_converter()
        return _convert_single(onnx_converter, convertee_path)

    logger.info(
        "Large PDF (%d pages) — processing in batches of %d pages",
        total_pages,
        pages_per_batch,
    )

    accumulated: list[str] = []
    warn_parts: list[str] = []
    tesseract_fallback_count = 0

    for start in range(0, total_pages, pages_per_batch):
        end = min(start + pages_per_batch, total_pages)
        tmp_path = _split_batch(convertee_path, start, end)

        if tmp_path is None:
            msg = f"[pages {start + 1}-{end}] skipped: could not create batch"
            warn_parts.append(msg)
            logger.warning("%s", msg)
            continue

        try:
            batch_md, engine = _convert_batch_with_fallback(tmp_path, start, end)
            accumulated.append(batch_md)
            if engine == "tesseract":
                tesseract_fallback_count += 1
            logger.debug("Converted pages %d-%d / %d [%s]", start + 1, end, total_pages, engine)

        except MemoryError as e:
            msg = (
                f"[pages {start + 1}-{end}] skipped: out of memory ({e!r}). "
                "Consider reducing PDF_PAGES_PER_BATCH."
            )
            warn_parts.append(msg)
            logger.warning("%s", msg)
        except Exception as e:
            msg = f"[pages {start + 1}-{end}] skipped: conversion error: {e!r}"
            warn_parts.append(msg)
            logger.warning("%s", msg)
        finally:
            os.unlink(tmp_path)

    if not accumulated:
        raise ParserError(
            convertee_path,
            "All PDF batches failed. The document may be empty or corrupted.",
        )

    if tesseract_fallback_count:
        logger.info(
            "Tesseract fallback was used for %d/%d batch(es) in %r.",
            tesseract_fallback_count,
            len(accumulated),
            os.path.basename(convertee_path),
        )

    combined = "\n\n".join(accumulated)
    if warn_parts:
        combined += "\n\n_" + "  ".join(warn_parts) + "_"

    return combined


# --------------------------------------------------------------------------- #
# Parser                                                                        #
# --------------------------------------------------------------------------- #


class PDFParser(BaseParser):
    """
    Parse PDF files using Docling with a tiered OCR strategy.

    OCR engine selection (per batch):
      1. RapidOCR + ONNXRuntime  — fast, 2-3x quicker than PyTorch on CPU
      2. Tesseract CLI            — fallback when ONNX returns empty/minimal text
         (auto-downloaded on Windows via TesseractManager; on Linux/macOS
          install with `sudo apt install tesseract-ocr` / `brew install tesseract`)

    Docling handles native-text-first internally: OCR only runs on detected
    bitmap regions, not on pages with a selectable text layer.
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
            total_pages = _count_pdf_pages(source_path)

            if total_pages is not None and total_pages >= PDF_SPLIT_THRESHOLD:
                raw_markdown = _convert_pdf_in_batches(
                    source_path, pages_per_batch=PDF_PAGES_PER_BATCH
                )
            else:
                # Small PDF: single pass with native-first tiered strategy
                # Use a temp copy so _convert_batch_with_fallback owns the file path
                tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, prefix="lsa_small_")
                tmp.close()
                try:
                    shutil.copy2(source_path, tmp.name)
                    raw_markdown, engine = _convert_batch_with_fallback(
                        tmp.name, 0, total_pages or 0
                    )
                    logger.debug(
                        "Small PDF %r converted [%s].",
                        os.path.basename(source_path),
                        engine,
                    )
                finally:
                    os.unlink(tmp.name)

            logger.debug(
                "Docling raw output for %r: %d chars, %d lines",
                os.path.basename(source_path),
                len(raw_markdown),
                raw_markdown.count("\n"),
            )

        except ParserError:
            raise
        except Exception as e:
            raise ParserError(source_path, f"Docling conversion failed: {e}", original=e)

        cleaned_text = clean(raw_markdown)

        return DocumentNode.from_file(
            source_path=source_path,
            text=cleaned_text,
            workspace=workspace,
            title=title,
        )
