"""
Global constants for the Local Search Agent framework.
"""

# Package version
__version__ = "0.2.1"

# Default server settings
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

# Default Meilisearch settings
DEFAULT_MEILI_URL = "http://localhost:7700"
DEFAULT_MEILI_MASTER_KEY = "local_search_master_key"
DEFAULT_INDEX_NAME = "documents"

# DocumentNode field names used for Meilisearch indexing
FIELD_DOC_ID = "doc_id"
FIELD_TITLE = "title"
FIELD_TEXT = "text"
FIELD_FILE_TYPE = "file_type"
FIELD_FOLDER_PATH = "folder_path"
FIELD_MODIFIED_AT = "modified_at"
FIELD_WORKSPACE = "workspace"
FIELD_SOURCE_PATH = "source_path"
FIELD_CONCEPTS = "concepts"
FIELD_SYNONYMS = "synonyms"
FIELD_SUMMARY = "summary"

# Meilisearch searchable attributes (order = relevance weight)
SEARCHABLE_ATTRIBUTES = [
    FIELD_TITLE,
    FIELD_TEXT,
    FIELD_CONCEPTS,
    FIELD_SYNONYMS,
    FIELD_SUMMARY,
]

# Meilisearch filterable attributes
FILTERABLE_ATTRIBUTES = [
    FIELD_FILE_TYPE,
    FIELD_FOLDER_PATH,
    FIELD_MODIFIED_AT,
    FIELD_WORKSPACE,
]

# Snippet context window (chars around a keyword match)
SNIPPET_CONTEXT_CHARS = 300

# Agent loop limits
DEFAULT_MAX_ITERATIONS = 50
DEFAULT_MAX_RETRIES = 5

# LangGraph recursion limit — maximum number of node-entry events before the
# graph automatically stops.  Should be >= DEFAULT_MAX_ITERATIONS; set it higher
# so that the LLM can call tools several times per iteration before the hard
# LangGraph-level ceiling is hit.
LANGGRAPH_RECURSION_LIMIT = 200
DEFAULT_TOP_K = 8

# Supported file types for ingestion
SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".html",
    ".htm",
    ".xlsx",
    ".pptx",
    ".txt",
    ".md",
    ".csv",
    ".json",
    ".xml",
    ".eml",
}

# Text endpoint prefix
TEXT_ENDPOINT_PREFIX = "/text"
DOCS_ENDPOINT_PREFIX = "/docs"

# ---------------------------------------------------------------------------
# Document chunking
# ---------------------------------------------------------------------------

# Documents with total text shorter than this are never chunked (single node).
# ~1000 chars ≈ roughly 1-2 pages of dense text.
CHUNK_MIN_CHARS = 1000

# Target chunk size in characters.  The chunker accumulates content until it
# reaches this size, then starts a new chunk at the next clean break point.
# ~8000 chars ≈ roughly 4-6 pages / one complete logical section.
CHUNK_TARGET_CHARS = 8000

# Hard maximum characters per chunk.  A single paragraph or table that exceeds
# this is split at sentence boundaries regardless of structure.
# ~20000 chars ≈ roughly 10-12 pages.
CHUNK_MAX_CHARS = 20000

# Overlap in characters carried from the end of one chunk into the start of
# the next.  Protects against mid-sentence / mid-list splits at boundaries.
# ~500 chars ≈ last paragraph / ~100 words.
CHUNK_OVERLAP_CHARS = 500

# Maximum table rows per chunk (CSV / Markdown-table documents).
# The header row is prepended to every chunk automatically.
TABLE_ROWS_PER_CHUNK = 100

# ---------------------------------------------------------------------------
# Large-file ingestion protection
# ---------------------------------------------------------------------------

# Minimum character count for a PDF batch OCR result to be considered non-empty.
# If RapidOCR returns fewer characters than this for a batch, it is treated as
# a failed/scanned page and the Tesseract fallback is triggered (if available).
# ~10 chars = a few words; well below any real content but above pure whitespace.
TESSERACT_FALLBACK_MIN_CHARS = 10

# Pages-per-batch for PDFs that are split before passing to docling.
# PDF_PAGES_PER_BATCH = 15 means extract_ocr+markdown for each batch
# independently, capping peak docling memory to ~15 pages at a time.
PDF_PAGES_PER_BATCH = 15

# Minimum page count to trigger PDF batch splitting.
# Files at or below this threshold are parsed in one call as before.
PDF_SPLIT_THRESHOLD = 15

# Max characters threshold for DOCX files before trigger-by-sections batching.
DOCX_CHAR_SPLIT_THRESHOLD = 6000

# Default pages-per-batch used by the PDF warm-path fallback (PyMuPDF first,
# pypdf fallback).  Keep in sync with PDF_PAGES_PER_BATCH unless you have a
# reason to use a different value for warm-path vs hot-path batches.
PDF_FALLBACK_PAGES_PER_BATCH = 15
