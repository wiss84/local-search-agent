# Ingestion Guide

## Overview

The ingestion pipeline transforms raw documents into clean, searchable content in Meilisearch. It handles everything from file discovery to chunking and indexing — you just point it at a directory.

## How It Works

```
Directory walk
     ↓
Delta check (skip unchanged files)
     ↓
Parser (PDF, DOCX, HTML, XLSX, ...)
     ↓
Text Cleaner (6-step pipeline)
     ↓
Chunker (sliding window with overlap)
     ↓
Semantic Enrichment (optional, Experimental)
     ↓
Meilisearch indexing (immediate, per file)
     ↓
WorkspaceManager registration (SQLite)
```

## Per-File Flush & Resume

Each file is flushed to Meilisearch and registered in SQLite **immediately after it finishes parsing** — not after all files are done. This means:

- If the process is killed or crashes mid-ingestion, all completed files are already safe in Meilisearch
- On restart, `document_needs_reindex()` checks modification time against the SQLite registry and skips already-indexed files
- Only the file that was actively being processed at the moment of the crash needs to be reprocessed

This makes ingestion resilient on large corpora where a single problematic file could otherwise cause hours of work to be lost.

## Delta Ingestion

By default the pipeline only re-indexes files whose `modified_at` timestamp has changed since the last run. On a 10,000-document corpus with 50 changed files, only the 50 are re-indexed. This makes scheduled re-ingestion fast.

To force a full re-index of all files:

```bash
local-search ingest --workspace finance --dirs "C:\my_docs" --force
```

```python
framework.ingest_and_index(force=True)
```

To wipe everything and start from scratch:

```bash
local-search ingest --workspace finance --dirs "C:\my_docs" --wipe
```

```python
framework.wipe_and_reingest()
```

---

## Supported File Types

| Extension | Parser | Notes |
|-----------|--------|-------|
| `.pdf` | Docling | Layout-aware extraction, multi-column support |
| `.docx` | Docling | Preserves headings, tables, lists |
| `.html`, `.htm` | BeautifulSoup4 + lxml | Strips navigation/ads, extracts main content |
| `.xlsx` | openpyxl | Converts sheets to Markdown tables |
| `.pptx` | python-pptx | Extracts text per slide with slide title |
| `.txt` | TextParser | Plain text, UTF-8 |
| `.md` | TextParser | Preserves Markdown |
| `.csv` | CSVParser | Converts to Markdown table |
| `.json` | JSONParser | Pretty-printed key-value extraction |
| `.xml` | XMLParser | Tag-aware text extraction |
| `.eml` | EMLParser | Email headers + body |

Files with unsupported extensions are silently skipped.

---

## PDF OCR Strategy

PDF ingestion uses a tiered OCR strategy designed to be as fast as possible while remaining accurate. Each batch of pages (15 pages at a time for large PDFs) goes through the following steps in order:

### Step 1 — Native text extraction (instant)

PyMuPDF attempts to extract the embedded text layer directly from the PDF. This is instant — no OCR, no ML models involved. If the batch has sufficient extractable text (more than `TESSERACT_FALLBACK_MIN_CHARS` characters), the pipeline runs Docling with OCR disabled for layout and Markdown conversion only, then moves to the next batch.

This path handles all digitally-created PDFs (research papers, reports, exported documents) at full speed.

### Step 2 — Tesseract OCR (fast, optional)

If native text extraction returns empty or near-empty text, the batch is identified as scanned or image-based. If Tesseract is installed and on PATH, it is used as the OCR engine (~1 second per page on CPU). Tesseract is detected automatically on all platforms via `shutil.which("tesseract")` — no configuration needed.

If Tesseract returns sufficient text, the pipeline moves to the next batch.

**Tesseract is optional.** If it is not installed, this step is skipped silently and the pipeline falls back to Step 3.

### Step 3 — RapidOCR + ONNXRuntime (last resort)

If Tesseract is unavailable or also returned empty text, RapidOCR with the ONNXRuntime backend is used. This is slower than Tesseract (minutes per page for heavily scanned documents) but more accurate on complex layouts, low-quality scans, and non-Latin scripts.

### Summary

| Scenario | Engine used | Speed |
|----------|-------------|-------|
| Digital PDF with text layer | Native (PyMuPDF, no OCR) | Instant |
| Scanned PDF, Tesseract installed | Tesseract CLI | ~1 sec/page |
| Scanned PDF, Tesseract not installed | RapidOCR + ONNXRuntime | Slow |
| RapidOCR also fails | Empty text, file still indexed | — |

### Installing Tesseract (optional but recommended)

Tesseract dramatically speeds up ingestion of scanned PDFs. Without it, a 100-page scanned document can take 30–120 minutes; with it, the same document typically finishes in under 2 minutes.

**Windows**

Download the installer from https://github.com/UB-Mannheim/tesseract/wiki
During installation, make sure to check **“Add Tesseract to the system PATH”**.

**Linux**
```bash
# Ubuntu / Debian
sudo apt install tesseract-ocr

# Fedora / RHEL
sudo dnf install tesseract

# Arch
sudo pacman -S tesseract
```

**macOS**
```bash
brew install tesseract
```

After installation, restart the application. The pipeline will detect Tesseract automatically and log:
```
Tesseract detected at: /usr/bin/tesseract
```

If Tesseract is not found, the pipeline logs:
```
Tesseract not found on PATH — scanned PDFs will use RapidOCR (slower).
```
and continues without it.

### Large PDF batching

PDFs with more than `PDF_SPLIT_THRESHOLD` pages (default: 15) are split into batches of `PDF_PAGES_PER_BATCH` pages (default: 15) before being passed to the OCR pipeline. Each batch is processed independently as a temporary file, which:

- Caps peak memory usage regardless of total page count
- Allows per-batch OCR engine selection (a mixed PDF can use native extraction for digital pages and Tesseract for scanned pages within the same file)
- Prevents a single bad page from aborting the entire document

Batch results are concatenated into a single Markdown document before chunking.

---

## Text Cleaning

Every parsed document goes through a 6-step cleaning pipeline before indexing:

1. **Remove control characters** — strips non-printable bytes (except newline and tab)
2. **Normalize Unicode** — NFC normalization, smart quotes → straight quotes, em/en dash → hyphen, zero-width chars removed
3. **Remove watermarks** — strips lines matching `CONFIDENTIAL`, `DRAFT`, `INTERNAL USE ONLY`, `PROPRIETARY`, `DO NOT DISTRIBUTE`
4. **Strip page numbers** — removes standalone `Page 3 of 12` / `- 3 -` lines
5. **Fix broken words** — re-joins hyphenated line-breaks from PDF column extraction (`hyphen-\nword` → `hyphenword`)
6. **Normalize whitespace** — collapses 3+ blank lines to 2, strips trailing spaces

---

## Chunking

Long documents are split into overlapping chunks so the agent can retrieve focused passages rather than massive walls of text. Short documents (under `CHUNK_MIN_CHARS`) are indexed as-is.

### Table documents

If more than 50% of non-empty lines start with `|` (Markdown table rows), the document uses row-based chunking: `TABLE_ROWS_PER_CHUNK` rows per chunk, with the header row prepended to every chunk. No overlap — rows are structurally independent.

### All other documents

Sliding-window chunking with overlap:

1. Text accumulates until it reaches `CHUNK_TARGET_CHARS`
2. The pipeline looks ahead for the best break point — in priority order: heading boundary, double blank line, single blank line, sentence end
3. The last `CHUNK_OVERLAP_CHARS` characters of each chunk are prepended to the next so boundary content is findable from either side

Each chunk becomes an independent Meilisearch document with a title like `Report [part 2/5]` and its own stable `doc_id`.

### Chunking constants

These live in `local_search_agent/core/constants.py` and can be adjusted:

| Constant | Default | Description |
|----------|---------|-------------|
| `CHUNK_MIN_CHARS` | `1000` | Documents shorter than this are not chunked |
| `CHUNK_TARGET_CHARS` | `8000` | Target chunk size |
| `CHUNK_MAX_CHARS` | `20000` | Hard maximum before a forced split |
| `CHUNK_OVERLAP_CHARS` | `500` | Characters of overlap between adjacent chunks |
| `TABLE_ROWS_PER_CHUNK` | `100` | Rows per chunk for table documents |
| `TABLE_LINE_RATIO` | `0.5` | Fraction of `\|` lines to classify as table document |

---

## File Discovery

The pipeline walks all configured `document_dirs` recursively. Rules:

- Hidden files and directories (starting with `.`) are skipped
- Files are processed in alphabetical order within each directory
- Multiple directories can be configured per workspace — they are walked in the order given

---

## Watch Mode (recommended)

Watch Mode reacts to filesystem events (`watchdog`) instead of polling on a fixed interval, so a changed file gets re-indexed within seconds rather than waiting for the next scheduled tick.

```bash
local-search watch start --workspace finance --dirs "C:\my_docs"
local-search watch status
local-search watch trigger --workspace finance
```

```python
framework.start_watch_mode()
framework.trigger_sync_now("finance")   # bypasses the debounce window
framework.stop_watch_mode()
```

Watch mode behaviour:
- One `watchdog` observer watches all of a workspace's `document_dirs` recursively
- A short debounce window (~2.5s) collapses bursts of filesystem events — a single save, or a folder copy with many files — into a single re-ingestion run
- Reuses the exact same delta logic and `IngestionPipeline` as a manual sync
- Whether semantic enrichment runs on a watch-triggered sync is controlled by `enrich_on_watch` (default `True`) — see [Configuration Guide](configuration.md#watch-mode) for details
- Sync state is written to SQLite before and after every run, same as the polling scheduler, so `local-search health` and sync history work identically regardless of which mechanism triggered the sync

See the [Configuration Guide](configuration.md#watch-mode) for the full Watch Mode vs. Scheduler comparison.

## Incremental Scheduler *(deprecated — use Watch Mode)*

The original polling-based scheduler runs ingestion automatically in the background on a fixed interval. Only changed files are re-indexed (delta logic applies). It is kept for backward compatibility; new code should prefer [Watch Mode](#watch-mode-recommended) above, which reacts to changes immediately instead of waiting for the next tick.

```bash
# Start the file server with scheduler enabled
local-search serve --workspace finance --scheduler --interval 15

# Or trigger a manual sync
local-search scheduler trigger --workspace finance
```

```python
framework.start_incremental_scheduler(interval_minutes=15)

# Later, force immediate sync
framework.trigger_sync_now("finance")
```

Scheduler behaviour:
- One APScheduler interval job per workspace
- `coalesce=True` — if the scheduler misses a tick (machine asleep, etc.), it runs once on wake, not multiple times
- `max_instances=1` — a sync job will not start if one is already running for the same workspace
- Sync state is written to SQLite before and after every run

Constructing `IncrementalSyncScheduler` directly now emits a `DeprecationWarning` pointing at `WorkspaceWatcher` / `framework.start_watch_mode()`.

---

## Progress Tracking

The UI shows a live progress bar during ingestion. If you use the Python API, you can get the same data via a callback:

```python
def on_progress(indexed, skipped, failed, total, current_file):
    if current_file == "__done__":
        print("Done!")
    else:
        print(f"{indexed}/{total} — {current_file}")

pipeline.run(force=False, progress_callback=on_progress)
```

---

## IngestStats

Every ingest call returns an `IngestStats` object:

```python
stats = framework.ingest_and_index()
print(stats)
# IngestStats(total=142, indexed=138, skipped=0, failed=4, duration=34.2s)

print(stats.total)        # files discovered
print(stats.indexed)      # chunks indexed into Meilisearch
print(stats.files_indexed) # source files successfully parsed
print(stats.skipped)      # files skipped (no change)
print(stats.failed)       # files that failed
print(stats.duration_s)   # wall-clock seconds
print(stats.errors)       # list of error messages for failed files
```

---

## Troubleshooting Ingestion

**Files are being skipped that I know changed**

The delta check uses the file's `modified_at` timestamp from the filesystem. If you copied files in a way that preserved the original timestamp (e.g. `robocopy /COPYALL`, `rsync -a`), the delta check won't see them as changed. Use `--force` or `--wipe` to override.

**PDF pages are garbled or missing**

Docling handles most PDF layouts well. For scanned PDFs, the pipeline automatically falls back to Tesseract (if installed) or RapidOCR. If results are poor, check whether the PDF is a scanned image-only document — install Tesseract for significantly better results on these files (see [PDF OCR Strategy](#pdf-ocr-strategy) above).

**Scanned PDF ingestion is very slow**

Without Tesseract, scanned PDFs fall back to RapidOCR which can take minutes per page on CPU. Install Tesseract to reduce this to ~1 second per page. See [Installing Tesseract](#installing-tesseract-optional-but-recommended) above.

**Large files take a long time**

Normal for scanned PDFs. Docling performs layout analysis which is CPU-intensive. For digitally-created PDFs (with a text layer), ingestion is much faster since OCR is skipped entirely. Consider running large initial ingestion jobs overnight.

**Memory errors on large Excel files**

openpyxl loads the entire sheet into memory. For very large `.xlsx` files (100k+ rows), consider splitting them first.
