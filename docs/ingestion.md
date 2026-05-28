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
Meilisearch batch indexing
     ↓
WorkspaceManager registration (SQLite)
```

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

If more than 60% of non-empty lines start with `|` (Markdown table rows), the document uses row-based chunking: `TABLE_ROWS_PER_CHUNK` rows per chunk, with the header row prepended to every chunk. No overlap — rows are structurally independent.

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
| `CHUNK_MIN_CHARS` | `2000` | Documents shorter than this are not chunked |
| `CHUNK_TARGET_CHARS` | `1500` | Target chunk size |
| `CHUNK_MAX_CHARS` | `3000` | Hard maximum before a forced split |
| `CHUNK_OVERLAP_CHARS` | `200` | Characters of overlap between adjacent chunks |
| `TABLE_ROWS_PER_CHUNK` | `50` | Rows per chunk for table documents |
| `TABLE_LINE_RATIO` | `0.6` | Fraction of `\|` lines to classify as table document |

---

## File Discovery

The pipeline walks all configured `document_dirs` recursively. Rules:

- Hidden files and directories (starting with `.`) are skipped
- Files are processed in alphabetical order within each directory
- Multiple directories can be configured per workspace — they are walked in the order given

---

## Incremental Scheduler

The scheduler runs ingestion automatically in the background on a fixed interval. Only changed files are re-indexed (delta logic applies).

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

Docling handles most PDF layouts well, but heavily image-based or scanned PDFs may produce poor results. Check that the PDF is searchable (text-layer present). Scanned PDFs without OCR will produce empty or near-empty text.

**Large files take a long time**

Normal. Docling performs layout analysis which is CPU-intensive. The pipeline processes files sequentially. For large initial loads, consider running ingestion overnight.

**Memory errors on large Excel files**

openpyxl loads the entire sheet into memory. For very large `.xlsx` files (100k+ rows), consider splitting them first.
