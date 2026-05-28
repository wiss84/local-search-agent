"""
Document chunker for the Local Search Agent ingestion pipeline.

Splits a parsed DocumentNode into overlapping chunks when the document text
exceeds CHUNK_MIN_CHARS.  Each chunk becomes an independent Meilisearch
document with its own doc_id and a descriptive title suffix.

Design goals
------------
- Keep logical units whole.  A recipe, a section, a subsection should not be
  split across chunks if it fits within CHUNK_TARGET_CHARS.
- Be format-agnostic.  Do not assume headings exist or that the document
  follows any particular structure.
- Provide overlap so that content near a chunk boundary is findable from
  either side.

Chunking pipeline (applied in order)
--------------------------------------
1. Short document check
   If len(text) < CHUNK_MIN_CHARS → return as-is (single node).

2. Table / CSV detection
   If >TABLE_LINE_RATIO of non-empty lines start with '|' → row-based split
   (TABLE_ROWS_PER_CHUNK rows per chunk, header prepended to every chunk).
   Table-mode does NOT use overlap since rows are structurally independent.

3. Sliding-window accumulation (all other documents)
   Collect natural break points in priority order:
     a. Blank line immediately after a Markdown heading (## / ### / etc.)  [priority 4]
     b. Double blank line  (strong paragraph boundary)                     [priority 3]
     c. Single blank line  (weak paragraph boundary)                       [priority 2]
     d. Sentence-ending punctuation ('. ' / '! ' / '? ')                  [priority 1]
   Accumulate text until the chunk reaches CHUNK_TARGET_CHARS, then cut at
   the next available break point.  If a block exceeds CHUNK_MAX_CHARS with
   no break point, force-split there.
   The last CHUNK_OVERLAP_CHARS characters of every chunk are prepended to
   the next chunk so that content near boundaries stays findable from both
   sides.

All constants live in core.constants — adjust them there.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Optional

from local_search_agent.core.constants import (
    CHUNK_MAX_CHARS,
    CHUNK_MIN_CHARS,
    CHUNK_OVERLAP_CHARS,
    CHUNK_TARGET_CHARS,
    TABLE_LINE_RATIO,
    TABLE_ROWS_PER_CHUNK,
)
from local_search_agent.core.document_node import DocumentNode

logger = logging.getLogger(__name__)

# Matches any Markdown heading line: # … through ###### …
_HEADING_RE = re.compile(r"^#{1,6}\s+.+$")

# Sentence-ending punctuation followed by whitespace
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")

# Break-point priority values (higher = stronger preference for cutting here)
_BP_HEADING   = 4
_BP_DOUBLE_NL = 3
_BP_SINGLE_NL = 2
_BP_SENTENCE  = 1


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def chunk_document(node: DocumentNode) -> list[DocumentNode]:
    """
    Split a DocumentNode into overlapping chunks when its text exceeds
    CHUNK_MIN_CHARS.

    Returns
    -------
    A list of DocumentNodes.  If no chunking is needed, returns [node]
    unchanged.  Otherwise returns only the chunks (original node excluded).
    """
    text = node.text

    if len(text) < CHUNK_MIN_CHARS:
        logger.debug(
            "Skipping chunking for short document %r (%d chars)",
            node.title, len(text),
        )
        return [node]

    if _is_table_document(text):
        logger.debug("Using table chunking for %r", node.title)
        raw_chunks = _chunk_table(text)
    else:
        logger.debug("Using sliding-window chunking for %r", node.title)
        raw_chunks = _chunk_sliding(text)

    raw_chunks = [c.strip() for c in raw_chunks if c.strip()]

    if len(raw_chunks) <= 1:
        return [node]

    total = len(raw_chunks)
    logger.info("Chunked %r into %d parts", node.title, total)

    return [
        _make_chunk_node(node, chunk_text, idx + 1, total)
        for idx, chunk_text in enumerate(raw_chunks)
    ]


# ---------------------------------------------------------------------------
# Strategy 1 — Table / CSV row splitting
# ---------------------------------------------------------------------------

def _is_table_document(text: str) -> bool:
    """Return True if the majority of non-empty lines look like Markdown table rows."""
    non_empty = [ln for ln in text.splitlines() if ln.strip()]
    if not non_empty:
        return False
    table_lines = sum(1 for ln in non_empty if ln.strip().startswith("|"))
    return (table_lines / len(non_empty)) >= TABLE_LINE_RATIO


def _chunk_table(text: str) -> list[str]:
    """
    Split a Markdown table into chunks of TABLE_ROWS_PER_CHUNK data rows.
    Header/separator rows are prepended to every chunk.
    Prose before/after the table is attached to the first/last chunk.
    """
    lines = text.splitlines()

    pre_table: list[str] = []
    table_start = 0
    for i, ln in enumerate(lines):
        if ln.strip().startswith("|"):
            table_start = i
            break
        pre_table.append(ln)

    table_lines = lines[table_start:]

    # Find end of table
    last_table_idx = len(table_lines)
    for i in range(len(table_lines) - 1, -1, -1):
        if table_lines[i].strip().startswith("|"):
            last_table_idx = i + 1
            break
    post_table  = table_lines[last_table_idx:]
    table_lines = table_lines[:last_table_idx]

    header_rows: list[str] = []
    data_rows:   list[str] = []
    separator_found = False

    for ln in table_lines:
        stripped = ln.strip()
        if not stripped:
            continue
        if not header_rows:
            header_rows.append(ln)
            continue
        if not separator_found and re.match(r"^\|[\s\-:|]+\|", stripped):
            header_rows.append(ln)
            separator_found = True
            continue
        data_rows.append(ln)

    if not header_rows:
        return [text]

    header_block = "\n".join(header_rows)
    pre_block    = "\n".join(pre_table).strip()
    post_block   = "\n".join(post_table).strip()

    chunks: list[str] = []
    for i in range(0, max(len(data_rows), 1), TABLE_ROWS_PER_CHUNK):
        row_slice = data_rows[i: i + TABLE_ROWS_PER_CHUNK]
        chunk = f"{header_block}\n" + "\n".join(row_slice)
        chunk = chunk.strip()
        if i == 0 and pre_block:
            chunk = f"{pre_block}\n\n{chunk}"
        if (i + TABLE_ROWS_PER_CHUNK) >= len(data_rows) and post_block:
            chunk = f"{chunk}\n\n{post_block}"
        chunks.append(chunk)

    return chunks if chunks else [text]


# ---------------------------------------------------------------------------
# Strategy 2 — Sliding-window with overlap  (all non-table documents)
# ---------------------------------------------------------------------------

def _find_break_points(text: str) -> list[tuple[int, int]]:
    """
    Scan *text* and return a sorted list of (position, priority) tuples.

    A position is the character index *after* which the text can cleanly be
    split.  Higher priority = more semantically meaningful boundary.
    """
    bps: dict[int, int] = {}

    def add(pos: int, priority: int) -> None:
        if 0 < pos < len(text):
            bps[pos] = max(bps.get(pos, 0), priority)

    lines = text.splitlines(keepends=True)
    cursor = 0
    prev_was_heading = False

    for line in lines:
        end    = cursor + len(line)
        stripped = line.rstrip("\n\r")
        is_blank   = stripped.strip() == ""
        is_heading = bool(_HEADING_RE.match(stripped))

        if is_blank:
            if prev_was_heading:
                # Blank line right after a heading — strongest natural break
                add(end, _BP_HEADING)
            else:
                # Double blank line vs single blank line
                if cursor > 0 and text[cursor - 1] == "\n":
                    add(end, _BP_DOUBLE_NL)
                else:
                    add(end, _BP_SINGLE_NL)

        prev_was_heading = is_heading
        cursor = end

    # Sentence boundaries fill the gaps between structural breaks
    for m in _SENTENCE_END_RE.finditer(text):
        add(m.end(), _BP_SENTENCE)

    return sorted(bps.items())  # sorted by position ascending


def _chunk_sliding(text: str) -> list[str]:
    """
    Accumulate text into chunks using a sliding window with overlap.

    Algorithm:
      - Once the accumulated chunk reaches CHUNK_TARGET_CHARS, scan forward
        for the highest-priority break point within the next
        (CHUNK_MAX_CHARS - CHUNK_TARGET_CHARS) characters and cut there.
      - If no break point exists before CHUNK_MAX_CHARS, force-cut there.
      - Prepend the last CHUNK_OVERLAP_CHARS of the previous chunk's raw text
        to the next chunk so boundary content is findable from either side.
    """
    bp_map: dict[int, int] = dict(_find_break_points(text))
    bp_positions: list[int] = sorted(bp_map)

    chunks: list[str] = []
    start          = 0       # current chunk start in original text
    overlap_prefix = ""      # tail of previous chunk

    while start < len(text):
        target_end = start + CHUNK_TARGET_CHARS
        hard_end   = start + CHUNK_MAX_CHARS

        if target_end >= len(text):
            # Everything remaining fits — take it all
            chunk = (overlap_prefix + text[start:]).strip()
            if chunk:
                chunks.append(chunk)
            break

        # Find the best break point in (start, hard_end].
        # Priority: highest priority in [target_end, hard_end] wins.
        # Fallback: last known break point before target_end.
        best_pos: Optional[int] = None
        best_pri = -1
        fallback_pos: Optional[int] = None  # last break before target_end

        for pos in bp_positions:
            if pos <= start:
                continue
            if pos > hard_end:
                break
            pri = bp_map[pos]
            if pos < target_end:
                fallback_pos = pos          # keep updating — want the last one
            else:
                # In the target → hard window: pick highest priority
                if pri > best_pri:
                    best_pri = pri
                    best_pos = pos

        if best_pos is None:
            # Nothing in the target→hard window; use last break before target,
            # or force-cut at hard_end if there's nothing at all.
            best_pos = fallback_pos if fallback_pos is not None else hard_end

        chunk_raw  = text[start:best_pos]
        chunk_full = (overlap_prefix + chunk_raw).strip()
        if chunk_full:
            chunks.append(chunk_full)

        # Overlap: last N chars of the raw (non-prefixed) section
        overlap_prefix = (
            chunk_raw[-CHUNK_OVERLAP_CHARS:]
            if len(chunk_raw) > CHUNK_OVERLAP_CHARS
            else chunk_raw
        )
        start = best_pos

    return chunks if chunks else [text]


# ---------------------------------------------------------------------------
# DocumentNode factory helpers
# ---------------------------------------------------------------------------

def _chunk_doc_id(source_path: str, index: int) -> str:
    key = f"{source_path}:chunk:{index}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _make_chunk_node(
    original: DocumentNode,
    chunk_text: str,
    index: int,
    total: int,
) -> DocumentNode:
    return DocumentNode(
        doc_id=_chunk_doc_id(original.source_path, index),
        title=f"{original.title} [part {index}/{total}]",
        text=chunk_text,
        file_type=original.file_type,
        source_path=original.source_path,
        folder_path=original.folder_path,
        workspace=original.workspace,
        modified_at=original.modified_at,
        indexed_at=original.indexed_at,
        concepts=list(original.concepts),
        synonyms=list(original.synonyms),
    )
