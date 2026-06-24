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
- Protect table integrity.  Tables in mixed-content documents must never be
  split mid-row; only pure-table documents use row-based chunking.

Chunking pipeline (applied in order)
--------------------------------------
1. Short document check
   If len(text) < CHUNK_MIN_CHARS → return as-is (single node).

2. Pure table / CSV detection
   If >=95% of non-empty lines start with '|' → row-based split
   (TABLE_ROWS_PER_CHUNK rows per chunk, header prepended to every chunk).
   Table-mode does NOT use overlap since rows are structurally independent.

3. Mixed content (all other documents)
   Split the text into semantic blocks at double-newline boundaries, then
   accumulate blocks into chunks respecting CHUNK_TARGET_CHARS / CHUNK_MAX_CHARS.
   Table blocks (>=90% pipe lines) are never split; prose blocks use a
   sliding-window with overlap.

All constants live in core.constants — adjust them there.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Optional

from local_search_agent.core import constants as _C
from local_search_agent.core.document_node import DocumentNode
from local_search_agent.core.key_manager import get_effective_constants

logger = logging.getLogger(__name__)

# Matches any Markdown heading line: # … through ###### …
_HEADING_RE = re.compile(r"^#{1,6}\s+.+$")

# Sentence-ending punctuation followed by whitespace
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")

# Break-point priority values (higher = stronger preference for cutting here)
_BP_HEADING = 4
_BP_DOUBLE_NL = 3
_BP_SINGLE_NL = 2
_BP_SENTENCE = 1

# Fraction of non-empty lines that must start with '|' for a document
# to be classified as a *pure* table (row-based chunking).
_PURE_TABLE_THRESHOLD = 0.95

# Fraction of non-empty lines in a block that must start with '|' for
# that block to be treated as a table block in mixed-content chunking.
_BLOCK_TABLE_THRESHOLD = 0.90


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

    Memory safety
    -------------
    If chunking fails with a MemoryError (low-RAM systems), the function
    retries with progressively halved CHUNK_TARGET_CHARS and CHUNK_MAX_CHARS.
    If all retries fail, the original node is returned as-is (un-chunked)
    so ingestion continues rather than crashing.
    """
    text = node.text

    if node.file_type == "md":
        logger.debug(
            "Skipping chunking for markdown file %r (always retained as single document)",
            node.title,
        )
        return [node]

    eff = get_effective_constants()
    chunk_min_chars = eff["CHUNK_MIN_CHARS"]
    rows_per_chunk = eff["TABLE_ROWS_PER_CHUNK"]
    overlap_chars = eff["CHUNK_OVERLAP_CHARS"]

    if len(text) < chunk_min_chars:
        logger.debug(
            "Skipping chunking for short document %r (%d chars)",
            node.title,
            len(text),
        )
        return [node]

    # --- Memory-safe retry loop ---
    target = eff["CHUNK_TARGET_CHARS"]
    max_c = eff["CHUNK_MAX_CHARS"]
    max_attempts = 4  # original + 3 halving attempts

    for attempt in range(max_attempts):
        try:
            if _detect_document_type(text) == "table":
                if attempt == 0:
                    logger.debug("Using pure-table (row-based) chunking for %r", node.title)
                raw_chunks = _chunk_table(text, rows_per_chunk=rows_per_chunk)
            else:
                if attempt == 0:
                    logger.debug("Using mixed-content chunking for %r", node.title)
                raw_chunks = _chunk_mixed_content(
                    text, target_chars=target, max_chars=max_c, overlap_chars=overlap_chars
                )

            raw_chunks = [c.strip() for c in raw_chunks if c.strip()]

            if len(raw_chunks) <= 1:
                return [node]

            total = len(raw_chunks)
            logger.info("Chunked %r into %d parts", node.title, total)

            return [
                _make_chunk_node(node, chunk_text, idx + 1, total)
                for idx, chunk_text in enumerate(raw_chunks)
            ]

        except MemoryError:
            target = target // 2
            max_c = max_c // 2
            logger.warning(
                "Chunking %r ran out of memory (attempt %d/%d). "
                "Retrying with reduced chunk size: target=%d max=%d.",
                node.title,
                attempt + 1,
                max_attempts,
                target,
                max_c,
            )

    # All retries failed — return original node un-chunked so ingestion continues
    logger.error(
        "Chunking %r failed after %d attempts due to MemoryError. "
        "Indexing as a single document. Consider reducing CHUNK_TARGET_CHARS "
        "in constants.py for low-memory environments.",
        node.title,
        max_attempts,
    )
    return [node]


# ---------------------------------------------------------------------------
# Document-type detection
# ---------------------------------------------------------------------------


def _detect_document_type(text: str) -> str:
    """
    Return 'table' if the document is a pure CSV/table (>=95% pipe lines),
    otherwise 'mixed'.

    A 95% threshold (rather than 100%) gracefully handles trailing blank
    lines or stray separator characters that would otherwise disqualify a
    genuine CSV-to-Markdown output.
    """
    non_empty = [ln for ln in text.splitlines() if ln.strip()]
    if not non_empty:
        return "mixed"
    table_lines = sum(1 for ln in non_empty if ln.strip().startswith("|"))
    return "table" if (table_lines / len(non_empty)) >= _PURE_TABLE_THRESHOLD else "mixed"


# ---------------------------------------------------------------------------
# Strategy 1 — Pure table / CSV row splitting
# ---------------------------------------------------------------------------


def _chunk_table(text: str, rows_per_chunk: int = _C.TABLE_ROWS_PER_CHUNK) -> list[str]:
    """
    Split a Markdown table into chunks of rows_per_chunk data rows.
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
    post_table = table_lines[last_table_idx:]
    table_lines = table_lines[:last_table_idx]

    header_rows: list[str] = []
    data_rows: list[str] = []
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
    pre_block = "\n".join(pre_table).strip()
    post_block = "\n".join(post_table).strip()

    chunks: list[str] = []
    for i in range(0, max(len(data_rows), 1), rows_per_chunk):
        row_slice = data_rows[i : i + rows_per_chunk]
        chunk = f"{header_block}\n" + "\n".join(row_slice)
        chunk = chunk.strip()
        if i == 0 and pre_block:
            chunk = f"{pre_block}\n\n{chunk}"
        if (i + rows_per_chunk) >= len(data_rows) and post_block:
            chunk = f"{chunk}\n\n{post_block}"
        chunks.append(chunk)

    return chunks if chunks else [text]


# ---------------------------------------------------------------------------
# Strategy 2 — Mixed content with table-boundary protection
# ---------------------------------------------------------------------------


def _split_into_semantic_blocks(text: str) -> list[str]:
    """Split text into blocks at two-or-more consecutive newlines."""
    raw_blocks = re.split(r"\n{2,}", text)
    return [b.strip() for b in raw_blocks if b.strip()]


def _is_table_block(block: str) -> bool:
    """True if >=90% of non-empty lines in the block look like Markdown table rows."""
    lines = [ln for ln in block.splitlines() if ln.strip()]
    if not lines:
        return False
    table_lines = sum(1 for ln in lines if ln.strip().startswith("|"))
    return (table_lines / len(lines)) >= _BLOCK_TABLE_THRESHOLD


def _get_overlap(text: str, overlap_chars: int = _C.CHUNK_OVERLAP_CHARS) -> str:
    """Return the last overlap_chars characters of text for use as an overlap prefix."""
    return text[-overlap_chars:] if len(text) > overlap_chars else text


def _chunk_mixed_content(
    text: str,
    target_chars: int = _C.CHUNK_TARGET_CHARS,
    max_chars: int = _C.CHUNK_MAX_CHARS,
    overlap_chars: int = _C.CHUNK_OVERLAP_CHARS,
) -> list[str]:
    """
    Chunk mixed content (prose + tables) by protecting table boundaries.

    Strategy:
    - Split the document into semantic blocks at double-newline boundaries.
    - Table blocks are never split; they are always kept intact.
    - Prose blocks use the existing sliding-window chunker with overlap.
    - Accumulate blocks into a current_chunk until it would exceed max_chars,
      then flush and start a new chunk.
    - The last CHUNK_OVERLAP_CHARS of a flushed prose section are prepended
      to the next chunk so boundary content stays findable from either side.
    """
    blocks = _split_into_semantic_blocks(text)

    if not blocks:
        return [text]

    result: list[str] = []
    current_parts: list[str] = []  # accumulated block strings for the current chunk
    current_len: int = 0
    overlap_prefix: str = ""

    def _flush(extra_block: str = "", is_table_flush: bool = False) -> None:
        nonlocal current_parts, current_len, overlap_prefix
        content = "\n\n".join(current_parts).strip()
        if extra_block:
            content = (content + "\n\n" + extra_block).strip() if content else extra_block.strip()
        full = (overlap_prefix + "\n\n" + content).strip() if overlap_prefix else content
        if full:
            result.append(full)
        # Tables are self-contained units — do not bleed table content into the
        # next chunk's overlap prefix.  Overlap is only meaningful for prose.
        overlap_prefix = (
            "" if is_table_flush else (_get_overlap(content, overlap_chars) if content else "")
        )
        current_parts = []
        current_len = 0

    for block in blocks:
        if _is_table_block(block):
            # Tables must stay intact and are never merged with other tables
            combined_len = current_len + len(block) + (2 if current_parts else 0)
            if current_parts and combined_len > max_chars:
                # Flush accumulated prose before the table
                _flush()
            # Flush the table itself (with any preceding prose) as its own chunk
            _flush(block, is_table_flush=True)
        else:
            # Prose block — try to accumulate
            addition = len(block) + (2 if current_parts else 0)
            if current_parts and (current_len + addition) > max_chars:
                # Current chunk is full — flush it, then try a sliding-window split
                # on the incoming block in case it's individually too large
                _flush()

            if len(block) > max_chars:
                # Single prose block larger than max_chars — use sliding-window split
                sub_chunks = _chunk_sliding(
                    block,
                    target_chars=target_chars,
                    max_chars=max_chars,
                    overlap_chars=overlap_chars,
                )
                for i, sc in enumerate(sub_chunks):
                    if i < len(sub_chunks) - 1:
                        full = (overlap_prefix + "\n\n" + sc).strip() if overlap_prefix else sc
                        result.append(full)
                        overlap_prefix = _get_overlap(sc, overlap_chars)
                    else:
                        # Last sub-chunk: carry it forward for accumulation.
                        # Reset overlap_prefix — the sub-chunk already embeds overlap
                        # from _chunk_sliding; we must not prepend it again in _flush.
                        overlap_prefix = ""
                        current_parts = [sc]
                        current_len = len(sc)
            else:
                current_parts.append(block)
                current_len += addition

    # Flush any remaining content
    if current_parts:
        _flush()

    return result if result else [text]


# ---------------------------------------------------------------------------
# Strategy 3 — Sliding-window with overlap  (prose blocks / fallback)
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
        end = cursor + len(line)
        stripped = line.rstrip("\n\r")
        is_blank = stripped.strip() == ""
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


def _chunk_sliding(
    text: str,
    target_chars: int = _C.CHUNK_TARGET_CHARS,
    max_chars: int = _C.CHUNK_MAX_CHARS,
    overlap_chars: int = _C.CHUNK_OVERLAP_CHARS,
) -> list[str]:
    """
    Accumulate text into chunks using a sliding window with overlap.

    Algorithm:
      - Once the accumulated chunk reaches target_chars, scan forward
        for the highest-priority break point within the next
        (max_chars - target_chars) characters and cut there.
      - If no break point exists before max_chars, force-cut there.
      - Prepend the last overlap_chars characters of the previous chunk's raw text
        to the next chunk so boundary content is findable from either side.
    """
    bp_map: dict[int, int] = dict(_find_break_points(text))
    bp_positions: list[int] = sorted(bp_map)

    chunks: list[str] = []
    start = 0
    overlap_prefix = ""

    while start < len(text):
        target_end = start + target_chars
        hard_end = start + max_chars

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
                fallback_pos = pos  # keep updating — want the last one
            else:
                # In the target → hard window: pick highest priority
                if pri > best_pri:
                    best_pri = pri
                    best_pos = pos

        if best_pos is None:
            # Nothing in the target→hard window; use last break before target,
            # or force-cut at hard_end if there's nothing at all.
            best_pos = fallback_pos if fallback_pos is not None else hard_end

        chunk_raw = text[start:best_pos]
        chunk_full = (overlap_prefix + chunk_raw).strip()
        if chunk_full:
            chunks.append(chunk_full)

        # Overlap: last N chars of the raw (non-prefixed) section
        overlap_prefix = chunk_raw[-overlap_chars:] if len(chunk_raw) > overlap_chars else chunk_raw
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
