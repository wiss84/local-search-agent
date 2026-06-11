"""
Production test suite for the ingestion chunker.

Tests are organised around observable behaviours a real document could trigger,
not around internal implementation details.  Every test uses document content
that resembles actual user files: paragraph-separated prose, Markdown tables
with header + separator + data rows, mixed reports, CSVs.

Fixture helpers
---------------
_para(n)         — n realistic paragraphs (~500 chars each), separated by
                   double newlines so _split_into_semantic_blocks sees them
                   as distinct blocks.
_big_para(chars) — a single very long paragraph (no internal double newlines),
                   used to exercise the large-block sliding-window path.
_table(rows, id) — a Markdown table with a unique marker column so individual
                   tables can be identified across chunks.
_report(before, after, table_rows, para_multiplier)
                 — a realistic mixed document: prose, one embedded table,
                   more prose.
"""

from __future__ import annotations

import re

from local_search_agent.core.constants import (
    CHUNK_MAX_CHARS,
    CHUNK_OVERLAP_CHARS,
    CHUNK_TARGET_CHARS,
    TABLE_ROWS_PER_CHUNK,
)
from local_search_agent.core.document_node import DocumentNode
from local_search_agent.ingestion.chunker import (
    _chunk_mixed_content,
    _chunk_sliding,
    _chunk_table,
    _detect_document_type,
    _is_table_block,
    _split_into_semantic_blocks,
    chunk_document,
)

# ---------------------------------------------------------------------------
# Realistic sentence used to build all prose fixtures
# ---------------------------------------------------------------------------
_SENTENCE = (
    "The quarterly results show strong performance across all divisions. "
    "Revenue grew by 12 percent year over year, driven primarily by the "
    "enterprise segment. Operating margins improved as headcount remained "
    "flat while productivity increased. "
)
assert len(_SENTENCE) == 239, "sanity-check sentence length"


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _para(n: int = 1) -> str:
    """Return n paragraphs (~500 chars each) joined by double newlines."""
    single = (_SENTENCE * 2).strip()
    return "\n\n".join([single] * n)


def _big_para(chars: int) -> str:
    """Single prose block (no double newlines) of exactly *chars* characters."""
    base = _SENTENCE * ((chars // len(_SENTENCE)) + 2)
    return base[:chars]


def _table(rows: int, marker: str = "T") -> str:
    """
    Markdown table with *rows* data rows.
    The marker is embedded so tables can be told apart in multi-table docs.
    """
    header = f"| {marker}_Quarter | {marker}_Revenue | {marker}_Growth |"
    sep = "| --- | --- | --- |"
    data = "\n".join(f"| {marker}Q{i:03d} | ${i * 100}M | {i}% |" for i in range(1, rows + 1))
    return f"{header}\n{sep}\n{data}"


def _report(
    before: int = 3,
    after: int = 3,
    table_rows: int = 10,
    para_multiplier: int = 1,
    marker: str = "T",
) -> str:
    """Prose before + embedded table + prose after, double-newline separated."""
    p = _para(before * para_multiplier)
    t = _table(table_rows, marker)
    a = _para(after * para_multiplier)
    return f"{p}\n\n{t}\n\n{a}"


def _make_node(
    text: str, tmp_path, file_type: str = "txt", modified_at: str = "2024-01-01T00:00:00"
) -> DocumentNode:
    src = str(tmp_path / "doc.txt")
    return DocumentNode(
        doc_id="test-id",
        title="Test Document",
        text=text,
        file_type=file_type,
        source_path=src,
        folder_path=str(tmp_path),
        workspace="test",
        modified_at=modified_at,
    )


# ---------------------------------------------------------------------------
# Counting helpers
# ---------------------------------------------------------------------------


def _data_row_count(text: str, marker: str = "T") -> int:
    """Count data rows (not header/separator) for the given table marker."""
    return sum(1 for line in text.splitlines() if re.match(rf"\|\s*{re.escape(marker)}Q\d", line))


def _total_data_rows(chunks: list[str], marker: str = "T") -> int:
    return sum(_data_row_count(c, marker) for c in chunks)


def _table_chunk_count(chunks: list[str], marker: str = "T") -> int:
    """Number of chunks that contain the header row of the given table."""
    return sum(1 for c in chunks if f"| {marker}_Quarter |" in c)


# ===========================================================================
# 1.  _detect_document_type
# ===========================================================================


class TestDetectDocumentType:
    def test_pure_markdown_table_detected_as_table(self):
        assert _detect_document_type(_table(50)) == "table"

    def test_pure_csv_with_250_rows_detected_as_table(self):
        header = "| ID | Name | Value |"
        sep = "| --- | --- | --- |"
        rows = "\n".join(f"| {i} | Item{i} | {i * 10} |" for i in range(250))
        assert _detect_document_type(f"{header}\n{sep}\n{rows}") == "table"

    def test_csv_with_single_title_line_still_classified_as_table(self):
        # 1 prose title + 52 pipe lines → 52/53 ≈ 0.981 ≥ 0.95 threshold
        header = "| ID | Name |"
        sep = "| --- | --- |"
        rows = "\n".join(f"| {i} | Item{i} |" for i in range(50))
        csv = f"Sales Report 2024\n{header}\n{sep}\n{rows}"
        assert _detect_document_type(csv) == "table"

    def test_trailing_blank_lines_do_not_break_table_detection(self):
        # Blank lines are excluded from ratio calculation (non-empty filter)
        csv = _table(50) + "\n\n\n"
        assert _detect_document_type(csv) == "table"

    def test_three_prose_lines_in_csv_falls_below_threshold(self):
        # 3 prose + 52 pipe → 52/55 ≈ 0.945 < 0.95 → mixed
        prose = "Summary line one.\nSummary line two.\nSummary line three."
        header = "| ID | Name |"
        sep = "| --- | --- |"
        rows = "\n".join(f"| {i} | Item{i} |" for i in range(50))
        doc = f"{prose}\n{header}\n{sep}\n{rows}"
        assert _detect_document_type(doc) == "mixed"

    def test_prose_only_document_detected_as_mixed(self):
        assert _detect_document_type(_para(5)) == "mixed"

    def test_empty_string_detected_as_mixed(self):
        assert _detect_document_type("") == "mixed"


# ===========================================================================
# 2.  _is_table_block
# ===========================================================================


class TestIsTableBlock:
    def test_pure_markdown_table_is_table_block(self):
        block = "| Name | Value |\n| --- | --- |\n| Foo | 1 |\n| Bar | 2 |"
        assert _is_table_block(block) is True

    def test_pure_prose_paragraph_is_not_table_block(self):
        assert _is_table_block(_para(1)) is False

    def test_empty_string_is_not_table_block(self):
        assert _is_table_block("") is False

    def test_block_with_majority_prose_is_not_table_block(self):
        mixed = "Intro line.\nAnother line.\nThird line.\n" + _table(2)
        assert _is_table_block(mixed) is False

    def test_ratio_boundary_at_90_percent(self):
        # 1 non-pipe line + 9 pipe lines → exactly 90% → True
        block = "non-table line\n" + "\n".join(f"| {i} | {i} |" for i in range(9))
        lines = [line for line in block.splitlines() if line.strip()]
        ratio = sum(1 for line in lines if line.strip().startswith("|")) / len(lines)
        expected = ratio >= 0.90
        assert _is_table_block(block) is expected


# ===========================================================================
# 3.  _split_into_semantic_blocks
# ===========================================================================


class TestSplitIntoSemanticBlocks:
    def test_three_paragraphs_produce_three_blocks(self):
        doc = "Para one.\n\nPara two.\n\nPara three."
        assert len(_split_into_semantic_blocks(doc)) == 3

    def test_multiple_blank_lines_treated_as_single_separator(self):
        doc = "A.\n\n\n\nB."
        assert len(_split_into_semantic_blocks(doc)) == 2

    def test_single_newlines_within_paragraph_not_split(self):
        doc = "Line one.\nLine two.\nLine three."
        assert len(_split_into_semantic_blocks(doc)) == 1

    def test_empty_string_returns_empty_list(self):
        assert _split_into_semantic_blocks("") == []

    def test_blocks_are_stripped_of_surrounding_whitespace(self):
        doc = "  Para one.  \n\n  Para two.  "
        blocks = _split_into_semantic_blocks(doc)
        assert blocks[0] == "Para one."
        assert blocks[1] == "Para two."


# ===========================================================================
# 4.  _chunk_table — pure-table / CSV row-based splitting
# ===========================================================================


class TestChunkTable:
    def test_small_table_fits_in_one_chunk(self):
        assert len(_chunk_table(_table(5))) == 1

    def test_250_rows_produces_three_chunks(self):
        # ceil(250 / TABLE_ROWS_PER_CHUNK) = 3
        header = "| ID | Name | Value |"
        sep = "| --- | --- | --- |"
        rows = "\n".join(f"| {i:03d} | Item{i} | {i * 10} |" for i in range(250))
        assert len(_chunk_table(f"{header}\n{sep}\n{rows}")) == 3

    def test_header_row_prepended_to_every_chunk(self):
        header = "| ID | Name | Value |"
        sep = "| --- | --- | --- |"
        rows = "\n".join(f"| {i:03d} | Item{i} | {i * 10} |" for i in range(250))
        csv = f"{header}\n{sep}\n{rows}"
        for i, chunk in enumerate(_chunk_table(csv)):
            assert "| ID |" in chunk, f"chunk {i} missing header"

    def test_each_chunk_respects_rows_per_chunk_limit(self):
        header = "| ID | Name |"
        sep = "| --- | --- |"
        rows = "\n".join(f"| {i:03d} | Item{i} |" for i in range(250))
        csv = f"{header}\n{sep}\n{rows}"
        for i, chunk in enumerate(_chunk_table(csv)):
            data = [line for line in chunk.splitlines() if re.match(r"\|\s*\d{3}", line)]
            assert len(data) <= TABLE_ROWS_PER_CHUNK, (
                f"chunk {i} has {len(data)} rows > limit {TABLE_ROWS_PER_CHUNK}"
            )

    def test_all_data_rows_present_across_chunks(self):
        n = 250
        header = "| ID | Name |"
        sep = "| --- | --- |"
        rows = "\n".join(f"| {i:03d} | Item{i} |" for i in range(n))
        csv = f"{header}\n{sep}\n{rows}"
        chunks = _chunk_table(csv)
        found = sum(
            sum(1 for line in c.splitlines() if re.match(r"\|\s*\d{3}", line)) for c in chunks
        )
        assert found == n

    def test_prose_before_table_attached_to_first_chunk(self):
        prose = "## Sales Report\n\nThis table summarises quarterly sales."
        csv = prose + "\n\n" + _table(10)
        assert "Sales Report" in _chunk_table(csv)[0]

    def test_prose_after_table_attached_to_last_chunk(self):
        footer = "Data sourced from the finance team."
        csv = _table(10) + "\n\n" + footer
        assert footer in _chunk_table(csv)[-1]

    def test_table_with_only_header_and_separator_returns_single_chunk(self):
        text = "| A | B |\n| --- | --- |"
        chunks = _chunk_table(text)
        assert len(chunks) == 1


# ===========================================================================
# 5.  _chunk_mixed_content — table integrity
# ===========================================================================


class TestMixedContentTableIntegrity:
    def test_table_appears_in_exactly_one_chunk(self):
        """An embedded table must not bleed into adjacent chunks via overlap."""
        chunks = _chunk_mixed_content(_report(before=2, after=2, table_rows=5))
        assert _table_chunk_count(chunks) == 1, (
            f"table header found in {_table_chunk_count(chunks)} chunks, expected 1"
        )

    def test_table_data_rows_not_duplicated(self):
        """Overlap must never copy table data rows into a neighbouring chunk."""
        chunks = _chunk_mixed_content(_report(before=2, after=2, table_rows=5))
        assert _total_data_rows(chunks) == 5, (
            f"expected 5 data rows total, got {_total_data_rows(chunks)}"
        )

    def test_table_intact_when_large_prose_surrounds_it(self):
        """With large prose sections (> TARGET) on both sides, table must stay whole."""
        # para_multiplier=35 → each prose section ≈ 35 × 500 = 17 500 chars > TARGET
        doc = _report(before=1, after=1, table_rows=5, para_multiplier=35)
        chunks = _chunk_mixed_content(doc)
        assert _table_chunk_count(chunks) == 1
        assert _total_data_rows(chunks) == 5

    def test_two_tables_each_in_separate_chunk(self):
        """Two tables in one doc must not be merged into a single chunk."""
        p = _para(2)
        t1 = _table(5, "T1")
        t2 = _table(5, "T2")
        doc = f"{p}\n\n{t1}\n\n{p}\n\n{t2}\n\n{p}"
        chunks = _chunk_mixed_content(doc)
        assert _table_chunk_count(chunks, "T1") == 1
        assert _table_chunk_count(chunks, "T2") == 1
        assert not any("| T1_Quarter |" in c and "| T2_Quarter |" in c for c in chunks), (
            "T1 and T2 must not share a chunk"
        )

    def test_three_tables_none_split_or_merged(self):
        sections = "\n\n".join(f"{_para(2)}\n\n{_table(5, f'T{i}')}" for i in range(1, 4))
        chunks = _chunk_mixed_content(sections)
        for i in range(1, 4):
            assert _table_chunk_count(chunks, f"T{i}") == 1, (
                f"T{i} found in {_table_chunk_count(chunks, f'T{i}')} chunks"
            )
            assert _total_data_rows(chunks, f"T{i}") == 5, (
                f"T{i}: expected 5 rows, got {_total_data_rows(chunks, f'T{i}')}"
            )

    def test_prose_flushed_before_table_when_combined_exceeds_max(self):
        """When accumulated prose + table would exceed CHUNK_MAX_CHARS, prose is
        flushed first so the table gets its own clean chunk."""
        prose = _big_para(CHUNK_MAX_CHARS - 1000)
        doc = prose + "\n\n" + _table(20)
        chunks = _chunk_mixed_content(doc)
        assert _table_chunk_count(chunks) == 1
        assert _total_data_rows(chunks) == 20


# ===========================================================================
# 6.  _chunk_mixed_content — prose overlap
# ===========================================================================


class TestMixedContentProseOverlap:
    def test_overlap_present_between_adjacent_prose_chunks(self):
        """The tail of chunk N must appear verbatim at the start of chunk N+1."""
        # 18 paragraphs × ~500 chars ≈ 21 500 chars > CHUNK_MAX_CHARS
        doc = _para(50)
        chunks = _chunk_mixed_content(doc)
        assert len(chunks) >= 2, "doc should produce at least 2 chunks"
        for i in range(len(chunks) - 1):
            tail_sample = chunks[i][-CHUNK_OVERLAP_CHARS:][:60]
            assert tail_sample.strip() in chunks[i + 1], (
                f"no overlap detected between chunk {i} and chunk {i + 1}"
            )

    def test_no_table_content_bleeds_into_post_table_prose_chunk(self):
        """Prose chunks that follow a table must not start with table rows."""
        doc = _report(before=2, after=2, table_rows=5)
        chunks = _chunk_mixed_content(doc)
        table_idx = next(i for i, c in enumerate(chunks) if "| T_Quarter |" in c)
        for post in chunks[table_idx + 1 :]:
            assert "| T_Quarter |" not in post, (
                "table header bled into a post-table chunk via overlap"
            )

    def test_no_chunk_exceeds_max_plus_overlap_tolerance(self):
        """No chunk may exceed CHUNK_MAX_CHARS + CHUNK_OVERLAP_CHARS + 100."""
        doc = _para(25)
        tol = CHUNK_MAX_CHARS + CHUNK_OVERLAP_CHARS + 100
        for i, chunk in enumerate(_chunk_mixed_content(doc)):
            assert len(chunk) <= tol, f"chunk {i} is {len(chunk)} chars, exceeds tolerance {tol}"


# ===========================================================================
# 7.  _chunk_sliding — break-point selection
# ===========================================================================


class TestChunkSliding:
    def test_heading_used_as_break_point_not_split(self):
        """A Markdown heading must not be split across chunks."""
        # section_text < TARGET so heading falls in the first chunk's window;
        # the cut lands just after the heading → heading is intact in one chunk.
        section = (_SENTENCE * 30).strip()  # ~7 169 chars < TARGET(8 000)
        doc = section + "\n\n## New Section\n\n" + section
        chunks = _chunk_sliding(doc)
        assert len(chunks) == 2
        heading_chunks = [c for c in chunks if "## New Section" in c]
        assert len(heading_chunks) == 1, "heading must appear intact in exactly one chunk"

    def test_force_cut_when_no_break_points_exist(self):
        """A solid block of non-breaking characters must still be split."""
        text = "A" * (CHUNK_MAX_CHARS * 2 + 500)
        chunks = _chunk_sliding(text)
        tol = CHUNK_MAX_CHARS + CHUNK_OVERLAP_CHARS + 100
        assert len(chunks) >= 2
        for i, c in enumerate(chunks):
            assert len(c) <= tol, f"force-cut chunk {i} is {len(c)} > {tol}"

    def test_large_single_prose_block_split_within_size_bounds(self):
        """A prose block > CHUNK_MAX_CHARS split via mixed-content path stays bounded."""
        block = _big_para(CHUNK_MAX_CHARS * 2)
        tol = CHUNK_MAX_CHARS + CHUNK_OVERLAP_CHARS + 100
        chunks = _chunk_mixed_content(block)
        assert len(chunks) >= 2
        for i, c in enumerate(chunks):
            assert len(c) <= tol, f"sub-chunk {i} is {len(c)} > {tol}"

    def test_two_prose_sections_exceeding_max_produce_two_chunks(self):
        """Two large prose blocks whose combined length exceeds CHUNK_MAX_CHARS
        must produce at least two chunks, each within the size tolerance."""
        # _SENTENCE = 239 chars; 42 × 239 = 10 038 > CHUNK_MAX_CHARS // 2
        sec_len = len(_SENTENCE) * 42  # 10 038 chars; combined ≈ 20 078 > MAX
        doc = _big_para(sec_len) + "\n\n" + _big_para(sec_len)
        chunks = _chunk_mixed_content(doc)
        tol = CHUNK_MAX_CHARS + CHUNK_OVERLAP_CHARS + 100
        assert len(chunks) >= 2, (
            f"expected >= 2 chunks for combined {sec_len * 2} chars, got {len(chunks)}"
        )
        for i, c in enumerate(chunks):
            assert len(c) <= tol, f"chunk {i} len={len(c)} > {tol}"

    def test_sentence_boundary_preferred_over_mid_word_cut(self):
        """When sentences exist, chunks should end at sentence boundaries."""
        text = _big_para(CHUNK_TARGET_CHARS + 500)
        chunks = _chunk_sliding(text)
        assert len(chunks) >= 2
        for i, chunk in enumerate(chunks[:-1]):
            stripped = chunk.rstrip()
            assert re.search(r"[.!?]$", stripped), (
                f"chunk {i} ends mid-sentence: ...{stripped[-40:]!r}"
            )


# ===========================================================================
# 8.  chunk_document — public entry-point integration
# ===========================================================================


class TestChunkDocumentPublicAPI:
    def test_short_document_returned_as_single_node(self, tmp_path):
        text = "This is a short document. " * 5  # well under CHUNK_MIN_CHARS
        node = _make_node(text, tmp_path)
        assert chunk_document(node) == [node]

    def test_markdown_file_never_chunked_regardless_of_length(self, tmp_path):
        text = _para(50)  # large enough to normally trigger chunking
        node = _make_node(text, tmp_path, file_type="md")
        assert chunk_document(node) == [node]

    def test_pure_csv_uses_row_based_chunking(self, tmp_path):
        header = "| ID | Name | Value |"
        sep = "| --- | --- | --- |"
        rows = "\n".join(f"| {i:03d} | Item{i} | {i * 10} |" for i in range(250))
        node = _make_node(f"{header}\n{sep}\n{rows}", tmp_path)
        chunks = chunk_document(node)
        assert len(chunks) == 3  # ceil(250 / 100)
        for chunk in chunks:
            assert "| ID |" in chunk.text

    def test_large_prose_document_produces_multiple_chunks(self, tmp_path):
        node = _make_node(_para(50), tmp_path)
        assert len(chunk_document(node)) >= 2

    def test_chunk_titles_carry_part_n_of_total_suffix(self, tmp_path):
        node = _make_node(_para(50), tmp_path)
        chunks = chunk_document(node)
        total = len(chunks)
        for i, chunk in enumerate(chunks, 1):
            assert f"[part {i}/{total}]" in chunk.title, (
                f"chunk {i} title missing suffix: {chunk.title!r}"
            )

    def test_chunk_doc_ids_are_all_unique(self, tmp_path):
        chunks = chunk_document(_make_node(_para(50), tmp_path))
        ids = [c.doc_id for c in chunks]
        assert len(ids) == len(set(ids)), "duplicate doc_ids detected"

    def test_chunk_source_path_preserved(self, tmp_path):
        node = _make_node(_para(50), tmp_path)
        for chunk in chunk_document(node):
            assert chunk.source_path == node.source_path

    def test_chunk_workspace_preserved(self, tmp_path):
        node = _make_node(_para(50), tmp_path)
        for chunk in chunk_document(node):
            assert chunk.workspace == node.workspace

    def test_mixed_report_all_table_rows_present_exactly_once(self, tmp_path):
        """End-to-end: a report with a 20-row table must preserve all rows."""
        doc = _report(before=3, after=3, table_rows=20, para_multiplier=10)
        node = _make_node(doc, tmp_path)
        chunks = chunk_document(node)
        texts = [c.text for c in chunks]
        assert _total_data_rows(texts) == 20, f"expected 20 rows, got {_total_data_rows(texts)}"
        assert _table_chunk_count(texts) == 1, "table must appear in exactly 1 chunk"

    def test_memory_error_falls_back_to_original_node(self, tmp_path, monkeypatch):
        """If all chunking attempts raise MemoryError, the original node is returned."""
        node = _make_node(_para(50), tmp_path)
        monkeypatch.setattr(
            "local_search_agent.ingestion.chunker._chunk_mixed_content",
            lambda *a, **kw: (_ for _ in ()).throw(MemoryError("OOM")),
        )
        monkeypatch.setattr(
            "local_search_agent.ingestion.chunker._chunk_table",
            lambda *a, **kw: (_ for _ in ()).throw(MemoryError("OOM")),
        )
        assert chunk_document(node) == [node]
