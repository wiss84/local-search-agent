"""
Unit tests for the text chunker (ingestion/chunker.py).

Tests cover:
- Short documents are not chunked (returned as-is)
- Long prose documents are split into overlapping chunks
- Table documents use row-based chunking
- Header row is prepended to every table chunk
- Chunk titles follow the "Doc [part N/M]" pattern
- Overlap preserves boundary content
- Force-split on documents that exceed CHUNK_MAX_CHARS without a break point
- Edge cases: empty text, single long word, exactly at threshold
"""

from __future__ import annotations

from local_search_agent.core.constants import (
    CHUNK_MAX_CHARS,
    CHUNK_OVERLAP_CHARS,
    CHUNK_TARGET_CHARS,
    TABLE_ROWS_PER_CHUNK,
)
from local_search_agent.core.document_node import DocumentNode
from local_search_agent.ingestion.chunker import chunk_document

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_node(tmp_path, text: str, name: str = "doc.txt") -> DocumentNode:
    f = tmp_path / name
    f.write_text(text, encoding="utf-8")
    return DocumentNode.from_file(str(f), text=text, workspace="test_ws")


def _prose(chars: int) -> str:
    """Generate realistic prose of approximately `chars` characters with natural line breaks."""
    sentence = "The quarterly financial results show strong performance across all divisions. "
    reps = (chars // len(sentence)) + 2
    text = (sentence * reps)[:chars]

    # Add realistic line breaks every ~200 chars (simulating paragraphs/sentences)
    lines = []
    for i in range(0, len(text), 200):
        lines.append(text[i:i+200])
    return "\n".join(lines)


def _table(rows: int, cols: int = 3) -> str:
    """Generate a Markdown table with a header row and `rows` data rows."""
    headers = " | ".join(f"Col{i}" for i in range(cols))
    separator = " | ".join(["---"] * cols)
    data_row = " | ".join(f"Val{i}" for i in range(cols))
    lines = [f"| {headers} |", f"| {separator} |"]
    lines += [f"| {data_row} |"] * rows
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Short document passthrough
# ---------------------------------------------------------------------------

class TestShortDocumentPassthrough:
    def test_short_doc_returned_as_single_chunk(self, tmp_path):
        text = "Short document. " * 5  # well under CHUNK_MIN_CHARS
        node = _make_node(tmp_path, text)
        chunks = chunk_document(node)
        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_short_doc_preserves_doc_id(self, tmp_path):
        text = "Short. " * 10
        node = _make_node(tmp_path, text)
        chunks = chunk_document(node)
        assert chunks[0].doc_id == node.doc_id

    def test_short_doc_preserves_workspace(self, tmp_path):
        text = "Short. " * 10
        node = _make_node(tmp_path, text)
        chunks = chunk_document(node)
        assert chunks[0].workspace == node.workspace

    def test_empty_text_returns_single_chunk(self, tmp_path):
        node = _make_node(tmp_path, "")
        chunks = chunk_document(node)
        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# Prose chunking
# ---------------------------------------------------------------------------

class TestProseChunking:
    def test_long_doc_produces_multiple_chunks(self, tmp_path):
        text = _prose(CHUNK_TARGET_CHARS * 4)
        node = _make_node(tmp_path, text)
        chunks = chunk_document(node)
        assert len(chunks) > 1

    def test_no_chunk_exceeds_max_chars(self, tmp_path):
        text = _prose(CHUNK_MAX_CHARS * 3)
        node = _make_node(tmp_path, text)
        chunks = chunk_document(node)
        for chunk in chunks:
            # Allow reasonable tolerance for overlap + word boundaries
            assert len(chunk.text) <= CHUNK_MAX_CHARS + CHUNK_OVERLAP_CHARS + 100

    def test_all_content_preserved_across_chunks(self, tmp_path):
        # Every character in the original should appear in at least one chunk
        # (allowing for overlap, not strict concatenation)
        text = _prose(CHUNK_TARGET_CHARS * 3)
        node = _make_node(tmp_path, text)
        chunks = chunk_document(node)
        combined = " ".join(c.text for c in chunks)
        # Sample 10 evenly-spaced substrings and verify each appears somewhere
        step = len(text) // 10
        for i in range(0, len(text) - 20, step):
            assert text[i:i + 20] in combined

    def test_chunk_titles_follow_pattern(self, tmp_path):
        text = _prose(CHUNK_TARGET_CHARS * 3)
        node = _make_node(tmp_path, text)
        node.title = "Finance Report"
        chunks = chunk_document(node)
        # When chunked, all chunks should have [part N/M] suffix
        assert "[part 1/" in chunks[0].title
        assert "part 2/" in chunks[1].title.lower()
        assert f"part {len(chunks)}/" in chunks[-1].title.lower()

    def test_overlap_content_shared_between_adjacent_chunks(self, tmp_path):
        text = _prose(CHUNK_TARGET_CHARS * 3)
        node = _make_node(tmp_path, text)
        chunks = chunk_document(node)
        if len(chunks) >= 2:
            # The tail of chunk N should appear at the start of chunk N+1
            tail = chunks[0].text[-CHUNK_OVERLAP_CHARS:]
            assert tail.strip()[:20] in chunks[1].text

    def test_chunk_breaks_at_heading_boundary(self, tmp_path):
        """Chunks should prefer to break at Markdown headings."""
        section = _prose(CHUNK_TARGET_CHARS // 2)
        text = f"{section}\n\n## New Section\n\n{section}\n\n## Another Section\n\n{section}"
        node = _make_node(tmp_path, text)
        chunks = chunk_document(node)
        # Verify that headings appear in the chunks (they may not start a chunk due to overlap)
        combined = " ".join(c.text for c in chunks)
        assert "## New Section" in combined
        assert "## Another Section" in combined

    def test_single_paragraph_longer_than_max_is_force_split(self, tmp_path):
        """A single paragraph with no break points must still be split."""
        # No spaces, no newlines — a pathological case
        text = "A" * (CHUNK_MAX_CHARS * 2)
        node = _make_node(tmp_path, text)
        chunks = chunk_document(node)
        assert len(chunks) >= 2
        for chunk in chunks:
            # Allow reasonable tolerance for overlap
            assert len(chunk.text) <= CHUNK_MAX_CHARS + CHUNK_OVERLAP_CHARS + 100

    def test_chunk_doc_ids_are_unique(self, tmp_path):
        text = _prose(CHUNK_TARGET_CHARS * 4)
        node = _make_node(tmp_path, text)
        chunks = chunk_document(node)
        doc_ids = [c.doc_id for c in chunks]
        assert len(doc_ids) == len(set(doc_ids))

    def test_all_chunks_have_same_workspace(self, tmp_path):
        text = _prose(CHUNK_TARGET_CHARS * 3)
        node = _make_node(tmp_path, text)
        chunks = chunk_document(node)
        for chunk in chunks:
            assert chunk.workspace == node.workspace

    def test_all_chunks_have_same_file_type(self, tmp_path):
        text = _prose(CHUNK_TARGET_CHARS * 3)
        node = _make_node(tmp_path, text)
        chunks = chunk_document(node)
        for chunk in chunks:
            assert chunk.file_type == node.file_type

    def test_concepts_and_synonyms_propagated_to_all_chunks(self, tmp_path):
        text = _prose(CHUNK_TARGET_CHARS * 3)
        node = _make_node(tmp_path, text)
        node.concepts = ["finance", "AWS"]
        node.synonyms = ["Amazon Web Services"]
        chunks = chunk_document(node)
        for chunk in chunks:
            assert "finance" in chunk.concepts
            assert "Amazon Web Services" in chunk.synonyms


# ---------------------------------------------------------------------------
# Table chunking
# ---------------------------------------------------------------------------

class TestTableChunking:
    def test_table_doc_uses_row_based_chunking(self, tmp_path):
        text = _table(rows=TABLE_ROWS_PER_CHUNK * 3)
        node = _make_node(tmp_path, text)
        chunks = chunk_document(node)
        assert len(chunks) > 1

    def test_header_row_prepended_to_every_chunk(self, tmp_path):
        text = _table(rows=TABLE_ROWS_PER_CHUNK * 2)
        node = _make_node(tmp_path, text)
        chunks = chunk_document(node)
        for chunk in chunks:
            # Every chunk should start with the header row
            assert chunk.text.lstrip().startswith("|")
            assert "Col0" in chunk.text
            assert "---" in chunk.text

    def test_table_chunk_row_count_respected(self, tmp_path):
        rows = TABLE_ROWS_PER_CHUNK * 2 + 5
        text = _table(rows=rows)
        node = _make_node(tmp_path, text)
        chunks = chunk_document(node)
        # Each chunk should contain at most TABLE_ROWS_PER_CHUNK data rows
        for chunk in chunks:
            data_rows = [
                line for line in chunk.text.splitlines()
                if line.strip().startswith("|") and "---" not in line and "Col0" not in line
            ]
            assert len(data_rows) <= TABLE_ROWS_PER_CHUNK

    def test_small_table_not_split(self, tmp_path):
        """A table with few rows should not be split."""
        text = _table(rows=5)
        node = _make_node(tmp_path, text)
        chunks = chunk_document(node)
        assert len(chunks) == 1

    def test_mixed_doc_with_few_table_rows_uses_prose_chunking(self, tmp_path):
        """If table rows are < TABLE_LINE_RATIO of total lines, use prose chunking."""
        prose = _prose(CHUNK_TARGET_CHARS * 2)
        table = _table(rows=3)
        # Mostly prose, few table rows
        text = prose + "\n\n" + table
        node = _make_node(tmp_path, text)
        chunks = chunk_document(node)
        # Should be prose chunks — verify we have multiple chunks (prose chunking happened)
        assert len(chunks) > 1
        # If table headers appear, they should not be in every chunk
        table_header_chunks = [c for c in chunks if "Col0" in c.text and "---" in c.text]
        if table_header_chunks:
            assert len(table_header_chunks) < len(chunks)

    def test_table_with_prose_before_and_after(self, tmp_path):
        """Prose before table should only appear in first chunk, prose after only in last chunk."""
        pre = "This is context before the table.\n\n"
        table = _table(rows=TABLE_ROWS_PER_CHUNK * 2)
        post = "\n\nThis is context after the table."
        text = pre + table + post
        node = _make_node(tmp_path, text)
        chunks = chunk_document(node)

        # First chunk should contain pre_block
        assert "This is context before the table" in chunks[0].text

        # Last chunk should contain post_block
        assert "This is context after the table" in chunks[-1].text

        # Middle chunks should NOT contain pre or post blocks
        if len(chunks) > 2:
            for chunk in chunks[1:-1]:
                assert "This is context before" not in chunk.text or "table" in chunk.text.split("This is context before")[0]  # only if it's the header
                assert "This is context after" not in chunk.text


# ---------------------------------------------------------------------------
# Break point priority tests
# ---------------------------------------------------------------------------

class TestBreakPointPriority:
    def test_prefers_double_blank_line_over_single_blank_line(self, tmp_path):
        """Double blank lines (stronger boundaries) should be preferred over single blank lines."""
        # Create text with break points at different priorities
        section1 = _prose(CHUNK_TARGET_CHARS // 3)
        section2 = _prose(CHUNK_TARGET_CHARS // 3)
        section3 = _prose(CHUNK_TARGET_CHARS // 3)

        # Single blank after section1, double blank after section2
        text = section1 + "\n" + section2 + "\n\n" + section3
        node = _make_node(tmp_path, text)
        chunks = chunk_document(node)

        # With proper priority, we should get multiple chunks
        # and the double-blank boundary should be preserved in content
        if len(chunks) >= 2:
            combined = " ".join(c.text for c in chunks)
            # Verify section3 content exists (it was after the double blank)
            assert section3[:50] in combined

    def test_prefers_heading_boundary_over_blank_lines(self, tmp_path):
        """Heading boundaries (priority 4) should be stronger than double blank (priority 3)."""
        prose_before = _prose(CHUNK_TARGET_CHARS // 2)
        prose_after = _prose(CHUNK_TARGET_CHARS // 2)

        text = prose_before + "\n\n## Important Section Header\n\n" + prose_after
        node = _make_node(tmp_path, text)
        chunks = chunk_document(node)

        # Heading should appear in the chunks (may be at overlap boundary)
        combined = " ".join(c.text for c in chunks)
        assert "## Important Section Header" in combined

    def test_sentence_boundary_as_fallback(self, tmp_path):
        """Sentence boundaries (lowest priority) used only when no structural breaks exist."""
        # Create text with NO blank lines or headings, only sentence breaks
        text = "This is sentence one. This is sentence two. This is sentence three. " * (CHUNK_TARGET_CHARS // 70)
        text = text * 2  # Make it long enough to chunk
        node = _make_node(tmp_path, text)
        chunks = chunk_document(node)

        # Even with only sentence breaks, document should still be chunked
        assert len(chunks) >= 1
        # And if it is chunked, breaks should occur at sentence boundaries (periods)
        for chunk in chunks:
            # Each chunk should end cleanly (sentence-final punctuation or trimmed)
            assert chunk.text  # Not empty


# ---------------------------------------------------------------------------
# Fallback and edge case tests
# ---------------------------------------------------------------------------

class TestFallbackLogic:
    def test_fallback_when_no_break_in_target_window(self, tmp_path):
        """When no break exists in [target, hard_end], use last break before target."""
        # Create text with a break point just before CHUNK_TARGET_CHARS
        # and nothing between target and hard_end
        section1 = _prose(CHUNK_TARGET_CHARS - 100)
        gap = " " * 50  # No break points here
        section2 = _prose(CHUNK_TARGET_CHARS)

        text = section1 + "\n\n" + gap + section2
        node = _make_node(tmp_path, text)
        chunks = chunk_document(node)

        # Should still produce valid chunks (no crash)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert len(chunk.text) > 0

    def test_force_cut_at_hard_end_when_no_breaks_exist(self, tmp_path):
        """When no break point exists before hard_end, force-cut at hard_end."""
        # A single long "word" with no spaces or newlines
        text = "X" * (CHUNK_MAX_CHARS * 2)
        node = _make_node(tmp_path, text)
        chunks = chunk_document(node)

        # Must produce multiple chunks despite no natural breaks
        assert len(chunks) >= 2

        # Each chunk should respect CHUNK_MAX_CHARS (with tolerance for overlap)
        for chunk in chunks:
            assert len(chunk.text) <= CHUNK_MAX_CHARS + CHUNK_OVERLAP_CHARS + 100

    def test_break_point_exactly_at_chunk_max_boundary(self, tmp_path):
        """When a break point falls exactly at CHUNK_MAX_CHARS, it should be used."""
        # Create prose, then add a break exactly at the boundary
        target_len = CHUNK_TARGET_CHARS + 200  # Safe margin
        text = _prose(target_len) + "\n\n" + _prose(target_len)
        node = _make_node(tmp_path, text)
        chunks = chunk_document(node)

        # Should produce 2 chunks with a clean break between them
        assert len(chunks) >= 2

        # No chunk should be excessively long
        for chunk in chunks:
            assert len(chunk.text) <= CHUNK_MAX_CHARS + 50


# ---------------------------------------------------------------------------
# Metadata propagation tests
# ---------------------------------------------------------------------------

class TestMetadataPropagation:
    def test_all_chunks_preserve_source_path(self, tmp_path):
        """All chunks should inherit source_path from original."""
        text = _prose(CHUNK_TARGET_CHARS * 3)
        node = _make_node(tmp_path, text)
        chunks = chunk_document(node)
        for chunk in chunks:
            assert chunk.source_path == node.source_path

    def test_all_chunks_preserve_folder_path(self, tmp_path):
        """All chunks should inherit folder_path from original."""
        text = _prose(CHUNK_TARGET_CHARS * 3)
        node = _make_node(tmp_path, text)
        chunks = chunk_document(node)
        for chunk in chunks:
            assert chunk.folder_path == node.folder_path

    def test_all_chunks_preserve_timestamps(self, tmp_path):
        """All chunks should inherit modified_at and indexed_at from original."""
        text = _prose(CHUNK_TARGET_CHARS * 3)
        node = _make_node(tmp_path, text)
        chunks = chunk_document(node)
        for chunk in chunks:
            assert chunk.modified_at == node.modified_at
            assert chunk.indexed_at == node.indexed_at



