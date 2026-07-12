"""
Unit tests for StructuralParser.to_searchable_text().

Covers:
- Empty metadata returns empty string
- Sections only
- Definitions only
- Key-values only
- References only
- Combined fields with deduplication
- Ordering: sections first, then definitions, then key_values, then references
"""

from __future__ import annotations

from local_search_agent.semantic.structural_parser import StructuralMetadata, StructuralParser


class TestToSearchableText:
    def test_empty_returns_empty_string(self):
        result = StructuralParser().to_searchable_text(StructuralMetadata())
        assert result == ""

    def test_sections_only(self):
        meta = StructuralMetadata(sections=["Section A", "Section B"])
        result = StructuralParser().to_searchable_text(meta)
        assert result == "Sections: Section A, Section B"

    def test_definitions_only(self):
        meta = StructuralMetadata(definitions=["AWS: Amazon Web Services", "Q3: Third Quarter"])
        result = StructuralParser().to_searchable_text(meta)
        lines = result.split("\n")
        assert lines == ["AWS: Amazon Web Services", "Q3: Third Quarter"]

    def test_key_values_only(self):
        meta = StructuralMetadata(key_values=["Budget: $1.2M", "Turnover: 5%"])
        result = StructuralParser().to_searchable_text(meta)
        lines = result.split("\n")
        assert lines == ["Budget: $1.2M", "Turnover: 5%"]

    def test_references_only(self):
        meta = StructuralMetadata(references=["Project Alpha Budget", "Employee Handbook"])
        result = StructuralParser().to_searchable_text(meta)
        assert result == "References: Project Alpha Budget, Employee Handbook"

    def test_combined_fields(self):
        meta = StructuralMetadata(
            sections=["Intro"],
            definitions=["AWS: Amazon Web Services"],
            key_values=["Spend: $1.2M"],
            references=["Q3 Report"],
        )
        result = StructuralParser().to_searchable_text(meta)
        lines = result.split("\n")
        assert lines[0] == "Sections: Intro"
        assert lines[1] == "AWS: Amazon Web Services"
        assert lines[2] == "Spend: $1.2M"
        assert lines[3] == "References: Q3 Report"

    def test_single_reference_no_trailing_comma(self):
        meta = StructuralMetadata(references=["Only Reference"])
        result = StructuralParser().to_searchable_text(meta)
        assert result == "References: Only Reference"

    def test_multiple_references_joined_with_comma_space(self):
        meta = StructuralMetadata(references=["Ref A", "Ref B", "Ref C"])
        result = StructuralParser().to_searchable_text(meta)
        assert "Ref A, Ref B, Ref C" in result
