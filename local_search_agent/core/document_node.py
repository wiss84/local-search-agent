"""
DocumentNode: the canonical schema for a single indexed document.

Every document that passes through ingestion becomes a DocumentNode.
This is the single source of truth between the parser, Meilisearch, and the file server.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Optional


def _local_now_iso() -> str:
    """
    Return the current local time as an ISO-8601 string with UTC offset.

    Detects the system timezone from the OS (via `datetime.now().astimezone()`),
    so every user sees timestamps in their own timezone.  The offset is embedded
    in the string (e.g. "2026-05-19T14:32:01+02:00") so timestamps are always
    unambiguous and comparable across machines.
    """
    return datetime.now().astimezone().isoformat()


def _file_mtime_iso(path: str) -> str:
    """
    Return the last-modified time of a file as a local-timezone ISO-8601 string.

    Uses the system timezone (same as _local_now_iso) so that modified_at
    timestamps are consistent with indexed_at on the same machine.
    """
    ts = os.stat(path).st_mtime
    return datetime.fromtimestamp(ts).astimezone().isoformat()


@dataclass
class DocumentNode:
    """
    Represents one indexed document.

    Fields
    ------
    doc_id        : Stable unique ID derived from the source file path.
                    Format: sha256(source_path)[:16]  — short but collision-resistant.
    title         : Human-readable title (filename stem or extracted <title> tag).
    text          : Full pre-cleaned Markdown text. Stored here and served by /text/{doc_id}.
    file_type     : Lowercase extension without dot: "pdf", "docx", "html", etc.
    source_path   : Absolute path to the original file on disk.
    folder_path   : Parent directory of source_path (used for folder-level filtering).
    workspace     : Logical workspace name this document belongs to.
    modified_at   : ISO-8601 local-timezone timestamp of last file modification.
    indexed_at    : ISO-8601 local-timezone timestamp of when this node was last indexed.
    concepts      : Optional list of concept/topic tags generated at ingest time (Phase 5 semantic layer).
    synonyms      : Optional list of synonym strings for query expansion (Phase 5 semantic layer).
    """

    doc_id: str
    title: str
    text: str
    file_type: str
    source_path: str
    folder_path: str
    workspace: str
    modified_at: str
    indexed_at: str = field(default_factory=_local_now_iso)
    concepts: list[str] = field(default_factory=list)
    synonyms: list[str] = field(default_factory=list)
    summary: str = ""

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @staticmethod
    def make_doc_id(source_path: str) -> str:
        """
        Derive a stable, short doc_id from the absolute file path.
        Using SHA-256 prefix keeps IDs URL-safe and collision-resistant.
        """
        return hashlib.sha256(source_path.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def from_file(
        cls,
        source_path: str,
        text: str,
        workspace: str,
        title: Optional[str] = None,
        concepts: Optional[list[str]] = None,
        synonyms: Optional[list[str]] = None,
        summary: str = "",
    ) -> "DocumentNode":
        """
        Convenience constructor: derive all metadata from the file path.

        Parameters
        ----------
        source_path : Absolute path to the original file.
        text        : Pre-cleaned Markdown content.
        workspace   : Logical workspace name.
        title       : Override title (defaults to filename stem).
        concepts    : Optional concept tags.
        synonyms    : Optional synonym strings.
        summary     : Optional 2-3 sentence summary from semantic enrichment.
        """
        abs_path = os.path.abspath(source_path)
        stem = os.path.splitext(os.path.basename(abs_path))[0]
        ext = os.path.splitext(abs_path)[1].lstrip(".").lower()

        return cls(
            doc_id=cls.make_doc_id(abs_path),
            title=title or stem,
            text=text,
            file_type=ext,
            source_path=abs_path,
            folder_path=os.path.dirname(abs_path),
            workspace=workspace,
            modified_at=_file_mtime_iso(abs_path),
            concepts=concepts or [],
            synonyms=synonyms or [],
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a plain dict suitable for JSON serialisation or Meilisearch indexing."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DocumentNode":
        """Reconstruct a DocumentNode from a plain dict (e.g. loaded from JSON)."""
        return cls(**data)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def snippet(self, query: str, context_chars: int = 300) -> str:
        """
        Return a short snippet of self.text centred around the first occurrence
        of any word in `query`. Falls back to the first `context_chars` chars.
        """
        lower_text = self.text.lower()
        for word in query.lower().split():
            idx = lower_text.find(word)
            if idx != -1:
                start = max(0, idx - context_chars // 2)
                end = min(len(self.text), idx + context_chars // 2)
                fragment = self.text[start:end].strip()
                prefix = "…" if start > 0 else ""
                suffix = "…" if end < len(self.text) else ""
                return f"{prefix}{fragment}{suffix}"
        return self.text[:context_chars].strip() + ("…" if len(self.text) > context_chars else "")

    def __repr__(self) -> str:
        return (
            f"DocumentNode(doc_id={self.doc_id!r}, title={self.title!r}, "
            f"file_type={self.file_type!r}, workspace={self.workspace!r})"
        )
