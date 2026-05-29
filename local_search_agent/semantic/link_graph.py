"""
Link graph: SQLite cross-document relationship store (opt-in).

Responsibility
--------------
Stores and queries directional relationships between documents:
  - "references"  : Document A cites Document B by name
  - "same_topic"  : Documents A and B share >=3 concepts (auto-detected at ingest)
  - "same_folder" : Documents in the same directory (structural proximity)
  - "custom"      : Manually added relationships

Used by the `get_related_docs` agent tool to answer "what else is related to
this document?" without requiring another keyword search.

Schema
------
  doc_links:
    source_doc_id   TEXT  — the referencing document
    target_doc_id   TEXT  — the referenced document
    relation_type   TEXT  — "references" | "same_topic" | "same_folder" | "custom"
    weight          REAL  — relevance weight (0.0–1.0), higher = stronger link
    created_at      TEXT  — ISO-8601 timestamp

Design
------
- Opt-in: only populated if enable_link_graph=True in SearchAgentConfig.
- Links are built at ingest time by the SemanticEnricher.
- Queries are O(log n) via indexed source_doc_id.
- Thread-safe (threading.Lock on writes).

Usage
-----
    from local_search_agent.semantic.link_graph import LinkGraph

    graph = LinkGraph(db_path="local_search_agent.db")
    graph.add_link("doc_a", "doc_b", "references", weight=0.9)
    related = graph.get_related("doc_a", limit=5)
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS doc_links (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source_doc_id   TEXT NOT NULL,
    target_doc_id   TEXT NOT NULL,
    relation_type   TEXT NOT NULL DEFAULT 'references',
    weight          REAL NOT NULL DEFAULT 1.0,
    created_at      TEXT NOT NULL,
    UNIQUE(source_doc_id, target_doc_id, relation_type)
);
CREATE INDEX IF NOT EXISTS idx_doc_links_source ON doc_links(source_doc_id);
CREATE INDEX IF NOT EXISTS idx_doc_links_target ON doc_links(target_doc_id);
"""

RELATION_REFERENCES = "references"
RELATION_SAME_TOPIC = "same_topic"
RELATION_SAME_FOLDER = "same_folder"
RELATION_CUSTOM = "custom"


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat()


class LinkGraph:
    """
    SQLite-backed cross-document link graph.

    Parameters
    ----------
    db_path : Path to the SQLite database (same file as WorkspaceManager).
    """

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_link(
        self,
        source_doc_id: str,
        target_doc_id: str,
        relation_type: str = RELATION_REFERENCES,
        weight: float = 1.0,
    ) -> None:
        """Add a directed link from source to target. Ignores duplicate links."""
        if source_doc_id == target_doc_id:
            return
        weight = max(0.0, min(1.0, weight))
        now = _now_iso()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO doc_links
                    (source_doc_id, target_doc_id, relation_type, weight, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (source_doc_id, target_doc_id, relation_type, weight, now),
            )

    def add_links_batch(self, links: list[tuple[str, str, str, float]]) -> None:
        """
        Bulk insert links. Each tuple: (source_doc_id, target_doc_id, relation_type, weight).
        """
        now = _now_iso()
        rows = [
            (src, tgt, rel, max(0.0, min(1.0, w)), now) for src, tgt, rel, w in links if src != tgt
        ]
        if not rows:
            return
        with self._lock, self._connect() as conn:
            conn.executemany(
                """
                INSERT OR IGNORE INTO doc_links
                    (source_doc_id, target_doc_id, relation_type, weight, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )

    def delete_links_for_doc(self, doc_id: str) -> None:
        """Remove all links where this document is the source (used when re-indexing)."""
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM doc_links WHERE source_doc_id=?", (doc_id,))

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_related(
        self,
        doc_id: str,
        relation_type: Optional[str] = None,
        limit: int = 10,
        min_weight: float = 0.0,
    ) -> list[dict]:
        """
        Return documents related to doc_id, ordered by weight descending.

        Parameters
        ----------
        doc_id        : Source document ID.
        relation_type : Filter by relation type (None = all types).
        limit         : Max results.
        min_weight    : Minimum weight threshold.

        Returns
        -------
        List of dicts: {target_doc_id, relation_type, weight, created_at}
        """
        query = """
            SELECT target_doc_id, relation_type, weight, created_at
            FROM doc_links
            WHERE source_doc_id=? AND weight>=?
        """
        params: list = [doc_id, min_weight]

        if relation_type:
            query += " AND relation_type=?"
            params.append(relation_type)

        query += " ORDER BY weight DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_link_count(self, doc_id: str) -> int:
        """Return total number of outgoing links for a document."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM doc_links WHERE source_doc_id=?", (doc_id,)
            ).fetchone()
        return row["cnt"] if row else 0

    def build_same_topic_links(
        self,
        nodes: list,  # list of DocumentNode
        min_shared_concepts: int = 3,
        weight: float = 0.7,
    ) -> int:
        """
        Auto-build "same_topic" links between documents sharing >= N concepts.

        Called by SemanticEnricher after concept compilation.
        Returns the number of links created.
        """
        links: list[tuple[str, str, str, float]] = []

        for i, node_a in enumerate(nodes):
            concepts_a = set(c.lower() for c in node_a.concepts)
            for node_b in nodes[i + 1 :]:
                concepts_b = set(c.lower() for c in node_b.concepts)
                shared = concepts_a & concepts_b
                if len(shared) >= min_shared_concepts:
                    # Bidirectional
                    w = min(1.0, weight + len(shared) * 0.05)
                    links.append((node_a.doc_id, node_b.doc_id, RELATION_SAME_TOPIC, w))
                    links.append((node_b.doc_id, node_a.doc_id, RELATION_SAME_TOPIC, w))

        if links:
            self.add_links_batch(links)

        return len(links) // 2  # number of document pairs linked
