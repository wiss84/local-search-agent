"""
Ingestion pipeline for the Local Search Agent framework.

Orchestrates document discovery, parsing, cleaning, optional semantic
enrichment (Phase 5), and Meilisearch indexing.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from local_search_agent.core.config import SearchAgentConfig
from local_search_agent.core.constants import SUPPORTED_EXTENSIONS
from local_search_agent.core.document_node import DocumentNode, _file_mtime_iso
from local_search_agent.ingestion.chunker import chunk_document
from local_search_agent.ingestion.parser import BaseParser, ParserError
from local_search_agent.ingestion.parsers import (
    CSVParser,
    DOCXParser,
    EMLParser,
    HTMLParser,
    JSONParser,
    PDFParser,
    PPTXParser,
    TextParser,
    XLSXParser,
    XMLParser,
)
from local_search_agent.workspace.workspace_manager import WorkspaceManager

logger = logging.getLogger(__name__)


@dataclass
class IngestStats:
    """Summary statistics returned after a pipeline run."""

    total: int = 0
    indexed: int = 0  # total chunks indexed into Meilisearch
    files_indexed: int = 0  # source files successfully parsed and indexed
    skipped: int = 0
    failed: int = 0
    duration_s: float = 0.0
    errors: list[str] = field(default_factory=list)  # failed file paths

    def failed_files(self) -> list[str]:
        """Return list of file paths that failed (basenames for display)."""
        return self.errors

    def __str__(self) -> str:
        return (
            f"IngestStats(total={self.total}, indexed={self.indexed}, "
            f"skipped={self.skipped}, failed={self.failed}, "
            f"duration={self.duration_s:.1f}s)"
        )


def _build_default_parsers() -> list[BaseParser]:
    return [
        PDFParser(),
        DOCXParser(),
        HTMLParser(),
        PPTXParser(),
        XLSXParser(),
        TextParser(),
        CSVParser(),
        JSONParser(),
        XMLParser(),
        EMLParser(),
    ]


class IngestionPipeline:
    """
    Orchestrates document discovery, parsing, cleaning, semantic enrichment,
    and Meilisearch indexing.

    Parameters
    ----------
    config            : SearchAgentConfig
    workspace_manager : WorkspaceManager for delta checks + document registration
    meili_client      : MeilisearchClient
    parsers           : Optional custom parser list (defaults to all built-ins)
    batch_size        : Documents per Meilisearch upload batch
    """

    def __init__(
        self,
        config: SearchAgentConfig,
        workspace_manager: WorkspaceManager,
        meili_client,
        parsers: Optional[list[BaseParser]] = None,
    ):
        self._config = config
        self._wm = workspace_manager
        self._meili = meili_client
        self._parsers = parsers or _build_default_parsers()
        self._enricher = None  # Lazy init (Phase 5)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, force: bool = False, progress_callback=None) -> IngestStats:
        """
        Run a full ingestion pass over all configured document directories.

        Parameters
        ----------
        force             : Re-index all files regardless of modification time.
        progress_callback : Optional callable(indexed, skipped, failed, total, current_file)
                            called after each file is processed. Used by the UI
                            to report live progress without polling the DB.
        """
        stats = IngestStats()
        start = time.monotonic()

        # First pass: count total eligible files so the UI can show X/N
        all_files: list[str] = []
        for doc_dir in self._config.document_dirs:
            if os.path.isdir(doc_dir):
                all_files.extend(self._walk(doc_dir))
        stats.total = len(all_files)

        if progress_callback:
            progress_callback(0, 0, 0, stats.total, "")

        for file_path in all_files:
            if not force:
                mtime = _file_mtime_iso(file_path)
                if not self._wm.document_needs_reindex(file_path, mtime):
                    stats.skipped += 1
                    if progress_callback:
                        progress_callback(
                            stats.indexed, stats.skipped, stats.failed, stats.total, file_path
                        )
                    continue

            nodes = self._parse_file(file_path, stats)
            if nodes:
                self._enrich_batch(nodes)
                self._flush_batch(nodes, stats)
                stats.files_indexed += 1

            if progress_callback:
                progress_callback(
                    stats.files_indexed, stats.skipped, stats.failed, stats.total, file_path
                )

        stats.duration_s = time.monotonic() - start
        logger.info("Ingestion complete: %s", stats)

        if progress_callback:
            progress_callback(
                stats.files_indexed, stats.skipped, stats.failed, stats.total, "__done__"
            )

        return stats

    # ------------------------------------------------------------------
    # Phase 5: Semantic enrichment
    # ------------------------------------------------------------------

    def _get_enricher(self):
        """Lazily build SemanticEnricher if semantic features are enabled."""
        if not self._config.enable_semantic:
            return None
        if self._enricher is None:
            try:
                from local_search_agent.agent.provider_factory import build_llm
                from local_search_agent.semantic.enricher import SemanticEnricher

                # Use semantic_model/provider override if set in settings
                if self._config.semantic_model:
                    from local_search_agent.core.config import SearchAgentConfig
                    from local_search_agent.core.key_manager import get_semantic_settings

                    sem_settings = get_semantic_settings()
                    sem_provider = sem_settings.get("semantic_provider") or self._config.provider
                    sem_config = SearchAgentConfig(
                        provider=sem_provider,
                        api_key=self._config.api_key,
                        model_name=self._config.semantic_model,
                    )
                    llm = build_llm(sem_config)
                else:
                    llm = build_llm(self._config)

                self._enricher = SemanticEnricher(
                    llm=llm,
                    enable_structural=True,
                )
                logger.info(
                    "SemanticEnricher initialised (model=%r).",
                    self._config.semantic_model or self._config.model_name,
                )
            except Exception as e:
                logger.warning("SemanticEnricher init failed: %s. Indexing without semantics.", e)
        return self._enricher

    def _enrich_batch(self, batch: list[DocumentNode]) -> None:
        """Run semantic enrichment on a batch if enabled."""
        enricher = self._get_enricher()
        if enricher is not None:
            enricher.enrich_batch(batch)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _walk(self, root: str):
        """Yield absolute paths of supported files, skipping hidden items."""
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for filename in sorted(filenames):
                if filename.startswith("."):
                    continue
                ext = os.path.splitext(filename)[1].lower()
                if ext not in SUPPORTED_EXTENSIONS:
                    continue
                yield os.path.join(dirpath, filename)

    def _find_parser(self, file_path: str) -> Optional[BaseParser]:
        for p in self._parsers:
            if p.can_parse(file_path):
                return p
        return None

    def _parse_file(self, file_path: str, stats: IngestStats) -> list[DocumentNode]:
        parser = self._find_parser(file_path)
        if parser is None:
            logger.warning("No parser found for: %s", file_path)
            stats.failed += 1
            stats.errors.append(f"No parser for {file_path}")
            return []

        try:
            node = parser.parse(source_path=file_path, workspace=self._config.workspace_name)
            return chunk_document(node)
        except FileNotFoundError:
            logger.error("File not found (deleted mid-ingestion?): %s", file_path)
            stats.failed += 1
            stats.errors.append(file_path)
            return []
        except ParserError as e:
            logger.error("Parser error for %s: %s", file_path, e)
            stats.failed += 1
            stats.errors.append(file_path)
            return []
        except Exception as e:
            logger.exception("Unexpected error parsing %s: %s", file_path, e)
            stats.failed += 1
            stats.errors.append(file_path)
            return []

    def _flush_batch(self, batch: list[DocumentNode], stats: IngestStats) -> None:
        try:
            self._meili.index_documents(batch)
        except Exception as e:
            logger.error("Meilisearch batch indexing failed: %s", e)
            stats.failed += len(batch)
            for node in batch:
                stats.errors.append(f"Indexing failed for {node.source_path}: {e}")
            return

        for node in batch:
            self._wm.register_document(node)
            stats.indexed += 1
            logger.debug("Indexed: %r (%r)", node.doc_id, node.title)
