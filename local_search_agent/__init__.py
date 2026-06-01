"""
local_search_agent
==================
An open-source, pip-installable Python framework that replaces vector-based RAG
with a deterministic, auditable local search system.

Phases
------
Phase 1: File server, DocumentNode, WorkspaceManager
Phase 2: Ingestion pipeline (PDF/DOCX/HTML/XLSX), text cleaner, Meilisearch indexing
Phase 3: LangGraph agent loop, multi-provider LLM, search + fetch tools
Phase 4: Multi-workspace isolation, APScheduler incremental sync, IndexMonitor
Phase 5: Semantic search (ConceptCompiler + StructuralParser + QueryExpander),
         LinkGraph cross-document relationships, Windows/LDAP access control
"""

from local_search_agent.agent.agent import LocalSearchAgent
from local_search_agent.core.config import SearchAgentConfig
from local_search_agent.core.document_node import DocumentNode
from local_search_agent.core.framework import SearchAgentFramework
from local_search_agent.ingestion.pipeline import IngestionPipeline, IngestStats
from local_search_agent.scheduler.incremental_sync import IncrementalSyncScheduler
from local_search_agent.scheduler.monitor import IndexHealthSummary, IndexMonitor
from local_search_agent.search.meilisearch_client import MeilisearchClient
from local_search_agent.search.query_builder import QueryBuilder
from local_search_agent.semantic.concept_compiler import ConceptCompiler, ConceptMetadata
from local_search_agent.semantic.enricher import SemanticEnricher
from local_search_agent.semantic.link_graph import LinkGraph
from local_search_agent.semantic.query_expander import QueryExpander
from local_search_agent.semantic.structural_parser import StructuralMetadata, StructuralParser
from local_search_agent.tools.search_tool import LocalSearchTool, ToolResult
from local_search_agent.workspace.metadata_db import MetadataDB
from local_search_agent.workspace.workspace_manager import WorkspaceManager

__all__ = [
    # Core
    "SearchAgentFramework",
    "SearchAgentConfig",
    "DocumentNode",
    # Tools
    "LocalSearchTool",
    "ToolResult",
    # Ingestion
    "IngestionPipeline",
    "IngestStats",
    # Search
    "MeilisearchClient",
    "QueryBuilder",
    # Agent
    "LocalSearchAgent",
    # Scheduler
    "IncrementalSyncScheduler",
    "IndexMonitor",
    "IndexHealthSummary",
    # Workspace
    "WorkspaceManager",
    "MetadataDB",
    # Semantic (Phase 5)
    "ConceptCompiler",
    "ConceptMetadata",
    "StructuralParser",
    "StructuralMetadata",
    "QueryExpander",
    "LinkGraph",
    "SemanticEnricher",
]
