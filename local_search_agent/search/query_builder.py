"""
Query builder for Meilisearch filter expressions.

Constructs well-formed Meilisearch filter strings from Python-friendly
keyword arguments. Used by the agent's search_local_index tool to build
filters without manually constructing filter syntax.

Meilisearch filter syntax reference:
  https://www.meilisearch.com/docs/reference/api/search#filter

Examples:
    builder = QueryBuilder(workspace="finance")
    expr = builder.build()
    # 'workspace = "finance"'

    builder = QueryBuilder(workspace="finance", file_type=["pdf", "docx"])
    expr = builder.build()
    # 'workspace = "finance" AND (file_type = "pdf" OR file_type = "docx")'

    builder = QueryBuilder(modified_after="2024-01-01T00:00:00")
    expr = builder.build()
    # 'modified_at > "2024-01-01T00:00:00"'
"""

from __future__ import annotations

from typing import Optional, Union

from local_search_agent.core.constants import (
    FIELD_FILE_TYPE,
    FIELD_FOLDER_PATH,
    FIELD_MODIFIED_AT,
    FIELD_WORKSPACE,
)


class QueryBuilder:
    """
    Build Meilisearch filter expressions from structured parameters.

    All parameters are optional. Unset parameters are not included in the filter.
    Multiple parameters are combined with AND.
    Multi-value parameters (lists) are combined with OR inside parentheses.

    Parameters
    ----------
    workspace       : Filter to a specific workspace name.
    file_type       : One or more file types (e.g. "pdf" or ["pdf", "docx"]).
    folder_path     : Filter to documents under a specific folder path.
    modified_after  : ISO-8601 timestamp — only return docs modified after this.
    modified_before : ISO-8601 timestamp — only return docs modified before this.
    raw             : Append a raw filter expression string (advanced usage).
    """

    def __init__(
        self,
        workspace: Optional[str] = None,
        file_type: Optional[Union[str, list[str]]] = None,
        folder_path: Optional[str] = None,
        modified_after: Optional[str] = None,
        modified_before: Optional[str] = None,
        raw: Optional[str] = None,
    ):
        self._workspace = workspace
        self._file_type = [file_type] if isinstance(file_type, str) else (file_type or [])
        self._folder_path = folder_path
        self._modified_after = modified_after
        self._modified_before = modified_before
        self._raw = raw

    def build(self) -> Optional[str]:
        """
        Build and return the Meilisearch filter expression string.
        Returns None if no filters are set (means: no filtering).
        """
        clauses: list[str] = []

        if self._workspace:
            clauses.append(f'{FIELD_WORKSPACE} = "{self._workspace}"')

        if self._file_type:
            if len(self._file_type) == 1:
                clauses.append(f'{FIELD_FILE_TYPE} = "{self._file_type[0]}"')
            else:
                or_parts = " OR ".join(f'{FIELD_FILE_TYPE} = "{ft}"' for ft in self._file_type)
                clauses.append(f"({or_parts})")

        if self._folder_path:
            clauses.append(f'{FIELD_FOLDER_PATH} = "{self._folder_path}"')

        if self._modified_after:
            clauses.append(f'{FIELD_MODIFIED_AT} > "{self._modified_after}"')

        if self._modified_before:
            clauses.append(f'{FIELD_MODIFIED_AT} < "{self._modified_before}"')

        if self._raw:
            clauses.append(f"({self._raw})")

        if not clauses:
            return None

        return " AND ".join(clauses)

    def __repr__(self) -> str:
        return f"QueryBuilder(filter={self.build()!r})"
