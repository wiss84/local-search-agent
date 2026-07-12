"""
provision_workspace_keys: creates and stores a scoped, member-level
Meilisearch API key for a workspace.

Called once per workspace, from api_routes.py's create_workspace route,
gated on `identity_provider is not None` -- single-user desktop installs
have no concept of a "member" role to scope a key for, so this is entirely
skipped there, same opt-in pattern as everything else in this feature.

Non-fatal by design: a workspace is fully usable without a scoped key --
member-level requests just fall back to the service-level master key
(AuthorizationMiddleware.MEILI_KEY_ATTR is None in that case). So provisioning failure here should
never block workspace creation itself; log and continue.
"""

from __future__ import annotations

import logging
from typing import Optional

from local_search_agent.auth.meili_key_crypto import encrypt_meili_key
from local_search_agent.search.meilisearch_client import MeilisearchClient
from local_search_agent.workspace.auth_db import AuthDB

logger = logging.getLogger(__name__)

# search-only -- member role never needs to add/delete documents or change
# index settings, only query them (see the Roles table: ingest is admin-only).
_MEMBER_ACTIONS = ["search"]


def provision_workspace_keys(
    workspace: str,
    meilisearch_url: str,
    meili_master_key: str,
    auth_db: AuthDB,
) -> Optional[str]:
    """
    Create a member-scoped Meilisearch key for `workspace` (search-only,
    limited to this workspace's index) and store it Fernet-encrypted in
    auth_db.meili_keys.

    Returns the new key's uid on success, or None if provisioning failed
    (logged as a warning, never raised -- see module docstring).

    The requesting client (`meili_master_key`) must be Meilisearch's own
    master key -- only the master key can create new keys.
    """
    try:
        client = MeilisearchClient(
            url=meilisearch_url, api_key=meili_master_key, index_name=workspace
        )
        key_uid, raw_key = client.create_scoped_key(
            actions=_MEMBER_ACTIONS,
            indexes=[workspace],
            description=f"member key for workspace={workspace}",
        )
        auth_db.store_meili_key(
            workspace=workspace,
            key_uid=key_uid,
            encrypted_key=encrypt_meili_key(raw_key),
        )
        logger.info("Provisioned scoped Meilisearch key for workspace=%r", workspace)
        return key_uid
    except Exception as e:
        logger.warning(
            "Could not provision scoped Meilisearch key for workspace=%r: %s. "
            "Member-level requests to this workspace will fall back to the "
            "service-level master key until this is retried.",
            workspace,
            e,
        )
        return None


def deprovision_workspace_keys(
    workspace: str,
    meilisearch_url: str,
    meili_master_key: str,
    auth_db: AuthDB,
) -> None:
    """
    Delete the scoped key for a workspace, both from Meilisearch and from
    auth_db.meili_keys. Called from workspace deletion. Best-effort on both
    sides -- see MeilisearchClient.delete_scoped_key's own swallow-and-log.
    """
    row = auth_db.get_meili_key_row(workspace)
    if row is None:
        return
    try:
        client = MeilisearchClient(
            url=meilisearch_url, api_key=meili_master_key, index_name=workspace
        )
        client.delete_scoped_key(row["key_uid"])
    finally:
        auth_db.delete_meili_key(workspace)
