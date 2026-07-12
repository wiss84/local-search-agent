"""
auth: multi-tenant identity + authorization subsystem.

This package holds the pluggable identity layer (Identity, IdentityProvider,
and its built-in implementations) and, in later phases, AuthorizationMiddleware.
Authorization *data* (workspace_members, activity_log, sessions, rate-limit
attempts, api_keys) lives in local_search_agent.workspace.auth_db.AuthDB —
this package is about resolving *who is calling*, not about what they're
allowed to do once resolved.
"""
