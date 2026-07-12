"""
Identity + IdentityProvider: the pluggable identity layer for multi-tenant RBAC.

The framework does not decide who a company's employees are or which IdP
they use — it accepts an already-authenticated identity from whichever
IdentityProvider the embedding company configures, then enforces
authorization against workspace_members (see auth_db.AuthDB) on top of
that. This module defines the seam between "who is calling" (this file)
and "what are they allowed to do" (AuthorizationMiddleware).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from starlette.requests import Request


@dataclass
class Identity:
    """
    A resolved caller identity.

    Parameters
    ----------
    subject       : Stable identifier, not a descriptive label — typically
                    the employee's email (e.g. "alice@acme.com"). This is
                    the value stored in workspace_members.subject and
                    everywhere else identity is referenced.
    display_name  : Human-readable name for UI display only. Never used
                    for authorization decisions.
    is_superadmin : Escape hatch for a framework-level operator (distinct
                    from a workspace `admin` role — see the design doc's
                    note that "Developer" is not an in-app role). Rarely
                    used; most identities should rely on workspace_members
                    grants instead of this flag.
    """

    subject: str
    display_name: str = ""
    is_superadmin: bool = False


@runtime_checkable
class IdentityProvider(Protocol):
    """
    Protocol every identity provider implements.

    Implementations must be fail-closed: any error, missing credential, or
    unverifiable token resolves to None (no identity), never a guessed or
    default identity. AuthorizationMiddleware treats a
    None return as an anonymous/unauthenticated caller and denies access
    to any workspace-scoped route.
    """

    def resolve(self, request: "Request") -> Optional["Identity"]:
        """Extract a verified identity from the incoming request, or None."""
        ...
