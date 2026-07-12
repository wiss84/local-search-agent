"""
Exception hierarchy for the authorization path.

"Error handling — principles": specific exception types, not bare Exception, so every
failure mode along the identity/authorization path is distinguishable in
logs and can be mapped to the correct HTTP status without guessing.

All of these are caught internally by AuthorizationMiddleware and turned
into a generic, non-leaking HTTP response (see that module's docstring for
why the response body deliberately doesn't distinguish "no access" from
"doesn't exist"). They exist as a hierarchy mainly so *logging* can be
specific even when the *response* isn't.
"""

from __future__ import annotations


class AuthError(Exception):
    """Base class for every error in the identity/authorization path."""


class IdentityResolutionError(AuthError):
    """The presented credential (API key, JWT, header) is bad, malformed, or expired."""


class WorkspaceNotFoundError(AuthError):
    """
    The request references a workspace that doesn't exist.

    Note: per the "no information leakage" principle, this must never
    produce a response distinguishable from InsufficientRoleError's
    response — both become a generic 403 at the HTTP boundary. This
    exception exists for internal logging only.
    """


class InsufficientRoleError(AuthError):
    """The resolved identity has a grant, but not at the role the route requires."""


class ProviderUnavailableError(AuthError):
    """
    The configured IdentityProvider itself failed (JWKS endpoint down, LDAP
    unreachable, etc.) — distinct from "no credential presented" or "bad
    credential." Maps to 503, and must never silently fall back to treating
    the caller as unauthenticated or serving a stale cached result.
    """
