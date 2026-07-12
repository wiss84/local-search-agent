"""
HeaderIdentityProvider: trusts an identity header set by a reverse proxy
that already terminated SSO.

See  ("generalizes existing X-Remote-User"). This is the same trust model
server/middleware/access_control.py already uses for its Windows-ACL/LDAP
checks (X-Remote-User set by nginx/IIS/Apache after authenticating the
caller) -- HeaderIdentityProvider makes that same pattern available to
AuthorizationMiddleware's workspace_members role checks, not just file
access control.

CRITICAL TRUST BOUNDARY -- read before configuring this in production
-----------------------------------------------------------------------
This provider does NOT authenticate anyone. It trusts whatever value is in
the configured header, verbatim, as the caller's identity. That is only
safe when a reverse proxy sits in front of this application and:

  1. Actually performs authentication (SSO, mTLS, etc.) before forwarding
     the request, AND
  2. Strips/overwrites this header from anything the original client sent,
     so a caller can't just set `X-Remote-User: admin@acme.com` themselves
     and walk in.

If this application is ever reachable directly (not exclusively through
that proxy), HeaderIdentityProvider is a full authentication bypass. This
is the same trust boundary AccessControlMiddleware already documents for
X-Remote-User -- nothing new is being introduced here, but it bears
repeating because a misconfiguration here is a total compromise, not a
partial one.

`trusted_proxy_ips`, if set, adds one more layer: reject (return None)
unless the immediate connecting peer's IP is in that set. This does NOT
make the header trustworthy if the network path in front of that peer
isn't itself locked down (e.g. spoofable X-Forwarded-For on an
untrusted network) -- it's defense in depth for the common case of "the
proxy runs on the same host or private subnet," not a substitute for
correctly configuring the proxy itself.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from local_search_agent.auth.identity import Identity

if TYPE_CHECKING:
    from starlette.requests import Request

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes", "on"}


class HeaderIdentityProvider:
    """
    Parameters
    ----------
    header_name          : Header carrying the stable subject identifier
                            (default "X-Remote-User", matching
                            AccessControlMiddleware's existing convention).
    display_name_header  : Optional header carrying a human-readable display
                            name (UI display only, never used for authorization).
    superadmin_header     : Optional header whose presence/value marks the
                             caller as a framework-level superadmin (see
                             Identity.is_superadmin's docstring -- rarely
                             needed; most identities should rely on
                             workspace_members grants instead).
    superadmin_values     : Case-insensitive values on superadmin_header that
                             count as "true" (default: "1", "true", "yes", "on").
    trusted_proxy_ips     : Optional set of IPs. If set, resolve() returns
                            None (fail closed, not "trust anyway") unless
                            request.client.host is in this set -- see the
                            module docstring's caveat before relying on this
                            alone.
    """

    def __init__(
        self,
        header_name: str = "X-Remote-User",
        display_name_header: Optional[str] = None,
        superadmin_header: Optional[str] = None,
        superadmin_values: frozenset[str] = frozenset(_TRUTHY),
        trusted_proxy_ips: Optional[frozenset[str]] = None,
    ):
        self._header_name = header_name
        self._display_name_header = display_name_header
        self._superadmin_header = superadmin_header
        self._superadmin_values = frozenset(v.lower() for v in superadmin_values)
        self._trusted_proxy_ips = trusted_proxy_ips

    def resolve(self, request: "Request") -> Optional[Identity]:
        """
        Extract an Identity from the configured header, or None if the
        header is absent/blank, or if trusted_proxy_ips is set and the
        connecting peer isn't in it. Never raises -- any malformed input
        here resolves to "no identity," consistent with every other
        IdentityProvider in this framework.
        """
        if self._trusted_proxy_ips is not None:
            peer = request.client.host if request.client else None
            if peer not in self._trusted_proxy_ips:
                logger.warning(
                    "HeaderIdentityProvider: rejecting request from untrusted peer %r "
                    "(not in trusted_proxy_ips).",
                    peer,
                )
                return None

        subject = request.headers.get(self._header_name, "").strip()
        if not subject:
            return None

        display_name = ""
        if self._display_name_header:
            display_name = request.headers.get(self._display_name_header, "").strip()

        is_superadmin = False
        if self._superadmin_header:
            raw = request.headers.get(self._superadmin_header, "").strip().lower()
            is_superadmin = raw in self._superadmin_values

        return Identity(subject=subject, display_name=display_name, is_superadmin=is_superadmin)
