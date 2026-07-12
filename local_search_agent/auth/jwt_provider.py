"""
JWTIdentityProvider: validates a bearer JWT against a configured issuer's
JWKS endpoint (Auth0, Okta, Azure AD, Google Workspace, or any standards-
compliant OIDC/OAuth2 issuer).

See ("three built-in providers") and the "Security checklist" section's JWT bullet.
This is the heaviest of the three built-in IdentityProviders -- unlike
HeaderIdentityProvider (trust a header, no cryptography involved) this
does real signature verification, so a bug here is a full authentication
bypass, not a partial one. Every item below maps directly to a line in
that checklist.

Security properties, stated explicitly
---------------------------------------
1. **Explicit algorithm allow-list.** `algorithms=` passed to jwt.decode()
   is *this provider's own configured list* (default ["RS256"]),  never
   derived from the token's own "alg" header. A token claiming
   `"alg": "none"` or `"alg": "HS256"` (algorithm confusion, where an
   attacker signs with the public RSA key treated as an HMAC secret) is
   rejected before signature verification even runs, because jwt.decode()
   only tries algorithms in the allow-list regardless of what the token
   itself claims. Constructing this provider with "none" in the allow-list
   is a hard error (see __init__), not just a discouraged option.
2. **iss/aud validated, not just the signature.** A token signed by a
   legitimate-but-wrong-tenant issuer, or issued for a different
   audience/client, is rejected even though the signature itself is valid.
3. **Expiry/not-before enforced with bounded clock skew** (default 60s,
   per the checklist) via jwt.decode()'s `leeway=`.
4. **JWKS cached with a sane TTL** (~10 min default) via a small
   thread-safe in-memory cache (_JWKSCache below) -- no new caching
   library, consistent with the framework's "don't over-engineer, in-
   memory is fine at this scale" philosophy used elsewhere (auth_attempts
   rate limiting, the meili-key cache, etc.). On an unknown `kid` (key
   rotation happened before the TTL expired), the cache forces one
   refresh before giving up, rather than waiting out the full TTL.
5. **Fail closed on provider failure, distinctly from fail closed on a bad
   token.** If the JWKS endpoint itself is unreachable or returns garbage,
   that's a `ProviderUnavailableError` (maps to 503 in
   AuthorizationMiddleware) -- not silently treated as "no identity" the
   way a merely-invalid/expired token is (returns None, denied same as any
   other IdentityProvider). Conflating these would make an outage at the
   IdP indistinguishable from every caller suddenly losing access, which
   is a much worse thing to debug at 3am.

What this deliberately does NOT do
------------------------------------
No OIDC discovery (`.well-known/openid-configuration`) fetch -- the caller
supplies `jwks_uri` directly. Auto-discovery is a nice-to-have, not a
security requirement, and skipping it avoids a second network call and a
second thing that can be down. If a future company needs discovery, it's
one line at their call site (fetch the discovery doc, read `jwks_uri`
from it, pass that here) -- not a reason to add an HTTP client dependency
graph to this module.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Optional

import httpx
import jwt
from jwt import PyJWK
from jwt.exceptions import InvalidTokenError, PyJWKError

from local_search_agent.auth.errors import ProviderUnavailableError
from local_search_agent.auth.identity import Identity

if TYPE_CHECKING:
    from starlette.requests import Request

logger = logging.getLogger(__name__)

_DEFAULT_JWKS_CACHE_TTL_SECONDS = 600  # ~10 min, per the security checklist
_DEFAULT_CLOCK_SKEW_SECONDS = 60  # <=60s, per the security checklist
_DEFAULT_JWKS_FETCH_TIMEOUT_SECONDS = 10.0


class _JWKSCache:
    """
    Thread-safe, in-memory JWKS cache keyed by `kid`.

    Refreshes (a) when nothing has ever been fetched, (b) when the TTL has
    elapsed, or (c) when the requested `kid` isn't in the current cache --
    (c) handles key rotation happening before the TTL naturally expires,
    without waiting out the full window. A network failure during refresh
    raises ProviderUnavailableError rather than returning stale data or
    silently treating the caller as unauthenticated.
    """

    def __init__(
        self,
        jwks_uri: str,
        ttl_seconds: int = _DEFAULT_JWKS_CACHE_TTL_SECONDS,
        timeout_seconds: float = _DEFAULT_JWKS_FETCH_TIMEOUT_SECONDS,
    ):
        self._jwks_uri = jwks_uri
        self._ttl_seconds = ttl_seconds
        self._timeout_seconds = timeout_seconds
        self._lock = threading.Lock()
        self._keys_by_kid: dict[str, dict] = {}
        self._fetched_at_monotonic: float = 0.0

    def get_jwk(self, kid: str) -> Optional[dict]:
        with self._lock:
            is_stale = (time.monotonic() - self._fetched_at_monotonic) > self._ttl_seconds
            if kid not in self._keys_by_kid or is_stale:
                self._refresh_locked()
            result = self._keys_by_kid.get(kid)
            if result is None:
                self._refresh_locked()
                result = self._keys_by_kid.get(kid)
            return result

    def _refresh_locked(self) -> None:
        """Must be called with self._lock held."""
        try:
            response = httpx.get(self._jwks_uri, timeout=self._timeout_seconds)
            response.raise_for_status()
            data = response.json()
            keys = data["keys"]
        except Exception as e:
            logger.error("JWTIdentityProvider: failed to fetch JWKS from %r: %s", self._jwks_uri, e)
            raise ProviderUnavailableError(
                f"Failed to fetch JWKS from {self._jwks_uri!r}: {e}"
            ) from e

        self._keys_by_kid = {k["kid"]: k for k in keys if "kid" in k}
        self._fetched_at_monotonic = time.monotonic()


class JWTIdentityProvider:
    """
    Parameters
    ----------
    issuer                : Expected `iss` claim -- must match exactly.
    audience               : Expected `aud` claim (string or, per PyJWT,
                             matches if present in a list-valued `aud`).
    jwks_uri               : JWKS endpoint URL (e.g.
                             "https://login.acme.com/.well-known/jwks.json").
                             No OIDC discovery -- pass this directly (see
                             module docstring).
    algorithms             : Explicit allow-list for signature verification.
                             Default ["RS256"] (the overwhelming majority of
                             enterprise IdPs -- Auth0/Okta/Azure AD/Google
                             all default to RS256). Must never contain
                             "none" -- enforced by a hard error in __init__.
    subject_claim          : Claim to use as Identity.subject (the stable
                             identifier stored in workspace_members --
                             typically the employee's email). Default "sub".
                             Many IdPs put the human-readable email in a
                             different claim (e.g. "email") -- set this to
                             match your IdP's token shape.
    display_name_claim     : Optional claim for UI display only (never used
                             for authorization). Default "name".
    superadmin_claim        : Optional claim marking a framework-level
                             superadmin (see Identity.is_superadmin's
                             docstring -- rarely needed).
    superadmin_values       : Case-insensitive claim values counting as
                             "true" for superadmin_claim.
    jwks_cache_ttl_seconds  : See _JWKSCache above. Default ~10 min per the
                             security checklist.
    clock_skew_seconds      : Leeway applied to exp/iat/nbf checks. Default
                             60s, the checklist's stated ceiling -- don't
                             raise this without a specific reason.
    jwks_fetch_timeout_seconds : HTTP timeout for the JWKS fetch itself.
    """

    def __init__(
        self,
        issuer: str,
        audience: str,
        jwks_uri: str,
        algorithms: Optional[list[str]] = None,
        subject_claim: str = "sub",
        display_name_claim: Optional[str] = "name",
        superadmin_claim: Optional[str] = None,
        superadmin_values: frozenset[str] = frozenset({"1", "true", "yes", "on"}),
        jwks_cache_ttl_seconds: int = _DEFAULT_JWKS_CACHE_TTL_SECONDS,
        clock_skew_seconds: int = _DEFAULT_CLOCK_SKEW_SECONDS,
        jwks_fetch_timeout_seconds: float = _DEFAULT_JWKS_FETCH_TIMEOUT_SECONDS,
    ):
        if not issuer:
            raise ValueError("JWTIdentityProvider: issuer is required.")
        if not audience:
            raise ValueError("JWTIdentityProvider: audience is required.")
        if not jwks_uri:
            raise ValueError("JWTIdentityProvider: jwks_uri is required.")

        # Distinguish "not passed" (None -> default to RS256) from "passed
        # as an explicit empty list" ([] -> a real misconfiguration, must
        # raise) -- `if algorithms` alone treats both the same way since an
        # empty list is falsy, which would silently swallow this check.
        resolved_algorithms = list(algorithms) if algorithms is not None else ["RS256"]
        if not resolved_algorithms:
            raise ValueError("JWTIdentityProvider: algorithms must not be empty.")
        if any(a.strip().lower() == "none" for a in resolved_algorithms):
            # Never relaxable -- "none" must never appear in the allow-list,
            # regardless of what a caller configures. See module docstring
            # point 1.
            raise ValueError(
                "JWTIdentityProvider: 'none' must never be in the algorithm allow-list."
            )

        self._issuer = issuer
        self._audience = audience
        self._algorithms = resolved_algorithms
        self._subject_claim = subject_claim
        self._display_name_claim = display_name_claim
        self._superadmin_claim = superadmin_claim
        self._superadmin_values = frozenset(v.lower() for v in superadmin_values)
        self._clock_skew_seconds = clock_skew_seconds
        self._jwks_cache = _JWKSCache(
            jwks_uri,
            ttl_seconds=jwks_cache_ttl_seconds,
            timeout_seconds=jwks_fetch_timeout_seconds,
        )

    def resolve(self, request: "Request") -> Optional[Identity]:
        """
        Extract and verify a bearer JWT, returning the resolved Identity.

        Returns None for any bad/expired/malformed/wrong-issuer/wrong-
        audience/wrong-algorithm token, or for a missing/malformed
        Authorization header -- fail-closed, consistent with every other
        IdentityProvider in this framework.

        Raises ProviderUnavailableError (NOT caught here) if the JWKS
        endpoint itself is unreachable -- this is a distinct failure mode
        from "bad token" and AuthorizationMiddleware must map it to a 503,
        not a 401/403. See module docstring point 5.
        """
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.lower().startswith("bearer "):
            return None
        token = auth_header[7:].strip()
        if not token:
            return None
        return self._verify(token)

    def _verify(self, token: str) -> Optional[Identity]:
        try:
            unverified_header = jwt.get_unverified_header(token)
        except InvalidTokenError as e:
            logger.debug("JWTIdentityProvider: malformed token header: %s", e)
            return None

        kid = unverified_header.get("kid")
        if not kid:
            logger.debug("JWTIdentityProvider: token header missing 'kid'.")
            return None

        # ProviderUnavailableError from here propagates up uncaught -- see
        # this method's own docstring reference in resolve().
        jwk_dict = self._jwks_cache.get_jwk(kid)
        if jwk_dict is None:
            # Known-good JWKS fetch, but no key matches this kid -- a bad
            # token (unknown/rotated-out key), not a provider failure.
            logger.debug("JWTIdentityProvider: no JWKS key matches kid=%r.", kid)
            return None

        try:
            signing_key = PyJWK(jwk_dict).key
        except PyJWKError as e:
            logger.warning("JWTIdentityProvider: malformed JWK for kid=%r: %s", kid, e)
            return None

        try:
            claims = jwt.decode(
                token,
                signing_key,
                # Explicit allow-list -- never derived from the token's own
                # "alg" header. See module docstring point 1.
                algorithms=self._algorithms,
                audience=self._audience,
                issuer=self._issuer,
                leeway=self._clock_skew_seconds,
                options={"require": ["exp"]},
            )
        except InvalidTokenError as e:
            # Covers expired, wrong iss/aud, bad signature, wrong alg,
            # missing exp, not-yet-valid (nbf), and everything else PyJWT
            # itself considers invalid -- all fail closed the same way.
            logger.debug("JWTIdentityProvider: token rejected: %s", e)
            return None

        subject = claims.get(self._subject_claim)
        if not subject or not isinstance(subject, str):
            logger.warning(
                "JWTIdentityProvider: token missing/invalid subject claim %r.",
                self._subject_claim,
            )
            return None

        display_name = ""
        if self._display_name_claim:
            raw_name = claims.get(self._display_name_claim)
            display_name = str(raw_name) if raw_name else ""

        is_superadmin = False
        if self._superadmin_claim:
            raw_flag = claims.get(self._superadmin_claim)
            is_superadmin = str(raw_flag).lower() in self._superadmin_values

        return Identity(subject=subject, display_name=display_name, is_superadmin=is_superadmin)
