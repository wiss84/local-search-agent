"""
APIKeyIdentityProvider: the built-in no-IdP fallback identity provider.

One option among three (Header/APIKey/JWT) — not the primary path for
companies that already have their own SSO, but the only one that works
day one with zero external infrastructure.

Key format & storage
---------------------
Raw keys are never stored. A generated key looks like:

    lsa_<key_id>_<secret>

`key_id` is a short, non-secret public identifier — storing it in plain
text lets verification do an indexed primary-key lookup in api_keys
instead of scanning every row and running an expensive argon2 verify
against each one (argon2 hashes can't be looked up by input the way a
fast deterministic hash can, since each has its own random salt).
`secret` is the actual credential; only its argon2 hash is persisted.

The raw key is returned to the caller exactly once, at creation time —
callers (CLI, admin API) must display/transmit it immediately and never
log it.

Browser sessions
-----------------
resolve() checks a session cookie first (see SESSION_COOKIE_NAME below),
falling through to the `Authorization: Bearer <key>` header if no valid
session cookie is present. Sessions are created via login() (see
auth/session_routes.py for the /api/auth/login, /api/auth/logout HTTP
endpoints) and are independent of the underlying API key after issuance --
by design, per the "what changes is what the browser holds onto after
login" principle: the raw long-lived key touches the wire once, at login;
after that the browser only ever presents the short-lived, revocable
session token. Revoking any one of a subject's API keys force-logs-out
ALL of that subject's active sessions (see revoke_key() below), even ones
not established via the revoked key -- an admin revoking a key for cause
wants that person locked out immediately, not merely blocked from their
next login attempt.

Sliding expiry, hard cap
--------------------------
Each successful session resolution extends expires_at by
_SESSION_IDLE_TIMEOUT from now, but never past `created_at + _SESSION_MAX_LIFETIME`
— an idle browser tab open for weeks doesn't grant an indefinite session,
and an actively-used one still gets force-logged-out eventually.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import uuid
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Optional

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

from local_search_agent.auth.identity import Identity
from local_search_agent.workspace.auth_db import AuthDB

if TYPE_CHECKING:
    from starlette.requests import Request

logger = logging.getLogger(__name__)

_KEY_PREFIX = "lsa"
_ph = PasswordHasher()

SESSION_COOKIE_NAME = "lsa_session"
_SESSION_IDLE_TIMEOUT = timedelta(hours=2)
_SESSION_MAX_LIFETIME = timedelta(hours=24)

# Brute-force protection on the login path (security checklist item in
# upcoming_features/04-multi-tenant-rbac-mode.md: "rate limit by IP ... on
# the key-validation path itself, via a small auth_attempts table").
# IP-based only -- the subject behind a wrong/malformed key isn't known
# until verify_key() succeeds, so there's nothing to key a per-subject
# limit on for the failed attempts that matter most here. Deliberately not
# applied to verify_key() itself (used on every bearer-header request, not
# just login) -- rate-limiting normal API traffic would punish legitimate
# high-frequency callers for a threat model that's specifically about
# repeated login guesses, not steady-state authenticated usage.
_MAX_FAILED_LOGIN_ATTEMPTS = 10
_LOGIN_ATTEMPT_WINDOW_MINUTES = 15


def _hash_token(raw_token: str) -> str:
    """sha256 hex digest — fast, deterministic, used only for session-token lookup
    (unlike API key secrets, session tokens are already high-entropy random
    values with no offline-guessing risk worth argon2's cost; a leaked
    token_hash from a DB dump can't be turned back into the token either
    way, and speed matters here since this runs on every request)."""
    return hashlib.sha256(raw_token.encode()).hexdigest()


class APIKeyIdentityProvider:
    """
    Identity provider backed by admin-generated, argon2-hashed API keys.

    Parameters
    ----------
    auth_db : Shared AuthDB instance (same db_path as the rest of the
              framework's SQLite state).
    """

    def __init__(self, auth_db: AuthDB):
        self._auth_db = auth_db

    # ------------------------------------------------------------------
    # Key issuance / management (admin-facing — called by CLI + admin API)
    # ------------------------------------------------------------------

    def create_key(
        self,
        subject: str,
        created_by: str,
        display_name: str = "",
        is_superadmin: bool = False,
    ) -> tuple[str, str]:
        """
        Generate a new API key for `subject`.

        Returns (key_id, raw_key). raw_key is shown to the caller exactly
        once — only its argon2 hash is ever persisted.
        """
        key_id = uuid.uuid4().hex[:12]
        secret = secrets.token_urlsafe(32)
        raw_key = f"{_KEY_PREFIX}_{key_id}_{secret}"
        key_hash = _ph.hash(secret)

        self._auth_db.create_api_key(
            key_id=key_id,
            subject=subject,
            key_hash=key_hash,
            display_name=display_name,
            is_superadmin=is_superadmin,
            created_by=created_by,
        )
        return key_id, raw_key

    def revoke_key(self, key_id: str) -> bool:
        """
        Revoke a key by its key_id, and immediately kill every active
        browser session belonging to that key's subject.

        The second half matters: without it, revoking a key only stops
        *future* logins with that credential -- a session established
        before revocation keeps working right up until its own idle/max
        lifetime expires (see module docstring's "Sliding expiry, hard
        cap"), since sessions are independent of the key by design. That's
        fine for routine key rotation, but wrong for the actual reason an
        admin revokes a key out-of-band: they want that person locked out
        *now*, not in up to 2 more hours. Every session for the subject is
        killed, not just ones tied to this specific key_id, since sessions
        aren't recorded as coming from one particular key -- a subject who
        holds multiple keys and gets one revoked for cause should not stay
        logged in on a different still-valid key without at least having to
        re-authenticate.

        Returns True if a matching active key was found.
        """
        row = self._auth_db.get_api_key(key_id)
        revoked = self._auth_db.revoke_api_key(key_id)
        if revoked and row is not None:
            killed = self._auth_db.delete_sessions_for_subject(row["subject"])
            if killed:
                logger.info(
                    "Revoked key_id=%r also force-logged-out %d active session(s) for subject=%r",
                    key_id,
                    killed,
                    row["subject"],
                )
        return revoked

    def list_keys(self, subject: Optional[str] = None) -> list[dict]:
        """List API key metadata for admin display. Never includes key_hash."""
        rows = self._auth_db.list_api_keys(subject=subject)
        return [{k: v for k, v in row.items() if k != "key_hash"} for row in rows]

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify_key(self, raw_key: str) -> Optional[Identity]:
        """
        Verify a raw API key and return the resolved Identity, or None if
        malformed, unknown, revoked, or the secret doesn't match.
        Fail-closed on every branch, including unexpected errors.
        """
        if not raw_key:
            return None
        parts = raw_key.split("_", 2)
        if len(parts) != 3 or parts[0] != _KEY_PREFIX:
            return None
        _, key_id, secret = parts

        row = self._auth_db.get_api_key(key_id)
        if row is None:
            return None
        if row["revoked_at"] is not None:
            logger.debug("API key %r rejected: revoked", key_id)
            return None

        try:
            _ph.verify(row["key_hash"], secret)
        except VerifyMismatchError:
            return None
        except Exception:
            logger.warning("API key verification error for key_id=%r", key_id)
            return None

        return Identity(
            subject=row["subject"],
            display_name=row["display_name"] or "",
            is_superadmin=bool(row["is_superadmin"]),
        )

    # ------------------------------------------------------------------
    # Browser session flow (see module docstring)
    # ------------------------------------------------------------------

    def login(
        self,
        raw_key: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> Optional[tuple[str, datetime]]:
        """
        Exchange a raw API key for a new browser session.

        Returns (raw_session_token, expires_at) on success, or None if the
        key is invalid/revoked, *or* if this ip_address has hit
        _MAX_FAILED_LOGIN_ATTEMPTS within _LOGIN_ATTEMPT_WINDOW_MINUTES --
        same generic failure as an invalid key, per the design doc's "no
        information leakage" principle (a rate-limited caller shouldn't be
        able to distinguish "too many attempts" from "wrong key"). The raw
        token is returned exactly once -- callers (session_routes.py) must
        place it directly into an HttpOnly cookie and never expose it to JS
        or log it. Only its sha256 hash is persisted.
        """
        if ip_address is not None:
            recent_failures = self._auth_db.count_recent_failed_attempts(
                ip_address=ip_address, window_minutes=_LOGIN_ATTEMPT_WINDOW_MINUTES
            )
            if recent_failures >= _MAX_FAILED_LOGIN_ATTEMPTS:
                logger.warning(
                    "Login rate-limited: ip_address=%r (%d recent failures)",
                    ip_address,
                    recent_failures,
                )
                return None

        identity = self.verify_key(raw_key)
        self._auth_db.record_attempt(
            subject=identity.subject if identity else None,
            ip_address=ip_address,
            success=identity is not None,
        )
        if identity is None:
            return None

        raw_token = secrets.token_urlsafe(32)
        token_hash = _hash_token(raw_token)
        expires_at = datetime.now().astimezone() + _SESSION_IDLE_TIMEOUT
        self._auth_db.create_session(
            token_hash=token_hash,
            subject=identity.subject,
            expires_at=expires_at.isoformat(),
            ip_address=ip_address,
            user_agent=user_agent,
            display_name=identity.display_name,
            is_superadmin=identity.is_superadmin,
        )
        self._auth_db.log_activity(
            subject=identity.subject,
            action="login",
            ip_address=ip_address,
        )
        logger.info("Session created: subject=%r", identity.subject)
        return raw_token, expires_at

    def logout(self, raw_token: str) -> None:
        """Immediately revoke a session. Idempotent — safe to call even if the token is already invalid."""
        token_hash = _hash_token(raw_token)
        row = self._auth_db.get_session(token_hash)
        self._auth_db.delete_session(token_hash)
        if row is not None:
            # Only log when the token actually resolved to a session --
            # idempotent repeat calls with an already-invalid token don't
            # get a phantom logout row attributed to nobody.
            self._auth_db.log_activity(subject=row["subject"], action="logout")

    def resolve_session(self, raw_token: str) -> Optional[Identity]:
        """
        Resolve a session cookie's token to an Identity, or None if
        missing/expired. On success, slides the session's expiry forward
        (capped at _SESSION_MAX_LIFETIME from creation). Fail-closed: any
        parsing error on the stored timestamps is treated as invalid.
        """
        if not raw_token:
            return None
        token_hash = _hash_token(raw_token)
        row = self._auth_db.get_session(token_hash)
        if row is None:
            return None

        try:
            expires_at = datetime.fromisoformat(row["expires_at"])
            created_at = datetime.fromisoformat(row["created_at"])
        except (ValueError, TypeError):
            logger.warning("Malformed session timestamps for token hash; treating as invalid.")
            return None

        now = datetime.now().astimezone()
        if now > expires_at:
            self._auth_db.delete_session(token_hash)  # sweep on the way out
            return None

        new_expires = min(now + _SESSION_IDLE_TIMEOUT, created_at + _SESSION_MAX_LIFETIME)
        if new_expires > expires_at:
            self._auth_db.extend_session(token_hash, new_expires.isoformat())

        return Identity(
            subject=row["subject"],
            display_name=row["display_name"] or "",
            is_superadmin=bool(row["is_superadmin"]),
        )

    # ------------------------------------------------------------------
    # IdentityProvider protocol
    # ------------------------------------------------------------------

    def resolve(self, request: "Request") -> Optional[Identity]:
        """
        Resolve an Identity, checking the session cookie first (browser
        flow) and falling through to the `Authorization: Bearer <key>`
        header (direct API/CLI flow) if no valid session cookie is present.
        """
        session_token = request.cookies.get(SESSION_COOKIE_NAME)
        if session_token:
            identity = self.resolve_session(session_token)
            if identity is not None:
                return identity

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.lower().startswith("bearer "):
            return None
        raw_key = auth_header[7:].strip()
        return self.verify_key(raw_key)
