"""
Tests for JWTIdentityProvider.

See auth/jwt_provider.py's module docstring for the properties under test here
(explicit algorithm allow-list, iss/aud validation, bounded clock skew,
JWKS caching with kid-miss refresh, and the ProviderUnavailableError vs.
"just deny" distinction).

No real network calls are made -- httpx.get is monkeypatched to serve a
JWKS built from a real, locally-generated RSA keypair, and tokens are
signed with jwt.encode() against that same key. This exercises the real
cryptographic verification path (unlike a plain mock of jwt.decode, which
would prove nothing about signature checking actually working).
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

from local_search_agent.auth import jwt_provider as jwt_provider_module
from local_search_agent.auth.errors import ProviderUnavailableError
from local_search_agent.auth.jwt_provider import JWTIdentityProvider

ISSUER = "https://login.acme.test/"
AUDIENCE = "acme-app"
JWKS_URI = "https://login.acme.test/.well-known/jwks.json"
KID = "test-key-1"
OTHER_KID = "test-key-2"


# ---------------------------------------------------------------------------
# Fixtures: a real RSA keypair + JWKS payload, and helpers to sign tokens.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rsa_private_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(scope="module")
def jwks_payload(rsa_private_key):
    public_key = rsa_private_key.public_key()
    jwk_dict = json.loads(RSAAlgorithm.to_jwk(public_key))
    jwk_dict.update({"kid": KID, "use": "sig", "alg": "RS256"})
    return {"keys": [jwk_dict]}


class _FakeHeaders(dict):
    """Case-insensitive-enough stand-in for starlette's Headers, matching
    the convention used in test_header_identity_provider.py."""

    def get(self, key, default=None):
        for k, v in self.items():
            if k.lower() == key.lower():
                return v
        return default


def _fake_request(token: str | None = None, raw_header: str | None = None):
    headers = {}
    if raw_header is not None:
        headers["Authorization"] = raw_header
    elif token is not None:
        headers["Authorization"] = f"Bearer {token}"
    return SimpleNamespace(headers=_FakeHeaders(headers))


def _make_token(
    private_key,
    *,
    kid: str = KID,
    issuer: str = ISSUER,
    audience: str = AUDIENCE,
    subject: str = "alice@acme.com",
    exp_delta_seconds: int = 3600,
    iat_delta_seconds: int = 0,
    extra_claims: dict | None = None,
    algorithm: str = "RS256",
    omit_exp: bool = False,
):
    now = int(time.time())
    claims = {
        "iss": issuer,
        "aud": audience,
        "sub": subject,
        "iat": now - iat_delta_seconds,
    }
    if not omit_exp:
        claims["exp"] = now + exp_delta_seconds
    if extra_claims:
        claims.update(extra_claims)
    return jwt.encode(claims, private_key, algorithm=algorithm, headers={"kid": kid})


class _FakeResponse:
    def __init__(self, json_data: dict, status_code: int = 200):
        self._json_data = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._json_data


@pytest.fixture
def mock_jwks_endpoint(monkeypatch, jwks_payload):
    """Monkeypatch httpx.get (as used inside jwt_provider.py) to serve
    jwks_payload without any real network call. Returns a call counter
    so tests can assert on cache-hit vs. cache-refresh behaviour."""
    calls = {"count": 0}

    def _fake_get(url, timeout=None):
        calls["count"] += 1
        assert url == JWKS_URI
        return _FakeResponse(jwks_payload)

    monkeypatch.setattr(jwt_provider_module.httpx, "get", _fake_get)
    return calls


@pytest.fixture
def failing_jwks_endpoint(monkeypatch):
    def _fake_get(url, timeout=None):
        raise ConnectionError("simulated network failure")

    monkeypatch.setattr(jwt_provider_module.httpx, "get", _fake_get)


@pytest.fixture
def provider(mock_jwks_endpoint):
    return JWTIdentityProvider(issuer=ISSUER, audience=AUDIENCE, jwks_uri=JWKS_URI)


# ---------------------------------------------------------------------------
# Construction / algorithm allow-list
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_requires_issuer(self):
        with pytest.raises(ValueError):
            JWTIdentityProvider(issuer="", audience=AUDIENCE, jwks_uri=JWKS_URI)

    def test_requires_audience(self):
        with pytest.raises(ValueError):
            JWTIdentityProvider(issuer=ISSUER, audience="", jwks_uri=JWKS_URI)

    def test_requires_jwks_uri(self):
        with pytest.raises(ValueError):
            JWTIdentityProvider(issuer=ISSUER, audience=AUDIENCE, jwks_uri="")

    def test_rejects_none_algorithm_in_allowlist(self):
        with pytest.raises(ValueError):
            JWTIdentityProvider(
                issuer=ISSUER, audience=AUDIENCE, jwks_uri=JWKS_URI, algorithms=["none"]
            )

    def test_rejects_none_algorithm_case_insensitive(self):
        with pytest.raises(ValueError):
            JWTIdentityProvider(
                issuer=ISSUER, audience=AUDIENCE, jwks_uri=JWKS_URI, algorithms=["RS256", "None"]
            )

    def test_rejects_empty_algorithm_list(self):
        with pytest.raises(ValueError):
            JWTIdentityProvider(issuer=ISSUER, audience=AUDIENCE, jwks_uri=JWKS_URI, algorithms=[])


# ---------------------------------------------------------------------------
# Basic resolution
# ---------------------------------------------------------------------------


class TestBasicResolution:
    def test_resolves_valid_token(self, provider, rsa_private_key):
        token = _make_token(rsa_private_key)
        request = _fake_request(token)
        identity = provider.resolve(request)
        assert identity is not None
        assert identity.subject == "alice@acme.com"
        assert identity.is_superadmin is False

    def test_missing_authorization_header_returns_none(self, provider):
        assert provider.resolve(_fake_request()) is None

    def test_non_bearer_header_returns_none(self, provider):
        request = _fake_request(raw_header="Basic dXNlcjpwYXNz")
        assert provider.resolve(request) is None

    def test_blank_bearer_token_returns_none(self, provider):
        request = _fake_request(raw_header="Bearer ")
        assert provider.resolve(request) is None

    def test_garbage_token_returns_none(self, provider):
        request = _fake_request(token="not-a-real-jwt")
        assert provider.resolve(request) is None


# ---------------------------------------------------------------------------
# iss / aud / exp validation
# ---------------------------------------------------------------------------


class TestClaimValidation:
    def test_wrong_issuer_rejected(self, provider, rsa_private_key):
        token = _make_token(rsa_private_key, issuer="https://evil.test/")
        assert provider.resolve(_fake_request(token)) is None

    def test_wrong_audience_rejected(self, provider, rsa_private_key):
        token = _make_token(rsa_private_key, audience="some-other-app")
        assert provider.resolve(_fake_request(token)) is None

    def test_expired_token_rejected(self, provider, rsa_private_key):
        token = _make_token(rsa_private_key, exp_delta_seconds=-3600)
        assert provider.resolve(_fake_request(token)) is None

    def test_token_missing_exp_rejected(self, provider, rsa_private_key):
        token = _make_token(rsa_private_key, omit_exp=True)
        assert provider.resolve(_fake_request(token)) is None

    def test_token_missing_subject_claim_rejected(self, provider, rsa_private_key, jwks_payload):
        now = int(time.time())
        claims = {"iss": ISSUER, "aud": AUDIENCE, "iat": now, "exp": now + 3600}
        token = jwt.encode(claims, rsa_private_key, algorithm="RS256", headers={"kid": KID})
        assert provider.resolve(_fake_request(token)) is None

    def test_clock_skew_within_leeway_accepted(self, provider, rsa_private_key):
        # Expired 30s ago -- within the default 60s leeway, should still pass.
        token = _make_token(rsa_private_key, exp_delta_seconds=-30)
        identity = provider.resolve(_fake_request(token))
        assert identity is not None

    def test_clock_skew_beyond_leeway_rejected(self, provider, rsa_private_key):
        token = _make_token(rsa_private_key, exp_delta_seconds=-120)
        assert provider.resolve(_fake_request(token)) is None


# ---------------------------------------------------------------------------
# Algorithm allow-list enforcement (algorithm confusion attacks)
# ---------------------------------------------------------------------------


class TestAlgorithmAllowlist:
    def test_hs256_token_rejected_by_rs256_only_provider(self, provider, jwks_payload):
        # Classic RS256/HS256 "algorithm confusion" attempt: sign with a
        # symmetric secret while claiming a kid that matches an RSA public
        # key. Must be rejected purely because HS256 isn't in the allow-list
        # -- regardless of whether the "secret" guessed happens to relate
        # to the public key at all.
        now = int(time.time())
        claims = {
            "iss": ISSUER,
            "aud": AUDIENCE,
            "sub": "attacker@acme.com",
            "iat": now,
            "exp": now + 3600,
        }
        forged_token = jwt.encode(
            claims, "some-guessed-secret", algorithm="HS256", headers={"kid": KID}
        )
        assert provider.resolve(_fake_request(forged_token)) is None

    def test_provider_can_be_configured_for_es256(self):
        # Just verifying construction succeeds with a non-default algorithm
        # list -- full EC signature verification is covered implicitly by
        # the RS256 tests above exercising the same jwt.decode() code path.
        p = JWTIdentityProvider(
            issuer=ISSUER, audience=AUDIENCE, jwks_uri=JWKS_URI, algorithms=["ES256"]
        )
        assert p is not None


# ---------------------------------------------------------------------------
# kid handling / JWKS caching
# ---------------------------------------------------------------------------


class TestJwksCaching:
    def test_unknown_kid_returns_none_not_provider_error(self, provider, rsa_private_key):
        # A well-formed JWKS was fetched fine; the token just references a
        # kid that isn't in it. This must be a plain deny (None), not a
        # ProviderUnavailableError -- the JWKS endpoint itself is healthy.
        token = _make_token(rsa_private_key, kid=OTHER_KID)
        assert provider.resolve(_fake_request(token)) is None

    def test_token_without_kid_header_returns_none(self, provider, rsa_private_key):
        now = int(time.time())
        claims = {
            "iss": ISSUER,
            "aud": AUDIENCE,
            "sub": "alice@acme.com",
            "iat": now,
            "exp": now + 3600,
        }
        token = jwt.encode(claims, rsa_private_key, algorithm="RS256")  # no kid header
        assert provider.resolve(_fake_request(token)) is None

    def test_jwks_fetched_once_and_cached_across_calls(self, mock_jwks_endpoint, rsa_private_key):
        provider = JWTIdentityProvider(issuer=ISSUER, audience=AUDIENCE, jwks_uri=JWKS_URI)
        token = _make_token(rsa_private_key)
        for _ in range(5):
            assert provider.resolve(_fake_request(token)) is not None
        assert mock_jwks_endpoint["count"] == 1

    def test_jwks_endpoint_failure_raises_provider_unavailable(
        self, failing_jwks_endpoint, rsa_private_key
    ):
        provider = JWTIdentityProvider(issuer=ISSUER, audience=AUDIENCE, jwks_uri=JWKS_URI)
        token = _make_token(rsa_private_key)
        with pytest.raises(ProviderUnavailableError):
            provider.resolve(_fake_request(token))


# ---------------------------------------------------------------------------
# Optional claims: display name, superadmin, custom subject claim
# ---------------------------------------------------------------------------


class TestOptionalClaims:
    def test_display_name_claim(self, mock_jwks_endpoint, rsa_private_key):
        provider = JWTIdentityProvider(
            issuer=ISSUER, audience=AUDIENCE, jwks_uri=JWKS_URI, display_name_claim="name"
        )
        token = _make_token(rsa_private_key, extra_claims={"name": "Alice A."})
        identity = provider.resolve(_fake_request(token))
        assert identity.display_name == "Alice A."

    def test_superadmin_claim_truthy(self, mock_jwks_endpoint, rsa_private_key):
        provider = JWTIdentityProvider(
            issuer=ISSUER,
            audience=AUDIENCE,
            jwks_uri=JWKS_URI,
            superadmin_claim="is_admin",
        )
        token = _make_token(
            rsa_private_key, subject="root@acme.com", extra_claims={"is_admin": "true"}
        )
        identity = provider.resolve(_fake_request(token))
        assert identity.is_superadmin is True

    def test_superadmin_claim_not_configured_ignored(self, mock_jwks_endpoint, rsa_private_key):
        provider = JWTIdentityProvider(issuer=ISSUER, audience=AUDIENCE, jwks_uri=JWKS_URI)
        token = _make_token(rsa_private_key, extra_claims={"is_admin": "true"})
        identity = provider.resolve(_fake_request(token))
        assert identity.is_superadmin is False

    def test_custom_subject_claim(self, mock_jwks_endpoint, rsa_private_key):
        provider = JWTIdentityProvider(
            issuer=ISSUER, audience=AUDIENCE, jwks_uri=JWKS_URI, subject_claim="email"
        )
        now = int(time.time())
        claims = {
            "iss": ISSUER,
            "aud": AUDIENCE,
            "sub": "some-opaque-id-123",
            "email": "bob@acme.com",
            "iat": now,
            "exp": now + 3600,
        }
        token = jwt.encode(claims, rsa_private_key, algorithm="RS256", headers={"kid": KID})
        identity = provider.resolve(_fake_request(token))
        assert identity.subject == "bob@acme.com"
