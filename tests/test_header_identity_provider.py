"""
Tests for HeaderIdentityProvider.

See auth/header_provider.py's module docstring for the trust-boundary caveats
this provider deliberately does NOT try to solve on its own (a reverse
proxy terminating auth and stripping client-supplied headers is assumed).
"""

from __future__ import annotations

from types import SimpleNamespace

from local_search_agent.auth.header_provider import HeaderIdentityProvider


class _FakeHeaders(dict):
    """Case-insensitive-enough stand-in for starlette's Headers for these tests."""

    def get(self, key, default=None):
        for k, v in self.items():
            if k.lower() == key.lower():
                return v
        return default


def _fake_request(headers: dict, client_host: str = "127.0.0.1"):
    return SimpleNamespace(
        headers=_FakeHeaders(headers),
        client=SimpleNamespace(host=client_host) if client_host else None,
    )


class TestBasicResolution:
    def test_resolves_subject_from_default_header(self):
        provider = HeaderIdentityProvider()
        request = _fake_request({"X-Remote-User": "alice@acme.com"})
        identity = provider.resolve(request)
        assert identity is not None
        assert identity.subject == "alice@acme.com"
        assert identity.display_name == ""
        assert identity.is_superadmin is False

    def test_custom_header_name(self):
        provider = HeaderIdentityProvider(header_name="X-Employee-Id")
        request = _fake_request({"X-Employee-Id": "bob@acme.com"})
        identity = provider.resolve(request)
        assert identity.subject == "bob@acme.com"

    def test_missing_header_returns_none(self):
        provider = HeaderIdentityProvider()
        request = _fake_request({})
        assert provider.resolve(request) is None

    def test_blank_header_returns_none(self):
        provider = HeaderIdentityProvider()
        request = _fake_request({"X-Remote-User": "   "})
        assert provider.resolve(request) is None

    def test_never_raises_on_malformed_input(self):
        provider = HeaderIdentityProvider()
        request = _fake_request({"X-Remote-User": "\x00\x01weird"})
        # Should resolve to *something* deterministic, never throw.
        identity = provider.resolve(request)
        assert identity is not None
        assert identity.subject == "\x00\x01weird"


class TestDisplayNameAndSuperadmin:
    def test_display_name_header(self):
        provider = HeaderIdentityProvider(display_name_header="X-Display-Name")
        request = _fake_request({"X-Remote-User": "alice@acme.com", "X-Display-Name": "Alice A."})
        identity = provider.resolve(request)
        assert identity.display_name == "Alice A."

    def test_superadmin_header_truthy_values(self):
        provider = HeaderIdentityProvider(superadmin_header="X-Superadmin")
        for truthy in ("1", "true", "TRUE", "yes", "on"):
            request = _fake_request({"X-Remote-User": "root@acme.com", "X-Superadmin": truthy})
            assert provider.resolve(request).is_superadmin is True

    def test_superadmin_header_falsy_or_absent(self):
        provider = HeaderIdentityProvider(superadmin_header="X-Superadmin")
        request = _fake_request({"X-Remote-User": "alice@acme.com", "X-Superadmin": "0"})
        assert provider.resolve(request).is_superadmin is False

        request2 = _fake_request({"X-Remote-User": "alice@acme.com"})
        assert provider.resolve(request2).is_superadmin is False

    def test_superadmin_header_not_configured_ignored(self):
        # No superadmin_header configured at all -- even a truthy-looking
        # header of that name must not accidentally grant superadmin.
        provider = HeaderIdentityProvider()
        request = _fake_request({"X-Remote-User": "alice@acme.com", "X-Superadmin": "true"})
        assert provider.resolve(request).is_superadmin is False


class TestTrustedProxyIps:
    def test_allows_request_from_trusted_ip(self):
        provider = HeaderIdentityProvider(trusted_proxy_ips=frozenset({"10.0.0.5"}))
        request = _fake_request({"X-Remote-User": "alice@acme.com"}, client_host="10.0.0.5")
        assert provider.resolve(request) is not None

    def test_rejects_request_from_untrusted_ip(self):
        provider = HeaderIdentityProvider(trusted_proxy_ips=frozenset({"10.0.0.5"}))
        request = _fake_request({"X-Remote-User": "alice@acme.com"}, client_host="1.2.3.4")
        assert provider.resolve(request) is None

    def test_rejects_when_client_is_none(self):
        provider = HeaderIdentityProvider(trusted_proxy_ips=frozenset({"10.0.0.5"}))
        request = _fake_request({"X-Remote-User": "alice@acme.com"}, client_host=None)
        assert provider.resolve(request) is None

    def test_no_restriction_when_trusted_proxy_ips_not_set(self):
        provider = HeaderIdentityProvider()
        request = _fake_request({"X-Remote-User": "alice@acme.com"}, client_host="1.2.3.4")
        assert provider.resolve(request) is not None
