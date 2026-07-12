"""
meili_key_crypto: Fernet encryption for scoped Meilisearch API keys at rest.

See  the Security checklist's "Credential storage" item: "Meilisearch keys
encrypted at rest with cryptography's Fernet, key sourced from env/secrets
manager, documented rotation procedure."

Key sourcing, in priority order
---------------------------------
1. LSA_FERNET_KEY environment variable (recommended for any real
   deployment -- lets the encryption key live in a secrets manager /
   deployment env rather than on disk next to the data it protects).
2. A key persisted to fernet.key in the same user-config directory as
   keys.json/settings.json (via platformdirs, see key_manager.py) --
   generated once on first use. Fine for single-machine desktop installs;
   NOT sufficient isolation for a real multi-server deployment, where
   LSA_FERNET_KEY should be used instead so the key doesn't ride along
   with the SQLite file it decrypts.

Rotation procedure (documented, not automated -- v1 scope)
-------------------------------------------------------------
1. Generate a new key: `Fernet.generate_key()`.
2. For every workspace: decrypt its stored meili_keys row with the OLD
   key, re-encrypt with the NEW key, overwrite the row (AuthDB.store_meili_key).
3. Only after every row is re-encrypted, swap LSA_FERNET_KEY (or
   fernet.key) to the new value and restart. A key rotation script is not
   provided in v1 -- this is a manual DBA-style operation until enough
   deployments need it to justify a CLI command.
"""

from __future__ import annotations

import logging
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken
from platformdirs import user_config_dir

from local_search_agent.core.key_manager import _APP_NAME

logger = logging.getLogger(__name__)

_ENV_VAR = "LSA_FERNET_KEY"


def _fernet_key_path() -> Path:
    config_dir = Path(user_config_dir(_APP_NAME))
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "fernet.key"


def _load_or_generate_key() -> bytes:
    """
    Fallback path when LSA_FERNET_KEY isn't set. Generates a key once and
    persists it so previously-encrypted meili_keys rows stay decryptable
    across restarts -- a fresh random key every startup would silently
    orphan every stored scoped key.
    """
    path = _fernet_key_path()
    if path.exists():
        return path.read_bytes().strip()
    key = Fernet.generate_key()
    path.write_bytes(key)
    logger.info(
        "Generated a new Fernet key at %r for encrypting scoped Meilisearch keys at rest. "
        "For production/multi-server deployments, set %s instead so the key doesn't live "
        "on disk next to the data it protects.",
        str(path),
        _ENV_VAR,
    )
    return key


def _get_fernet() -> Fernet:
    import os

    env_key = os.environ.get(_ENV_VAR)
    if env_key:
        return Fernet(env_key.encode() if isinstance(env_key, str) else env_key)
    return Fernet(_load_or_generate_key())


def encrypt_meili_key(raw_key: str) -> str:
    """Encrypt a raw Meilisearch API key for storage in AuthDB.meili_keys."""
    return _get_fernet().encrypt(raw_key.encode()).decode()


def decrypt_meili_key(encrypted: str) -> str:
    """
    Decrypt a stored Meilisearch API key.

    Raises InvalidToken if the encryption key has changed since this row
    was written (e.g. LSA_FERNET_KEY rotated without following the
    rotation procedure above) -- callers should treat this as "no usable
    scoped key" and fall back to the service-level key rather than
    crashing the request, since decrypt failures here are a
    defense-in-depth data-layer concern, not an authorization decision
    (AuthorizationMiddleware's role check already ran).
    """
    return _get_fernet().decrypt(encrypted.encode()).decode()


__all__ = ["encrypt_meili_key", "decrypt_meili_key", "InvalidToken"]
