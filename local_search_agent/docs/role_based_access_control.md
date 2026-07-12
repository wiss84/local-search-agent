# Role-Based Access Control (Multi-Tenant Mode)

By default, Local Search Agent runs in single-user mode — anyone who can
reach the file server or dashboard can do anything. Multi-tenant mode adds
three roles (`superadmin` / `admin` / `member`) on top of a pluggable
identity layer, so one shared deployment can serve multiple employees or
teams with different access levels, without giving up the "no vector
store, fully local" design.

Multi-tenant mode is entirely opt-in. If you never set
`identity_provider` on `SearchAgentConfig`, nothing about your deployment
changes — no new middleware runs, no new checks happen, and every existing
CLI command, Python API call, and endpoint behaves exactly as before.

---

## Concepts

### Identity

An `Identity` is a resolved caller: a stable `subject` (typically an
email, e.g. `"alice@acme.com"`), an optional `display_name` for the UI,
and an `is_superadmin` flag. Superadmin is a real, third role tier in
practice — typically held by whoever runs IT/ops for the deployment, not
a single person's personal escape hatch — see "Roles and grants" below
for what it actually controls.

### IdentityProvider

An `IdentityProvider` answers one question: *given this HTTP request, who
is calling?* It never decides *what* they're allowed to do — that's a
separate layer (below). The framework ships three built-in providers, and
you can write your own by implementing one method:

```python
from local_search_agent.auth.identity import Identity

class MyIdentityProvider:
    def resolve(self, request) -> Identity | None:
        # Return None for anyone you can't verify — never guess or
        # default to an identity. A None here means "not authenticated",
        # not "let it through".
        ...
```

Every built-in provider is fail-closed: any error, missing credential, or
unverifiable token resolves to `None`, which `AuthorizationMiddleware`
treats as an anonymous caller and denies.

### Roles and grants

Three tiers, not two:

- **`member`** — workspace-scoped. Search, ask questions, and manage your
  own conversations (rename, delete — but only ones you created; deleting
  someone else's conversation in a shared workspace is still refused).
- **`admin`** — workspace-scoped. Everything a member can do in that
  workspace, plus ordinary ingest/sync, toggling watch mode, and managing
  API keys/grants for *members* only (never for other admins or
  superadmins). Notably, an admin can **not** create or delete workspaces,
  trigger a force re-ingest or wipe-and-reingest, or touch concurrency/
  rate-limit settings — those moved to superadmin-only (see below).
- **`superadmin`** — not workspace-scoped at all. Unconditional access to
  everything, every workspace, bypassing every grant check in the system
  — there is no `workspace_members` row for a superadmin to have or lack,
  the bypass happens directly in code. Held by whoever actually deployed
  and provisioned the server (typically IT/ops), not assigned per-workspace
  the way `member`/`admin` are.

There is no `curator` or fourth tier — that was considered and
deliberately dropped to keep the model simple.

`member`/`admin` roles are granted per subject, per workspace, via
`workspace_members` rows — a subject can be `member` in `finance` and
`admin` in `marketing` at the same time. A subject with no grant for a
workspace has no access to it at all (fail-closed, not "read-only by
default"). `superadmin` is a separate flag set at key-creation time
(`auth create-key --superadmin`), not a `workspace_members` grant — it
applies everywhere, immediately, with no per-workspace step.

Actions split into three enforcement tiers, not two — worth being
precise about which is which, since the difference is deliberate, not
accidental:

- **Workspace-scoped** (`admin` role in that specific workspace): ingest/
  sync, watch-mode toggle, scheduler, deleting *any* conversation in that
  workspace (a member can only delete their own).
- **Global + admin-only** (`admin` in *any* workspace, since the
  underlying config (`app_state.config`) is one shared object across the
  whole deployment, not because these are considered less sensitive):
  managing LLM provider API keys, adding/removing model names, semantic/
  reranking/advanced settings, and RBAC administration for *members*
  (granting/revoking member access, managing member API keys).
- **Superadmin-only**: creating or deleting workspaces (requires a
  `document_dirs` path that already exists on the *server's own disk* —
  inherently a provisioning action, not day-to-day workspace
  administration, and not something every workspace admin should be able
  to trigger from a browser); restarting the whole dashboard process
  (`PATCH`ing a new `db_path` — affects every workspace and every other
  user at once); force re-ingest and wipe-and-reingest (an ordinary
  incremental sync stays open to workspace admins; the heavier,
  destructive variants don't); granting/revoking the `admin` role itself
  (a plain admin can promote/demote members, never other admins);
  managing another admin's API key; and concurrency/rate-limit
  configuration (see [Rate Limits & Concurrency](#rate-limits--concurrency)
  below).

---

## Choosing an IdentityProvider

### `HeaderIdentityProvider` — you already have a reverse proxy doing auth

Trusts an HTTP header set by something in front of this app — an
authenticating reverse proxy, an API gateway, or a service mesh sidecar
that already verified the caller. No cryptography happens here; the
provider is exactly as trustworthy as whatever sets that header.

```python
from local_search_agent.auth.header_provider import HeaderIdentityProvider

identity_provider = HeaderIdentityProvider(
    header_name="X-Auth-Subject",
    trusted_proxy_ips=["10.0.0.0/8"],   # only trust the header from these IPs
)
```

**Only use this if you control the network path completely.** If this
header can reach the app from anywhere the header itself isn't stripped
and re-set by a trusted proxy, anyone can claim to be anyone. This is the
right choice when you already run something like Nginx with `auth_request`,
Cloudflare Access, or an internal API gateway doing real authentication in
front of the app.

### `APIKeyIdentityProvider` — no existing auth infrastructure

Issues long-lived API keys (`lsa_<key_id>_<secret>` format, argon2-hashed
at rest) and short-lived browser session cookies on top of them. This is
the right default when you don't already have SSO or a reverse proxy
doing authentication, and want something that works out of the box.

```python
from local_search_agent.auth.api_key_provider import APIKeyIdentityProvider
from local_search_agent.workspace.auth_db import AuthDB

identity_provider = APIKeyIdentityProvider(AuthDB(db_path=config.db_path))
```

Create keys via the CLI (see the walkthrough below) or the admin API. The
browser dashboard uses this provider's session cookie flow automatically
— `POST /api/auth/login` with a raw key sets an `HttpOnly`, `Secure`,
`SameSite=Strict` cookie; API/CLI callers instead send
`Authorization: Bearer <raw_key>` directly on every request.

### `JWTIdentityProvider` — you already have an enterprise IdP

Validates a bearer JWT against your identity provider's JWKS endpoint —
Auth0, Okta, Azure AD, Google Workspace, or any standards-compliant
OIDC/OAuth2 issuer. This is the right choice when employees already sign
in via company SSO and you want that same session to carry over.

```python
from local_search_agent.auth.jwt_provider import JWTIdentityProvider

identity_provider = JWTIdentityProvider(
    issuer="https://login.acme.com/",
    audience="local-search-agent",
    jwks_uri="https://login.acme.com/.well-known/jwks.json",
    subject_claim="email",   # match whatever claim your IdP puts the email in
)
```

Signature verification uses an explicit algorithm allow-list (default
`["RS256"]"`, matching the overwhelming majority of enterprise IdPs) that
is never derived from the token's own header — this closes the classic
"algorithm confusion" attack where a forged token claims a different
algorithm than the one actually configured. `iss`/`aud` are validated,
not just the signature, and expiry is enforced with a bounded clock-skew
allowance (60 seconds by default). The JWKS itself is cached (~10 minutes)
and refreshed early if a token references a `kid` the cache doesn't
recognize yet, so key rotation on your IdP's side doesn't require
restarting this app.

If the JWKS endpoint itself becomes unreachable, requests using this
provider fail with a `503`, not a silent "let everyone through" or a
silent "deny everyone" — see [Failure modes](#failure-modes) below.

---

## Enabling multi-tenant mode

Set `identity_provider` on your config, same as any other field:

```python
from local_search_agent import SearchAgentConfig, SearchAgentFramework
from local_search_agent.auth.api_key_provider import APIKeyIdentityProvider
from local_search_agent.workspace.auth_db import AuthDB

config = SearchAgentConfig(
    workspace_name="finance",
    db_path="local_search_agent.db",
)
config.identity_provider = APIKeyIdentityProvider(AuthDB(db_path=config.db_path))

framework = SearchAgentFramework(config)
framework.start_file_server()
```

For the desktop dashboard, pass `--multi-tenant` on the CLI instead —
this wires up `APIKeyIdentityProvider` against the same `--db` you give
it. `HeaderIdentityProvider` and `JWTIdentityProvider` don't have a CLI
flag; they're configured directly in your own deployment code (as shown
above), since they need company-specific values (your header name and
trusted proxy IPs, or your IdP's issuer/audience/JWKS URL) that don't
make sense as generic CLI flags.

```bash
local-search ui --multi-tenant --db /var/lib/local-search-agent/prod.db
```

### A note on testing across devices on a LAN

If you're trying this out across two machines on the same network (not
just `localhost`), point `--host` at your machine's LAN IP so the other
device can reach it: `local-search ui --multi-tenant --host 192.168.1.50`.
You'll also need `--insecure-cookies`:

```bash
local-search ui --multi-tenant --host 192.168.1.50 --insecure-cookies
```

Why: `APIKeyIdentityProvider`'s browser session cookie is marked `Secure`,
which browsers only store/send over HTTPS -- **except** on
`localhost`/`127.0.0.1`, which browsers treat as a secure context even
over plain HTTP. That's why single-machine testing works with no extra
flags. The moment `--host` is a real LAN IP over plain HTTP, that
exception no longer applies: the login request still succeeds on the
server (a session really is created), but the browser silently drops the
cookie, so every subsequent page load looks unauthenticated again --
indistinguishable from "login did nothing" unless you know this mechanism
exists. `--insecure-cookies` removes the `Secure` flag so the cookie
survives on plain HTTP.

**Only use this on a network you trust** (e.g. your own home/office
LAN) — the session cookie travels in cleartext with this flag on. For
anything reachable beyond a LAN you control, terminate real TLS with a
reverse proxy instead (see [Production Deployment](production-deployment.md))
and leave `--insecure-cookies` off.

---

## CLI walkthrough

This example uses `APIKeyIdentityProvider`, bootstrapped from scratch.

**1. Create a workspace** (if you haven't already):

```bash
local-search workspace create finance "/srv/docs/finance"
```

**2. Create a superadmin key first** — workspace creation itself, and a
few other deployment-wide actions (see "Roles and grants" above), require
superadmin, so this is usually the very first key created for a fresh
deployment:

```bash
local-search auth create-key --subject root@acme.com --display-name "IT/Ops" --superadmin
```

**3. Create an ordinary API key for yourself, as the first workspace admin:**

```bash
local-search auth create-key --subject alice@acme.com --display-name "Alice" --created-by root@acme.com
```

```
✓ API key created for 'alice@acme.com' (key_id=a1b2c3d4)

  lsa_a1b2c3d4_9f8e7d6c5b4a3928170695...

  This key will not be shown again. Store it securely now.
```

The raw key is shown exactly once — only its argon2 hash is stored.
Anyone who wants to authenticate as `alice@acme.com` needs this exact
string; there is no way to retrieve it again later (create a new key
instead).

**4. Grant Alice admin access to the workspace:**

```bash
local-search grant-access --subject alice@acme.com --workspace finance --role admin
```

**5. Grant a colleague member access:**

```bash
local-search auth create-key --subject bob@acme.com --display-name "Bob" --created-by alice@acme.com
local-search grant-access --subject bob@acme.com --workspace finance --role member
```

**6. Check who has access to what:**

```bash
local-search list-access --workspace finance
```

```
Workspace            Subject                        Role       Granted By           Granted At
----------------------------------------------------------------------------------------------
  finance            alice@acme.com                 admin      root@acme.com        2026-07-05T10:02:11+00:00
  finance            bob@acme.com                   member     alice@acme.com       2026-07-05T10:04:33+00:00
```

**7. Revoke access** (from one workspace, or entirely):

```bash
local-search revoke-access --subject bob@acme.com --workspace finance
local-search revoke-access --subject bob@acme.com   # revokes everything, all workspaces
```

**8. Revoke a compromised key without touching workspace grants:**

```bash
local-search auth list-keys --subject bob@acme.com
local-search auth revoke-key <key_id>
```

Revoking a key immediately force-logs-out any active browser session tied
to that subject too, not just the raw key itself — if Bob is mid-session
in the dashboard when his key is revoked, his very next request gets a
`401` and his browser is redirected straight to the login page, the same
as if he'd clicked Sign Out himself. No separate "kill session" step
needed.

Grants (`workspace_members`) and keys (`api_keys`) are deliberately
separate: revoking a key stops that specific credential from
authenticating at all, while revoking a grant stops a subject from
accessing a workspace regardless of which valid key they present. Rotate
a leaked key with `auth revoke-key` + `auth create-key`; change what
someone's *allowed* to do with `grant-access`/`revoke-access`.

A plain admin (not superadmin) may create/revoke keys and grants for
*members* only — attempting either against a subject who already holds
`admin` anywhere, or a superadmin key, is refused. Only a superadmin may
grant or revoke the `admin` role itself, or manage another admin's key.

Full command reference: see [CLI Reference](cli-reference.md#grant-access)
and [CLI Reference — auth](cli-reference.md#auth).

---

## Model / Provider Access Control

A separate, optional layer on top of workspace access: which **provider +
model combinations** each role may use for their own queries — a cost
control, not a workspace permission. Like everything else in this doc,
it's multi-tenant-only; in single-user mode every configured model stays
fully usable with no allow-list involved at all.

### How it's scoped

Two flat allow-lists, one per non-superadmin role — **not** per
individual person, and **not** per workspace. Every `member` anywhere
shares one allowed set; every `admin` anywhere shares a (presumably
broader) one; `superadmin` always has access to every configured model,
unconditionally, with nothing to configure. A role with nothing granted
has access to **nothing** — fail-closed, same principle as workspace
access. This means: **grant at least one model to each role before
anyone tries to query**, or their requests will be refused with a clear
403, not a confusing failure.

If a subject holds different roles in different workspaces (admin in
Finance, member in Marketing), whichever role applies to the workspace
they're *currently* querying is the allow-list that's checked — not a
single "overall" role for that person.

### True per-request selection

Each query can specify its own provider/model explicitly — it isn't
locked to one shared, deployment-wide default. A member and an admin (or
two different admins) can genuinely use two different, independently
allowed models at the same time without affecting each other's queries.
If a request doesn't specify one, it falls back to the shared deployment
default (`app_state.config.provider`/`model_name`, the same one
`PATCH /api/ui/config` still controls) — but whichever provider/model
actually ends up being used, explicit or fallback, is checked against the
caller's current role's allow-list before the query runs.

### Managing the allow-lists

Through the dashboard: Settings → Model Manager → "Model access by
role" — a role dropdown, a provider dropdown, and a model dropdown
(filtered to that provider's configured models, same pattern as the
sidebar's own Provider/Model selector), with Add/Remove. Visible and
editable by superadmin only — an ordinary admin can see and manage the
Model Manager's provider/model list itself, but not who's allowed to use
what.

Through the API: `GET/POST/DELETE /api/ui/models/access` (superadmin
only — see [HTTP endpoints](#http-endpoints) below).

Through the CLI:

```bash
local-search grant-model-access --role member --provider google --model-name gemma-4-31b-it
local-search revoke-model-access --role member --provider google --model-name gemma-4-31b-it
local-search list-model-access [--role member|admin]
```

Through the Python API: `framework.grant_model_access()` /
`revoke_model_access()` / `list_model_access()` — see [Python API
Reference](api-reference.md#modelprovider-access-control).

---

## Rate Limits & Concurrency

A separate concern from Model/Provider Access Control above — that
controls *who* may use a model; this controls *how much load* this
deployment puts on a given provider or piece of hardware, deployment-wide,
regardless of who's asking. Two independent things, both fully
admin-configurable so a company on a paid-tier account with real,
much-higher limits than the free tier can set their own numbers rather
than being stuck with free-tier defaults:

- **Concurrency** — the max number of LLM calls for a given provider
  allowed in flight at once. For Ollama, this is the framework-side
  mirror of Ollama's own `OLLAMA_NUM_PARALLEL` — set it to whatever your
  actual hardware supports; this framework has no way to introspect your
  VRAM itself. For cloud providers it's a burst control layered on top of
  (not instead of) RPM/TPM tracking below. Left unset, a provider has no
  cap at all (today's behavior, unchanged until you opt in).

  A concurrent call beyond the limit waits in line rather than failing
  immediately — the dashboard shows "N requests ahead of you" while it
  waits. Only if a slot never frees up within 120 seconds does the
  request fail outright, with a clear message rather than the browser tab
  hanging indefinitely.

  **Single-user mode only shows the Quota section, not Concurrency** — a
  single-user desktop install has no separate "deployment" to protect
  from itself, so a deployment-wide concurrency cap has nothing useful to
  do there.

- **Quota overrides (RPM / TPM / RPD)** — Google gets real sliding-window
  rate-limit tracking automatically, auto-detected from the free tier
  (15 RPM / 250K TPM for `gemini-*`, more generous for `gemma-4-*`).
  Every *other* provider (OpenAI, Anthropic, Ollama) tracks **nothing** by
  default — just exponential-backoff retry on errors — until you
  explicitly configure an override, which is also how you'd override
  Google's own auto-detected default for a paid-tier account. Any of
  requests-per-minute/tokens-per-minute/requests-per-day may be set
  independently; an omitted dimension means "don't track this", not
  "unlimited".

  This section stays visible and editable in **both** single-user and
  multi-tenant mode (a solo user on a paid-tier account benefits from
  real tracking just as much as a company would) — unlike Concurrency.

### Single-user and multi-tenant settings are completely independent

If you run this framework single-user on your own machine *and* also
test/run a multi-tenant deployment using the same OS user account on the
same machine, these are stored in two separate namespaces in the same
`rate_limits.json` file — changing one never touches or overwrites the
other. The CLI's `--multi-tenant` flag (see below) picks which namespace
a command edits; the dashboard picks automatically based on which mode
the running server is actually in.

### Managing rate limits

Through the dashboard: Settings → Model Manager → Concurrency /
Quota overrides sections, superadmin only in multi-tenant mode (hidden
entirely for an ordinary admin, not just disabled).

Through the CLI:

```bash
# Concurrency
local-search config set-concurrency --provider ollama --limit 2
local-search config delete-concurrency --provider ollama

# Quota overrides -- e.g. a paid-tier OpenAI account
local-search config set-rate-limit --provider openai --model-name gpt-5 --rpm 500 --tpm 2000000
local-search config delete-rate-limit --provider openai --model-name gpt-5

# Show everything currently configured
local-search config show-rate-limits
```

Add `--multi-tenant` to any of the above to edit the multi-tenant
namespace instead of single-user's (see above). Changes take effect
immediately — no restart needed, the next query for that provider/model
picks up the new setting.

Through the Python API: `framework.set_concurrency_limit()` /
`delete_concurrency_limit()` / `get_concurrency_limits()` /
`set_quota_override()` / `delete_quota_override()` /
`get_quota_overrides()`, each taking an explicit `multi_tenant: bool` —
see [Python API Reference](api-reference.md#rate-limits--concurrency).

---

## HTTP endpoints

These only exist (return meaningful data / are protected) when
`config.identity_provider` is set. `GET /api/ui/whoami` is the one
exception, always mounted but reporting `multi_tenant: false` in
single-user mode.

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/ui/whoami` | GET | Identity + role introspection, used by the frontend for UI role-gating (hiding admin-only buttons from members). Never the actual enforcement point — that's always `AuthorizationMiddleware`, server-side, on every protected route. |
| `/api/auth/login` | POST | `APIKeyIdentityProvider` only. Exchanges a raw API key for a session cookie. |
| `/api/auth/logout` | POST | `APIKeyIdentityProvider` only. Clears the session cookie. Idempotent. |
| `/api/admin/grants` | GET / POST / DELETE | List / create / revoke `workspace_members` grants. Global admin only; granting/revoking `admin` itself requires superadmin. |
| `/api/admin/keys` | GET / POST / DELETE | List / create / revoke API keys. `APIKeyIdentityProvider` only, global admin only; creating/revoking another admin's key requires superadmin. |
| `/api/ui/models/allowed` | GET | Filtered provider/model list for the caller's current role, used to populate the sidebar's per-query selector. Workspace-scoped (role is per-workspace). |
| `/api/ui/models/access` | GET / POST / DELETE | Manage the two role-level model allow-lists (see [Model / Provider Access Control](#model--provider-access-control)). Superadmin only. |
| `/api/ui/rate-limits` | GET | Full concurrency + quota-override config for this deployment's own mode. Superadmin only. |
| `/api/ui/rate-limits/concurrency` | POST / DELETE | Set/remove a provider's concurrency cap. Superadmin only. |
| `/api/ui/rate-limits/quota` | POST / DELETE | Set/remove a provider+model's RPM/TPM/RPD override. Superadmin only. |

All other existing routes (`/api/ui/query`, `/workspaces/{name}/docs`,
ingest/watch/scheduler endpoints, etc.) gain enforcement automatically
once `identity_provider` is set — see
[Architecture — Authorization middleware](architecture.md#authorization-middleware-multi-tenant-rbac)
for the full route-by-route table, and how it decides which role a given
route needs.

---

## Failure modes

Every built-in provider is fail-closed, but *how* a failure surfaces
depends on what actually failed — this distinction matters operationally:

- **Bad, expired, or missing credential** → `401`/`403`. The caller
  presented something, and it didn't check out. Ordinary and expected;
  no alerting needed.
- **The identity provider itself couldn't do its job** — e.g. `JWTIdentityProvider`'s
  JWKS endpoint is unreachable — → `503`. This is a deployment problem
  (your IdP is down, or a network path between this app and it broke),
  not a "someone tried to log in with a bad password" event. Point
  uptime monitoring at this distinction: a spike in `401`s from real
  users trying bad credentials is normal; a spike in `503`s from the
  identity layer itself is an incident.

This same liveness/readiness distinction shows up in
[Production Deployment](production-deployment.md#health-checks) at the
infrastructure level — `GET /health` is a plain liveness probe, while
`GET /health/ready` additionally checks that Meilisearch is reachable,
for the same "these are different failure classes" reason.

---

## Limitations

- **Three roles, not a general permission system.** `member`/`admin`/
  `superadmin` covers the common case. If you need finer-grained
  permissions (e.g. read-only vs. read-write within the same workspace),
  that's out of scope today.
- **No cross-workspace queries.** This was already true before
  multi-tenant mode — one Meilisearch index per workspace, by design —
  and RBAC doesn't change it. A `member` of both `finance` and `hr` still
  queries them one at a time.
- **File-level access control is separate and Windows/LDAP-only.** RBAC
  here controls *which workspaces* someone can reach; it says nothing
  about which specific files within a workspace's directory a given
  employee's OS-level permissions would normally block them from. See
  `config.enable_access_control` / `AccessControlMiddleware` for that
  orthogonal (and currently Windows/LDAP-only) concern — this doc doesn't
  cover it further.
- **No UI for creating custom `IdentityProvider`s.** `HeaderIdentityProvider`,
  `APIKeyIdentityProvider`, and `JWTIdentityProvider` cover the common
  deployment shapes; anything else means writing a small Python class as
  shown above, not a config toggle.
