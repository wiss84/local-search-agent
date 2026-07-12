# Production Deployment (Self-Hosting)

This page is for running Local Search Agent as a shared server for a team
or company, instead of the default single-user desktop setup — the
prerequisite for turning on [multi-tenant RBAC](role_based_access_control.md).

> **Note on `pip install`:** the Dockerfile, systemd unit files, and Caddy
> config referenced below live at the root of the
> [GitHub repo](https://github.com/wiss84/local-search-agent) (`Dockerfile`,
> `deploy/systemd/`, `deploy/Caddyfile`), **not** inside the PyPI package
> itself — `pip install local-search-agent` only installs the Python
> package. Every file's full content is reproduced inline below so this
> page is self-contained either way; grab them from GitHub only if you'd
> rather not copy-paste.

## What you do NOT need

No separate Meilisearch service to stand up. The framework already
downloads and manages the Meilisearch binary itself as a background
process (see [Installation](installation.md)) — that's true on a server
exactly the same way it's true on a laptop. Don't run a `meilisearch`
container/service alongside the app; it will just conflict with the one
the framework starts.

---

## Option A: Bare Metal / VM (simplest, no Docker)

```bash
pip install local-search-agent
local-search setup
local-search workspace create finance "/srv/docs/finance"
local-search ingest --workspace finance --dirs "/srv/docs/finance"
local-search serve --workspace finance --host 0.0.0.0 --port 8000
```

Run `watch start` as its own supervised process alongside `serve` for
live re-indexing — it's a separate foreground process, not a flag on
`serve` (the old `serve --scheduler` flag is deprecated; `watch` replaces
it):

```bash
local-search watch start --workspace finance
```

Run `serve` under a process supervisor so it restarts on crash/reboot —
`systemd` is the standard choice on a Linux server. Two separate unit
files, since `serve` and `watch start` are two independent long-running
processes:

```bash
sudo useradd --system --create-home --home-dir /var/lib/local-search-agent search-agent
sudo mkdir -p /etc/local-search-agent
# Create /etc/local-search-agent/env with at minimum:
#   MEILI_MASTER_KEY=<a real generated secret, not the dev default>
```

`/etc/systemd/system/local-search-agent.service`:

```ini
[Unit]
Description=Local Search Agent — file server
After=network.target

[Service]
Type=simple
User=search-agent
Group=search-agent
EnvironmentFile=/etc/local-search-agent/env
WorkingDirectory=/var/lib/local-search-agent
ExecStart=/usr/local/bin/local-search serve --workspace finance --host 127.0.0.1 --port 8000
Restart=on-failure
RestartSec=5
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/local-search-agent
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

`/etc/systemd/system/local-search-agent-watch.service`:

```ini
[Unit]
Description=Local Search Agent — watch mode
After=local-search-agent.service

[Service]
Type=simple
User=search-agent
Group=search-agent
EnvironmentFile=/etc/local-search-agent/env
WorkingDirectory=/var/lib/local-search-agent
ExecStart=/usr/local/bin/local-search watch start --workspace finance
Restart=on-failure
RestartSec=5
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/local-search-agent
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now local-search-agent.service
sudo systemctl enable --now local-search-agent-watch.service
```

`--host 127.0.0.1`, not `0.0.0.0`, in the unit file — bind locally and let
the reverse proxy (below) be the only thing actually facing the network.
Run as a dedicated non-root `search-agent` user, not root.

---

## Option B: Docker (optional, for teams that already standardize on it)

One container, no separate Meilisearch service — the same binary
auto-download behavior happens inside the container:

```dockerfile
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir local-search-agent

RUN useradd --create-home appuser
USER appuser
WORKDIR /home/appuser

EXPOSE 8000

CMD ["local-search", "serve", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t local-search-agent .

docker run -d --name lsa-serve \
  -v lsa_data:/home/appuser/.local-search-agent \
  -e MEILI_MASTER_KEY="$MEILI_MASTER_KEY" \
  -p 127.0.0.1:8000:8000 \
  local-search-agent
```

For watch mode in Docker, run it as a second container from the same
image (sharing the same data volume) rather than a flag — `serve` and
`watch start` are independent processes, same as in the bare-metal
option:

```bash
docker run -d --name lsa-watch \
  -v lsa_data:/home/appuser/.local-search-agent \
  -e MEILI_MASTER_KEY="$MEILI_MASTER_KEY" \
  local-search-agent local-search watch start --workspace finance
```

The volume mount matters: it's the only thing that survives a container
recreate — it holds both `metadata_db`'s SQLite file and the
auto-downloaded Meilisearch binary + index data.

---

## Reverse Proxy / TLS (needed in both options)

The app should never be directly internet-facing or terminate TLS
itself. Minimal Caddy example (auto-provisions certs):

```
search.acme-internal.com {
    reverse_proxy localhost:8000

    request_body {
        max_size 50MB
    }

    header {
        Strict-Transport-Security "max-age=31536000"
        X-Content-Type-Options "nosniff"
        X-Frame-Options "DENY"
    }
}
```

Bind to an internal DNS name reachable via VPN/internal network, not the
public internet — this is an employee tool, not a public product.

---

## Health Checks

Two separate endpoints, deliberately not one:

- **`GET /health`** — liveness. "Is this process up at all?" Always
  `200` once the process has started, regardless of Meilisearch's state.
  Point your process supervisor's restart-on-failure here (that's what
  the systemd units above already do implicitly via `Restart=on-failure`
  combined with the process crashing, not by polling this endpoint
  directly — but if your supervisor of choice does poll a liveness URL,
  this is it).
- **`GET /health/ready`** — readiness. "Can this process actually serve a
  search right now?" Also verifies Meilisearch is reachable; returns
  `503` with `{"status": "degraded", "meilisearch": false, ...}` if not.
  Point your load balancer's traffic-gating healthcheck here instead, so
  a momentarily-Meilisearch-less-but-otherwise-fine process gets held out
  of rotation without a liveness probe killing and restarting it for the
  same transient reason.

`local-search health` (CLI) reports per-workspace sync freshness — a
different, workspace-level concept from either endpoint above.

---

## Secrets

`MEILI_MASTER_KEY` is already an env var the CLI reads (see
[CLI Reference](cli-reference.md#environment-variables)) — no new
mechanism needed there. If you turn on
[multi-tenant RBAC](role_based_access_control.md) with `JWTIdentityProvider`,
your IdP's issuer/audience/JWKS URL aren't secrets themselves, but keep
them out of source control the same way — env vars at minimum, a real
secrets manager (cloud provider secret store, Docker/k8s secrets) for
teams with the ops maturity for one. Never commit any of these or ship a
real-looking default value.

---

## Backups

Back up the single data directory documented in
[Installation](installation.md#first-run-meilisearch-downloads-automatically)'s
cache-path table (per OS) — it contains `metadata_db`'s SQLite file and
the Meilisearch index data together. One directory, one thing to
snapshot.

---

## What This Deliberately Doesn't Try to Be

Not a Kubernetes/Helm setup or multi-region HA — the realistic deployment
target is one or a few instances on infrastructure a company already has.
Horizontal scaling is a future topic once a concrete deployment actually
needs it.
