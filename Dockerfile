# syntax=docker/dockerfile:1
#
# Local Search Agent -- production container image.
#
# Docker:"
# One container, no separate Meilisearch service -- the framework
# downloads and manages its own Meilisearch binary as a background process
# exactly the same way it does on bare metal (see docs/installation.md's
# cache-path table). Do NOT run a `meilisearch` container/service alongside
# this one; it will conflict with the one the framework starts itself.
#
# Build:
#   docker build -t local-search-agent .
#
# Run (serve):
#   docker run -d --name lsa-serve \
#     -v lsa_data:/home/appuser/.local-search-agent \
#     -e MEILI_MASTER_KEY="$MEILI_MASTER_KEY" \
#     -p 127.0.0.1:8000:8000 \
#     local-search-agent
#
# Run (watch mode -- a second, independent container sharing the same
# volume, NOT a flag on serve; see the design doc's rationale):
#   docker run -d --name lsa-watch \
#     -v lsa_data:/home/appuser/.local-search-agent \
#     -e MEILI_MASTER_KEY="$MEILI_MASTER_KEY" \
#     local-search-agent local-search watch start --workspace finance
#
# The volume mount is the only thing that survives a container recreate --
# it holds both metadata_db's SQLite file and the auto-downloaded
# Meilisearch binary + index data.

FROM python:3.12-slim

# tesseract-ocr : third-tier OCR fallback in the PDF ingestion pipeline
#                 (see docs/ingestion.md's three-tier OCR strategy)
# libgl1        : required by some OCR/image-processing dependencies
#                 (opencv-based bits pulled in transitively)
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir local-search-agent

# Dedicated non-root user -- never run this as root, matching the
# bare-metal option's "dedicated non-root search-agent user" guidance.
RUN useradd --create-home appuser
USER appuser
WORKDIR /home/appuser

EXPOSE 8000

# --host 0.0.0.0 is correct *inside* the container -- the reverse proxy
# (see /Caddyfile in this repo) 
# Publish this container's port bound to 127.0.0.1 on the host (see the `docker run`
# example above), same "bind locally, let the proxy face the network"
# principle as the bare-metal --host 127.0.0.1 guidance.
CMD ["local-search", "serve", "--host", "0.0.0.0", "--port", "8000"]
