"""
Static mount helpers for the FastAPI file server.

In Phase 4 these helpers allow mounting arbitrary local folders as
static directories so the agent can discover new files without restarting.
Currently used by the ingestion pipeline to register source directories.
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles


def mount_directory(
    app: FastAPI,
    directory: str,
    mount_path: str,
    name: Optional[str] = None,
) -> None:
    """
    Mount a local directory as a static file path on the FastAPI app.

    Parameters
    ----------
    app        : FastAPI application instance.
    directory  : Absolute path to the local folder to expose.
    mount_path : URL prefix under which files are served (e.g. "/static/finance").
    name       : Internal route name (defaults to the directory basename).
    """
    if not os.path.isdir(directory):
        raise ValueError(f"Cannot mount {directory!r}: not a directory or does not exist.")

    route_name = name or os.path.basename(directory.rstrip("/\\"))
    app.mount(
        mount_path,
        StaticFiles(directory=directory),
        name=route_name,
    )
