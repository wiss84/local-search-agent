"""
FastAPI file server for the Local Search Agent framework.

Endpoints
---------
GET /health                      → liveness check
GET /health/indexes              → index freshness summary (Phase 4)
GET /docs/{doc_id}               → serve the original raw file
GET /text/{doc_id}               → serve pre-cleaned Markdown text
GET /workspaces                  → list registered workspaces
GET /workspaces/{name}/docs      → list all documents in a workspace
GET /workspaces/{name}/history   → sync history for a workspace
GET /help/{filename}             → documentation files (e.g. /help/getting-started.md)
"""

from __future__ import annotations

import logging
import mimetypes
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from local_search_agent.core.config import SearchAgentConfig
from local_search_agent.core.constants import __version__
from local_search_agent.workspace.workspace_manager import WorkspaceManager

logger = logging.getLogger(__name__)


def build_app(
    config: SearchAgentConfig,
    workspace_manager: WorkspaceManager,
    metadata_db=None,
) -> FastAPI:
    app = FastAPI(
        title="Local Search Agent — File Server",
        description=(
            "Serves raw documents and pre-cleaned text for the Local Search Agent framework."
        ),
        version=__version__,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    if config.enable_access_control:
        from local_search_agent.server.middleware.access_control import AccessControlMiddleware

        app.add_middleware(
            AccessControlMiddleware,
            config=config,
            workspace_manager=workspace_manager,
        )
        logger.info("Access control middleware enabled.")

    app.state.config = config
    app.state.workspace_manager = workspace_manager
    app.state.metadata_db = metadata_db

    @app.get("/health", tags=["meta"])
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok", "version": __version__})

    @app.get("/health/indexes", tags=["meta"])
    async def health_indexes() -> JSONResponse:
        if metadata_db is None:
            return JSONResponse(
                {"error": "Index health monitoring not available (MetadataDB not configured)."},
                status_code=503,
            )
        from local_search_agent.scheduler.monitor import IndexMonitor

        monitor = IndexMonitor(metadata_db)
        return JSONResponse(monitor.get_health_summary().to_dict())

    @app.get("/workspaces", tags=["meta"])
    async def list_workspaces() -> JSONResponse:
        return JSONResponse({"workspaces": workspace_manager.list_workspaces()})

    @app.get("/workspaces/{workspace_name}/docs", tags=["meta"])
    async def list_workspace_docs(workspace_name: str) -> JSONResponse:
        docs = workspace_manager.list_documents(workspace_name)
        if docs is None:
            raise HTTPException(404, detail=f"Workspace {workspace_name!r} not found.")
        return JSONResponse({"workspace": workspace_name, "documents": docs})

    @app.get("/workspaces/{workspace_name}/history", tags=["meta"])
    async def get_workspace_sync_history(workspace_name: str, limit: int = 20) -> JSONResponse:
        if metadata_db is None:
            raise HTTPException(503, detail="Sync history not available.")
        if workspace_manager.get_workspace(workspace_name) is None:
            raise HTTPException(404, detail=f"Workspace {workspace_name!r} not found.")
        history = metadata_db.get_sync_history(workspace_name, limit=limit)
        return JSONResponse({"workspace": workspace_name, "history": history})

    @app.get("/text/{doc_id}", tags=["documents"])
    async def get_text(doc_id: str) -> PlainTextResponse:
        node = workspace_manager.get_document(doc_id)
        if node is None:
            raise HTTPException(404, detail=f"Document {doc_id!r} not found.")
        return PlainTextResponse(content=node.text, media_type="text/plain; charset=utf-8")

    @app.get("/preview/{doc_id}", tags=["documents"])
    async def get_preview(
        doc_id: str,
        query: str = "",
        context_chars: int = 400,
    ) -> JSONResponse:
        """
        Return document text with the position of the best-matching span
        for the given query, so the frontend can render a highlighted preview.

        Response shape
        --------------
        {
            "doc_id":     "...",
            "title":      "...",
            "file_type":  "...",
            "text":       "full cleaned markdown text",
            "match_start": N,   # char offset of match start (-1 if no match)
            "match_end":   M,   # char offset of match end   (-1 if no match)
            "snippet":     "... short context window around the match ..."
        }
        """
        node = workspace_manager.get_document(doc_id)
        if node is None:
            raise HTTPException(404, detail=f"Document {doc_id!r} not found.")

        match_start = -1
        match_end = -1
        snippet = ""

        if query.strip():
            lower_text = node.text.lower()
            # Find the earliest word in the query that matches
            for word in query.lower().split():
                if len(word) < 3:  # skip very short stop words
                    continue
                idx = lower_text.find(word)
                if idx != -1:
                    match_start = idx
                    match_end = idx + len(word)
                    # Build context snippet around the match
                    start = max(0, idx - context_chars // 2)
                    end = min(len(node.text), idx + context_chars // 2)
                    prefix = "\u2026" if start > 0 else ""
                    suffix = "\u2026" if end < len(node.text) else ""
                    snippet = prefix + node.text[start:end].strip() + suffix
                    break

        if not snippet and node.text:
            snippet = node.text[:context_chars].strip()
            if len(node.text) > context_chars:
                snippet += "\u2026"

        return JSONResponse(
            {
                "doc_id": node.doc_id,
                "title": node.title,
                "file_type": node.file_type,
                "source_path": node.source_path,
                "text": node.text,
                "match_start": match_start,
                "match_end": match_end,
                "snippet": snippet,
            }
        )

    @app.get("/docs/{doc_id}", tags=["documents"])
    async def get_raw_doc(doc_id: str) -> FileResponse:
        node = workspace_manager.get_document(doc_id)
        if node is None:
            raise HTTPException(404, detail=f"Document {doc_id!r} not found.")
        source_path = node.source_path
        if not os.path.isfile(source_path):
            raise HTTPException(
                410,
                detail=f"Source file no longer exists at {source_path!r}. Re-run ingestion.",
            )
        media_type, _ = mimetypes.guess_type(source_path)
        return FileResponse(
            path=source_path,
            media_type=media_type or "application/octet-stream",
            filename=os.path.basename(source_path),
        )

    # Resolve the docs/ directory relative to the installed package root.
    # When pip-installed, __file__ is inside site-packages/local_search_agent/server/,
    # so parents[1] is the package root (local_search_agent/) and the docs/ folder
    # sits one level above that alongside pyproject.toml — but that only exists in
    # the source tree. For installed packages we ship docs/ inside the package so
    # it is always co-located with the Python code.
    _pkg_root = Path(__file__).resolve().parent.parent  # local_search_agent/
    _docs_dir = _pkg_root.parent / "docs"  # source-tree location
    if not _docs_dir.is_dir():
        _docs_dir = _pkg_root / "docs"  # installed package location
    if _docs_dir.is_dir():
        _generate_docs_index(_docs_dir)
        app.mount("/help", StaticFiles(directory=str(_docs_dir), html=True), name="docs")
        logger.info("Documentation endpoint mounted at /help (source: %s)", _docs_dir)
    else:
        logger.warning("docs/ directory not found at %s — /help endpoint not available.", _docs_dir)

    return app


def _generate_docs_index(docs_dir: Path) -> None:
    from local_search_agent.core.constants import __version__

    groups = [
        (
            "Introduction",
            [
                "architecture.md",
                "getting-started.md",
                "installation.md",
                "configuration.md",
            ],
        ),
        (
            "Core Concepts",
            [
                "ingestion.md",
                "semantic-search.md",
                "multi-workspace.md",
            ],
        ),
        (
            "Reference",
            [
                "cli-reference.md",
                "api-reference.md",
            ],
        ),
        (
            "Help",
            [
                "troubleshooting.md",
            ],
        ),
    ]

    def _clean_name(stem: str):
        return stem.replace("-", " ").replace("_", " ").title()

    group_items = []
    default_file = ""

    for group_title, file_list in groups:
        existing = [f for f in file_list if (docs_dir / f).is_file()]
        if not existing:
            continue

        inner = []
        for f in existing:
            inner.append(f'<li><a href="#" data-file="{f}">{_clean_name(Path(f).stem)}</a></li>')

        if not default_file:
            default_file = existing[0]

        group_items.append(f"""
        <div class="group">
          <div class="group-title">{group_title}</div>
          <ul class="group-list">
            {"".join(inner)}
          </ul>
        </div>
        """)

    nav_items = "".join(group_items)
    if not nav_items:
        return

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Docs</title>

<script src="https://cdnjs.cloudflare.com/ajax/libs/marked/12.0.0/marked.min.js"></script>
<link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-tomorrow.min.css" rel="stylesheet"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js"></script>

<style>
:root {{
  --bg:#F8FAFC;
  --surface:#fff;
  --border:#E2E8F0;
  --text:#0F172A;
  --muted:#64748B;
  --accent:#4F6EF7;
  --code:#EEF2F7;
  --radius:10px;
  --sidebar:260px;
  --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}}

[data-theme="dark"] {{
  --bg:#0B1220;
  --surface:#0F172A;
  --border:#1F2937;
  --text:#E5E7EB;
  --muted:#94A3B8;
  --code:#0F172A;
  --accent:#7C9CFF;
}}

* {{ box-sizing:border-box; margin:0; padding:0; }}

body {{
  font-family:system-ui,-apple-system,Segoe UI,Roboto;
  background:var(--bg);
  color:var(--text);
  height:100vh;
  display:flex;
  flex-direction:column;
  overflow:hidden;
}}

header {{
  display:flex;
  align-items:center;
  padding:14px 18px;
  background:var(--surface);
  border-bottom:1px solid var(--border);
}}

header h1 {{ font-size:14px; font-weight:700; }}

.version {{
  margin-left:10px;
  font-size:11px;
  color:var(--muted);
  border:1px solid var(--border);
  padding:2px 8px;
  border-radius:999px;
}}

button.theme-btn {{
  margin-left:auto;
  width:34px; height:34px;
  border-radius:8px;
  border:1px solid var(--border);
  background:var(--surface);
  display:flex; align-items:center; justify-content:center;
  cursor:pointer;
  font-size:16px;
}}

#layout {{ display:flex; flex:1; overflow:hidden; }}

nav {{
  width:var(--sidebar);
  background:var(--surface);
  border-right:1px solid var(--border);
  overflow:auto;
  padding:12px;
}}

.group-title {{
  font-size:11px;
  text-transform:uppercase;
  letter-spacing:.08em;
  color:var(--text);
  font-weight:700;
  margin:14px 8px 8px;
}}

nav a {{
  display:block;
  padding:7px 10px;
  margin:2px 4px;
  border-radius:6px;
  text-decoration:none;
  color:var(--muted);
  font-size:13px;
}}

nav a:hover {{ background:#EEF2FF; color:var(--text); }}
nav a.active {{ background:#EEF2FF; color:var(--accent); font-weight:600; }}

#content {{ flex:1; overflow:auto; padding:48px 24px; }}

.inner {{ max-width:860px; margin:0 auto; line-height:1.75; }}
.inner p {{ margin:12px 0; }}

code:not(pre code) {{
  background:#FEE2E2;
  color:#B91C1C;
  padding:2px 6px;
  border-radius:6px;
}}

pre {{
  position:relative;
  background:var(--code);
  border:1px solid var(--border);
  border-radius:10px;
  padding:16px;
  margin:18px 0;
  overflow:auto;
}}

pre code {{ font-family:var(--mono); font-size:13px; }}

.copy-btn {{
  position:absolute; top:8px; right:8px;
  width:30px; height:30px;
  border-radius:8px;
  border:1px solid var(--border);
  background:var(--surface);
  display:flex; align-items:center; justify-content:center;
  cursor:pointer;
}}

.copy-btn svg {{ width:15px; height:15px; fill:var(--muted); }}
</style>
</head>

<body>
<header>
  <h1>Docs</h1>
  <span class="version">v{__version__}</span>
  <button class="theme-btn" id="themeBtn" onclick="toggleTheme()">🌙</button>
</header>

<div id="layout">
  <nav>{nav_items}</nav>
  <div id="content">
    <div class="inner">Loading...</div>
  </div>
</div>

<script>
const defaultFile = {repr(default_file)};
const links = document.querySelectorAll("nav a");

/* ── THEME ──────────────────────────────────────────────────────────────── */
function _applyTheme(theme) {{
  document.documentElement.setAttribute("data-theme", theme);
  document.getElementById("themeBtn").textContent = theme === "dark" ? "☀️" : "🌙";
}}
function toggleTheme() {{
  const next = (document.documentElement.getAttribute("data-theme") || "light") === "dark"
    ? "light" : "dark";
  localStorage.setItem("theme", next);
  _applyTheme(next);
}}
_applyTheme(localStorage.getItem("theme") || "light");

/* ── COPY BUTTONS ───────────────────────────────────────────────────────── */
function addCopyButtons() {{
  document.querySelectorAll("pre").forEach(pre => {{
    if (pre.querySelector(".copy-btn")) return;
    const btn = document.createElement("button");
    btn.className = "copy-btn";
    btn.innerHTML = `<svg viewBox="0 0 24 24"><path d="M16 1H4a2 2 0 0 0-2 2v14h2V3h12V1zm4 4H8a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2zm0 18H8V7h12v16z"/></svg>`;
    btn.onclick = () => navigator.clipboard.writeText(pre.innerText);
    pre.appendChild(btn);
  }});
}}

/* ── MARKED RENDERER ────────────────────────────────────────────────────
   marked v12 via new marked.Renderer() uses positional args:
     link(href, title, text)
   NOT a destructured token object. The token API only applies when using
   marked.use() with useNewRenderer:true (v13+).

   We rewrite .md hrefs to href="#" and store the filename in data-md so
   the browser never sees a navigable URL. Without this, the browser starts
   navigation synchronously on click — before any JS handler runs — which
   is why e.preventDefault() in a delegated listener was always too late.
────────────────────────────────────────────────────────────────────────── */
const renderer = new marked.Renderer();
renderer.link = function(href, title, text) {{
  if (href && href.endsWith(".md")) {{
    const file = href.split("/").pop();
    return `<a href="#" data-md="${{file}}">${{text}}</a>`;
  }}
  const titleAttr = title ? ` title="${{title}}"` : "";
  return `<a href="${{href}}"${{titleAttr}} target="_blank" rel="noopener">${{text}}</a>`;
}};
marked.setOptions({{ renderer }});

/* ── INTERNAL LINK CLICKS (attached once) ───────────────────────────────── */
document.querySelector(".inner").addEventListener("click", (e) => {{
  const a = e.target.closest("a[data-md]");
  if (!a) return;
  e.preventDefault();
  loadDoc(a.dataset.md);
}});

/* ── LOAD DOC ───────────────────────────────────────────────────────────── */
async function loadDoc(file) {{
  const res = await fetch("/help/" + file);
  const md = await res.text();

  const inner = document.querySelector(".inner");
  inner.innerHTML = marked.parse(md);

  Prism.highlightAll();
  addCopyButtons();

  document.getElementById("content").scrollTop = 0;

  links.forEach(l => l.classList.toggle("active", l.dataset.file === file));
}}

links.forEach(l => {{
  l.onclick = e => {{ e.preventDefault(); loadDoc(l.dataset.file); }};
}});

loadDoc(defaultFile);
</script>

</body>
</html>
"""

    (docs_dir / "index.html").write_text(html, encoding="utf-8")
