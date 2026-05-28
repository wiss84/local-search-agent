"""
CLI entry point for the Local Search Agent framework.

Usage
-----
    python -m local_search_agent [command] [options]

Or after pip install:
    local-search [command] [options]

Commands (Phase 1)
------------------
    serve       Start the FastAPI file server.
    workspace   Manage workspaces (create, list, delete).

Commands (Phase 2+, stubs)
--------------------------
    ingest      Ingest and index documents.
    query       Ask the agent a question (interactive or one-shot).
"""

from local_search_agent.cli.commands import main

if __name__ == "__main__":
    main()
