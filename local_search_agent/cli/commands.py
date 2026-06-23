"""
CLI commands for the Local Search Agent framework.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Terminal UI helpers
# ---------------------------------------------------------------------------


def _print_banner() -> None:
    """Print the AGENT ASCII art banner using pyfiglet + rich."""
    try:
        from pyfiglet import figlet_format
        from rich.console import Console
        from rich.rule import Rule
        from rich.text import Text

        console = Console()
        banner = figlet_format("AGENT", font="slant")
        console.print(Text(banner, style="bold cyan"))
        console.print(Rule(style="cyan"))
        console.print(
            "  [bold white]Local Search Agent[/bold white]  "
            "[dim]deterministic, auditable, local-first[/dim]",
            justify="center",
        )
        console.print(Rule(style="cyan"))
        console.print()
    except Exception:
        print("=" * 60)
        print("  LOCAL SEARCH AGENT")
        print("=" * 60)
        print()


def _console():
    """Return a rich Console instance, falling back to plain print if unavailable."""
    try:
        from rich.console import Console

        return Console()
    except ImportError:
        return None


def _print_answer(answer: str, iterations: int, truncated: bool) -> None:
    """Render the agent answer in a styled rich panel."""
    try:
        from rich.console import Console
        from rich.markdown import Markdown
        from rich.panel import Panel

        console = Console()
        console.print()
        console.print(
            Panel(
                Markdown(answer),
                title="[bold green]Answer[/bold green]",
                border_style="green",
                padding=(1, 2),
            )
        )
        footer = f"Iterations used: {iterations}"
        if truncated:
            footer += "  [yellow]Answer may be incomplete (max iterations reached).[/yellow]"
        console.print(f"  [dim]{footer}[/dim]")
        console.print()
    except Exception:
        print("\n" + "=" * 60)
        print(answer)
        print("=" * 60)
        print(f"Iterations used: {iterations}")
        if truncated:
            print("Answer may be incomplete (max iterations reached).")
        print()


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def cmd_config_set_key(args: argparse.Namespace) -> None:
    """Save an API key for a provider."""
    from local_search_agent.core.key_manager import keys_file_path, set_key

    try:
        set_key(args.provider, args.key)
        print(f"\u2713 API key saved for provider '{args.provider}'.")
        print(f"  Stored at: {keys_file_path()}")
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_config_delete_key(args: argparse.Namespace) -> None:
    """Remove the saved API key for a provider."""
    from local_search_agent.core.key_manager import delete_key

    deleted = delete_key(args.provider)
    if deleted:
        print(f"\u2713 API key for '{args.provider}' removed.")
    else:
        print(f"No saved key found for provider '{args.provider}'.")


def cmd_config_list_keys(args: argparse.Namespace) -> None:
    """List all saved API keys (masked)."""
    from local_search_agent.core.key_manager import keys_file_path, list_keys

    keys = list_keys()
    print(f"Saved keys ({keys_file_path()}):")
    if not keys:
        print("  (none)")
        return
    for provider, masked in keys.items():
        print(f"  {provider:<12} {masked}")


def cmd_config_add_model(args: argparse.Namespace) -> None:
    """Add a model name for a provider."""
    from local_search_agent.core.key_manager import add_model, models_file_path

    try:
        add_model(args.provider, args.model_name)
        print(f"\u2713 Model '{args.model_name}' added for provider '{args.provider}'.")
        print(f"  Stored at: {models_file_path()}")
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_config_delete_model(args: argparse.Namespace) -> None:
    """Remove a model name for a provider."""
    from local_search_agent.core.key_manager import delete_model

    deleted = delete_model(args.provider, args.model_name)
    if deleted:
        print(f"\u2713 Model '{args.model_name}' removed from provider '{args.provider}'.")
    else:
        print(f"Model '{args.model_name}' not found for provider '{args.provider}'.")


def cmd_config_list_models(args: argparse.Namespace) -> None:
    """List all saved model names per provider."""
    from local_search_agent.core.key_manager import get_models, models_file_path

    all_models = get_models()
    print(f"Saved models ({models_file_path()}):")
    for provider, models in all_models.items():
        if models:
            print(f"  {provider}:")
            for m in models:
                print(f"    - {m}")
        else:
            print(f"  {provider}: (none)")


def cmd_config_set_semantic(args: argparse.Namespace) -> None:
    """Configure semantic features: toggles and model/provider overrides."""
    from local_search_agent.core.key_manager import (
        get_semantic_settings,
        set_all_semantic_settings,
        settings_file_path,
    )

    current = get_semantic_settings()
    changed = False

    if args.enable is not None:
        current["enable_semantic"] = args.enable
        changed = True
        print(f"\u2713 semantic indexing {'enabled' if args.enable else 'disabled'}.")

    if args.query_expansion is not None:
        current["enable_query_expansion"] = args.query_expansion
        changed = True
        print(f"\u2713 query expansion {'enabled' if args.query_expansion else 'disabled'}.")

    if args.provider is not None:
        val = "" if args.provider.lower() in ("none", "default") else args.provider
        current["semantic_provider"] = val
        changed = True
        print(f"\u2713 semantic provider set to {(val or '(default - uses main provider)'):!r}.")

    if args.model is not None:
        val = "" if args.model.lower() in ("none", "default") else args.model
        current["semantic_model"] = val
        changed = True
        print(f"\u2713 semantic model set to {(val or '(default - uses main model)'):!r}.")

    if changed:
        set_all_semantic_settings(**current)
        print(f"  Stored at: {settings_file_path()}")
    else:
        print("Nothing changed. Use --enable, --query-expansion, --provider, or --model flags.")
        print("  Run 'local-search config show-semantic' to see current settings.")


def cmd_config_show_semantic(args: argparse.Namespace) -> None:
    """Show current semantic feature flag settings."""
    from local_search_agent.core.key_manager import get_semantic_settings, settings_file_path

    settings = get_semantic_settings()
    print(f"Semantic settings ({settings_file_path()}):")
    print(f"  {'enable_semantic':<28} {'ON' if settings['enable_semantic'] else 'off'}")
    print(
        f"  {'enable_query_expansion':<28} {'ON' if settings['enable_query_expansion'] else 'off'}"
    )
    sem_provider = settings.get("semantic_provider") or "(default - uses main provider)"
    sem_model = settings.get("semantic_model") or "(default - uses main model)"
    print(f"  {'semantic_provider':<28} {sem_provider}")
    print(f"  {'semantic_model':<28} {sem_model}")


def cmd_config_show(args: argparse.Namespace) -> None:
    """Show all current config -- keys, models, semantic settings, LangSmith."""
    from local_search_agent.core.constants import __version__
    from local_search_agent.core.key_manager import (
        advanced_settings_file_path,
        get_advanced_settings,
        get_effective_constants,
        get_langsmith,
        get_models,
        get_semantic_settings,
        keys_file_path,
        list_keys,
        models_file_path,
        settings_file_path,
    )

    print(f"Local Search Agent v{__version__}")
    print("=" * 60)

    # API Keys
    print(f"\nAPI Keys  ({keys_file_path()}):")
    keys = {k: v for k, v in list_keys().items() if not k.startswith("langsmith")}
    if keys:
        for provider, masked in keys.items():
            print(f"  {provider:<12} {masked}")
    else:
        print("  (none saved)")

    # Models
    print(f"\nModels  ({models_file_path()}):")
    for provider, models in get_models().items():
        label = ", ".join(models) if models else "(none)"
        print(f"  {provider:<12} {label}")

    # Semantic settings
    print(f"\nSemantic Settings  ({settings_file_path()}):")
    s = get_semantic_settings()
    for key, val in s.items():
        if isinstance(val, bool):
            print(f"  {key:<28} {'ON' if val else 'off'}")
        else:
            print(f"  {key:<28} {val or '(default)'}")

    # Advanced settings
    print(f"\nAdvanced Settings  ({advanced_settings_file_path()}):")
    overrides = get_advanced_settings()
    effective = get_effective_constants()
    if not overrides:
        print("  (all defaults)")
    for key, eff_val in effective.items():
        override_marker = " [OVERRIDE]" if key in overrides else ""
        print(f"  {key:<35} {eff_val}{override_marker}")

    # LangSmith
    print("\nLangSmith Tracing:")
    ls = get_langsmith()
    if ls["configured"]:
        print(f"  Configured   {ls['api_key_masked']}  project={ls['project']}")
    else:
        print("  Not configured")


def cmd_config_set_advanced(args: argparse.Namespace) -> None:
    """Set one advanced setting override, or reset all to defaults."""
    from local_search_agent.core.key_manager import (
        advanced_settings_file_path,
        get_advanced_settings,
        get_effective_constants,
        set_advanced_settings,
    )

    if args.reset:
        set_advanced_settings({})
        print("\u2713 All advanced settings reset to compiled-in defaults.")
        print(f"  File: {advanced_settings_file_path()}")
        return

    if args.key is None or args.value is None:
        print("Provide --key and --value, or --reset.")
        print("  Example: local-search config set-advanced --key PDF_PAGES_PER_BATCH --value 10")
        return

    current = get_advanced_settings()
    current[args.key] = args.value
    set_advanced_settings(current)
    effective = get_effective_constants()
    print(f"\u2713 {args.key} = {effective.get(args.key, args.value)}")
    print(f"  Stored at: {advanced_settings_file_path()}")


def cmd_setup(args: argparse.Namespace) -> None:
    """Download the Meilisearch binary for the current platform."""
    from local_search_agent.core.meilisearch_manager import run_setup

    run_setup(force=args.force)


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the FastAPI file server, optionally with the incremental scheduler."""
    from local_search_agent.core.config import SearchAgentConfig
    from local_search_agent.core.framework import SearchAgentFramework

    config = SearchAgentConfig(
        document_dirs=args.dirs or [],
        workspace_name=args.workspace,
        host=args.host,
        port=args.port,
        meilisearch_url=args.meili_url,
        meili_master_key=args.meili_key,
        provider="ollama",
        db_path=args.db,
    )
    framework = SearchAgentFramework(config)

    if args.dirs:
        for d in args.dirs:
            framework.create_workspace(name=args.workspace, document_dir=d)
        print(f"Ingesting {args.dirs} into workspace {args.workspace!r} ...")
        stats = framework.ingest_and_index()
        print(f"Done. {stats}")

    if args.scheduler:
        framework.start_incremental_scheduler(interval_minutes=args.interval)
        print(f"Incremental scheduler started (every {args.interval}m).")

    print(f"Starting file server on http://{args.host}:{args.port} ...")
    framework.start_file_server(block=True)


# ---------------------------------------------------------------------------
# workspace
# ---------------------------------------------------------------------------


def cmd_workspace_create(args: argparse.Namespace) -> None:
    from local_search_agent.workspace.metadata_db import MetadataDB
    from local_search_agent.workspace.workspace_manager import WorkspaceManager

    wm = WorkspaceManager(db_path=args.db)
    mdb = MetadataDB(db_path=args.db)
    wm.create_workspace(name=args.name, document_dir=args.dir)
    mdb.upsert_sync_job(workspace=args.name)
    print(f"Workspace {args.name!r} created -> {args.dir}")


def cmd_workspace_list(args: argparse.Namespace) -> None:
    from local_search_agent.core.config import SearchAgentConfig
    from local_search_agent.core.framework import SearchAgentFramework

    config = SearchAgentConfig(workspace_name="default", db_path=args.db)
    framework = SearchAgentFramework(config)
    workspaces = framework.list_workspaces()
    if not workspaces:
        print("No workspaces registered.")
        return
    print(f"{'Name':<25} {'Document Directory'}")
    print("-" * 70)
    for ws in workspaces:
        print(f"  {ws['name']:<23} {ws['document_dir']}")


def cmd_workspace_delete(args: argparse.Namespace) -> None:
    from local_search_agent.core.config import SearchAgentConfig
    from local_search_agent.core.framework import SearchAgentFramework

    config = SearchAgentConfig(workspace_name=args.name, db_path=args.db)
    framework = SearchAgentFramework(config)
    framework.delete_workspace(name=args.name, wipe_index=args.wipe)
    print(f"Workspace {args.name!r} deleted" + (" (index wiped)." if args.wipe else "."))


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------


def cmd_ingest(args: argparse.Namespace) -> None:
    from local_search_agent.core.config import SearchAgentConfig
    from local_search_agent.core.framework import SearchAgentFramework

    config = SearchAgentConfig(
        document_dirs=args.dirs,
        workspace_name=args.workspace,
        meilisearch_url=args.meili_url,
        meili_master_key=args.meili_key,
        provider="ollama",
        db_path=args.db,
    )
    framework = SearchAgentFramework(config)

    if args.wipe:
        print(f"Wiping index and re-ingesting workspace {args.workspace!r} ...")
        stats = framework.wipe_and_reingest(workspace_name=args.workspace)
    else:
        print(f"Ingesting {args.dirs} into workspace {args.workspace!r} ...")
        stats = framework.ingest_and_index(force=args.force)

    print(f"Done. {stats}")
    if stats.errors:
        print(f"\n{len(stats.errors)} error(s):")
        for err in stats.errors[:10]:
            print(f"  - {err}")
        if len(stats.errors) > 10:
            print(f"  ... and {len(stats.errors) - 10} more. Check logs for details.")


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------


def cmd_query(args: argparse.Namespace) -> None:
    from local_search_agent.core.config import SearchAgentConfig
    from local_search_agent.core.framework import SearchAgentFramework

    provider = args.provider
    api_key = args.api_key or None

    config = SearchAgentConfig(
        workspace_name=args.workspace,
        provider=provider,
        api_key=api_key,
        model_name=args.model,
        meilisearch_url=args.meili_url,
        meili_master_key=args.meili_key,
        max_iterations=args.max_iterations,
        top_k=args.top_k,
        db_path=args.db,
    )
    framework = SearchAgentFramework(config)

    if args.question:
        try:
            from rich.console import Console
            from rich.status import Status

            console = Console()
            with Status("[cyan]Searching...[/cyan]", console=console):
                response = framework.query(question=args.question, workspace=args.workspace)
        except ImportError:
            print(f"Searching... (max {args.max_iterations} iterations)")
            response = framework.query(question=args.question, workspace=args.workspace)
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        _print_answer(response["answer"], response["iterations_used"], response["truncated"])
        return

    _print_banner()

    try:
        from rich.console import Console
        from rich.prompt import Prompt
        from rich.status import Status

        console = Console()
        console.print(
            f"  Workspace: [bold cyan]{args.workspace}[/bold cyan]  "
            f"Provider: [bold cyan]{provider}[/bold cyan]  "
            f"Model: [bold cyan]{args.model}[/bold cyan]"
        )
        console.print(
            "  Type your question and press Enter. Type [bold]exit[/bold] or Ctrl+C to quit.\n"
        )
        use_rich = True
    except ImportError:
        console = None
        use_rich = False
        print(f"Workspace: {args.workspace}  Provider: {provider}  Model: {args.model}")
        print("Type your question and press Enter. Type 'exit' or Ctrl+C to quit.\n")

    while True:
        try:
            if use_rich:
                question = Prompt.ask("[bold cyan]You[/bold cyan]")
            else:
                question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        question = question.strip()
        if not question:
            continue
        if question.lower() in ("exit", "quit", "q", ":q"):
            print("Goodbye.")
            break

        try:
            if use_rich:
                with Status("[cyan]Searching...[/cyan]", console=console):
                    response = framework.query(question=question, workspace=args.workspace)
            else:
                print("Searching...")
                response = framework.query(question=question, workspace=args.workspace)
        except Exception as e:
            if use_rich:
                console.print(f"[red]ERROR:[/red] {e}")
            else:
                print(f"ERROR: {e}", file=sys.stderr)
            continue

        _print_answer(response["answer"], response["iterations_used"], response["truncated"])


# ---------------------------------------------------------------------------
# scheduler
# ---------------------------------------------------------------------------


def cmd_scheduler_start(args: argparse.Namespace) -> None:
    """Start the incremental scheduler as a foreground process (blocks)."""
    import signal
    import time

    from local_search_agent.core.config import SearchAgentConfig
    from local_search_agent.core.framework import SearchAgentFramework

    config = SearchAgentConfig(
        document_dirs=args.dirs or [],
        workspace_name=args.workspace,
        meilisearch_url=args.meili_url,
        meili_master_key=args.meili_key,
        provider="ollama",
        db_path=args.db,
    )
    framework = SearchAgentFramework(config)

    if args.dirs:
        for d in args.dirs:
            framework.create_workspace(name=args.workspace, document_dir=d)

    framework.start_incremental_scheduler(interval_minutes=args.interval)
    print(
        f"Incremental scheduler running (workspace={args.workspace!r}, "
        f"interval={args.interval}m). Press Ctrl+C to stop."
    )

    def _shutdown(sig, frame):
        print("\nShutting down scheduler...")
        framework.stop_incremental_scheduler()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    while True:
        time.sleep(5)


def cmd_scheduler_status(args: argparse.Namespace) -> None:
    """DEPRECATED (use 'watch status'): Show scheduler status -- which workspaces are scheduled and next run times."""
    from local_search_agent.core.config import SearchAgentConfig
    from local_search_agent.core.framework import SearchAgentFramework

    config = SearchAgentConfig(workspace_name="default", db_path=args.db)
    framework = SearchAgentFramework(config)
    status = framework.get_scheduler_status()

    if not status["running"]:
        print("Scheduler is not running.")
        return

    jobs = status.get("scheduled_jobs", [])
    print(f"Scheduler running -- {len(jobs)} job(s)")
    print("-" * 60)
    for job in jobs:
        print(
            f"  workspace={job.get('workspace', '?'):<25} "
            f"interval={job.get('interval_minutes', '?')}m  "
            f"next_run={job.get('next_run', 'unknown')}"
        )


def cmd_scheduler_trigger(args: argparse.Namespace) -> None:
    """DEPRECATED (use 'watch trigger'): Trigger an immediate sync for a workspace."""
    from local_search_agent.core.config import SearchAgentConfig
    from local_search_agent.core.framework import SearchAgentFramework

    config = SearchAgentConfig(
        document_dirs=args.dirs or [],
        workspace_name=args.workspace,
        meilisearch_url=args.meili_url,
        meili_master_key=args.meili_key,
        provider="ollama",
        db_path=args.db,
    )
    framework = SearchAgentFramework(config)
    print(f"Triggering immediate sync for workspace {args.workspace!r} ...")
    stats = framework.ingest_and_index(force=args.force)
    print(f"Done. {stats}")


# ---------------------------------------------------------------------------
# watch (filesystem-event-driven, replaces the polling scheduler)
# ---------------------------------------------------------------------------


def cmd_watch_start(args: argparse.Namespace) -> None:
    """Start Watch Mode as a foreground process (blocks). Reacts to file changes instantly."""
    import signal
    import time

    from local_search_agent.core.config import SearchAgentConfig
    from local_search_agent.core.framework import SearchAgentFramework

    config = SearchAgentConfig(
        document_dirs=args.dirs or [],
        workspace_name=args.workspace,
        meilisearch_url=args.meili_url,
        meili_master_key=args.meili_key,
        provider="ollama",
        db_path=args.db,
        enrich_on_watch=not args.no_enrich,
    )
    framework = SearchAgentFramework(config)

    if args.dirs:
        for d in args.dirs:
            framework.create_workspace(name=args.workspace, document_dir=d)

    framework.start_watch_mode()
    print(
        f"Watch mode running (workspace={args.workspace!r}, "
        f"enrich_on_watch={not args.no_enrich}). Press Ctrl+C to stop."
    )

    def _shutdown(sig, frame):
        print("\nShutting down watch mode...")
        framework.stop_watch_mode()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    while True:
        time.sleep(5)


def cmd_watch_status(args: argparse.Namespace) -> None:
    """Show watch-mode status -- which workspaces/directories are being watched."""
    from local_search_agent.core.config import SearchAgentConfig
    from local_search_agent.core.framework import SearchAgentFramework

    config = SearchAgentConfig(workspace_name="default", db_path=args.db)
    framework = SearchAgentFramework(config)
    status = framework.get_watch_mode_status()

    if not status["running"]:
        print("Watch mode is not running.")
        return

    watched = status.get("watched_directories", {})
    print(f"Watch mode running -- {len(watched)} workspace(s)")
    print("-" * 60)
    for workspace, dir_count in watched.items():
        print(f"  workspace={workspace:<25} watched_directories={dir_count}")


def cmd_watch_trigger(args: argparse.Namespace) -> None:
    """Trigger an immediate sync for a workspace (bypassing the debounce window)."""
    from local_search_agent.core.config import SearchAgentConfig
    from local_search_agent.core.framework import SearchAgentFramework

    config = SearchAgentConfig(
        document_dirs=args.dirs or [],
        workspace_name=args.workspace,
        meilisearch_url=args.meili_url,
        meili_master_key=args.meili_key,
        provider="ollama",
        db_path=args.db,
        enrich_on_watch=not args.no_enrich,
    )
    framework = SearchAgentFramework(config)
    print(f"Triggering immediate sync for workspace {args.workspace!r} ...")
    stats = framework.ingest_and_index(force=args.force)
    print(f"Done. {stats}")


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------


def cmd_health(args: argparse.Namespace) -> None:
    """Show index health and freshness across all workspaces."""
    from local_search_agent.scheduler.monitor import IndexMonitor
    from local_search_agent.workspace.metadata_db import MetadataDB

    mdb = MetadataDB(db_path=args.db)
    monitor = IndexMonitor(metadata_db=mdb, stale_threshold_minutes=args.stale_threshold)
    summary = monitor.get_health_summary()

    print(f"Index Health Summary -- {summary.total_workspaces} workspace(s)")
    print("-" * 60)
    print(f"  Healthy      : {summary.healthy}")
    print(f"  Stale        : {summary.stale}")
    print(f"  Never synced : {summary.never_synced}")
    print(f"  Error        : {summary.error}")
    print(f"  Running      : {summary.running}")
    print(f"  Total docs   : {summary.total_docs}")
    print()

    if not summary.workspaces:
        print("No workspaces registered. Run 'local-search workspace create' first.")
        return

    for ws in summary.workspaces:
        age_str = f"{ws.age_minutes:.0f}m ago" if ws.age_minutes is not None else "never"
        status_icon = {
            "healthy": "\u2713",
            "stale": "\u26a0",
            "never_synced": "\u25cb",
            "error": "\u2717",
            "running": "\u21bb",
        }.get(ws.status, "?")
        print(
            f"  {status_icon} {ws.workspace:<25} "
            f"status={ws.status:<12} "
            f"docs={ws.doc_count:<6} "
            f"last_sync={age_str}"
        )
        if ws.last_error:
            print(f"    \u2514\u2500 Error: {ws.last_error}")

    if not summary.all_healthy:
        print(
            "\n\u26a0  Some workspaces need attention. "
            "Run 'local-search scheduler trigger --workspace <name>' to sync manually."
        )


# ---------------------------------------------------------------------------
# ui (desktop dashboard)
# ---------------------------------------------------------------------------


def cmd_ui(args: argparse.Namespace) -> None:
    """Open the desktop dashboard window."""
    from local_search_agent.ui.dashboard import run

    run(
        host=args.host,
        port=args.port,
        db_path=args.db,
        provider=args.provider,
        model=args.model,
        meili_url=args.meili_url,
        meili_key=args.meili_key,
        scheduler_interval=args.scheduler_interval,
        open_window=not args.headless,
    )


# ---------------------------------------------------------------------------
# Parser assembly
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="local-search",
        description="Local Search Agent -- deterministic, auditable local document RAG.",
    )
    from local_search_agent.core.config import _default_db_path

    parser.add_argument(
        "--db",
        default=os.environ.get("LSA_DB_PATH") or _default_db_path(),
        help="SQLite metadata database path (default: user config dir).",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # -- config --------------------------------------------------------------
    p_config = sub.add_parser("config", help="Manage configuration (API keys etc.).")
    config_sub = p_config.add_subparsers(dest="config_command", required=True)

    p_cfg_set = config_sub.add_parser("set-key", help="Save an API key for a provider.")
    p_cfg_set.add_argument(
        "--provider", required=True, choices=["google", "openai", "anthropic"], help="LLM provider."
    )
    p_cfg_set.add_argument("--key", required=True, help="Your API key.")
    p_cfg_set.set_defaults(func=cmd_config_set_key)

    p_cfg_del = config_sub.add_parser("delete-key", help="Remove a saved API key.")
    p_cfg_del.add_argument("--provider", required=True, choices=["google", "openai", "anthropic"])
    p_cfg_del.set_defaults(func=cmd_config_delete_key)

    p_cfg_list = config_sub.add_parser("list-keys", help="List all saved API keys (masked).")
    p_cfg_list.set_defaults(func=cmd_config_list_keys)

    p_cfg_add_model = config_sub.add_parser("add-model", help="Add a model name for a provider.")
    p_cfg_add_model.add_argument(
        "--provider", required=True, choices=["google", "openai", "anthropic", "ollama"]
    )
    p_cfg_add_model.add_argument("--model-name", required=True, help="Model name to add.")
    p_cfg_add_model.set_defaults(func=cmd_config_add_model)

    p_cfg_del_model = config_sub.add_parser(
        "delete-model", help="Remove a model name for a provider."
    )
    p_cfg_del_model.add_argument(
        "--provider", required=True, choices=["google", "openai", "anthropic", "ollama"]
    )
    p_cfg_del_model.add_argument("--model-name", required=True, help="Model name to remove.")
    p_cfg_del_model.set_defaults(func=cmd_config_delete_model)

    p_cfg_list_models = config_sub.add_parser(
        "list-models", help="List all saved model names per provider."
    )
    p_cfg_list_models.set_defaults(func=cmd_config_list_models)

    # set-semantic: all flags, can be combined in one call
    p_cfg_semantic = config_sub.add_parser(
        "set-semantic",
        help="Configure semantic search settings.",
        description=(
            "Configure semantic search features. All flags are optional and can be combined.\n\n"
            "Examples:\n"
            "  local-search config set-semantic --enable true\n"
            "  local-search config set-semantic --provider google --model gemma-4-26b-a4b-it\n"
            "  local-search config set-semantic --enable true --query-expansion true "
            "--provider google --model gemma-4-26b-a4b-it\n"
            "  local-search config set-semantic --model none  # reset to main model"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_cfg_semantic.add_argument(
        "--enable",
        type=lambda v: v.lower() in ("true", "1", "on", "yes", "enable"),
        metavar="true|false",
        default=None,
        help="Enable or disable semantic indexing (ConceptCompiler + StructuralParser).",
    )
    p_cfg_semantic.add_argument(
        "--query-expansion",
        type=lambda v: v.lower() in ("true", "1", "on", "yes", "enable"),
        metavar="true|false",
        default=None,
        dest="query_expansion",
        help="Enable or disable query expansion at search time.",
    )
    p_cfg_semantic.add_argument(
        "--provider",
        default=None,
        choices=["google", "openai", "anthropic", "ollama", "none"],
        help="Provider to use for semantic indexing. 'none' resets to main provider.",
    )
    p_cfg_semantic.add_argument(
        "--model",
        default=None,
        metavar="MODEL_NAME",
        help="Model to use for semantic indexing. 'none' resets to main model.",
    )
    p_cfg_semantic.set_defaults(func=cmd_config_set_semantic)

    p_cfg_show_semantic = config_sub.add_parser(
        "show-semantic", help="Show current semantic feature flag settings."
    )
    p_cfg_show_semantic.set_defaults(func=cmd_config_show_semantic)

    p_cfg_show = config_sub.add_parser(
        "show", help="Show all current config (keys, models, semantic, advanced, LangSmith)."
    )
    p_cfg_show.set_defaults(func=cmd_config_show)

    # set-advanced: override a single ingestion/search constant
    p_cfg_adv = config_sub.add_parser(
        "set-advanced",
        help="Override an ingestion/search constant, or reset all to defaults.",
        description=(
            "Override compiled-in constants stored in advanced_settings.json.\n\n"
            "Examples:\n"
            "  local-search config set-advanced --key PDF_PAGES_PER_BATCH --value 10\n"
            "  local-search config set-advanced --key CHUNK_TARGET_CHARS --value 12000\n"
            "  local-search config set-advanced --reset  # back to all defaults\n\n"
            "Valid keys:\n"
            "  CHUNK_MIN_CHARS, CHUNK_TARGET_CHARS, CHUNK_MAX_CHARS, CHUNK_OVERLAP_CHARS,\n"
            "  TABLE_ROWS_PER_CHUNK,\n"
            "  PDF_PAGES_PER_BATCH, PDF_SPLIT_THRESHOLD, PDF_FALLBACK_PAGES_PER_BATCH,\n"
            "  DOCX_CHAR_SPLIT_THRESHOLD, TESSERACT_FALLBACK_MIN_CHARS,\n"
            "  DEFAULT_TOP_K, DEFAULT_MAX_ITERATIONS, SNIPPET_CONTEXT_CHARS"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_cfg_adv.add_argument(
        "--key", default=None, metavar="CONSTANT_NAME", help="Name of the constant to override."
    )
    p_cfg_adv.add_argument("--value", default=None, metavar="VALUE", help="New value (number).")
    p_cfg_adv.add_argument(
        "--reset", action="store_true", help="Reset ALL advanced settings to compiled-in defaults."
    )
    p_cfg_adv.set_defaults(func=cmd_config_set_advanced)

    # -- setup ---------------------------------------------------------------
    p_setup = sub.add_parser(
        "setup",
        help="Download the Meilisearch binary for this platform (run once after install).",
    )
    p_setup.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the binary already exists.",
    )
    p_setup.set_defaults(func=cmd_setup)

    # -- serve ---------------------------------------------------------------
    p_serve = sub.add_parser("serve", help="Start the FastAPI file server.")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--workspace", default="default")
    p_serve.add_argument("--meili-url", default="http://localhost:7700")
    p_serve.add_argument("--meili-key", default="local_search_master_key")
    p_serve.add_argument("--dirs", nargs="*", metavar="DIR")
    p_serve.add_argument(
        "--scheduler",
        action="store_true",
        help="(deprecated, use 'local-search watch start') Also start the incremental sync scheduler.",
    )
    p_serve.add_argument(
        "--interval",
        type=int,
        default=15,
        help="(deprecated) Scheduler interval in minutes (default 15).",
    )
    p_serve.set_defaults(func=cmd_serve)

    # -- workspace -----------------------------------------------------------
    p_ws = sub.add_parser("workspace", help="Manage workspaces.")
    ws_sub = p_ws.add_subparsers(dest="ws_command", required=True)

    p_ws_create = ws_sub.add_parser("create", help="Register a new workspace.")
    p_ws_create.add_argument("name")
    p_ws_create.add_argument("dir")
    p_ws_create.set_defaults(func=cmd_workspace_create)

    p_ws_list = ws_sub.add_parser("list", help="List registered workspaces.")
    p_ws_list.set_defaults(func=cmd_workspace_list)

    p_ws_delete = ws_sub.add_parser("delete", help="Delete a workspace.")
    p_ws_delete.add_argument("name")
    p_ws_delete.add_argument(
        "--wipe", action="store_true", help="Also delete all documents from the Meilisearch index."
    )
    p_ws_delete.set_defaults(func=cmd_workspace_delete)

    # -- ingest --------------------------------------------------------------
    p_ingest = sub.add_parser("ingest", help="Ingest and index documents into Meilisearch.")
    p_ingest.add_argument("--workspace", default="default")
    p_ingest.add_argument("--dirs", nargs="+", required=True, metavar="DIR")
    p_ingest.add_argument("--meili-url", default="http://localhost:7700")
    p_ingest.add_argument("--meili-key", default="local_search_master_key")
    p_ingest.add_argument(
        "--force", action="store_true", help="Force full re-index (ignore delta logic)."
    )
    p_ingest.add_argument(
        "--wipe",
        action="store_true",
        help="Wipe the index and all DB records, then force full re-ingest.",
    )
    p_ingest.set_defaults(func=cmd_ingest)

    # -- query ---------------------------------------------------------------
    p_query = sub.add_parser("query", help="Ask the agent a question.")
    p_query.add_argument("question", nargs="?", help="Question to ask. Omit for interactive mode.")
    p_query.add_argument("--workspace", default="default")
    p_query.add_argument(
        "--provider", default="google", choices=["google", "ollama", "openai", "anthropic"]
    )
    p_query.add_argument("--model", default="gemma-4-31b-it")
    p_query.add_argument("--api-key", default=None)
    p_query.add_argument("--meili-url", default="http://localhost:7700")
    p_query.add_argument("--meili-key", default="local_search_master_key")
    p_query.add_argument("--max-iterations", type=int, default=10)
    p_query.add_argument("--top-k", type=int, default=5)
    p_query.set_defaults(func=cmd_query)

    # -- scheduler -------------------------------------------------------------
    p_sched = sub.add_parser(
        "scheduler",
        help="(deprecated, use 'watch') Manage the polling-based incremental sync scheduler.",
    )
    sched_sub = p_sched.add_subparsers(dest="sched_command", required=True)

    p_sched_status = sched_sub.add_parser(
        "status",
        help="(deprecated, use 'watch status') Show scheduler status and next run times.",
    )
    p_sched_status.set_defaults(func=cmd_scheduler_status)

    p_sched_start = sched_sub.add_parser(
        "start",
        help="(deprecated, use 'watch start') Run the scheduler as a foreground process.",
    )
    p_sched_start.add_argument("--workspace", default="default")
    p_sched_start.add_argument("--dirs", nargs="*", metavar="DIR")
    p_sched_start.add_argument("--meili-url", default="http://localhost:7700")
    p_sched_start.add_argument("--meili-key", default="local_search_master_key")
    p_sched_start.add_argument(
        "--interval", type=int, default=15, help="Sync interval in minutes (default 15)."
    )
    p_sched_start.set_defaults(func=cmd_scheduler_start)

    p_sched_trigger = sched_sub.add_parser(
        "trigger",
        help="(deprecated, use 'watch trigger') Trigger an immediate one-off sync.",
    )
    p_sched_trigger.add_argument("--workspace", default="default")
    p_sched_trigger.add_argument("--dirs", nargs="*", metavar="DIR")
    p_sched_trigger.add_argument("--meili-url", default="http://localhost:7700")
    p_sched_trigger.add_argument("--meili-key", default="local_search_master_key")
    p_sched_trigger.add_argument("--force", action="store_true", help="Force full re-index.")
    p_sched_trigger.set_defaults(func=cmd_scheduler_trigger)

    # -- watch (filesystem-event-driven, recommended) -------------------------
    p_watch = sub.add_parser(
        "watch", help="Manage Watch Mode (filesystem-event-driven incremental sync)."
    )
    watch_sub = p_watch.add_subparsers(dest="watch_command", required=True)

    p_watch_status = watch_sub.add_parser(
        "status", help="Show watch-mode status and watched directories."
    )
    p_watch_status.set_defaults(func=cmd_watch_status)

    p_watch_start = watch_sub.add_parser(
        "start", help="Run watch mode as a foreground process (reacts to file changes instantly)."
    )
    p_watch_start.add_argument("--workspace", default="default")
    p_watch_start.add_argument("--dirs", nargs="*", metavar="DIR")
    p_watch_start.add_argument("--meili-url", default="http://localhost:7700")
    p_watch_start.add_argument("--meili-key", default="local_search_master_key")
    p_watch_start.add_argument(
        "--no-enrich",
        action="store_true",
        help="Skip semantic enrichment on watch-triggered syncs (faster, no LLM calls).",
    )
    p_watch_start.set_defaults(func=cmd_watch_start)

    p_watch_trigger = watch_sub.add_parser(
        "trigger", help="Trigger an immediate one-off sync, bypassing the debounce window."
    )
    p_watch_trigger.add_argument("--workspace", default="default")
    p_watch_trigger.add_argument("--dirs", nargs="*", metavar="DIR")
    p_watch_trigger.add_argument("--meili-url", default="http://localhost:7700")
    p_watch_trigger.add_argument("--meili-key", default="local_search_master_key")
    p_watch_trigger.add_argument("--force", action="store_true", help="Force full re-index.")
    p_watch_trigger.add_argument(
        "--no-enrich",
        action="store_true",
        help="Skip semantic enrichment for this sync (faster, no LLM calls).",
    )
    p_watch_trigger.set_defaults(func=cmd_watch_trigger)

    # -- health --------------------------------------------------------------
    p_health = sub.add_parser("health", help="Show index health across all workspaces.")
    p_health.add_argument(
        "--stale-threshold",
        type=int,
        default=30,
        help="Minutes after which a workspace is considered stale (default 30).",
    )
    p_health.set_defaults(func=cmd_health)

    # -- ui ------------------------------------------------------------------
    p_ui = sub.add_parser("ui", help="Open the desktop dashboard.")
    p_ui.add_argument(
        "--host",
        default=os.environ.get("LSA_HOST", "127.0.0.1"),
        help="Host for the dashboard API server (default 127.0.0.1).",
    )
    p_ui.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("LSA_PORT", "8765")),
        help="Port for the dashboard API server (default 8765).",
    )
    p_ui.add_argument(
        "--provider",
        default=os.environ.get("LSA_PROVIDER", "google"),
        choices=["google", "ollama", "openai", "anthropic"],
    )
    p_ui.add_argument("--model", default=os.environ.get("LSA_MODEL", "gemma-4-31b-it"))
    p_ui.add_argument("--meili-url", default=os.environ.get("MEILI_URL", "http://localhost:7700"))
    p_ui.add_argument(
        "--meili-key", default=os.environ.get("MEILI_MASTER_KEY", "local_search_master_key")
    )
    p_ui.add_argument(
        "--scheduler-interval",
        type=int,
        default=0,
        help="Start ingestion scheduler with this interval in minutes (0 = disabled).",
    )
    p_ui.add_argument(
        "--headless", action="store_true", help="Run API server only, no window (for debugging)."
    )
    p_ui.set_defaults(func=cmd_ui)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args.func(args)
