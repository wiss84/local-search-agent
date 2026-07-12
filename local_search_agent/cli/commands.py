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
# config: concurrency & rate limits (07-concurrency-and-model-serving)
# ---------------------------------------------------------------------------


def cmd_config_set_concurrency(args: argparse.Namespace) -> None:
    """Set the max simultaneous in-flight LLM calls for a provider."""
    from local_search_agent.agent.rate_limit_handler import reset_shared_rate_limit_handlers
    from local_search_agent.core.key_manager import rate_limits_file_path, set_concurrency_limit

    try:
        set_concurrency_limit(args.provider, args.limit, args.multi_tenant)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    reset_shared_rate_limit_handlers()
    mode = "multi-tenant" if args.multi_tenant else "single-user"
    print(f"\u2713 Concurrency limit for '{args.provider}' set to {args.limit} ({mode} mode).")
    print(f"  Stored at: {rate_limits_file_path()}")


def cmd_config_delete_concurrency(args: argparse.Namespace) -> None:
    """Remove a provider's concurrency cap (reverts to unbounded)."""
    from local_search_agent.agent.rate_limit_handler import reset_shared_rate_limit_handlers
    from local_search_agent.core.key_manager import delete_concurrency_limit

    deleted = delete_concurrency_limit(args.provider, args.multi_tenant)
    reset_shared_rate_limit_handlers()
    mode = "multi-tenant" if args.multi_tenant else "single-user"
    if deleted:
        print(
            f"\u2713 Concurrency limit for '{args.provider}' removed ({mode} mode, now unbounded)."
        )
    else:
        print(f"No concurrency limit was set for '{args.provider}' ({mode} mode).")


def cmd_config_set_rate_limit(args: argparse.Namespace) -> None:
    """Set the RPM/TPM/RPD override for one provider+model."""
    from local_search_agent.agent.rate_limit_handler import reset_shared_rate_limit_handlers
    from local_search_agent.core.key_manager import rate_limits_file_path, set_quota_override

    if args.rpm is None and args.tpm is None and args.rpd is None:
        print("Provide at least one of --rpm, --tpm, --rpd.", file=sys.stderr)
        sys.exit(1)

    try:
        set_quota_override(
            args.provider,
            args.model_name,
            args.multi_tenant,
            rpm=args.rpm,
            tpm=args.tpm,
            rpd=args.rpd,
        )
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    reset_shared_rate_limit_handlers()
    mode = "multi-tenant" if args.multi_tenant else "single-user"
    print(f"\u2713 Rate limit override set for '{args.provider}/{args.model_name}' ({mode} mode).")
    print(f"  Stored at: {rate_limits_file_path()}")


def cmd_config_delete_rate_limit(args: argparse.Namespace) -> None:
    """Remove a provider+model's RPM/TPM/RPD override."""
    from local_search_agent.agent.rate_limit_handler import reset_shared_rate_limit_handlers
    from local_search_agent.core.key_manager import delete_quota_override

    deleted = delete_quota_override(args.provider, args.model_name, args.multi_tenant)
    reset_shared_rate_limit_handlers()
    mode = "multi-tenant" if args.multi_tenant else "single-user"
    if deleted:
        print(
            f"\u2713 Rate limit override for '{args.provider}/{args.model_name}' removed ({mode} mode)."
        )
    else:
        print(
            f"No rate limit override was set for '{args.provider}/{args.model_name}' ({mode} mode)."
        )


def cmd_config_show_rate_limits(args: argparse.Namespace) -> None:
    """Show all configured concurrency limits and RPM/TPM/RPD overrides."""
    from local_search_agent.core.key_manager import (
        get_concurrency_limits,
        get_quota_overrides,
        rate_limits_file_path,
    )

    mode = "multi-tenant" if args.multi_tenant else "single-user"
    print(f"Rate limits & concurrency -- {mode} mode ({rate_limits_file_path()}):")

    concurrency = get_concurrency_limits(args.multi_tenant)
    print("\n  Concurrency (max simultaneous LLM calls):")
    if not concurrency:
        print("    (none configured -- unbounded for every provider)")
    for provider, limit in concurrency.items():
        print(f"    {provider:<12} {limit}")

    overrides = get_quota_overrides(args.multi_tenant)
    print("\n  Quota overrides (RPM / TPM / RPD):")
    if not overrides:
        print("    (none configured -- Google uses auto-detected free-tier defaults; ")
        print("     other providers run retry-only with no quota tracking)")
    for provider, models in overrides.items():
        for model_name, limits in models.items():
            parts = [f"{k}={v}" for k, v in limits.items()]
            print(f"    {provider}/{model_name:<25} {', '.join(parts)}")


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

    if args.multi_tenant:
        # Same non-fatal-by-design provisioning the UI's create_workspace
        # route calls (see auth/meili_key_provisioning.py) -- a workspace
        # is fully usable without a scoped key either way, so a failure
        # here is printed but never turns workspace creation itself into
        # an error.
        from local_search_agent.auth.meili_key_provisioning import provision_workspace_keys
        from local_search_agent.workspace.auth_db import AuthDB

        auth_db = AuthDB(db_path=args.db)
        key_uid = provision_workspace_keys(
            workspace=args.name,
            meilisearch_url=args.meili_url,
            meili_master_key=args.meili_key,
            auth_db=auth_db,
        )
        if key_uid:
            print(f"✓ Scoped member-level Meilisearch key provisioned (uid={key_uid}).")
        else:
            print(
                "⚠ Could not provision a scoped Meilisearch key (see logs). "
                "Member-level requests will fall back to the master key until retried."
            )


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

    if args.multi_tenant:
        from local_search_agent.auth.meili_key_provisioning import deprovision_workspace_keys
        from local_search_agent.workspace.auth_db import AuthDB

        auth_db = AuthDB(db_path=args.db)
        deprovision_workspace_keys(
            workspace=args.name,
            meilisearch_url=args.meili_url,
            meili_master_key=args.meili_key,
            auth_db=auth_db,
        )

    config = SearchAgentConfig(workspace_name=args.name, db_path=args.db)
    framework = SearchAgentFramework(config)
    framework.delete_workspace(name=args.name, wipe_index=args.wipe)
    print(f"Workspace {args.name!r} deleted" + (" (index wiped)." if args.wipe else "."))


# ---------------------------------------------------------------------------
# multi-tenant RBAC: grant-access / revoke-access / list-access
# (see docs/role_based_access_control.md)
# ---------------------------------------------------------------------------


def cmd_grant_access(args: argparse.Namespace) -> None:
    """Grant a subject a role across one or more workspaces (single atomic call)."""
    import getpass

    from local_search_agent.core.config import SearchAgentConfig
    from local_search_agent.core.framework import SearchAgentFramework

    config = SearchAgentConfig(workspace_name="default", db_path=args.db)
    framework = SearchAgentFramework(config)
    granted_by = args.granted_by or getpass.getuser()

    try:
        framework.grant_workspace_access(
            workspaces=args.workspace,
            subject=args.subject,
            role=args.role,
            granted_by=granted_by,
        )
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    ws_list = ", ".join(args.workspace)
    print(f"\u2713 Granted {args.subject!r} role={args.role!r} on workspace(s): {ws_list}")
    print(f"  (granted_by={granted_by!r})")


def cmd_revoke_access(args: argparse.Namespace) -> None:
    """Revoke a subject's access to one or more workspaces, or all of it if --workspace is omitted."""
    from local_search_agent.core.config import SearchAgentConfig
    from local_search_agent.core.framework import SearchAgentFramework

    config = SearchAgentConfig(workspace_name="default", db_path=args.db)
    framework = SearchAgentFramework(config)
    workspaces = args.workspace or None
    deleted = framework.revoke_workspace_access(subject=args.subject, workspaces=workspaces)

    if workspaces is None:
        print(f"\u2713 Revoked all access for {args.subject!r} ({deleted} grant(s) removed).")
    else:
        ws_list = ", ".join(workspaces)
        print(
            f"\u2713 Revoked {args.subject!r} from workspace(s): {ws_list} ({deleted} grant(s) removed)."
        )


def cmd_list_access(args: argparse.Namespace) -> None:
    """List grants, filtered by --subject and/or --workspace (either, both, or neither)."""
    from local_search_agent.core.config import SearchAgentConfig
    from local_search_agent.core.framework import SearchAgentFramework

    config = SearchAgentConfig(workspace_name="default", db_path=args.db)
    framework = SearchAgentFramework(config)
    rows = framework.list_workspace_access(subject=args.subject, workspace=args.workspace)

    if not rows:
        print("No grants found.")
        return

    print(f"{'Workspace':<20} {'Subject':<30} {'Role':<10} {'Granted By':<20} {'Granted At'}")
    print("-" * 100)
    for row in rows:
        print(
            f"  {row['workspace']:<18} {row['subject']:<30} {row['role']:<10} "
            f"{row['granted_by']:<20} {row['granted_at']}"
        )


# ---------------------------------------------------------------------------
# Model / Provider Access Control: grant-model-access / revoke-model-access /
# list-model-access (see docs/role_based_access_control.md)
# ---------------------------------------------------------------------------


def cmd_grant_model_access(args: argparse.Namespace) -> None:
    """Grant a role permission to use a provider+model for their own queries."""
    import getpass

    from local_search_agent.core.config import SearchAgentConfig
    from local_search_agent.core.framework import SearchAgentFramework

    config = SearchAgentConfig(workspace_name="default", db_path=args.db)
    framework = SearchAgentFramework(config)
    granted_by = args.granted_by or getpass.getuser()

    try:
        framework.grant_model_access(
            role=args.role,
            provider=args.provider,
            model_name=args.model_name,
            granted_by=granted_by,
        )
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\u2713 Granted role={args.role!r} access to {args.provider}/{args.model_name}")
    print(f"  (granted_by={granted_by!r})")


def cmd_revoke_model_access(args: argparse.Namespace) -> None:
    """Revoke a role's permission to use a provider+model."""
    from local_search_agent.core.config import SearchAgentConfig
    from local_search_agent.core.framework import SearchAgentFramework

    config = SearchAgentConfig(workspace_name="default", db_path=args.db)
    framework = SearchAgentFramework(config)
    revoked = framework.revoke_model_access(
        role=args.role, provider=args.provider, model_name=args.model_name
    )
    if revoked:
        print(f"\u2713 Revoked role={args.role!r} access to {args.provider}/{args.model_name}")
    else:
        print(f"No grant found for role={args.role!r} on {args.provider}/{args.model_name}.")


def cmd_list_model_access(args: argparse.Namespace) -> None:
    """List model-access grants, optionally filtered by --role."""
    from local_search_agent.core.config import SearchAgentConfig
    from local_search_agent.core.framework import SearchAgentFramework

    config = SearchAgentConfig(workspace_name="default", db_path=args.db)
    framework = SearchAgentFramework(config)
    rows = framework.list_model_access(role=args.role)

    if not rows:
        print("No model-access grants found.")
        return

    print(f"{'Role':<10} {'Provider':<12} {'Model':<30} {'Granted By':<20} {'Granted At'}")
    print("-" * 100)
    for row in rows:
        print(
            f"  {row['role']:<8} {row['provider']:<12} {row['model_name']:<30} "
            f"{row['granted_by']:<20} {row['granted_at']}"
        )


# ---------------------------------------------------------------------------
# multi-tenant RBAC: auth create-key / revoke-key / list-keys
# (APIKeyIdentityProvider -- see docs/role_based_access_control.md)
# ---------------------------------------------------------------------------


def cmd_auth_create_key(args: argparse.Namespace) -> None:
    """Generate a new API key for a subject. The raw key is shown exactly once."""
    import getpass

    from local_search_agent.core.config import SearchAgentConfig
    from local_search_agent.core.framework import SearchAgentFramework

    config = SearchAgentConfig(workspace_name="default", db_path=args.db)
    framework = SearchAgentFramework(config)
    created_by = args.created_by or getpass.getuser()

    key_id, raw_key = framework.create_api_key(
        subject=args.subject,
        created_by=created_by,
        display_name=args.display_name or "",
        is_superadmin=args.superadmin,
    )
    print(f"\u2713 API key created for {args.subject!r} (key_id={key_id})")
    print()
    print(f"  {raw_key}")
    print()
    print("  This key will not be shown again. Store it securely now.")


def cmd_auth_revoke_key(args: argparse.Namespace) -> None:
    """Revoke an API key by its key_id (see 'auth list-keys' for key_ids)."""
    from local_search_agent.core.config import SearchAgentConfig
    from local_search_agent.core.framework import SearchAgentFramework

    config = SearchAgentConfig(workspace_name="default", db_path=args.db)
    framework = SearchAgentFramework(config)
    revoked = framework.revoke_api_key(args.key_id)
    if revoked:
        print(f"\u2713 API key {args.key_id!r} revoked.")
    else:
        print(f"No active key found with key_id {args.key_id!r}.")


def cmd_auth_list_keys(args: argparse.Namespace) -> None:
    """List API keys (metadata only -- never the raw key), optionally filtered by --subject."""
    from local_search_agent.core.config import SearchAgentConfig
    from local_search_agent.core.framework import SearchAgentFramework

    config = SearchAgentConfig(workspace_name="default", db_path=args.db)
    framework = SearchAgentFramework(config)
    rows = framework.list_api_keys(subject=args.subject)

    if not rows:
        print("No API keys found.")
        return

    print(f"{'Key ID':<14} {'Subject':<30} {'Display Name':<20} {'Status':<10} {'Created At'}")
    print("-" * 100)
    for row in rows:
        status = "revoked" if row["revoked_at"] else "active"
        print(
            f"  {row['key_id']:<12} {row['subject']:<30} {row['display_name']:<20} "
            f"{status:<10} {row['created_at']}"
        )


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
        multi_tenant=args.multi_tenant,
        insecure_cookies=args.insecure_cookies,
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

    # set-concurrency / delete-concurrency / set-rate-limit / delete-rate-limit / show-rate-limits
    p_cfg_concurrency = config_sub.add_parser(
        "set-concurrency",
        help="Set the max simultaneous in-flight LLM calls for a provider.",
        description=(
            "Cap how many LLM calls for a provider may be in flight at once, "
            "deployment-wide. For Ollama this is the framework-side mirror of "
            "OLLAMA_NUM_PARALLEL -- set it based on your actual hardware's real "
            "capacity (this framework cannot introspect VRAM itself). For cloud "
            "providers this is a burst control on top of (not instead of) any "
            "RPM/TPM override set via set-rate-limit.\n\n"
            "Example: local-search config set-concurrency --provider ollama --limit 2"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_cfg_concurrency.add_argument(
        "--provider", required=True, choices=["google", "openai", "anthropic", "ollama"]
    )
    p_cfg_concurrency.add_argument(
        "--limit", type=int, required=True, help="Max simultaneous in-flight LLM calls."
    )
    p_cfg_concurrency.add_argument(
        "--multi-tenant",
        action="store_true",
        dest="multi_tenant",
        help=(
            "Edit the multi-tenant namespace instead of single-user. These are "
            "completely independent settings (see rate_limits.json's own structure) "
            "-- pass this if you're configuring a 'local-search ui --multi-tenant' "
            "deployment; omit it for a single-user desktop install."
        ),
    )
    p_cfg_concurrency.set_defaults(func=cmd_config_set_concurrency)

    p_cfg_del_concurrency = config_sub.add_parser(
        "delete-concurrency", help="Remove a provider's concurrency cap (reverts to unbounded)."
    )
    p_cfg_del_concurrency.add_argument(
        "--provider", required=True, choices=["google", "openai", "anthropic", "ollama"]
    )
    p_cfg_del_concurrency.add_argument(
        "--multi-tenant",
        action="store_true",
        dest="multi_tenant",
        help="Edit the multi-tenant namespace instead of single-user.",
    )
    p_cfg_del_concurrency.set_defaults(func=cmd_config_delete_concurrency)

    p_cfg_rate_limit = config_sub.add_parser(
        "set-rate-limit",
        help="Set an RPM/TPM/RPD override for a provider+model.",
        description=(
            "Override the auto-detected free-tier limits (Google) or add quota "
            "tracking where there otherwise is none (OpenAI/Anthropic/Ollama) -- "
            "use this for a paid-tier account with real, much-higher limits than "
            "the free tier. At least one of --rpm/--tpm/--rpd is required; an "
            "omitted dimension means 'don't track this', not 'unlimited'.\n\n"
            "Example: local-search config set-rate-limit --provider google "
            "--model-name gemini-3-flash --rpm 100 --tpm 2000000 --rpd 10000"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_cfg_rate_limit.add_argument(
        "--provider", required=True, choices=["google", "openai", "anthropic", "ollama"]
    )
    p_cfg_rate_limit.add_argument("--model-name", required=True, dest="model_name")
    p_cfg_rate_limit.add_argument("--rpm", type=int, default=None, help="Requests per minute.")
    p_cfg_rate_limit.add_argument("--tpm", type=int, default=None, help="Tokens per minute.")
    p_cfg_rate_limit.add_argument("--rpd", type=int, default=None, help="Requests per day.")
    p_cfg_rate_limit.add_argument(
        "--multi-tenant",
        action="store_true",
        dest="multi_tenant",
        help="Edit the multi-tenant namespace instead of single-user.",
    )
    p_cfg_rate_limit.set_defaults(func=cmd_config_set_rate_limit)

    p_cfg_del_rate_limit = config_sub.add_parser(
        "delete-rate-limit", help="Remove a provider+model's RPM/TPM/RPD override."
    )
    p_cfg_del_rate_limit.add_argument(
        "--provider", required=True, choices=["google", "openai", "anthropic", "ollama"]
    )
    p_cfg_del_rate_limit.add_argument("--model-name", required=True, dest="model_name")
    p_cfg_del_rate_limit.add_argument(
        "--multi-tenant",
        action="store_true",
        dest="multi_tenant",
        help="Edit the multi-tenant namespace instead of single-user.",
    )
    p_cfg_del_rate_limit.set_defaults(func=cmd_config_delete_rate_limit)

    p_cfg_show_rate_limits = config_sub.add_parser(
        "show-rate-limits", help="Show all configured concurrency limits and RPM/TPM/RPD overrides."
    )
    p_cfg_show_rate_limits.add_argument(
        "--multi-tenant",
        action="store_true",
        dest="multi_tenant",
        help="Show the multi-tenant namespace instead of single-user.",
    )
    p_cfg_show_rate_limits.set_defaults(func=cmd_config_show_rate_limits)

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
    p_ws_create.add_argument(
        "--meili-url", default=os.environ.get("MEILI_URL", "http://localhost:7700")
    )
    p_ws_create.add_argument(
        "--meili-key", default=os.environ.get("MEILI_MASTER_KEY", "local_search_master_key")
    )
    p_ws_create.add_argument(
        "--multi-tenant",
        action="store_true",
        dest="multi_tenant",
        help=(
            "Also provision a scoped, member-level Meilisearch key for this "
            "Only meaningful if you're running this framework in "
            "multi-tenant mode elsewhere (e.g. 'local-search ui --multi-tenant') "
            "against this same --db; the workspace is fully usable either way."
        ),
    )
    p_ws_create.set_defaults(func=cmd_workspace_create)

    p_ws_list = ws_sub.add_parser("list", help="List registered workspaces.")
    p_ws_list.set_defaults(func=cmd_workspace_list)

    p_ws_delete = ws_sub.add_parser("delete", help="Delete a workspace.")
    p_ws_delete.add_argument("name")
    p_ws_delete.add_argument(
        "--wipe", action="store_true", help="Also delete all documents from the Meilisearch index."
    )
    p_ws_delete.add_argument(
        "--meili-url", default=os.environ.get("MEILI_URL", "http://localhost:7700")
    )
    p_ws_delete.add_argument(
        "--meili-key", default=os.environ.get("MEILI_MASTER_KEY", "local_search_master_key")
    )
    p_ws_delete.add_argument(
        "--multi-tenant",
        action="store_true",
        dest="multi_tenant",
        help="Also delete this workspace's scoped Meilisearch key (if one was provisioned via 'workspace create --multi-tenant').",
    )
    p_ws_delete.set_defaults(func=cmd_workspace_delete)

    # -- grant-access / revoke-access / list-access (multi-tenant RBAC) ------
    p_grant = sub.add_parser(
        "grant-access", help="Grant a subject a role across one or more workspaces."
    )
    p_grant.add_argument("--subject", required=True, help="Stable identity, e.g. an email.")
    p_grant.add_argument(
        "--workspace",
        nargs="+",
        required=True,
        metavar="WORKSPACE",
        help="One or more workspace names.",
    )
    p_grant.add_argument("--role", required=True, choices=["member", "admin"])
    p_grant.add_argument(
        "--granted-by",
        default=None,
        dest="granted_by",
        help="Identity performing the grant, for audit purposes (default: current OS user).",
    )
    p_grant.set_defaults(func=cmd_grant_access)

    p_revoke = sub.add_parser("revoke-access", help="Revoke a subject's workspace access.")
    p_revoke.add_argument("--subject", required=True)
    p_revoke.add_argument(
        "--workspace",
        nargs="*",
        metavar="WORKSPACE",
        help="Workspace(s) to revoke. Omit to revoke ALL of the subject's access.",
    )
    p_revoke.set_defaults(func=cmd_revoke_access)

    p_list_access = sub.add_parser("list-access", help="List workspace access grants.")
    p_list_access.add_argument("--subject", default=None, help="Filter by subject.")
    p_list_access.add_argument("--workspace", default=None, help="Filter by workspace.")
    p_list_access.set_defaults(func=cmd_list_access)

    # -- grant-model-access / revoke-model-access / list-model-access --------
    # (Model/Provider Access Control -- see docs/role_based_access_control.md)
    p_grant_model = sub.add_parser(
        "grant-model-access",
        help="Grant a role permission to use a provider+model for their own queries.",
        description=(
            "A role with nothing granted has access to NOTHING -- fail-closed, same "
            "as workspace access. Grant at least one model to each role before anyone "
            "in that role tries to query.\n\n"
            "Example: local-search grant-model-access --role member --provider google "
            "--model-name gemma-4-31b-it"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_grant_model.add_argument("--role", required=True, choices=["member", "admin"])
    p_grant_model.add_argument(
        "--provider", required=True, choices=["google", "openai", "anthropic", "ollama"]
    )
    p_grant_model.add_argument("--model-name", required=True, dest="model_name")
    p_grant_model.add_argument(
        "--granted-by",
        default=None,
        dest="granted_by",
        help="Identity performing the grant, for audit purposes (default: current OS user).",
    )
    p_grant_model.set_defaults(func=cmd_grant_model_access)

    p_revoke_model = sub.add_parser(
        "revoke-model-access", help="Revoke a role's permission to use a provider+model."
    )
    p_revoke_model.add_argument("--role", required=True, choices=["member", "admin"])
    p_revoke_model.add_argument(
        "--provider", required=True, choices=["google", "openai", "anthropic", "ollama"]
    )
    p_revoke_model.add_argument("--model-name", required=True, dest="model_name")
    p_revoke_model.set_defaults(func=cmd_revoke_model_access)

    p_list_model_access = sub.add_parser(
        "list-model-access", help="List model-access grants (which roles may use which models)."
    )
    p_list_model_access.add_argument(
        "--role", default=None, choices=["member", "admin"], help="Filter by role."
    )
    p_list_model_access.set_defaults(func=cmd_list_model_access)

    # -- auth (API key management for APIKeyIdentityProvider) ----------------
    p_auth = sub.add_parser("auth", help="Manage API keys (APIKeyIdentityProvider).")
    auth_sub = p_auth.add_subparsers(dest="auth_command", required=True)

    p_auth_create = auth_sub.add_parser("create-key", help="Generate a new API key for a subject.")
    p_auth_create.add_argument("--subject", required=True, help="Stable identity, e.g. an email.")
    p_auth_create.add_argument("--display-name", default=None, dest="display_name")
    p_auth_create.add_argument(
        "--superadmin",
        action="store_true",
        help="Mark this identity as a framework-level superadmin (rarely needed).",
    )
    p_auth_create.add_argument(
        "--created-by",
        default=None,
        dest="created_by",
        help="Identity creating the key, for audit purposes (default: current OS user).",
    )
    p_auth_create.set_defaults(func=cmd_auth_create_key)

    p_auth_revoke = auth_sub.add_parser("revoke-key", help="Revoke an API key by its key_id.")
    p_auth_revoke.add_argument("key_id")
    p_auth_revoke.set_defaults(func=cmd_auth_revoke_key)

    p_auth_list = auth_sub.add_parser("list-keys", help="List API keys (metadata only).")
    p_auth_list.add_argument("--subject", default=None, help="Filter by subject.")
    p_auth_list.set_defaults(func=cmd_auth_list_keys)

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
    p_ui.add_argument(
        "--multi-tenant",
        action="store_true",
        dest="multi_tenant",
        help=(
            "Enable multi-tenant RBAC (APIKeyIdentityProvider) against this same --db. "
            "Bootstrap keys/grants first with 'local-search auth create-key' and "
            "'local-search grant-access' against the same --db path. "
            "See docs/role_based_access_control.md."
        ),
    )
    p_ui.add_argument(
        "--insecure-cookies",
        action="store_true",
        dest="insecure_cookies",
        help=(
            "Allow the multi-tenant session cookie over plain HTTP -- needed when "
            "--host is a real LAN IP rather than 127.0.0.1/localhost, since browsers "
            "treat only localhost as a secure context and otherwise silently refuse "
            "to store a Secure cookie over non-HTTPS (login will appear to silently "
            "do nothing without this flag). Only use on a trusted local network "
            "(e.g. testing across two laptops on the same home/office LAN); never "
            "for anything internet-facing -- use a TLS reverse proxy instead, see "
            "docs/production-deployment.md."
        ),
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
