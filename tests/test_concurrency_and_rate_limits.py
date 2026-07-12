"""
Tests for concurrency & rate-limit configuration (07-concurrency-and-
model-serving.md), covering:

1. core/key_manager.py's concurrency/quota_overrides CRUD (rate_limits.json),
   namespaced independently by single_user vs multi_tenant mode.
2. route_policy.py's superadmin_only entries for the new routes.
3. agent/rate_limit_handler.py's ConcurrencyGate (threading.Semaphore-based
   bounded concurrency with queue-position tracking and fail-fast timeout).
4. The shared-instance registry (get_shared_rate_limit_handler /
   reset_shared_rate_limit_handlers) -- this is the fix for the double-
   counting bug Model/Provider Access Control's per-workspace agent
   caching introduced (two workspaces using the same provider+model must
   share ONE RateLimitHandler, not build their own).
5. RateLimitHandler's generalized quota tracking -- active for Google
   always (auto-detected), and for any other provider only once an
   explicit override is configured.
"""

from __future__ import annotations

import threading
import time

import pytest

from local_search_agent.auth.route_policy import match_policy

# ---------------------------------------------------------------------------
# Layer 1: key_manager.py CRUD
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_config_dir(tmp_path, monkeypatch):
    """Point key_manager's config dir AND the rate-limit persistence
    file's cwd-relative path at tmp_path, so tests don't touch the real
    user config directory or leave a stray rate_limit_state.json in the
    repo (RateLimitHandler resolves PERSISTENCE_FILE relative to cwd)."""
    monkeypatch.setattr(
        "local_search_agent.core.key_manager.user_config_dir", lambda app: str(tmp_path)
    )
    monkeypatch.chdir(tmp_path)
    yield tmp_path


class TestConcurrencyConfigCRUD:
    def test_no_limits_configured_by_default(self, isolated_config_dir):
        from local_search_agent.core.key_manager import get_concurrency_limits

        assert get_concurrency_limits(multi_tenant=False) == {}
        assert get_concurrency_limits(multi_tenant=True) == {}

    def test_set_and_get_concurrency_limit(self, isolated_config_dir):
        from local_search_agent.core.key_manager import (
            get_concurrency_limits,
            set_concurrency_limit,
        )

        set_concurrency_limit("ollama", 2, multi_tenant=False)
        assert get_concurrency_limits(multi_tenant=False) == {"ollama": 2}

    def test_set_concurrency_limit_below_one_rejected(self, isolated_config_dir):
        from local_search_agent.core.key_manager import set_concurrency_limit

        with pytest.raises(ValueError):
            set_concurrency_limit("ollama", 0, multi_tenant=False)

    def test_set_concurrency_limit_unknown_provider_rejected(self, isolated_config_dir):
        from local_search_agent.core.key_manager import set_concurrency_limit

        with pytest.raises(ValueError):
            set_concurrency_limit("not-a-real-provider", 5, multi_tenant=False)

    def test_delete_concurrency_limit(self, isolated_config_dir):
        from local_search_agent.core.key_manager import (
            delete_concurrency_limit,
            get_concurrency_limits,
            set_concurrency_limit,
        )

        set_concurrency_limit("ollama", 2, multi_tenant=False)
        assert delete_concurrency_limit("ollama", multi_tenant=False) is True
        assert get_concurrency_limits(multi_tenant=False) == {}
        assert delete_concurrency_limit("ollama", multi_tenant=False) is False

    def test_single_user_and_multi_tenant_concurrency_are_independent(self, isolated_config_dir):
        """The exact bug the user flagged: a single-user desktop change
        must not leak into (or be overwritten by) a multi-tenant change
        on the same machine/config dir, and vice versa."""
        from local_search_agent.core.key_manager import (
            get_concurrency_limits,
            set_concurrency_limit,
        )

        set_concurrency_limit("ollama", 2, multi_tenant=False)
        set_concurrency_limit("ollama", 8, multi_tenant=True)

        assert get_concurrency_limits(multi_tenant=False) == {"ollama": 2}
        assert get_concurrency_limits(multi_tenant=True) == {"ollama": 8}

        # Changing one must not touch the other.
        set_concurrency_limit("ollama", 3, multi_tenant=False)
        assert get_concurrency_limits(multi_tenant=False) == {"ollama": 3}
        assert get_concurrency_limits(multi_tenant=True) == {"ollama": 8}


class TestQuotaOverrideCRUD:
    def test_no_overrides_by_default(self, isolated_config_dir):
        from local_search_agent.core.key_manager import get_quota_overrides

        assert get_quota_overrides(multi_tenant=False) == {}
        assert get_quota_overrides(multi_tenant=False, provider="openai") == {}

    def test_set_and_get_quota_override(self, isolated_config_dir):
        from local_search_agent.core.key_manager import get_quota_overrides, set_quota_override

        set_quota_override("openai", "gpt-5", multi_tenant=False, rpm=60, tpm=500_000)
        overrides = get_quota_overrides(multi_tenant=False, provider="openai")
        assert overrides == {"gpt-5": {"rpm": 60, "tpm": 500_000}}

    def test_set_quota_override_requires_at_least_one_field(self, isolated_config_dir):
        from local_search_agent.core.key_manager import set_quota_override

        with pytest.raises(ValueError):
            set_quota_override("openai", "gpt-5", multi_tenant=False)

    def test_delete_quota_override(self, isolated_config_dir):
        from local_search_agent.core.key_manager import (
            delete_quota_override,
            get_quota_overrides,
            set_quota_override,
        )

        set_quota_override("openai", "gpt-5", multi_tenant=False, rpm=60)
        assert delete_quota_override("openai", "gpt-5", multi_tenant=False) is True
        assert get_quota_overrides(multi_tenant=False, provider="openai") == {}
        assert delete_quota_override("openai", "gpt-5", multi_tenant=False) is False

    def test_single_user_and_multi_tenant_quota_are_independent(self, isolated_config_dir):
        """Same independence guarantee as concurrency -- a single user's
        free-tier-adjacent override and a company's paid-tier override
        for the SAME provider+model must not collide."""
        from local_search_agent.core.key_manager import get_quota_overrides, set_quota_override

        set_quota_override("openai", "gpt-5", multi_tenant=False, rpm=10)
        set_quota_override("openai", "gpt-5", multi_tenant=True, rpm=500, tpm=2_000_000)

        assert get_quota_overrides(multi_tenant=False, provider="openai") == {"gpt-5": {"rpm": 10}}
        assert get_quota_overrides(multi_tenant=True, provider="openai") == {
            "gpt-5": {"rpm": 500, "tpm": 2_000_000}
        }


# ---------------------------------------------------------------------------
# Layer 2: RoutePolicy entries
# ---------------------------------------------------------------------------


class TestRateLimitRoutePolicyEntries:
    def test_get_rate_limits_is_superadmin_only(self):
        policy = match_policy("GET", "/api/ui/rate-limits")
        assert policy is not None
        assert policy.scope == "superadmin_only"

    @pytest.mark.parametrize(
        "method,path",
        [
            ("POST", "/api/ui/rate-limits/concurrency"),
            ("DELETE", "/api/ui/rate-limits/concurrency"),
            ("POST", "/api/ui/rate-limits/quota"),
            ("DELETE", "/api/ui/rate-limits/quota"),
        ],
    )
    def test_all_write_routes_are_superadmin_only(self, method, path):
        policy = match_policy(method, path)
        assert policy is not None
        assert policy.scope == "superadmin_only"


# ---------------------------------------------------------------------------
# Layer 3: ConcurrencyGate
# ---------------------------------------------------------------------------


class TestConcurrencyGate:
    def test_limit_below_one_rejected(self):
        from local_search_agent.agent.rate_limit_handler import ConcurrencyGate

        with pytest.raises(ValueError):
            ConcurrencyGate(0)

    def test_acquire_release_allows_sequential_use(self):
        from local_search_agent.agent.rate_limit_handler import ConcurrencyGate

        gate = ConcurrencyGate(1)
        gate.acquire()
        gate.release()
        gate.acquire()  # should not block -- released above
        gate.release()

    def test_second_caller_blocks_until_release(self):
        from local_search_agent.agent.rate_limit_handler import ConcurrencyGate

        gate = ConcurrencyGate(1, max_wait_seconds=5)
        gate.acquire()

        released_by_second = []

        def _second_caller():
            gate.acquire()
            released_by_second.append(True)
            gate.release()

        t = threading.Thread(target=_second_caller)
        t.start()
        time.sleep(0.2)
        assert gate.waiting_count() == 1
        assert released_by_second == []  # still blocked

        gate.release()  # frees the slot for the second caller
        t.join(timeout=5)
        assert released_by_second == [True]
        assert gate.waiting_count() == 0

    def test_timeout_raises_when_queue_never_clears(self):
        from local_search_agent.agent.rate_limit_handler import ConcurrencyGate

        gate = ConcurrencyGate(1, max_wait_seconds=0.2)
        gate.acquire()  # holds the only slot, never released in this test
        with pytest.raises(TimeoutError):
            gate.acquire()

    def test_on_wait_not_called_when_slot_immediately_free(self):
        from local_search_agent.agent.rate_limit_handler import ConcurrencyGate

        gate = ConcurrencyGate(2)
        calls = []
        gate.acquire(on_wait=lambda n: calls.append(n))
        assert calls == []

    def test_on_wait_called_once_with_waiting_count_when_blocking(self):
        from local_search_agent.agent.rate_limit_handler import ConcurrencyGate

        gate = ConcurrencyGate(1, max_wait_seconds=5)
        gate.acquire()  # hold the only slot
        calls = []

        def _second_caller():
            gate.acquire(on_wait=lambda n: calls.append(n))
            gate.release()

        t = threading.Thread(target=_second_caller)
        t.start()
        time.sleep(0.2)
        gate.release()  # let the second caller through
        t.join(timeout=5)
        assert calls == [1]


class TestQueuedCallbackThreadLocal:
    def test_default_is_none(self):
        from local_search_agent.agent.rate_limit_handler import get_queued_callback

        assert get_queued_callback() is None

    def test_set_and_get_on_same_thread(self):
        from local_search_agent.agent.rate_limit_handler import (
            get_queued_callback,
            set_queued_callback,
        )

        cb = lambda n: None  # noqa: E731
        set_queued_callback(cb)
        try:
            assert get_queued_callback() is cb
        finally:
            set_queued_callback(None)

    def test_isolated_per_thread(self):
        """The whole point of using a thread-local: two threads (standing
        in for two concurrent requests sharing the same cached
        RateLimitHandler) must never see each other's callback."""
        from local_search_agent.agent.rate_limit_handler import (
            get_queued_callback,
            set_queued_callback,
        )

        seen_in_other_thread = []

        def _other_thread():
            # Never set anything on this thread -- should see None, not
            # whatever the main thread set below.
            seen_in_other_thread.append(get_queued_callback())

        set_queued_callback(lambda n: None)
        try:
            t = threading.Thread(target=_other_thread)
            t.start()
            t.join()
            assert seen_in_other_thread == [None]
            assert get_queued_callback() is not None  # main thread's own is untouched
        finally:
            set_queued_callback(None)


# ---------------------------------------------------------------------------
# Layer 4: shared-instance registry
# ---------------------------------------------------------------------------


class TestSharedRateLimitHandlerRegistry:
    def setup_method(self):
        from local_search_agent.agent.rate_limit_handler import reset_shared_rate_limit_handlers

        reset_shared_rate_limit_handlers()

    def teardown_method(self):
        from local_search_agent.agent.rate_limit_handler import reset_shared_rate_limit_handlers

        reset_shared_rate_limit_handlers()

    def test_same_provider_model_returns_same_instance(self, isolated_config_dir):
        from local_search_agent.agent.rate_limit_handler import get_shared_rate_limit_handler

        h1 = get_shared_rate_limit_handler("google", "gemma-4-31b-it", multi_tenant=True)
        h2 = get_shared_rate_limit_handler("google", "gemma-4-31b-it", multi_tenant=True)
        assert h1 is h2

    def test_different_workspace_same_provider_model_still_shares_instance(
        self, isolated_config_dir
    ):
        """This is the actual bug fix: two LocalSearchAgent instances built
        for different workspaces (Model/Provider Access Control's per-
        workspace agent cache) but the same provider+model must NOT each
        get their own RateLimitHandler -- they'd double-count RPM/TPM
        against one shared real quota."""
        from local_search_agent.agent.rate_limit_handler import get_shared_rate_limit_handler

        # Simulates agent.py calling this once per workspace's agent build
        h_workspace_a = get_shared_rate_limit_handler("google", "gemma-4-31b-it", multi_tenant=True)
        h_workspace_b = get_shared_rate_limit_handler("google", "gemma-4-31b-it", multi_tenant=True)
        assert h_workspace_a is h_workspace_b

    def test_different_model_returns_different_instance(self, isolated_config_dir):
        from local_search_agent.agent.rate_limit_handler import get_shared_rate_limit_handler

        h1 = get_shared_rate_limit_handler("google", "gemma-4-31b-it", multi_tenant=True)
        h2 = get_shared_rate_limit_handler("google", "gemini-3.1-flash-lite", multi_tenant=True)
        assert h1 is not h2

    def test_single_user_and_multi_tenant_are_different_instances(self, isolated_config_dir):
        """Same (provider, model) but different mode must NOT share an
        instance -- they read from independent config namespaces and
        must track quota/concurrency independently."""
        from local_search_agent.agent.rate_limit_handler import get_shared_rate_limit_handler

        h_single = get_shared_rate_limit_handler("google", "gemma-4-31b-it", multi_tenant=False)
        h_multi = get_shared_rate_limit_handler("google", "gemma-4-31b-it", multi_tenant=True)
        assert h_single is not h_multi

    def test_reset_clears_cache_so_next_call_rebuilds(self, isolated_config_dir):
        from local_search_agent.agent.rate_limit_handler import (
            get_shared_rate_limit_handler,
            reset_shared_rate_limit_handlers,
        )

        h1 = get_shared_rate_limit_handler("google", "gemma-4-31b-it", multi_tenant=True)
        reset_shared_rate_limit_handlers()
        h2 = get_shared_rate_limit_handler("google", "gemma-4-31b-it", multi_tenant=True)
        assert h1 is not h2

    def test_picks_up_configured_concurrency_limit(self, isolated_config_dir):
        from local_search_agent.agent.rate_limit_handler import get_shared_rate_limit_handler
        from local_search_agent.core.key_manager import set_concurrency_limit

        set_concurrency_limit("ollama", 3, multi_tenant=True)
        handler = get_shared_rate_limit_handler("ollama", "llama3", multi_tenant=True)
        assert handler._gate is not None
        assert handler._gate.limit == 3

    def test_picks_up_configured_quota_override_for_non_google_provider(self, isolated_config_dir):
        from local_search_agent.agent.rate_limit_handler import get_shared_rate_limit_handler
        from local_search_agent.core.key_manager import set_quota_override

        set_quota_override("openai", "gpt-5", multi_tenant=True, rpm=100, tpm=1_000_000)
        handler = get_shared_rate_limit_handler("openai", "gpt-5", multi_tenant=True)
        assert handler._track_quota is True
        # safety_margin defaults to 0.8
        assert handler.rpm_limit == 80
        assert handler.tpm_limit == 800_000

    def test_multi_tenant_handler_unaffected_by_single_user_config(self, isolated_config_dir):
        """A single-user-mode config change must not affect an already-
        or freshly-built multi-tenant handler for the same provider+model."""
        from local_search_agent.agent.rate_limit_handler import get_shared_rate_limit_handler
        from local_search_agent.core.key_manager import set_concurrency_limit

        set_concurrency_limit("ollama", 2, multi_tenant=False)
        handler = get_shared_rate_limit_handler("ollama", "llama3", multi_tenant=True)
        assert handler._gate is None  # nothing configured in the multi_tenant namespace


# ---------------------------------------------------------------------------
# Layer 5: RateLimitHandler's generalized quota tracking
# ---------------------------------------------------------------------------


class TestQuotaTrackingGeneralization:
    def test_google_tracks_quota_by_default(self, tmp_path, monkeypatch):
        from local_search_agent.agent.rate_limit_handler import RateLimitHandler

        monkeypatch.chdir(tmp_path)
        handler = RateLimitHandler(provider="google", model_name="gemma-4-31b-it")
        assert handler._track_quota is True
        assert handler.rpm_limit is not None

    def test_non_google_no_tracking_without_override(self, tmp_path, monkeypatch):
        from local_search_agent.agent.rate_limit_handler import RateLimitHandler

        monkeypatch.chdir(tmp_path)
        handler = RateLimitHandler(provider="openai", model_name="gpt-5")
        assert handler._track_quota is False

    def test_non_google_tracks_quota_once_override_given(self, tmp_path, monkeypatch):
        from local_search_agent.agent.rate_limit_handler import RateLimitHandler

        monkeypatch.chdir(tmp_path)
        handler = RateLimitHandler(
            provider="openai", model_name="gpt-5", requests_per_minute=60, tokens_per_minute=500_000
        )
        assert handler._track_quota is True
        assert handler.rpm_limit == 48  # 60 * 0.8 safety margin
        assert handler.tpm_limit == 400_000

    def test_status_reports_none_tracking_for_retry_only_mode(self, tmp_path, monkeypatch):
        from local_search_agent.agent.rate_limit_handler import RateLimitHandler

        monkeypatch.chdir(tmp_path)
        handler = RateLimitHandler(provider="ollama", model_name="llama3")
        status = handler.status()
        assert status["tracking"] == "none"

    def test_concurrency_limit_wired_into_status(self, tmp_path, monkeypatch):
        from local_search_agent.agent.rate_limit_handler import RateLimitHandler

        monkeypatch.chdir(tmp_path)
        handler = RateLimitHandler(provider="ollama", model_name="llama3", concurrency_limit=2)
        status = handler.status()
        assert status["concurrency_limit"] == 2
        assert status["concurrency_waiting"] == 0
