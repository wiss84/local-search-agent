"""
Rate Limit Handler for Local Search Agent
==========================================

Wraps any callable (typically an LLM invoke call) with:

  - Proactive delay before hitting limits (RPM / TPM / RPD tracking)
  - Reactive retry on transient errors with exponential backoff
  - Per-model daily request count persisted to JSON so quotas survive restarts
    and reset automatically at calendar midnight
  - A bounded concurrency gate capping how many calls for this
    provider+model may be in flight at once, deployment-wide

Provider behaviour
------------------
  google  : Full rate limiting -- RPM / TPM / RPD tracking with sliding
            windows, retries on Google-specific 429 / 503 / 500 errors.
            Free-tier limits are auto-detected from the model name prefix
            (gemini-* / gemma-4-*) unless overridden (see below).

  others (openai, anthropic, ollama)
          : Simple exponential-backoff retry on any transient error by
            default -- no quota tracking, matching the free/no-quota-info
            reality for these providers out of the box. RPM/TPM/RPD
            tracking activates for these too the moment an explicit
            override is configured (see core/key_manager.py's
            get_quota_overrides/set_quota_override) -- this is how a
            company running paid-tier OpenAI/Anthropic accounts with
            real, much-higher-than-free-tier limits gets the same sliding-
            window tracking Google gets automatically. Fully admin-
            configurable per (provider, model_name), not hardcoded.

Concurrency
-----------
Every provider (including Google) can additionally have a concurrency cap
-- the max number of LLM calls for that provider allowed in flight at
once, deployment-wide, configured via
core/key_manager.py's get_concurrency_limits()/set_concurrency_limit().
For Ollama this is the framework-side mirror of Ollama's own
OLLAMA_NUM_PARALLEL setting -- the admin sets this based on their actual
hardware's real capacity (this module has no way to introspect VRAM
itself). For cloud providers it caps simultaneous requests as a burst
control on top of (not instead of) RPM/TPM tracking.

The gate is a threading.Semaphore, not asyncio.Semaphore -- the LLM calls
this wraps happen synchronously inside a background worker thread (see
ui/api_routes.py's _run_agent_streaming -> threading.Thread), not on the
asyncio event loop, so an asyncio primitive would be the wrong tool here.

One shared RateLimitHandler instance per (provider, model_name), NOT one
per agent
-------------------------------------------------------------------------
Use get_shared_rate_limit_handler() below rather than constructing
RateLimitHandler directly in application code. Model/Provider Access
Control (Option B) caches a separate LocalSearchAgent instance per
(workspace, meili_key, provider, model_name) -- if each agent constructed
its own RateLimitHandler, two workspaces both using google/gemini-3-flash
would each track RPM/TPM against their own private in-memory sliding
window, each believing it has the FULL quota, when in reality they share
one account's actual limit. The shared-instance registry keeps exactly
one RateLimitHandler (and one concurrency gate) per (provider,
model_name) process-wide, so tracking and concurrency are correctly
shared across every agent instance that targets the same underlying
account/hardware, regardless of which workspace or role requested it.

Usage
-----
    handler = get_shared_rate_limit_handler(
        provider="google", model_name="gemma-4-31b-it", multi_tenant=False
    )
    response = handler.call_with_retry(llm_with_tools.invoke, messages)
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from collections import deque
from datetime import datetime
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Google error imports — graceful fallback if google packages not installed
# ---------------------------------------------------------------------------

try:
    from google.api_core.exceptions import (
        InternalServerError,
        ResourceExhausted,
        ServiceUnavailable,
    )
except ImportError:
    ResourceExhausted = None
    ServiceUnavailable = None
    InternalServerError = None

try:
    from google.genai.errors import ClientError as _GenaiClientError
    from google.genai.errors import ServerError as _GenaiServerError
except ImportError:
    _GenaiServerError = None
    _GenaiClientError = None

# Build exception tuples — filter out None entries so except clauses stay valid
_GOOGLE_SERVER_ERRORS = tuple(
    e
    for e in (
        ServiceUnavailable,
        InternalServerError,
        _GenaiServerError,
    )
    if e is not None
)

_GOOGLE_RATE_ERRORS = tuple(
    e
    for e in (
        ResourceExhausted,
        _GenaiClientError,
    )
    if e is not None
)

# ---------------------------------------------------------------------------
# Free-tier rate limits per Google model family (auto-detected default;
# overridden by core/key_manager.py's quota_overrides when configured)
# ---------------------------------------------------------------------------

_GOOGLE_LIMITS = {
    "gemini": {
        "requests_per_minute": 15,
        "tokens_per_minute": 250_000,  # free-tier TPM enforced
        "requests_per_day": 500,
    },
    "gemma-4": {
        "requests_per_minute": 15,
        "tokens_per_minute": 10_000_000,  # effectively unlimited — not enforced
        "requests_per_day": 1_500,
    },
}

_GOOGLE_LIMITS_DEFAULT = _GOOGLE_LIMITS["gemma-4"]

PERSISTENCE_FILE = "rate_limit_state.json"


def _detect_google_limits(model_name: str) -> dict:
    if model_name.startswith("gemini-"):
        return _GOOGLE_LIMITS["gemini"].copy()
    if model_name.startswith("gemma-4-"):
        return _GOOGLE_LIMITS["gemma-4"].copy()
    logger.warning(
        "RateLimitHandler: unrecognised Google model %r — defaulting to Gemma 4 free-tier limits.",
        model_name,
    )
    return _GOOGLE_LIMITS_DEFAULT.copy()


# ---------------------------------------------------------------------------
# Concurrency gate
# ---------------------------------------------------------------------------


class ConcurrencyGate:
    """
    Bounded concurrency control for actual LLM calls.

    Uses threading.Semaphore (see module docstring for why, not
    asyncio.Semaphore) with an explicit waiting-count counter alongside
    it, since neither threading nor asyncio semaphores expose queue depth
    through any public API -- tracking it ourselves is what lets the
    caller report "N requests ahead of you" (07-concurrency-and-model-
    serving.md's queued SSE event).

    acquire() has a bounded max wait rather than blocking forever -- if
    the queue doesn't clear within max_wait_seconds, it raises
    TimeoutError so the caller can fail the request with a clear message
    rather than the browser tab hanging indefinitely (07's own stated
    goal: fail fast with a clear response, don't silently pile up
    requests the provider would eventually reject anyway).
    """

    def __init__(self, limit: int, max_wait_seconds: float = 120.0):
        if limit < 1:
            raise ValueError("Concurrency limit must be at least 1.")
        self.limit = limit
        self.max_wait_seconds = max_wait_seconds
        self._sem = threading.Semaphore(limit)
        self._count_lock = threading.Lock()
        self._waiting = 0

    def waiting_count(self) -> int:
        with self._count_lock:
            return self._waiting

    def acquire(self, on_wait: Optional[Callable[[int], None]] = None) -> None:
        """
        Acquire a slot, blocking if none are free.

        on_wait : Called ONCE, synchronously, with the current waiting
                  count, if (and only if) this call actually has to wait
                  (a slot wasn't immediately free). Lets the caller emit a
                  "queued" signal (e.g. an SSE event) before blocking --
                  see set_queued_callback()/get_queued_callback() above
                  for how agent/agent.py wires this up. Never called at
                  all on the common fast path where a slot is free.
        """
        # Fast path: try a non-blocking acquire first -- if it succeeds, no
        # queueing/callback needed at all, avoids the waiting-count lock
        # entirely for the common case where a slot is already free.
        if self._sem.acquire(blocking=False):
            return
        with self._count_lock:
            self._waiting += 1
        if on_wait is not None:
            try:
                on_wait(self.waiting_count())
            except Exception:
                logger.warning("ConcurrencyGate on_wait callback raised.", exc_info=True)
        try:
            acquired = self._sem.acquire(timeout=self.max_wait_seconds)
        finally:
            with self._count_lock:
                self._waiting -= 1
        if not acquired:
            raise TimeoutError(
                f"Too many concurrent requests for this provider right now "
                f"(waited {self.max_wait_seconds:.0f}s without a free slot). "
                f"Please try again shortly."
            )

    def release(self) -> None:
        self._sem.release()


# ---------------------------------------------------------------------------
# Queued-callback registry -- lets the concurrency gate notify the SSE
# stream when a call has to wait, without RateLimitHandler/ConcurrencyGate
# needing any knowledge of SSE, queues, or the web layer at all.
#
# Thread-local, not a plain module attribute: each HTTP request's agent
# execution runs in its own dedicated background thread (see
# ui/api_routes.py's _agent_thread), and multiple concurrent requests may
# share the SAME cached LocalSearchAgent/RateLimitHandler instance --
# that's the whole point of the concurrency gate. A plain attribute on
# the handler or agent would race between those concurrent requests;
# thread-local doesn't, since each request's own thread only ever sees
# the callback it set for itself.
# ---------------------------------------------------------------------------

_queued_callback_local = threading.local()


def set_queued_callback(callback: Optional[Callable[[int], None]]) -> None:
    """Set the on-queued callback for the CURRENT THREAD only. Called by
    ui/api_routes.py's _agent_thread before running the agent, and cleared
    (None) when the thread is done with this request."""
    _queued_callback_local.callback = callback


def get_queued_callback() -> Optional[Callable[[int], None]]:
    """Read back whatever set_queued_callback() set on THIS thread. Called
    by agent/agent.py's call_llm node, once per LLM call."""
    return getattr(_queued_callback_local, "callback", None)


# ---------------------------------------------------------------------------
# RateLimitHandler
# ---------------------------------------------------------------------------


class RateLimitHandler:
    """
    Provider-aware rate limiter and retry wrapper.

    Parameters
    ----------
    provider        : "google" | "ollama" | "openai" | "anthropic"
    model_name      : Model identifier string (used for limit detection + persistence key)
    max_retries     : Maximum retry attempts on transient errors
    base_backoff    : Base delay (seconds) for exponential backoff
    safety_margin   : Fraction of the hard limit to treat as the effective limit
    requests_per_minute / tokens_per_minute / requests_per_day :
                      Explicit overrides (any provider, from
                      core/key_manager.py's quota_overrides). For Google,
                      overrides the auto-detected free-tier default. For
                      any other provider, ANY of these being set activates
                      sliding-window quota tracking for it, which
                      otherwise defaults to off (retry-only mode).
    concurrency_limit : Max simultaneous in-flight calls for this
                      provider+model, deployment-wide. None = unbounded
                      (today's behavior, unchanged unless an admin
                      configures a limit via core/key_manager.py's
                      set_concurrency_limit).
    concurrency_max_wait_seconds : How long a call may wait for a free
                      concurrency slot before failing with a clear error.
    """

    def __init__(
        self,
        provider: str,
        model_name: str,
        max_retries: int = 5,
        base_backoff: float = 2.0,
        safety_margin: float = 0.8,
        requests_per_minute: Optional[int] = None,
        tokens_per_minute: Optional[int] = None,
        requests_per_day: Optional[int] = None,
        concurrency_limit: Optional[int] = None,
        concurrency_max_wait_seconds: float = 120.0,
    ):
        self.provider = provider.lower()
        self.model_name = model_name
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self._is_google = self.provider == "google"

        # Quota tracking is active for Google always (auto-detected
        # free-tier default, override-able) and for any OTHER provider
        # only once an explicit override is configured -- this is what
        # generalises tracking past Google-only.
        has_explicit_override = bool(requests_per_minute or tokens_per_minute or requests_per_day)
        self._track_quota = self._is_google or has_explicit_override

        if self._track_quota:
            if self._is_google:
                detected = _detect_google_limits(model_name)
            else:
                detected = {
                    "requests_per_minute": 0,
                    "tokens_per_minute": 0,
                    "requests_per_day": 0,
                }
            rpm = requests_per_minute or detected["requests_per_minute"]
            tpm = tokens_per_minute or detected["tokens_per_minute"]
            rpd = requests_per_day or detected["requests_per_day"]

            # A dimension with no configured/detected value at all means
            # "don't track this dimension" (None), not "block on a limit
            # of zero".
            self.rpm_limit = int(rpm * safety_margin) if rpm else None
            self.tpm_limit = int(tpm * safety_margin) if tpm else None
            self.rpd_limit = int(rpd * safety_margin) if rpd else None

            self._minute_requests: deque = deque()
            self._minute_tokens: deque = deque()
            self._day_request_count: int = 0
            self._load_state()

            logger.info(
                "RateLimitHandler [%s/%s]: RPM=%s TPM=%s RPD=%s",
                self.provider,
                model_name,
                self.rpm_limit,
                self.tpm_limit,
                self.rpd_limit,
            )
        else:
            logger.info(
                "RateLimitHandler [%s/%s]: retry-only mode (no quota tracking configured)",
                self.provider,
                model_name,
            )

        self._gate: Optional[ConcurrencyGate] = (
            ConcurrencyGate(concurrency_limit, concurrency_max_wait_seconds)
            if concurrency_limit
            else None
        )
        if self._gate:
            logger.info(
                "RateLimitHandler [%s/%s]: concurrency limit=%d",
                self.provider,
                model_name,
                concurrency_limit,
            )

    # ------------------------------------------------------------------
    # Concurrency
    # ------------------------------------------------------------------

    def waiting_count(self) -> int:
        """Current number of calls waiting for a concurrency slot (0 if
        no concurrency limit is configured for this provider)."""
        return self._gate.waiting_count() if self._gate else 0

    # ------------------------------------------------------------------
    # Persistence (quota tracking only)
    # ------------------------------------------------------------------

    def _today(self) -> str:
        return datetime.now().strftime("%Y-%m-%d")

    def _now_str(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _state_key(self) -> str:
        return f"{self.provider}:{self.model_name}"

    def _load_all(self) -> dict:
        if not os.path.exists(PERSISTENCE_FILE):
            return {}
        try:
            with open(PERSISTENCE_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning("RateLimitHandler: could not read %s, starting fresh.", PERSISTENCE_FILE)
            return {}

    def _save_all(self, data: dict) -> None:
        try:
            with open(PERSISTENCE_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            logger.warning("RateLimitHandler: could not save %s: %s", PERSISTENCE_FILE, e)

    def _load_state(self) -> None:
        data = self._load_all()
        entry = data.get(self._state_key(), {})
        if entry.get("date") == self._today():
            self._day_request_count = entry.get("day_request_count", 0)
            logger.info(
                "RateLimitHandler [%s]: resumed — %d requests used today.",
                self._state_key(),
                self._day_request_count,
            )
        else:
            self._day_request_count = 0

    def _save_state(self) -> None:
        data = self._load_all()
        data[self._state_key()] = {
            "date": self._today(),
            "day_request_count": self._day_request_count,
            "saved_at": self._now_str(),
            "minute_requests": f"{self._current_minute_requests()}/{self.rpm_limit}",
            "minute_tokens": f"{self._current_minute_tokens()}/{self.tpm_limit}",
            "day_requests": f"{self._day_request_count}/{self.rpd_limit}",
        }
        self._save_all(data)

    # ------------------------------------------------------------------
    # Sliding window helpers (quota tracking only)
    # ------------------------------------------------------------------

    def _cleanup_windows(self) -> None:
        now = time.time()
        minute_ago = now - 60
        while self._minute_requests and self._minute_requests[0] < minute_ago:
            self._minute_requests.popleft()
        while self._minute_tokens and self._minute_tokens[0][0] < minute_ago:
            self._minute_tokens.popleft()

    def _current_minute_requests(self) -> int:
        self._cleanup_windows()
        return len(self._minute_requests)

    def _current_minute_tokens(self) -> int:
        self._cleanup_windows()
        return sum(t for _, t in self._minute_tokens)

    def _record_request(self, tokens_used: int) -> None:
        now = time.time()
        self._minute_requests.append(now)
        self._minute_tokens.append((now, tokens_used))
        self._day_request_count += 1
        self._save_state()

    # ------------------------------------------------------------------
    # Proactive delay (quota tracking only)
    # ------------------------------------------------------------------

    def _wait_if_needed(self, estimated_tokens: int) -> None:
        while True:
            self._cleanup_windows()
            now = time.time()
            wait_time = 0.0

            # Daily quota
            if self.rpd_limit and self._day_request_count >= self.rpd_limit:
                midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                until_midnight = (midnight.timestamp() + 86_400) - now
                wait_time = max(wait_time, until_midnight + 1)
                logger.warning(
                    "RateLimitHandler [%s]: daily limit reached (%d/%d). "
                    "Waiting %.0fs until midnight.",
                    self._state_key(),
                    self._day_request_count,
                    self.rpd_limit,
                    wait_time,
                )

            # RPM
            if self.rpm_limit and self._current_minute_requests() >= self.rpm_limit:
                oldest = self._minute_requests[0]
                wait_time = max(wait_time, (oldest + 60) - now + 1)
                logger.info(
                    "RateLimitHandler [%s]: RPM limit reached. Waiting %.1fs.",
                    self._state_key(),
                    wait_time,
                )

            # TPM
            if (
                self.tpm_limit
                and self._current_minute_tokens() + estimated_tokens >= self.tpm_limit
            ):
                if self._minute_tokens:
                    oldest = self._minute_tokens[0][0]
                    wait_time = max(wait_time, (oldest + 60) - now + 1)
                    logger.info(
                        "RateLimitHandler [%s]: TPM limit reached. Waiting %.1fs.",
                        self._state_key(),
                        wait_time,
                    )

            if wait_time <= 0:
                break

            logger.info(
                "RateLimitHandler [%s]: sleeping %.1fs for rate limit.",
                self._state_key(),
                wait_time,
            )
            time.sleep(wait_time)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token estimate: 1 token ≈ 4 chars."""
        return max(1, len(text) // 4)

    @staticmethod
    def _parse_retry_delay(error_message: str) -> Optional[float]:
        """Extract retryDelay from Google error messages if present."""
        match = re.search(r"retryDelay.*?(\d+)s", str(error_message))
        if match:
            return float(match.group(1)) + 2.0
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _user_friendly_error(self, e: Exception) -> str:
        """
        Convert a raw API exception into a clean user-facing message.
        Only called after all retries are exhausted.
        """
        msg = str(e)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
            return (
                "Rate limit reached — quota exhausted for this minute or day. "
                "Please wait a few minutes and try again."
            )
        if "503" in msg or "UNAVAILABLE" in msg or "high demand" in msg.lower():
            return (
                "The model is temporarily unavailable due to high demand. "
                "Please try again in a few minutes."
            )
        if "500" in msg or "INTERNAL" in msg:
            return "The model returned an internal server error. Please try again shortly."
        clean = msg.split("\n")[0][:120]
        return f"Model error: {clean}"

    def call_with_retry(
        self,
        fn: Callable,
        *args,
        estimated_tokens: int = 500,
        on_queued: Optional[Callable[[int], None]] = None,
        **kwargs,
    ) -> Any:
        """
        Call fn(*args, **kwargs) with rate limiting, concurrency gating,
        and retry logic.

        For Google provider:
          - Proactive wait if RPM / TPM / RPD limits are close
          - Retry on 429 (ResourceExhausted / ClientError) with Google-suggested
            delay or exponential backoff
          - Retry on 503 / 500 (ServiceUnavailable / ServerError) with backoff
          - Non-retryable errors are re-raised immediately

        For all other providers:
          - Proactive RPM/TPM/RPD wait too, IF an explicit override was
            configured for this provider+model (see core/key_manager.py)
          - Simple exponential backoff retry on any Exception
          - Re-raises after max_retries exhausted

        A concurrency slot (if configured) is held only around the actual
        fn(*args, **kwargs) call itself, not during backoff sleeps between
        retries -- holding it while sleeping would waste capacity another
        waiting caller could have used.

        Parameters
        ----------
        fn              : Callable to invoke (e.g. llm_with_tools.invoke)
        *args           : Positional arguments forwarded to fn
        estimated_tokens: Approximate tokens for TPM tracking
        on_queued       : Optional callback(waiting_count), forwarded to
                          ConcurrencyGate.acquire()'s on_wait -- see that
                          method's docstring. None (the default) if the
                          caller isn't running inside an SSE-streamed
                          request (e.g. CLI/API direct use).
        **kwargs        : Keyword arguments forwarded to fn
        """
        if self._is_google:
            return self._call_google(
                fn, *args, estimated_tokens=estimated_tokens, on_queued=on_queued, **kwargs
            )
        else:
            return self._call_generic(
                fn, *args, estimated_tokens=estimated_tokens, on_queued=on_queued, **kwargs
            )

    def _call_gated(
        self, fn: Callable, *args, on_queued: Optional[Callable[[int], None]] = None, **kwargs
    ) -> Any:
        """Invoke fn, holding the concurrency slot (if any) only for the duration of the call itself."""
        if self._gate is not None:
            self._gate.acquire(on_wait=on_queued)
            try:
                return fn(*args, **kwargs)
            finally:
                self._gate.release()
        return fn(*args, **kwargs)

    def _call_google(
        self,
        fn: Callable,
        *args,
        estimated_tokens: int,
        on_queued: Optional[Callable[[int], None]] = None,
        **kwargs,
    ) -> Any:
        last_exc = None
        server_attempts = 0

        for attempt in range(self.max_retries):
            self._wait_if_needed(estimated_tokens)

            try:
                result = self._call_gated(fn, *args, on_queued=on_queued, **kwargs)
                self._record_request(estimated_tokens)
                return result

            except _GOOGLE_RATE_ERRORS as e:
                last_exc = e
                suggested = self._parse_retry_delay(str(e))
                backoff = suggested if suggested else self.base_backoff * (2**attempt)
                logger.warning(
                    "RateLimitHandler [%s]: rate limit (attempt %d/%d). Waiting %.1fs.",
                    self._state_key(),
                    attempt + 1,
                    self.max_retries,
                    backoff,
                )
                time.sleep(backoff)

            except _GOOGLE_SERVER_ERRORS as e:
                last_exc = e
                server_attempts += 1
                backoff = self.base_backoff * (2**attempt)
                logger.warning(
                    "RateLimitHandler [%s]: server error %s (attempt %d/%d). Waiting %.1fs.",
                    self._state_key(),
                    type(e).__name__,
                    server_attempts,
                    self.max_retries,
                    backoff,
                )
                if server_attempts >= self.max_retries:
                    logger.error(
                        "RateLimitHandler [%s]: server error retries exhausted.",
                        self._state_key(),
                    )
                    raise last_exc
                time.sleep(backoff)

            except Exception as e:
                logger.error("RateLimitHandler [%s]: non-retryable error: %s", self._state_key(), e)
                raise

        logger.error(
            "RateLimitHandler [%s]: all %d retries exhausted.", self._state_key(), self.max_retries
        )
        raise RuntimeError(self._user_friendly_error(last_exc)) from last_exc

    def _call_generic(
        self,
        fn: Callable,
        *args,
        estimated_tokens: int = 500,
        on_queued: Optional[Callable[[int], None]] = None,
        **kwargs,
    ) -> Any:
        last_exc = None
        for attempt in range(self.max_retries):
            if self._track_quota:
                self._wait_if_needed(estimated_tokens)
            try:
                result = self._call_gated(fn, *args, on_queued=on_queued, **kwargs)
                if self._track_quota:
                    self._record_request(estimated_tokens)
                return result
            except TimeoutError:
                # Concurrency-gate timeout -- the gate itself already
                # waited up to its own max_wait_seconds; retrying here
                # would just multiply that wait across every retry
                # attempt for no benefit, contrary to this module's own
                # "fail fast with a clear response" goal (see
                # ConcurrencyGate's docstring). Surface immediately.
                logger.warning(
                    "RateLimitHandler [%s]: concurrency queue timed out.", self._state_key()
                )
                raise
            except Exception as e:
                last_exc = e
                backoff = self.base_backoff * (2**attempt)
                logger.warning(
                    "RateLimitHandler [%s]: error on attempt %d/%d — %s. Waiting %.1fs.",
                    self._state_key(),
                    attempt + 1,
                    self.max_retries,
                    type(e).__name__,
                    backoff,
                )
                time.sleep(backoff)

        logger.error(
            "RateLimitHandler [%s]: all %d retries exhausted.", self._state_key(), self.max_retries
        )
        raise RuntimeError(self._user_friendly_error(last_exc)) from last_exc

    def status(self) -> dict:
        """Return current rate limit + concurrency counters."""
        base = {
            "provider": self.provider,
            "model": self.model_name,
            "concurrency_limit": self._gate.limit if self._gate else None,
            "concurrency_waiting": self.waiting_count(),
        }
        if not self._track_quota:
            base["tracking"] = "none"
            return base
        self._cleanup_windows()
        base.update(
            {
                "minute_requests": f"{self._current_minute_requests()}/{self.rpm_limit or '-'}",
                "minute_tokens": f"{self._current_minute_tokens()}/{self.tpm_limit or '-'}",
                "day_requests": f"{self._day_request_count}/{self.rpd_limit or '-'}",
            }
        )
        return base


# ---------------------------------------------------------------------------
# Shared-instance registry — see module docstring for why this exists.
# ---------------------------------------------------------------------------

_shared_handlers: dict[tuple[str, str], RateLimitHandler] = {}
_shared_handlers_lock = threading.Lock()


def get_shared_rate_limit_handler(
    provider: str, model_name: str, multi_tenant: bool, max_retries: int = 5
) -> RateLimitHandler:
    """
    Return the process-wide-shared RateLimitHandler for (provider,
    model_name, multi_tenant), building it on first use from
    core/key_manager.py's configured concurrency limit and quota
    overrides FOR THIS MODE (single-user and multi-tenant settings are
    independent namespaces -- see key_manager.py's own comment on why).
    Application code (agent/agent.py) should always go through this
    rather than constructing RateLimitHandler directly -- see module
    docstring.
    """
    key = (provider.lower(), model_name, bool(multi_tenant))
    with _shared_handlers_lock:
        handler = _shared_handlers.get(key)
        if handler is None:
            from local_search_agent.core.key_manager import (
                get_concurrency_limits,
                get_quota_overrides,
            )

            concurrency_limits = get_concurrency_limits(multi_tenant)
            overrides = get_quota_overrides(multi_tenant, provider).get(model_name, {})

            handler = RateLimitHandler(
                provider=provider,
                model_name=model_name,
                max_retries=max_retries,
                requests_per_minute=overrides.get("rpm"),
                tokens_per_minute=overrides.get("tpm"),
                requests_per_day=overrides.get("rpd"),
                concurrency_limit=concurrency_limits.get(provider.lower()),
            )
            _shared_handlers[key] = handler
        return handler


def reset_shared_rate_limit_handlers() -> None:
    """
    Drop every cached shared handler so the next get_shared_rate_limit_handler()
    call for any (provider, model_name) rebuilds from current
    core/key_manager.py settings. Call this whenever concurrency limits or
    quota overrides are changed via the UI/CLI/API, so updated limits take
    effect immediately rather than only after a process restart --
    correctness (never silently running under stale limits) over
    caching efficiency, same principle AppState.invalidate_agents() already
    follows for the agent cache.
    """
    with _shared_handlers_lock:
        _shared_handlers.clear()
