"""
Rate Limit Handler for Local Search Agent
==========================================

Wraps any callable (typically an LLM invoke call) with:

  - Proactive delay before hitting limits (RPM / TPM / RPD tracking)
  - Reactive retry on transient errors with exponential backoff
  - Per-model daily request count persisted to JSON so quotas survive restarts
    and reset automatically at calendar midnight

Provider behaviour
------------------
  google  : Full rate limiting — RPM / TPM / RPD tracking with sliding windows,
            retries on Google-specific 429 / 503 / 500 errors.
            Free-tier limits are auto-detected from the model name prefix:
              gemini-*   →  15 RPM  |  250,000 TPM  |  500 RPD
              gemma-4-*  →  15 RPM  |  unlimited TPM |  1,500 RPD

  others  : No quota tracking. Simple exponential-backoff retry on any
            transient error (configurable max_retries). Persistence file
            is not written for non-Google providers.

Usage
-----
    handler = RateLimitHandler(provider="google", model_name="gemma-4-31b-it")
    response = handler.call_with_retry(llm_with_tools.invoke, messages)
"""

from __future__ import annotations

import json
import logging
import os
import re
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
# Free-tier rate limits per Google model family
# ---------------------------------------------------------------------------

_GOOGLE_LIMITS = {
    "gemini": {
        "requests_per_minute": 15,
        "tokens_per_minute": 250_000,
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
    safety_margin   : Fraction of the hard limit to treat as the effective limit (Google only)
    requests_per_minute / tokens_per_minute / requests_per_day :
                      Override auto-detected Google limits if needed
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
    ):
        self.provider = provider.lower()
        self.model_name = model_name
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self._is_google = self.provider == "google"

        if self._is_google:
            detected = _detect_google_limits(model_name)
            rpm = requests_per_minute or detected["requests_per_minute"]
            tpm = tokens_per_minute or detected["tokens_per_minute"]
            rpd = requests_per_day or detected["requests_per_day"]

            self.rpm_limit = int(rpm * safety_margin)
            self.tpm_limit = int(tpm * safety_margin)
            self.rpd_limit = int(rpd * safety_margin)

            self._minute_requests: deque = deque()
            self._minute_tokens: deque = deque()
            self._day_request_count: int = 0
            self._load_state()

            logger.info(
                "RateLimitHandler [google/%s]: RPM=%d TPM=%d RPD=%d",
                model_name,
                self.rpm_limit,
                self.tpm_limit,
                self.rpd_limit,
            )
        else:
            logger.info(
                "RateLimitHandler [%s/%s]: retry-only mode (no quota tracking)",
                provider,
                model_name,
            )

    # ------------------------------------------------------------------
    # Persistence (Google only)
    # ------------------------------------------------------------------

    def _today(self) -> str:
        return datetime.now().strftime("%Y-%m-%d")

    def _now_str(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
        entry = data.get(self.model_name, {})
        if entry.get("date") == self._today():
            self._day_request_count = entry.get("day_request_count", 0)
            logger.info(
                "RateLimitHandler [%s]: resumed — %d requests used today.",
                self.model_name,
                self._day_request_count,
            )
        else:
            self._day_request_count = 0

    def _save_state(self) -> None:
        data = self._load_all()
        data[self.model_name] = {
            "date": self._today(),
            "day_request_count": self._day_request_count,
            "saved_at": self._now_str(),
            "minute_requests": f"{self._current_minute_requests()}/{self.rpm_limit}",
            "minute_tokens": f"{self._current_minute_tokens()}/{self.tpm_limit}",
            "day_requests": f"{self._day_request_count}/{self.rpd_limit}",
        }
        self._save_all(data)

    # ------------------------------------------------------------------
    # Sliding window helpers (Google only)
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
    # Proactive delay (Google only)
    # ------------------------------------------------------------------

    def _wait_if_needed(self, estimated_tokens: int) -> None:
        while True:
            self._cleanup_windows()
            now = time.time()
            wait_time = 0.0

            # Daily quota
            if self._day_request_count >= self.rpd_limit:
                midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                until_midnight = (midnight.timestamp() + 86_400) - now
                wait_time = max(wait_time, until_midnight + 1)
                logger.warning(
                    "RateLimitHandler [%s]: daily limit reached (%d/%d). "
                    "Waiting %.0fs until midnight.",
                    self.model_name,
                    self._day_request_count,
                    self.rpd_limit,
                    wait_time,
                )

            # RPM
            if self._current_minute_requests() >= self.rpm_limit:
                oldest = self._minute_requests[0]
                wait_time = max(wait_time, (oldest + 60) - now + 1)
                logger.info(
                    "RateLimitHandler [%s]: RPM limit reached. Waiting %.1fs.",
                    self.model_name,
                    wait_time,
                )

            # TPM
            if self._current_minute_tokens() + estimated_tokens >= self.tpm_limit:
                if self._minute_tokens:
                    oldest = self._minute_tokens[0][0]
                    wait_time = max(wait_time, (oldest + 60) - now + 1)
                    logger.info(
                        "RateLimitHandler [%s]: TPM limit reached. Waiting %.1fs.",
                        self.model_name,
                        wait_time,
                    )

            if wait_time <= 0:
                break

            logger.info(
                "RateLimitHandler [%s]: sleeping %.1fs for rate limit.",
                self.model_name,
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

    def call_with_retry(
        self,
        fn: Callable,
        *args,
        estimated_tokens: int = 500,
        **kwargs,
    ) -> Any:
        """
        Call fn(*args, **kwargs) with rate limiting and retry logic.

        For Google provider:
          - Proactive wait if RPM / TPM / RPD limits are close
          - Retry on 429 (ResourceExhausted / ClientError) with Google-suggested
            delay or exponential backoff
          - Retry on 503 / 500 (ServiceUnavailable / ServerError) with backoff
          - Non-retryable errors are re-raised immediately

        For all other providers:
          - Simple exponential backoff retry on any Exception
          - Re-raises after max_retries exhausted

        Parameters
        ----------
        fn              : Callable to invoke (e.g. llm_with_tools.invoke)
        *args           : Positional arguments forwarded to fn
        estimated_tokens: Approximate tokens for TPM tracking (Google only)
        **kwargs        : Keyword arguments forwarded to fn
        """
        if self._is_google:
            return self._call_google(fn, *args, estimated_tokens=estimated_tokens, **kwargs)
        else:
            return self._call_generic(fn, *args, **kwargs)

    def _call_google(
        self,
        fn: Callable,
        *args,
        estimated_tokens: int,
        **kwargs,
    ) -> Any:
        last_exc = None
        server_attempts = 0

        for attempt in range(self.max_retries):
            self._wait_if_needed(estimated_tokens)

            try:
                result = fn(*args, **kwargs)
                self._record_request(estimated_tokens)
                return result

            except _GOOGLE_RATE_ERRORS as e:
                last_exc = e
                suggested = self._parse_retry_delay(str(e))
                backoff = suggested if suggested else self.base_backoff * (2**attempt)
                logger.warning(
                    "RateLimitHandler [%s]: rate limit (attempt %d/%d). Waiting %.1fs.",
                    self.model_name,
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
                    self.model_name,
                    type(e).__name__,
                    server_attempts,
                    self.max_retries,
                    backoff,
                )
                if server_attempts >= self.max_retries:
                    logger.error(
                        "RateLimitHandler [%s]: server error retries exhausted.", self.model_name
                    )
                    raise last_exc
                time.sleep(backoff)

            except Exception as e:
                logger.error("RateLimitHandler [%s]: non-retryable error: %s", self.model_name, e)
                raise

        logger.error(
            "RateLimitHandler [%s]: all %d retries exhausted.", self.model_name, self.max_retries
        )
        raise last_exc

    def _call_generic(self, fn: Callable, *args, **kwargs) -> Any:
        last_exc = None
        for attempt in range(self.max_retries):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_exc = e
                backoff = self.base_backoff * (2**attempt)
                logger.warning(
                    "RateLimitHandler [%s]: error on attempt %d/%d — %s. Waiting %.1fs.",
                    self.model_name,
                    attempt + 1,
                    self.max_retries,
                    type(e).__name__,
                    backoff,
                )
                time.sleep(backoff)

        logger.error(
            "RateLimitHandler [%s]: all %d retries exhausted.", self.model_name, self.max_retries
        )
        raise last_exc

    def status(self) -> dict:
        """Return current rate limit counters. Google only — returns empty dict for other providers."""
        if not self._is_google:
            return {"provider": self.provider, "model": self.model_name, "tracking": "none"}
        self._cleanup_windows()
        return {
            "provider": self.provider,
            "model": self.model_name,
            "minute_requests": f"{self._current_minute_requests()}/{self.rpm_limit}",
            "minute_tokens": f"{self._current_minute_tokens()}/{self.tpm_limit}",
            "day_requests": f"{self._day_request_count}/{self.rpd_limit}",
        }
