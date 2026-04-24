"""Token-bucket rate limiters for each external API used by AlphaLens.

Each limiter is a module-level singleton so all callers share the same bucket.
"""

import logging
import threading
import time

logger = logging.getLogger(__name__)


class TokenBucket:
    """Thread-safe token bucket rate limiter.

    Args:
        rate: Tokens added per second.
        capacity: Maximum number of tokens the bucket can hold.
    """

    def __init__(self, rate: float, capacity: float) -> None:
        self._rate = rate
        self._capacity = capacity
        self._tokens = capacity
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        added = elapsed * self._rate
        self._tokens = min(self._capacity, self._tokens + added)
        self._last_refill = now

    def acquire(self, tokens: float = 1.0, timeout: float = 120.0) -> bool:
        """Block until `tokens` are available, then consume them.

        Args:
            tokens: Number of tokens to consume (default 1 = one request).
            timeout: Maximum seconds to wait before returning False.

        Returns:
            True if tokens were acquired, False if timeout exceeded.
        """
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True
                wait = (tokens - self._tokens) / self._rate

            if time.monotonic() + wait > deadline:
                logger.warning(
                    "Rate limiter timeout: %.1fs needed, %.1fs left",
                    wait, deadline - time.monotonic(),
                )
                return False

            time.sleep(min(wait, 0.1))


# ── Per-API singletons ────────────────────────────────────────────────────────
# Gemini free tier: 15 requests per minute
gemini_limiter = TokenBucket(rate=15 / 60, capacity=15)

# SEC EDGAR: 10 requests per second (be polite: cap at 8)
edgar_limiter = TokenBucket(rate=8.0, capacity=10)

# Alpha Vantage: 25 requests per day (~0.00029/s), throttle to 1 per 4s for safety
alpha_vantage_limiter = TokenBucket(rate=1 / 4, capacity=1)

# FRED: generous limits, cap at 5 req/s as courtesy
fred_limiter = TokenBucket(rate=5.0, capacity=10)

# yfinance: no hard limit, but courtesy cap
yfinance_limiter = TokenBucket(rate=2.0, capacity=5)


def wait_for_gemini() -> None:
    """Block until a Gemini request slot is available."""
    gemini_limiter.acquire()


def wait_for_edgar() -> None:
    """Block until an EDGAR request slot is available."""
    edgar_limiter.acquire()


def wait_for_alpha_vantage() -> None:
    """Block until an Alpha Vantage request slot is available."""
    alpha_vantage_limiter.acquire()


def wait_for_fred() -> None:
    """Block until a FRED request slot is available."""
    fred_limiter.acquire()


def wait_for_yfinance() -> None:
    """Block until a yfinance slot is available."""
    yfinance_limiter.acquire()
