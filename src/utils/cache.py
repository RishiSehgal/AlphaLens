"""Semantic cache keyed by (ticker, date) with 24-hour TTL.

Stores full pipeline state snapshots in .alphalens_cache.json.
Avoids re-running the entire pipeline for the same ticker within a trading day.
"""

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_CACHE_PATH = Path(__file__).resolve().parents[3] / ".alphalens_cache.json"
_TTL_SECONDS = 86_400  # 24 hours


def _load_cache() -> dict:
    if not _CACHE_PATH.exists():
        return {}
    try:
        return json.loads(_CACHE_PATH.read_text())
    except Exception:
        return {}


def _save_cache(data: dict) -> None:
    try:
        _CACHE_PATH.write_text(json.dumps(data, indent=2, default=str))
    except Exception as exc:
        logger.debug("cache write failed: %s", exc)


def _cache_key(ticker: str) -> str:
    from datetime import date
    return f"{ticker.upper()}_{date.today().isoformat()}"


def get_cached_state(ticker: str) -> dict | None:
    """Return a cached pipeline state if one exists and is not expired.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        Cached AlphaLensState dict, or None if cache miss / expired.
    """
    cache = _load_cache()
    key = _cache_key(ticker)
    entry = cache.get(key)

    if not entry:
        return None

    age = time.time() - entry.get("cached_at", 0)
    if age > _TTL_SECONDS:
        logger.debug("Cache expired for %s (age %.0fs)", ticker, age)
        return None

    logger.info("Cache HIT for %s (age %.0fs)", ticker, age)
    return entry.get("state")


def set_cached_state(ticker: str, state: dict) -> None:
    """Store a pipeline state in the cache.

    Args:
        ticker: Stock ticker symbol.
        state: Completed AlphaLensState to cache.
    """
    cache = _load_cache()
    key = _cache_key(ticker)
    cache[key] = {
        "cached_at": time.time(),
        "state": state,
    }
    _save_cache(cache)
    logger.info("Cached state for %s", ticker)


def invalidate(ticker: str) -> None:
    """Remove a specific ticker's cache entry.

    Args:
        ticker: Stock ticker symbol.
    """
    cache = _load_cache()
    key = _cache_key(ticker)
    if key in cache:
        del cache[key]
        _save_cache(cache)
        logger.info("Invalidated cache for %s", ticker)


def clear_all() -> None:
    """Delete the entire cache file."""
    if _CACHE_PATH.exists():
        _CACHE_PATH.unlink()
        logger.info("Cache cleared")


def cache_stats() -> dict:
    """Return cache statistics.

    Returns:
        Dict with: total_entries, live_entries, expired_entries, size_bytes.
    """
    cache = _load_cache()
    now = time.time()
    live = sum(1 for e in cache.values() if (now - e.get("cached_at", 0)) <= _TTL_SECONDS)
    return {
        "total_entries": len(cache),
        "live_entries": live,
        "expired_entries": len(cache) - live,
        "size_bytes": _CACHE_PATH.stat().st_size if _CACHE_PATH.exists() else 0,
    }
