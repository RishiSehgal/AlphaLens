"""Graceful degradation wrappers for AlphaLens agent calls.

Provides a decorator and a context-manager that:
- Catch any exception from a callable
- Log it to the pipeline error_log
- Return a safe fallback value so the pipeline continues
"""

import functools
import logging
import traceback
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def with_fallback(
    fallback: Any,
    error_prefix: str = "",
    error_log: list[str] | None = None,
) -> Callable:
    """Decorator: catch all exceptions and return `fallback` instead of raising.

    Args:
        fallback: Value to return if the wrapped function raises.
        error_prefix: Short label prepended to the error message (e.g. "EDGAR_FETCH").
        error_log: Mutable list; if provided, the error string is appended here.

    Example::

        @with_fallback(fallback={}, error_prefix="MARKET_DATA")
        def get_price(ticker: str) -> dict:
            ...
    """
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> T:
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                msg = f"{error_prefix}: {type(exc).__name__}: {exc}" if error_prefix else str(exc)
                logger.error("with_fallback caught: %s", msg)
                if error_log is not None:
                    error_log.append(msg)
                return fallback  # type: ignore[return-value]
        return wrapper
    return decorator


def safe_call(
    fn: Callable[..., T],
    *args,
    fallback: T,
    error_log: list[str] | None = None,
    label: str = "",
    **kwargs,
) -> T:
    """Call `fn(*args, **kwargs)` and return `fallback` on any exception.

    Preferred over the decorator when you can't annotate the function definition.

    Args:
        fn: Callable to invoke.
        *args: Positional arguments for fn.
        fallback: Return value on exception.
        error_log: If provided, append the error string here.
        label: Short label for log messages.
        **kwargs: Keyword arguments for fn.

    Returns:
        fn(*args, **kwargs) on success, `fallback` on exception.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        prefix = f"[{label}] " if label else ""
        msg = f"{prefix}{type(exc).__name__}: {exc}"
        logger.error("safe_call error: %s", msg)
        logger.debug(traceback.format_exc())
        if error_log is not None:
            error_log.append(msg)
        return fallback


def agent_error_boundary(
    agent_name: str,
    state: dict,
    fallback_output: dict,
) -> Callable:
    """Decorator factory for LangGraph agent functions.

    Wraps the agent so that any unhandled exception:
    1. Logs the full traceback
    2. Appends a structured error to ``state["error_log"]``
    3. Returns ``fallback_output`` so LangGraph continues

    Usage::

        @agent_error_boundary("risk_scanner", state, {"risk_flags": []})
        def risk_scanner_agent(state): ...

    Args:
        agent_name: Name used in log messages and error_log entries.
        state: The LangGraph state dict (used to read the ticker for logging).
        fallback_output: Partial state dict returned on crash.
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            ticker = state.get("ticker", "?")
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                msg = f"{agent_name.upper()}_CRASH[{ticker}]: {type(exc).__name__}: {exc}"
                logger.error(msg, exc_info=True)
                # Merge error into fallback output
                out = dict(fallback_output)
                out["error_log"] = [msg]
                return out
        return wrapper
    return decorator


class SectionUnavailable(Exception):
    """Raised when a pipeline section cannot be generated and the caller should skip it."""
