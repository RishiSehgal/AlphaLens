"""Token usage and cost logger for AlphaLens.

Appends one JSON record per pipeline run to .alphalens_costs.json in the project root.
Gemini 2.0 Flash pricing (as of 2025): input $0.10/1M tokens, output $0.40/1M tokens.
All free-tier usage stays at $0 — tracker records token counts so costs become visible
if the account is ever upgraded.
"""

import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Gemini 2.0 Flash pricing (USD per 1M tokens)
_PRICE_INPUT_PER_M = 0.10
_PRICE_OUTPUT_PER_M = 0.40

_COST_LOG_PATH = Path(__file__).resolve().parents[3] / ".alphalens_costs.json"


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost for a single Gemini call.

    Args:
        input_tokens: Number of input tokens consumed.
        output_tokens: Number of output tokens generated.

    Returns:
        Estimated cost in USD (float).
    """
    return round(
        (input_tokens / 1_000_000) * _PRICE_INPUT_PER_M
        + (output_tokens / 1_000_000) * _PRICE_OUTPUT_PER_M,
        8,
    )


def log_usage(
    ticker: str,
    agent_name: str,
    input_tokens: int,
    output_tokens: int,
    api: str = "gemini",
    extra: dict | None = None,
) -> None:
    """Append a usage record to the cost log file.

    Args:
        ticker: Stock ticker this call was made for.
        agent_name: Name of the calling agent (e.g. "report_synthesis").
        input_tokens: Input token count.
        output_tokens: Output token count.
        api: API name tag (default "gemini").
        extra: Optional dict of extra fields to merge into the record.
    """
    record = {
        "ts": time.time(),
        "ticker": ticker,
        "agent": agent_name,
        "api": api,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": estimate_cost(input_tokens, output_tokens),
    }
    if extra:
        record.update(extra)

    try:
        existing: list[dict] = []
        if _COST_LOG_PATH.exists():
            try:
                existing = json.loads(_COST_LOG_PATH.read_text())
            except json.JSONDecodeError:
                existing = []

        existing.append(record)
        _COST_LOG_PATH.write_text(json.dumps(existing, indent=2))
    except Exception as exc:
        logger.debug("cost_tracker write failed: %s", exc)


def get_session_summary(ticker: str | None = None) -> dict:
    """Return aggregated usage stats from the cost log.

    Args:
        ticker: Filter by ticker. If None, returns stats for all tickers.

    Returns:
        Dict with: total_calls, total_input_tokens, total_output_tokens, total_cost_usd,
        and per_agent breakdown dict.
    """
    if not _COST_LOG_PATH.exists():
        return {"total_calls": 0, "total_input_tokens": 0, "total_output_tokens": 0,
                "total_cost_usd": 0.0, "per_agent": {}}

    try:
        records: list[dict] = json.loads(_COST_LOG_PATH.read_text())
    except Exception:
        return {"total_calls": 0, "total_input_tokens": 0, "total_output_tokens": 0,
                "total_cost_usd": 0.0, "per_agent": {}}

    if ticker:
        records = [r for r in records if r.get("ticker") == ticker.upper()]

    total_in = sum(r.get("input_tokens", 0) for r in records)
    total_out = sum(r.get("output_tokens", 0) for r in records)
    total_cost = sum(r.get("cost_usd", 0.0) for r in records)

    per_agent: dict[str, dict] = {}
    for r in records:
        agent = r.get("agent", "unknown")
        if agent not in per_agent:
            per_agent[agent] = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
        per_agent[agent]["calls"] += 1
        per_agent[agent]["input_tokens"] += r.get("input_tokens", 0)
        per_agent[agent]["output_tokens"] += r.get("output_tokens", 0)
        per_agent[agent]["cost_usd"] += r.get("cost_usd", 0.0)

    return {
        "total_calls": len(records),
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "total_cost_usd": round(total_cost, 6),
        "per_agent": per_agent,
    }


def clear_log() -> None:
    """Delete the cost log file (used in tests)."""
    if _COST_LOG_PATH.exists():
        _COST_LOG_PATH.unlink()
