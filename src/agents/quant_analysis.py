"""Quant Analysis Agent (Agent 3) — Quantitative Analyst replacement.

Computes three independent quantitative views from structured financial data
and live price history:

1. **DCF** — 3-stage discounted cash-flow model with bear / base / bull
   scenarios plus a 3×3 sensitivity table.  Always labelled MEDIUM confidence
   per project spec.
2. **Technicals** — RSI(14) and MACD(12/26/9) from 6-month price history.
3. **Earnings Surprise** — most-recent actual EPS vs. consensus estimate.

All computations are wrapped in try/except so a data gap in any section
produces a graceful None rather than a pipeline crash.
"""

import logging
import time
from typing import Any, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from src.data.market_data import MarketDataClient
from src.state import AlphaLensState

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
_EQUITY_RISK_PREMIUM: float = 0.055   # 5.5 % ERP assumed throughout
_WACC_MIN: float = 0.06
_WACC_MAX: float = 0.18
_TERMINAL_GROWTH: float = 0.025
_STAGE1_YEARS: int = 5
_STAGE2_YEARS: int = 5  # years 6-10

_STAGE1_GROWTH: dict[str, float] = {"bull": 0.18, "base": 0.10, "bear": 0.03}

# Sensitivity axes (growth_pct key = string like "0.10", wacc_pct = "0.10")
_SENS_GROWTH_RATES: list[float] = [0.03, 0.10, 0.18]
_SENS_WACC_ADJUSTMENTS: list[float] = [-0.02, 0.0, +0.02]  # relative to base WACC


# ── Helper: DCF single scenario ────────────────────────────────────────────────

def _dcf_scenario(
    fcf: float,
    wacc: float,
    stage1_growth: float,
    terminal_growth: float = _TERMINAL_GROWTH,
    stage1_years: int = _STAGE1_YEARS,
    stage2_years: int = _STAGE2_YEARS,
) -> float:
    """Return undiscounted-equivalent terminal-equivalent DCF equity value (total, not per share).

    Args:
        fcf: Most-recent trailing free cash flow (dollars).
        wacc: Weighted average cost of capital as a decimal.
        stage1_growth: Annual FCF growth rate for years 1-5.
        terminal_growth: Perpetuity growth rate after year 10.
        stage1_years: Number of high-growth years (Stage 1).
        stage2_years: Number of transition years (Stage 2).

    Returns:
        Total DCF value (sum of discounted FCFs + discounted terminal value).
    """
    total_pv: float = 0.0
    cf = fcf

    # Stage 1: constant high-growth FCF
    for year in range(1, stage1_years + 1):
        cf = cf * (1 + stage1_growth)
        pv = cf / ((1 + wacc) ** year)
        total_pv += pv

    # Stage 2: linearly interpolate growth from stage1_growth → terminal_growth
    cf_stage2_start = cf
    for step in range(1, stage2_years + 1):
        year = stage1_years + step
        # linear blend: step=1 → still near stage1_growth, step=5 → near terminal
        blend = step / stage2_years
        g = stage1_growth + blend * (terminal_growth - stage1_growth)
        cf_stage2_start = cf_stage2_start * (1 + g)
        pv = cf_stage2_start / ((1 + wacc) ** year)
        total_pv += pv

    # Terminal value at end of year 10
    terminal_fcf = cf_stage2_start * (1 + terminal_growth)
    terminal_value = terminal_fcf / (wacc - terminal_growth)
    total_years = stage1_years + stage2_years
    total_pv += terminal_value / ((1 + wacc) ** total_years)

    return total_pv


# ── Sub-section: DCF ───────────────────────────────────────────────────────────

def _compute_dcf(financial_data: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Build the 3-stage DCF and sensitivity table.

    Args:
        financial_data: Unified financial data dict from Agent 1.

    Returns:
        DCF result dict or None if critical inputs are missing.
    """
    try:
        # --- Extract inputs ---------------------------------------------------
        fcf: Optional[float] = financial_data.get("free_cash_flow")
        beta_raw = financial_data.get("beta") or financial_data.get("yfinance", {}).get("beta")
        beta: float = float(beta_raw) if beta_raw is not None else 1.0

        # Fed funds rate from FRED macro data
        macro: dict = financial_data.get("macro", {})
        fed_funds_raw = (
            macro.get("FEDFUNDS", {}).get("current_value")
            if isinstance(macro.get("FEDFUNDS"), dict)
            else macro.get("FEDFUNDS")
        )
        fed_funds: float = float(fed_funds_raw) / 100.0 if fed_funds_raw is not None else 0.05

        if fcf is None:
            # Attempt fallback: derive FCF from yfinance cash flow data
            yf_info = financial_data.get("yfinance", {})
            op_cf = yf_info.get("operatingCashflow") or yf_info.get("info_raw", {}).get("operatingCashflow")
            capex = yf_info.get("capitalExpenditures") or yf_info.get("info_raw", {}).get("capitalExpenditures")
            if op_cf is not None and capex is not None:
                fcf = float(op_cf) - abs(float(capex))
                logger.info("DCF: derived FCF from yfinance operatingCashflow − |capex| = %.2f", fcf)
            else:
                logger.warning("DCF: free_cash_flow missing and fallback unavailable — skipping DCF")
                return None

        fcf = float(fcf)
        if fcf <= 0:
            logger.warning("DCF: FCF is non-positive (%.2f) — skipping DCF", fcf)
            return None

        # --- WACC -------------------------------------------------------------
        wacc_raw = fed_funds + beta * _EQUITY_RISK_PREMIUM
        wacc = float(np.clip(wacc_raw, _WACC_MIN, _WACC_MAX))
        logger.info(
            "DCF: beta=%.2f fed_funds=%.4f raw_wacc=%.4f clamped_wacc=%.4f",
            beta, fed_funds, wacc_raw, wacc,
        )

        # --- Shares outstanding -----------------------------------------------
        shares: Optional[float] = None
        yf_info = financial_data.get("yfinance", {})
        shares_raw = yf_info.get("shares_outstanding") or yf_info.get("info_raw", {}).get("sharesOutstanding")
        if shares_raw:
            shares = float(shares_raw)
        else:
            # derive from market_cap / price
            mc = financial_data.get("market_cap") or yf_info.get("market_cap")
            price = financial_data.get("price") or yf_info.get("price")
            if mc and price and float(price) > 0:
                shares = float(mc) / float(price)

        if not shares or shares <= 0:
            logger.warning("DCF: shares_outstanding unavailable — per-share values will be None")

        # --- Scenarios --------------------------------------------------------
        scenarios: dict[str, float] = {}
        for scenario, g1 in _STAGE1_GROWTH.items():
            total_value = _dcf_scenario(fcf, wacc, g1)
            per_share = total_value / shares if shares else None
            scenarios[scenario] = round(per_share, 2) if per_share is not None else None

        # --- Sensitivity table (3×3) ──────────────────────────────────────────
        sensitivity: dict[str, float] = {}
        for g in _SENS_GROWTH_RATES:
            for delta_w in _SENS_WACC_ADJUSTMENTS:
                adj_wacc = float(np.clip(wacc + delta_w, _WACC_MIN, _WACC_MAX))
                total_val = _dcf_scenario(fcf, adj_wacc, g)
                per_share_val = (total_val / shares) if shares else None
                key = f"g={g:.0%},w={adj_wacc:.0%}"
                sensitivity[key] = round(per_share_val, 2) if per_share_val is not None else None

        return {
            "bear": scenarios.get("bear"),
            "base": scenarios.get("base"),
            "bull": scenarios.get("bull"),
            "wacc": round(wacc, 4),
            "confidence": "MEDIUM",
            "sensitivity": sensitivity,
        }

    except Exception as exc:
        logger.error("DCF computation failed: %s", exc, exc_info=True)
        return None


# ── Sub-section: Technicals ────────────────────────────────────────────────────

def _compute_technicals(ticker: str) -> Optional[dict[str, Any]]:
    """Compute RSI(14) and MACD(12/26/9) from 6-month price history.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        Dict with rsi, rsi_signal, macd, macd_signal_line, macd_histogram,
        macd_signal — or None if price history is unavailable.
    """
    try:
        client = MarketDataClient()
        hist: pd.DataFrame = client.get_price_history(ticker, period="6mo")

        if hist.empty or "Close" not in hist.columns:
            logger.warning("Technicals: price history unavailable for %s", ticker)
            return None

        close: pd.Series = hist["Close"].dropna()
        if len(close) < 27:  # need at least 27 rows for EMA-26
            logger.warning(
                "Technicals: insufficient price history (%d rows) for %s", len(close), ticker
            )
            return None

        # ---- RSI(14) ---------------------------------------------------------
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss
        rsi_series = 100 - 100 / (1 + rs)
        rsi: float = float(rsi_series.iloc[-1])

        if rsi < 30:
            rsi_signal = "oversold"
        elif rsi > 70:
            rsi_signal = "overbought"
        else:
            rsi_signal = "neutral"

        # ---- MACD(12/26/9) ---------------------------------------------------
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_series = ema12 - ema26
        signal_series = macd_series.ewm(span=9, adjust=False).mean()

        macd_line: float = float(macd_series.iloc[-1])
        signal_line: float = float(signal_series.iloc[-1])
        histogram: float = macd_line - signal_line
        macd_signal = "bullish" if histogram > 0 else "bearish"

        logger.info(
            "Technicals: %s RSI=%.2f (%s) MACD=%.4f signal=%.4f hist=%.4f (%s)",
            ticker, rsi, rsi_signal, macd_line, signal_line, histogram, macd_signal,
        )

        return {
            "rsi": round(rsi, 2),
            "rsi_signal": rsi_signal,
            "macd": round(macd_line, 4),
            "macd_signal_line": round(signal_line, 4),
            "macd_histogram": round(histogram, 4),
            "macd_signal": macd_signal,
        }

    except Exception as exc:
        logger.error("Technicals computation failed for %s: %s", ticker, exc, exc_info=True)
        return None


# ── Sub-section: Earnings Surprise ─────────────────────────────────────────────

def _compute_earnings_surprise(ticker: str) -> Optional[dict[str, Any]]:
    """Fetch most-recent EPS actual vs. estimate and compute the surprise.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        Dict with actual_eps, estimate_eps, surprise_pct, label — or None if
        earnings history is unavailable or malformed.
    """
    try:
        t = yf.Ticker(ticker)
        history = t.earnings_history  # DataFrame: epsActual, epsEstimate

        if history is None or not isinstance(history, pd.DataFrame) or history.empty:
            logger.warning("Earnings surprise: no earnings_history for %s", ticker)
            return None

        # Normalise column names (yfinance may use camelCase or lowercase)
        col_map: dict[str, str] = {}
        for col in history.columns:
            if col.lower() in ("epsactual", "eps actual"):
                col_map[col] = "epsActual"
            elif col.lower() in ("epsestimate", "eps estimate"):
                col_map[col] = "epsEstimate"
        if col_map:
            history = history.rename(columns=col_map)

        if "epsActual" not in history.columns or "epsEstimate" not in history.columns:
            logger.warning(
                "Earnings surprise: expected columns not found for %s. Columns: %s",
                ticker, list(history.columns),
            )
            return None

        # Most-recent row (first row after sorting descending by index date)
        recent = history.sort_index(ascending=False).iloc[0]
        actual: Optional[float] = recent.get("epsActual")
        estimate: Optional[float] = recent.get("epsEstimate")

        if actual is None or estimate is None or pd.isna(actual) or pd.isna(estimate):
            logger.warning("Earnings surprise: NaN EPS values for %s", ticker)
            return None

        actual = float(actual)
        estimate = float(estimate)

        if estimate == 0:
            logger.warning("Earnings surprise: estimate EPS is zero for %s — cannot compute %%", ticker)
            return None

        surprise_pct = (actual - estimate) / abs(estimate) * 100.0

        if surprise_pct > 5:
            label = "beat"
        elif surprise_pct < -5:
            label = "miss"
        else:
            label = "inline"

        logger.info(
            "Earnings surprise: %s actual=%.4f estimate=%.4f surprise=%.2f%% (%s)",
            ticker, actual, estimate, surprise_pct, label,
        )

        return {
            "actual_eps": round(actual, 4),
            "estimate_eps": round(estimate, 4),
            "surprise_pct": round(surprise_pct, 2),
            "label": label,
        }

    except Exception as exc:
        logger.error("Earnings surprise computation failed for %s: %s", ticker, exc, exc_info=True)
        return None


# ── Agent entry point ──────────────────────────────────────────────────────────

def quant_analysis_agent(state: AlphaLensState) -> dict:
    """LangGraph node: Quant Analysis Agent.

    Reads ``state["financial_data"]`` and produces three independent
    quantitative views: DCF valuation, technical indicators, and earnings
    surprise.  All sections are computed independently so a failure in one
    does not block the others.

    Args:
        state: Shared LangGraph pipeline state.

    Returns:
        Partial state dict with keys ``quant_results`` and ``metadata``.
        Never raises — any unrecoverable error is logged and surfaced via the
        returned ``error_log`` key on the state.
    """
    t0 = time.monotonic()
    ticker: str = state.get("ticker", "UNKNOWN").upper()
    financial_data: dict = state.get("financial_data", {})

    logger.info("quant_analysis_agent: starting for %s", ticker)

    error_messages: list[str] = []

    # 1. DCF -----------------------------------------------------------------
    dcf_result = _compute_dcf(financial_data)
    if dcf_result is None:
        error_messages.append(f"[quant_analysis] DCF unavailable for {ticker} — insufficient data")
        dcf_result = {
            "bear": None,
            "base": None,
            "bull": None,
            "wacc": None,
            "confidence": "MEDIUM",
            "sensitivity": {},
        }

    # 2. Technicals ----------------------------------------------------------
    technicals_result = _compute_technicals(ticker)
    if technicals_result is None:
        error_messages.append(
            f"[quant_analysis] Technicals unavailable for {ticker} — price history missing or too short"
        )

    # 3. Earnings Surprise ---------------------------------------------------
    earnings_result = _compute_earnings_surprise(ticker)
    if earnings_result is None:
        error_messages.append(
            f"[quant_analysis] Earnings surprise unavailable for {ticker} — no earnings history"
        )

    elapsed = time.monotonic() - t0
    logger.info("quant_analysis_agent: completed for %s in %.2fs", ticker, elapsed)

    output: dict = {
        "quant_results": {
            "dcf": dcf_result,
            "technicals": technicals_result,
            "earnings_surprise": earnings_result,
        },
        "metadata": {
            "agent_latencies": {
                "quant_analysis": round(elapsed, 3),
            }
        },
    }

    if error_messages:
        output["error_log"] = error_messages

    return output
