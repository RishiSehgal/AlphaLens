"""Market data layer — Alpha Vantage (primary) with yfinance fallback."""

import logging
import time
from typing import Any, Optional

import requests
import yfinance as yf

from src.config import ALPHA_VANTAGE_API_KEY

logger = logging.getLogger(__name__)

_AV_BASE = "https://www.alphavantage.co/query"
# Alpha Vantage free tier: 25 req/day — space calls ~4s apart to be safe
_AV_INTERVAL = 4.0


def _is_av_error(data: dict) -> bool:
    """Return True if Alpha Vantage returned a rate-limit or error payload."""
    return (
        "Note" in data
        or "Information" in data
        or "Error Message" in data
        or not data
    )


class MarketDataClient:
    """Fetches structured financial data from Alpha Vantage and yfinance.

    Alpha Vantage is used for income statement, balance sheet, and cash flow.
    yfinance covers real-time price, ratios, and basic fundamentals. If Alpha
    Vantage fails or hits its daily limit, the client falls back to yfinance
    and sets ``sources_status["alpha_vantage"] = "degraded"``.

    Args:
        av_api_key: Alpha Vantage API key. Defaults to value from config.
    """

    def __init__(self, av_api_key: Optional[str] = None) -> None:
        self._av_key = av_api_key or ALPHA_VANTAGE_API_KEY
        self._session = requests.Session()
        self._last_av_call: float = 0.0

    # ── Private helpers ────────────────────────────────────────────────────────

    def _av_throttle(self) -> None:
        """Enforce minimum gap between Alpha Vantage calls."""
        elapsed = time.monotonic() - self._last_av_call
        if elapsed < _AV_INTERVAL:
            time.sleep(_AV_INTERVAL - elapsed)
        self._last_av_call = time.monotonic()

    def _fetch_av(self, function: str, ticker: str) -> Optional[dict]:
        """Call one Alpha Vantage endpoint.

        Args:
            function: AV function name, e.g. "INCOME_STATEMENT".
            ticker: Stock ticker symbol.

        Returns:
            Parsed JSON dict, or None on error/rate-limit.
        """
        self._av_throttle()
        params = {"function": function, "symbol": ticker, "apikey": self._av_key}
        try:
            resp = self._session.get(_AV_BASE, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            if _is_av_error(data):
                logger.warning(
                    "Alpha Vantage error/limit for %s %s: %s", function, ticker, list(data.keys())
                )
                return None
            return data
        except requests.exceptions.Timeout:
            logger.warning("Alpha Vantage timeout for %s %s", function, ticker)
            return None
        except requests.exceptions.RequestException as exc:
            logger.warning("Alpha Vantage request failed for %s %s: %s", function, ticker, exc)
            return None
        except ValueError as exc:
            logger.warning("Alpha Vantage JSON decode error for %s %s: %s", function, ticker, exc)
            return None

    def _fetch_yfinance(self, ticker: str) -> dict[str, Any]:
        """Pull all available data from yfinance for a ticker.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Dict with price, ratios, fundamentals, and sector info.
        """
        try:
            t = yf.Ticker(ticker)
            info: dict = t.info or {}

            def _safe(key: str, default: Any = None) -> Any:
                return info.get(key, default)

            return {
                "price": _safe("currentPrice") or _safe("regularMarketPrice"),
                "market_cap": _safe("marketCap"),
                "pe_ratio": _safe("trailingPE"),
                "forward_pe": _safe("forwardPE"),
                "dividend_yield": _safe("dividendYield"),
                "week_52_high": _safe("fiftyTwoWeekHigh"),
                "week_52_low": _safe("fiftyTwoWeekLow"),
                "sector": _safe("sector"),
                "industry": _safe("industry"),
                "revenue_ttm": _safe("totalRevenue"),
                "net_income_ttm": _safe("netIncomeToCommon"),
                "total_debt": _safe("totalDebt"),
                "total_cash": _safe("totalCash"),
                "shares_outstanding": _safe("sharesOutstanding"),
                "beta": _safe("beta"),
                "currency": _safe("currency", "USD"),
                "long_name": _safe("longName", ticker),
                "info_raw": info,
            }
        except Exception as exc:
            logger.error("yfinance fetch failed for %s: %s", ticker, exc)
            return {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_financial_data(self, ticker: str) -> dict[str, Any]:
        """Return a unified financial data dict for a ticker.

        Attempts Alpha Vantage first for structured income/balance/cashflow
        statements. Falls back to yfinance if AV fails. Always returns at
        least yfinance data regardless of AV status.

        Args:
            ticker: Stock ticker symbol (e.g. "AAPL").

        Returns:
            Dict with keys:
                ticker (str),
                sources_status (dict: per-source "ok" | "degraded" | "failed"),
                yfinance (dict),
                income_statement (dict | None),
                balance_sheet (dict | None),
                cash_flow (dict | None).
        """
        result: dict[str, Any] = {
            "ticker": ticker.upper(),
            "sources_status": {
                "alpha_vantage": "ok",
                "yfinance": "ok",
            },
            "yfinance": {},
            "income_statement": None,
            "balance_sheet": None,
            "cash_flow": None,
        }

        logger.info("Fetching yfinance data for %s", ticker)
        yf_data = self._fetch_yfinance(ticker)
        if yf_data:
            result["yfinance"] = yf_data
        else:
            result["sources_status"]["yfinance"] = "failed"
            logger.error("yfinance returned no data for %s", ticker)

        if not self._av_key:
            logger.warning("ALPHA_VANTAGE_API_KEY not set — skipping AV, using yfinance only")
            result["sources_status"]["alpha_vantage"] = "degraded"
            return result

        av_ok = True
        for func, key in [
            ("INCOME_STATEMENT", "income_statement"),
            ("BALANCE_SHEET", "balance_sheet"),
            ("CASH_FLOW", "cash_flow"),
        ]:
            logger.info("Fetching Alpha Vantage %s for %s", func, ticker)
            data = self._fetch_av(func, ticker)
            if data is not None:
                result[key] = data
            else:
                av_ok = False
                logger.warning(
                    "Alpha Vantage %s unavailable for %s — using yfinance only", func, ticker
                )

        if not av_ok:
            result["sources_status"]["alpha_vantage"] = "degraded"

        return result

    def get_price_history(self, ticker: str, period: str = "6mo") -> Any:
        """Return OHLCV price history DataFrame via yfinance.

        Args:
            ticker: Stock ticker symbol.
            period: yfinance period string (e.g. "6mo", "1y").

        Returns:
            pandas DataFrame with OHLCV columns, or empty DataFrame on error.
        """
        import pandas as pd

        try:
            t = yf.Ticker(ticker)
            hist = t.history(period=period)
            if hist.empty:
                logger.warning("yfinance history empty for %s (%s)", ticker, period)
            return hist
        except Exception as exc:
            logger.error("yfinance price history failed for %s: %s", ticker, exc)
            return pd.DataFrame()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    client = MarketDataClient()
    data = client.get_financial_data("AAPL")

    print("\n=== Sources Status ===")
    for src, status in data["sources_status"].items():
        print(f"  {src}: {status}")

    print("\n=== yfinance Snapshot ===")
    yf_snap = data["yfinance"]
    print(f"  Name:       {yf_snap.get('long_name')}")
    print(f"  Price:      ${yf_snap.get('price')}")
    mc = yf_snap.get("market_cap")
    print(f"  Market Cap: ${mc:,}" if mc else "  Market Cap: N/A")
    print(f"  P/E:        {yf_snap.get('pe_ratio')}")
    print(f"  Sector:     {yf_snap.get('sector')}")
    print(f"  52w Range:  {yf_snap.get('week_52_low')} – {yf_snap.get('week_52_high')}")

    print("\n=== Alpha Vantage ===")
    for key in ["income_statement", "balance_sheet", "cash_flow"]:
        val = data[key]
        if val:
            annual = val.get("annualReports", [])
            print(f"  {key}: {len(annual)} annual period(s) available")
        else:
            print(f"  {key}: unavailable (degraded mode)")

    print("\n=== Price History (6mo, last 5 rows) ===")
    hist = client.get_price_history("AAPL")
    if not hist.empty:
        print(hist.tail(5)[["Open", "Close", "Volume"]].to_string())
    else:
        print("  No history available")
