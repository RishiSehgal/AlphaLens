"""Data Fusion Agent — Agent 1 of the AlphaLens LangGraph pipeline.

Replaces: Junior Analyst
Role: Gather and normalize all raw financial data for a given ticker from
      SEC EDGAR, Alpha Vantage (via MarketDataClient), and FRED in parallel.
      Returns a partial state dict that LangGraph merges into AlphaLensState.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from src.config import RATE_LIMITS
from src.data.edgar_client import EdgarClient
from src.data.fred_client import FredClient
from src.data.market_data import MarketDataClient
from src.state import AlphaLensState

logger = logging.getLogger(__name__)


# ── Helper ─────────────────────────────────────────────────────────────────────

def _av_float(report_list: list, field: str, index: int = 0) -> float | None:
    """Safely extract a float from an Alpha Vantage annualReports list.

    Args:
        report_list: List of annual report dicts from Alpha Vantage.
        field: Key to extract from the report dict.
        index: Which report to read (0 = most recent).

    Returns:
        Parsed float, or None if the value is absent, "None", or non-numeric.
    """
    try:
        val = report_list[index].get(field, "None")
        return float(val) if val not in (None, "None", "") else None
    except (IndexError, TypeError, ValueError):
        return None


# ── Core normalisation ─────────────────────────────────────────────────────────

def _normalize(
    market_data: dict[str, Any],
    filing_urls: list[dict],
    macro: dict[str, Any],
) -> dict[str, Any]:
    """Build the canonical ``financial_data`` dict from raw source payloads.

    Prefers Alpha Vantage structured statements for income / balance / cashflow
    fields. Falls back to yfinance where AV data is absent or degraded.

    Args:
        market_data: Output of ``MarketDataClient.get_financial_data()``.
        filing_urls: Output of ``EdgarClient.get_filing_urls()``.
        macro: Output of ``FredClient.get_macro_snapshot()``.

    Returns:
        Normalized dict containing all financial fields plus ``sources_status``.
    """
    yf: dict = market_data.get("yfinance", {})
    income: dict | None = market_data.get("income_statement")
    balance: dict | None = market_data.get("balance_sheet")
    cashflow: dict | None = market_data.get("cash_flow")

    annual_income: list = (income or {}).get("annualReports", [])
    annual_balance: list = (balance or {}).get("annualReports", [])
    annual_cashflow: list = (cashflow or {}).get("annualReports", [])

    info_raw: dict = yf.get("info_raw", {})

    # ── Identity ───────────────────────────────────────────────────────────────
    ticker: str = market_data.get("ticker", "")
    company_name: str = yf.get("long_name", ticker)
    sector: str | None = yf.get("sector")
    industry: str | None = yf.get("industry")

    # ── Price & market ratios (always from yfinance) ───────────────────────────
    price: float | None = yf.get("price")
    market_cap: float | None = yf.get("market_cap")
    pe_ratio: float | None = yf.get("pe_ratio")
    forward_pe: float | None = yf.get("forward_pe")
    dividend_yield: float | None = yf.get("dividend_yield")
    beta: float | None = yf.get("beta")
    week_52_high: float | None = yf.get("week_52_high")
    week_52_low: float | None = yf.get("week_52_low")

    # ── Income statement ───────────────────────────────────────────────────────
    av_revenue = _av_float(annual_income, "totalRevenue")
    revenue: float | None = av_revenue if av_revenue is not None else yf.get("revenue_ttm")

    av_net_income = _av_float(annual_income, "netIncome")
    net_income: float | None = av_net_income if av_net_income is not None else yf.get("net_income_ttm")

    av_gross_profit = _av_float(annual_income, "grossProfit")
    gross_margin: float | None = None
    if av_gross_profit is not None and revenue:
        try:
            gross_margin = av_gross_profit / revenue
        except ZeroDivisionError:
            gross_margin = None

    av_operating_income = _av_float(annual_income, "operatingIncome")
    operating_margin: float | None = None
    if av_operating_income is not None and revenue:
        try:
            operating_margin = av_operating_income / revenue
        except ZeroDivisionError:
            operating_margin = None

    # EPS — always from yfinance trailing EPS
    eps: float | None = info_raw.get("trailingEps")

    # ── Cash flow ──────────────────────────────────────────────────────────────
    av_op_cf = _av_float(annual_cashflow, "operatingCashflow")
    cash_flow_from_operations: float | None = av_op_cf if av_op_cf is not None else (
        info_raw.get("operatingCashflow")
    )

    av_capex = _av_float(annual_cashflow, "capitalExpenditures")
    free_cash_flow: float | None = None
    if cash_flow_from_operations is not None and av_capex is not None:
        free_cash_flow = cash_flow_from_operations - abs(av_capex)

    # ── Balance sheet ──────────────────────────────────────────────────────────
    total_debt: float | None = yf.get("total_debt")
    total_cash: float | None = yf.get("total_cash")

    av_equity = _av_float(annual_balance, "totalShareholderEquity")
    av_total_debt_bs = _av_float(annual_balance, "shortLongTermDebtTotal")
    # Prefer yfinance total_debt; AV balance sheet as secondary
    effective_debt = total_debt if total_debt is not None else av_total_debt_bs
    debt_to_equity: float | None = None
    if effective_debt is not None and av_equity and av_equity != 0:
        try:
            debt_to_equity = effective_debt / av_equity
        except ZeroDivisionError:
            debt_to_equity = None

    av_current_assets = _av_float(annual_balance, "totalCurrentAssets")
    av_current_liabilities = _av_float(annual_balance, "totalCurrentLiabilities")
    current_ratio: float | None = None
    if av_current_assets is not None and av_current_liabilities and av_current_liabilities != 0:
        try:
            current_ratio = av_current_assets / av_current_liabilities
        except ZeroDivisionError:
            current_ratio = None

    # ── Sources status ─────────────────────────────────────────────────────────
    sources_status: dict[str, str] = dict(market_data.get("sources_status", {}))
    sources_status["edgar"] = "ok" if filing_urls else "failed"
    sources_status["fred"] = "ok" if macro.get("fetch_success") else "degraded"

    return {
        # Identity
        "ticker": ticker,
        "company_name": company_name,
        "sector": sector,
        "industry": industry,
        # Price / market
        "price": price,
        "market_cap": market_cap,
        "pe_ratio": pe_ratio,
        "forward_pe": forward_pe,
        "dividend_yield": dividend_yield,
        "beta": beta,
        "week_52_high": week_52_high,
        "week_52_low": week_52_low,
        # Income statement
        "revenue": revenue,
        "net_income": net_income,
        "gross_margin": gross_margin,
        "operating_margin": operating_margin,
        "eps": eps,
        # Cash flow
        "cash_flow_from_operations": cash_flow_from_operations,
        "free_cash_flow": free_cash_flow,
        # Balance sheet
        "total_debt": total_debt,
        "total_cash": total_cash,
        "debt_to_equity": debt_to_equity,
        "current_ratio": current_ratio,
        # Downstream inputs
        "filing_urls": filing_urls,
        "macro": macro,
        "sources_status": sources_status,
    }


# ── Agent entry point ──────────────────────────────────────────────────────────

def data_fusion_agent(state: AlphaLensState) -> dict:
    """Fetch and normalize all raw financial data for the ticker in ``state``.

    Runs three data fetches in parallel:
      1. ``MarketDataClient.get_financial_data`` — price, ratios, AV statements
      2. ``EdgarClient.get_filing_urls`` — SEC EDGAR filing metadata
      3. ``FredClient.get_macro_snapshot`` — macro economic indicators

    Normalizes the combined payload into a flat ``financial_data`` dict and
    records its own wall-clock latency under ``metadata["agent_latencies"]``.

    Args:
        state: Shared LangGraph state. Reads ``state["ticker"]``.

    Returns:
        Partial state dict with updated keys:
          ``financial_data``, ``metadata``, ``error_log``.
        Never raises — errors are captured in ``error_log`` and the pipeline
        continues with whatever data was successfully retrieved.
    """
    ticker: str = state["ticker"].upper().strip()
    logger.info("[DataFusion] Starting for ticker=%s", ticker)

    start_time = time.monotonic()
    error_log: list[str] = []

    # ── Parallel fetch ─────────────────────────────────────────────────────────
    market_data: dict[str, Any] = {}
    filing_urls: list[dict] = []
    macro: dict[str, Any] = {"fetch_success": False}

    def _fetch_market() -> dict[str, Any]:
        logger.info("[DataFusion] Fetching market data for %s", ticker)
        return MarketDataClient().get_financial_data(ticker)

    def _fetch_edgar() -> list[dict]:
        logger.info("[DataFusion] Fetching EDGAR filing URLs for %s", ticker)
        return EdgarClient(max_filings_per_type=2).get_filing_urls(ticker)

    def _fetch_fred() -> dict[str, Any]:
        logger.info("[DataFusion] Fetching FRED macro snapshot")
        return FredClient().get_macro_snapshot()

    tasks = {
        "market": _fetch_market,
        "edgar": _fetch_edgar,
        "fred": _fetch_fred,
    }

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(fn): name for name, fn in tasks.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                if name == "market":
                    market_data = result
                elif name == "edgar":
                    filing_urls = result
                elif name == "fred":
                    macro = result
                logger.info("[DataFusion] '%s' fetch complete", name)
            except Exception as exc:
                msg = f"[DataFusion] '{name}' fetch raised an exception: {exc}"
                logger.error(msg)
                error_log.append(msg)

    # ── Normalize ──────────────────────────────────────────────────────────────
    try:
        financial_data = _normalize(market_data, filing_urls, macro)
    except Exception as exc:
        msg = f"[DataFusion] Normalization failed: {exc}"
        logger.error(msg)
        error_log.append(msg)
        # Return minimal stub so downstream agents don't crash on missing key
        financial_data = {
            "ticker": ticker,
            "sources_status": {"alpha_vantage": "failed", "yfinance": "failed",
                               "edgar": "failed", "fred": "failed"},
            "filing_urls": filing_urls,
            "macro": macro,
        }

    # ── Latency recording ──────────────────────────────────────────────────────
    elapsed = round(time.monotonic() - start_time, 2)
    logger.info("[DataFusion] Completed in %.2fs for %s", elapsed, ticker)

    return {
        "financial_data": financial_data,
        "metadata": {
            "agent_latencies": {"data_fusion": elapsed},
        },
        "error_log": error_log,
    }
