"""Verification Agent — Agent 5 of the AlphaLens LangGraph pipeline.

Replaces: Senior Analyst
Role: Cross-reference all upstream agent outputs to identify divergences between
      management's narrative in SEC filings and the hard quantitative data from
      market APIs. Assigns per-section confidence scores and writes a verification
      verdict that feeds into the final report.

This agent is the "crown jewel" of the pipeline — it is the only component that
reasons about internal consistency across the full research package.
"""

import json
import logging
import time
from typing import Any

from google import genai
from google.genai import types

from src.config import GOOGLE_API_KEY, GEMINI_MODEL
from src.state import AlphaLensState

logger = logging.getLogger(__name__)

# ── Gemini client (module-level singleton) ────────────────────────────────────

_gemini_client: genai.Client | None = None

if GOOGLE_API_KEY:
    try:
        _gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
        logger.info("[Verification] Gemini client initialised (model=%s)", GEMINI_MODEL)
    except Exception as _init_exc:
        logger.warning(
            "[Verification] Failed to initialise Gemini client: %s", _init_exc
        )
        _gemini_client = None
else:
    logger.warning("[Verification] GOOGLE_API_KEY not set — LLM calls will be skipped")


# ── Gemini helper ─────────────────────────────────────────────────────────────

def _call_gemini(
    prompt: str,
    max_tokens: int = 1024,
    max_retries: int = 3,
) -> str:
    """Send a prompt to Gemini and return the raw text response.

    Handles two classes of transient failure:
    - HTTP 429 (rate-limit): waits 65 seconds before retrying.
    - Any other exception: exponential back-off (2^attempt seconds).

    Args:
        prompt: The fully-rendered prompt string to send.
        max_tokens: Maximum output tokens to request from the model.
        max_retries: Number of retry attempts before propagating the exception.

    Returns:
        The model's text response as a plain string.

    Raises:
        RuntimeError: If ``_gemini_client`` is None (API key not configured).
        Exception: The last exception raised after all retries are exhausted.
    """
    if _gemini_client is None:
        raise RuntimeError("Gemini client is not initialised — check GOOGLE_API_KEY")

    last_exc: Exception | None = None

    for attempt in range(max_retries):
        try:
            response = _gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.1,  # low temperature for factual cross-checking
                ),
            )
            return response.text or ""
        except Exception as exc:
            last_exc = exc
            exc_str = str(exc)

            if "429" in exc_str or "RESOURCE_EXHAUSTED" in exc_str.upper():
                wait = 65
                logger.warning(
                    "[Verification] Gemini 429 rate-limit on attempt %d — sleeping %ds",
                    attempt + 1,
                    wait,
                )
                time.sleep(wait)
            else:
                wait = 2 ** attempt
                logger.warning(
                    "[Verification] Gemini error on attempt %d (%s) — backing off %ds",
                    attempt + 1,
                    exc_str[:120],
                    wait,
                )
                time.sleep(wait)

    raise last_exc  # type: ignore[misc]


# ── Step 1: Build data summary ────────────────────────────────────────────────

def _build_data_summary(financial_data: dict[str, Any]) -> str:
    """Format a concise (<400 token) summary of key financial metrics.

    Args:
        financial_data: Normalized financial data dict from the Data Fusion agent.

    Returns:
        A single multi-line string with labelled metric rows and source status.
    """

    def _fmt_currency(val: float | None, unit: str = "B") -> str:
        """Format a currency value in billions or millions."""
        if val is None:
            return "N/A"
        divisor = 1e9 if unit == "B" else 1e6
        return f"${val / divisor:.2f}{unit}"

    def _fmt_pct(val: float | None) -> str:
        """Format a ratio as a percentage string."""
        if val is None:
            return "N/A"
        return f"{val * 100:.1f}%"

    def _fmt_float(val: float | None, decimals: int = 2) -> str:
        """Format a plain float."""
        if val is None:
            return "N/A"
        return f"{val:.{decimals}f}"

    sources: dict = financial_data.get("sources_status", {})
    av_status = sources.get("alpha_vantage", "unknown")
    yf_status = sources.get("yfinance", "unknown")

    revenue = _fmt_currency(financial_data.get("revenue"))
    net_income = _fmt_currency(financial_data.get("net_income"))
    gross_margin = _fmt_pct(financial_data.get("gross_margin"))
    op_margin = _fmt_pct(financial_data.get("operating_margin"))
    pe_ratio = _fmt_float(financial_data.get("pe_ratio"))
    market_cap = _fmt_currency(financial_data.get("market_cap"))
    eps = _fmt_float(financial_data.get("eps"))
    fcf = _fmt_currency(financial_data.get("free_cash_flow"))
    debt_equity = _fmt_float(financial_data.get("debt_to_equity"))
    current_ratio = _fmt_float(financial_data.get("current_ratio"))

    return (
        f"Revenue: {revenue} | Net Income: {net_income} | "
        f"Gross Margin: {gross_margin} | Op. Margin: {op_margin}\n"
        f"P/E: {pe_ratio} | Market Cap: {market_cap} | EPS: {eps}\n"
        f"Free Cash Flow: {fcf} | Debt/Equity: {debt_equity} | "
        f"Current Ratio: {current_ratio}\n"
        f"AV status: {av_status} | yfinance status: {yf_status}"
    )


# ── Step 2: Select top RAG excerpts ──────────────────────────────────────────

def _select_rag_excerpts(rag_chunks: list[dict], max_chunks: int = 6) -> list[dict]:
    """Pick the most relevant RAG chunks for verification context.

    Prioritises MD&A sections, then Risk Factors, then any remaining chunks
    up to ``max_chunks`` total.

    Args:
        rag_chunks: List of retrieved chunk dicts from the RAG Citation agent.
        max_chunks: Maximum number of chunks to return (default 6).

    Returns:
        Ordered list of up to ``max_chunks`` chunk dicts.
    """
    mda_chunks: list[dict] = []
    risk_chunks: list[dict] = []
    other_chunks: list[dict] = []

    for chunk in rag_chunks:
        section = chunk.get("metadata", {}).get("section_name", "").lower()
        if "management" in section or "discussion" in section or "md&a" in section:
            mda_chunks.append(chunk)
        elif "risk" in section:
            risk_chunks.append(chunk)
        else:
            other_chunks.append(chunk)

    selected: list[dict] = []
    for pool in (mda_chunks, risk_chunks, other_chunks):
        for chunk in pool:
            if len(selected) >= max_chunks:
                break
            selected.append(chunk)
        if len(selected) >= max_chunks:
            break

    return selected


def _build_excerpts_block(selected_chunks: list[dict], max_chars_each: int = 400) -> str:
    """Render selected chunks into a numbered plain-text block for the prompt.

    Args:
        selected_chunks: Chunks chosen by ``_select_rag_excerpts``.
        max_chars_each: Maximum characters to include per chunk (default 400).

    Returns:
        Formatted string with numbered excerpts and section labels.
    """
    if not selected_chunks:
        return "(No filing excerpts available — verification based on quantitative data only)"

    lines: list[str] = []
    for i, chunk in enumerate(selected_chunks, start=1):
        section = chunk.get("metadata", {}).get("section_name", "Unknown Section")
        text: str = chunk.get("text", chunk.get("content", "")).strip()
        excerpt = text[:max_chars_each]
        if len(text) > max_chars_each:
            excerpt += "…"
        lines.append(f"[{i}] Section: {section}\n{excerpt}")

    return "\n\n".join(lines)


# ── Step 3: Build and fire the Gemini prompt ─────────────────────────────────

def _build_prompt(
    data_summary: str,
    quant_results: dict[str, Any],
    excerpts_block: str,
    risk_count: int,
    high_count: int,
) -> str:
    """Assemble the verification prompt from rendered sub-sections.

    Keeps the total token count well under 2000 tokens.

    Args:
        data_summary: Output of ``_build_data_summary``.
        quant_results: Dict from the Quant Analysis agent.
        excerpts_block: Output of ``_build_excerpts_block``.
        risk_count: Total number of risk flags from the Risk Scanner agent.
        high_count: Number of HIGH-severity risk flags.

    Returns:
        The fully-rendered prompt string ready for Gemini.
    """
    # Extract quant fields safely
    dcf: dict = quant_results.get("dcf", {})
    dcf_base = dcf.get("base_case") or dcf.get("base") or "N/A"
    if isinstance(dcf_base, (int, float)):
        dcf_base = f"{dcf_base:.2f}"

    technicals: dict = quant_results.get("technicals", {})
    rsi = technicals.get("rsi")
    rsi_str = f"{rsi:.1f}" if isinstance(rsi, (int, float)) else "N/A"

    rsi_signal = "N/A"
    if isinstance(rsi, (int, float)):
        if rsi < 30:
            rsi_signal = "oversold"
        elif rsi > 70:
            rsi_signal = "overbought"
        else:
            rsi_signal = "neutral"

    macd_signal = technicals.get("macd_signal", technicals.get("macd_crossover", "N/A"))
    if not isinstance(macd_signal, str):
        macd_signal = str(macd_signal)

    return f"""You are a senior equity analyst performing fact-checking on a company's SEC filing disclosures.

FINANCIAL DATA (from market APIs, considered ground truth):
{data_summary}

QUANT RESULTS:
DCF base fair value: ${dcf_base} | RSI: {rsi_str} ({rsi_signal}) | MACD: {macd_signal}

FILING EXCERPTS (what management says in their 10-K):
{excerpts_block}

RISK FLAGS IDENTIFIED: {risk_count} flags ({high_count} HIGH severity)

Your task: Identify divergences between management's narrative and the quantitative data.
Look for: overstated growth claims, margin discrepancies, selective disclosure, contradictions with risk flags.

Respond with valid JSON only:
{{
  "divergences": [
    {{"claim": "what management says", "actual_data": "what numbers show", "severity": "HIGH|MEDIUM|LOW"}}
  ],
  "confidence_scores": {{
    "financial_health": "HIGH|MEDIUM|LOW",
    "risk_assessment": "HIGH|MEDIUM|LOW",
    "valuation": "MEDIUM",
    "overall": "HIGH|MEDIUM|LOW"
  }},
  "verification_verdict": "one paragraph summary of findings"
}}"""


# ── Step 4: Parse Gemini response ────────────────────────────────────────────

_SAFE_DEFAULTS: dict[str, Any] = {
    "divergences": [],
    "confidence_scores": {
        "financial_health": "MEDIUM",
        "risk_assessment": "MEDIUM",
        "valuation": "MEDIUM",
        "overall": "MEDIUM",
    },
    "verification_verdict": (
        "Verification unavailable — Gemini rate limit reached. "
        "Manual review recommended."
    ),
    "llm_available": False,
}


def _parse_gemini_response(raw_text: str) -> dict[str, Any]:
    """Extract and validate the JSON payload from the Gemini response.

    Strips markdown fences if present, then parses the JSON. Enforces:
    - ``divergences`` is a list with at most 5 entries.
    - ``confidence_scores.valuation`` is always ``"MEDIUM"`` (architecture spec).
    - All confidence values are one of ``HIGH|MEDIUM|LOW``; defaults to MEDIUM.
    - ``verification_verdict`` is a non-empty string.

    Args:
        raw_text: Raw string response from the Gemini API.

    Returns:
        Validated dict with keys ``divergences``, ``confidence_scores``,
        ``verification_verdict``, and ``llm_available=True``.
    """
    valid_levels = {"HIGH", "MEDIUM", "LOW"}

    # Strip markdown code fences if Gemini wrapped the JSON
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        # Drop first and last fence lines
        inner_lines = []
        in_block = False
        for line in lines:
            if line.strip().startswith("```") and not in_block:
                in_block = True
                continue
            if line.strip() == "```" and in_block:
                break
            if in_block:
                inner_lines.append(line)
        cleaned = "\n".join(inner_lines).strip()

    data = json.loads(cleaned)  # raises json.JSONDecodeError on bad input

    # ── Validate divergences ───────────────────────────────────────────────────
    raw_divergences = data.get("divergences", [])
    if not isinstance(raw_divergences, list):
        raw_divergences = []

    validated_divergences: list[dict] = []
    for item in raw_divergences[:5]:  # cap at 5
        if not isinstance(item, dict):
            continue
        severity = str(item.get("severity", "MEDIUM")).upper()
        if severity not in valid_levels:
            severity = "MEDIUM"
        validated_divergences.append(
            {
                "claim": str(item.get("claim", "")),
                "actual_data": str(item.get("actual_data", "")),
                "severity": severity,
            }
        )

    # ── Validate confidence scores ────────────────────────────────────────────
    raw_scores = data.get("confidence_scores", {})
    if not isinstance(raw_scores, dict):
        raw_scores = {}

    def _safe_level(key: str, default: str = "MEDIUM") -> str:
        val = str(raw_scores.get(key, default)).upper()
        return val if val in valid_levels else default

    confidence_scores = {
        "financial_health": _safe_level("financial_health"),
        "risk_assessment": _safe_level("risk_assessment"),
        "valuation": "MEDIUM",  # always MEDIUM per architecture spec
        "overall": _safe_level("overall"),
    }

    # ── Validate verdict ──────────────────────────────────────────────────────
    verdict = str(data.get("verification_verdict", "")).strip()
    if not verdict:
        verdict = "No verification verdict returned by the model."

    return {
        "divergences": validated_divergences,
        "confidence_scores": confidence_scores,
        "verification_verdict": verdict,
        "llm_available": True,
    }


# ── Agent entry point ─────────────────────────────────────────────────────────

def verification_agent(state: AlphaLensState) -> dict:
    """Cross-reference all upstream outputs to identify narrative vs. data divergences.

    This is Agent 5 — the Senior Analyst replacement and pipeline crown jewel.
    It performs a single, context-rich Gemini call that compares management's
    10-K disclosures against hard quantitative data, then returns structured
    divergences, per-section confidence scores, and an overall verdict.

    Steps:
        1. Build a concise (<400 token) financial data summary.
        2. Select up to 6 RAG chunks (preferring MD&A then Risk Factors).
        3. Fire a single Gemini call with a structured verification prompt.
        4. Parse and validate the JSON response; cap divergences at 5.

    Fallback: if Gemini is unavailable (persistent 429 or network error),
    returns empty divergences, all-MEDIUM confidence, and a user-visible
    "unavailable" verdict. The pipeline never crashes.

    Args:
        state: LangGraph pipeline state. Reads:
            - ``state["financial_data"]``: normalized financial metrics dict.
            - ``state["rag_chunks"]``: list of retrieved filing chunks.
            - ``state["quant_results"]``: DCF, technicals, earnings surprise.
            - ``state["risk_flags"]``: list of risk flag dicts with ``severity``.

    Returns:
        Partial state dict with keys:
            ``verification``: dict with ``divergences``, ``confidence_scores``,
                ``verification_verdict``, and ``llm_available``.
            ``metadata``: merged dict including ``agent_latencies.verification``.
            ``error_log``: list of error strings (appended, not replaced).
    """
    t_start = time.monotonic()
    error_log: list[str] = []

    ticker: str = state.get("ticker", "UNKNOWN").upper().strip()
    logger.info("[Verification] Starting for ticker=%s", ticker)

    financial_data: dict[str, Any] = state.get("financial_data") or {}
    rag_chunks: list[dict] = state.get("rag_chunks") or []
    quant_results: dict[str, Any] = state.get("quant_results") or {}
    risk_flags: list[dict] = state.get("risk_flags") or []

    # ── Risk flag counts for the prompt ──────────────────────────────────────
    risk_count = len(risk_flags)
    high_count = sum(
        1 for f in risk_flags
        if str(f.get("severity", "")).upper() == "HIGH"
    )
    logger.info(
        "[Verification] Context: %d rag_chunks, %d risk_flags (%d HIGH)",
        len(rag_chunks), risk_count, high_count,
    )

    # ── Step 1: Build concise data summary ───────────────────────────────────
    try:
        data_summary = _build_data_summary(financial_data)
    except Exception as exc:
        msg = f"verification: _build_data_summary raised: {exc}"
        logger.error("[Verification] %s", msg)
        error_log.append(msg)
        data_summary = "(Financial data unavailable)"

    # ── Step 2: Select top RAG excerpts ──────────────────────────────────────
    try:
        selected_chunks = _select_rag_excerpts(rag_chunks, max_chunks=6)
        excerpts_block = _build_excerpts_block(selected_chunks)
    except Exception as exc:
        msg = f"verification: RAG excerpt selection raised: {exc}"
        logger.error("[Verification] %s", msg)
        error_log.append(msg)
        excerpts_block = "(Filing excerpts unavailable)"

    # ── Step 3: Gemini call ───────────────────────────────────────────────────
    verification_result: dict[str, Any]

    if _gemini_client is None:
        logger.warning("[Verification] Gemini client not available — returning safe defaults")
        error_log.append("verification: Gemini client not initialised — skipping LLM step")
        verification_result = dict(_SAFE_DEFAULTS)
    else:
        try:
            prompt = _build_prompt(
                data_summary=data_summary,
                quant_results=quant_results,
                excerpts_block=excerpts_block,
                risk_count=risk_count,
                high_count=high_count,
            )
            logger.debug("[Verification] Prompt length: %d chars", len(prompt))

            raw_response = _call_gemini(prompt, max_tokens=1024, max_retries=3)
            logger.info(
                "[Verification] Gemini responded (%d chars)", len(raw_response)
            )

            # ── Step 4: Parse response ────────────────────────────────────────
            try:
                verification_result = _parse_gemini_response(raw_response)
                logger.info(
                    "[Verification] Parsed %d divergence(s), overall confidence=%s",
                    len(verification_result["divergences"]),
                    verification_result["confidence_scores"]["overall"],
                )
            except json.JSONDecodeError as parse_exc:
                msg = (
                    f"verification: JSON parse failed — {parse_exc}. "
                    f"Raw snippet: {raw_response[:200]!r}"
                )
                logger.error("[Verification] %s", msg)
                error_log.append(msg)
                verification_result = dict(_SAFE_DEFAULTS)

        except RuntimeError as rt_exc:
            msg = f"verification: Gemini client runtime error: {rt_exc}"
            logger.error("[Verification] %s", msg)
            error_log.append(msg)
            verification_result = dict(_SAFE_DEFAULTS)

        except Exception as exc:
            exc_str = str(exc)
            if "429" in exc_str or "RESOURCE_EXHAUSTED" in exc_str.upper():
                msg = "verification: Gemini rate-limit exhausted after retries — returning defaults"
            else:
                msg = f"verification: Gemini call failed after retries: {exc}"
            logger.error("[Verification] %s", msg)
            error_log.append(msg)
            verification_result = dict(_SAFE_DEFAULTS)

    elapsed = round(time.monotonic() - t_start, 2)
    logger.info(
        "[Verification] Completed in %.2fs for %s (llm_available=%s)",
        elapsed,
        ticker,
        verification_result.get("llm_available", False),
    )

    return {
        "verification": verification_result,
        "metadata": {
            "agent_latencies": {"verification": elapsed},
        },
        "error_log": error_log,
    }
