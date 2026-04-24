"""Agent 4: Risk Scanner Agent — replaces a Risk/Compliance Officer.

Two-stage pipeline:
  1. Keyword/regex pattern matching across all RAG chunks.
  2. LLM confirmation via Gemini to validate severity and produce reasoning.

Returns a partial AlphaLensState dict with ``risk_flags`` and updated
``metadata`` (agent latency).
"""

import json
import logging
import re
import time
from typing import Any

from google import genai
from google.genai import types

from src.config import GEMINI_MODEL, GOOGLE_API_KEY
from src.state import AlphaLensState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gemini client (module-level singleton)
# ---------------------------------------------------------------------------

_gemini_client = genai.Client(api_key=GOOGLE_API_KEY)


def _call_gemini(prompt: str, max_retries: int = 3) -> str:
    """Call Gemini with exponential back-off on transient errors.

    Args:
        prompt: The full text prompt to send to Gemini.
        max_retries: Maximum number of attempts before giving up.

    Returns:
        The model's text response, or an empty string if all attempts fail.
    """
    for attempt in range(max_retries):
        try:
            resp = _gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=256,
                ),
            )
            return resp.text or ""
        except Exception as exc:
            is_quota = "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)
            wait = 65.0 if is_quota else (2.0 ** attempt)
            logger.warning(
                "Gemini attempt %d failed: %s — waiting %.0fs", attempt + 1, exc, wait
            )
            if attempt < max_retries - 1:
                time.sleep(wait)
    return ""


# ---------------------------------------------------------------------------
# Risk pattern registry
# ---------------------------------------------------------------------------

_RISK_PATTERNS: dict[str, tuple[list[str], str]] = {
    "going_concern": (
        [
            "going concern",
            "substantial doubt",
            r"ability to continue as a going concern",
        ],
        "HIGH",
    ),
    "auditor_change": (
        [
            r"change in (?:independent )?auditor",
            "auditor resigned",
            r"new (?:independent )?registered public accounting",
        ],
        "HIGH",
    ),
    "revenue_recognition": (
        [
            "change in revenue recognition",
            "adopted asc 606",
            "revenue recognition policy change",
        ],
        "MEDIUM",
    ),
    "related_party": (
        [
            r"related.party transaction",
            "related party",
        ],
        "MEDIUM",
    ),
    "material_weakness": (
        [
            "material weakness",
            "significant deficiency",
            r"ineffective.*internal control",
        ],
        "HIGH",
    ),
    "litigation": (
        [
            r"class action",
            r"regulatory.{0,20}investigation",
            "enforcement action",
            r"securities.{0,20}lawsuit",
        ],
        "MEDIUM",
    ),
    "insider_selling": (
        [
            r"10b5.1 plan",
            r"officers?.{0,10}sold",
            "insider selling",
        ],
        "LOW",
    ),
    "covenant_violation": (
        [
            "covenant violation",
            "breach of covenant",
            "waiver of covenant",
        ],
        "HIGH",
    ),
    "customer_concentration": (
        [
            r"customer.{0,20}concentration",
            r"\d+% of (?:net )?revenue",
            "single customer",
        ],
        "MEDIUM",
    ),
}

# Pre-compile patterns once at import time for speed.
_COMPILED_PATTERNS: dict[str, tuple[list[re.Pattern], str]] = {
    risk_type: (
        [re.compile(p, re.IGNORECASE) for p in patterns],
        default_severity,
    )
    for risk_type, (patterns, default_severity) in _RISK_PATTERNS.items()
}

# Maximum number of flagged chunks to send to LLM (rate-limit guard).
_MAX_LLM_FLAGS = 10


# ---------------------------------------------------------------------------
# Stage 1: keyword/pattern scan
# ---------------------------------------------------------------------------


def _scan_chunks(chunks: list[dict]) -> list[dict]:
    """Scan RAG chunks for risk patterns and return candidate flag dicts.

    Args:
        chunks: List of RAG chunk dicts.  Each dict must have at minimum a
            ``text`` key and a ``metadata`` key (which itself is a dict).

    Returns:
        List of candidate flag dicts (not yet LLM-confirmed).  May contain
        multiple flags for the same chunk if it matched several patterns.
    """
    candidates: list[dict] = []

    for chunk in chunks:
        text: str = chunk.get("text", "") or chunk.get("content", "")
        metadata: dict = chunk.get("metadata", {})

        for risk_type, (compiled_pats, default_severity) in _COMPILED_PATTERNS.items():
            matched = any(pat.search(text) for pat in compiled_pats)
            if not matched:
                continue

            candidate: dict[str, Any] = {
                "flag_type": risk_type,
                "severity": default_severity,
                "description": f"Pattern match for '{risk_type}' in {metadata.get('section_name', 'unknown')} section.",
                "text_excerpt": text[:300],
                "full_text": text,
                "citation": {
                    "source_file": metadata.get("source_file", ""),
                    "section_name": metadata.get("section_name", ""),
                    "estimated_page": metadata.get("estimated_page", metadata.get("page_number", 0)),
                    "filing_type": metadata.get("filing_type", metadata.get("form_type", "")),
                    "filing_date": metadata.get("filing_date", ""),
                },
                "confirmed_by_llm": False,
                "llm_reasoning": "",
            }
            candidates.append(candidate)

    return candidates


# ---------------------------------------------------------------------------
# Stage 2: LLM confirmation
# ---------------------------------------------------------------------------


def _build_llm_prompt(
    ticker: str,
    flag: dict,
) -> str:
    """Build a concise (<500 token) Gemini prompt for risk confirmation.

    Args:
        ticker: The equity ticker symbol.
        flag: Candidate flag dict produced by ``_scan_chunks``.

    Returns:
        Formatted prompt string.
    """
    citation = flag["citation"]
    form_type = citation.get("filing_type", "filing")
    filing_date = citation.get("filing_date", "unknown date")
    section_name = citation.get("section_name", "unknown section")
    risk_type = flag["flag_type"]
    excerpt = flag["full_text"][:800]

    prompt = (
        "You are a financial risk analyst reviewing an SEC filing excerpt.\n"
        f"Ticker: {ticker}, Filing: {form_type} {filing_date}, Section: {section_name}\n\n"
        "Excerpt:\n"
        f"{excerpt}\n\n"
        f"Is this a real {risk_type} risk requiring investor attention?\n"
        'Respond with valid JSON only: {"confirmed": true/false, "severity": "HIGH"|"MEDIUM"|"LOW", "reasoning": "one sentence"}'
    )
    return prompt


def _confirm_with_llm(ticker: str, candidates: list[dict]) -> list[dict]:
    """Run LLM confirmation on up to ``_MAX_LLM_FLAGS`` candidate flags.

    Flags beyond the cap are passed through with ``confirmed_by_llm=False``.

    Args:
        ticker: The equity ticker symbol.
        candidates: Output of ``_scan_chunks``.

    Returns:
        Confirmed and cleaned flag dicts.
    """
    confirmed_flags: list[dict] = []

    for idx, flag in enumerate(candidates):
        if idx >= _MAX_LLM_FLAGS:
            # Beyond the rate-limit cap — pass through keyword result.
            cleaned = _clean_flag(flag)
            confirmed_flags.append(cleaned)
            continue

        prompt = _build_llm_prompt(ticker, flag)

        try:
            raw_response = _call_gemini(prompt)
        except Exception as exc:
            logger.warning("Unexpected error calling Gemini for flag %d: %s", idx, exc)
            raw_response = ""

        if not raw_response:
            # Gemini failed (quota / parse) — keep keyword result.
            cleaned = _clean_flag(flag)
            confirmed_flags.append(cleaned)
            continue

        # Strip markdown code fences if present.
        cleaned_response = raw_response.strip()
        if cleaned_response.startswith("```"):
            cleaned_response = re.sub(r"^```(?:json)?\s*", "", cleaned_response)
            cleaned_response = re.sub(r"\s*```$", "", cleaned_response)
            cleaned_response = cleaned_response.strip()

        try:
            llm_result: dict = json.loads(cleaned_response)
        except json.JSONDecodeError as exc:
            logger.warning(
                "JSON parse failed for LLM risk confirmation (flag %d): %s — raw: %r",
                idx,
                exc,
                raw_response[:200],
            )
            cleaned = _clean_flag(flag)
            confirmed_flags.append(cleaned)
            continue

        confirmed = bool(llm_result.get("confirmed", True))
        if not confirmed:
            # LLM judged it a false positive — skip this flag.
            logger.debug("LLM rejected flag type='%s' for %s", flag["flag_type"], ticker)
            continue

        # Accept LLM's severity if it provided one, else keep pattern default.
        llm_severity = llm_result.get("severity", flag["severity"])
        if llm_severity not in ("HIGH", "MEDIUM", "LOW"):
            llm_severity = flag["severity"]

        reasoning = str(llm_result.get("reasoning", "")).strip()

        updated_flag = dict(flag)
        updated_flag["severity"] = llm_severity
        updated_flag["description"] = reasoning if reasoning else flag["description"]
        updated_flag["confirmed_by_llm"] = True
        updated_flag["llm_reasoning"] = reasoning

        confirmed_flags.append(_clean_flag(updated_flag))

    return confirmed_flags


def _clean_flag(flag: dict) -> dict:
    """Strip internal-only keys before storing in state.

    Args:
        flag: Raw flag dict that may contain ``full_text``.

    Returns:
        Public flag dict matching the AlphaLens contract.
    """
    return {
        "flag_type": flag["flag_type"],
        "severity": flag["severity"],
        "description": flag["description"],
        "text_excerpt": flag["text_excerpt"],
        "citation": flag["citation"],
        "confirmed_by_llm": flag["confirmed_by_llm"],
        "llm_reasoning": flag["llm_reasoning"],
    }


# ---------------------------------------------------------------------------
# Agent entry point
# ---------------------------------------------------------------------------


def risk_scanner_agent(state: AlphaLensState) -> dict:
    """Scan RAG chunks for SEC-filing risk signals and confirm with Gemini.

    This is Agent 4 in the AlphaLens LangGraph pipeline.  It reads
    ``state["rag_chunks"]`` and ``state["financial_data"]``, runs a two-stage
    keyword + LLM risk detection process, and returns a partial state dict.

    Stage 1 — Keyword/regex pattern matching:
        Every chunk is scanned against ``_RISK_PATTERNS``.  Matches produce
        candidate flag dicts with default severity.

    Stage 2 — LLM confirmation:
        Up to ``_MAX_LLM_FLAGS`` candidates are sent to Gemini to validate
        whether the match is a true risk and to refine severity.  Rate-limit
        errors (HTTP 429) keep the keyword-matched result unchanged.

    Args:
        state: The current LangGraph shared state.

    Returns:
        Partial state dict with keys ``risk_flags`` (list[dict]) and
        ``metadata`` (dict containing ``agent_latencies.risk_scanner`` in
        seconds).
    """
    start_time = time.time()

    rag_chunks: list[dict] = state.get("rag_chunks") or []
    financial_data: dict = state.get("financial_data") or {}
    ticker: str = state.get("ticker", "UNKNOWN")

    if not rag_chunks:
        logger.info("risk_scanner: no RAG chunks available for %s — skipping", ticker)
        return {
            "risk_flags": [],
            "metadata": {"agent_latencies": {"risk_scanner": 0}},
        }

    logger.info(
        "risk_scanner: scanning %d chunks for %s", len(rag_chunks), ticker
    )

    # Stage 1 — keyword scan.
    try:
        candidates = _scan_chunks(rag_chunks)
    except Exception as exc:
        logger.error("risk_scanner stage-1 failed for %s: %s", ticker, exc, exc_info=True)
        candidates = []

    logger.info(
        "risk_scanner: found %d candidate flags after pattern matching", len(candidates)
    )

    # Stage 2 — LLM confirmation.
    if candidates:
        try:
            risk_flags = _confirm_with_llm(ticker, candidates)
        except Exception as exc:
            logger.error(
                "risk_scanner stage-2 failed for %s: %s", ticker, exc, exc_info=True
            )
            # Fall back to unconfirmed keyword results.
            risk_flags = [_clean_flag(c) for c in candidates]
    else:
        risk_flags = []

    elapsed = time.time() - start_time
    logger.info(
        "risk_scanner: completed for %s — %d flags confirmed in %.1fs",
        ticker,
        len(risk_flags),
        elapsed,
    )

    return {
        "risk_flags": risk_flags,
        "metadata": {"agent_latencies": {"risk_scanner": round(elapsed, 2)}},
    }
