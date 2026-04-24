"""Central configuration: env vars, model constants, API rate limits, UI tokens.

Key resolution order:
  1. Environment variables (.env via python-dotenv)
  2. st.secrets (Streamlit Cloud) — resolved lazily at runtime via get_api_key()
  3. Empty string (graceful degradation)
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def get_api_key(key: str) -> str:
    """Resolve an API key from env vars, then st.secrets (lazy, never at import time)."""
    val = os.getenv(key, "")
    if val:
        return val
    try:
        import streamlit as st
        val = st.secrets.get(key, "")
        if val:
            return str(val)
    except Exception:
        pass
    return ""


# ── API keys — read from env at import time (st.secrets never touched here) ──
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
ALPHA_VANTAGE_API_KEY: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")
FRED_API_KEY: str = os.getenv("FRED_API_KEY", "")

if not GOOGLE_API_KEY:
    logger.warning("GOOGLE_API_KEY not set — Gemini calls will fail")

# ── Model identifiers ─────────────────────────────────────────────────────────
GEMINI_MODEL: str = "gemini-2.5-flash"
EMBEDDING_MODEL: str = "gemini-embedding-001"  # text-embedding-004 retired; use via google-genai SDK

# ── RAG parameters ────────────────────────────────────────────────────────────
CHUNK_SIZE: int = 500       # tokens
CHUNK_OVERLAP: int = 100    # tokens
TOP_K: int = 5

# ── Rate limits (requests per minute unless noted) ────────────────────────────
RATE_LIMITS: dict[str, int] = {
    "gemini_rpm": 15,
    "edgar_rps": 10,        # requests per second
    "alpha_vantage_rpd": 25,  # requests per day
}

# ── UI color system (dark theme) ──────────────────────────────────────────────
COLORS: dict[str, str] = {
    "bg_primary": "#0A0A0F",
    "bg_card": "#12121A",
    "bg_elevated": "#1A1A25",
    "bg_input": "#1E1E2A",
    "accent": "#3B82F6",
    "green": "#10B981",
    "amber": "#F59E0B",
    "red": "#EF4444",
    "text_primary": "#F1F5F9",
    "text_secondary": "#94A3B8",
    "text_muted": "#64748B",
    "border": "#1E293B",
    "border_hover": "#334155",
}

# ── Plotly chart template ─────────────────────────────────────────────────────
PLOTLY_TEMPLATE: dict = {
    "paper_bgcolor": COLORS["bg_card"],
    "plot_bgcolor": COLORS["bg_card"],
    "font": {
        "color": COLORS["text_secondary"],
        "family": "Inter, sans-serif",
        "size": 12,
    },
    "xaxis": {
        "gridcolor": COLORS["border"],
        "zerolinecolor": COLORS["border"],
    },
    "yaxis": {
        "gridcolor": COLORS["border"],
        "zerolinecolor": COLORS["border"],
    },
    "colorway": [
        COLORS["accent"],
        COLORS["green"],
        COLORS["amber"],
        COLORS["red"],
        "#8B5CF6",
    ],
}
