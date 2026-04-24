"""Plotly chart builders for AlphaLens — all using the dark PLOTLY_TEMPLATE."""

import logging
from typing import Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import COLORS, PLOTLY_TEMPLATE

logger = logging.getLogger(__name__)


def _base_layout(**extra) -> dict:
    layout = {
        **PLOTLY_TEMPLATE,
        "margin": {"l": 40, "r": 20, "t": 40, "b": 40},
        "showlegend": False,
    }
    layout.update(extra)
    return layout


def create_dcf_heatmap(quant_results: dict) -> Optional[go.Figure]:
    """Growth rate × discount rate → fair value heatmap.

    Args:
        quant_results: quant_analysis_agent output; expects ["dcf"].

    Returns:
        Plotly Figure or None if data is insufficient.
    """
    try:
        dcf = quant_results.get("dcf", {})
        base_price = dcf.get("base")
        if base_price is None:
            return None

        wacc = dcf.get("wacc", 0.10)
        g1 = dcf.get("growth_stage1", 0.10)

        discount_rates = [round(wacc - 0.03 + i * 0.01, 3) for i in range(7)]
        growth_rates = [round(g1 - 0.04 + i * 0.02, 3) for i in range(5)]

        def fair_value(g, d):
            terminal_g = min(g * 0.3, 0.035)
            if d <= terminal_g:
                return base_price
            scale = ((1 + g) / (1 + d)) ** 5
            return round(base_price * scale * max(0.5, (terminal_g / max(d - terminal_g, 0.01))), 2)

        z = [[fair_value(g, d) for d in discount_rates] for g in growth_rates]
        x_labels = [f"{r:.0%}" for r in discount_rates]
        y_labels = [f"{r:.0%}" for r in growth_rates]

        fig = go.Figure(
            go.Heatmap(
                z=z,
                x=x_labels,
                y=y_labels,
                colorscale=[
                    [0.0, COLORS["red"]],
                    [0.5, COLORS["amber"]],
                    [1.0, COLORS["green"]],
                ],
                text=[[f"${v:.0f}" for v in row] for row in z],
                texttemplate="%{text}",
                textfont={"size": 11, "color": "white"},
                showscale=True,
                colorbar=dict(
                    tickfont=dict(color=COLORS["text_secondary"], size=10),
                    outlinecolor=COLORS["border"],
                    outlinewidth=1,
                ),
            )
        )

        fig.update_layout(
            **_base_layout(
                title=dict(
                    text="DCF Sensitivity — Fair Value ($)",
                    font=dict(color=COLORS["text_primary"], size=13),
                    x=0,
                ),
                xaxis=dict(
                    title="Discount Rate (WACC)",
                    titlefont=dict(color=COLORS["text_muted"], size=11),
                    tickfont=dict(color=COLORS["text_secondary"], size=10),
                    gridcolor=COLORS["border"],
                ),
                yaxis=dict(
                    title="Revenue Growth Rate",
                    titlefont=dict(color=COLORS["text_muted"], size=11),
                    tickfont=dict(color=COLORS["text_secondary"], size=10),
                    gridcolor=COLORS["border"],
                ),
            )
        )
        return fig

    except Exception as exc:
        logger.warning("create_dcf_heatmap failed: %s", exc)
        return None


def create_rsi_gauge(rsi_value: float) -> Optional[go.Figure]:
    """Semicircular gauge with RSI zones: oversold/neutral/overbought.

    Args:
        rsi_value: RSI(14) float, expected 0–100.

    Returns:
        Plotly Figure or None on error.
    """
    try:
        rsi_value = float(rsi_value)
        if rsi_value < 30:
            needle_color = COLORS["green"]
            zone_label = "Oversold"
        elif rsi_value > 70:
            needle_color = COLORS["red"]
            zone_label = "Overbought"
        else:
            needle_color = COLORS["accent"]
            zone_label = "Neutral"

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=rsi_value,
                number={"font": {"color": COLORS["text_primary"], "size": 28}},
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickwidth": 1,
                        "tickcolor": COLORS["border"],
                        "tickfont": {"color": COLORS["text_secondary"], "size": 10},
                        "nticks": 6,
                    },
                    "bar": {"color": needle_color, "thickness": 0.25},
                    "bgcolor": COLORS["bg_card"],
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 30], "color": "rgba(16,185,129,0.12)"},
                        {"range": [30, 70], "color": "rgba(59,130,246,0.07)"},
                        {"range": [70, 100], "color": "rgba(239,68,68,0.12)"},
                    ],
                    "threshold": {
                        "line": {"color": needle_color, "width": 2},
                        "thickness": 0.75,
                        "value": rsi_value,
                    },
                },
                title={
                    "text": f"RSI (14)  ·  {zone_label}",
                    "font": {"color": COLORS["text_secondary"], "size": 12},
                },
                domain={"x": [0, 1], "y": [0, 1]},
            )
        )

        fig.update_layout(
            **_base_layout(
                height=220,
                margin={"l": 20, "r": 20, "t": 30, "b": 10},
            )
        )
        return fig

    except Exception as exc:
        logger.warning("create_rsi_gauge failed: %s", exc)
        return None


def create_price_macd_chart(ticker: str) -> Optional[go.Figure]:
    """6-month price line with MACD subplot fetched from yfinance.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        Plotly Figure or None if fetch fails.
    """
    try:
        import yfinance as yf

        hist = yf.Ticker(ticker).history(period="6mo")
        if hist.empty:
            return None

        closes = hist["Close"]
        dates = hist.index

        ema12 = closes.ewm(span=12, adjust=False).mean()
        ema26 = closes.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.65, 0.35],
            vertical_spacing=0.04,
        )

        fig.add_trace(
            go.Scatter(
                x=dates, y=closes,
                mode="lines",
                line=dict(color=COLORS["accent"], width=1.5),
                hovertemplate="$%{y:.2f}<extra>Close</extra>",
            ),
            row=1, col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=dates, y=macd_line,
                mode="lines",
                line=dict(color=COLORS["accent"], width=1.2),
                hovertemplate="%{y:.3f}<extra>MACD</extra>",
            ),
            row=2, col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=dates, y=signal_line,
                mode="lines",
                line=dict(color=COLORS["amber"], width=1.2),
                hovertemplate="%{y:.3f}<extra>Signal</extra>",
            ),
            row=2, col=1,
        )

        bar_colors = [COLORS["green"] if v >= 0 else COLORS["red"] for v in histogram]
        fig.add_trace(
            go.Bar(
                x=dates, y=histogram,
                marker_color=bar_colors,
                hovertemplate="%{y:.3f}<extra>Histogram</extra>",
            ),
            row=2, col=1,
        )

        shared_axis = dict(
            gridcolor=COLORS["border"],
            zerolinecolor=COLORS["border"],
            tickfont=dict(color=COLORS["text_secondary"], size=10),
        )
        fig.update_layout(
            paper_bgcolor=COLORS["bg_card"],
            plot_bgcolor=COLORS["bg_card"],
            font=dict(color=COLORS["text_secondary"], family="Inter, sans-serif", size=11),
            margin={"l": 50, "r": 20, "t": 40, "b": 30},
            showlegend=False,
            title=dict(
                text=f"{ticker.upper()} — Price & MACD (6mo)",
                font=dict(color=COLORS["text_primary"], size=13),
                x=0,
            ),
            hovermode="x unified",
            hoverlabel=dict(
                bgcolor=COLORS["bg_elevated"],
                bordercolor=COLORS["border"],
                font=dict(color=COLORS["text_primary"], size=12),
            ),
        )
        fig.update_xaxes(**shared_axis)
        fig.update_yaxes(**shared_axis)
        fig.update_yaxes(
            title_text=f"{ticker.upper()} ($)",
            title_font=dict(color=COLORS["text_muted"], size=10),
            row=1, col=1,
        )
        fig.update_yaxes(
            title_text="MACD",
            title_font=dict(color=COLORS["text_muted"], size=10),
            row=2, col=1,
        )

        return fig

    except Exception as exc:
        logger.warning("create_price_macd_chart failed for %s: %s", ticker, exc)
        return None


def create_earnings_chart(quant_results: dict) -> Optional[go.Figure]:
    """Horizontal bar: actual EPS vs consensus estimate.

    Args:
        quant_results: quant_analysis_agent output; uses ["earnings_surprise"].

    Returns:
        Plotly Figure or None if data unavailable.
    """
    try:
        es = quant_results.get("earnings_surprise", {})
        if not es or es.get("actual") is None or es.get("estimate") is None:
            return None

        actual = float(es["actual"])
        estimate = float(es["estimate"])
        surprise_pct = es.get("surprise_pct", 0.0)
        beat = actual >= estimate
        bar_color = COLORS["green"] if beat else COLORS["red"]
        beat_label = f"{'Beat' if beat else 'Miss'} {abs(surprise_pct):.1f}%"

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                y=["Consensus", "Actual"],
                x=[estimate, actual],
                orientation="h",
                marker=dict(
                    color=[COLORS["text_muted"], bar_color],
                    opacity=[0.5, 0.9],
                    line=dict(color=COLORS["border"], width=0.5),
                ),
                text=[f"${estimate:.2f}", f"${actual:.2f} ({beat_label})"],
                textposition="outside",
                textfont=dict(color=COLORS["text_primary"], size=12),
                hovertemplate="%{text}<extra></extra>",
            )
        )

        fig.update_layout(
            **_base_layout(
                title=dict(
                    text="Earnings Per Share — Actual vs Estimate",
                    font=dict(color=COLORS["text_primary"], size=13),
                    x=0,
                ),
                xaxis=dict(
                    title="EPS ($)",
                    titlefont=dict(color=COLORS["text_muted"], size=11),
                    tickfont=dict(color=COLORS["text_secondary"], size=10),
                    gridcolor=COLORS["border"],
                    zeroline=True,
                    zerolinecolor=COLORS["border"],
                ),
                yaxis=dict(
                    tickfont=dict(color=COLORS["text_secondary"], size=11),
                    gridcolor=COLORS["border"],
                ),
                height=160,
                margin={"l": 80, "r": 100, "t": 40, "b": 30},
            )
        )

        return fig

    except Exception as exc:
        logger.warning("create_earnings_chart failed: %s", exc)
        return None
