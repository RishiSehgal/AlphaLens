"""Custom HTML/CSS component functions for AlphaLens dark-theme UI.

All functions return HTML strings intended for st.markdown(html, unsafe_allow_html=True).
"""

from src.config import COLORS

# ── Pulse animation CSS (injected once via render_agent_progress) ─────────────
_PULSE_CSS = """
<style>
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}
.agent-running { animation: pulse 1.5s ease-in-out infinite; color: #3B82F6; }
</style>
"""


def render_header() -> str:
    """Top-of-page header with gradient rule and AlphaLens branding."""
    return f"""
<div style="margin-bottom: 1.5rem;">
    <div style="
        height: 2px;
        background: linear-gradient(90deg, {COLORS['accent']}, #8B5CF6);
        border-radius: 1px;
        margin-bottom: 1rem;
    "></div>
    <div style="display: flex; align-items: baseline; gap: 12px;">
        <span style="
            font-size: 28px;
            font-weight: 700;
            color: {COLORS['text_primary']};
            letter-spacing: -0.5px;
            font-family: Inter, sans-serif;
        ">AlphaLens</span>
        <span style="
            font-size: 13px;
            color: {COLORS['text_muted']};
            font-family: Inter, sans-serif;
        ">AI Equity Research · 90-second deep-dive</span>
    </div>
</div>
"""


def render_metric_card(label: str, value: str, subtitle: str = "", color: str = "") -> str:
    """Dark card with muted uppercase label, large value, and colored left border.

    Args:
        label: Short uppercase label text.
        value: Primary display value.
        subtitle: Optional smaller text below value.
        color: CSS color for the left accent border. Defaults to accent blue.
    """
    border_color = color or COLORS["accent"]
    sub_html = (
        f'<div style="font-size:12px;color:{COLORS["text_muted"]};margin-top:4px;">'
        f"{subtitle}</div>"
        if subtitle
        else ""
    )
    return f"""
<div style="
    background: {COLORS['bg_card']};
    border: 1px solid {COLORS['border']};
    border-left: 3px solid {border_color};
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 10px;
">
    <div style="
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: {COLORS['text_muted']};
        margin-bottom: 6px;
        font-family: Inter, sans-serif;
    ">{label}</div>
    <div style="
        font-size: 22px;
        font-weight: 700;
        color: {COLORS['text_primary']};
        font-family: Inter, sans-serif;
        line-height: 1.2;
    ">{value}</div>
    {sub_html}
</div>
"""


def render_report_section(
    title: str,
    confidence: str,
    body_html: str,
    citations: list[dict] | None = None,
) -> str:
    """Report section card with confidence badge, body text, and citation links.

    Args:
        title: Section heading.
        confidence: "HIGH", "MEDIUM", or "LOW".
        body_html: HTML body content rendered inside the card.
        citations: List of citation dicts with keys: source_file, section_name, page_number.
    """
    conf_upper = (confidence or "LOW").upper()
    badge_color = {
        "HIGH": COLORS["green"],
        "MEDIUM": COLORS["amber"],
        "LOW": COLORS["red"],
    }.get(conf_upper, COLORS["red"])

    citation_html = ""
    if citations:
        links = []
        for c in citations:
            sec = c.get("section_name", "")
            pg = c.get("page_number", "")
            src = c.get("source_file", "")
            label = sec or src or "source"
            pg_str = f" p.{pg}" if pg else ""
            links.append(
                f'<span style="'
                f"display:inline-block;background:{COLORS['bg_elevated']};"
                f"border:1px solid {COLORS['border']};border-radius:4px;"
                f"padding:2px 8px;font-size:11px;color:{COLORS['accent']};"
                f'margin:2px 4px 2px 0;font-family:Inter,sans-serif;">'
                f"[{label}{pg_str}]</span>"
            )
        citation_html = (
            f'<div style="margin-top:12px;padding-top:10px;'
            f"border-top:1px solid {COLORS['border']}\">"
            + "".join(links)
            + "</div>"
        )

    return f"""
<div style="
    background: {COLORS['bg_card']};
    border: 1px solid {COLORS['border']};
    border-radius: 10px;
    padding: 20px 22px;
    margin-bottom: 14px;
">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;">
        <div style="
            font-size: 15px;
            font-weight: 600;
            color: {COLORS['text_primary']};
            font-family: Inter, sans-serif;
        ">{title}</div>
        <div style="
            background: {badge_color}22;
            border: 1px solid {badge_color}55;
            color: {badge_color};
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.06em;
            padding: 3px 10px;
            border-radius: 12px;
            font-family: Inter, sans-serif;
        ">{conf_upper}</div>
    </div>
    <div style="
        color: {COLORS['text_secondary']};
        font-size: 14px;
        line-height: 1.65;
        font-family: Inter, sans-serif;
    ">{body_html}</div>
    {citation_html}
</div>
"""


def render_risk_flag(
    flag_type: str,
    severity: str,
    description: str,
    citation: str = "",
) -> str:
    """Red-tinted card for a single risk flag with severity accent border.

    Args:
        flag_type: Category name (e.g. "going_concern").
        severity: "HIGH", "MEDIUM", or "LOW".
        description: Plain-text risk description.
        citation: Optional source reference string.
    """
    sev_upper = (severity or "LOW").upper()
    sev_color = {
        "HIGH": COLORS["red"],
        "MEDIUM": COLORS["amber"],
        "LOW": COLORS["green"],
    }.get(sev_upper, COLORS["amber"])

    icon = {"HIGH": "⚠", "MEDIUM": "▲", "LOW": "ℹ"}.get(sev_upper, "▲")

    cite_html = (
        f'<div style="margin-top:8px;font-size:11px;color:{COLORS["text_muted"]};">'
        f"Source: {citation}</div>"
        if citation
        else ""
    )

    label = flag_type.replace("_", " ").title()

    return f"""
<div style="
    background: rgba(239,68,68,0.06);
    border: 1px solid rgba(239,68,68,0.18);
    border-left: 3px solid {sev_color};
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 10px;
">
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
        <span style="color:{sev_color};font-size:14px;">{icon}</span>
        <span style="
            font-size: 13px;
            font-weight: 600;
            color: {COLORS['text_primary']};
            font-family: Inter, sans-serif;
        ">{label}</span>
        <span style="
            margin-left:auto;
            background: {sev_color}22;
            border: 1px solid {sev_color}44;
            color: {sev_color};
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 0.06em;
            padding: 2px 8px;
            border-radius: 10px;
            font-family: Inter, sans-serif;
        ">{sev_upper}</span>
    </div>
    <div style="
        font-size: 13px;
        color: {COLORS['text_secondary']};
        line-height: 1.55;
        font-family: Inter, sans-serif;
    ">{description}</div>
    {cite_html}
</div>
"""


def render_confidence_bar(label: str, level: str) -> str:
    """Horizontal progress bar colored by confidence level.

    Args:
        label: Left-side label text.
        level: "HIGH", "MEDIUM", or "LOW".
    """
    level_upper = (level or "LOW").upper()
    color = {
        "HIGH": COLORS["green"],
        "MEDIUM": COLORS["amber"],
        "LOW": COLORS["red"],
    }.get(level_upper, COLORS["red"])
    pct = {"HIGH": "88%", "MEDIUM": "55%", "LOW": "22%"}.get(level_upper, "22%")

    return f"""
<div style="margin-bottom:10px;">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
        <span style="font-size:12px;color:{COLORS['text_secondary']};font-family:Inter,sans-serif;">{label}</span>
        <span style="font-size:11px;font-weight:600;color:{color};font-family:Inter,sans-serif;">{level_upper}</span>
    </div>
    <div style="height:6px;background:{COLORS['border']};border-radius:3px;overflow:hidden;">
        <div style="
            height:100%;
            width:{pct};
            background:linear-gradient(90deg,{color}88,{color});
            border-radius:3px;
            transition:width 0.4s ease;
        "></div>
    </div>
</div>
"""


def render_agent_progress(agents_status: dict[str, str]) -> str:
    """Vertical agent progress list.

    Args:
        agents_status: Mapping of node_name → status string:
            "done", "running", "waiting", or "error".
            Status may also include " (Xs)" suffix for elapsed time.
    """
    _DISPLAY = {
        "data_fusion": "Data Fusion",
        "rag_citation": "RAG Citation",
        "quant_analysis": "Quant Analysis",
        "risk_scanner": "Risk Scanner",
        "verify": "Verification",
        "report_synthesis": "Report Synthesis",
    }

    rows = []
    for node, raw_status in agents_status.items():
        label = _DISPLAY.get(node, node.replace("_", " ").title())
        # Status may be "done (3.2s)" — split for display
        parts = raw_status.split(" ", 1)
        status_key = parts[0]
        time_str = parts[1] if len(parts) > 1 else ""

        if status_key == "done":
            icon_html = f'<span style="color:{COLORS["green"]};font-size:16px;">✓</span>'
            label_style = f'color:{COLORS["text_secondary"]};'
        elif status_key == "running":
            icon_html = f'<span class="agent-running" style="font-size:16px;">●</span>'
            label_style = f'color:{COLORS["text_primary"]};font-weight:600;'
        elif status_key == "error":
            icon_html = f'<span style="color:{COLORS["red"]};font-size:16px;">✗</span>'
            label_style = f'color:{COLORS["red"]};'
        else:
            icon_html = f'<span style="color:{COLORS["text_muted"]};font-size:16px;">○</span>'
            label_style = f'color:{COLORS["text_muted"]};'

        time_html = (
            f'<span style="font-size:11px;color:{COLORS["text_muted"]};margin-left:auto;'
            f'font-family:Inter,sans-serif;">{time_str}</span>'
            if time_str
            else ""
        )

        rows.append(f"""
<div style="
    display:flex;align-items:center;gap:10px;
    padding:8px 12px;
    border-radius:6px;
    margin-bottom:4px;
    background:{COLORS['bg_elevated']};
">
    {icon_html}
    <span style="font-size:13px;font-family:Inter,sans-serif;{label_style}">{label}</span>
    {time_html}
</div>""")

    return _PULSE_CSS + "\n".join(rows)


def render_divergence(claim: str, actual_data: str, severity: str = "MEDIUM") -> str:
    """Two-column divergence card: management narrative vs quantitative data.

    Args:
        claim: What management stated or implied.
        actual_data: What the numbers actually show.
        severity: "HIGH", "MEDIUM", or "LOW".
    """
    sev_upper = (severity or "MEDIUM").upper()
    sev_color = {
        "HIGH": COLORS["red"],
        "MEDIUM": COLORS["amber"],
        "LOW": COLORS["green"],
    }.get(sev_upper, COLORS["amber"])

    return f"""
<div style="
    background: {COLORS['bg_card']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 10px;
">
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;">
        <div>
            <div style="
                font-size:11px;font-weight:700;letter-spacing:0.06em;
                text-transform:uppercase;color:{COLORS['text_muted']};
                margin-bottom:6px;font-family:Inter,sans-serif;
            ">Management Says</div>
            <div style="
                font-size:13px;color:{COLORS['text_secondary']};
                line-height:1.55;font-family:Inter,sans-serif;
                padding:10px;background:{COLORS['bg_elevated']};
                border-radius:6px;border-left:2px solid {COLORS['border']};
            ">{claim}</div>
        </div>
        <div>
            <div style="
                font-size:11px;font-weight:700;letter-spacing:0.06em;
                text-transform:uppercase;color:{COLORS['text_muted']};
                margin-bottom:6px;font-family:Inter,sans-serif;
            ">Data Shows</div>
            <div style="
                font-size:13px;color:{COLORS['text_secondary']};
                line-height:1.55;font-family:Inter,sans-serif;
                padding:10px;background:{COLORS['bg_elevated']};
                border-radius:6px;border-left:2px solid {sev_color};
            ">{actual_data}</div>
        </div>
    </div>
    <div style="
        margin-top:10px;text-align:right;
        font-size:11px;font-weight:700;color:{sev_color};
        font-family:Inter,sans-serif;letter-spacing:0.06em;
    ">DIVERGENCE · {sev_upper}</div>
</div>
"""


def render_chat_message(role: str, content: str) -> str:
    """Chat bubble: user right-aligned blue, assistant left-aligned dark.

    Args:
        role: "user" or "assistant".
        content: Message text (plain text; newlines become <br>).
    """
    safe_content = content.replace("\n", "<br>")
    if role == "user":
        return f"""
<div style="display:flex;justify-content:flex-end;margin-bottom:10px;">
    <div style="
        background:{COLORS['accent']};
        color:white;
        padding:10px 14px;
        border-radius:14px 14px 4px 14px;
        max-width:75%;
        font-size:14px;
        line-height:1.5;
        font-family:Inter,sans-serif;
    ">{safe_content}</div>
</div>
"""
    else:
        return f"""
<div style="display:flex;justify-content:flex-start;margin-bottom:10px;">
    <div style="
        background:{COLORS['bg_elevated']};
        border:1px solid {COLORS['border']};
        color:{COLORS['text_secondary']};
        padding:10px 14px;
        border-radius:14px 14px 14px 4px;
        max-width:80%;
        font-size:14px;
        line-height:1.5;
        font-family:Inter,sans-serif;
    ">{safe_content}</div>
</div>
"""


def render_footer() -> str:
    """Disclaimer, data source attribution, and GitHub link."""
    return f"""
<div style="
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid {COLORS['border']};
    text-align: center;
">
    <div style="
        font-size: 11px;
        color: {COLORS['text_muted']};
        font-family: Inter, sans-serif;
        line-height: 1.7;
        max-width: 700px;
        margin: 0 auto;
    ">
        <strong style="color:{COLORS['text_secondary']};">NOT FINANCIAL ADVICE.</strong>
        AlphaLens is an AI research tool for educational purposes only.
        All analysis is generated by large language models and may contain errors.
        Do not make investment decisions based solely on this output.<br><br>
        Data sources: SEC EDGAR · Alpha Vantage · FRED (Federal Reserve) · Yahoo Finance ·
        <a href="https://github.com" target="_blank"
           style="color:{COLORS['accent']};text-decoration:none;">GitHub</a>
    </div>
</div>
"""
