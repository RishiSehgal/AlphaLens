# AlphaLens — CLAUDE.md (Master Architecture Document)

## What This Project Is
AlphaLens is a multi-agent AI equity research platform that compresses 8+ hours of professional equity research into 90 seconds at zero cost. Each agent maps to a real analyst role on a traditional research team. This is a one-person research desk — proof that one engineer with AI agents can replicate the output of an entire equity research team.

## Positioning
- **Target user**: Retail investors who own stocks but don't read 10-Q filings
- **Demo audience**: Hiring managers for AI Engineer roles + MS capstone evaluators at Northeastern
- **Interview line**: "Each agent maps to a real analyst role. I replaced the team with a LangGraph state machine."
- **Assignment**: Generative AI course capstone — must implement RAG, Prompt Engineering, Multimodal Integration, and Evaluation Metrics

## Tech Stack (LOCKED)
- **Orchestration**: LangGraph (multi-agent state machine with parallel execution)
- **LLM**: Google Gemini 2.0 Flash (free tier: 15 RPM, 1M tokens/day)
- **Frontend**: Streamlit with custom embedded HTML/CSS components (NOT default Streamlit look)
- **Vector DB**: ChromaDB (local, no cost)
- **Data Sources** (all free):
  - SEC EDGAR: 10-K/10-Q filings (10 req/sec, no key needed)
  - Alpha Vantage: Market data + financials (25 req/day, free key)
  - FRED: Macro economic data (free key, public domain U.S. government data)
  - yfinance: Stock prices + basic financials (no key)
- **Language**: Python 3.11+
- **Key Libraries**: langchain, langgraph, chromadb, streamlit, plotly, pandas, requests, google-generativeai, yfinance, fredapi, beautifulsoup4

## Project Structure
```
AlphaLens/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── .env / .env.example
├── .gitignore
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml.example
├── app.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── state.py
│   ├── graph.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── data_fusion.py
│   │   ├── rag_citation.py
│   │   ├── quant_analysis.py
│   │   ├── risk_scanner.py
│   │   ├── verification.py
│   │   └── report_synthesis.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── edgar_client.py
│   │   ├── market_data.py
│   │   ├── fred_client.py
│   │   └── filing_parser.py
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── chunker.py
│   │   ├── embeddings.py
│   │   └── retriever.py
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── test_cases.py
│   │   └── runner.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── rate_limiter.py
│   │   ├── cost_tracker.py
│   │   ├── cache.py
│   │   └── error_handler.py
│   └── ui/
│       ├── __init__.py
│       ├── components.py
│       ├── report_view.py
│       ├── sidebar.py
│       ├── charts.py
│       └── chat.py
├── tests/
│   ├── test_data_fusion.py
│   ├── test_rag.py
│   ├── test_verification.py
│   └── test_eval.py
├── docs/
│   ├── architecture.md
│   └── eval_results.md
├── examples/
│   └── sample_output_NVDA.json
└── web/
    └── index.html
```

---

## The 6 Agents

### Agent 1: Data Fusion Agent (replaces: Junior Analyst)
- **Input**: Ticker symbol
- **Output**: Structured financial data dict
- **Sources**: SEC EDGAR, yfinance, Alpha Vantage, FRED
- **Behavior**: Parallel API calls. If Alpha Vantage fails, fall back to yfinance with degraded confidence. Never crashes — returns partial data with sources_status dict.

### Agent 2: RAG Citation Agent (replaces: Research Associate)
- **Input**: Ticker + filing URLs from Agent 1
- **Output**: Retrieved chunks with section metadata and citation info
- **Process**: Download filing → parse sections (MD&A, Risk Factors, Financial Statements, Notes) → chunk (500 tokens, 100 overlap) → embed with Gemini → store ChromaDB → retrieve top-k per query dimension
- **Citation format**: {source_file, section_name, page_number}

### Agent 3: Quant Analysis Agent (replaces: Quantitative Analyst)
- **Input**: Financial data from Agent 1
- **Output**: DCF range, technicals, earnings surprise
- **DCF**: 3-stage model, bear/base/bull output, sensitivity table. ALWAYS MEDIUM confidence.
- **Technicals**: RSI (14-day), MACD (12/26/9)
- **Earnings Surprise**: actual EPS vs consensus, % surprise

### Agent 4: Risk Scanner Agent (replaces: Risk/Compliance Officer)
- **Input**: RAG chunks from Agent 2
- **Output**: Red flags with severity and citations
- **Two-stage**: keyword/pattern match FIRST → LLM classifier to confirm severity
- **Patterns**: going concern, auditor change, revenue recognition change, related-party, material weakness, litigation, insider selling, covenant violations, customer concentration

### Agent 5: Verification Agent (replaces: Senior Analyst) — CROWN JEWEL
- **Input**: ALL outputs from Agents 1-4
- **Output**: Divergence list + per-section confidence scores
- **Process**: Extract quantitative claims → extract narrative claims → cross-reference → flag divergences with evidence → assign confidence

### Agent 6: Report Synthesis Agent (replaces: Publishing Editor)
- **Input**: ALL outputs from Agents 1-5
- **Output**: Structured report with sections, citations, confidence scores
- **Sections**: Executive Summary, Financial Health, Risk Flags, Valuation, Verification Verdict

---

## LangGraph State Machine

```python
class AlphaLensState(TypedDict):
    ticker: str
    financial_data: dict
    rag_chunks: list[dict]
    quant_results: dict
    risk_flags: list[dict]
    verification: dict
    report: dict
    metadata: dict
    error_log: list[str]
    chat_history: list[dict]
```

### Execution Flow:
```
START → data_fusion ──┐
                       ├── parallel ──→ rag_citation ──→ risk_scanner ──┐
                       └── parallel ──→ quant_analysis ─────────────────┤
                                                                         ├──→ verification → report_synthesis → END
```

---

## UI/UX Design System

### CRITICAL: DO NOT USE DEFAULT STREAMLIT STYLING
All visible UI must be rendered as custom HTML/CSS via st.markdown(html, unsafe_allow_html=True) and st.components.v1.html(). The app must look like a real financial product — dark theme, data-dense, professional.

### Color System (Dark Theme)
```python
COLORS = {
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
```

### .streamlit/config.toml
```toml
[theme]
primaryColor = "#3B82F6"
backgroundColor = "#0A0A0F"
secondaryBackgroundColor = "#12121A"
textColor = "#F1F5F9"
font = "sans serif"
[server]
headless = true
[browser]
gatherUsageStats = false
```

### Custom HTML Components to Build (src/ui/components.py)

1. **render_header()** — Gradient top border (blue→purple, 2px), "AlphaLens" 28px bold, subtitle 14px muted
2. **render_metric_card(label, value, subtitle, color)** — Dark card, muted uppercase label 12px, large value 24px, colored left border 3px
3. **render_report_section(title, confidence, body_html, citations, source_chunks)** — Card with confidence badge pill (HIGH=green, MEDIUM=amber, LOW=red), body text with inline citation links, CSS-only expandable "Show source chunks" section
4. **render_risk_flag(flag_type, severity, description, citation)** — Red-tinted card (rgba(239,68,68,0.08)), severity left-border, icon, description, citation link
5. **render_confidence_bar(label, level)** — 6px height bar, label left, fill colored by level, smooth gradient
6. **render_agent_progress(agents_status)** — Vertical list: ✓ green (done), ● blue pulse animation (running), ○ gray (waiting), time right-aligned
7. **render_divergence(claim, actual_data, severity)** — Two-column card: "Management says" left vs "Data shows" right, severity badge bottom
8. **render_chat_message(role, content)** — User bubbles right-aligned blue, assistant bubbles left-aligned dark
9. **render_footer()** — Disclaimer + data source attribution + GitHub link

### Agent Progress Animation CSS
```css
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}
.agent-running { animation: pulse 1.5s ease-in-out infinite; color: #3B82F6; }
```

### Plotly Chart Template
```python
PLOTLY_TEMPLATE = {
    "paper_bgcolor": "#12121A",
    "plot_bgcolor": "#12121A",
    "font": {"color": "#94A3B8", "family": "Inter, sans-serif", "size": 12},
    "xaxis": {"gridcolor": "#1E293B", "zerolinecolor": "#1E293B"},
    "yaxis": {"gridcolor": "#1E293B", "zerolinecolor": "#1E293B"},
    "colorway": ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"],
}
```

### Charts to Build
1. **DCF Sensitivity Heatmap** — growth rate × discount rate → fair value, green-amber-red scale
2. **RSI Gauge** — semicircular, zones: oversold(0-30 green), neutral(30-70 blue), overbought(70-100 red)
3. **Price + MACD Chart** — candlestick/line with MACD subplot, 6-month yfinance data
4. **Earnings Surprise Bar** — horizontal, actual vs estimate, beat=green miss=red

### Layout Code Pattern
```python
# app.py structure
st.set_page_config(page_title="AlphaLens", layout="wide", initial_sidebar_state="expanded")

# Inject global CSS to override ALL default Streamlit styling
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# Header
render_header()

# Ticker input row
col_input, col_btn = st.columns([4, 1])
with col_input:
    ticker = st.text_input("", placeholder="Enter ticker (e.g., NVDA)", label_visibility="collapsed")
with col_btn:
    analyze = st.button("Analyze", type="primary", use_container_width=True)

# Main content + sidebar
col_main, col_side = st.columns([7, 3])

with col_main:
    # Agent progress (during execution)
    # Report sections (after completion)
    # Follow-up chat

with col_side:
    # Metric cards (time, tokens, cost)
    # Sources breakdown
    # Confidence bars
    # Agent latency
    # Cache status
```

### Global CSS Override (inject at top of app.py)
```python
GLOBAL_CSS = """
<style>
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Override Streamlit containers */
    .stApp {
        background-color: #0A0A0F;
    }
    .block-container {
        padding: 1rem 2rem;
        max-width: 1400px;
    }

    /* Style all Streamlit text inputs */
    .stTextInput input {
        background-color: #1E1E2A;
        border: 1px solid #1E293B;
        border-radius: 8px;
        color: #F1F5F9;
        padding: 12px 16px;
        font-size: 16px;
    }
    .stTextInput input:focus {
        border-color: #3B82F6;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
    }
    .stTextInput input::placeholder {
        color: #64748B;
    }

    /* Style primary button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #3B82F6, #8B5CF6);
        border: none;
        border-radius: 8px;
        color: white;
        font-weight: 600;
        padding: 12px 24px;
        font-size: 16px;
        transition: opacity 0.2s;
    }
    .stButton > button[kind="primary"]:hover {
        opacity: 0.9;
    }

    /* Style expanders */
    .streamlit-expanderHeader {
        background-color: #12121A;
        border: 1px solid #1E293B;
        border-radius: 8px;
        color: #94A3B8;
        font-size: 13px;
    }

    /* Style chat input */
    .stChatInput textarea {
        background-color: #1E1E2A;
        border: 1px solid #1E293B;
        color: #F1F5F9;
    }

    /* Remove default padding on sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0A0A0F;
        border-right: 1px solid #1E293B;
    }

    /* Plotly chart containers */
    .stPlotlyChart {
        border-radius: 8px;
        overflow: hidden;
    }

    /* Scrollbar styling */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0A0A0F; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
</style>
"""
```

---

## Evaluation Framework

### Metrics:
1. Retrieval Precision@5 ≥ 0.60
2. Retrieval Recall@5 ≥ 0.80
3. Faithfulness/Groundedness ≥ 0.90
4. Numerical Accuracy ≥ 0.95
5. Earnings Surprise Accuracy = 1.00

### Golden test set: AAPL, NVDA, MSFT

---

## Graceful Degradation
| Failure | Behavior | UI Message |
|---------|----------|------------|
| Alpha Vantage limit | yfinance-only | "Limited data — confidence adjusted" |
| EDGAR not found | Skip RAG | "Filing unavailable — market data only" |
| Gemini rate limit | Exponential backoff | "Processing..." |
| Agent crash | Continue pipeline | "Section unavailable" |

## Code Conventions
- Type hints on all functions
- Google-style docstrings
- logging module (no print)
- try/except on all API calls
- Rate limiting on all external calls
- Environment variables for keys
- @st.cache_data / @st.cache_resource in Streamlit

## Ethical Considerations
- NOT financial advice — disclaimer everywhere
- All data from public sources, compliant for educational use
- Confidence scores communicate uncertainty
- No user data collected
- DCF capped at MEDIUM confidence

## CLAUDE.md Maintenance Rule
After completing each major milestone (data layer, agents, frontend, eval), Claude Code should update this file with:
- What was built and what actually works
- Any deviations from the original plan
- Current bugs or known issues
- What to build next
To trigger this, prompt: "Update CLAUDE.md with current project state."