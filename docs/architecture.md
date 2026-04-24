# AlphaLens — Technical Architecture

## Overview

AlphaLens is a LangGraph-orchestrated multi-agent pipeline that produces equity research reports from public data. This document covers the data flow, agent internals, error handling strategy, caching design, and extension points.

---

## LangGraph State Machine

### State Definition (`src/state.py`)

```python
class AlphaLensState(TypedDict):
    ticker: str
    financial_data: dict        # Agent 1 output
    rag_chunks: list[dict]      # Agent 2 output
    quant_results: dict         # Agent 3 output
    risk_flags: list[dict]      # Agent 4 output
    verification: dict          # Agent 5 output
    report: dict                # Agent 6 output
    metadata: Annotated[dict, _merge_metadata]   # reducer: deep-merge
    error_log: Annotated[list[str], operator.add] # reducer: append
    chat_history: list[dict]
```

**Parallel-safety**: `metadata` and `error_log` use LangGraph `Annotated` reducers. During the parallel branch (rag_citation ∥ quant_analysis), both agents write to `metadata["agent_latencies"]` without clobbering each other.

### Execution Graph

```
START
  │
  ▼
data_fusion                    ← sequential: fetches all raw data
  │
  ├──────────────────┐
  ▼                  ▼
rag_citation      quant_analysis   ← parallel: independent computation
  │                  │
  ▼                  │
risk_scanner         │
  │                  │
  └──────────┬───────┘
             ▼
         verification           ← fan-in: waits for BOTH branches
             │
             ▼
      report_synthesis
             │
             ▼
            END
```

`add_edge([NODE_RISK_SCANNER, NODE_QUANT_ANALYSIS], NODE_VERIFICATION)` is the LangGraph list-syntax fan-in that implements the barrier.

---

## Agent Internals

### Agent 1: Data Fusion

**Inputs**: ticker string  
**Outputs**: `financial_data` dict, `metadata.agent_latencies.data_fusion`, `metadata.sources_status`

Three API clients run concurrently via `ThreadPoolExecutor(max_workers=3)`:

| Client | Data | Rate limit |
|--------|------|-----------|
| `MarketDataClient` | Income statement, balance sheet, cash flow, current price | 25 req/day (AV) → yfinance fallback |
| `EdgarClient` | Filing URLs (10-K, 10-Q), CIK lookup | 10 req/sec |
| `FredClient` | FEDFUNDS, GDP, CPIAUCSL, UNRATE | Unlimited |

Output normalization extracts ~25 canonical fields (revenue, net_income, gross_margin, eps, dcf inputs, etc.) into a flat dict regardless of which data source provided them.

### Agent 2: RAG Citation

**Inputs**: `financial_data.filing_urls`  
**Outputs**: `rag_chunks` list, `metadata.rag_available`, `metadata.rag_filing_date`

```
Download 10-K HTML (EDGAR)
         │
         ▼
  parse_filing() → [{section_name, text_content, estimated_page}]
  (3-strategy: heading tags → TOC anchors → text scan)
         │
         ▼
  chunk_sections() → [{text, metadata}]
  (RecursiveCharacterTextSplitter, 500 tok, 100 overlap, cl100k_base)
         │
         ▼
  collection_count(ticker) > 0?  ──yes──→ skip embed (idempotent)
         │ no
         ▼
  embed_chunks() → Gemini gemini-embedding-001 (3072-dim, RETRIEVAL_DOCUMENT)
         │
         ▼
  ChromaDB upsert (cosine, persistent)
         │
         ▼
  retrieve_relevant() × 3 queries → deduplicated top-k chunks
```

Embedding idempotency: on repeated runs for the same ticker, ChromaDB already has the chunks embedded. The agent checks `collection_count > 0` before calling Gemini, saving embedding quota.

### Agent 3: Quant Analysis

**Inputs**: `financial_data`  
**Outputs**: `quant_results.{dcf, technicals, earnings_surprise}`

**DCF Model (3-stage)**:
- WACC = `(FEDFUNDS_rate / 100) + beta × 0.055`, clamped to [0.06, 0.18]
- Stage 1 (years 1–5): constant growth `growth_stage1` from revenue trend
- Stage 2 (years 6–10): linear interpolation → terminal growth rate
- Stage 3: Gordon Growth Model terminal value
- Three scenarios: bear (g1 × 0.6), base (g1), bull (g1 × 1.4)
- Confidence: always `MEDIUM` per design spec (no DCF is HIGH confidence)

**Technicals**:
- RSI(14): 14-day relative strength from yfinance 6-month history
- MACD(12/26/9): EMA crossover signal

**Earnings Surprise**:
- `yfinance.earnings_history` → most recent quarter actual vs estimate
- `surprise_pct = (actual - estimate) / abs(estimate) × 100`

### Agent 4: Risk Scanner

**Inputs**: `rag_chunks`  
**Outputs**: `risk_flags` list of `{type, severity, description, citation, confirmed}`

Two-stage detection:

**Stage 1 — Keyword/Regex scan** across all RAG chunks:
```
9 risk categories:
  going_concern       → ["going concern", "substantial doubt"] → HIGH
  material_weakness   → ["material weakness", "significant deficiency"] → HIGH
  auditor_change      → ["deloitte", "pwc", "kpmg", "ernst"] + ["resigned", "appointed"] → HIGH
  revenue_recognition → ["revenue recognition", "606", "performance obligation"] → MEDIUM
  related_party       → ["related party", "transactions with officers"] → MEDIUM
  litigation          → ["lawsuit", "legal proceedings", "regulatory investigation"] → MEDIUM
  insider_selling     → ["sale of shares by executive", "10b5-1"] → MEDIUM
  covenant_violation  → ["covenant", "waiver", "default"] → HIGH
  customer_concentration → ["significant customer", "accounts for more than"] → MEDIUM
```

**Stage 2 — Gemini LLM confirmation** (capped at 10 chunks to save quota):
- Sends chunk + pattern context to Gemini
- Confirms severity or downgrades to LOW if pattern is benign
- 65-second retry on 429

### Agent 5: Verification (Crown Jewel)

**Inputs**: All of `financial_data`, `rag_chunks`, `quant_results`, `risk_flags`  
**Outputs**: `verification.{divergences, confidence_scores, verification_verdict}`

```
Build data summary (<400 tokens):
  revenue, margins, FCF, DCF range, RSI, risk flag count

Select 6 RAG excerpts:
  MD&A chunks first (management narrative), then Risk Factors

Single Gemini call with structured prompt:
  "Compare management's narrative claims against these financial metrics.
   List divergences. Assign confidence scores per section."

Post-processing:
  - Force valuation confidence = MEDIUM (DCF invariant)
  - Parse divergences into [{field, management_claim, data_evidence, severity}]
```

### Agent 6: Report Synthesis

**Inputs**: All previous agent outputs  
**Outputs**: `report.{sections, overall_confidence, disclaimer}`

5 Gemini calls (one per section) with 4-second inter-call throttle:

| Section | Input context | Confidence source |
|---------|--------------|------------------|
| executive_summary | financial_data + verification_verdict | min(all) |
| financial_health | financial_data + rag_chunks(financial) | numerical_accuracy proxy |
| valuation | quant_results + verification | always MEDIUM |
| risk_flags | risk_flags + rag_chunks(risk) | flag severity distribution |
| verification_verdict | verification.divergences | verification agent output |

`overall_confidence = min(all section confidences)` using HIGH > MEDIUM > LOW ordering.

---

## Error Handling Strategy

### Hierarchy

```
Level 1: try/except in each API call → log + return None/empty
Level 2: agent-level try/except → log + return partial state
Level 3: graph.py run_pipeline() → catch pipeline crash → return error state with disclaimer
Level 4: Streamlit app → display error section, never crash
```

### Graceful Degradation Table

| Component | Failure | Fallback |
|-----------|---------|----------|
| Alpha Vantage | Rate limit (25/day) | yfinance full substitution |
| Alpha Vantage | 5xx error | yfinance partial substitution |
| EDGAR download | Network error | Skip RAG; rag_available=False |
| EDGAR filing | Filing not found | Skip RAG |
| Gemini | 429 rate limit | 65-second sleep, retry ×4 |
| Gemini | Other error | 2^attempt second backoff, retry ×4 |
| Gemini | All retries fail | Template-based fallback report |
| ChromaDB | Write error | Skip embed; retrieve returns [] |
| Any agent | Uncaught exception | error_log append; pipeline continues |

---

## Caching Strategy

### 24-Hour State Cache (`src/utils/cache.py`)

Key: `{TICKER}_{YYYY-MM-DD}` — same ticker, same calendar day → cache hit.

```
get_cached_state(ticker)
  │
  ├── cache miss → run_pipeline() → set_cached_state() → return state
  └── cache hit (< 24h) → return cached state immediately
```

Stored at `.alphalens_cache.json` (project root, gitignored).

**Why 24h TTL**: Financial data updates daily (new price, possible 8-K). Within a trading day, data doesn't change meaningfully. The cache eliminates all API calls and LLM inference on repeat visits.

### ChromaDB Embedding Cache

Embedding idempotency (separate from state cache):
- `FilingRetriever.collection_count(ticker)` returns 0 on first run
- Embeddings are stored in `.chroma/` (persistent directory)
- On subsequent runs: skip embed, query directly from ChromaDB
- `upsert` prevents duplicates even if the check is bypassed

---

## RAG Pipeline Details

### Chunking Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Chunk size | 500 tokens | Fits in Gemini context window for single-chunk prompts |
| Overlap | 100 tokens | Prevents context loss at section boundaries |
| Tokenizer | cl100k_base (tiktoken) | Compatible with GPT-4 token counts; ~4 chars/token |
| Splitter | RecursiveCharacterTextSplitter | Respects paragraph → sentence → word hierarchy |

### Embedding Model

`gemini-embedding-001`: 3072-dimensional dense vectors, optimized for semantic similarity.

Two task types:
- `RETRIEVAL_DOCUMENT`: for storing filing chunks (indexed vectors)
- `RETRIEVAL_QUERY`: for embedding user/agent queries (query-optimized)

These are different embedding spaces — mixing them degrades retrieval quality.

### Retrieval Queries

Three queries per run, deduplicated by `chunk_index`:
1. `"risk factors financial risks regulatory compliance"` — for Risk Scanner
2. `"revenue growth operating margins business performance"` — for Report Synthesis
3. `"management discussion outlook future guidance"` — for Verification

ChromaDB cosine similarity search; optional `where` filter for section-scoped retrieval.

---

## Rate Limiting

Token bucket implementation (`src/utils/rate_limiter.py`):

| API | Bucket Rate | Capacity | Notes |
|-----|------------|---------|-------|
| Gemini | 0.25 req/s (15/min) | 15 | Free tier hard limit |
| EDGAR | 8 req/s | 10 | Polite limit (official: 10/s) |
| Alpha Vantage | 0.25 req/s (1/4s) | 1 | 25/day → space them out |
| FRED | 5 req/s | 10 | Generous official limits |
| yfinance | 2 req/s | 5 | Courtesy; no hard limit |

All buckets are module-level singletons — shared across concurrent calls.

---

## Evaluation Framework

### Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| Precision@k | `relevant_in_top_k / k` | ≥ 0.60 |
| Recall@k | `sections_covered / required_sections` | ≥ 0.80 |
| Faithfulness | `grounded_claims / total_claims` | ≥ 0.90 |
| Numerical Accuracy | `within_tolerance / checked_fields` | ≥ 0.95 |
| Earnings Acc. | `direction_correct` (binary) | 1.00 |

### Golden Dataset

AAPL, NVDA, MSFT — chosen for:
- High filing quality and completeness on EDGAR
- Diverse sector/growth profile
- Well-documented consensus estimates for EPS validation
- Large enough filing volume to test chunking and retrieval

---

## Deployment

### Local Development

```bash
streamlit run app.py
```

### Streamlit Cloud

Set secrets in the Streamlit Cloud dashboard (or `secrets.toml`):
```toml
GOOGLE_API_KEY = "..."
ALPHA_VANTAGE_API_KEY = "..."
FRED_API_KEY = "..."
```

`app.py` reads from `st.secrets` first, then falls back to environment variables.

ChromaDB persistence: on Streamlit Cloud, `.chroma/` is ephemeral (resets on redeploy). Embeddings are regenerated on each cold start — ~30s overhead for the first run per ticker.

---

## Security Notes

- All API keys are loaded from environment variables / `st.secrets` — never hardcoded
- `.env` and `secrets.toml` are gitignored
- No user data is collected, logged, or transmitted
- All data fetched is from public APIs under their respective terms of service
- The EDGAR User-Agent header identifies the application per SEC requirements
