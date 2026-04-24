"""LangGraph orchestrator for the AlphaLens multi-agent pipeline.

Execution flow:
    START → data_fusion ─┬─→ rag_citation ─→ risk_scanner ─┐
                         └─→ quant_analysis ────────────────┤
                                                             ↓
                                                       verification
                                                             ↓
                                                     report_synthesis
                                                             ↓
                                                            END
"""

import logging
import time
from typing import Any, Generator, Optional

from langgraph.graph import END, START, StateGraph

from src.agents.data_fusion import data_fusion_agent
from src.agents.quant_analysis import quant_analysis_agent
from src.agents.rag_citation import rag_citation_agent
from src.agents.report_synthesis import report_synthesis_agent
from src.agents.risk_scanner import risk_scanner_agent
from src.agents.verification import verification_agent
from src.state import AlphaLensState

logger = logging.getLogger(__name__)

# Node name constants — referenced by the UI for progress tracking
NODE_DATA_FUSION = "data_fusion"
NODE_RAG_CITATION = "rag_citation"
NODE_QUANT_ANALYSIS = "quant_analysis"
NODE_RISK_SCANNER = "risk_scanner"
NODE_VERIFICATION = "verify"
NODE_REPORT_SYNTHESIS = "report_synthesis"

_ALL_NODES = [
    NODE_DATA_FUSION,
    NODE_RAG_CITATION,
    NODE_QUANT_ANALYSIS,
    NODE_RISK_SCANNER,
    NODE_VERIFICATION,
    NODE_REPORT_SYNTHESIS,
]


def _build_graph() -> Any:
    """Construct and compile the LangGraph StateGraph.

    Returns:
        Compiled LangGraph runnable (CompiledStateGraph).
    """
    workflow = StateGraph(AlphaLensState)

    # ── Register nodes ─────────────────────────────────────────────────────────
    workflow.add_node(NODE_DATA_FUSION, data_fusion_agent)
    workflow.add_node(NODE_RAG_CITATION, rag_citation_agent)
    workflow.add_node(NODE_QUANT_ANALYSIS, quant_analysis_agent)
    workflow.add_node(NODE_RISK_SCANNER, risk_scanner_agent)
    workflow.add_node(NODE_VERIFICATION, verification_agent)
    workflow.add_node(NODE_REPORT_SYNTHESIS, report_synthesis_agent)

    # ── Wire edges ─────────────────────────────────────────────────────────────
    # Sequential start
    workflow.add_edge(START, NODE_DATA_FUSION)

    # Fan-out after data_fusion — both run in parallel
    workflow.add_edge(NODE_DATA_FUSION, NODE_RAG_CITATION)
    workflow.add_edge(NODE_DATA_FUSION, NODE_QUANT_ANALYSIS)

    # rag_citation feeds risk_scanner
    workflow.add_edge(NODE_RAG_CITATION, NODE_RISK_SCANNER)

    # Fan-in — verification only starts after BOTH risk_scanner AND quant_analysis finish
    workflow.add_edge([NODE_RISK_SCANNER, NODE_QUANT_ANALYSIS], NODE_VERIFICATION)

    # Sequential tail
    workflow.add_edge(NODE_VERIFICATION, NODE_REPORT_SYNTHESIS)
    workflow.add_edge(NODE_REPORT_SYNTHESIS, END)

    return workflow.compile()


# Compile once at module import — re-used across calls
_COMPILED_GRAPH = _build_graph()


def _initial_state(ticker: str) -> AlphaLensState:
    """Build the initial AlphaLensState for a fresh pipeline run.

    Args:
        ticker: Stock ticker symbol (will be upper-cased).

    Returns:
        Fully-initialised AlphaLensState TypedDict.
    """
    return AlphaLensState(
        ticker=ticker.upper().strip(),
        financial_data={},
        rag_chunks=[],
        quant_results={},
        risk_flags=[],
        verification={},
        report={},
        metadata={
            "agent_latencies": {},
            "sources_status": {},
            "pipeline_start": time.time(),
        },
        error_log=[],
        chat_history=[],
    )


def run_pipeline(ticker: str) -> AlphaLensState:
    """Run the full AlphaLens pipeline synchronously and return the final state.

    Args:
        ticker: Stock ticker symbol (e.g. "AAPL").

    Returns:
        Final AlphaLensState with all agents' outputs populated.
        Never raises — errors are recorded in ``state["error_log"]``.
    """
    logger.info("Starting AlphaLens pipeline for %s", ticker.upper())
    state = _initial_state(ticker)

    try:
        result: AlphaLensState = _COMPILED_GRAPH.invoke(state)
    except Exception as exc:
        logger.error("Pipeline crashed for %s: %s", ticker, exc, exc_info=True)
        state["error_log"].append(f"PIPELINE_CRASH: {exc}")
        state["report"] = {
            "ticker": ticker.upper(),
            "sections": {},
            "overall_confidence": "LOW",
            "disclaimer": "Pipeline encountered an error. Results unavailable.",
        }
        return state

    elapsed = time.time() - result["metadata"].get("pipeline_start", time.time())
    result["metadata"]["pipeline_total_seconds"] = round(elapsed, 2)
    logger.info(
        "Pipeline complete for %s in %.1fs — %d errors",
        ticker.upper(), elapsed, len(result["error_log"]),
    )
    return result


def stream_pipeline(ticker: str) -> Generator[dict, None, None]:
    """Stream node-completion events from the pipeline for live UI updates.

    Each yielded dict has keys:
        ``node`` (str): Node name that just finished,
        ``state_snapshot`` (AlphaLensState): State after that node,
        ``elapsed`` (float): Wall-clock seconds since pipeline start.

    Args:
        ticker: Stock ticker symbol.

    Yields:
        Progress event dicts as each agent completes.
    """
    state = _initial_state(ticker)
    pipeline_start = state["metadata"]["pipeline_start"]

    try:
        for event in _COMPILED_GRAPH.stream(state, stream_mode="updates"):
            for node_name, node_output in event.items():
                elapsed = round(time.time() - pipeline_start, 2)
                logger.debug("Node '%s' completed in %.1fs", node_name, elapsed)
                yield {
                    "node": node_name,
                    "partial_state": node_output,
                    "elapsed": elapsed,
                }
    except Exception as exc:
        logger.error("stream_pipeline crashed for %s: %s", ticker, exc, exc_info=True)
        yield {
            "node": "error",
            "partial_state": {"error_log": [f"PIPELINE_CRASH: {exc}"]},
            "elapsed": round(time.time() - pipeline_start, 2),
        }


def get_node_order() -> list[str]:
    """Return the ordered list of pipeline node names for UI progress display.

    Returns:
        List of node names in expected execution order.
    """
    return _ALL_NODES.copy()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    import sys

    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(f"\nRunning AlphaLens pipeline for {ticker}…\n")

    final_state = run_pipeline(ticker)

    print(f"\n{'='*60}")
    print(f"Pipeline complete — {ticker}")
    print(f"{'='*60}")

    meta = final_state.get("metadata", {})
    print(f"Total time:   {meta.get('pipeline_total_seconds', '?')}s")
    print(f"Errors:       {len(final_state.get('error_log', []))}")

    latencies = meta.get("agent_latencies", {})
    print("\nAgent latencies:")
    for node in get_node_order():
        t = latencies.get(node, "—")
        print(f"  {node:25s} {t}s" if isinstance(t, (int, float)) else f"  {node:25s} {t}")

    report = final_state.get("report", {})
    print(f"\nReport confidence: {report.get('overall_confidence', '?')}")
    print(f"Risk flags:        {len(final_state.get('risk_flags', []))}")
    print(f"RAG chunks:        {len(final_state.get('rag_chunks', []))}")

    sections = report.get("sections", {})
    if sections:
        print("\nReport sections generated:")
        for key, section in sections.items():
            conf = section.get("confidence", "?")
            preview = (section.get("content", "") or "")[:80].replace("\n", " ")
            print(f"  [{conf:6s}] {key}: {preview}…")

    if final_state.get("error_log"):
        print("\nErrors encountered:")
        for err in final_state["error_log"]:
            print(f"  • {err}")
