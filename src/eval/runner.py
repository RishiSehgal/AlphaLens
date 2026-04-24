"""Evaluation runner — runs the full AlphaLens pipeline on golden tickers,
computes all metrics, and saves a markdown table to docs/eval_results.md.

Usage:
    python -m src.eval.runner
    python -m src.eval.runner --tickers AAPL NVDA
    python -m src.eval.runner --k 10
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

from src.eval.metrics import compute_all_metrics
from src.eval.test_cases import GOLDEN_CASES, GoldenCase
from src.graph import run_pipeline

logger = logging.getLogger(__name__)

DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"
EVAL_RESULTS_PATH = DOCS_DIR / "eval_results.md"

# Target thresholds from CLAUDE.md
THRESHOLDS: dict[str, float] = {
    "retrieval_precision_at_k": 0.60,
    "retrieval_recall_at_k": 0.80,
    "faithfulness_score": 0.90,
    "numerical_accuracy": 0.95,
    "earnings_surprise_accuracy": 1.00,
}

METRIC_DISPLAY: dict[str, str] = {
    "retrieval_precision_at_k": "Precision@k",
    "retrieval_recall_at_k": "Recall@k",
    "faithfulness_score": "Faithfulness",
    "numerical_accuracy": "Numerical Acc.",
    "earnings_surprise_accuracy": "Earnings Acc.",
}


def _pass_fail(score: float, threshold: float) -> str:
    return "✅" if score >= threshold else "⚠️"


def run_evaluation(
    tickers: list[str] | None = None,
    k: int = 5,
    save_states: bool = False,
) -> dict[str, dict[str, float]]:
    """Run the pipeline and compute metrics for each golden test case.

    Args:
        tickers: Subset of ticker symbols to evaluate. If None, all golden cases are run.
        k: Top-k chunks for retrieval metrics.
        save_states: If True, save full pipeline state to docs/<ticker>_state.json.

    Returns:
        Nested dict: ticker → metric_name → score.
    """
    golden_map: dict[str, GoldenCase] = {g["ticker"]: g for g in GOLDEN_CASES}
    target_tickers = tickers or list(golden_map.keys())

    all_results: dict[str, dict[str, float]] = {}

    for ticker in target_tickers:
        golden = golden_map.get(ticker)
        if not golden:
            logger.warning("No golden case for %s — skipping", ticker)
            continue

        logger.info("Evaluating %s …", ticker)
        t0 = time.time()

        state = run_pipeline(ticker)
        elapsed = round(time.time() - t0, 1)

        if state.get("error_log"):
            logger.warning("%s pipeline errors: %s", ticker, state["error_log"][:3])

        metrics = compute_all_metrics(state, golden, k=k)
        metrics["pipeline_seconds"] = elapsed
        all_results[ticker] = metrics

        logger.info(
            "%s metrics: P@k=%.2f R@k=%.2f Faith=%.2f NumAcc=%.2f EPS=%.2f  (%.1fs)",
            ticker,
            metrics["retrieval_precision_at_k"],
            metrics["retrieval_recall_at_k"],
            metrics["faithfulness_score"],
            metrics["numerical_accuracy"],
            metrics["earnings_surprise_accuracy"],
            elapsed,
        )

        if save_states:
            state_path = DOCS_DIR / f"{ticker.lower()}_state.json"
            try:
                with open(state_path, "w") as f:
                    json.dump(state, f, indent=2, default=str)
                logger.info("Saved state to %s", state_path)
            except Exception as exc:
                logger.warning("Could not save state for %s: %s", ticker, exc)

    return all_results


def _markdown_table(results: dict[str, dict[str, float]], k: int) -> str:
    """Render evaluation results as a Markdown table string."""
    metric_keys = list(METRIC_DISPLAY.keys())
    col_names = ["Ticker"] + [METRIC_DISPLAY[m] for m in metric_keys] + ["Time (s)", "Pass/Fail"]
    sep = "|".join(["---"] * len(col_names))
    header = "| " + " | ".join(col_names) + " |"
    divider = "| " + sep + " |"

    rows = []
    for ticker, metrics in results.items():
        scores = [f"{metrics.get(m, 0.0):.3f}" for m in metric_keys]
        elapsed = f"{metrics.get('pipeline_seconds', 0):.1f}"
        passed_all = all(
            metrics.get(m, 0.0) >= THRESHOLDS[m] for m in metric_keys
        )
        pf = "✅ Pass" if passed_all else "⚠️ Review"
        rows.append("| " + " | ".join([ticker] + scores + [elapsed, pf]) + " |")

    # Threshold row
    thresh_row = "| " + " | ".join(
        ["**Threshold**"] + [f"≥ {THRESHOLDS[m]:.2f}" for m in metric_keys] + ["—", "—"]
    ) + " |"

    return "\n".join([header, divider] + rows + [divider, thresh_row])


def save_eval_results(
    results: dict[str, dict[str, float]],
    k: int = 5,
    path: Path = EVAL_RESULTS_PATH,
) -> None:
    """Write evaluation results to a Markdown file.

    Args:
        results: Output of :func:`run_evaluation`.
        k: Top-k value used for retrieval metrics (included in the report header).
        path: Output file path.
    """
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    table = _markdown_table(results, k)

    # Compute averages
    metric_keys = list(METRIC_DISPLAY.keys())
    avgs = {}
    for m in metric_keys:
        vals = [v[m] for v in results.values() if m in v]
        avgs[m] = round(sum(vals) / len(vals), 4) if vals else 0.0

    avg_row_scores = " | ".join(f"{avgs[m]:.3f}" for m in metric_keys)

    content = f"""# AlphaLens Evaluation Results

Generated: {now}
Golden tickers: {", ".join(results.keys())}
Retrieval k={k}

## Metric Definitions

| Metric | Description | Target |
|--------|-------------|--------|
| Precision@k | Fraction of top-{k} retrieved chunks containing relevant keywords | ≥ 0.60 |
| Recall@k | Fraction of required filing sections covered in top-{k} chunks | ≥ 0.80 |
| Faithfulness | Fraction of monetary claims in report grounded in retrieved chunks | ≥ 0.90 |
| Numerical Acc. | Fraction of key financial figures within 15% of ground-truth range | ≥ 0.95 |
| Earnings Acc. | Direction accuracy of most-recent earnings surprise (beat/miss) | 1.00 |

## Results

{table}

## Averages

| Metric | Score |
|--------|-------|
{chr(10).join(f"| {METRIC_DISPLAY[m]} | {avgs[m]:.3f} |" for m in metric_keys)}

## Notes

- Numerical accuracy uses ±15% tolerance to account for data-source variation (Alpha Vantage vs yfinance vs EDGAR)
- Faithfulness is a lexical proxy (digit match), not semantic entailment — optimistic upper bound
- Earnings surprise accuracy checks beat/miss *direction* only, not magnitude
- All data sourced from free public APIs; no proprietary data used
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    logger.info("Evaluation results saved to %s", path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run AlphaLens evaluation suite")
    parser.add_argument("--tickers", nargs="+", default=None, help="Tickers to evaluate (default: all golden)")
    parser.add_argument("--k", type=int, default=5, help="Top-k for retrieval metrics")
    parser.add_argument("--save-states", action="store_true", help="Save full pipeline states to docs/")
    args = parser.parse_args()

    results = run_evaluation(tickers=args.tickers, k=args.k, save_states=args.save_states)
    save_eval_results(results, k=args.k)

    print(f"\n{'='*60}")
    print("Evaluation complete")
    print(f"{'='*60}")
    for ticker, metrics in results.items():
        print(f"\n{ticker}:")
        for m, v in metrics.items():
            threshold = THRESHOLDS.get(m)
            if threshold is not None:
                icon = "✅" if v >= threshold else "⚠️"
                print(f"  {icon} {METRIC_DISPLAY.get(m, m):25s} {v:.4f}  (target ≥ {threshold})")
            elif m == "pipeline_seconds":
                print(f"     {'Time':25s} {v:.1f}s")
    print(f"\nResults saved to {EVAL_RESULTS_PATH}")
