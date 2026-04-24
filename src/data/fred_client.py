"""FRED client — fetches macro indicators via the fredapi library."""

import logging
from typing import Any, Optional

from src.config import FRED_API_KEY

logger = logging.getLogger(__name__)

_INDICATORS: dict[str, str] = {
    "FEDFUNDS": "Federal Funds Rate (%)",
    "GDP": "Real GDP (Billions $)",
    "CPIAUCSL": "CPI (All Urban Consumers)",
    "UNRATE": "Unemployment Rate (%)",
}

_TREND_THRESHOLDS: dict[str, float] = {
    "FEDFUNDS": 0.05,
    "GDP": 50.0,
    "CPIAUCSL": 0.5,
    "UNRATE": 0.1,
}


def _compute_trend(series: Any, threshold: float) -> str:
    """Determine directional trend from the last two observations.

    Args:
        series: pandas Series from fredapi (index=date, values=float).
        threshold: Minimum absolute delta to label as up/down.

    Returns:
        "up", "down", or "stable".
    """
    if series is None or len(series) < 2:
        return "stable"
    delta = float(series.iloc[-1]) - float(series.iloc[-2])
    if delta > threshold:
        return "up"
    if delta < -threshold:
        return "down"
    return "stable"


class FredClient:
    """Wrapper around fredapi that returns latest macro indicator values.

    Args:
        api_key: FRED API key. Defaults to value from config.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key or FRED_API_KEY

    def _build_fred(self) -> Any:
        """Instantiate fredapi.Fred.

        Raises:
            ImportError: If fredapi is not installed.
        """
        try:
            from fredapi import Fred
            return Fred(api_key=self._api_key)
        except ImportError as exc:
            raise ImportError("fredapi is required: pip install fredapi") from exc

    def get_macro_snapshot(self) -> dict[str, Any]:
        """Fetch the latest values and recent trend for each tracked indicator.

        Returns:
            Dict keyed by FRED series ID, each containing:
                label (str): Human-readable name,
                current_value (float | None): Most recent observation,
                previous_value (float | None): Second-most-recent observation,
                trend (str): "up" | "down" | "stable",
                unit (str): Series units string from FRED metadata,
                error (str): Non-empty if this series could not be fetched.

            Top-level key ``fetch_success`` (bool) is True only if all series
            were retrieved without error.
        """
        if not self._api_key:
            logger.error("FRED_API_KEY not set — macro data unavailable")
            return {
                "fetch_success": False,
                "error": "FRED_API_KEY not configured",
                **{
                    sid: {
                        "label": label,
                        "current_value": None,
                        "previous_value": None,
                        "trend": "stable",
                        "unit": "",
                        "error": "no API key",
                    }
                    for sid, label in _INDICATORS.items()
                },
            }

        try:
            fred = self._build_fred()
        except ImportError as exc:
            logger.error("fredapi not installed: %s", exc)
            return {"fetch_success": False, "error": str(exc)}

        results: dict[str, Any] = {"fetch_success": True}
        all_ok = True

        for series_id, label in _INDICATORS.items():
            try:
                series = fred.get_series(series_id, observation_start="2020-01-01")
                series = series.dropna()

                current = float(series.iloc[-1]) if len(series) >= 1 else None
                previous = float(series.iloc[-2]) if len(series) >= 2 else None
                threshold = _TREND_THRESHOLDS.get(series_id, 0.1)
                trend = _compute_trend(series, threshold)

                try:
                    info = fred.get_series_info(series_id)
                    unit = str(info.get("units", ""))
                except Exception:
                    unit = ""

                results[series_id] = {
                    "label": label,
                    "current_value": current,
                    "previous_value": previous,
                    "trend": trend,
                    "unit": unit,
                    "error": "",
                }
                logger.info("FRED %s: %.4f (%s)", series_id, current or 0, trend)

            except Exception as exc:
                logger.error("FRED fetch failed for %s: %s", series_id, exc)
                results[series_id] = {
                    "label": label,
                    "current_value": None,
                    "previous_value": None,
                    "trend": "stable",
                    "unit": "",
                    "error": str(exc),
                }
                all_ok = False

        results["fetch_success"] = all_ok
        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    client = FredClient()
    snapshot = client.get_macro_snapshot()

    print(f"\n=== Macro Snapshot (fetch_success={snapshot['fetch_success']}) ===")
    arrows = {"up": "↑", "down": "↓", "stable": "→"}
    for sid, label in _INDICATORS.items():
        entry = snapshot.get(sid, {})
        val = entry.get("current_value")
        trend = entry.get("trend", "?")
        err = entry.get("error", "")
        if err:
            print(f"  {sid:12s} {label:35s} ERROR: {err}")
        elif val is None:
            print(f"  {sid:12s} {label:35s} {'N/A':>10s}  {arrows.get(trend, '?')}")
        else:
            print(f"  {sid:12s} {label:35s} {val:>10.4f}  {arrows.get(trend, '?')}")
