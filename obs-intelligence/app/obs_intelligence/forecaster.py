"""
obs_intelligence/forecaster.py
────────────────────────────────
Linear regression forecasting using numpy, with exponential growth detection.

Workflow
────────
For each configured metric:
  1. Fetch a historical time-series from Prometheus (/api/v1/query_range).
  2. Fit a linear model: value = slope × t + intercept   (numpy.polyfit deg=1).
  3. Detect exponential growth: compare first-half vs second-half slope.
  4. Project forward up to FORECAST_HORIZON_MINUTES minutes.
  5. Find the first forecast point that crosses the alerting threshold.
  6. Return a ForecastResult with 95 % CI bands (residual-based ± 1.96σ).
"""

from __future__ import annotations

import logging
import os
import time as _time
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import numpy as np

from obs_intelligence.models import ForecastResult

logger = logging.getLogger("obs_intelligence.forecaster")

_DEFAULT_PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
_FORECAST_HORIZON_MINUTES = int(os.getenv("FORECAST_HORIZON_MINUTES", "60"))


# ── Metric configurations ────────────────────────────────────────────────────

_FORECAST_CONFIGS: list[dict] = [
    {
        "metric_name": "pool_fill_pct",
        "promql": "max(storage_pool_used_bytes / storage_pool_capacity_bytes) * 100",
        "threshold": 85.0,
        "step": "1m",
        "lookback_minutes": 60,
    },
    {
        "metric_name": "error_rate_pct",
        "promql": (
            '100 * sum(rate(http_server_duration_count{http_response_status_code=~"5.."}[5m]))'
            ' / sum(rate(http_server_duration_count[5m]))'
        ),
        "threshold": 10.0,
        "step": "30s",
        "lookback_minutes": 30,
    },
    {
        "metric_name": "io_latency_ms",
        "promql": "max(storage_io_latency_seconds) * 1000",
        "threshold": 200.0,
        "step": "30s",
        "lookback_minutes": 30,
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

async def run_forecasts(
    http: httpx.AsyncClient,
    prometheus_url: str | None = None,
) -> list[ForecastResult]:
    """Run forecasts for all configured metrics and return results."""
    url = prometheus_url or _DEFAULT_PROMETHEUS_URL
    results: list[ForecastResult] = []

    for cfg in _FORECAST_CONFIGS:
        try:
            result = await _forecast_one(cfg, http, url)
            if result is not None:
                results.append(result)
        except Exception as exc:
            logger.warning("Forecast failed for %s: %s", cfg["metric_name"], exc)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _forecast_one(
    cfg: dict,
    http: httpx.AsyncClient,
    prometheus_url: str,
) -> ForecastResult | None:
    """Fetch historical data and produce a linear-regression forecast."""
    now = int(_time.time())
    lookback_secs = cfg["lookback_minutes"] * 60

    raw = await _range_query(
        cfg["promql"], http, prometheus_url,
        start=str(now - lookback_secs),
        end=str(now),
        step=cfg["step"],
    )

    if not raw or len(raw) < 4:
        logger.debug(
            "Not enough history for forecast: metric=%s (got %d points)",
            cfg["metric_name"], len(raw),
        )
        return None

    ts_arr = np.array([float(r[0]) for r in raw])
    val_arr = np.array([float(r[1]) for r in raw])

    # Normalise time to minutes from first sample
    t0 = ts_arr[0]
    t_min = (ts_arr - t0) / 60.0

    # Linear regression: value = slope × t + intercept
    coeffs = np.polyfit(t_min, val_arr, deg=1)
    slope, intercept = float(coeffs[0]), float(coeffs[1])

    # Forecast the next <horizon> minutes at 1-min resolution
    last_t = float(t_min[-1])
    horizon = _FORECAST_HORIZON_MINUTES
    fcast_t = np.arange(1.0, horizon + 1, dtype=float)
    fcast_vals = np.clip(slope * (last_t + fcast_t) + intercept, 0.0, None)

    # Absolute forecast timestamps
    base_dt = datetime.fromtimestamp(float(ts_arr[-1]), tz=timezone.utc)
    fcast_dts = [base_dt + timedelta(minutes=int(m)) for m in fcast_t]

    # 95 % CI from residuals
    predicted_hist = slope * t_min + intercept
    residuals = val_arr - predicted_hist
    std_resid = float(np.std(residuals))
    ci_lower = (fcast_vals - 1.96 * std_resid).tolist()
    ci_upper = (fcast_vals + 1.96 * std_resid).tolist()

    # Predicted breach time
    threshold = cfg.get("threshold")
    predicted_breach: datetime | None = None
    if threshold is not None:
        for i, fv in enumerate(fcast_vals):
            if fv >= threshold:
                predicted_breach = fcast_dts[i]
                break

    # Exponential growth detection: second-half slope > 2× first-half slope
    model_str = "linear"
    mid = len(t_min) // 2
    if len(t_min) >= 8 and mid > 0:
        s1 = float(np.polyfit(t_min[:mid], val_arr[:mid], 1)[0])
        s2 = float(np.polyfit(t_min[mid:], val_arr[mid:], 1)[0])
        if s2 > s1 * 2.0 and s2 > 0:
            model_str = "exponential_growth"
            logger.info(
                "Exponential growth detected: metric=%s slope_ratio=%.2f",
                cfg["metric_name"], s2 / max(s1, 1e-9),
            )

    return ForecastResult(
        metric_name=cfg["metric_name"],
        forecast_values=fcast_vals.tolist(),
        forecast_timestamps=fcast_dts,
        predicted_breach=predicted_breach,
        threshold=threshold,
        confidence_interval_lower=ci_lower,
        confidence_interval_upper=ci_upper,
        model_used=model_str,
        horizon_minutes=horizon,
    )


async def _range_query(
    promql: str,
    http: httpx.AsyncClient,
    prometheus_url: str,
    start: str,
    end: str,
    step: str,
    timeout: float = 10.0,
) -> list[list[Any]]:
    """Execute a Prometheus range query and return [[ts, val], ...] for the first series."""
    try:
        resp = await http.get(
            f"{prometheus_url}/api/v1/query_range",
            params={"query": promql, "start": start, "end": end, "step": step},
            timeout=timeout,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        results = data.get("data", {}).get("result", [])
        if not results:
            return []
        return results[0].get("values", [])
    except Exception as exc:
        logger.debug("Range query failed (%s): %s", promql[:60], exc)
        return []
