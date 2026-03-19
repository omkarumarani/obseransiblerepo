"""
obs_intelligence/anomaly_detector.py
──────────────────────────────────────
Statistical anomaly detection using Prometheus PromQL subqueries.

Algorithm
─────────
For each metric definition:
  1. Query current value via an instant PromQL expression.
  2. Query rolling mean   via avg_over_time(expr[window:step]).
  3. Query rolling stddev via stddev_over_time(expr[window:step]).
  4. Compute Z = (current − mean) / stddev.
  5. Emit AnomalySignal when |Z| ≥ threshold.

The subquery syntax `avg_over_time(expr[30m:30s])` lets Prometheus
evaluate `expr` as an instant vector at 30-second intervals over the
last 30 minutes and compute the aggregate — no need for recording rules.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import httpx

from obs_intelligence.models import AnomalySignal

logger = logging.getLogger("obs_intelligence.anomaly_detector")

_DEFAULT_PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
_Z_THRESHOLD = float(os.getenv("ANOMALY_Z_THRESHOLD", "2.5"))
_WINDOW_MINUTES = int(os.getenv("ANOMALY_WINDOW_MINUTES", "30"))
_SUBQUERY_STEP = "30s"


# ─────────────────────────────────────────────────────────────────────────────
# Metric definitions per domain
# ─────────────────────────────────────────────────────────────────────────────

_COMPUTE_METRICS: list[dict] = [
    {
        "name": "error_rate_pct",
        "current": (
            '100 * sum(rate(http_server_duration_count{http_response_status_code=~"5.."}[5m]))'
            ' / sum(rate(http_server_duration_count[5m]))'
        ),
        "domain": "compute",
    },
    {
        "name": "latency_p99_ms",
        "current": (
            "histogram_quantile(0.99,"
            " sum(rate(http_server_duration_bucket[5m])) by (le)) * 1000"
        ),
        "domain": "compute",
    },
]

_STORAGE_METRICS: list[dict] = [
    {
        "name": "pool_fill_pct",
        "current": "max(storage_pool_used_bytes / storage_pool_capacity_bytes) * 100",
        "domain": "storage",
    },
    {
        "name": "io_latency_ms",
        "current": "max(storage_io_latency_seconds) * 1000",
        "domain": "storage",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

async def detect_anomalies(
    domain: str,
    http: httpx.AsyncClient,
    service_name: str = "",
    prometheus_url: str | None = None,
) -> list[AnomalySignal]:
    """
    Run Z-score anomaly detection across all metric definitions for *domain*.

    Uses Prometheus avg_over_time / stddev_over_time subqueries over a rolling
    window to establish a dynamic baseline, then computes the Z-score of the
    current value against that baseline.

    Returns a list of AnomalySignal (empty if Prometheus is unavailable or
    all Z-scores are within the normal range).
    """
    url = prometheus_url or _DEFAULT_PROMETHEUS_URL
    metric_defs = _COMPUTE_METRICS if domain == "compute" else _STORAGE_METRICS
    window = f"{_WINDOW_MINUTES}m"
    signals: list[AnomalySignal] = []

    for mdef in metric_defs:
        try:
            current_expr = mdef["current"]
            current_val = await _scalar_query(current_expr, http, url)
            mean_val = await _scalar_query(
                f"avg_over_time(({current_expr})[{window}:{_SUBQUERY_STEP}])", http, url
            )
            std_val = await _scalar_query(
                f"stddev_over_time(({current_expr})[{window}:{_SUBQUERY_STEP}])", http, url
            )

            if current_val is None or mean_val is None or std_val is None:
                continue
            if std_val < 1e-9:
                # Flat signal — Z-score undefined; skip
                continue

            z = (current_val - mean_val) / std_val
            anomaly_type = "spike" if z > 0 else "drop"
            confidence = min(1.0, abs(z) / (_Z_THRESHOLD * 2))

            if abs(z) >= _Z_THRESHOLD:
                signal = AnomalySignal(
                    metric_name=mdef["name"],
                    current_value=round(current_val, 4),
                    baseline_mean=round(mean_val, 4),
                    baseline_stddev=round(std_val, 4),
                    z_score=round(z, 3),
                    anomaly_type=anomaly_type,
                    detected_at=datetime.now(timezone.utc),
                    confidence=round(confidence, 3),
                )
                signals.append(signal)
                logger.info(
                    "Anomaly detected: metric=%s domain=%s z=%.2f type=%s",
                    mdef["name"], domain, z, anomaly_type,
                )
        except Exception as exc:
            logger.warning(
                "Anomaly check failed for %s/%s: %s", domain, mdef["name"], exc
            )

    return signals


# ─────────────────────────────────────────────────────────────────────────────
# Prometheus helper
# ─────────────────────────────────────────────────────────────────────────────

async def _scalar_query(
    promql: str,
    http: httpx.AsyncClient,
    prometheus_url: str,
    timeout: float = 8.0,
) -> float | None:
    """Execute a Prometheus instant query and return the first scalar result."""
    try:
        resp = await http.get(
            f"{prometheus_url}/api/v1/query",
            params={"query": promql},
            timeout=timeout,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        results = data.get("data", {}).get("result", [])
        if not results:
            return None
        val = float(results[0]["value"][1])
        # Discard NaN — Prometheus can return "NaN" for zero-division expressions
        return None if val != val else val
    except Exception as exc:
        logger.debug("Prometheus scalar query failed (%s): %s", promql[:80], exc)
        return None
