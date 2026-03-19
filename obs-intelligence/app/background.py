"""
obs-intelligence/app/background.py
─────────────────────────────────────
APScheduler background loops for the Obs-Intelligence Engine.

Schedules
─────────
  run_analysis_loop()  — every 60 seconds
    • detect_anomalies() for compute + storage domains
    • publish Z-scores to Prometheus gauges
    • update current_intelligence["anomalies"]

  run_forecasting()    — every 5 minutes
    • run_forecasts() for configured metrics
    • publish breach-time gauges
    • update current_intelligence["forecasts"]

The current_intelligence dict is the backing store for GET /intelligence/current
and is read by domain agents to enrich their LLM analysis context.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from obs_intelligence.anomaly_detector import detect_anomalies
from obs_intelligence.forecaster import run_forecasts
from obs_intelligence.metrics_publisher import (
    obs_intelligence_analysis_loop_duration_seconds,
    obs_intelligence_analysis_loop_runs_total,
    obs_intelligence_anomaly_detected_total,
    obs_intelligence_anomaly_z_score,
    obs_intelligence_forecast_breach_minutes,
    obs_intelligence_forecast_loop_runs_total,
)

logger = logging.getLogger("obs-intelligence.background")

# ── Shared state: written by background loops, read by GET /intelligence/current
current_intelligence: dict[str, Any] = {
    "anomalies": [],
    "forecasts": [],
    "last_analysis_at": None,
    "last_forecast_at": None,
    "analysis_loop_count": 0,
    "forecast_loop_count": 0,
}

_scheduler: AsyncIOScheduler | None = None
_http: httpx.AsyncClient | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Background jobs
# ─────────────────────────────────────────────────────────────────────────────

async def run_analysis_loop() -> None:
    """Detect anomalies for both domains and update gauges + shared state."""
    if _http is None:
        return

    start = time.perf_counter()
    try:
        compute_signals = await detect_anomalies("compute", _http)
        storage_signals = await detect_anomalies("storage", _http)
        all_signals = compute_signals + storage_signals

        for sig in all_signals:
            domain = "compute" if sig.metric_name in {"error_rate_pct", "latency_p99_ms"} else "storage"
            obs_intelligence_anomaly_z_score.labels(
                metric_name=sig.metric_name,
                domain=domain,
            ).set(sig.z_score)
            obs_intelligence_anomaly_detected_total.labels(
                metric_name=sig.metric_name,
                anomaly_type=sig.anomaly_type,
                domain=domain,
            ).inc()

        current_intelligence["anomalies"] = [
            {
                "metric_name": s.metric_name,
                "z_score": s.z_score,
                "current_value": s.current_value,
                "baseline_mean": s.baseline_mean,
                "anomaly_type": s.anomaly_type,
                "confidence": s.confidence,
                "detected_at": s.detected_at.isoformat() if s.detected_at else None,
            }
            for s in all_signals
        ]
        current_intelligence["last_analysis_at"] = datetime.now(timezone.utc).isoformat()
        current_intelligence["analysis_loop_count"] += 1

        elapsed = time.perf_counter() - start
        obs_intelligence_analysis_loop_duration_seconds.observe(elapsed)
        obs_intelligence_analysis_loop_runs_total.labels(status="success").inc()
        logger.info(
            "Analysis loop #%d: anomalies=%d elapsed=%.2fs",
            current_intelligence["analysis_loop_count"],
            len(all_signals),
            elapsed,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - start
        obs_intelligence_analysis_loop_runs_total.labels(status="error").inc()
        logger.error("Analysis loop error (%.2fs): %s", elapsed, exc)


async def run_forecasting() -> None:
    """Run metric forecasts and update breach-time gauges + shared state."""
    if _http is None:
        return

    try:
        forecasts = await run_forecasts(_http)
        now_dt = datetime.now(timezone.utc)

        for fc in forecasts:
            if fc.predicted_breach is not None:
                breach_dt = fc.predicted_breach
                if breach_dt.tzinfo is None:
                    breach_dt = breach_dt.replace(tzinfo=timezone.utc)
                minutes_left = max(0.0, (breach_dt - now_dt).total_seconds() / 60.0)
            else:
                minutes_left = 0.0
            obs_intelligence_forecast_breach_minutes.labels(
                metric_name=fc.metric_name
            ).set(minutes_left)

        current_intelligence["forecasts"] = [
            {
                "metric_name": fc.metric_name,
                "model_used": fc.model_used,
                "horizon_minutes": fc.horizon_minutes,
                "predicted_breach": fc.predicted_breach.isoformat() if fc.predicted_breach else None,
                "threshold": fc.threshold,
                "forecast_values_sample": fc.forecast_values[:5],
            }
            for fc in forecasts
        ]
        current_intelligence["last_forecast_at"] = datetime.now(timezone.utc).isoformat()
        current_intelligence["forecast_loop_count"] += 1

        obs_intelligence_forecast_loop_runs_total.labels(status="success").inc()
        logger.info(
            "Forecast loop #%d: forecasts=%d",
            current_intelligence["forecast_loop_count"],
            len(forecasts),
        )
    except Exception as exc:
        obs_intelligence_forecast_loop_runs_total.labels(status="error").inc()
        logger.error("Forecasting loop error: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler lifecycle
# ─────────────────────────────────────────────────────────────────────────────

def start_scheduler(http_client: httpx.AsyncClient) -> None:
    """Create and start the APScheduler with both background jobs."""
    global _scheduler, _http
    _http = http_client
    _scheduler = AsyncIOScheduler()
    _scheduler.add_job(
        run_analysis_loop, "interval", seconds=60, id="analysis_loop",
        max_instances=1, misfire_grace_time=30,
    )
    _scheduler.add_job(
        run_forecasting, "interval", minutes=5, id="forecast_loop",
        max_instances=1, misfire_grace_time=60,
    )
    _scheduler.start()
    logger.info("Background scheduler started: analysis=60s forecast=5min")


def stop_scheduler() -> None:
    """Gracefully stop the scheduler."""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("Background scheduler stopped")
