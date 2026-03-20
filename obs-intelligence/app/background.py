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
import os
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
    obs_intelligence_predictive_alerts_sent_total,
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

# ── Domain agent URLs (read once at scheduler start) ─────────────────────────
_COMPUTE_AGENT_URL: str = os.getenv("COMPUTE_AGENT_URL", "http://compute-agent:9000")
_STORAGE_AGENT_URL: str = os.getenv("STORAGE_AGENT_URL", "http://storage-agent:9001")
_PROMETHEUS_URL: str = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")

# Thresholds for predictive alert dispatch
_PREDICTIVE_RISK_THRESHOLD: float = float(os.getenv("PREDICTIVE_RISK_THRESHOLD", "0.75"))
_PREDICTIVE_CONFIDENCE_THRESHOLD: float = float(os.getenv("PREDICTIVE_CONFIDENCE_THRESHOLD", "0.7"))
_PREDICTIVE_Z_THRESHOLD: float = float(os.getenv("ANOMALY_Z_THRESHOLD", "2.5"))


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

        # After updating shared state, check if we should fire predictive alerts
        await _dispatch_predictive_alerts(all_signals)

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
# Predictive alert dispatch
# ─────────────────────────────────────────────────────────────────────────────

async def _has_active_prometheus_alerts(domain: str) -> bool:
    """
    Query Prometheus to check whether any alerts are actively firing for
    the given domain.  Returns True if at least one firing alert exists.
    Uses `ALERTS{alertstate="firing",domain="<domain>"}` via the
    Prometheus HTTP API.  Falls back to False on any error so predictive
    alerts can still fire if Prometheus is temporarily unreachable.
    """
    if _http is None:
        return False
    try:
        # Try domain-scoped check first, then fall back to any firing alert
        queries = [
            f'ALERTS{{alertstate="firing",domain="{domain}"}}',
            f'ALERTS{{alertstate="firing",team="{domain}"}}',
        ]
        for q in queries:
            r = await _http.get(
                f"{_PROMETHEUS_URL}/api/v1/query",
                params={"query": q},
                timeout=5.0,
            )
            data = r.json()
            if data.get("status") == "success" and data.get("data", {}).get("result"):
                return True
        return False
    except Exception as exc:
        logger.debug("Could not query Prometheus alerts (%s) — assuming no active alerts", exc)
        return False


async def _dispatch_predictive_alerts(all_signals: list) -> None:
    """
    After each analysis loop, check if any domain has anomalies meeting the
    predictive alert criteria:
      - abs(z_score) > _PREDICTIVE_Z_THRESHOLD
      - confidence > _PREDICTIVE_CONFIDENCE_THRESHOLD
      - computed risk_score > _PREDICTIVE_RISK_THRESHOLD
      - no active Prometheus alert currently firing for that domain

    If criteria are met, POST to the appropriate domain agent's
    POST /predictive-alert endpoint and update the Prometheus counter.
    """
    if _http is None:
        return

    # Group strongest anomaly per domain
    best_by_domain: dict[str, Any] = {}
    for sig in all_signals:
        domain = "compute" if sig.metric_name in {
            "error_rate_pct", "latency_p99_ms", "rps", "cpu_usage_pct", "memory_usage_pct"
        } else "storage"
        if abs(sig.z_score) < _PREDICTIVE_Z_THRESHOLD:
            continue
        if sig.confidence < _PREDICTIVE_CONFIDENCE_THRESHOLD:
            continue
        existing = best_by_domain.get(domain)
        if existing is None or abs(sig.z_score) > abs(existing.z_score):
            best_by_domain[domain] = sig

    for domain, sig in best_by_domain.items():
        # Risk score: normalised z-score * confidence, capped at 1.0
        risk_score = min(1.0, (abs(sig.z_score) / 5.0) * sig.confidence)
        if risk_score < _PREDICTIVE_RISK_THRESHOLD:
            continue

        # Check no Prometheus alert is already firing for this domain
        if await _has_active_prometheus_alerts(domain):
            logger.debug(
                "Skipping predictive alert for %s — Prometheus alert already firing", domain
            )
            continue

        # Find forecast context
        forecast_minutes = 0
        for fc in current_intelligence.get("forecasts", []):
            if fc.get("predicted_breach") and fc.get("metric_name", "").startswith(
                "storage" if domain == "storage" else "http"
            ):
                breach_dt = datetime.fromisoformat(fc["predicted_breach"].replace("Z", "+00:00"))
                now_dt = datetime.now(timezone.utc)
                if breach_dt > now_dt:
                    forecast_minutes = int((breach_dt - now_dt).total_seconds() / 60)
                    break

        agent_url = _COMPUTE_AGENT_URL if domain == "compute" else _STORAGE_AGENT_URL
        payload = {
            "service_name": sig.service_name if hasattr(sig, "service_name") else "unknown",
            "domain": domain,
            "scenario_id": f"anomaly_{sig.metric_name}",
            "risk_score": round(risk_score, 3),
            "confidence": round(sig.confidence, 3),
            "description": (
                f"Metric `{sig.metric_name}` is anomalous: "
                f"current={sig.current_value:.3f}, "
                f"baseline={sig.baseline_mean:.3f}, "
                f"z_score={sig.z_score:.2f}."
            ),
            "forecast_breach_minutes": forecast_minutes,
            "anomaly_metric": sig.metric_name,
            "anomaly_z_score": round(sig.z_score, 3),
        }

        try:
            resp = await _http.post(
                f"{agent_url}/predictive-alert",
                json=payload,
                timeout=10.0,
            )
            if resp.status_code < 300:
                obs_intelligence_predictive_alerts_sent_total.labels(domain=domain).inc()
                logger.info(
                    "Predictive alert dispatched  domain=%s  metric=%s  risk=%.2f  agent=%s",
                    domain, sig.metric_name, risk_score, agent_url,
                )
            else:
                logger.warning(
                    "Predictive alert rejected by %s  status=%d  body=%s",
                    agent_url, resp.status_code, resp.text[:200],
                )
        except Exception as exc:
            logger.warning(
                "Failed to dispatch predictive alert to %s: %s", agent_url, exc
            )


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
