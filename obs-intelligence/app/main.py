"""
obs-intelligence/app/main.py
────────────────────────────────────────────────────────────────────────────
Obs-Intelligence Engine — FastAPI service (Phase 3).

Endpoints
─────────
  GET  /health                — liveness check + loop iteration counts
  GET  /metrics               — Prometheus /metrics exposition
  GET  /intelligence/current  — latest anomalies + forecasts from background loops
  POST /analyze               — on-demand analysis for a given domain / alert

Port: 9100
"""

import logging
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .background import current_intelligence, start_scheduler, stop_scheduler
from .telemetry import bootstrap
from obs_intelligence.metrics_publisher import obs_intelligence_scenario_outcome_total

logger = logging.getLogger("obs-intelligence")

_PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
_LOKI_URL = os.getenv("LOKI_URL", "http://loki:3100")

_http: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http
    bootstrap(app)
    _http = httpx.AsyncClient()
    start_scheduler(_http)
    logger.info("Obs-intelligence engine started (port 9100)")
    yield
    stop_scheduler()
    if _http:
        await _http.aclose()
    logger.info("Obs-intelligence engine stopped")


app = FastAPI(
    title="Obs-Intelligence Engine",
    description="Shared intelligence core for the multi-agent AIOps platform.",
    version="3.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "service": "obs-intelligence",
        "phase": "3",
        "analysis_loop_count": current_intelligence.get("analysis_loop_count", 0),
        "forecast_loop_count": current_intelligence.get("forecast_loop_count", 0),
        "last_analysis_at": current_intelligence.get("last_analysis_at"),
        "last_forecast_at": current_intelligence.get("last_forecast_at"),
        "active_anomalies": len(current_intelligence.get("anomalies", [])),
    }


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    return PlainTextResponse(
        content=generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/intelligence/current")
async def intelligence_current() -> dict:
    """Return the latest pre-computed intelligence state from background loops."""
    return {
        "status": "ok",
        "anomalies": current_intelligence.get("anomalies", []),
        "forecasts": current_intelligence.get("forecasts", []),
        "last_analysis_at": current_intelligence.get("last_analysis_at"),
        "last_forecast_at": current_intelligence.get("last_forecast_at"),
        "analysis_loop_count": current_intelligence.get("analysis_loop_count", 0),
        "forecast_loop_count": current_intelligence.get("forecast_loop_count", 0),
    }


@app.post("/intelligence/record-outcome")
async def record_outcome(body: dict) -> dict:
    """
    Record a scenario outcome after an alert is resolved or disposition is made.

    Called by domain agents when Alertmanager sends status=resolved,
    or when a human approves/declines a remediation.

    Body: {
        "scenario_id": "recurring_failure_signature",
        "outcome":     "resolved",   # resolved | escalated | declined | timedout
        "service_name": "frontend-api",  # optional, for logging
        "domain":       "compute"        # optional, for logging
    }

    Increments obs_intelligence_scenario_outcome_total{scenario_id, outcome}.
    """
    scenario_id = str(body.get("scenario_id", "unknown"))
    outcome     = str(body.get("outcome", "resolved"))
    service     = str(body.get("service_name", ""))
    domain      = str(body.get("domain", ""))

    obs_intelligence_scenario_outcome_total.labels(
        scenario_id=scenario_id,
        outcome=outcome,
    ).inc()

    logger.info(
        "Scenario outcome recorded  scenario=%s  outcome=%s  service=%s  domain=%s",
        scenario_id, outcome, service, domain,
    )
    return {"status": "ok", "scenario_id": scenario_id, "outcome": outcome}


@app.post("/analyze")
async def analyze_on_demand(body: dict) -> dict:
    """
    On-demand analysis for a given domain + service.

    Body: {
        "domain": "compute|storage",
        "service_name": "frontend-api",
        "alert_name": "HighErrorRate",
        "severity": "warning"
    }
    """
    if _http is None:
        return {"status": "error", "message": "HTTP client not ready"}

    from obs_intelligence.anomaly_detector import detect_anomalies
    from obs_intelligence.forecaster import run_forecasts

    domain = body.get("domain", "compute")
    service_name = body.get("service_name", "")

    anomalies = await detect_anomalies(domain, _http, service_name=service_name)
    forecasts = await run_forecasts(_http)

    return {
        "status": "ok",
        "domain": domain,
        "service_name": service_name,
        "alert_name": body.get("alert_name", ""),
        "severity": body.get("severity", ""),
        "anomalies": [
            {
                "metric_name": s.metric_name,
                "z_score": s.z_score,
                "current_value": s.current_value,
                "baseline_mean": s.baseline_mean,
                "anomaly_type": s.anomaly_type,
                "confidence": s.confidence,
            }
            for s in anomalies
        ],
        "forecasts": [
            {
                "metric_name": fc.metric_name,
                "model_used": fc.model_used,
                "predicted_breach": fc.predicted_breach.isoformat() if fc.predicted_breach else None,
                "threshold": fc.threshold,
            }
            for fc in forecasts
        ],
    }
