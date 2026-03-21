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
  POST /webhook               — Alertmanager webhook receiver (learning tap)

Port: 9100
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .background import current_intelligence, start_scheduler, stop_scheduler
from .telemetry import bootstrap
from obs_intelligence.learning_store import LearningStore
from obs_intelligence.local_llm_enricher import local_llm_enricher
from obs_intelligence.outcome_store import OutcomeStore
from obs_intelligence.incident_coordinator import IncidentCoordinator
from obs_intelligence.metrics_publisher import (
        obs_intelligence_cross_domain_correlations_total,
        obs_intelligence_external_validation_total,
        obs_intelligence_local_validation_duration_seconds,
        obs_intelligence_local_validation_total,
        obs_intelligence_scenario_outcome_total,
        obs_intelligence_webhook_alerts_total,
    )
from obs_intelligence.cross_domain_correlator import cross_domain_correlator
logger = logging.getLogger("obs-intelligence")

_PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")

# Shared outcome store — persists to /data/outcomes.db inside the container
_outcome_store = OutcomeStore()
_learning_store = LearningStore()

# Cross-domain incident coordinator — in-memory, best-effort
_incident_coordinator = IncidentCoordinator()
_LOKI_URL = os.getenv("LOKI_URL", "http://loki:3100")

_http: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http
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

# CORS middleware — allow browser access from ui-backend and command-center
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3005", "http://localhost:3500", "http://localhost:9005", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Bootstrap OTel instrumentation at module load time (must run before first request)
bootstrap(app)


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
        "learning_entries": _learning_store.learning_stats().get("knowledge_entries_total", 0),
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

    # Persist to SQLite and recalculate weight adjustment
    run_id = str(body.get("run_id", ""))
    resolution_time_seconds = body.get("resolution_time_seconds")
    _outcome_store.record(
        scenario_id=scenario_id,
        outcome=outcome,
        run_id=run_id,
        domain=domain or "compute",
    )
    updated = _learning_store.update_outcome(
        scenario_id=scenario_id,
        service_name=service,
        run_id=run_id,
        outcome=outcome,
        resolution_time_seconds=float(resolution_time_seconds) if resolution_time_seconds is not None else None,
    )

    # Best-effort: update the ChromaDB entry for this run_id
    if run_id:
        asyncio.create_task(
            local_llm_enricher.update_incident_outcome(
                run_id=run_id,
                outcome=outcome,
                service_name=service,
                alert_name=scenario_id,
            )
        )

    logger.info(
        "Scenario outcome recorded  scenario=%s  outcome=%s  service=%s  domain=%s  learning_updates=%d",
        scenario_id, outcome, service, domain, updated,
    )
    return {"status": "ok", "scenario_id": scenario_id, "outcome": outcome}


@app.get("/intelligence/scenario-stats")
async def scenario_stats() -> dict:
    """
    Return per-scenario outcome statistics for the React Learning tab.

    Each entry contains:
      scenario_id, weight_adjustment, total_seen, success_count,
      success_rate, last_updated.
    """
    return {"status": "ok", "scenarios": _outcome_store.stats_all()}


@app.post("/intelligence/validate-external-analysis")
async def validate_external_analysis(body: dict) -> dict:
    """
    Record an external LLM analysis and run local dual-validation via ChromaDB + local LLM.

    This endpoint is the canonical entry point for any caller that wants to
    trigger the full Block F validation flow on-demand (e.g. storage-agent,
    external scripts, or manual curl calls).

    The normal pipeline path (compute-agent / storage-agent) does NOT call
    this endpoint directly — the inline dual-validation in llm_enricher.enrich()
    calls local_llm_enricher directly.  This endpoint exists for:
      - External / manual invocation
      - Storage-agent (which shares the same llm_enricher)
      - Backward compatibility with any existing callers
    """
    service_name      = str(body.get("service_name", ""))
    alert_name        = str(body.get("alert_name", ""))
    domain            = str(body.get("domain", "compute"))
    scenario_id       = str(body.get("scenario_id", ""))
    run_id            = str(body.get("run_id", ""))
    external_analysis = dict(body.get("external_analysis") or {})
    ev_lines          = list(body.get("evidence_lines") or [])

    root_cause = str(
        external_analysis.get("root_cause")
        or external_analysis.get("rca_detail", {}).get("probable_cause")
        or external_analysis.get("rca_summary", "")
    )
    incident_text = " ".join(filter(None, [
        f"service={service_name}",
        f"alert={alert_name}",
        f"domain={domain}",
        f"scenario={scenario_id}",
        f"root_cause={root_cause}",
        f"action={external_analysis.get('recommended_action', '')}",
        " ".join(ev_lines[:8]),
    ]))

    incident_context = {
        "service_name": service_name,
        "alert_name":   alert_name,
        "domain":       domain,
        "scenario_id":  scenario_id,
        "risk_score":   float(body.get("risk_score") or 0.0),
        "description":  incident_text,
        "run_id":       run_id,
    }

    # ── Step 1: ChromaDB similarity retrieval ─────────────────────────
    started = time.perf_counter()
    similar = await local_llm_enricher.query_similar_incidents(incident_text, domain)

    # ── Step 2: Local LLM corroboration (best-effort) ─────────────────
    local_val = None
    try:
        local_val = await asyncio.wait_for(
            local_llm_enricher.validate_external_result(
                incident_context=incident_context,
                external_result=external_analysis,
                similar=similar,
            ),
            timeout=15.0,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Local LLM validation timed out in validate-external-analysis  run_id=%s", run_id
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Local LLM validation error: %s", exc)
    obs_intelligence_local_validation_duration_seconds.observe(time.perf_counter() - started)

    # ── Step 3: Persist to ChromaDB (fire-and-forget) ─────────────────
    entry_id = await local_llm_enricher.store_incident_resolution(
        incident_context=incident_context,
        external_result=external_analysis,
        local_validation=local_val,
        similar=similar,
        outcome="pending",
        run_id=run_id,
    )

    # ── Step 4: Prometheus counters ────────────────────────────────
    obs_intelligence_external_validation_total.labels(
        domain=domain,
        provider=str(external_analysis.get("provider", "external")),
        status="recorded",
    ).inc()
    verdict = local_val.validation_status if local_val else "unavailable"
    obs_intelligence_local_validation_total.labels(
        domain=domain,
        verdict=verdict,
    ).inc()

    # ── Response  (backward-compatible schema) ───────────────────────
    # Map new LocalValidationResult field names → legacy names so existing
    # callers that read "status" / "reason" / "completed" keep working.
    validation_dict: dict = {
        "status":               verdict,
        "confidence":           local_val.confidence      if local_val else 0.0,
        "reason":               local_val.reasoning_summary if local_val else "",
        "rca_alignment":        local_val.rca_alignment   if local_val else "",
        "action_alignment":     local_val.action_alignment if local_val else "",
        "suggested_adjustment": local_val.suggested_adjustment if local_val else "",
        "top_similarity":       local_val.top_similarity  if local_val else 0.0,
        "similar_count":        local_val.similar_count   if local_val else 0,
        "local_model":          os.getenv("LOCAL_LLM_MODEL", "qwen3.5"),
        "completed":            local_val is not None,
    }
    return {
        "status":        "ok",
        "stored":        entry_id is not None,
        "entry_id":      entry_id or "",
        "local_validation": validation_dict,
        "top_matches": [
            {
                "id":          e.id,
                "similarity":  e.similarity(),
                "scenario_id": e.metadata.get("scenario_id", ""),
                "outcome":     e.metadata.get("outcome", "pending"),
            }
            for e in similar[:3]
        ],
    }


@app.get("/intelligence/knowledge-entries")
async def knowledge_entries(service_name: str = "", scenario_id: str = "", limit: int = 50) -> dict:
    """
    List knowledge entries from ChromaDB (primary store).
    Falls back to SQLite learning_store entries when ChromaDB returns nothing.
    """
    chroma_entries = await local_llm_enricher.list_entries(
        service_name=service_name,
        scenario_id=scenario_id,
        limit=limit,
    )
    if chroma_entries:
        return {"status": "ok", "source": "chromadb", "entries": chroma_entries}

    # Fallback to SQLite if ChromaDB is empty / unavailable
    sqlite_entries = _learning_store.list_entries(
        service_name=service_name,
        scenario_id=scenario_id,
        limit=limit,
    )
    return {"status": "ok", "source": "sqlite", "entries": sqlite_entries}


@app.get("/intelligence/learning-stats")
async def learning_stats() -> dict:
    """
    Dual-validation learning telemetry.
    Merges ChromaDB-based stats with SQLite-based stats.
    ChromaDB stats are primary; SQLite stats fill any gaps.
    """
    chroma_stats = await local_llm_enricher.knowledge_stats()
    sqlite_stats  = _learning_store.learning_stats()

    # Merge: ChromaDB fields take precedence; supplement with SQLite where absent
    merged = {
        **sqlite_stats,
        **chroma_stats,
        # Always expose these descriptive fields from spec F.7
        "external_llm_calls_30d":           sqlite_stats.get("external_llm_calls_30d", 0),
        "local_validation_attempts_30d":     sqlite_stats.get("local_validation_attempts_30d", 0),
        "local_validation_completed_30d":    chroma_stats.get("knowledge_entries_total", 0),
        "corroborated_count_30d":            chroma_stats.get("corroborated_count", 0),
        "weak_support_count_30d":            chroma_stats.get("weak_support_count", 0),
        "divergent_count_30d":               chroma_stats.get("divergent_count", 0),
        "insufficient_context_count_30d":    chroma_stats.get("insufficient_context_count", 0),
        "avg_top_similarity_30d":            chroma_stats.get("avg_top_similarity", 0.0),
        "knowledge_entries_total":           chroma_stats.get("knowledge_entries_total", 0),
        "knowledge_entries_with_success_outcome": chroma_stats.get("success_outcome_count", 0),
        "local_validation_coverage_pct":     chroma_stats.get("local_validation_coverage_pct", 0.0),
        "corroboration_rate_pct":            chroma_stats.get("corroboration_rate_pct", 0.0),
    }
    return merged


@app.post("/intelligence/record-incident")
async def record_incident(body: dict) -> dict:
    """
    Record a domain incident and detect cross-domain co-occurrence.

    Extended payload (v2) adds a ``signals`` dict so the CrossDomainCorrelator
    can produce a fully-typed UnifiedSREAssessment without an extra Prometheus
    query round-trip.

    Body: {
        "domain":       "compute" | "storage",
        "service_name": "frontend-api",
        "alert_name":   "HighErrorRate",
        "risk_score":   0.72,
        "scenario_id":  "high_error_rate",
        "run_id":       "run-abc123",
        "signals": {                    # optional — enriches correlation
            "error_rate":       0.15,   # compute fraction in [0,1]
            "latency_p99_ms":   620,    # compute p99 in ms
            "cpu_usage_pct":    0.71,   # compute CPU fraction in [0,1]
            "io_latency_s":     0.28,   # storage seconds
            "pool_usage_pct":   0.82,   # storage fraction in [0,1]
            "osd_up":           9,
            "osd_total":        10
        }
    }

    Returns {"status": "ok", "cross_domain_event": null | {...},
             "unified_assessment": null | {UnifiedSREAssessment}}.
    """
    domain       = str(body.get("domain", "compute"))
    service_name = str(body.get("service_name", ""))
    alert_name   = str(body.get("alert_name", ""))
    risk_score   = float(body.get("risk_score", 0.0))
    scenario_id  = str(body.get("scenario_id", ""))
    run_id       = str(body.get("run_id", ""))
    signals      = dict(body.get("signals") or {})

    cross_domain_event = _incident_coordinator.record_incident(
        domain=domain,
        service_name=service_name,
        alert_name=alert_name,
        risk_score=risk_score,
        scenario_id=scenario_id,
        run_id=run_id,
        signals=signals,
    )

    unified_assessment: dict | None = None

    if cross_domain_event:
        # Log without internal _-prefixed entries
        loggable = {k: v for k, v in cross_domain_event.items() if not k.startswith("_")}
        logger.warning(
            "Cross-domain correlation detected  domains=%s  services=%s  "
            "combined_risk=%.3f  alert_a=%s  alert_b=%s",
            loggable["domains"],
            loggable["services"],
            loggable["combined_risk_score"],
            loggable.get("alert_a", ""),
            loggable.get("alert_b", ""),
        )

        # Produce the unified SREAssessment
        compute_entry = cross_domain_event.pop("_compute_entry", {})
        storage_entry = cross_domain_event.pop("_storage_entry", {})
        try:
            unified = cross_domain_correlator.assess(compute_entry, storage_entry)
            unified_assessment = unified.to_dict()
            _incident_coordinator.store_unified_assessment(unified_assessment)
            obs_intelligence_cross_domain_correlations_total.labels(
                correlation_type=unified.correlation_type,
            ).inc()
            logger.info(
                "Unified SREAssessment  type=%s  primary=%s  combined_risk=%.2f",
                unified.correlation_type,
                unified.primary_domain,
                unified.combined_risk_score,
            )
        except Exception as exc:
            logger.warning("CrossDomainCorrelator.assess() failed: %s", exc)

    return {
        "status": "ok",
        "cross_domain_event": {k: v for k, v in cross_domain_event.items() if not k.startswith("_")} if cross_domain_event else None,
        "unified_assessment": unified_assessment,
    }


@app.get("/intelligence/correlation/current")
async def correlation_current() -> dict:
    """
    Return the most recently produced UnifiedSREAssessment (within 10-min TTL).

    Polled by the Streamlit Agent Mesh tab to display the live cross-domain
    correlation panel whenever two domains are simultaneously degraded.

    Returns {"status": "ok", "assessment": null} when nothing active.
    """
    assessment = _incident_coordinator.get_active_correlation()
    return {"status": "ok", "assessment": assessment}


@app.post("/webhook")
async def alertmanager_webhook(body: dict) -> dict:
    """
    Alertmanager webhook receiver (learning tap).

    Alertmanager sends a copy of every alert here via the `obs-intelligence`
    receiver (continue: true in alertmanager.yml).  This endpoint:
      - Logs each alert for cross-domain pattern learning
      - Increments obs_intelligence_webhook_alerts_total{status, severity}
      - Triggers an on-demand anomaly analysis for each firing alert

    Returns 200 quickly so Alertmanager does not time out.
    """
    status   = body.get("status", "unknown")          # "firing" | "resolved"
    alerts   = body.get("alerts", [])
    receiver = body.get("receiver", "obs-intelligence")

    for alert in alerts:
        labels      = alert.get("labels", {})
        annotations = alert.get("annotations", {})
        alert_name  = labels.get("alertname", "unknown")
        severity    = labels.get("severity", "unknown")
        service     = labels.get("service_name", labels.get("job", ""))
        domain      = labels.get("domain", "compute")
        alert_status = alert.get("status", status)

        obs_intelligence_webhook_alerts_total.labels(
            status=alert_status,
            severity=severity,
        ).inc()

        logger.info(
            "Alertmanager alert received  alert=%s  status=%s  severity=%s  "
            "service=%s  domain=%s  receiver=%s",
            alert_name, alert_status, severity, service, domain, receiver,
        )

        # Trigger background anomaly analysis for firing alerts so the
        # intelligence state is refreshed while the incident is active.
        if alert_status == "firing" and _http is not None:
            try:
                from obs_intelligence.anomaly_detector import detect_anomalies
                anomalies = await detect_anomalies(domain, _http, service_name=service)
                logger.debug(
                    "Webhook-triggered analysis  alert=%s  anomalies_found=%d",
                    alert_name, len(anomalies),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Webhook analysis failed for %s: %s", alert_name, exc)

    return {"status": "ok", "alerts_processed": len(alerts)}


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
