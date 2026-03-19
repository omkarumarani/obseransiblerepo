"""
storage-agent/app/main.py
──────────────────────────────────────────────────────────────────────────────
Storage AIOps Agent — FastAPI service.

Receives Alertmanager webhooks for storage alerts (domain=storage),
runs a 6-step agent pipeline, and exposes Prometheus metrics.

Endpoints
─────────
  GET  /health                          — liveness probe
  GET  /metrics                         — Prometheus scrape endpoint
  POST /webhook                         — Alertmanager webhook receiver
  POST /approval/{session_id}/decision  — human approval callback
  GET  /approvals/pending               — list pending approval requests

Pipeline (called by xyOps workflow nodes):
  POST /pipeline/start
  POST /pipeline/agent/storage-metrics
  POST /pipeline/agent/logs
  POST /pipeline/agent/analyze
  POST /pipeline/agent/ticket
  POST /pipeline/agent/approval

Ports
─────
  9001  HTTP (obs-net internal + host)
──────────────────────────────────────────────────────────────────────────────
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request, Response, status
from opentelemetry import trace
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest, REGISTRY
from pydantic import BaseModel

from .pipeline import init_pipeline, _sessions
from .storage_analyst import AI_ENABLED
from .telemetry import (
    alert_processing_histogram,
    get_tracer,
    setup_telemetry,
    storage_agent_webhook_received_total,
    storage_agent_alert_processing_seconds,
)
from .xyops_provisioner import ensure_storage_workflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("storage-agent")

# ── FastAPI app ────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(application: FastAPI):
    """Startup: provision xyOps storage workflow."""
    xyops_url = os.getenv("XYOPS_URL", "http://xyops:5522")
    xyops_api_key = os.getenv("XYOPS_API_KEY", "")
    headers = {"Content-Type": "application/json"}
    if xyops_api_key:
        headers["X-API-Key"] = xyops_api_key

    try:
        async with httpx.AsyncClient() as http:
            async def _post(path: str, body: dict) -> dict:
                try:
                    r = await http.post(f"{xyops_url}{path}", json=body, headers=headers, timeout=10.0)
                    return r.json()
                except Exception as e:
                    return {"error": str(e)}

            async def _get(path: str) -> dict:
                try:
                    r = await http.get(f"{xyops_url}{path}", headers=headers, timeout=10.0)
                    return r.json()
                except Exception as e:
                    return {"error": str(e)}

            await ensure_storage_workflow(_post, _get)
    except Exception as exc:
        logger.warning("xyOps provisioning skipped (xyOps may not be ready): %s", exc)

    logger.info("Storage agent ready  port=9001  ai_enabled=%s", AI_ENABLED)
    yield


app = FastAPI(
    title="Storage AIOps Agent",
    description="Receives Alertmanager storage alerts, runs AI analysis, and creates xyOps tickets.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Bootstrap OTel (before routes) ────────────────────────────────────────────
setup_telemetry(fastapi_app=app, service_name="storage-agent")
tracer = get_tracer()

# ── Register pipeline endpoints ────────────────────────────────────────────────
init_pipeline(app)

# ── Config ─────────────────────────────────────────────────────────────────────
XYOPS_URL: str = os.getenv("XYOPS_URL", "http://xyops:5522")
XYOPS_API_KEY: str = os.getenv("XYOPS_API_KEY", "")
BRIDGE_INTERNAL_URL: str = os.getenv("BRIDGE_INTERNAL_URL", "http://storage-agent:9001")

# ── Shared HTTP client ─────────────────────────────────────────────────────────
_http: httpx.AsyncClient | None = None


def _xyops_headers() -> dict[str, str]:
    h = {"Content-Type": "application/json"}
    if XYOPS_API_KEY:
        h["X-API-Key"] = XYOPS_API_KEY
    return h


async def _xyops_post(path: str, body: dict) -> dict:
    if not _http:
        return {"error": "http client not ready"}
    try:
        resp = await _http.post(
            f"{XYOPS_URL}{path}", json=body, headers=_xyops_headers(), timeout=10.0
        )
        return resp.json()
    except Exception as exc:
        logger.warning("xyOps POST %s failed: %s", path, exc)
        return {"error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# HTTP client lifecycle (separate from lifespan — reused across requests)
# ─────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def _startup():
    global _http
    _http = httpx.AsyncClient()


@app.on_event("shutdown")
async def _shutdown():
    if _http:
        await _http.aclose()


# ═══════════════════════════════════════════════════════════════════════════════
# Health + Metrics
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "service": "storage-agent",
        "xyops_url": XYOPS_URL,
        "ai_enabled": AI_ENABLED,
        "active_sessions": len(_sessions),
    }


@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus scrape endpoint — exposes storage agent action counters."""
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


# ═══════════════════════════════════════════════════════════════════════════════
# Approval callback
# ═══════════════════════════════════════════════════════════════════════════════

class ApprovalDecision(BaseModel):
    approved: bool
    decided_by: str = "unknown"
    notes: str = ""


@app.post("/approval/{session_id}/decision")
async def approval_decision(session_id: str, decision: ApprovalDecision) -> dict:
    """
    Called by a human (or xyOps event) to approve or decline a storage remediation.
    session_id matches the pipeline session / service_name used in the pipeline.
    """
    from .pipeline import _sessions, _run_playbook, _increment_action_counter
    from .telemetry import storage_agent_autonomous_remediations_total

    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"No session: {session_id}")

    if session.status not in ("awaiting_approval",):
        return {"status": session.status, "message": "Not awaiting approval"}

    action = session.ai_result.get("recommended_action", "unknown")

    async with httpx.AsyncClient() as http:
        if decision.approved:
            logger.info("Approval GRANTED by %s for session=%s action=%s", decision.decided_by, session_id, action)
            session.status = "executing"
            _increment_action_counter(action)
            storage_agent_autonomous_remediations_total.inc()

            await http.post(
                f"{XYOPS_URL}/api/app/add_ticket_change/v1",
                json={"id": session.ticket_id, "change": {
                    "type": "comment",
                    "body": f"[OK] **Approved by {decision.decided_by}** — executing `{action}` playbook",
                }},
                headers=_xyops_headers(),
                timeout=10.0,
            )
            await _run_playbook(session, http)
            session.status = "executed"
        else:
            logger.info("Approval DECLINED by %s for session=%s", decision.decided_by, session_id)
            session.status = "declined"
            await http.post(
                f"{XYOPS_URL}/api/app/add_ticket_change/v1",
                json={"id": session.ticket_id, "change": {
                    "type": "comment",
                    "body": f"[!!] **Declined by {decision.decided_by}** — remediation will NOT execute. Notes: {decision.notes}",
                }},
                headers=_xyops_headers(),
                timeout=10.0,
            )

    return {"status": session.status, "session_id": session_id, "decided_by": decision.decided_by}


@app.get("/approvals/pending")
async def list_pending_approvals() -> dict:
    pending = [
        {"session_id": s.session_id, "alert_name": s.alert_name, "action": s.ai_result.get("recommended_action")}
        for s in _sessions.values()
        if s.status == "awaiting_approval"
    ]
    return {"count": len(pending), "items": pending}


# ═══════════════════════════════════════════════════════════════════════════════
# Alertmanager webhook receiver
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/webhook", status_code=status.HTTP_200_OK)
async def alertmanager_webhook(request: Request) -> dict:
    """
    Receive Alertmanager webhook for storage domain alerts.
    For each firing alert, triggers the storage pipeline via /pipeline/start
    in the background (non-blocking, so Alertmanager always gets 200).
    """
    payload: dict[str, Any] = await request.json()
    group_status: str = payload.get("status", "unknown")
    alerts: list[dict] = payload.get("alerts", [])

    storage_agent_webhook_received_total.labels(group_status=group_status).inc()

    logger.info("Storage webhook received  status=%s  alerts=%d", group_status, len(alerts))

    for alert in alerts:
        alert_name: str = alert["labels"].get("alertname", "unknown")
        service_name: str = alert["labels"].get(
            "service_name", alert["labels"].get("job", "storage-simulator")
        )
        severity: str = alert["labels"].get("severity", "warning")
        alert_status: str = alert.get("status", group_status)
        summary: str = alert.get("annotations", {}).get("summary", alert_name)
        description: str = alert.get("annotations", {}).get("description", "")
        dashboard_url: str = alert.get("annotations", {}).get(
            "dashboard_url", "http://grafana:3000/d/agentic-ai-overview"
        )

        if alert_status == "firing":
            # Fire-and-forget: kick off the pipeline without blocking Alertmanager
            asyncio.create_task(
                _run_storage_pipeline(alert_name, service_name, severity, summary, description, dashboard_url)
            )
        else:
            logger.info("Storage alert resolved: %s / %s", alert_name, service_name)

    return {"status": "ok", "received": len(alerts)}


async def _run_storage_pipeline(
    alert_name: str,
    service_name: str,
    severity: str,
    summary: str,
    description: str,
    dashboard_url: str,
) -> None:
    """Drive the 6-step storage pipeline end-to-end (background task)."""
    t_start = time.perf_counter()
    base = BRIDGE_INTERNAL_URL

    async with httpx.AsyncClient() as http:
        try:
            # Step 1: start
            r1 = await http.post(f"{base}/pipeline/start", json={
                "service_name": service_name,
                "alert_name": alert_name,
                "severity": severity,
                "summary": summary,
                "description": description,
                "dashboard_url": dashboard_url,
            }, timeout=30.0)
            r1.raise_for_status()

            body = {"session_id": service_name}
            for path in [
                "/pipeline/agent/storage-metrics",
                "/pipeline/agent/logs",
                "/pipeline/agent/analyze",
                "/pipeline/agent/ticket",
                "/pipeline/agent/approval",
            ]:
                r = await http.post(f"{base}{path}", json=body, timeout=120.0)
                r.raise_for_status()

            elapsed = time.perf_counter() - t_start
            storage_agent_alert_processing_seconds.observe(elapsed)
            logger.info("Storage pipeline complete  alert=%s  elapsed=%.2fs", alert_name, elapsed)

        except Exception as exc:
            elapsed = time.perf_counter() - t_start
            storage_agent_alert_processing_seconds.observe(elapsed)
            logger.error("Storage pipeline failed for %s: %s", alert_name, exc)
