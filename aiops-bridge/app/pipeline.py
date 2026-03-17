"""
aiops-bridge/app/pipeline.py
────────────────────────────────────────────────────────────────
Agent-to-Agent Pipeline — HTTP endpoints for the xyOps Workflow canvas.

Architecture
────────────
Each endpoint is one "Agent node" in the xyOps Scheduler → Workflows
visual canvas.  When you click Run on the "AIOps Agent Pipeline" workflow,
xyOps fires each httpplug node in sequence.  You watch:
  • Each node turn green (success) or red (failure) on the canvas
  • Step-by-step [>>]/[OK]/[!!] comments appear live on the incident ticket

Pipeline session design
───────────────────────
All agent nodes share a session keyed by `session_id` (defaults to
`service_name`).  Node 1 creates the session + skeleton ticket.
Nodes 2–6 retrieve the session and add their results.  This way every
httpplug node can send a simple fixed JSON body without needing to
chain output from a previous node.

Endpoints (each = one xyOps workflow node)
──────────────────────────────────────────
  POST /pipeline/start          Agent 1 — creates session + skeleton ticket
  POST /pipeline/agent/logs     Agent 2 — Loki log fetcher
  POST /pipeline/agent/metrics  Agent 3 — Prometheus analyst
  POST /pipeline/agent/analyze  Agent 4 — Claude AI analyst
  POST /pipeline/agent/ticket   Agent 5 — Incident scribe (enriches body)
  POST /pipeline/agent/approval Agent 6 — Approval gateway

GET  /pipeline/session/{id}    — inspect current session state (debug)
────────────────────────────────────────────────────────────────
"""

import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .ai_analyst import (
    AI_ENABLED,
    build_enriched_ticket_body,
    fetch_loki_logs,
    fetch_prometheus_context,
    generate_ai_analysis,
    get_notify_list,
)
from .approval_workflow import request_approval
from .xyops_client import TOTAL_STEPS, post_step_comment

logger = logging.getLogger("aiops-bridge.pipeline")

XYOPS_URL: str = os.getenv("XYOPS_URL", "http://xyops:5522")
REQUIRE_APPROVAL: bool = os.getenv("REQUIRE_APPROVAL", "true").lower() != "false"
APPROVAL_SEVERITY_THRESHOLD: set[str] = {"warning", "critical"}
SESSION_TTL_SECONDS: int = 3600  # sessions expire after 1 hour
# Seconds each workflow node visibly "runs" before completing — lets you watch
# the xyOps canvas step by step.  Set to 0 to disable.
WORKFLOW_STEP_DELAY: int = int(os.getenv("WORKFLOW_STEP_DELAY_SECONDS", "5"))


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline session store
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineSession:
    session_id: str
    service_name: str
    alert_name: str
    severity: str
    summary: str
    description: str
    dashboard_url: str
    starts_at: str
    bridge_trace_id: str = ""
    created_at: float = field(default_factory=time.time)

    # Skeleton ticket (created in /start so comments can land immediately)
    ticket_id: str = ""
    ticket_num: int = 0

    # Accumulated per-agent results
    logs: str = ""
    metrics: dict = field(default_factory=dict)
    analysis: dict = field(default_factory=dict)

    # Approval gate
    approval_id: str = ""
    approval_ticket_id: str = ""
    approval_ticket_num: int = 0

    # Current pipeline stage (for debugging/inspection)
    stage: str = "created"


# {session_id: PipelineSession}
_sessions: dict[str, PipelineSession] = {}


# ── Module-level client refs — set by init_pipeline() on bridge startup ────────
_http: httpx.AsyncClient | None = None
_xyops_post_fn = None


def init_pipeline(http: httpx.AsyncClient, xyops_post_fn) -> None:
    """
    Called once from aiops-bridge main.py startup after _http is ready.
    Gives pipeline agents access to the shared httpx client and the
    xyOps POST helper (which already carries auth headers + base URL).
    """
    global _http, _xyops_post_fn
    _http = http
    _xyops_post_fn = xyops_post_fn
    logger.info("Pipeline agents initialized")


async def _post(path: str, body: dict) -> dict:
    """Delegate to main.py's _xyops_post (shared httpx client + auth headers)."""
    if _xyops_post_fn:
        return await _xyops_post_fn(path, body)
    return {"error": "pipeline not initialized — init_pipeline() not yet called"}


def _require_session(session_id: str) -> PipelineSession:
    """Load session or raise 404 if not found."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No pipeline session '{session_id}'. "
                "Run Agent 1 (POST /pipeline/start) first."
            ),
        )
    return session


def _gc_sessions() -> None:
    """Evict sessions older than SESSION_TTL_SECONDS."""
    cutoff = time.time() - SESSION_TTL_SECONDS
    expired = [k for k, s in _sessions.items() if s.created_at < cutoff]
    for k in expired:
        del _sessions[k]
        logger.info("Evicted expired pipeline session: %s", k)


# ═══════════════════════════════════════════════════════════════════════════════
# Request / Response models
# ═══════════════════════════════════════════════════════════════════════════════

class StartRequest(BaseModel):
    service_name: str
    alert_name: str
    severity: str = "warning"
    summary: str = ""
    description: str = ""
    dashboard_url: str = "http://grafana:3000/d/obs-overview"
    starts_at: str = ""
    session_id: str = ""     # defaults to service_name if not provided


class AgentRequest(BaseModel):
    session_id: str


# ═══════════════════════════════════════════════════════════════════════════════
# Router
# ═══════════════════════════════════════════════════════════════════════════════

pipeline_router = APIRouter(prefix="/pipeline", tags=["pipeline-agents"])


# ── Agent 1: Pipeline Start ────────────────────────────────────────────────────

@pipeline_router.post("/start")
async def pipeline_start(req: StartRequest) -> dict:
    """
    Agent 1 — Create the pipeline session and a skeleton incident ticket.

    Called by the first httpplug node in the xyOps "AIOps Agent Pipeline"
    workflow.  Returns the ticket number so you can open it immediately
    to watch the live step comments as subsequent agents run.
    """
    _gc_sessions()

    session_id = req.session_id or req.service_name
    bridge_trace_id = uuid.uuid4().hex

    session = PipelineSession(
        session_id=session_id,
        service_name=req.service_name,
        alert_name=req.alert_name,
        severity=req.severity,
        summary=req.summary or req.alert_name,
        description=req.description,
        dashboard_url=req.dashboard_url,
        starts_at=req.starts_at or datetime.now(timezone.utc).isoformat(),
        bridge_trace_id=bridge_trace_id,
    )
    _sessions[session_id] = session

    # Create skeleton ticket immediately so subsequent agents can post comments
    skeleton_body = (
        f"## Automated Incident — AIOps Agent Pipeline\n\n"
        f"| Field | Value |\n|---|---|\n"
        f"| **Service** | `{req.service_name}` |\n"
        f"| **Alert** | `{req.alert_name}` |\n"
        f"| **Severity** | `{req.severity.upper()}` |\n"
        f"| **Detected at** | {session.starts_at} |\n"
        f"| **Dashboard** | [{req.dashboard_url}]({req.dashboard_url}) |\n"
        f"| **Pipeline ID** | `{session_id}` |\n"
        f"| **OTel Trace** | `{bridge_trace_id}` — paste in Grafana → Tempo |\n\n"
        f"*AI pipeline agents running — watch the activity feed for live step updates.*\n\n"
        f"*Simultaneously open **Scheduler → Workflows → AIOps Agent Pipeline** "
        f"to watch each agent node turn green on the workflow canvas.*"
    )
    notify = get_notify_list(req.severity)
    create_payload: dict[str, Any] = {
        "subject": (
            f"[AIOPS] {req.alert_name} on {req.service_name} "
            f"[{req.severity.upper()}]: {session.summary}"
        ),
        "body": skeleton_body,
        "type": "issue",
        "status": "open",
    }
    if notify:
        create_payload["notify"] = notify

    create_result = await _post("/api/app/create_ticket/v1", create_payload)
    if create_result.get("error"):
        logger.warning(
            "Failed to create skeleton ticket  session=%s  error=%s",
            session_id, create_result.get("error"),
        )
        return {
            "status": "error",
            "session_id": session_id,
            "agent": "pipeline-start",
            "message": f"Ticket creation failed: {create_result.get('error')}",
        }

    session.ticket_id = create_result.get("ticket", {}).get("id", "")
    session.ticket_num = create_result.get("ticket", {}).get("num", 0)
    session.stage = "started"

    logger.info(
        "Pipeline session started  session=%s  ticket=#%s (%s)  service=%s  alert=%s",
        session_id, session.ticket_num, session.ticket_id,
        req.service_name, req.alert_name,
    )
    await asyncio.sleep(WORKFLOW_STEP_DELAY)
    return {
        "status": "started",
        "session_id": session_id,
        "agent": "pipeline-start",
        "ticket_id": session.ticket_id,
        "ticket_num": session.ticket_num,
        "message": (
            f"Pipeline started: {req.service_name} / {req.alert_name} "
            f"[{req.severity.upper()}] — Ticket #{session.ticket_num} created. "
            f"Open it in xyOps to watch live [>>]/[OK] comments as agents run."
        ),
    }


# ── Agent 2: Loki Log Fetcher ──────────────────────────────────────────────────

@pipeline_router.post("/agent/logs")
async def agent_logs(req: AgentRequest) -> dict:
    """
    Agent 2 — Fetch the last 50 Loki log lines for the service.

    Posts [>>] started and [OK] done comments to the incident ticket.
    Stores log text in the session for Agent 4 (Claude) to consume.
    """
    session = _require_session(req.session_id)

    await post_step_comment(
        session.ticket_id, 1, "started",
        f"Fetching log context for **{session.service_name}** (last 50 lines)...",
        _post,
    )

    if _http:
        session.logs = await fetch_loki_logs(session.service_name, _http)

    log_lines = session.logs.count("\n") + (1 if session.logs.strip() else 0)
    warn_count = session.logs.upper().count("WARN")
    session.stage = "logs"

    await post_step_comment(
        session.ticket_id, 1, "done",
        f"Retrieved **{log_lines}** log lines ({warn_count} WARN events)",
        _post,
    )

    logger.info("Agent logs complete  session=%s  lines=%d", req.session_id, log_lines)
    await asyncio.sleep(WORKFLOW_STEP_DELAY)
    return {
        "status": "ok",
        "session_id": req.session_id,
        "agent": "loki-log-fetcher",
        "ticket_num": session.ticket_num,
        "log_lines": log_lines,
        "warn_count": warn_count,
        "message": (
            f"Fetched {log_lines} log lines ({warn_count} WARN) "
            f"for {session.service_name}"
        ),
    }


# ── Agent 3: Prometheus Analyst ────────────────────────────────────────────────

@pipeline_router.post("/agent/metrics")
async def agent_metrics(req: AgentRequest) -> dict:
    """
    Agent 3 — Fetch Prometheus golden signals for the service.

    Fetches: error_rate, p99_latency, p50_latency, RPS.
    Posts step comments and stores metrics in session for Agent 4.
    """
    session = _require_session(req.session_id)

    await post_step_comment(
        session.ticket_id, 2, "started",
        f"Fetching Prometheus golden signals for **{session.service_name}**...",
        _post,
    )

    if _http:
        session.metrics = await fetch_prometheus_context(session.service_name, _http)

    m_parts: list[str] = []
    if session.metrics.get("error_rate_pct") not in (None, "no data"):
        m_parts.append(f"error_rate={session.metrics['error_rate_pct']}%")
    if session.metrics.get("p99_latency_ms") not in (None, "no data"):
        m_parts.append(f"p99={session.metrics['p99_latency_ms']}ms")
    if session.metrics.get("rps") not in (None, "no data"):
        m_parts.append(f"rps={session.metrics['rps']}")
    metrics_str = "  ".join(m_parts) if m_parts else "no metrics available"
    session.stage = "metrics"

    await post_step_comment(
        session.ticket_id, 2, "done",
        f"Metrics snapshot: `{metrics_str}`",
        _post,
    )

    logger.info("Agent metrics complete  session=%s  %s", req.session_id, metrics_str)
    await asyncio.sleep(WORKFLOW_STEP_DELAY)
    return {
        "status": "ok",
        "session_id": req.session_id,
        "agent": "prometheus-analyst",
        "ticket_num": session.ticket_num,
        "metrics": session.metrics,
        "message": f"Golden signals for {session.service_name}: {metrics_str}",
    }


# ── Agent 4: Claude AI Analyst ─────────────────────────────────────────────────

@pipeline_router.post("/agent/analyze")
async def agent_analyze(req: AgentRequest) -> dict:
    """
    Agent 4 — Call Claude AI for root cause analysis.

    Requires CLAUDE_API_KEY to be set.  If not set, posts a [--] SKIPPED
    comment and returns ai_enabled=false so subsequent agents still run.
    Stores analysis dict in session (contains ansible_playbook, rca_summary,
    test_plan, rollback_steps, pr_description, confidence).
    """
    session = _require_session(req.session_id)

    if AI_ENABLED and _http:
        await post_step_comment(
            session.ticket_id, 3, "started",
            "Calling **Claude AI** (`claude-3-5-haiku-20241022`) for RCA...",
            _post,
        )
        session.analysis = await generate_ai_analysis(
            alert_name=session.alert_name,
            service_name=session.service_name,
            severity=session.severity,
            description=session.description,
            logs=session.logs,
            metrics=session.metrics,
            http=_http,
        )
        confidence = session.analysis.get("confidence", "?")
        rca_words = len(session.analysis.get("rca_summary", "").split())
        has_playbook = bool(session.analysis.get("ansible_playbook"))
        session.stage = "analyzed"

        await post_step_comment(
            session.ticket_id, 3, "done",
            f"RCA complete — confidence: **{confidence}** | "
            f"{rca_words}-word analysis | playbook: {'yes' if has_playbook else 'no'}",
            _post,
        )
        logger.info(
            "Agent analyze complete  session=%s  confidence=%s  playbook=%s",
            req.session_id, confidence, has_playbook,
        )
        await asyncio.sleep(WORKFLOW_STEP_DELAY)
        return {
            "status": "ok",
            "session_id": req.session_id,
            "agent": "claude-ai-analyst",
            "ticket_num": session.ticket_num,
            "ai_enabled": True,
            "confidence": confidence,
            "has_playbook": has_playbook,
            "rca_summary": session.analysis.get("rca_summary", ""),
            "message": (
                f"AI analysis: confidence={confidence}, "
                f"playbook={'yes' if has_playbook else 'no'}, "
                f"{rca_words} words of RCA"
            ),
        }
    else:
        await post_step_comment(
            session.ticket_id, 3, "skipped",
            "AI analysis — SKIPPED (`CLAUDE_API_KEY` not set — add key to enable)",
            _post,
        )
        session.stage = "analyzed"
        await asyncio.sleep(WORKFLOW_STEP_DELAY)
        return {
            "status": "ok",
            "session_id": req.session_id,
            "agent": "claude-ai-analyst",
            "ticket_num": session.ticket_num,
            "ai_enabled": False,
            "message": "AI skipped — CLAUDE_API_KEY not configured",
        }


# ── Agent 5: Incident Scribe ───────────────────────────────────────────────────

@pipeline_router.post("/agent/ticket")
async def agent_ticket(req: AgentRequest) -> dict:
    """
    Agent 5 — Replace the skeleton ticket body with the full enriched content.

    Uses all context accumulated by agents 2-4 to build the complete
    AI-enriched incident body (metrics table, RCA, Ansible playbook,
    test plan, rollback steps, GitHub PR suggestion).
    """
    session = _require_session(req.session_id)

    await post_step_comment(
        session.ticket_id, 4, "started",
        "Building full incident body with diagnostic context...",
        _post,
    )

    ticket_body = build_enriched_ticket_body(
        service_name=session.service_name,
        alert_name=session.alert_name,
        severity=session.severity,
        description=session.description,
        starts_at=session.starts_at,
        dashboard_url=session.dashboard_url,
        bridge_trace_id=session.bridge_trace_id,
        metrics=session.metrics,
        analysis=session.analysis,
    )

    await _post(
        "/api/app/update_ticket/v1",
        {"id": session.ticket_id, "body": ticket_body},
    )
    word_count = len(ticket_body.split())
    session.stage = "ticket_enriched"

    await post_step_comment(
        session.ticket_id, 4, "done",
        f"Incident body enriched — {word_count} words of AI diagnostic context",
        _post,
    )

    logger.info(
        "Agent ticket complete  session=%s  ticket=#%s  words=%d",
        req.session_id, session.ticket_num, word_count,
    )
    await asyncio.sleep(WORKFLOW_STEP_DELAY)
    return {
        "status": "ok",
        "session_id": req.session_id,
        "agent": "incident-scribe",
        "ticket_id": session.ticket_id,
        "ticket_num": session.ticket_num,
        "word_count": word_count,
        "message": (
            f"Ticket #{session.ticket_num} enriched with "
            f"{word_count} words of AI diagnostic context"
        ),
    }


# ── Agent 6: Approval Gateway ──────────────────────────────────────────────────

@pipeline_router.post("/agent/approval")
async def agent_approval(req: AgentRequest) -> dict:
    """
    Agent 6 — Create a human-approval gate ticket if warranted.

    Creates a second xyOps ticket of type 'change' containing:
      - Full RCA summary and confidence score
      - The proposed Ansible playbook YAML
      - Test plan and rollback steps
      - A curl command the approver can run to POST their decision

    If no playbook was generated (AI disabled or low confidence),
    posts a [--] N/A comment and returns cleanly.
    """
    session = _require_session(req.session_id)

    needs_approval = (
        REQUIRE_APPROVAL
        and session.severity in APPROVAL_SEVERITY_THRESHOLD
        and session.analysis.get("ansible_playbook")
        and session.ticket_id
        and _http
    )

    if needs_approval:
        await post_step_comment(
            session.ticket_id, 5, "started",
            "Creating **approval gate** ticket for human review...",
            _post,
        )
        session.approval_id = f"apr-{uuid.uuid4().hex[:12]}"
        approval_req = await request_approval(
            approval_id=session.approval_id,
            incident_ticket_id=session.ticket_id,
            alert_name=session.alert_name,
            service_name=session.service_name,
            severity=session.severity,
            analysis=session.analysis,
            bridge_trace_id=session.bridge_trace_id,
            xyops_post=_post,
            xyops_url=XYOPS_URL,
            http=_http,
        )
        session.approval_ticket_id = approval_req.approval_ticket_id
        session.approval_ticket_num = approval_req.approval_ticket_num
        session.stage = "awaiting_approval"

        await post_step_comment(
            session.ticket_id, 5, "waiting",
            f"Awaiting human approval — see **Ticket #{session.approval_ticket_num}**  \n"
            f"Approve: `POST /approval/{session.approval_id}/decision` "
            f"`{{\"approved\":true,\"decided_by\":\"your-name\"}}`",
            _post,
        )

        logger.info(
            "Agent approval complete  session=%s  approval_id=%s  approval_ticket=#%s",
            req.session_id, session.approval_id, session.approval_ticket_num,
        )
        await asyncio.sleep(WORKFLOW_STEP_DELAY)
        return {
            "status": "ok",
            "session_id": req.session_id,
            "agent": "approval-gateway",
            "ticket_num": session.ticket_num,
            "approval_id": session.approval_id,
            "approval_ticket_id": session.approval_ticket_id,
            "approval_ticket_num": session.approval_ticket_num,
            "message": (
                f"Approval gate: Ticket #{session.approval_ticket_num} created — "
                f"POST /approval/{session.approval_id}/decision to decide"
            ),
        }
    else:
        await post_step_comment(
            session.ticket_id, 5, "skipped",
            "Approval gate — N/A (no Ansible playbook or approval not required)",
            _post,
        )
        session.stage = "complete"
        logger.info(
            "Agent approval skipped  session=%s  reason=no_playbook_or_not_required",
            req.session_id,
        )
        await asyncio.sleep(WORKFLOW_STEP_DELAY)
        return {
            "status": "ok",
            "session_id": req.session_id,
            "agent": "approval-gateway",
            "ticket_num": session.ticket_num,
            "message": "No approval gate needed — pipeline complete",
        }


# ── Debug: inspect session state ───────────────────────────────────────────────

@pipeline_router.get("/session/{session_id}")
async def get_session(session_id: str) -> dict:
    """Return current session state (useful for debugging from the host)."""
    session = _require_session(session_id)
    return {
        "session_id": session.session_id,
        "service_name": session.service_name,
        "alert_name": session.alert_name,
        "severity": session.severity,
        "stage": session.stage,
        "ticket_id": session.ticket_id,
        "ticket_num": session.ticket_num,
        "approval_id": session.approval_id,
        "approval_ticket_num": session.approval_ticket_num,
        "has_logs": bool(session.logs),
        "has_metrics": bool(session.metrics),
        "has_analysis": bool(session.analysis),
        "created_at": session.created_at,
        "age_seconds": round(time.time() - session.created_at),
    }
