"""
UI Backend - FastAPI aggregator service

Consolidates data from compute-agent, storage-agent, obs-intelligence,
Gitea, and xyOps to provide a unified API for the React frontend.
"""
import asyncio
import os
import logging
import sqlite3
import time
import httpx
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# OpenTelemetry instrumentation
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.resources import Resource

from .models import (
    PipelineRunSummary,
    PipelineRunFull,
    PipelineState,
    HistoryFilters,
    ScenarioCard,
    ScenarioDetail,
    AutonomyStatusResponse,
    HealthResponse,
)
from .storage import (
    init_db,
    get_pipeline_history,
    get_pipeline_snapshot,
    get_all_scenarios,
    get_scenario_runs,
    seed_scenario_metadata,
    save_pipeline_run,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Environment variables
COMPUTE_AGENT_URL = os.getenv("COMPUTE_AGENT_URL", "http://compute-agent:9000")
STORAGE_AGENT_URL = os.getenv("STORAGE_AGENT_URL", "http://storage-agent:9001")
OBS_INTELLIGENCE_URL = os.getenv("OBS_INTELLIGENCE_URL", "http://obs-intelligence:9100")
GITEA_URL = os.getenv("GITEA_URL", "http://gitea:3000")
XYOPS_URL = os.getenv("XYOPS_URL", "http://xyops:5522")
XYOPS_API_KEY = os.getenv("XYOPS_API_KEY", "")
PIPELINE_SESSIONS_DB_PATH = os.getenv("PIPELINE_SESSIONS_DB_PATH", "/compute-data/pipeline.db")

# OpenTelemetry setup
resource = Resource.create({"service.name": "ui-backend"})
trace.set_tracer_provider(TracerProvider(resource=resource))
otlp_exporter = OTLPSpanExporter(
    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4317"),
    insecure=True
)
trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(otlp_exporter))

# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database and seed scenario metadata on startup"""
    logger.info("Starting UI Backend service")
    
    # Initialize database
    await init_db()
    
    # Seed scenario metadata (read from scenarios directory)
    scenarios = await load_scenario_metadata()
    if scenarios:
        await seed_scenario_metadata(scenarios)
        logger.info(f"Seeded {len(scenarios)} scenarios")
    
    yield
    
    logger.info("Shutting down UI Backend service")


# Create FastAPI app
app = FastAPI(
    title="AIOps UI Backend",
    description="Aggregator API for AIOps Command Center",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware - allow browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3005", "http://localhost:3500", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instrument FastAPI with OpenTelemetry
FastAPIInstrumentor.instrument_app(app)
HTTPXClientInstrumentor().instrument()


# ─────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────

async def load_scenario_metadata() -> List[dict]:
    """Load scenario metadata from configuration"""
    # In a real implementation, read from scenarios directory
    # For now, return hardcoded 20 scenarios
    compute_scenarios = [
        {
            "scenario_id": "HighCPUUsage",
            "display_name": "High CPU Usage",
            "domain": "compute",
            "autonomy_badge": "autonomous",
            "action": "Scale horizontally",
            "confidence_threshold": 0.85,
        },
        {
            "scenario_id": "HighMemoryUsage",
            "display_name": "High Memory Usage",
            "domain": "compute",
            "autonomy_badge": "approval_gated",
            "action": "Increase memory limit",
            "confidence_threshold": 0.80,
        },
        {
            "scenario_id": "ServiceUnhealthy",
            "display_name": "Service Unhealthy",
            "domain": "compute",
            "autonomy_badge": "approval_gated",
            "action": "Restart service",
            "confidence_threshold": 0.90,
        },
        {
            "scenario_id": "HighErrorRate",
            "display_name": "High Error Rate",
            "domain": "compute",
            "autonomy_badge": "human_only",
            "action": "Investigate logs",
            "confidence_threshold": 0.75,
        },
        {
            "scenario_id": "HighLatency",
            "display_name": "High Latency",
            "domain": "compute",
            "autonomy_badge": "autonomous",
            "action": "Scale workers",
            "confidence_threshold": 0.80,
        },
        {
            "scenario_id": "MemoryLeak",
            "display_name": "Memory Leak Detected",
            "domain": "compute",
            "autonomy_badge": "approval_gated",
            "action": "Restart with heap dump",
            "confidence_threshold": 0.85,
        },
        {
            "scenario_id": "DiskFull",
            "display_name": "Disk Full",
            "domain": "compute",
            "autonomy_badge": "approval_gated",
            "action": "Clean logs and expand volume",
            "confidence_threshold": 0.95,
        },
        {
            "scenario_id": "NetworkSaturation",
            "display_name": "Network Saturation",
            "domain": "compute",
            "autonomy_badge": "human_only",
            "action": "Investigate network topology",
            "confidence_threshold": 0.70,
        },
        {
            "scenario_id": "PodCrashLoop",
            "display_name": "Pod Crash Loop",
            "domain": "compute",
            "autonomy_badge": "approval_gated",
            "action": "Rollback deployment",
            "confidence_threshold": 0.90,
        },
        {
            "scenario_id": "DatabaseConnectionPoolExhausted",
            "display_name": "DB Connection Pool Exhausted",
            "domain": "compute",
            "autonomy_badge": "autonomous",
            "action": "Increase pool size",
            "confidence_threshold": 0.85,
        },
    ]
    
    storage_scenarios = [
        {
            "scenario_id": "CephOSDDown",
            "display_name": "Ceph OSD Down",
            "domain": "storage",
            "autonomy_badge": "approval_gated",
            "action": "Reweight OSD",
            "confidence_threshold": 0.90,
        },
        {
            "scenario_id": "CephPoolFull",
            "display_name": "Ceph Pool Full",
            "domain": "storage",
            "autonomy_badge": "approval_gated",
            "action": "Expand pool",
            "confidence_threshold": 0.95,
        },
        {
            "scenario_id": "CephPoolNearFull",
            "display_name": "Ceph Pool Near Full",
            "domain": "storage",
            "autonomy_badge": "autonomous",
            "action": "Expand pool preemptively",
            "confidence_threshold": 0.85,
        },
        {
            "scenario_id": "CephMultipleOSDDown",
            "display_name": "Multiple Ceph OSDs Down",
            "domain": "storage",
            "autonomy_badge": "human_only",
            "action": "Escalate to SRE",
            "confidence_threshold": 0.95,
        },
        {
            "scenario_id": "PVCHighLatency",
            "display_name": "PVC High Latency",
            "domain": "storage",
            "autonomy_badge": "approval_gated",
            "action": "Investigate IO bottleneck",
            "confidence_threshold": 0.80,
        },
        {
            "scenario_id": "NoisyPVCDetected",
            "display_name": "Noisy PVC Detected",
            "domain": "storage",
            "autonomy_badge": "autonomous",
            "action": "Throttle IO",
            "confidence_threshold": 0.85,
        },
        {
            "scenario_id": "CephSlowOps",
            "display_name": "Ceph Slow Ops",
            "domain": "storage",
            "autonomy_badge": "approval_gated",
            "action": "Rebalance OSDs",
            "confidence_threshold": 0.75,
        },
        {
            "scenario_id": "StorageQuotaExceeded",
            "display_name": "Storage Quota Exceeded",
            "domain": "storage",
            "autonomy_badge": "approval_gated",
            "action": "Increase quota",
            "confidence_threshold": 0.90,
        },
        {
            "scenario_id": "BackupFailure",
            "display_name": "Backup Failure",
            "domain": "storage",
            "autonomy_badge": "human_only",
            "action": "Verify backup infrastructure",
            "confidence_threshold": 0.85,
        },
        {
            "scenario_id": "SnapshotDeletionPending",
            "display_name": "Snapshot Deletion Pending",
            "domain": "storage",
            "autonomy_badge": "autonomous",
            "action": "Clean old snapshots",
            "confidence_threshold": 0.80,
        },
    ]
    
    return compute_scenarios + storage_scenarios


async def fetch_json(url: str, headers: dict = None, timeout: float = 10.0):
    """Fetch JSON from a URL with error handling"""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, headers=headers or {})
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        logger.error(f"HTTP error fetching {url}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error fetching {url}: {e}")
        return None


def _read_value_metrics() -> dict:
    cutoff = time.time() - (30 * 24 * 60 * 60)
    with sqlite3.connect(PIPELINE_SESSIONS_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        def scalar(query: str, params: tuple = ()) -> float:
            row = conn.execute(query, params).fetchone()
            value = row[0] if row and row[0] is not None else 0
            return float(value)

        avg_mttr_automated = scalar(
            "SELECT AVG(mttr_seconds) FROM pipeline_sessions WHERE autonomy_decision = 'AUTONOMOUS' AND mttr_seconds > 0"
        )
        avg_mttr_manual = scalar(
            "SELECT AVG(mttr_seconds) FROM pipeline_sessions WHERE autonomy_decision != 'AUTONOMOUS' AND mttr_seconds > 0"
        )
        automated_count = scalar(
            "SELECT COUNT(*) FROM pipeline_sessions WHERE autonomy_decision = 'AUTONOMOUS' AND outcome = 'success'"
        )
        manual_count = scalar(
            "SELECT COUNT(*) FROM pipeline_sessions WHERE autonomy_decision != 'AUTONOMOUS'"
        )
        incidents_last_30d = scalar(
            "SELECT COUNT(*) FROM pipeline_sessions WHERE created_at > ?",
            (cutoff,),
        )
        time_saved_minutes = max(avg_mttr_manual - avg_mttr_automated, 0.0) * automated_count / 60.0

        return {
            "avg_mttr_automated": round(avg_mttr_automated, 1),
            "avg_mttr_manual": round(avg_mttr_manual, 1),
            "automated_count": int(automated_count),
            "manual_count": int(manual_count),
            "incidents_last_30d": int(incidents_last_30d),
            "time_saved_minutes": round(time_saved_minutes, 1),
        }


# ─────────────────────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    # Check connectivity to backend services
    services = {}
    
    compute_health = await fetch_json(f"{COMPUTE_AGENT_URL}/health", timeout=5.0)
    services["compute-agent"] = "healthy" if compute_health else "unhealthy"
    
    storage_health = await fetch_json(f"{STORAGE_AGENT_URL}/health", timeout=5.0)
    services["storage-agent"] = "healthy" if storage_health else "unhealthy"
    
    obs_health = await fetch_json(f"{OBS_INTELLIGENCE_URL}/health", timeout=5.0)
    services["obs-intelligence"] = "healthy" if obs_health else "unhealthy"
    
    return HealthResponse(status="healthy", services=services)


# ─────────────────────────────────────────────────────────────
# Section 1.1: Live Pipeline State
# ─────────────────────────────────────────────────────────────

@app.get("/pipeline/session/{session_id}", response_model=PipelineState)
async def get_live_session(session_id: str, domain: str = "compute"):
    """Get current state of a live pipeline session"""
    agent_url = COMPUTE_AGENT_URL if domain == "compute" else STORAGE_AGENT_URL
    
    data = await fetch_json(f"{agent_url}/pipeline/session/{session_id}")
    if not data:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return PipelineState(**data)


@app.get("/pipeline/active", response_model=List[str])
async def get_active_sessions():
    """Get list of active session IDs across all domains"""
    sessions = []
    
    # Query compute agent
    compute_data = await fetch_json(f"{COMPUTE_AGENT_URL}/pipeline/active")
    if compute_data:
        sessions.extend(compute_data.get("sessions", []))
    
    # Query storage agent
    storage_data = await fetch_json(f"{STORAGE_AGENT_URL}/pipeline/active")
    if storage_data:
        sessions.extend(storage_data.get("sessions", []))
    
    return sessions


# ─────────────────────────────────────────────────────────────
# Section 1.2: Pipeline History & Playback
# ─────────────────────────────────────────────────────────────

@app.get("/pipeline/history", response_model=List[PipelineRunSummary])
async def get_history(
    domain: Optional[str] = Query(None),
    scenario: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    autonomy_decision: Optional[str] = Query(None),
    outcome: Optional[str] = Query(None),
):
    """Get pipeline execution history with optional filters"""
    filters = HistoryFilters(
        domain=domain,
        scenario=scenario,
        start_date=start_date,
        end_date=end_date,
        autonomy_decision=autonomy_decision,
        outcome=outcome,
    )
    
    history = await get_pipeline_history(filters)
    return history


@app.get("/pipeline/session/{session_id}/snapshot", response_model=PipelineRunFull)
async def get_session_snapshot(session_id: str):
    """Get full historical snapshot for playback mode"""
    snapshot = await get_pipeline_snapshot(session_id)
    if not snapshot:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    
    return snapshot


@app.get("/pipeline/session/{session_id}/risk-history")
async def get_risk_history(session_id: str, domain: str = "compute"):
    """
    Returns a risk score time-series for the given pipeline session.

    Queries Prometheus for the rate of compute_agent_actions_total over
    the session's time window.  This gives a real measure of agent activity
    pressure — a useful proxy for operational risk.

    Falls back to a flat line at the stored risk_score if Prometheus is
    unreachable or returns no data, so the sparkline always has something to show.
    """
    import time as _time

    PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")

    # Fetch the live session to learn service_name and created_at
    agent_url = COMPUTE_AGENT_URL if domain == "compute" else STORAGE_AGENT_URL
    session_data = await fetch_json(f"{agent_url}/pipeline/session/{session_id}")

    risk_score = 0.5
    start_ts = None
    if session_data:
        risk_score = float(session_data.get("risk_score") or 0.5)
        created_at = session_data.get("created_at")
        if created_at:
            start_ts = int(float(created_at))

    now_ts = int(_time.time())
    window_start = start_ts if start_ts else now_ts - 1800   # 30 min default
    window_start = max(window_start, now_ts - 3600)           # max 60 min lookback
    step = max(60, (now_ts - window_start) // 20)             # ~20 data points

    series = []
    source = "fallback"

    try:
        prom_url = (
            f"{PROMETHEUS_URL}/api/v1/query_range"
            f"?query=rate(compute_agent_actions_total[2m])"
            f"&start={window_start}&end={now_ts}&step={step}"
        )
        prom_data = await fetch_json(prom_url, timeout=5.0)
        if prom_data and prom_data.get("status") == "success":
            results = prom_data.get("data", {}).get("result", [])
            # Sum all action_type series per timestamp
            ts_map: dict = {}
            for r in results:
                for ts, val in r.get("values", []):
                    ts_map[float(ts)] = ts_map.get(float(ts), 0.0) + float(val)
            if ts_map:
                max_val = max(ts_map.values()) or 1.0
                series = [
                    {"t": i + 1, "risk": round((v / max_val) * 100, 1)}
                    for i, (_, v) in enumerate(sorted(ts_map.items()))
                ]
                source = "prometheus"
    except Exception as e:
        logger.warning(f"Prometheus risk-history query failed: {e}")

    if not series:
        base = round(risk_score * 100, 1)
        series = [{"t": i + 1, "risk": base} for i in range(15)]
        source = "fallback"

    return {"session_id": session_id, "series": series, "source": source}
    """Get all scenarios with aggregated statistics"""
    scenarios = await get_all_scenarios()
    
    # Filter by domain if specified
    if domain:
        scenarios = [s for s in scenarios if s.domain == domain]
    
    return scenarios


@app.get("/scenarios/{scenario_id}", response_model=ScenarioDetail)
async def get_scenario_detail(scenario_id: str, limit: int = Query(20, le=100)):
    """Get full scenario details with historical runs"""
    # Get scenario metadata
    all_scenarios = await get_all_scenarios()
    scenario_card = next((s for s in all_scenarios if s.scenario_id == scenario_id), None)
    
    if not scenario_card:
        raise HTTPException(status_code=404, detail="Scenario not found")
    
    # Get historical runs
    runs = await get_scenario_runs(scenario_id, limit)
    historical_runs = [
        {"session_id": r.session_id, "timestamp": r.timestamp, "outcome": r.outcome}
        for r in runs
    ]
    
    # Build YAML conditions (placeholder - would read from scenario files)
    yaml_conditions = [
        {
            "field": "cpu_usage_percent",
            "operator": ">",
            "value": 80,
            "description": "CPU usage exceeds 80%"
        },
        {
            "field": "cpu_trend",
            "operator": "==",
            "value": "increasing",
            "description": "CPU trend is increasing"
        },
    ]
    
    return ScenarioDetail(
        scenario_id=scenario_card.scenario_id,
        display_name=scenario_card.display_name,
        domain=scenario_card.domain,
        autonomy_badge=scenario_card.autonomy_badge,
        yaml_conditions=yaml_conditions,
        rca_template="High CPU usage detected. Investigate recent deployments.",
        playbook_hint="Scale horizontally: kubectl scale deployment --replicas=+1",
        historical_runs=historical_runs,
        statistics=scenario_card,
    )


# ─────────────────────────────────────────────────────────────
# Autonomy Status
# ─────────────────────────────────────────────────────────────

@app.get("/autonomy/status", response_model=AutonomyStatusResponse)
async def get_autonomy_status():
    """Get current autonomy tier status for all services"""
    # Query compute agent for autonomy status
    data = await fetch_json(f"{COMPUTE_AGENT_URL}/autonomy/status")
    
    if not data:
        return AutonomyStatusResponse(services=[])
    
    return AutonomyStatusResponse(**data)


@app.get("/autonomy/status/{service_name}")
async def get_autonomy_status_for_service(service_name: str):
    """Proxy service-level autonomy status with trust metrics from compute-agent."""
    data = await fetch_json(f"{COMPUTE_AGENT_URL}/autonomy/status/{service_name}")
    if not data:
        raise HTTPException(status_code=404, detail="Service autonomy status unavailable")
    return data


@app.get("/metrics/value")
async def get_value_metrics():
    """Return management-facing MTTR and automation value metrics."""
    try:
        return await asyncio.to_thread(_read_value_metrics)
    except Exception as e:
        logger.error(f"Error reading value metrics: {e}")
        return {
            "avg_mttr_automated": 0.0,
            "avg_mttr_manual": 0.0,
            "automated_count": 0,
            "manual_count": 0,
            "incidents_last_30d": 0,
            "time_saved_minutes": 0.0,
        }


@app.get("/intelligence/scenario-stats")
async def get_intelligence_scenario_stats():
    data = await fetch_json(f"{OBS_INTELLIGENCE_URL}/intelligence/scenario-stats")
    return data or {"status": "ok", "scenarios": []}


@app.get("/intelligence/knowledge-entries")
async def get_knowledge_entries(service_name: str = "", scenario_id: str = "", limit: int = 50):
    query = []
    if service_name:
        query.append(f"service_name={service_name}")
    if scenario_id:
        query.append(f"scenario_id={scenario_id}")
    query.append(f"limit={limit}")
    suffix = f"?{'&'.join(query)}" if query else ""
    data = await fetch_json(f"{OBS_INTELLIGENCE_URL}/intelligence/knowledge-entries{suffix}")
    return data or {"entries": []}


@app.get("/intelligence/learning-stats")
async def get_learning_stats():
    """Graceful proxy for Learning tab while Block F is not yet fully deployed."""
    data = await fetch_json(f"{OBS_INTELLIGENCE_URL}/intelligence/learning-stats")
    if data:
        return data

    scenario_stats = await fetch_json(f"{OBS_INTELLIGENCE_URL}/intelligence/scenario-stats")
    scenarios = (scenario_stats or {}).get("scenarios", [])
    return {
        "external_llm_calls_30d": 0,
        "local_validation_attempts_30d": 0,
        "local_validation_completed_30d": 0,
        "corroborated_count_30d": 0,
        "weak_support_count_30d": 0,
        "divergent_count_30d": 0,
        "insufficient_context_count_30d": 0,
        "avg_top_similarity_30d": 0.0,
        "knowledge_entries_total": 0,
        "knowledge_entries_with_success_outcome": 0,
        "local_validation_coverage_pct": 0.0,
        "corroboration_rate_pct": 0.0,
        "weekly_hit_rate": [],
        "scenario_count": len(scenarios),
    }


# ─────────────────────────────────────────────────────────────
# Admin Endpoints (for testing)
# ─────────────────────────────────────────────────────────────

@app.post("/admin/save_run")
async def save_run(run: PipelineRunFull):
    """Save a pipeline run manually (for testing)"""
    try:
        await save_pipeline_run(run.summary, run)
        return {"status": "saved", "session_id": run.session_id}
    except Exception as e:
        logger.error(f"Error saving run: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9005)
