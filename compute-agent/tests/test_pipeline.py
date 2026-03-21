"""
tests/test_pipeline.py
─────────────────────────────────────────────────────────────────────────────
Integration tests for pipeline.py agent endpoints via the FastAPI test client.

All external calls (xyOps API, Loki, Prometheus, OpenAI) are mocked.
Tests verify:
  - Each endpoint returns {status: "ok" | "started"} for xyOps success_match
  - Session state is correctly propagated between agents
  - Error handling: missing session → 404, bad xyOps ticket creation → error
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _ticket_post_mock(ticket_id="t_test_001", num=42):
    """Mock _post that returns a successful ticket creation response."""
    async def _mock(path, body):
        if "create_ticket" in path:
            return {"code": 0, "ticket": {"id": ticket_id, "num": num}}
        if "add_ticket_change" in path:
            return {"code": 0}
        if "update_ticket" in path:
            return {"code": 0}
        if "create_ticket" in path:
            return {"code": 0, "ticket": {"id": ticket_id, "num": num + 1}}
        return {"code": 0}
    return _mock


def _loki_mock_http(lines=None):
    """Mock httpx client with Loki returning log lines."""
    lines = lines or ["INFO request ok", "WARN high latency", "ERROR payment failed"]
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "data": {
            "result": [
                {"stream": {}, "values": [["1", line] for line in lines]}
            ]
        }
    }
    mock_http = AsyncMock()
    mock_http.get = AsyncMock(return_value=resp)
    return mock_http


MOCK_ANALYSIS = {
    "rca_summary": "DB connection pool exhausted under load.",
    "rca_detail": {"symptoms": [], "probable_cause": "pool exhausted",
                   "contributing_factors": [], "blast_radius": "all users"},
    "confidence": "high",
    "ansible_playbook": "---\n- name: fix\n  hosts: all\n  tasks: []",
    "ansible_description": "Restart payment service",
    "pr_description": "Fix pool size",
    "pr_title": "fix: increase pool",
    "test_plan": ["Step 1: dry run"],
    "estimated_fix_time_minutes": 5,
    "rollback_steps": ["Step 1: revert"],
}


class _FakeEnrichment:
    provider = "test-local"

    def __init__(self, analysis: dict):
        self._analysis = analysis

    def to_analysis_dict(self):
        return dict(self._analysis)


# ═══════════════════════════════════════════════════════════════════════════════
# Session lifecycle helpers
# ═══════════════════════════════════════════════════════════════════════════════

async def _start_session(session_id="frontend-api"):
    """Create a pipeline session directly via the module (bypasses HTTP)."""
    from app import pipeline
    from app.pipeline import PipelineSession
    import time
    s = PipelineSession(
        session_id=session_id,
        service_name="frontend-api",
        alert_name="HighErrorRate",
        severity="warning",
        summary="High error rate",
        description="Error rate exceeded 5%",
        dashboard_url="http://grafana:3000",
        starts_at="2026-03-17T10:00:00Z",
        bridge_trace_id="trace_abc123",
        ticket_id="t_test_001",
        ticket_num=42,
    )
    pipeline._sessions[session_id] = s
    return s


# ═══════════════════════════════════════════════════════════════════════════════
# Agent 1: /pipeline/start
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
class TestPipelineStart:

    async def test_returns_started_status(self):
        from app import pipeline
        pipeline._http = _loki_mock_http()
        pipeline._xyops_post_fn = _ticket_post_mock()
        from httpx import AsyncClient, ASGITransport
        from app.main import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/pipeline/start", json={
                "service_name": "frontend-api",
                "alert_name": "HighErrorRate",
                "severity": "warning",
                "summary": "Test",
                "description": "Testing",
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "started"

    async def test_returns_ticket_num(self):
        from app import pipeline
        pipeline._http = _loki_mock_http()
        pipeline._xyops_post_fn = _ticket_post_mock(num=77)
        from httpx import AsyncClient, ASGITransport
        from app.main import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/pipeline/start", json={
                "service_name": "frontend-api",
                "alert_name": "HighErrorRate",
                "severity": "warning",
            })
        assert resp.json()["ticket_num"] == 77

    async def test_creates_session_in_store(self):
        from app import pipeline
        pipeline._sessions.clear()
        pipeline._xyops_post_fn = _ticket_post_mock()
        from httpx import AsyncClient, ASGITransport
        from app.main import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post("/pipeline/start", json={
                "service_name": "test-svc-unique",
                "alert_name": "TestAlert",
                "severity": "warning",
            })
        assert "test-svc-unique" in pipeline._sessions

    async def test_returns_error_when_ticket_creation_fails(self):
        from app import pipeline
        async def _failing_post(path, body):
            if "create_ticket" in path:
                return {"error": "xyops unavailable"}
            return {"code": 0}
        pipeline._xyops_post_fn = _failing_post
        from httpx import AsyncClient, ASGITransport
        from app.main import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/pipeline/start", json={
                "service_name": "fail-svc",
                "alert_name": "TestAlert",
                "severity": "warning",
            })
        assert resp.status_code == 200
        assert resp.json()["status"] == "error"


# ═══════════════════════════════════════════════════════════════════════════════
# Agent 2: /pipeline/agent/logs
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
class TestAgentLogs:

    async def test_returns_ok_status(self):
        from app import pipeline
        await _start_session("frontend-api")
        pipeline._http = _loki_mock_http(["WARN line1", "WARN line2"])
        pipeline._xyops_post_fn = AsyncMock(return_value={"code": 0})
        from httpx import AsyncClient, ASGITransport
        from app.main import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/pipeline/agent/logs",
                                     json={"session_id": "frontend-api"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    async def test_returns_log_line_count(self):
        from app import pipeline
        await _start_session("frontend-api")
        pipeline._http = _loki_mock_http(["line1", "line2", "line3"])
        pipeline._xyops_post_fn = AsyncMock(return_value={"code": 0})
        from httpx import AsyncClient, ASGITransport
        from app.main import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/pipeline/agent/logs",
                                     json={"session_id": "frontend-api"})
        assert resp.json()["log_lines"] == 3

    async def test_returns_404_for_unknown_session(self):
        from httpx import AsyncClient, ASGITransport
        from app.main import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/pipeline/agent/logs",
                                     json={"session_id": "no-such-session-xyz"})
        assert resp.status_code == 404

    async def test_stores_logs_in_session(self):
        from app import pipeline
        pipeline._sessions.clear()
        await _start_session("frontend-api")
        pipeline._http = _loki_mock_http(["INFO foo", "WARN bar"])
        pipeline._xyops_post_fn = AsyncMock(return_value={"code": 0})
        from httpx import AsyncClient, ASGITransport
        from app.main import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post("/pipeline/agent/logs",
                              json={"session_id": "frontend-api"})
        assert pipeline._sessions["frontend-api"].logs != ""


# ═══════════════════════════════════════════════════════════════════════════════
# Agent 3: /pipeline/agent/metrics
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
class TestAgentMetrics:

    def _prom_http(self, value=10.0):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "data": {"result": [{"value": ["1", str(value)]}]}
        }
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=resp)
        return mock_http

    async def test_returns_ok_status(self):
        from app import pipeline
        await _start_session("frontend-api")
        pipeline._http = self._prom_http()
        pipeline._xyops_post_fn = AsyncMock(return_value={"code": 0})
        from httpx import AsyncClient, ASGITransport
        from app.main import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/pipeline/agent/metrics",
                                     json={"session_id": "frontend-api"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    async def test_stores_metrics_in_session(self):
        from app import pipeline
        pipeline._sessions.clear()
        await _start_session("frontend-api")
        pipeline._http = self._prom_http(5.5)
        pipeline._xyops_post_fn = AsyncMock(return_value={"code": 0})
        from httpx import AsyncClient, ASGITransport
        from app.main import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post("/pipeline/agent/metrics",
                              json={"session_id": "frontend-api"})
        assert pipeline._sessions["frontend-api"].metrics != {}


# ═══════════════════════════════════════════════════════════════════════════════
# Agent 4: /pipeline/agent/analyze
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
class TestAgentAnalyze:

    async def test_returns_ok_with_ai_enabled(self):
        from app import pipeline, ai_analyst
        await _start_session("frontend-api")
        pipeline._xyops_post_fn = AsyncMock(return_value={"code": 0})
        pipeline._http = _loki_mock_http()
        with patch.object(ai_analyst, "generate_ai_analysis",
                          AsyncMock(return_value=MOCK_ANALYSIS)):
            from httpx import AsyncClient, ASGITransport
            from app.main import app
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.post("/pipeline/agent/analyze",
                                         json={"session_id": "frontend-api"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    async def test_returns_correct_confidence(self):
        from app import pipeline
        await _start_session("frontend-api")
        pipeline._xyops_post_fn = AsyncMock(return_value={"code": 0})
        pipeline._http = _loki_mock_http()
        fake_analysis = {
            **MOCK_ANALYSIS,
            "confidence": "0.91",
        }
        with patch("app.pipeline._llm_enrich",
                   AsyncMock(return_value=_FakeEnrichment(fake_analysis))), \
             patch("app.pipeline._recommend") as mock_recommend:
            mock_recommend.return_value = type("Rec", (), {
                "action_type": "restart_service",
                "confidence": 0.91,
                "autonomous": False,
                "display_name": "Restart service",
                "rollback_plan": "rollback",
                "description": "restart",
                "estimated_duration": "5 minutes",
            })()
            from httpx import AsyncClient, ASGITransport
            from app.main import app
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.post("/pipeline/agent/analyze",
                                         json={"session_id": "frontend-api"})
        assert resp.json()["confidence"] == "0.91"

    async def test_returns_ok_when_ai_disabled(self):
        from app import pipeline
        await _start_session("frontend-api")
        pipeline._xyops_post_fn = AsyncMock(return_value={"code": 0})
        # Patch AI_ENABLED in the pipeline module where it's bound
        with patch("app.pipeline.AI_ENABLED", False):
            from httpx import AsyncClient, ASGITransport
            from app.main import app
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.post("/pipeline/agent/analyze",
                                         json={"session_id": "frontend-api"})
        assert resp.json()["status"] == "ok"
        assert resp.json()["ai_enabled"] is False

    async def test_stores_analysis_in_session(self):
        from app import pipeline
        pipeline._sessions.clear()
        await _start_session("frontend-api")
        pipeline._xyops_post_fn = AsyncMock(return_value={"code": 0})
        pipeline._http = _loki_mock_http()
        with patch("app.pipeline._llm_enrich",
                   AsyncMock(return_value=_FakeEnrichment(MOCK_ANALYSIS))):
            from httpx import AsyncClient, ASGITransport
            from app.main import app
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                await client.post("/pipeline/agent/analyze",
                                  json={"session_id": "frontend-api"})
        analysis = pipeline._sessions["frontend-api"].analysis
        assert analysis["rca_summary"] == MOCK_ANALYSIS["rca_summary"]
        assert analysis["confidence"] == MOCK_ANALYSIS["confidence"]
        assert "risk_score" in analysis
        assert "evidence_lines" in analysis


# ═══════════════════════════════════════════════════════════════════════════════
# Agent 5: /pipeline/agent/ticket
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
class TestAgentTicket:

    async def test_returns_ok_status(self):
        from app import pipeline
        s = await _start_session("frontend-api")
        s.analysis = MOCK_ANALYSIS
        pipeline._xyops_post_fn = AsyncMock(return_value={"code": 0})
        from httpx import AsyncClient, ASGITransport
        from app.main import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/pipeline/agent/ticket",
                                     json={"session_id": "frontend-api"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    async def test_returns_word_count(self):
        from app import pipeline
        s = await _start_session("frontend-api")
        s.analysis = MOCK_ANALYSIS
        pipeline._xyops_post_fn = AsyncMock(return_value={"code": 0})
        from httpx import AsyncClient, ASGITransport
        from app.main import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/pipeline/agent/ticket",
                                     json={"session_id": "frontend-api"})
        assert resp.json()["word_count"] > 10

    async def test_calls_update_ticket_api(self):
        from app import pipeline
        s = await _start_session("frontend-api")
        s.analysis = MOCK_ANALYSIS
        mock_post = AsyncMock(return_value={"code": 0})
        pipeline._xyops_post_fn = mock_post
        from httpx import AsyncClient, ASGITransport
        from app.main import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post("/pipeline/agent/ticket",
                              json={"session_id": "frontend-api"})
        update_calls = [c for c in mock_post.call_args_list
                        if "update_ticket" in c[0][0]]
        assert len(update_calls) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Agent 6: /pipeline/agent/approval
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
class TestAgentApproval:

    async def test_skips_when_no_playbook(self):
        from app import pipeline
        s = await _start_session("frontend-api")
        s.analysis = {}  # no ansible_playbook
        pipeline._xyops_post_fn = AsyncMock(return_value={"code": 0})
        from httpx import AsyncClient, ASGITransport
        from app.main import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/pipeline/agent/approval",
                                     json={"session_id": "frontend-api"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        # No approval_id when skipped
        assert resp.json().get("approval_id", "") == ""

    async def test_creates_approval_when_playbook_exists(self):
        from app import pipeline, approval_workflow
        s = await _start_session("frontend-api")
        s.analysis = MOCK_ANALYSIS
        s.severity = "warning"

        async def _mock_request_approval(**kwargs):
            from app.approval_workflow import ApprovalRequest
            req = ApprovalRequest(
                approval_id=kwargs["approval_id"],
                session_id=kwargs["session_id"],
                incident_ticket_id=kwargs["incident_ticket_id"],
                alert_name="HighErrorRate",
                service_name="frontend-api",
                severity="warning",
                ansible_playbook="---",
                ansible_description="test",
                test_plan=[],
                rca_summary="rca",
                bridge_trace_id="trace",
                action_type="restart_service",
                env_tier="production",
                approval_ticket_id="t_apr_001",
                approval_ticket_num=99,
            )
            return req

        pipeline._xyops_post_fn = AsyncMock(return_value={"code": 0})
        pipeline._http = _loki_mock_http()
        # Patch in pipeline module where the name is bound
        with patch("app.pipeline.request_approval",
                   AsyncMock(side_effect=_mock_request_approval)):
            from httpx import AsyncClient, ASGITransport
            from app.main import app
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.post("/pipeline/agent/approval",
                                         json={"session_id": "frontend-api"})
        data = resp.json()
        assert data["status"] == "ok"
        assert data["approval_ticket_num"] == 99


# ═══════════════════════════════════════════════════════════════════════════════
# GET /pipeline/session/{id}
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
class TestGetSession:

    async def test_returns_session_state(self):
        from app import pipeline
        await _start_session("frontend-api")
        from httpx import AsyncClient, ASGITransport
        from app.main import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/pipeline/session/frontend-api")
        assert resp.status_code == 200
        data = resp.json()
        assert data["service_name"] == "frontend-api"
        assert data["alert_name"] == "HighErrorRate"

    async def test_returns_404_for_missing_session(self):
        from httpx import AsyncClient, ASGITransport
        from app.main import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/pipeline/session/does-not-exist-xyz")
        assert resp.status_code == 404
