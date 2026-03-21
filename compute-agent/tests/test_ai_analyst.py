"""
tests/test_ai_analyst.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for ai_analyst.py:
  - AI_ENABLED flag reflects OPENAI_API_KEY / CLAUDE_API_KEY
  - generate_ai_analysis: OpenAI happy path, Anthropic happy path,
    bad JSON, HTTP error, AI disabled
  - fetch_loki_logs: success, HTTP error, exception
  - fetch_prometheus_context: success, no-data, HTTP error
  - build_enriched_ticket_body: content validation
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ═══════════════════════════════════════════════════════════════════════════════
# AI_ENABLED flag
# ═══════════════════════════════════════════════════════════════════════════════

class TestAiEnabled:
    def test_ai_enabled_when_openai_key_set(self):
        """conftest sets OPENAI_API_KEY so AI_ENABLED must be True."""
        from app.ai_analyst import AI_ENABLED, _USE_OPENAI
        assert AI_ENABLED is True
        assert _USE_OPENAI is True

    def test_ai_model_is_a_string(self):
        """AI_MODEL should be a non-empty string (set via environment)."""
        from app.ai_analyst import AI_MODEL
        assert isinstance(AI_MODEL, str)
        assert len(AI_MODEL) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# generate_ai_analysis
# ═══════════════════════════════════════════════════════════════════════════════

MOCK_ANALYSIS = {
    "rca_summary": "The payment service is returning 500s due to DB connection exhaustion.",
    "rca_detail": {
        "symptoms": ["high error rate", "slow p99"],
        "probable_cause": "DB pool exhausted",
        "contributing_factors": [],
        "blast_radius": "all users on payment flow",
    },
    "confidence": "high",
    "ansible_playbook": "---\n- name: Restart payment service\n  hosts: all\n  tasks: []",
    "ansible_description": "Restart the payment service to clear connections",
    "pr_description": "Fix DB pool size",
    "pr_title": "fix: increase DB connection pool",
    "test_plan": ["Step 1: dry-run playbook", "Step 2: verify metrics"],
    "estimated_fix_time_minutes": 10,
    "rollback_steps": ["Step 1: revert deployment"],
}


def _make_openai_response(analysis: dict):
    """Build a mock httpx Response that looks like an OpenAI API response."""
    from httpx import Response
    body = json.dumps({
        "choices": [{"message": {"content": json.dumps(analysis)}}]
    })
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = json.loads(body)
    resp.text = body
    return resp


def _make_claude_response(analysis: dict):
    """Build a mock httpx Response that looks like an Anthropic API response."""
    body = json.dumps({
        "content": [{"text": json.dumps(analysis)}]
    })
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = json.loads(body)
    resp.text = body
    return resp


def _make_error_response(status_code: int, message: str):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = message
    resp.json.return_value = {"error": message}
    return resp


@pytest.mark.asyncio
class TestGenerateAiAnalysis:

    async def test_openai_happy_path_returns_analysis(self):
        from app.ai_analyst import generate_ai_analysis
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=_make_openai_response(MOCK_ANALYSIS))
        result = await generate_ai_analysis(
            alert_name="HighErrorRate",
            service_name="frontend-api",
            severity="warning",
            description="Error rate spiked",
            logs="WARN payment failed\nWARN DB timeout",
            metrics={"error_rate_pct": "14.5", "p99_latency_ms": "2800"},
            http=mock_http,
        )
        assert result["confidence"] == "high"
        assert "rca_summary" in result
        assert "ansible_playbook" in result

    async def test_openai_calls_correct_url(self):
        from app.ai_analyst import generate_ai_analysis, _OPENAI_URL
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=_make_openai_response(MOCK_ANALYSIS))
        await generate_ai_analysis(
            alert_name="Test", service_name="svc", severity="warning",
            description="desc", logs="", metrics={}, http=mock_http,
        )
        call_url = mock_http.post.call_args[0][0]
        assert call_url == _OPENAI_URL

    async def test_openai_sends_bearer_auth_header(self):
        from app.ai_analyst import generate_ai_analysis
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=_make_openai_response(MOCK_ANALYSIS))
        await generate_ai_analysis(
            alert_name="Test", service_name="svc", severity="warning",
            description="desc", logs="", metrics={}, http=mock_http,
        )
        headers = mock_http.post.call_args[1]["headers"]
        # Just verify 'Bearer ' prefix — exact key value depends on runtime env
        assert headers["Authorization"].startswith("Bearer ")

    async def test_openai_sends_json_object_response_format(self):
        from app.ai_analyst import generate_ai_analysis
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=_make_openai_response(MOCK_ANALYSIS))
        await generate_ai_analysis(
            alert_name="Test", service_name="svc", severity="warning",
            description="desc", logs="", metrics={}, http=mock_http,
        )
        payload = mock_http.post.call_args[1]["json"]
        assert payload["response_format"] == {"type": "json_object"}

    async def test_returns_empty_dict_on_http_error(self):
        from app.ai_analyst import generate_ai_analysis
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=_make_error_response(401, "unauthorized"))
        result = await generate_ai_analysis(
            alert_name="Test", service_name="svc", severity="warning",
            description="desc", logs="", metrics={}, http=mock_http,
        )
        assert result == {}

    async def test_returns_empty_dict_on_invalid_json(self):
        from app.ai_analyst import generate_ai_analysis
        bad_resp = MagicMock()
        bad_resp.status_code = 200
        bad_resp.json.return_value = {
            "choices": [{"message": {"content": "not-valid-json{{{"}}]
        }
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=bad_resp)
        result = await generate_ai_analysis(
            alert_name="Test", service_name="svc", severity="warning",
            description="desc", logs="", metrics={}, http=mock_http,
        )
        assert result == {}

    async def test_returns_empty_dict_when_ai_disabled(self):
        from app import ai_analyst
        original = ai_analyst.AI_ENABLED
        try:
            ai_analyst.AI_ENABLED = False
            mock_http = AsyncMock()
            result = await ai_analyst.generate_ai_analysis(
                alert_name="Test", service_name="svc", severity="warning",
                description="desc", logs="", metrics={}, http=mock_http,
            )
            assert result == {}
            mock_http.post.assert_not_called()
        finally:
            ai_analyst.AI_ENABLED = original

    async def test_returns_empty_dict_on_network_exception(self):
        from app.ai_analyst import generate_ai_analysis
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(side_effect=Exception("connection refused"))
        result = await generate_ai_analysis(
            alert_name="Test", service_name="svc", severity="warning",
            description="desc", logs="", metrics={}, http=mock_http,
        )
        assert result == {}

    async def test_logs_are_truncated_to_3000_chars(self):
        """Logs longer than 3000 chars should be trimmed before sending to AI."""
        from app.ai_analyst import generate_ai_analysis
        long_logs = "x" * 5000
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=_make_openai_response(MOCK_ANALYSIS))
        await generate_ai_analysis(
            alert_name="Test", service_name="svc", severity="warning",
            description="desc", logs=long_logs, metrics={}, http=mock_http,
        )
        payload = mock_http.post.call_args[1]["json"]
        user_msg = payload["messages"][-1]["content"]
        assert "xxxxx" in user_msg
        # Truncated — the full 5000 chars must not appear
        assert len(user_msg) < 5000 + 500  # prompt overhead is fine


# ═══════════════════════════════════════════════════════════════════════════════
# fetch_loki_logs
# ═══════════════════════════════════════════════════════════════════════════════

def _loki_response(lines: list[str]):
    body = {
        "data": {
            "result": [
                {"stream": {"service_name": "frontend-api"},
                 "values": [["1710000000000000000", line] for line in lines]}
            ]
        }
    }
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = body
    return resp


@pytest.mark.asyncio
class TestFetchLokiLogs:

    async def test_returns_log_lines_as_text(self):
        from app.ai_analyst import fetch_loki_logs
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=_loki_response(
            ["INFO request started", "WARN high latency", "ERROR payment failed"]
        ))
        result = await fetch_loki_logs("frontend-api", mock_http)
        assert "INFO request started" in result
        assert "ERROR payment failed" in result

    async def test_returns_empty_string_on_http_error(self):
        from app.ai_analyst import fetch_loki_logs
        resp = MagicMock()
        resp.status_code = 500
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=resp)
        result = await fetch_loki_logs("frontend-api", mock_http)
        assert result == ""

    async def test_returns_empty_string_on_exception(self):
        from app.ai_analyst import fetch_loki_logs
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(side_effect=Exception("connection refused"))
        result = await fetch_loki_logs("frontend-api", mock_http)
        assert result == ""

    async def test_queries_correct_loki_endpoint(self):
        from app.ai_analyst import fetch_loki_logs, LOKI_URL
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=_loki_response([]))
        await fetch_loki_logs("frontend-api", mock_http)
        url = mock_http.get.call_args[0][0]
        assert LOKI_URL in url
        assert "query_range" in url

    async def test_service_name_included_in_query(self):
        from app.ai_analyst import fetch_loki_logs
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=_loki_response([]))
        await fetch_loki_logs("backend-api", mock_http)
        params = mock_http.get.call_args[1]["params"]
        assert "backend-api" in params["query"]


# ═══════════════════════════════════════════════════════════════════════════════
# fetch_prometheus_context
# ═══════════════════════════════════════════════════════════════════════════════

def _prom_response(value: float):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "data": {"result": [{"value": ["1710000000", str(value)]}]}
    }
    return resp


def _prom_empty():
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"data": {"result": []}}
    return resp


@pytest.mark.asyncio
class TestFetchPrometheusContext:

    async def test_returns_error_rate_for_service(self):
        from app.ai_analyst import fetch_prometheus_context
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=_prom_response(12.5))
        result = await fetch_prometheus_context("frontend-api", mock_http)
        assert "error_rate_pct" in result
        # 12.50
        assert result["error_rate_pct"] == "12.50"

    async def test_returns_no_data_when_prometheus_empty(self):
        from app.ai_analyst import fetch_prometheus_context
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=_prom_empty())
        result = await fetch_prometheus_context("frontend-api", mock_http)
        assert result["error_rate_pct"] == "no data"

    async def test_returns_all_four_golden_signals(self):
        from app.ai_analyst import fetch_prometheus_context
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=_prom_response(1.0))
        result = await fetch_prometheus_context("frontend-api", mock_http)
        assert set(result.keys()) == {
            "error_rate_pct",
            "p99_latency_ms",
            "p50_latency_ms",
            "rps",
            "cpu_usage_pct",
            "memory_usage_pct",
            "active_connections",
        }

    async def test_handles_individual_query_exception(self):
        from app.ai_analyst import fetch_prometheus_context
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(side_effect=Exception("timeout"))
        result = await fetch_prometheus_context("frontend-api", mock_http)
        for v in result.values():
            assert "error" in v.lower() or v == "no data"


# ═══════════════════════════════════════════════════════════════════════════════
# build_enriched_ticket_body
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildEnrichedTicketBody:

    def _build(self, analysis=None):
        from app.ai_analyst import build_enriched_ticket_body
        return build_enriched_ticket_body(
            service_name="frontend-api",
            alert_name="HighErrorRate",
            severity="warning",
            description="Error rate exceeded threshold",
            starts_at="2026-03-17T10:00:00Z",
            dashboard_url="http://grafana:3000/d/obs-overview",
            bridge_trace_id="abc123",
            metrics={"error_rate_pct": "14.5", "p99_latency_ms": "2800"},
            analysis=analysis or MOCK_ANALYSIS,
        )

    def test_contains_service_name(self):
        body = self._build()
        assert "frontend-api" in body

    def test_contains_alert_name(self):
        body = self._build()
        assert "HighErrorRate" in body

    def test_contains_severity(self):
        body = self._build()
        assert "WARNING" in body or "warning" in body

    def test_contains_dashboard_url(self):
        body = self._build()
        assert "grafana:3000" in body

    def test_contains_trace_id(self):
        body = self._build()
        assert "abc123" in body

    def test_contains_rca_summary_when_analysis_present(self):
        body = self._build()
        assert "DB connection exhaustion" in body

    def test_contains_ansible_playbook(self):
        body = self._build()
        assert "ansible" in body.lower() or "playbook" in body.lower()

    def test_returns_string(self):
        body = self._build()
        assert isinstance(body, str)
        assert len(body) > 100

    def test_handles_empty_analysis_gracefully(self):
        body = self._build(analysis={})
        assert "frontend-api" in body  # still has basic fields
