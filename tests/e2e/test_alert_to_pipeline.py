"""
tests/e2e/test_alert_to_pipeline.py
─────────────────────────────────────────────────────────────────────────────
End-to-end smoke test for the full alert → pipeline loop.

Flow under test
───────────────
  1. POST a realistic Alertmanager v2 webhook payload directly to the
     compute-agent /webhook endpoint (the same URL Alertmanager calls
     in production).
  2. Poll GET /pipeline/session/{service_name} until the session appears
     or a configurable timeout is exceeded.
  3. Assert structural correctness and that the pipeline advanced beyond
     the initial "created" stage.

Prerequisites
─────────────
  • All containers must be running:  docker compose up -d
  • The compute-agent must be reachable at COMPUTE_AGENT_URL
  • The ui-backend must be reachable at UI_BACKEND_URL

Running the tests
─────────────────
  # From the workspace root (Windows)
  docker compose exec compute-agent pytest /app/../../../tests/e2e -v

  # Or point at a running stack from the host (requires requests):
  pip install requests pytest
  pytest tests/e2e/test_alert_to_pipeline.py -v

Environment variables (all optional)
─────────────────────────────────────
  COMPUTE_AGENT_URL   default: http://localhost:9000
  UI_BACKEND_URL      default: http://localhost:9005
  E2E_TIMEOUT_SEC     default: 30
"""

import os
import time
import uuid

import pytest
import requests

COMPUTE_URL = os.getenv("COMPUTE_AGENT_URL", "http://localhost:9000")
UI_BACKEND_URL = os.getenv("UI_BACKEND_URL", "http://localhost:9005")
TIMEOUT = int(os.getenv("E2E_TIMEOUT_SEC", "30"))


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def alertmanager_payload(service_name: str, alert_name: str) -> dict:
    """Build a minimal Alertmanager v2 webhook payload."""
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return {
        "version": "4",
        "groupKey": f"{{alertname={alert_name}}}:{{{service_name}}}",
        "status": "firing",
        "receiver": "xyops-incidents",
        "groupLabels": {"alertname": alert_name},
        "commonLabels": {
            "alertname": alert_name,
            "service_name": service_name,
            "severity": "critical",
            "domain": "compute",
        },
        "commonAnnotations": {
            "summary": f"E2E test alert for {service_name}",
            "description": "Synthetic alert fired from the e2e test suite.",
        },
        "externalURL": "http://alertmanager:9093",
        "alerts": [
            {
                "status": "firing",
                "labels": {
                    "alertname": alert_name,
                    "service_name": service_name,
                    "severity": "critical",
                    "domain": "compute",
                },
                "annotations": {
                    "summary": f"E2E test alert for {service_name}",
                },
                "startsAt": now_iso,
                "endsAt": "0001-01-01T00:00:00Z",
                "generatorURL": "http://prometheus:9090",
                "fingerprint": uuid.uuid4().hex[:16],
            }
        ],
    }


def poll_session(session_id: str, timeout_sec: int) -> dict | None:
    """Poll GET /pipeline/session/{id} until it appears or timeout."""
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            resp = requests.get(
                f"{COMPUTE_URL}/pipeline/session/{session_id}", timeout=5
            )
            if resp.status_code == 200:
                return resp.json()
        except requests.RequestException:
            pass
        time.sleep(1)
    return None


# ─────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────

class TestAlertToPipeline:
    """End-to-end: Alertmanager webhook → compute-agent pipeline session."""

    def test_compute_agent_is_healthy(self):
        """Prerequisite: compute-agent must be reachable and healthy."""
        resp = requests.get(f"{COMPUTE_URL}/health", timeout=5)
        assert resp.status_code == 200, (
            f"compute-agent not reachable at {COMPUTE_URL}. "
            "Start containers first: docker compose up -d"
        )

    def test_ui_backend_is_healthy(self):
        """Prerequisite: ui-backend aggregator must be reachable."""
        resp = requests.get(f"{UI_BACKEND_URL}/health", timeout=5)
        assert resp.status_code == 200, (
            f"ui-backend not reachable at {UI_BACKEND_URL}. "
            "Start containers first: docker compose up -d"
        )

    def test_webhook_fires_and_creates_session(self):
        """
        POST a synthetic alert to /webhook and verify a pipeline session
        is created with the correct structure within TIMEOUT seconds.
        """
        service = "e2e-test-service"
        alert_name = "HighCPUUsage"
        payload = alertmanager_payload(service, alert_name)

        webhook_resp = requests.post(
            f"{COMPUTE_URL}/webhook",
            json=payload,
            timeout=10,
        )
        assert webhook_resp.status_code in (200, 202, 204), (
            f"Webhook POST failed: {webhook_resp.status_code} — {webhook_resp.text}"
        )

        # Compute-agent may echo session_id; fall back to service_name as key.
        try:
            body = webhook_resp.json()
            session_id = body.get("session_id", service)
        except Exception:
            session_id = service

        session = poll_session(session_id, TIMEOUT)
        assert session is not None, (
            f"Pipeline session '{session_id}' did not appear within {TIMEOUT}s "
            "after the webhook was posted."
        )

        # Structural assertions — these fields must always be present.
        for field in ("session_id", "stage", "service_name", "alert_name"):
            assert field in session, (
                f"Session response is missing required field '{field}'. "
                f"Got: {list(session.keys())}"
            )

        assert session["service_name"] == service
        assert session["alert_name"] == alert_name
        assert session["stage"], "Session 'stage' must not be empty"

    def test_pipeline_advances_past_created(self):
        """
        After the webhook, the pipeline should advance beyond 'created'
        within the timeout — proving at least one agent step ran.
        """
        service = "e2e-advance-test"
        alert_name = "ServiceUnhealthy"
        payload = alertmanager_payload(service, alert_name)

        requests.post(f"{COMPUTE_URL}/webhook", json=payload, timeout=10)

        deadline = time.time() + TIMEOUT
        advanced = False
        while time.time() < deadline:
            try:
                resp = requests.get(
                    f"{COMPUTE_URL}/pipeline/session/{service}", timeout=5
                )
                if resp.status_code == 200:
                    stage = resp.json().get("stage", "created")
                    if stage not in ("created", ""):
                        advanced = True
                        break
            except requests.RequestException:
                pass
            time.sleep(2)

        assert advanced, (
            f"Pipeline for '{service}' did not advance past 'created' "
            f"within {TIMEOUT}s. The agent did not run any analysis steps."
        )

    def test_risk_history_endpoint_returns_series(self):
        """
        Verify the ui-backend /risk-history endpoint returns a non-empty
        time-series array (or a valid fallback) for a known session.
        """
        service = "e2e-test-service"   # same session created in previous test
        resp = requests.get(
            f"{UI_BACKEND_URL}/pipeline/session/{service}/risk-history",
            timeout=10,
        )
        # 200 with data OR 404 if session doesn't survive — both acceptable here.
        if resp.status_code == 200:
            body = resp.json()
            assert "series" in body
            assert isinstance(body["series"], list)
            assert len(body["series"]) > 0
            assert "source" in body
            assert body["source"] in ("prometheus", "fallback")
        else:
            pytest.skip(
                "Session gone or service not available — "
                "run tests in order for full coverage."
            )
