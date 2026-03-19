"""
tests/test_approval_workflow.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for approval_workflow.py:
  - request_approval()   → creates ApprovalRequest + xyOps gate ticket
  - process_decision()   → approved=True / declined paths
  - get_pending()        → lookup by approval_id
  - list_pending()       → filters by status=="pending"
"""

import pytest
from unittest.mock import AsyncMock, patch


MOCK_ANALYSIS = {
    "ansible_playbook": "---\n- name: fix\n  hosts: all\n  tasks: []",
    "ansible_description": "Restart payment service",
    "test_plan": ["Step 1: dry run", "Step 2: verify"],
    "rca_summary": "DB connection pool exhausted.",
    "confidence": "high",
    "pr_title": "fix: increase pool",
    "pr_description": "Increase DB pool to 50.",
    "rollback_steps": ["Step 1: revert pool size"],
}


def _make_xyops_post(ticket_id="t_apr_001", num=88):
    """Mock xyops_post that returns a ticket-create response for create_ticket calls."""
    async def _post(path, body):
        if "create_ticket" in path:
            return {"code": 0, "ticket": {"id": ticket_id, "num": num}}
        return {"code": 0}
    return _post


# ═══════════════════════════════════════════════════════════════════════════════
# request_approval()
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
class TestRequestApproval:

    async def _call(self, xyops_post=None, analysis=None, approval_id="apr-001"):
        from app.approval_workflow import request_approval, _pending
        _pending.clear()
        return await request_approval(
            approval_id=approval_id,
            incident_ticket_id="t_incident_001",
            alert_name="HighErrorRate",
            service_name="frontend-api",
            severity="warning",
            analysis=analysis or MOCK_ANALYSIS,
            bridge_trace_id="trace_abc123",
            xyops_post=xyops_post or _make_xyops_post(),
            xyops_url="http://xyops:5522",
        )

    async def test_returns_approval_request(self):
        from app.approval_workflow import ApprovalRequest
        req = await self._call()
        assert isinstance(req, ApprovalRequest)

    async def test_stores_in_pending(self):
        from app.approval_workflow import _pending
        _pending.clear()
        await self._call(approval_id="apr-store-test")
        assert "apr-store-test" in _pending

    async def test_status_is_pending(self):
        req = await self._call(approval_id="apr-status-test")
        assert req.status == "pending"

    async def test_sets_approval_ticket_id(self):
        req = await self._call()
        assert req.approval_ticket_id == "t_apr_001"

    async def test_sets_approval_ticket_num(self):
        req = await self._call()
        assert req.approval_ticket_num == 88

    async def test_stores_playbook(self):
        req = await self._call()
        assert req.ansible_playbook == MOCK_ANALYSIS["ansible_playbook"]

    async def test_calls_create_ticket_api(self):
        mock_post = AsyncMock(return_value={"code": 0, "ticket": {"id": "t", "num": 1}})
        await self._call(xyops_post=mock_post)
        create_calls = [c for c in mock_post.call_args_list
                        if "create_ticket" in c[0][0]]
        assert len(create_calls) == 1

    async def test_ticket_subject_contains_alert_name(self):
        captured = []

        async def _capturing_post(path, body):
            captured.append((path, body))
            if "create_ticket" in path:
                return {"code": 0, "ticket": {"id": "t", "num": 1}}
            return {"code": 0}

        await self._call(xyops_post=_capturing_post)
        create_call_bodies = [body for path, body in captured if "create_ticket" in path]
        assert create_call_bodies, "Expected at least one create_ticket call"
        assert "HighErrorRate" in create_call_bodies[0]["subject"]

    async def test_body_contains_playbook_yaml(self):
        captured_bodies = []

        async def _capturing_post(path, body):
            captured_bodies.append((path, body))
            if "create_ticket" in path:
                return {"code": 0, "ticket": {"id": "t", "num": 1}}
            return {"code": 0}

        await self._call(xyops_post=_capturing_post)
        create_bodies = [body for path, body in captured_bodies if "create_ticket" in path]
        assert "ansible" in create_bodies[0]["body"].lower() or "playbook" in create_bodies[0]["body"].lower()

    async def test_empty_analysis_does_not_raise(self):
        req = await self._call(analysis={})
        assert req is not None


# ═══════════════════════════════════════════════════════════════════════════════
# process_decision()
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
class TestProcessDecision:

    async def _setup_approval(self, approval_id="proc-001"):
        """Create a pending approval and return it."""
        from app.approval_workflow import request_approval, _pending
        _pending.clear()
        req = await request_approval(
            approval_id=approval_id,
            incident_ticket_id="t_inc_001",
            alert_name="HighErrorRate",
            service_name="frontend-api",
            severity="warning",
            analysis=MOCK_ANALYSIS,
            bridge_trace_id="trace_xyz",
            xyops_post=_make_xyops_post(),
            xyops_url="http://xyops:5522",
        )
        return req

    async def test_declined_returns_declined_status(self):
        from app.approval_workflow import process_decision, _pending
        _pending.clear()
        await self._setup_approval("proc-decline")
        result = await process_decision(
            approval_id="proc-decline",
            approved=False,
            decided_by="john.doe",
            notes="Not ready for prod",
            http=AsyncMock(),
            xyops_post=AsyncMock(return_value={"code": 0}),
        )
        assert result["status"] == "declined"

    async def test_approved_returns_approved_status(self):
        from app.approval_workflow import process_decision, _pending
        _pending.clear()
        await self._setup_approval("proc-approve")

        mock_http = AsyncMock()
        ansible_resp = AsyncMock()
        ansible_resp.status_code = 200
        ansible_resp.json.return_value = {
            "status": "success",
            "return_code": 0,
            "stdout": "playbook ran",
            "duration_seconds": 5,
        }
        mock_http.post = AsyncMock(return_value=ansible_resp)

        result = await process_decision(
            approval_id="proc-approve",
            approved=True,
            decided_by="jane.doe",
            notes="Looks good",
            http=mock_http,
            xyops_post=AsyncMock(return_value={"code": 0}),
        )
        assert result["status"] == "approved"

    async def test_unknown_approval_id_returns_error(self):
        from app.approval_workflow import process_decision, _pending
        _pending.clear()
        result = await process_decision(
            approval_id="no-such-id",
            approved=True,
            decided_by="someone",
            notes="",
            http=AsyncMock(),
            xyops_post=AsyncMock(return_value={"code": 0}),
        )
        assert "error" in result

    async def test_already_processed_returns_error(self):
        from app.approval_workflow import process_decision, _pending
        _pending.clear()
        await self._setup_approval("proc-double")
        # First decision
        await process_decision(
            approval_id="proc-double",
            approved=False,
            decided_by="user",
            notes="",
            http=AsyncMock(),
            xyops_post=AsyncMock(return_value={"code": 0}),
        )
        # Second call on same ID should return error
        result = await process_decision(
            approval_id="proc-double",
            approved=True,
            decided_by="user2",
            notes="",
            http=AsyncMock(),
            xyops_post=AsyncMock(return_value={"code": 0}),
        )
        assert "error" in result

    async def test_decline_updates_request_status(self):
        from app.approval_workflow import process_decision, _pending
        _pending.clear()
        req = await self._setup_approval("proc-status-check")
        await process_decision(
            approval_id="proc-status-check",
            approved=False,
            decided_by="admin",
            notes="",
            http=AsyncMock(),
            xyops_post=AsyncMock(return_value={"code": 0}),
        )
        assert req.status == "declined"
        assert req.decided_by == "admin"


# ═══════════════════════════════════════════════════════════════════════════════
# get_pending() / list_pending()
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
class TestPendingHelpers:

    async def test_get_pending_returns_none_for_unknown(self):
        from app.approval_workflow import get_pending
        result = get_pending("does-not-exist")
        assert result is None

    async def test_get_pending_returns_approval(self):
        from app.approval_workflow import get_pending, request_approval, _pending
        _pending.clear()
        await request_approval(
            approval_id="gp-test",
            incident_ticket_id="t_inc",
            alert_name="TestAlert",
            service_name="test-svc",
            severity="warning",
            analysis=MOCK_ANALYSIS,
            bridge_trace_id="trace_123",
            xyops_post=_make_xyops_post(),
            xyops_url="http://xyops:5522",
        )
        req = get_pending("gp-test")
        assert req is not None
        assert req.approval_id == "gp-test"

    async def test_list_pending_only_includes_pending(self):
        from app.approval_workflow import list_pending, request_approval, process_decision, _pending
        _pending.clear()
        # Create 2 approvals
        for i in range(2):
            await request_approval(
                approval_id=f"list-{i}",
                incident_ticket_id=f"t_inc_{i}",
                alert_name="TestAlert",
                service_name="test-svc",
                severity="warning",
                analysis=MOCK_ANALYSIS,
                bridge_trace_id=f"trace_{i}",
                xyops_post=_make_xyops_post(),
                xyops_url="http://xyops:5522",
            )
        # Decline one
        await process_decision(
            approval_id="list-0",
            approved=False,
            decided_by="admin",
            notes="",
            http=AsyncMock(),
            xyops_post=AsyncMock(return_value={"code": 0}),
        )
        pending = list_pending()
        assert all(r.status == "pending" for r in pending)
        assert len(pending) == 1
        assert pending[0].approval_id == "list-1"
