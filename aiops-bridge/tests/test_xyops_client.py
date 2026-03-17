"""
tests/test_xyops_client.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for xyops_client.py:
  - post_step_comment  (formats and POSTs a step progress comment)
  - post_outcome_comment  (formats and POSTs a final outcome comment)
  - ensure_aiops_workflow  (creates/updates the xyOps workflow event)
"""

import pytest
from unittest.mock import AsyncMock, call


# ── post_step_comment ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestPostStepComment:

    async def test_posts_to_correct_api_path(self):
        from app.xyops_client import post_step_comment
        mock_post = AsyncMock(return_value={"code": 0})
        await post_step_comment("t_abc", 1, "started", "Fetching logs", mock_post)
        mock_post.assert_called_once()
        path, body = mock_post.call_args[0]
        assert path == "/api/app/add_ticket_change/v1"

    async def test_body_contains_ticket_id(self):
        from app.xyops_client import post_step_comment
        mock_post = AsyncMock(return_value={"code": 0})
        await post_step_comment("t_xyz99", 2, "done", "Metrics fetched", mock_post)
        _, body = mock_post.call_args[0]
        assert body["id"] == "t_xyz99"

    async def test_body_contains_comment_type(self):
        from app.xyops_client import post_step_comment
        mock_post = AsyncMock(return_value={"code": 0})
        await post_step_comment("t_xyz99", 2, "done", "Metrics fetched", mock_post)
        _, body = mock_post.call_args[0]
        assert body["change"]["type"] == "comment"

    async def test_done_status_renders_ok_icon(self):
        from app.xyops_client import post_step_comment
        mock_post = AsyncMock(return_value={"code": 0})
        await post_step_comment("t_abc", 3, "done", "AI analysis complete", mock_post)
        _, body = mock_post.call_args[0]
        assert "[OK]" in body["change"]["body"]

    async def test_started_status_renders_right_arrow(self):
        from app.xyops_client import post_step_comment
        mock_post = AsyncMock(return_value={"code": 0})
        await post_step_comment("t_abc", 1, "started", "Starting up", mock_post)
        _, body = mock_post.call_args[0]
        assert "[>>]" in body["change"]["body"]

    async def test_error_status_renders_bang_icon(self):
        from app.xyops_client import post_step_comment
        mock_post = AsyncMock(return_value={"code": 0})
        await post_step_comment("t_abc", 2, "error", "Failed!", mock_post)
        _, body = mock_post.call_args[0]
        assert "[!!]" in body["change"]["body"]

    async def test_skipped_status_renders_dash_icon(self):
        from app.xyops_client import post_step_comment
        mock_post = AsyncMock(return_value={"code": 0})
        await post_step_comment("t_abc", 4, "skipped", "AI not enabled", mock_post)
        _, body = mock_post.call_args[0]
        assert "[--]" in body["change"]["body"]

    async def test_comment_body_contains_step_fraction(self):
        from app.xyops_client import post_step_comment
        mock_post = AsyncMock(return_value={"code": 0})
        await post_step_comment("t_abc", 3, "done", "Done", mock_post, total_steps=5)
        _, body = mock_post.call_args[0]
        assert "3/5" in body["change"]["body"]

    async def test_skips_silently_when_ticket_id_empty(self):
        from app.xyops_client import post_step_comment
        mock_post = AsyncMock(return_value={"code": 0})
        await post_step_comment("", 1, "started", "msg", mock_post)
        mock_post.assert_not_called()

    async def test_warning_logged_on_api_error(self, caplog):
        import logging
        from app.xyops_client import post_step_comment
        mock_post = AsyncMock(return_value={"error": "unauthorized"})
        with caplog.at_level(logging.WARNING, logger="aiops-bridge.xyops_client"):
            await post_step_comment("t_abc", 1, "done", "msg", mock_post)
        assert any("unauthorized" in r.message for r in caplog.records)


# ── post_outcome_comment ──────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestPostOutcomeComment:

    async def test_approved_renders_ok_icon(self):
        from app.xyops_client import post_outcome_comment
        mock_post = AsyncMock(return_value={"code": 0})
        await post_outcome_comment("t_123", "approved", "Engineer approved", mock_post)
        _, body = mock_post.call_args[0]
        assert "[OK]" in body["change"]["body"]

    async def test_declined_renders_bang_icon(self):
        from app.xyops_client import post_outcome_comment
        mock_post = AsyncMock(return_value={"code": 0})
        await post_outcome_comment("t_123", "declined", "Declined", mock_post)
        _, body = mock_post.call_args[0]
        assert "[!!]" in body["change"]["body"]

    async def test_skips_when_no_ticket_id(self):
        from app.xyops_client import post_outcome_comment
        mock_post = AsyncMock(return_value={"code": 0})
        await post_outcome_comment("", "approved", "msg", mock_post)
        mock_post.assert_not_called()


# ── ensure_aiops_workflow ─────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestEnsureAiopsWorkflow:

    async def test_creates_workflow_when_not_exists(self):
        from app.xyops_client import ensure_aiops_workflow
        mock_post = AsyncMock(return_value={"code": 0})
        mock_get = AsyncMock(return_value={})  # no "event" key → not found
        await ensure_aiops_workflow(mock_post, mock_get)
        # Should call create_event/v1
        create_calls = [
            c for c in mock_post.call_args_list
            if "create_event" in c[0][0]
        ]
        assert len(create_calls) == 1

    async def test_updates_workflow_when_already_exists(self):
        from app.xyops_client import ensure_aiops_workflow
        mock_post = AsyncMock(return_value={"code": 0})
        mock_get = AsyncMock(return_value={"event": {"id": "aiops_pipeline_wf"}})
        await ensure_aiops_workflow(mock_post, mock_get)
        # Should call update_event/v1, not create
        update_calls = [c for c in mock_post.call_args_list if "update_event" in c[0][0]]
        create_calls = [c for c in mock_post.call_args_list if "create_event" in c[0][0]]
        assert len(update_calls) == 1
        assert len(create_calls) == 0

    async def test_workflow_payload_has_type_workflow(self):
        from app.xyops_client import ensure_aiops_workflow
        mock_post = AsyncMock(return_value={"code": 0})
        mock_get = AsyncMock(return_value={})
        await ensure_aiops_workflow(mock_post, mock_get)
        _, payload = mock_post.call_args[0]
        assert payload.get("type") == "workflow"

    async def test_workflow_payload_has_six_agent_nodes(self):
        from app.xyops_client import ensure_aiops_workflow
        mock_post = AsyncMock(return_value={"code": 0})
        mock_get = AsyncMock(return_value={})
        await ensure_aiops_workflow(mock_post, mock_get)
        _, payload = mock_post.call_args[0]
        nodes = payload["workflow"]["nodes"]
        # trigger + 6 agent job nodes = 7
        assert len(nodes) == 7

    async def test_workflow_connections_form_linear_chain(self):
        from app.xyops_client import ensure_aiops_workflow
        mock_post = AsyncMock(return_value={"code": 0})
        mock_get = AsyncMock(return_value={})
        await ensure_aiops_workflow(mock_post, mock_get)
        _, payload = mock_post.call_args[0]
        conns = payload["workflow"]["connections"]
        # 6 connections for 7 nodes in a chain
        assert len(conns) == 6
        # First connection starts at the trigger
        assert conns[0]["source"] == "wf_trigger"
        assert conns[0]["dest"] == "wf_n1"

    async def test_logs_warning_on_api_failure(self, caplog):
        import logging
        from app.xyops_client import ensure_aiops_workflow
        mock_post = AsyncMock(return_value={"code": 1, "description": "server error"})
        mock_get = AsyncMock(return_value={})
        with caplog.at_level(logging.WARNING, logger="aiops-bridge.xyops_client"):
            await ensure_aiops_workflow(mock_post, mock_get)
        assert any("server error" in r.message for r in caplog.records)
