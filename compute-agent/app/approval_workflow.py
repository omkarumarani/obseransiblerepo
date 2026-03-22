"""
aiops-bridge/app/approval_workflow.py
───────────────────────────────────────────────────────────────
Human-in-the-loop approval workflow.

Flow:
  1. A ticket is created in xyOps (by main.py) with full RCA +
     Ansible playbook + test plan.
  2. approval_workflow.request_approval() is called → creates a
     second "approval-gate" ticket in xyOps with:
       - Link to the original incident ticket
       - The Ansible playbook to be run
       - Test plan results (dry-run output)
       - Clear YES/NO action buttons (xyOps ticket events)
  3. Human opens xyOps, reads the RCA, reviews the playbook,
     sees the dry-run output, clicks "Approve Remediation".
  4. Approval triggers a xyOps job event which calls back:
       POST /approval/{approval_id}/decision  {"approved": true}
  5. The bridge receives the decision:
       - approved=true  → runs the real Ansible playbook via the
                          ansible-runner service (POST /run)
       - approved=false → closes the approval ticket as "declined"
  6. Playbook execution results are posted back to the original
     incident ticket as a comment.

POST /approval/{approval_id}/decision
  Body: {"approved": bool, "decided_by": str, "notes": str}

GET  /approval/{approval_id}
  Returns current state of a pending approval.

All state is held in-memory (dict).  For production, replace with
Redis or a database-backed store behind the same FastAPI app.
"""

import asyncio
import hashlib
import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

from .git_client import (
    GITEA_ENABLED,
    auto_merge_pull_request,
    check_pr_merged,
    close_pull_request,
    commit_playbook,
    create_pull_request,
    merge_pull_request,
)
from .approval_history import history_store
from .tier_registry import get_service_tier
from .xyops_client import create_approval_events

logger = logging.getLogger("aiops-bridge.approval")

ANSIBLE_RUNNER_URL: str = os.getenv("ANSIBLE_RUNNER_URL", "http://ansible-runner:8080")
_OBS_INTELLIGENCE_URL: str = os.getenv("OBS_INTELLIGENCE_URL", "http://obs-intelligence:9100")

_APR_DB = os.getenv("PIPELINE_DB_PATH", "/data/pipeline.db")


async def _record_llm_outcome(req: "ApprovalRequest", outcome: str) -> None:
    """
    Best-effort POST to obs-intelligence /intelligence/record-outcome.

    Updates both the SQLite outcome history and the ChromaDB entry for this
    run_id from outcome="pending" to the final outcome.  Never blocks the
    pipeline — all errors are silently logged at DEBUG level.
    """
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            await client.post(
                f"{_OBS_INTELLIGENCE_URL}/intelligence/record-outcome",
                json={
                    "scenario_id":          req.action_type or req.alert_name,
                    "outcome":              outcome,
                    "run_id":               req.bridge_trace_id,
                    "domain":               "compute",
                    "service_name":         req.service_name,
                    "alert_name":           req.alert_name,
                    "action_taken":         req.action_type,
                    "autonomy_decision":    req.status,
                    "validation_source":    "external_llm",
                },
            )
    except Exception as exc:  # noqa: BLE001
        logger.debug("_record_llm_outcome failed (best-effort): %s", exc)


def _apr_conn() -> sqlite3.Connection:
    import pathlib
    pathlib.Path(_APR_DB).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_APR_DB)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_approval_schema() -> None:
    with _apr_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pending_approvals (
                approval_id      TEXT PRIMARY KEY,
                session_id       TEXT,
                service_name     TEXT,
                action_type      TEXT,
                ansible_playbook TEXT,
                risk_score       REAL,
                tier             TEXT,
                status           TEXT DEFAULT 'pending',
                created_at       TEXT,
                decided_at       TEXT,
                decided_by       TEXT
            )
        """)


def _persist_approval(req: "ApprovalRequest") -> None:
    """UPSERT the ApprovalRequest state into pending_approvals."""
    try:
        with _apr_conn() as conn:
            conn.execute(
                """
                INSERT INTO pending_approvals
                    (approval_id, session_id, service_name, action_type,
                     ansible_playbook, risk_score, tier, status,
                     created_at, decided_at, decided_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(approval_id) DO UPDATE SET
                    status     = excluded.status,
                    decided_at = excluded.decided_at,
                    decided_by = excluded.decided_by
                """,
                (
                    req.approval_id,
                    req.session_id,
                    req.service_name,
                    req.action_type,
                    req.ansible_playbook[:4000] if req.ansible_playbook else "",
                    req.risk_score,
                    req.env_tier,
                    req.status,
                    req.created_at,
                    req.decided_at or None,
                    req.decided_by or None,
                ),
            )
    except Exception as exc:
        logger.warning("_persist_approval failed: %s", exc)


def _load_approvals_from_db() -> None:
    """Rebuild _pending from DB rows where status='pending' and created < 48 h ago."""
    from datetime import timedelta
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
    try:
        with _apr_conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM pending_approvals
                WHERE status = 'pending' AND created_at > ?
                """,
                (cutoff,),
            ).fetchall()
        for row in rows:
            if row["approval_id"] in _pending:
                continue
            req = ApprovalRequest(
                approval_id         = row["approval_id"],
                session_id          = row["session_id"] or "",
                incident_ticket_id  = "",
                alert_name          = "",
                service_name        = row["service_name"] or "",
                severity            = "warning",
                ansible_playbook    = row["ansible_playbook"] or "",
                ansible_description = "",
                test_plan           = [],
                rca_summary         = "",
                bridge_trace_id     = "",
                created_at          = row["created_at"] or "",
                status              = row["status"] or "pending",
                action_type         = row["action_type"] or "",
                env_tier            = row["tier"] or "",
            )
            _pending[req.approval_id] = req
        if rows:
            logger.info("Restored %d pending approvals from DB", len(rows))
    except Exception as exc:
        logger.warning("_load_approvals_from_db failed: %s", exc)

# ── In-memory state ────────────────────────────────────────────────────────────
# {approval_id: ApprovalRequest}
_pending: dict[str, "ApprovalRequest"] = {}

@dataclass
class ApprovalRequest:
    approval_id: str
    session_id: str
    incident_ticket_id: str
    alert_name: str
    service_name: str
    severity: str
    ansible_playbook: str          # raw YAML string
    ansible_description: str
    test_plan: list[str]
    rca_summary: str
    bridge_trace_id: str
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    status: str = "pending"        # pending | approved | declined | executed
    decided_by: str = ""
    decided_at: str = ""
    execution_result: dict = field(default_factory=dict)

    # xyOps approval-gate ticket ID and number
    approval_ticket_id: str = ""
    approval_ticket_num: int = 0
    # xyOps event IDs for the Approve / Decline ticket buttons
    approve_event_id: str = ""
    decline_event_id: str = ""
    # Structured test cases from AI analysis
    test_cases: list[dict] = field(default_factory=list)
    # Gitea PR details
    gitea_pr_url: str = ""
    gitea_pr_num: int = 0
    gitea_branch: str = ""
    # Pre-commit Ansible validation
    validation_passed: bool = False
    validation_result: dict = field(default_factory=dict)
    # Autonomy engine fields (set by caller when applicable)
    action_type: str = ""          # recommended action from scenario catalog
    env_tier: str = ""             # tier at the time of the request
    risk_score: float = 0.0
    gitea_commit_sha: str = ""


_ensure_approval_schema()
_load_approvals_from_db()


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

async def request_approval(
    session_id: str,
    approval_id: str,
    incident_ticket_id: str,
    alert_name: str,
    service_name: str,
    severity: str,
    analysis: dict[str, Any],
    bridge_trace_id: str,
    xyops_post,   # callable: async (path, body) -> dict
    xyops_url: str,
    http: "httpx.AsyncClient | None" = None,
    action_type: str = "",
    env_tier: str = "",
    risk_score: float = 0.0,
) -> ApprovalRequest:
    """
    Create an ApprovalRequest and open a gating ticket in xyOps.

    The gating ticket body explains exactly what will happen if approved,
    including the full Ansible playbook and test plan.  The approver
    clicks "Approve Remediation" in the xyOps UI which should call back
    POST /approval/{approval_id}/decision with approved=true.
    """
    playbook = analysis.get("ansible_playbook", "")
    description = analysis.get("ansible_description", "")
    test_plan = analysis.get("test_plan", [])
    test_cases = analysis.get("test_cases", [])
    rca_summary = analysis.get("rca_summary", "")
    pr_title = analysis.get("pr_title", "")
    pr_description = analysis.get("pr_description", "")
    rollback_steps = analysis.get("rollback_steps", [])
    confidence = analysis.get("confidence", "unknown")

    req = ApprovalRequest(
        approval_id=approval_id,
        session_id=session_id,
        incident_ticket_id=incident_ticket_id,
        alert_name=alert_name,
        service_name=service_name,
        severity=severity,
        ansible_playbook=playbook,
        ansible_description=description,
        test_plan=test_plan,
        test_cases=test_cases,
        rca_summary=rca_summary,
        bridge_trace_id=bridge_trace_id,
        action_type=action_type or analysis.get("recommended_action", ""),
        env_tier=env_tier or get_service_tier(service_name).value,
        risk_score=risk_score,
    )
    _pending[approval_id] = req
    _persist_approval(req)
    # ── Step 0: Validate playbook BEFORE any git operations ───────────────────
    # Ansible code is tested first.  Only if validation passes do we commit to
    # Gitea and open a PR.  If it fails the user sees a clear explanation in the
    # incident ticket and no code is pushed anywhere.
    if http and playbook:
        val_payload = {
            "playbook_yaml": playbook,
            "service_name":  service_name,
            "alert_name":    alert_name,
            "trace_id":      bridge_trace_id,
            "test_cases":    test_cases,
        }
        try:
            val_resp = await http.post(
                f"{ANSIBLE_RUNNER_URL}/validate",
                json=val_payload,
                timeout=90.0,
            )
            if val_resp.status_code == 200:
                val               = val_resp.json()
                all_passed        = val.get("all_passed", True)
                test_results      = val.get("test_results", [])
                passed_count      = sum(1 for t in test_results if t.get("status") == "PASSED")
                total_count       = len(test_results)
                req.validation_passed = all_passed
                req.validation_result = {
                    "test_results": test_results,
                    "stdout":       val.get("stdout", ""),
                    "all_passed":   all_passed,
                }
                logger.info(
                    "Pre-commit validation: %s/%s passed  alert=%s",
                    passed_count, total_count, alert_name,
                )

                if not all_passed:
                    # Build failure comment for the incident ticket
                    fail_lines = [
                        "## \u274c Ansible Pre-commit Validation FAILED",
                        "",
                        "The auto-generated playbook **failed validation** and was "
                        "**NOT pushed to Gitea**.  No automated changes will be made.",
                        "",
                        f"**Result:** {passed_count}/{total_count} test cases passed",
                        "",
                        "| Test Case | Status | Detail |",
                        "|---|---|---|",
                    ]
                    for tc in test_results:
                        icon = "\u2705" if tc.get("status") == "PASSED" else "\u274c"
                        fail_lines.append(
                            f"| `{tc.get('id', '?')}` {tc.get('name', '')} "
                            f"| {icon} {tc.get('status', '?')} "
                            f"| {tc.get('output', '')[:100]} |"
                        )
                    fail_lines += [
                        "",
                        "```",
                        val.get("stdout", "")[:1500],
                        "```",
                        "",
                        "> **Manual review required.** "
                        "Fix the playbook or investigate the alert manually.",
                    ]
                    await xyops_post(
                        "/api/app/add_ticket_change/v1",
                        {
                            "id":     incident_ticket_id,
                            "change": {"type": "comment", "body": "\n".join(fail_lines)},
                        },
                    )
                    req.status = "validation_failed"
                    logger.warning(
                        "Approval blocked — playbook failed pre-commit validation  alert=%s",
                        alert_name,
                    )
                    return req
            else:
                # Non-200 from validator → treat as passed (fail-open) and log
                req.validation_passed = True
                logger.warning(
                    "Validator returned HTTP %d — treating as passed", val_resp.status_code
                )
        except Exception as exc:
            # Network error → treat as passed and log
            req.validation_passed = True
            logger.warning("Validation call failed (fail-open): %s", exc)
    else:
        # No playbook or no http client → skip validation
        req.validation_passed = True

    # ── Step 1: Create Approve / Decline events in xyOps ─────────────────────
    bridge_host = os.getenv("BRIDGE_INTERNAL_URL", "http://compute-agent:9000")
    try:
        approve_evt, decline_evt = await create_approval_events(
            approval_id, bridge_host, xyops_post
        )
        req.approve_event_id = approve_evt
        req.decline_event_id = decline_evt
    except Exception as exc:
        logger.warning("Could not create approval events: %s", exc)
        approve_evt = ""
        decline_evt = ""

    # ── Step 2: Gitea: commit playbook to a branch and open a PR ─────────────────────
    # Validation already passed so it’s safe to commit the code to git.
    if http and GITEA_ENABLED and playbook:
        try:
            git_info = await commit_playbook(
                approval_id=approval_id,
                playbook_yaml=playbook,
                service_name=service_name,
                alert_name=alert_name,
                http=http,
            )
            if not git_info.get("error"):
                req.gitea_branch = git_info.get("branch", "")
                req.gitea_commit_sha = git_info.get("sha", "")
                pr_info = await create_pull_request(
                    branch=req.gitea_branch,
                    service_name=service_name,
                    alert_name=alert_name,
                    rca_summary=rca_summary,
                    approval_id=approval_id,
                    http=http,
                )
                req.gitea_pr_url = pr_info.get("pr_url", "")
                req.gitea_pr_num = pr_info.get("pr_number", 0)
                logger.info(
                    "Gitea PR #%d created  url=%s", req.gitea_pr_num, req.gitea_pr_url
                )
        except Exception as exc:
            logger.warning("Gitea git commit/PR failed (non-fatal): %s", exc)

    # ── Build the approval ticket body ─────────────────────
    body = _build_approval_body(
        req=req,
        confidence=confidence,
        pr_title=pr_title,
        pr_description=pr_description,
        rollback_steps=rollback_steps,
        xyops_url=xyops_url,
        approval_id=approval_id,
    )

    ticket_payload: dict[str, Any] = {
        "subject": f"[APPROVAL REQUIRED] Remediate `{alert_name}` on `{service_name}` — {severity.upper()}",
        "body": body,
        "type": "change",   # xyOps "change" type = change management / approval
        "status": "open",
    }
    if approve_evt and decline_evt:
        ticket_payload["events"] = [
            {"id": approve_evt},
            {"id": decline_evt},
        ]

    result = await xyops_post("/api/app/create_ticket/v1", ticket_payload)
    req.approval_ticket_id = result.get("ticket", {}).get("id", "")
    req.approval_ticket_num = result.get("ticket", {}).get("num", 0)
    _persist_approval(req)

    logger.info(
        "Approval gate ticket #%s (%s) created  approval_id=%s  incident=%s",
        req.approval_ticket_num,
        req.approval_ticket_id,
        approval_id,
        incident_ticket_id,
    )
    return req


def get_pending(approval_id: str) -> ApprovalRequest | None:
    return _pending.get(approval_id)


def list_pending() -> list[ApprovalRequest]:
    return [r for r in _pending.values() if r.status == "pending"]


async def execute_autonomous(
    *,
    session_id: str,
    approval_id: str,
    incident_ticket_id: str,
    alert_name: str,
    service_name: str,
    severity: str,
    analysis: dict[str, Any],
    bridge_trace_id: str,
    action_type: str,
    env_tier: str,
    risk_score: float,
    auto_merge_pr: bool,
    http: httpx.AsyncClient,
    xyops_post,
) -> dict[str, Any]:
    """
    Execute an Ansible playbook fully autonomously — no approval gate ticket.

    Called by pipeline.py Agent 6 when the Autonomy Engine grants autonomous
    execution.  Performs:
      1. (Optional) commit playbook to Gitea branch + create PR
      2. (Optional) auto-merge the PR
      3. Run ansible-runner /run directly
      4. Record decision + outcome in approval history

    Returns a dict with execution status for the pipeline response.
    """
    playbook  = analysis.get("ansible_playbook", "")
    test_cases = analysis.get("test_cases", [])

    # Build a minimal ApprovalRequest to reuse _execute_playbook
    req = ApprovalRequest(
        approval_id=approval_id,
        session_id=session_id,
        incident_ticket_id=incident_ticket_id,
        alert_name=alert_name,
        service_name=service_name,
        severity=severity,
        ansible_playbook=playbook,
        ansible_description=analysis.get("ansible_description", ""),
        test_plan=analysis.get("test_plan", []),
        test_cases=test_cases,
        rca_summary=analysis.get("rca_summary", ""),
        bridge_trace_id=bridge_trace_id,
        action_type=action_type,
        env_tier=env_tier,
        risk_score=risk_score,
    )
    _pending[approval_id] = req
    _persist_approval(req)

    # ── Record autonomous decision in history (outcome = pending for now) ─────
    history_store.record_decision(
        approval_id=approval_id,
        service_name=service_name,
        alert_name=alert_name,
        action_type=action_type,
        env_tier=env_tier,
        decided_by="autonomous",
        decision="autonomous",
        risk_score=risk_score,
        notes="Autonomy engine granted execution — trust threshold met",
    )

    # ── Commit + PR + (optional) auto-merge in Gitea ─────────────────────────
    if GITEA_ENABLED and playbook:
        try:
            git_info = await commit_playbook(
                approval_id=approval_id,
                playbook_yaml=playbook,
                service_name=service_name,
                alert_name=alert_name,
                http=http,
            )
            if not git_info.get("error"):
                req.gitea_branch = git_info.get("branch", "")
                req.gitea_commit_sha = git_info.get("sha", "")
                pr_info = await create_pull_request(
                    branch=req.gitea_branch,
                    service_name=service_name,
                    alert_name=alert_name,
                    rca_summary=req.rca_summary,
                    approval_id=approval_id,
                    http=http,
                )
                req.gitea_pr_url = pr_info.get("pr_url", "")
                req.gitea_pr_num = pr_info.get("pr_number", 0)

                if auto_merge_pr and req.gitea_pr_num:
                    await auto_merge_pull_request(req.gitea_pr_num, approval_id, http)
                    logger.info(
                        "Autonomous PR #%d auto-merged  service=%s  alert=%s",
                        req.gitea_pr_num, service_name, alert_name,
                    )
        except Exception as exc:
            logger.warning("Gitea autonomous commit/PR/merge failed (non-fatal): %s", exc)

    # ── Execute playbook ──────────────────────────────────────────────────────
    req.status = "approved"
    _persist_approval(req)
    _sync_pipeline_session(req, stage="autonomous_executing", autonomy_decision="AUTONOMOUS")
    asyncio.create_task(
        _execute_playbook(req=req, http=http, xyops_post=xyops_post)
    )

    logger.info(
        "Autonomous execution started  approval_id=%s  service=%s  alert=%s  tier=%s",
        approval_id, service_name, alert_name, env_tier,
    )
    return {
        "status": "autonomous",
        "approval_id": approval_id,
        "gitea_pr_url": req.gitea_pr_url,
        "gitea_pr_num": req.gitea_pr_num,
        "message": "Autonomous execution started — playbook running without human approval",
    }


async def process_decision(
    approval_id: str,
    approved: bool,
    decided_by: str,
    notes: str,
    http: httpx.AsyncClient,
    xyops_post,
) -> dict[str, Any]:
    """
    Called when a human approves or declines via POST /approval/{id}/decision.

    - approved=True  → POST to ansible-runner → execute playbook
    - approved=False → update approval ticket as declined, no action
    """
    req = _pending.get(approval_id)
    if not req:
        return {"error": f"No pending approval with id={approval_id}"}
    if req.status != "pending":
        return {"error": f"Approval {approval_id} already in state={req.status}"}

    # Gate: reject approval if playbook validation failed
    if approved and not req.validation_passed:
        return {
            "error": "Cannot approve — playbook validation failed. Fix the playbook first.",
            "validation_result": req.validation_result,
        }

    now = datetime.now(timezone.utc).isoformat()
    req.decided_by = decided_by
    req.decided_at = now

    if not approved:
        req.status = "declined"
        logger.info(
            "Approval %s DECLINED by %s  alert=%s", approval_id, decided_by, req.alert_name
        )
        # Record the human decline in history
        history_store.record_decision(
            approval_id=approval_id,
            service_name=req.service_name,
            alert_name=req.alert_name,
            action_type=req.action_type,
            env_tier=req.env_tier,
            decided_by=decided_by,
            decision="declined",
            risk_score=0.0,
            notes=notes or "",
        )
        # Close the Gitea PR so the branch is cleanly rejected
        if req.gitea_pr_num:
            try:
                await close_pull_request(req.gitea_pr_num, http)
                logger.info("Gitea PR #%d closed (declined)", req.gitea_pr_num)
            except Exception as exc:
                logger.warning("Could not close Gitea PR: %s", exc)
        decline_msg = (
            f"## Remediation Declined\n\n"
            f"Declined by **{decided_by}** at {now}\n\n"
            f"Notes: {notes or '(none)'}\n\n"
            f"No automated changes were made. Manual investigation required."
        )
        await _update_approval_ticket(req=req, comment=decline_msg, xyops_post=xyops_post)
        # Post outcome back to the original incident ticket
        await _post_to_incident(
            req=req,
            status="declined",
            message=f"Remediation DECLINED by {decided_by} — no automated changes made",
            xyops_post=xyops_post,
        )
        return {"status": "declined", "approval_id": approval_id}

    # ── Approved → check PR is merged, then trigger Ansible execution ─────────
    req.status = "approved"
    _persist_approval(req)
    _sync_pipeline_session(req, stage="approved", autonomy_decision="APPROVAL_GATED")
    logger.info(
        "Approval %s APPROVED by %s  alert=%s — checking PR merge status",
        approval_id, decided_by, req.alert_name,
    )

    # Record the human approval in history (outcome updated after execution)
    history_store.record_decision(
        approval_id=approval_id,
        service_name=req.service_name,
        alert_name=req.alert_name,
        action_type=req.action_type,
        env_tier=req.env_tier,
        decided_by=decided_by,
        decision="approved",
        risk_score=0.0,
        notes=notes or "",
    )

    # If a PR exists, check merge status.
    # For tiers that allow auto_merge on approval (staging/dev/sandbox) and the
    # PR is still open, the agent merges it automatically so the user only needs
    # to click "Approve" once without a separate trip to Gitea.
    if req.gitea_pr_num:
        try:
            is_merged = await check_pr_merged(req.gitea_pr_num, http)
        except Exception as exc:
            is_merged = False
            logger.warning("Could not check PR merge status: %s", exc)

        if not is_merged:
            # Determine if this tier/policy allows auto-merge on human approval
            from .tier_registry import get_service_tier, get_tier_policy
            tier = get_service_tier(req.service_name)
            policy = get_tier_policy(tier)

            if policy.auto_merge_pr:
                # Automatically merge the PR on behalf of the approving human
                logger.info(
                    "Tier '%s' allows auto_merge — merging PR #%d automatically  approval_id=%s",
                    tier.value, req.gitea_pr_num, approval_id,
                )
                merged = await auto_merge_pull_request(req.gitea_pr_num, approval_id, http)
                if not merged:
                    logger.warning(
                        "Auto-merge of PR #%d failed — falling back to manual merge reminder",
                        req.gitea_pr_num,
                    )
                    req.status = "pending"
                    _persist_approval(req)
                    pr_remind = (
                        f"## ⚠️ Auto-Merge Failed\n\n"
                        f"The agent attempted to auto-merge PR "
                        f"[#{req.gitea_pr_num}]({req.gitea_pr_url}) "
                        f"but the merge was rejected by Gitea.\n\n"
                        f"**Please complete both steps in order:**\n\n"
                        f"1. 🔗 [Merge PR #{req.gitea_pr_num}]({req.gitea_pr_url}) manually on Gitea\n"
                        f"2. ✅ Click **Approve Remediation** again\n\n"
                        f"The playbook will execute automatically once both conditions are met."
                    )
                    await _update_approval_ticket(req=req, comment=pr_remind, xyops_post=xyops_post)
                    return {
                        "status": "pending_pr_merge",
                        "approval_id": approval_id,
                        "message": f"Auto-merge failed — please merge PR #{req.gitea_pr_num} manually then re-approve",
                    }
                logger.info("PR #%d auto-merged — proceeding to Ansible execution", req.gitea_pr_num)
            else:
                # Production tier: human must merge PR manually
                req.status = "pending"
                _persist_approval(req)
                pr_remind = (
                    f"## ⚠️ PR Not Yet Merged\n\n"
                    f"You clicked **Approve** but PR [#{req.gitea_pr_num}]({req.gitea_pr_url}) "
                    f"has **not been merged** yet.\n\n"
                    f"**Please complete both steps in order:**\n\n"
                    f"1. 🔗 [Merge PR #{req.gitea_pr_num}]({req.gitea_pr_url}) on Gitea "
                    f"(review the YAML diff, then click 'Merge Pull Request')\n"
                    f"2. ✅ Return here and click **Approve Remediation** again\n\n"
                    f"The playbook will execute automatically once both conditions are met.\n\n"
                    f"> **Note:** Service tier is `{tier.value}` — "
                    f"manual PR review is required before execution on this tier."
                )
                await _update_approval_ticket(req=req, comment=pr_remind, xyops_post=xyops_post)
                logger.info(
                    "Approval %s reset to pending — PR #%d not merged yet (tier=%s, auto_merge=False)",
                    approval_id, req.gitea_pr_num, tier.value,
                )
                return {
                    "status": "pending_pr_merge",
                    "approval_id": approval_id,
                    "message": f"Please merge PR #{req.gitea_pr_num} first, then re-approve",
                }
        else:
            # PR is already merged by the user — proceed straight to execution
            logger.info("PR #%d already merged — proceeding to Ansible execution", req.gitea_pr_num)

    approve_msg = (
        f"## ✅ Remediation Approved\n\n"
        f"Approved by **{decided_by}** at {now}\n\n"
        f"Notes: {notes or '(none)'}\n\n"
        f"{'PR [#' + str(req.gitea_pr_num) + '](' + req.gitea_pr_url + ') merged ✓  ' if req.gitea_pr_num else ''}"
        f"Ansible playbook execution started..."
    )
    await _update_approval_ticket(req=req, comment=approve_msg, xyops_post=xyops_post)
    # Post outcome back to the original incident ticket
    await _post_to_incident(
        req=req,
        status="approved",
        message=f"Remediation APPROVED by {decided_by} — Ansible playbook executing now",
        xyops_post=xyops_post,
    )

    # Trigger Ansible playbook execution (non-blocking)
    asyncio.create_task(
        _execute_playbook(req=req, http=http, xyops_post=xyops_post)
    )

    return {
        "status": "approved",
        "approval_id": approval_id,
        "message": "Ansible playbook execution started",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

async def _execute_playbook(
    req: ApprovalRequest,
    http: httpx.AsyncClient,
    xyops_post,
) -> None:
    """
    Validation was already performed pre-commit in request_approval().
    Go straight to /run — no duplicate validate step needed.
    """
    run_payload = {
        "playbook_yaml": req.ansible_playbook,
        "service_name": req.service_name,
        "alert_name": req.alert_name,
        "trace_id": req.bridge_trace_id,
        "test_cases": req.test_cases,
    }

    # ── Execute ───────────────────────────────────────────────────────────────
    try:
        resp = await http.post(
            f"{ANSIBLE_RUNNER_URL}/run",
            json=run_payload,
            timeout=120.0,
        )
        if resp.status_code == 200:
            result = resp.json()
            req.execution_result = result
            req.status = "executed"
            _persist_approval(req)

            rc = result.get("return_code", -1)
            stdout = result.get("stdout", "")[:3000]
            status_icon = "✅" if rc == 0 else "❌"
            test_results = result.get("test_results", [])

            comment = (
                f"## {status_icon} Ansible Playbook Execution Result\n\n"
                f"| Field | Value |\n|---|---|\n"
                f"| **Return code** | `{rc}` |\n"
                f"| **Duration** | {result.get('duration_seconds', '?')}s |\n"
                f"| **Mode** | `{result.get('mode', 'simulated')}` |\n"
                f"| **Executed at** | {datetime.now(timezone.utc).isoformat()} |\n\n"
            )
            if test_results:
                passed_count = sum(1 for t in test_results if t.get("status") == "PASSED")
                comment += (
                    f"### Test Results ({passed_count}/{len(test_results)} passed)\n\n"
                    f"| Test Case | Status | Detail |\n|---|---|---|\n"
                )
                for tc in test_results:
                    icon = "✅" if tc.get("status") == "PASSED" else "❌"
                    comment += (
                        f"| `{tc.get('id', '?')}` {tc.get('name', '')} "
                        f"| {icon} {tc.get('status', '?')} "
                        f"| {tc.get('output', '')[:80]} |\n"
                    )
                comment += "\n"
            comment += f"### Output\n\n```\n{stdout}\n```\n"
            if rc != 0:
                comment += (
                    "\n\n> ⚠️ Playbook failed. Review the output above. "
                    "Manual rollback may be required."
                )
        else:
            req.status = "executed"
            _persist_approval(req)
            comment = (
                f"## ❌ Ansible Runner Error\n\n"
                f"ansible-runner returned HTTP {resp.status_code}\n\n"
                f"```\n{resp.text[:500]}\n```"
            )
    except Exception as exc:
        req.status = "executed"
        _persist_approval(req)
        comment = (
            f"## ❌ Ansible Runner Unreachable\n\n"
            f"Could not contact ansible-runner at `{ANSIBLE_RUNNER_URL}`\n\n"
            f"Error: `{exc}`\n\n"
            f"Playbook content is in this ticket — run manually:\n\n"
            f"```yaml\n{req.ansible_playbook[:2000]}\n```"
        )
        logger.error("Ansible runner call failed: %s", exc)

    await _update_approval_ticket(req=req, comment=comment, xyops_post=xyops_post)

    rc = req.execution_result.get("return_code", -1) if req.execution_result else -1
    if rc == 0:
        outcome_status = "executed"
        outcome_msg = "Ansible playbook executed successfully — service should be recovering"
        history_store.update_outcome(req.approval_id, "success")
        _sync_pipeline_session(req, stage="complete", outcome="success", complete=True)
        asyncio.create_task(_record_llm_outcome(req, "success"))
    else:
        outcome_status = "failed"
        outcome_msg = f"Ansible playbook FAILED (rc={rc}) — manual intervention required"
        history_store.update_outcome(req.approval_id, "failure")
        _sync_pipeline_session(req, stage="complete", outcome="failure", complete=True)
        asyncio.create_task(_record_llm_outcome(req, "failure"))
    await _append_change_audit_record(req=req, outcome=outcome_status, xyops_post=xyops_post)
    await _post_to_incident(req=req, status=outcome_status, message=outcome_msg, xyops_post=xyops_post)


def _sync_pipeline_session(
    req: ApprovalRequest,
    *,
    stage: str | None = None,
    autonomy_decision: str | None = None,
    outcome: str | None = None,
    complete: bool = False,
) -> None:
    if not req.session_id:
        return
    try:
        from .pipeline import update_pipeline_session_state

        update_pipeline_session_state(
            req.session_id,
            stage=stage,
            autonomy_decision=autonomy_decision,
            outcome=outcome,
            completed=complete,
        )
    except Exception as exc:
        logger.warning("Could not sync pipeline session %s: %s", req.session_id, exc)


async def _append_change_audit_record(
    req: ApprovalRequest,
    *,
    outcome: str,
    xyops_post,
) -> None:
    if not req.incident_ticket_id:
        return
    validated_at = req.validation_result.get("validated_at") or req.created_at
    decided_at = req.decided_at or req.created_at
    executed_at = datetime.now(timezone.utc).isoformat()
    playbook_sha256 = hashlib.sha256(req.ansible_playbook.encode("utf-8")).hexdigest()
    audit_block = (
        "═══════════════════ CHANGE AUDIT RECORD ═══════════════════\n"
        f"RFC ID:           {req.approval_id}\n"
        f"Change Request:   {req.action_type} on {req.service_name}\n"
        "Requested by:     AIOps Autonomous Engine\n"
        f"Validation:       Ansible dry-run {'PASSED' if req.validation_passed else 'FAILED'} at {validated_at}\n"
        f"Evidence:         Git commit {req.gitea_commit_sha or 'n/a'} in branch {req.gitea_branch or 'n/a'}\n"
        f"PR link:          {req.gitea_pr_url or 'n/a'}\n"
        f"Approved by:      {req.decided_by or 'autonomous'} at {decided_at}\n"
        f"Executed at:      {executed_at}\n"
        f"OpenTelemetry:    trace_id={req.bridge_trace_id}\n"
        f"Outcome:          {outcome}\n"
        f"Playbook hash:    SHA256:{playbook_sha256}\n"
        "════════════════════════════════════════════════════════════"
    )
    await xyops_post(
        "/api/app/add_ticket_change/v1",
        {"id": req.incident_ticket_id, "change": {"type": "comment", "body": audit_block}},
    )


async def _update_approval_ticket(
    req: ApprovalRequest,
    comment: str,
    xyops_post,
) -> None:
    """Append a comment to the approval-gate ticket."""
    if not req.approval_ticket_id:
        return
    await xyops_post(
        "/api/app/add_ticket_change/v1",
        {"id": req.approval_ticket_id, "change": {"type": "comment", "body": comment}},
    )


async def _post_to_incident(
    req: ApprovalRequest,
    status: str,
    message: str,
    xyops_post,
) -> None:
    """Post a free-form outcome comment to the original incident ticket."""
    if not req.incident_ticket_id:
        return
    now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    icons = {"approved": "[OK]", "declined": "[!!]", "executing": "[>>]", "executed": "[OK]", "failed": "[!!]"}
    icon = icons.get(status, "[??]")
    body = f"{icon} **{message}**  \n`{now}`"
    await xyops_post(
        "/api/app/add_ticket_change/v1",
        {"id": req.incident_ticket_id, "change": {"type": "comment", "body": body}},
    )


def _build_approval_body(
    req: ApprovalRequest,
    confidence: str,
    pr_title: str,
    pr_description: str,
    rollback_steps: list[str],
    xyops_url: str,
    approval_id: str,
) -> str:
    bridge_host = os.getenv("BRIDGE_INTERNAL_URL", "http://compute-agent:9000")

    body = (
        f"## 🔐 Change Approval Required\n\n"
        f"An automated Ansible remediation has been proposed for the following incident.\n"
        f"**Please review the full context below and approve or decline.**\n\n"
        f"| Field | Value |\n"
        f"|---|---|\n"
        f"| **Alert** | `{req.alert_name}` |\n"
        f"| **Service** | `{req.service_name}` |\n"
        f"| **Severity** | `{req.severity.upper()}` |\n"
        f"| **AI Confidence** | `{confidence}` |\n"
        f"| **Incident Ticket** | `{req.incident_ticket_id}` — see xyOps Tickets |\n"
        f"| **Trace ID** | `{req.bridge_trace_id}` — open in Grafana → Tempo |\n\n"
        f"### Root Cause Summary\n\n"
        f"{req.rca_summary}\n\n"
        f"---\n\n"
        f"### Playbook: {req.ansible_description}\n\n"
        f"```yaml\n{req.ansible_playbook}\n```\n\n"
    )

    # ── Test cases (structured from AI) ───────────────────────────────────────
    if req.test_cases:
        pre_cases  = [tc for tc in req.test_cases if tc.get("phase") == "pre"]
        post_cases = [tc for tc in req.test_cases if tc.get("phase") == "post"]
        body += "### 🧪 Test Cases\n\n"
        if pre_cases:
            body += "**Pre-execution validation** (run before applying changes):\n\n"
            for tc in pre_cases:
                body += (
                    f"- [ ] `{tc.get('id', '')}` **{tc.get('name', '')}**"
                    f" — _{tc.get('assertion', '')}_\n"
                )
            body += "\n"
        if post_cases:
            body += "**Post-execution verification** (confirm recovery):\n\n"
            for tc in post_cases:
                body += (
                    f"- [ ] `{tc.get('id', '')}` **{tc.get('name', '')}**"
                    f" — _{tc.get('assertion', '')}_\n"
                )
            body += "\n"
    elif req.test_plan:
        body += "### ✅ Test Plan (what will be validated)\n\n"
        for step in req.test_plan:
            body += f"- {step}\n"
        body += "\n"

    if rollback_steps:
        body += "### ⏪ Rollback Steps (automatic if playbook fails)\n\n"
        for i, step in enumerate(rollback_steps, 1):
            body += f"{i}. {step}\n"
        body += "\n"

    if req.gitea_pr_url:
        from .git_client import GITEA_ORG, GITEA_REPO, GITEA_URL  # noqa: PLC0415
        # Show pre-commit validation summary inline with the PR link
        val_summary = ""
        if req.validation_result:
            vr = req.validation_result
            test_results = vr.get("test_results", [])
            passed  = sum(1 for t in test_results if t.get("status") == "PASSED")
            total   = len(test_results)
            val_icon = "\u2705" if vr.get("all_passed") else "\u26a0\ufe0f"
            val_summary = f" {val_icon} Validation: {passed}/{total} test cases passed"
        body += (
            f"### \U0001f4c1 Gitea \u2014 Ansible Playbook PR\n\n"
            f"The playbook YAML has been **validated**{val_summary} and **committed** to Gitea:\n\n"
            f"| Field | Value |\n|---|---|\n"
            f"| **Pull Request** | [View PR \#{req.gitea_pr_num}]({req.gitea_pr_url}) |\n"
            f"| **Branch** | `{req.gitea_branch}` |\n"
            f"| **Repository** | {GITEA_URL}/{GITEA_ORG}/{GITEA_REPO} |\n\n"
            f"---\n\n"
            f"## \U0001f6a6 ACTION REQUIRED \u2014 TWO STEPS TO AUTHORIZE\n\n"
            f"**Step 1 \u2014 Merge the PR on Gitea:**\n\n"
            f"> \U0001f517 Open [PR \#{req.gitea_pr_num}]({req.gitea_pr_url}), review the "
            f"YAML diff, then click \u2018**Merge Pull Request**\u2019.\n\n"
            f"Merging records your authorisation in Git history.\n\n"
            f"**Step 2 \u2014 Click Approve below:**\n\n"
            f"> After merging, return to this xyOps ticket and click "
            f"**\u25b6 Run \u2018Approve Remediation\u2019** (or use the curl command).\n\n"
            f"The Ansible playbook will execute automatically "
            f"once the PR is merged **and** you click Approve.\n\n"
        )

    if pr_title:
        body += (
            f"### Code/Config Change (GitHub PR)\n\n"
            f"After remediation, raise this PR to prevent recurrence:\n\n"
            f"> **{pr_title}**\n\n"
            f"{pr_description}\n\n"
        )

    # ── Approve / Decline instructions ────────────────────────────────────────
    body += "---\n\n## ▶️ HOW TO APPROVE\n\n"

    if req.approve_event_id and req.decline_event_id:
        body += (
            f"This ticket has two runnable actions attached "
            f"(scroll up to the **Events** section):\n\n"
            f"| Button | Action |\n"
            f"|---|---|\n"
            f"| **▶ Run \"{req.approve_event_id}\"** | Executes the Ansible playbook via ansible-runner |\n"
            f"| **▶ Run \"{req.decline_event_id}\"** | Closes this ticket without making any changes |\n\n"
            f"Click **▶ Run** on the appropriate event above.\n\n"
            f"---\n\n"
            f"_Fallback: use the curl commands below if the events don't appear._\n\n"
        )
    else:
        body += "Use the commands below to approve or decline:\n\n"

    body += (
        f"```powershell\n"
        f'$body = \'{{"approved": true, "decided_by": "YOUR_NAME", "notes": "Reviewed and approved"}}\'\n'
        f"Invoke-RestMethod -Method POST `\n"
        f'  -Uri "{bridge_host}/approval/{approval_id}/decision" `\n'
        f'  -ContentType "application/json" -Body $body\n'
        f"```\n\n"
        f"To decline:\n\n"
        f"```powershell\n"
        f'$body = \'{{"approved": false, "decided_by": "YOUR_NAME", "notes": "Reason for decline"}}\'\n'
        f"Invoke-RestMethod -Method POST `\n"
        f'  -Uri "{bridge_host}/approval/{approval_id}/decision" `\n'
        f'  -ContentType "application/json" -Body $body\n'
        f"```\n\n"
        f"*This ticket was created automatically by the AIOps Bridge.*\n"
    )
    return body
