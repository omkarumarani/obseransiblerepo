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
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

from .git_client import (
    GITEA_ENABLED,
    close_pull_request,
    commit_playbook,
    create_pull_request,
    merge_pull_request,
)
from .xyops_client import create_approval_events

logger = logging.getLogger("aiops-bridge.approval")

ANSIBLE_RUNNER_URL: str = os.getenv("ANSIBLE_RUNNER_URL", "http://ansible-runner:8080")

# ── In-memory state ────────────────────────────────────────────────────────────
# {approval_id: ApprovalRequest}
_pending: dict[str, "ApprovalRequest"] = {}


@dataclass
class ApprovalRequest:
    approval_id: str
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


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

async def request_approval(
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
    )
    _pending[approval_id] = req

    # ── Create Approve / Decline events in xyOps (ticket action buttons) ──────
    bridge_host = os.getenv("BRIDGE_INTERNAL_URL", "http://aiops-bridge:9000")
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

    # ── Gitea: commit playbook to a branch and open a PR ──────────────────────
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

    now = datetime.now(timezone.utc).isoformat()
    req.decided_by = decided_by
    req.decided_at = now

    if not approved:
        req.status = "declined"
        logger.info(
            "Approval %s DECLINED by %s  alert=%s", approval_id, decided_by, req.alert_name
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

    # ── Approved → trigger Ansible execution ───────────────
    req.status = "approved"
    logger.info(
        "Approval %s APPROVED by %s  alert=%s — triggering Ansible",
        approval_id, decided_by, req.alert_name,
    )
    # Merge the Gitea PR now that the human has approved
    if req.gitea_pr_num:
        try:
            merged = await merge_pull_request(req.gitea_pr_num, http)
            logger.info("Gitea PR #%d merged=%s", req.gitea_pr_num, merged)
        except Exception as exc:
            logger.warning("Could not merge Gitea PR: %s", exc)
    approve_msg = (
        f"## Remediation Approved\n\n"
        f"Approved by **{decided_by}** at {now}\n\n"
        f"Notes: {notes or '(none)'}\n\n"
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
    1. POST to /validate (dry-run with test cases) — post results to ticket.
    2. POST to /run (real or simulated execution) — post final results.
    """
    run_payload = {
        "playbook_yaml": req.ansible_playbook,
        "service_name": req.service_name,
        "alert_name": req.alert_name,
        "trace_id": req.bridge_trace_id,
        "test_cases": req.test_cases,
    }

    # ── Step 1: Validate (dry-run) ──────────────────────────────────────────
    try:
        val_resp = await http.post(
            f"{ANSIBLE_RUNNER_URL}/validate",
            json=run_payload,
            timeout=90.0,
        )
        if val_resp.status_code == 200:
            val = val_resp.json()
            test_results = val.get("test_results", [])
            all_passed   = val.get("all_passed", True)
            status_icon  = "✅" if all_passed else "⚠️"
            passed_count = sum(1 for t in test_results if t.get("status") == "PASSED")
            total_count  = len(test_results)

            val_comment = (
                f"## {status_icon} Pre-execution Validation ({passed_count}/{total_count} passed)\n\n"
                f"| Test Case | Status | Detail |\n|---|---|---|\n"
            )
            for tc in test_results:
                icon = "✅" if tc.get("status") == "PASSED" else "❌"
                val_comment += (
                    f"| `{tc.get('id', '?')}` {tc.get('name', '')} "
                    f"| {icon} {tc.get('status', '?')} "
                    f"| {tc.get('output', '')[:80]} |\n"
                )
            val_comment += f"\n```\n{val.get('stdout', '')[:1500]}\n```\n"
            if not all_passed:
                val_comment += "\n> ⚠️ Some pre-checks failed — playbook will still execute, review output carefully.\n"

            await _update_approval_ticket(req=req, comment=val_comment, xyops_post=xyops_post)
    except Exception as exc:
        logger.warning("Validation step failed (non-fatal): %s", exc)
        await _update_approval_ticket(
            req=req,
            comment=f"## ℹ️ Validation Skipped\n\n`{exc}`\n\nProceeding with execution.",
            xyops_post=xyops_post,
        )

    # ── Step 2: Execute ───────────────────────────────────────────────────────
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
            comment = (
                f"## ❌ Ansible Runner Error\n\n"
                f"ansible-runner returned HTTP {resp.status_code}\n\n"
                f"```\n{resp.text[:500]}\n```"
            )
    except Exception as exc:
        req.status = "executed"
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
    else:
        outcome_status = "failed"
        outcome_msg = f"Ansible playbook FAILED (rc={rc}) — manual intervention required"
    await _post_to_incident(req=req, status=outcome_status, message=outcome_msg, xyops_post=xyops_post)


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
    bridge_host = os.getenv("BRIDGE_INTERNAL_URL", "http://aiops-bridge:9000")

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
        body += (
            f"### 📁 Gitea — Ansible Playbook PR\n\n"
            f"The playbook YAML has been committed for code review before execution:\n\n"
            f"| Field | Value |\n|---|---|\n"
            f"| **Pull Request** | [View PR #{req.gitea_pr_num}]({req.gitea_pr_url}) |\n"
            f"| **Branch** | `{req.gitea_branch}` |\n"
            f"| **Repository** | {GITEA_URL}/{GITEA_ORG}/{GITEA_REPO} |\n\n"
            f"Review the YAML diff in Gitea before approving.\n\n"
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
