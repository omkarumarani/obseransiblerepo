"""
AIOps Command Center — Streamlit Dashboard
==========================================
Real-time visibility over the full AIOps pipeline:
  Alertmanager → compute-agent → xyOps → Gitea → ansible-runner

Tabs:
  📊 Overview         — summary metrics, service health, activity feed
  🔴 Live Pipeline    — current session: 6 agents, AI analysis, trust score
  ⏳ Pending Approvals — every waiting approval + Gitea open PRs, Approve/Reject
  ✅ Workflow Outcomes — per-run end-to-end results, stage breakdown chart
  📜 Pipeline History  — filterable table with per-session analysis drill-down
  🤖 Autonomy & Trust — per-service tier, trust progress, decision distribution

Environment variables:
  COMPUTE_AGENT_URL   default: http://compute-agent:9000
  STORAGE_AGENT_URL   default: http://storage-agent:9001
  OBS_INTELLIGENCE_URL default: http://obs-intelligence:9100
  GITEA_URL           default: http://gitea:3000
  GITEA_ADMIN_USER    default: aiops
  GITEA_ADMIN_PASS    default: Aiops1234!
  GITEA_ORG           default: aiops-org
  GITEA_REPO          default: ansible-playbooks
  XYOPS_URL           default: http://xyops:5522
  POLL_INTERVAL_SECONDS default: 5
"""

import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# ─── Configuration ─────────────────────────────────────────────────────────────

COMPUTE_AGENT_URL    = os.getenv("COMPUTE_AGENT_URL",    "http://compute-agent:9000")
STORAGE_AGENT_URL    = os.getenv("STORAGE_AGENT_URL",    "http://storage-agent:9001")
OBS_INTELLIGENCE_URL = os.getenv("OBS_INTELLIGENCE_URL", "http://obs-intelligence:9100")
GITEA_URL            = os.getenv("GITEA_URL",            "http://gitea:3000")
GITEA_USER           = os.getenv("GITEA_ADMIN_USER",     "aiops")
GITEA_PASS           = os.getenv("GITEA_ADMIN_PASS",     "Aiops1234!")
GITEA_ORG            = os.getenv("GITEA_ORG",            "aiops-org")
GITEA_REPO           = os.getenv("GITEA_REPO",           "ansible-playbooks")
XYOPS_URL            = os.getenv("XYOPS_URL",            "http://xyops:5522")
POLL_INTERVAL        = int(os.getenv("POLL_INTERVAL_SECONDS", "5"))

# ─── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AIOps Command Center",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  /* tighten column gaps */
  div[data-testid="stHorizontalBlock"] { gap: 0.4rem; }
  /* agent status badges */
  .agent-box {
    background: #1a1f3a;
    border-radius: 8px;
    padding: 0.6rem 0.4rem;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.08);
  }
  .badge-completed { color: #4caf50; font-size: 1.6rem; }
  .badge-running   { color: #ff9800; font-size: 1.6rem; }
  .badge-idle      { color: #555;    font-size: 1.6rem; }
  .badge-failed    { color: #f44336; font-size: 1.6rem; }
  .badge-skipped   { color: #9e9e9e; font-size: 1.6rem; }
  /* severity colours */
  .sev-critical { color: #f44336; font-weight: 700; }
  .sev-warning  { color: #ff9800; font-weight: 700; }
  .sev-info     { color: #2196f3; font-weight: 700; }
  /* pending count badge */
  .pending-badge {
    background: #f44336;
    color: white;
    border-radius: 12px;
    padding: 2px 10px;
    font-weight: bold;
    font-size: 0.85rem;
  }
</style>
""", unsafe_allow_html=True)

# ─── Helpers ───────────────────────────────────────────────────────────────────

def api_get(url: str, timeout: float = 5.0) -> Optional[Dict]:
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def api_post(url: str, payload: Dict, timeout: float = 8.0) -> Optional[Dict]:
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def gitea_get(path: str) -> Optional[Any]:
    try:
        r = requests.get(
            f"{GITEA_URL}{path}",
            auth=(GITEA_USER, GITEA_PASS),
            timeout=5.0,
        )
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def since_str(val) -> str:
    """Return human-readable age string from unix timestamp or ISO string."""
    try:
        if isinstance(val, (int, float)) and val > 0:
            dt = datetime.fromtimestamp(val, tz=timezone.utc)
        else:
            dt = datetime.fromisoformat(str(val).replace("Z", "+00:00"))
        secs = int((datetime.now(timezone.utc) - dt).total_seconds())
        if secs < 60:
            return f"{secs}s ago"
        if secs < 3600:
            return f"{secs // 60}m {secs % 60}s ago"
        return f"{secs // 3600}h {(secs % 3600) // 60}m ago"
    except Exception:
        return str(val)


def sev_icon(sev: str) -> str:
    return {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(str(sev).lower(), "⚪")


def status_icon(st_: str) -> str:
    return {
        "completed": "✅",
        "running":   "⏳",
        "idle":      "⬜",
        "failed":    "❌",
        "skipped":   "⏭️",
    }.get(str(st_).lower(), "⬜")

# ─── Fetch all data ────────────────────────────────────────────────────────────
# All fetched once per rerun; Streamlit caching kept off intentionally so
# every auto-refresh gets fresh data.

live_session      = api_get(f"{COMPUTE_AGENT_URL}/pipeline/session/default")
pipeline_history  = api_get(f"{COMPUTE_AGENT_URL}/pipeline/history")
pending_approvals = api_get(f"{COMPUTE_AGENT_URL}/approvals/pending")
autonomy_history  = api_get(f"{COMPUTE_AGENT_URL}/autonomy/history")
autonomy_tiers    = api_get(f"{COMPUTE_AGENT_URL}/autonomy/tiers")
compute_health    = api_get(f"{COMPUTE_AGENT_URL}/health")
storage_health    = api_get(f"{STORAGE_AGENT_URL}/health")
obs_health        = api_get(f"{OBS_INTELLIGENCE_URL}/health")
gitea_prs         = gitea_get(
    f"/api/v1/repos/{GITEA_ORG}/{GITEA_REPO}/pulls?state=open&limit=50&type=pulls"
)

history_items = pipeline_history.get("history", []) if pipeline_history else []
pending_items = pending_approvals.get("items", []) if pending_approvals else []
pending_count = pending_approvals.get("count", 0) if pending_approvals else 0
open_prs      = gitea_prs if isinstance(gitea_prs, list) else []

# ─── Header ────────────────────────────────────────────────────────────────────

title_col, time_col = st.columns([5, 1])
title_col.title("🤖 AIOps Command Center")
time_col.markdown(
    f"<br><small style='color:#8b949e'>Refreshed {datetime.now().strftime('%H:%M:%S')}</small>",
    unsafe_allow_html=True,
)

# ─── Tabs ──────────────────────────────────────────────────────────────────────

pending_label = f"⏳ Pending Approvals"
if pending_count:
    pending_label = f"⏳ Pending Approvals 🔴{pending_count}"

tab_overview, tab_live, tab_approvals, tab_outcomes, tab_history, tab_autonomy, tab_mesh = st.tabs([
    "📊 Overview",
    "🔴 Live Pipeline",
    pending_label,
    "✅ Workflow Outcomes",
    "📜 Pipeline History",
    "🤖 Autonomy & Trust",
    "🌐 Agent Mesh",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB: Overview
# ══════════════════════════════════════════════════════════════════════════════

with tab_overview:

    # ── Service Health ──────────────────────────────────────────────────────
    st.subheader("Service Health")
    h = st.columns(5)
    h[0].metric("Compute Agent",     "🟢 healthy" if compute_health else "🔴 offline")
    h[1].metric("Storage Agent",     "🟢 healthy" if storage_health else "🔴 offline")
    h[2].metric("Obs Intelligence",  "🟢 healthy" if obs_health     else "🔴 offline")
    h[3].metric("Gitea (Git)",       "🟢 reachable" if gitea_prs is not None else "🔴 offline")
    h[4].metric("Pipeline DB",       "🟢 ok" if history_items else "⚠️ empty")

    st.divider()

    # ── Pipeline KPIs ───────────────────────────────────────────────────────
    st.subheader("Pipeline KPIs (last 24 h)")
    ah = autonomy_history or {}
    total_runs     = len(history_items)
    completed_runs = sum(1 for x in history_items if x.get("stage") == "complete")
    approved_cnt   = ah.get("approved", 0)
    autonomous_cnt = ah.get("autonomous", 0)
    declined_cnt   = ah.get("declined", 0)
    successes      = ah.get("successes", 0)
    failures       = ah.get("failures", 0)
    total_decisions = ah.get("recent_records", 0) or 1
    success_rate   = round(successes / total_decisions * 100, 1)
    mttrs          = [x.get("mttr_seconds", 0) for x in history_items if x.get("mttr_seconds", 0) > 0]
    avg_mttr       = round(sum(mttrs) / len(mttrs), 1) if mttrs else 0

    k = st.columns(7)
    k[0].metric("Total Incidents",       total_runs)
    k[1].metric("Completed",             completed_runs)
    k[2].metric("⏳ Pending Approvals",  pending_count,
                delta=f"{len(open_prs)} open PRs" if open_prs else None)
    k[3].metric("Approved by Human",     approved_cnt)
    k[4].metric("Autonomous Executions", autonomous_cnt)
    k[5].metric("Success Rate",          f"{success_rate}%")
    k[6].metric("Avg MTTR",              f"{avg_mttr}s")

    st.divider()

    # ── Recent Runs + Distribution chart side by side ───────────────────────
    left, right = st.columns([3, 1])

    with left:
        st.subheader("Recent Pipeline Runs")
        if history_items:
            rows = []
            for x in history_items:
                rows.append({
                    "Session":   x.get("session_id", ""),
                    "Service":   x.get("service_name", ""),
                    "Alert":     x.get("alert_name", ""),
                    "Sev":       x.get("severity", ""),
                    "Stage":     x.get("stage", ""),
                    "Autonomy":  x.get("autonomy_decision", ""),
                    "MTTR":      f"{x.get('mttr_seconds', 0):.1f}s" if x.get("mttr_seconds") else "⏳",
                    "Started":   since_str(x.get("created_at", 0)),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No pipeline history yet. Trigger a Prometheus alert to start.")

    with right:
        st.subheader("Decision Split")
        dist = {"Approved": approved_cnt, "Autonomous": autonomous_cnt, "Declined": declined_cnt}
        dist = {k: v for k, v in dist.items() if v > 0}
        if dist:
            fig = px.pie(
                values=list(dist.values()),
                names=list(dist.keys()),
                color_discrete_sequence=["#4caf50", "#2196f3", "#f44336"],
                hole=0.4,
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e1e4e8",
                margin=dict(t=20, b=0, l=0, r=0),
                showlegend=True,
                legend=dict(orientation="h"),
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No decision data yet.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB: Live Pipeline
# ══════════════════════════════════════════════════════════════════════════════

with tab_live:

    if not live_session:
        st.warning("No pipeline session found — compute-agent returned no data.")
        st.info(
            "**Manually trigger a test alert:**\n"
            "```bash\n"
            "curl -X POST http://localhost:9000/webhook \\\n"
            "  -H 'Content-Type: application/json' \\\n"
            "  -d '{\"status\":\"firing\",\"alerts\":[{\"status\":\"firing\","
            "\"labels\":{\"alertname\":\"HighErrorRate\",\"service_name\":\"frontend-api\","
            "\"severity\":\"warning\"},\"annotations\":{\"summary\":\"Test alert\"},"
            "\"startsAt\":\"2026-03-22T10:00:00Z\"}]}'\n"
            "```"
        )
    else:
        s = live_session
        sev  = s.get("severity", "info")
        stage = s.get("stage", "unknown")
        status = s.get("status", "unknown")

        # ── Session header row ────────────────────────────────────────────
        c = st.columns(4)
        c[0].metric("Session ID",  s.get("session_id", "-"))
        c[1].metric("Service",     s.get("service_name", "-"))
        c[2].metric(f"Severity {sev_icon(sev)}", sev.upper())
        c[3].metric("Stage",       stage)

        c2 = st.columns(4)
        c2[0].metric("Alert",        s.get("alert_name", "-"))
        c2[1].metric("Status",       status.upper())
        c2[2].metric("Risk Score",   f"{s.get('risk_score', 0):.2f}   ({s.get('risk_level', '-')})")
        c2[3].metric(
            "MTTR",
            f"{s.get('mttr_seconds', 0):.1f}s" if s.get("mttr_seconds") else "⏳ running",
        )

        st.divider()

        # ── 6-Agent Pipeline Visualisation ───────────────────────────────
        st.subheader("Agent Pipeline")

        AGENTS = [
            ("ticket-creator",   "1️⃣ Ticket\nCreator"),
            ("log-fetcher",      "2️⃣ Log\nFetcher"),
            ("metrics-fetcher",  "3️⃣ Metrics\nFetcher"),
            ("ai-analyst",       "4️⃣ AI\nAnalyst"),
            ("ticket-writer",    "5️⃣ Ticket\nWriter"),
            ("approval-gateway", "6️⃣ Approval\nGateway"),
        ]

        agents_data   = {a["name"]: a for a in s.get("agents", [])}
        completed_cnt = sum(1 for a in s.get("agents", []) if a.get("status") == "completed")
        total_agents  = len(s.get("agents", [])) or 6

        agent_cols = st.columns(6)
        for i, (agent_key, agent_label) in enumerate(AGENTS):
            agent  = agents_data.get(agent_key, {"name": agent_key, "status": "idle"})
            ast    = agent.get("status", "idle")
            icon   = status_icon(ast)
            with agent_cols[i]:
                st.markdown(
                    f'<div class="agent-box">'
                    f'<div style="font-size:0.75rem;font-weight:600;margin-bottom:4px">'
                    f'{agent_label}</div>'
                    f'<div class="badge-{ast}">{icon}</div>'
                    f'<div style="font-size:0.7rem;color:#8b949e;margin-top:4px">'
                    f'{ast.upper()}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.progress(
            completed_cnt / total_agents,
            text=f"{completed_cnt} / {total_agents} agents completed",
        )

        st.divider()

        # ── Incident Info + AI Analysis in two columns ────────────────────
        left_col, right_col = st.columns(2)

        with left_col:
            incident = s.get("incident", {})
            if incident:
                st.subheader("Incident Details")
                st.markdown(f"**Service:** `{incident.get('service_name', '-')}`")
                st.markdown(f"**Alert:** `{incident.get('alert_name', '-')}`")
                st.markdown(f"**Severity:** {sev_icon(incident.get('severity', ''))} {incident.get('severity', '').upper()}")
                st.markdown(f"**Risk Score:** `{incident.get('risk_score', 0):.2f}`")
                grafana = incident.get("grafana_url", "")
                if grafana:
                    link = grafana.replace("grafana:3000", "localhost:3001")
                    st.markdown(f"[📊 Open Grafana Dashboard]({link})")

            # Ticket info
            ticket_num = s.get("ticket_num", 0)
            ticket_id  = s.get("ticket_id", "")
            if ticket_num:
                st.markdown(f"**xyOps Ticket:** [#{ticket_num}](http://localhost:5522) `{ticket_id}`")

            approval_id  = s.get("approval_id", "")
            approval_num = s.get("approval_ticket_num", 0)
            if approval_id:
                st.markdown(f"**Approval ID:** `{approval_id}`")
            if approval_num:
                st.markdown(f"**Approval Ticket:** [#{approval_num}](http://localhost:5522)")

            st.markdown(f"**Approval Required:** {'✅ Yes' if s.get('approval_required') else '❌ No'}")

        with right_col:
            analysis = s.get("analysis", {})
            if analysis:
                st.subheader("🧠 AI Analysis")
                root_cause = analysis.get("root_cause", "Analyzing...")
                action     = analysis.get("recommended_action", "Pending...")

                if root_cause and root_cause != "Analyzing...":
                    st.info(f"**Root Cause:** {root_cause}")
                    st.success(f"**Recommended Action:** {action}")
                    with st.expander("Analysis details"):
                        st.json({
                            "provider":               analysis.get("provider"),
                            "model":                  analysis.get("model"),
                            "confidence":             analysis.get("confidence"),
                            "scenario_id":            analysis.get("scenario_id"),
                            "scenario_confidence":    analysis.get("scenario_confidence"),
                            "local_validation":       analysis.get("local_validation_status"),
                            "local_validation_conf":  analysis.get("local_validation_confidence"),
                            "local_model":            analysis.get("local_model"),
                            "knowledge_similarity":   analysis.get("knowledge_top_similarity"),
                        })
                else:
                    st.caption("⏳ AI analysis in progress...")

        st.divider()

        # ── Trust & Autonomy ─────────────────────────────────────────────
        trust = s.get("trust_metrics")
        if trust:
            st.subheader("🏆 Trust & Autonomy Progress")
            t = st.columns(4)
            t[0].metric("Autonomy Decision",    s.get("autonomy_decision", "-"))
            t[1].metric("Approvals Recorded",   trust.get("approvals_recorded", 0))
            t[2].metric("Success Rate",         f"{trust.get('success_rate', 0) * 100:.1f}%")
            t[3].metric("Progress",             s.get("trust_progress", "-"))

            next_tier = trust.get("next_tier", {})
            if next_tier:
                needed   = max(next_tier.get("approvals_needed", 1), 1)
                recorded = trust.get("approvals_recorded", 0)
                pct      = min(recorded / needed, 1.0)
                st.progress(pct, text=trust.get("path_to_next_tier", ""))

        # ── Full JSON toggle ─────────────────────────────────────────────
        with st.expander("🔍 Full raw session JSON"):
            st.json(s)

# ══════════════════════════════════════════════════════════════════════════════
# TAB: Pending Approvals
# ══════════════════════════════════════════════════════════════════════════════

with tab_approvals:

    # ── Gitea Open PRs ─────────────────────────────────────────────────────
    st.subheader(f"📋 Open Gitea Pull Requests — {len(open_prs)} open")

    if open_prs:
        pr_rows = []
        for pr in open_prs:
            pr_rows.append({
                "PR #":    pr.get("number"),
                "Title":   pr.get("title", ""),
                "Branch":  pr.get("head", {}).get("label", pr.get("head", {}).get("ref", "")),
                "Author":  pr.get("user", {}).get("login", ""),
                "Created": since_str(pr.get("created_at", "")),
            })
        pr_df = pd.DataFrame(pr_rows)
        st.dataframe(pr_df, use_container_width=True, hide_index=True)

        st.markdown("**Direct links:**")
        link_cols = st.columns(min(len(open_prs), 3))
        for i, pr in enumerate(open_prs[:6]):
            url = pr.get("html_url", "").replace("gitea:3000", "localhost:3002")
            link_cols[i % 3].markdown(f"[PR #{pr.get('number')}: {pr.get('title', '')[:50]}]({url})")
        if len(open_prs) > 6:
            st.caption(f"… and {len(open_prs) - 6} more")
    else:
        st.info("No open Gitea PRs. All playbooks have been reviewed or no alerts processed yet.")

    st.divider()

    # ── Approval Queue ─────────────────────────────────────────────────────
    st.subheader(f"⏳ Approval Queue — {pending_count} pending")

    if not pending_items:
        st.success("✅ No pending approvals — everything is resolved!")
    else:
        # Batch summary
        services_waiting = {}
        for item in pending_items:
            svc = item.get("service_name", "unknown")
            services_waiting[svc] = services_waiting.get(svc, 0) + 1

        sw_cols = st.columns(min(len(services_waiting), 5))
        for i, (svc, cnt) in enumerate(services_waiting.items()):
            sw_cols[i % 5].metric(f"Service: {svc}", f"{cnt} waiting")

        st.divider()
        st.caption("💡 Approve or reject each pending remediation below.")

        for item in pending_items:
            approval_id = item.get("approval_id", "")
            svc         = item.get("service_name", "unknown")
            sev         = item.get("severity", "warning")
            alert       = item.get("alert_name", "")
            created     = since_str(item.get("created_at", ""))
            ticket      = item.get("approval_ticket_id", "")

            with st.container(border=True):
                h1, h2, h3 = st.columns([4, 2, 2])
                h1.markdown(
                    f"**`{approval_id}`**  \n"
                    f"Service: **{svc}** · Alert: **{alert or 'N/A'}**  \n"
                    f"Ticket: `{ticket or 'N/A'}` · {sev_icon(sev)} {sev.upper()} · 🕐 {created}"
                )

                approve_key = f"approve_{approval_id}"
                reject_key  = f"reject_{approval_id}"

                if h2.button("✅ Approve", key=approve_key, use_container_width=True):
                    result = api_post(
                        f"{COMPUTE_AGENT_URL}/approval/{approval_id}/decision",
                        {"decision": "approved", "approver": "command-center-ui", "notes": "Approved via Streamlit UI"},
                    )
                    if result and "error" not in result:
                        st.toast(f"✅ Approved {approval_id}", icon="✅")
                    else:
                        st.error(f"Approval failed: {result}")
                    st.rerun()

                if h3.button("❌ Decline", key=reject_key, use_container_width=True):
                    result = api_post(
                        f"{COMPUTE_AGENT_URL}/approval/{approval_id}/decision",
                        {"decision": "declined", "approver": "command-center-ui", "notes": "Declined via Streamlit UI"},
                    )
                    if result and "error" not in result:
                        st.toast(f"❌ Declined {approval_id}", icon="❌")
                    else:
                        st.error(f"Decline failed: {result}")
                    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB: Workflow Outcomes
# ══════════════════════════════════════════════════════════════════════════════

with tab_outcomes:
    st.subheader("✅ Workflow Execution Outcomes")

    ah = autonomy_history or {}

    # ── KPI row ────────────────────────────────────────────────────────────
    k = st.columns(6)
    k[0].metric("Total Decisions",       ah.get("recent_records", 0))
    k[1].metric("Approved",              ah.get("approved", 0))
    k[2].metric("Autonomous",            ah.get("autonomous", 0))
    k[3].metric("Declined",              ah.get("declined", 0))
    k[4].metric("Successful Executions", ah.get("successes", 0))
    k[5].metric("Failed Executions",     ah.get("failures", 0))

    st.divider()

    # ── Per-run end-to-end table ────────────────────────────────────────────
    st.subheader("End-to-End Pipeline Results")

    if not history_items:
        st.info("No pipeline runs recorded yet.")
    else:
        outcome_rows = []
        for x in history_items:
            completed_ts = x.get("completed_at")
            outcome_rows.append({
                "Session":     x.get("session_id", ""),
                "Service":     x.get("service_name", ""),
                "Alert":       x.get("alert_name", ""),
                "Severity":    sev_icon(x.get("severity", "")) + " " + x.get("severity", "").upper(),
                "Final Stage": x.get("stage", ""),
                "Autonomy":    x.get("autonomy_decision", ""),
                "Outcome":     {
                    "success": "✅ success",
                    "failure": "❌ failure",
                    "pending": "⏳ pending",
                }.get(x.get("outcome", ""), x.get("outcome", "-")),
                "MTTR":        f"{x.get('mttr_seconds', 0):.1f}s" if x.get("mttr_seconds") else "—",
                "Started":     since_str(x.get("created_at", 0)),
                "Completed":   since_str(completed_ts) if completed_ts else "⏳ running",
            })

        df_outcomes = pd.DataFrame(outcome_rows)
        st.dataframe(df_outcomes, use_container_width=True, hide_index=True)

        # ── Charts row ─────────────────────────────────────────────────────
        ch1, ch2 = st.columns(2)

        with ch1:
            stage_counts = (
                pd.DataFrame(history_items)["stage"]
                .value_counts()
                .reset_index()
            )
            stage_counts.columns = ["Stage", "Count"]
            fig = px.bar(
                stage_counts, x="Stage", y="Count",
                title="Sessions by Final Stage",
                color="Stage",
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e1e4e8",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        with ch2:
            autonomy_counts = (
                pd.DataFrame(history_items)["autonomy_decision"]
                .value_counts()
                .reset_index()
            )
            autonomy_counts.columns = ["Decision", "Count"]
            fig2 = px.bar(
                autonomy_counts, x="Decision", y="Count",
                title="Sessions by Autonomy Decision",
                color="Decision",
                color_discrete_sequence=["#4caf50", "#2196f3", "#ff9800", "#f44336"],
            )
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e1e4e8",
                showlegend=False,
            )
            st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB: Pipeline History
# ══════════════════════════════════════════════════════════════════════════════

with tab_history:
    st.subheader("📜 Full Pipeline History")

    if not history_items:
        st.info("No pipeline history yet.")
    else:
        # ── Filters ───────────────────────────────────────────────────────
        f1, f2, f3 = st.columns(3)
        all_services  = sorted({x.get("service_name", "") for x in history_items})
        all_stages    = sorted({x.get("stage", "") for x in history_items})
        all_decisions = sorted({x.get("autonomy_decision", "") for x in history_items})

        sel_services  = f1.multiselect("Service",          all_services,  default=all_services)
        sel_stages    = f2.multiselect("Final Stage",      all_stages,    default=all_stages)
        sel_decisions = f3.multiselect("Autonomy Decision", all_decisions, default=all_decisions)

        filtered = [
            x for x in history_items
            if x.get("service_name", "") in sel_services
            and x.get("stage", "") in sel_stages
            and x.get("autonomy_decision", "") in sel_decisions
        ]

        st.caption(f"Showing {len(filtered)} / {len(history_items)} sessions")

        # ── Expandable per-session cards ──────────────────────────────────
        for x in filtered:
            complete = x.get("stage") == "complete"
            label = (
                f"{'✅' if complete else '⏳'}  "
                f"{sev_icon(x.get('severity', ''))}  "
                f"**{x.get('session_id', '')}** — "
                f"{x.get('service_name', '')} / {x.get('alert_name', '')} — "
                f"{since_str(x.get('created_at', 0))}"
            )
            with st.expander(label, expanded=False):
                r1 = st.columns(4)
                r1[0].metric("Stage",         x.get("stage", "-"))
                r1[1].metric("Risk Score",    f"{x.get('risk_score', 0):.2f}")
                r1[2].metric("MTTR",          f"{x.get('mttr_seconds', 0):.1f}s" if x.get("mttr_seconds") else "—")
                r1[3].metric("Outcome",       x.get("outcome", "-"))

                r2 = st.columns(4)
                r2[0].metric("Severity",      x.get("severity", "-").upper())
                r2[1].metric("Autonomy",      x.get("autonomy_decision", "-"))
                r2[2].metric("Domain",        x.get("domain", "-"))
                r2[3].metric("Completed",     since_str(x.get("completed_at")) if x.get("completed_at") else "—")

                # Load full analysis on demand
                if st.button("🔍 Load full analysis", key=f"detail_{x.get('session_id')}"):
                    full = api_get(f"{COMPUTE_AGENT_URL}/pipeline/session/{x.get('session_id')}")
                    if full:
                        analysis = full.get("analysis", {})
                        if analysis and analysis.get("root_cause") and analysis.get("root_cause") != "Analyzing...":
                            st.info(f"**Root Cause:** {analysis.get('root_cause', '-')}")
                            st.success(f"**Recommended Action:** {analysis.get('recommended_action', '-')}")
                            acols = st.columns(3)
                            acols[0].metric("Provider",   analysis.get("provider", "-"))
                            acols[1].metric("Confidence", f"{analysis.get('confidence', 0):.0%}")
                            acols[2].metric("Scenario",   analysis.get("scenario_id", "-"))
                            st.json({
                                "local_validation":  analysis.get("local_validation_status"),
                                "local_confidence":  analysis.get("local_validation_confidence"),
                                "local_model":       analysis.get("local_model"),
                                "knowledge_similarity": analysis.get("knowledge_top_similarity"),
                            })
                        else:
                            st.warning("Analysis not yet available for this session.")
                    else:
                        st.error("Could not fetch session details.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB: Autonomy & Trust
# ══════════════════════════════════════════════════════════════════════════════

with tab_autonomy:
    st.subheader("🤖 Autonomy Engine & Trust Scores")

    # ── Service tier table ─────────────────────────────────────────────────
    if autonomy_tiers:
        st.subheader("Service Tier Configuration")
        raw_tiers = (
            autonomy_tiers if isinstance(autonomy_tiers, list)
            else autonomy_tiers.get("tiers", autonomy_tiers.get("services", [autonomy_tiers]))
        )
        if isinstance(raw_tiers, list) and raw_tiers:
            tier_rows = []
            for t in raw_tiers:
                tier_rows.append({
                    "Service":           t.get("service_name", t.get("name", "-")),
                    "Tier":              t.get("tier", "-"),
                    "Risk Ceiling":      t.get("risk_ceiling", "-"),
                    "Min Approvals":     t.get("min_approvals_for_autonomy", "-"),
                    "Min Success Rate":  f"{float(t.get('min_success_rate', 0))*100:.0f}%",
                    "Auto-execute?":     "✅" if t.get("autonomous") else "❌",
                })
            st.dataframe(pd.DataFrame(tier_rows), use_container_width=True, hide_index=True)
        else:
            st.json(autonomy_tiers)
    else:
        st.info("Autonomy tier configuration unavailable.")

    st.divider()

    # ── Live trust progress for current session ────────────────────────────
    if live_session and live_session.get("trust_metrics"):
        trust = live_session["trust_metrics"]
        svc   = live_session.get("service_name", "current service")
        st.subheader(f"Trust Ladder: {svc}")

        t = st.columns(4)
        t[0].metric("Approvals Recorded",  trust.get("approvals_recorded", 0))
        t[1].metric("Successful Runs",     trust.get("successful_runs", 0))
        t[2].metric("Executed Runs",       trust.get("executed_runs", 0))
        t[3].metric("Success Rate",        f"{trust.get('success_rate', 0)*100:.1f}%")

        next_tier = trust.get("next_tier", {})
        if next_tier:
            needed   = max(next_tier.get("approvals_needed", 1), 1)
            recorded = trust.get("approvals_recorded", 0)
            pct      = min(recorded / needed, 1.0)
            tier_name = next_tier.get("name", "next tier")
            st.progress(pct, text=f"Progress to {tier_name}: {trust.get('path_to_next_tier', '')}")
            g1, g2 = st.columns(2)
            g1.metric(
                "Approvals: Have / Need",
                f"{recorded} / {needed}",
                delta=f"Need {needed - recorded} more" if needed > recorded else "✅ threshold met",
                delta_color="inverse",
            )
            g2.metric(
                "Success Rate: Have / Need",
                f"{trust.get('success_rate', 0)*100:.1f}% / {next_tier.get('success_rate_needed', 0)*100:.0f}%",
            )

    st.divider()

    # ── Historical decision distribution ──────────────────────────────────
    ah = autonomy_history or {}
    st.subheader("Decision History (last 90 days)")

    if ah:
        stat_col, chart_col = st.columns([1, 1])

        with stat_col:
            st.markdown(f"""
| Metric | Value |
|--------|-------|
| Total Records | **{ah.get('total_records', 0)}** |
| Recent (90d)  | **{ah.get('recent_records', 0)}** |
| Approved      | **{ah.get('approved', 0)}** |
| Autonomous    | **{ah.get('autonomous', 0)}** |
| Declined      | **{ah.get('declined', 0)}** |
| Successes     | **{ah.get('successes', 0)}** |
| Failures      | **{ah.get('failures', 0)}** |
| Services seen | `{', '.join(ah.get('services', []))}` |
""")

        with chart_col:
            dist = {
                "Approved":  ah.get("approved", 0),
                "Autonomous": ah.get("autonomous", 0),
                "Declined":  ah.get("declined", 0),
            }
            dist = {k: v for k, v in dist.items() if v > 0}
            if dist:
                fig = px.pie(
                    values=list(dist.values()),
                    names=list(dist.keys()),
                    color_discrete_sequence=["#4caf50", "#2196f3", "#f44336"],
                    hole=0.45,
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#e1e4e8",
                    margin=dict(t=20, b=0),
                )
                fig.update_traces(textinfo="percent+label")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No autonomy history data available.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB: Agent Mesh  — live topology showing agents + platforms + data flows
# ══════════════════════════════════════════════════════════════════════════════

def build_agent_mesh_html(session: Optional[Dict], health: Optional[Dict]) -> str:
    """
    Build a vis.js Network graph showing the AIOps agent mesh.
    Node border/background colour changes based on the current pipeline stage.
    Active edges animate (dashes flow) when data is flowing on that link.
    """
    stage = (session or {}).get("stage", "")

    # ── Determine which nodes are active right now ─────────────────────────────
    # stage → set of node IDs that are currently "working"
    STAGE_ACTIVE = {
        "started":            {"alertmanager", "compute"},
        "logs":               {"compute", "loki"},
        "metrics":            {"compute", "prometheus"},
        "analyzed":           {"compute", "obs_intel", "local_llm"},
        "ticket_enriched":    {"compute", "xyops"},
        "awaiting_approval":  {"compute", "gitea"},
        "autonomous_executing": {"compute", "ansible"},
        "complete":           set(),
    }
    active_nodes = STAGE_ACTIVE.get(stage, set())

    # Completed stages — nodes that are done
    STAGE_ORDER = ["started","logs","metrics","analyzed","ticket_enriched",
                   "awaiting_approval","autonomous_executing","complete"]
    stage_idx = STAGE_ORDER.index(stage) if stage in STAGE_ORDER else -1

    STAGE_COMPLETE_NODES = {
        0: set(),
        1: {"alertmanager","compute"},
        2: {"alertmanager","compute","loki"},
        3: {"alertmanager","compute","loki","prometheus"},
        4: {"alertmanager","compute","loki","prometheus","obs_intel","local_llm"},
        5: {"alertmanager","compute","loki","prometheus","obs_intel","local_llm","xyops"},
        6: {"alertmanager","compute","loki","prometheus","obs_intel","local_llm","xyops","gitea"},
        7: {"alertmanager","compute","loki","prometheus","obs_intel","local_llm","xyops","gitea","ansible"},
    }
    done_nodes = STAGE_COMPLETE_NODES.get(stage_idx, set())

    svc_h = (health or {}).get("services", {})

    def node_colors(nid: str):
        if nid in active_nodes:
            return "#ff9800", "#fff3e0", 3     # orange fill, light border, thick border
        if nid in done_nodes:
            return "#1b5e20", "#4caf50", 2     # dark-green fill, green border
        # check if the actual service is down
        svc_map = {
            "compute":    "compute-agent",
            "obs_intel":  "obs-intelligence",
        }
        svc_key = svc_map.get(nid)
        if svc_key and svc_h.get(svc_key) == "unhealthy":
            return "#b71c1c", "#ef9a9a", 2
        return "#1a1f3a", "#37474f", 1         # default dark

    def edge_active(a: str, b: str) -> bool:
        return a in active_nodes and b in active_nodes

    # ── Build node + edge JSON strings ────────────────────────────────────────
    def make_node(nid, label, shape, size, x, y, icon=None):
        bg, border_col, bw = node_colors(nid)
        glow = "true" if nid in active_nodes else "false"
        font_col = "#ffffff"
        node = {
            "id": nid, "label": label, "shape": shape, "size": size,
            "x": x, "y": y, "fixed": True,
            "color": {
                "background": bg,
                "border": border_col,
                "highlight": {"background": bg, "border": "#ffeb3b"},
            },
            "borderWidth": bw,
            "font": {"color": font_col, "size": 13, "face": "monospace"},
            "shadow": {"enabled": glow == "true", "color": "#ff9800", "size": 15, "x": 0, "y": 0},
        }
        return node

    nodes = [
        make_node("alertmanager", "🚨\nAlertmanager",  "box",     60, -500,  0),
        make_node("compute",      "🤖\nCompute Agent", "box",     80,  -50,  0),
        make_node("prometheus",   "📈\nPrometheus",    "ellipse", 55,  300, -250),
        make_node("loki",         "📋\nLoki",          "ellipse", 55,  300,  250),
        make_node("obs_intel",    "🧠\nObs-Intelligence", "box",  65,  300,    0),
        make_node("local_llm",    "🦙\nLocal LLM\n(qwen3.5)", "ellipse", 50, 600, 0),
        make_node("xyops",        "🎫\nxyOps",         "box",     60, -200,  300),
        make_node("gitea",        "🔀\nGitea\n(PR/Approval)", "box", 55, -200, -300),
        make_node("ansible",      "⚙️\nAnsible Runner","box",     55, -450,  300),
    ]

    def make_edge(src, dst, label="", dashed=False):
        active = edge_active(src, dst)
        done_edge = src in done_nodes and dst in done_nodes
        col = "#ff9800" if active else ("#4caf50" if done_edge else "#37474f")
        width = 4 if active else (2 if done_edge else 1)
        anim = "true" if active else "false"
        e = {
            "from": src, "to": dst, "label": label,
            "arrows": "to",
            "color": {"color": col, "highlight": "#ffeb3b"},
            "width": width,
            "dashes": dashed or active,
            "font": {"color": "#90caf9", "size": 11, "align": "middle"},
            "smooth": {"type": "curvedCW", "roundness": 0.2},
        }
        return e

    edges = [
        make_edge("alertmanager", "compute",   "webhook"),
        make_edge("compute",      "loki",      "fetch logs"),
        make_edge("compute",      "prometheus","fetch metrics"),
        make_edge("compute",      "obs_intel", "AI analysis"),
        make_edge("obs_intel",    "local_llm", "corroborate"),
        make_edge("local_llm",    "obs_intel", "verdict",     dashed=True),
        make_edge("obs_intel",    "compute",   "enriched result"),
        make_edge("compute",      "xyops",     "create ticket"),
        make_edge("compute",      "gitea",     "create PR"),
        make_edge("gitea",        "compute",   "approved ✓",  dashed=True),
        make_edge("compute",      "ansible",   "run playbook"),
        make_edge("ansible",      "compute",   "result",      dashed=True),
        make_edge("ansible",      "xyops",     "update ticket"),
    ]

    import json as _json
    nodes_json = _json.dumps(nodes)
    edges_json = _json.dumps(edges)

    # Current stage label + session info
    alive = session is not None
    stage_label = stage.replace("_", " ").upper() if stage else "IDLE"
    svc  = (session or {}).get("service_name", "—")
    alrt = (session or {}).get("alert_name",   "—")
    sev  = (session or {}).get("severity",     "—")
    risk = (session or {}).get("risk_score",   0)
    dec  = (session or {}).get("autonomy_decision", "—")
    sid  = (session or {}).get("session_id",   "—")

    stage_color = "#ff9800" if stage not in ("complete","") else "#4caf50"

    # Stage-to-narrative map
    narratives = {
        "started":            "🚨 Alert received. Compute agent opening pipeline session.",
        "logs":               "📋 Compute agent querying Loki for recent log lines.",
        "metrics":            "📈 Compute agent fetching Prometheus metrics.",
        "analyzed":           "🧠 Obs-Intelligence running AI analysis + Local LLM corroboration.",
        "ticket_enriched":    "🎫 Enriched ticket created and updated in xyOps.",
        "awaiting_approval":  "🔀 PR created in Gitea. Waiting for human approval.",
        "autonomous_executing":"⚙️ Autonomous execution — Ansible playbook running.",
        "complete":           "✅ Pipeline complete. All agents idle.",
        "":                   "😴 No active pipeline. Waiting for next alert.",
    }
    narrative = narratives.get(stage, "Processing…")

    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<script src="https://unpkg.com/vis-network@9.1.9/dist/vis-network.min.js"></script>
<style>
  body  {{ margin:0; padding:0; background:#0a0e27; color:#e1e4e8; font-family: monospace; }}
  #net  {{ width:100%; height:520px; border:1px solid #1e2a4a; border-radius:8px; }}
  #info {{ padding:12px 16px; background:#111827; border-radius:8px; margin-top:8px;
           display:flex; gap:24px; flex-wrap:wrap; align-items:center; }}
  .info-item {{ display:flex; flex-direction:column; }}
  .info-label {{ font-size:11px; color:#64748b; text-transform:uppercase; letter-spacing:.05em; }}
  .info-val   {{ font-size:14px; font-weight:700; color:#e1e4e8; margin-top:2px; }}
  .stage-badge {{
    display:inline-block; padding:4px 14px; border-radius:20px;
    background:{stage_color}22; border:1px solid {stage_color};
    color:{stage_color}; font-weight:700; font-size:13px;
  }}
  #narrative {{
    margin-top:8px; padding:10px 16px;
    background:#0d1b2a; border-left:3px solid {stage_color};
    border-radius:0 8px 8px 0; font-size:13px; color:#90caf9;
  }}
  .legend {{ display:flex; gap:16px; margin-top:8px; font-size:12px; flex-wrap:wrap; }}
  .leg-item {{ display:flex; align-items:center; gap:6px; }}
  .leg-dot  {{ width:12px; height:12px; border-radius:50%; }}
</style>
</head>
<body>

<div id="net"></div>

<div id="info">
  <div class="info-item">
    <span class="info-label">Pipeline Stage</span>
    <span class="stage-badge">{stage_label}</span>
  </div>
  <div class="info-item">
    <span class="info-label">Session</span>
    <span class="info-val">{sid}</span>
  </div>
  <div class="info-item">
    <span class="info-label">Service</span>
    <span class="info-val">{svc}</span>
  </div>
  <div class="info-item">
    <span class="info-label">Alert</span>
    <span class="info-val">{alrt}</span>
  </div>
  <div class="info-item">
    <span class="info-label">Severity</span>
    <span class="info-val">{sev.upper()}</span>
  </div>
  <div class="info-item">
    <span class="info-label">Risk Score</span>
    <span class="info-val">{risk:.2f}</span>
  </div>
  <div class="info-item">
    <span class="info-label">Autonomy</span>
    <span class="info-val">{dec}</span>
  </div>
</div>

<div id="narrative">{narrative}</div>

<div class="legend">
  <div class="leg-item"><div class="leg-dot" style="background:#ff9800;"></div> Active now</div>
  <div class="leg-item"><div class="leg-dot" style="background:#4caf50;"></div> Completed</div>
  <div class="leg-item"><div class="leg-dot" style="background:#1a1f3a; border:1px solid #37474f;"></div> Idle</div>
  <div class="leg-item"><div class="leg-dot" style="background:#b71c1c;"></div> Unhealthy</div>
</div>

<script>
const nodes = new vis.DataSet({nodes_json});
const edges = new vis.DataSet({edges_json});

const container = document.getElementById("net");
const options = {{
  physics: {{ enabled: false }},
  interaction: {{ dragNodes: false, zoomView: true, dragView: true }},
  nodes: {{ margin: 10 }},
  edges: {{
    smooth: {{ type: "curvedCW", roundness: 0.2 }},
    arrows: {{ to: {{ enabled: true, scaleFactor: 0.8 }} }},
  }},
}};

const network = new vis.Network(container, {{ nodes, edges }}, options);
network.fit({{ animation: false }});
</script>
</body>
</html>
"""


with tab_mesh:
    st.subheader("🌐 Agent Mesh — Live Data Flow")
    st.caption("Shows which agents and platforms are actively communicating right now. Orange = active, Green = done, Dark = idle.")

    # ── Fetch all data needed ──────────────────────────────────────────────────
    mesh_session = api_get(f"{COMPUTE_AGENT_URL}/pipeline/session/default")
    mesh_health  = api_get(f"{COMPUTE_AGENT_URL}/health")

    # ── vis.js topology ───────────────────────────────────────────────────────
    html_src = build_agent_mesh_html(mesh_session, mesh_health)
    st.components.v1.html(html_src, height=680, scrolling=False)

    st.divider()

    # ── Live agent activity feed ───────────────────────────────────────────────
    st.subheader("📡 Live Agent Activity Feed")

    if mesh_session:
        stage   = mesh_session.get("stage", "")
        agents  = mesh_session.get("agents", [])
        anal    = mesh_session.get("analysis", {})
        metrics = mesh_session.get("metrics", {})
        logs_raw = mesh_session.get("logs", "")

        # Two-column layout: left = agent status grid, right = data details
        left, right = st.columns([1, 1])

        with left:
            st.markdown("**Agent Status**")
            AGENT_ICONS = {
                "completed": "✅", "running": "🟠", "idle": "⚪",
                "failed": "❌", "skipped": "⏭️",
            }
            AGENT_LABELS = {
                "ticket-creator":   ("🚨", "Alertmanager → Compute",     "Receives alert webhook, opens session"),
                "log-fetcher":      ("📋", "Compute → Loki",             "Queries Loki for log stream"),
                "metrics-fetcher":  ("📈", "Compute → Prometheus",       "Runs PromQL metric queries"),
                "ai-analyst":       ("🧠", "Compute → Obs-Intelligence", "AI root-cause analysis + LLM corroboration"),
                "ticket-writer":    ("🎫", "Compute → xyOps",            "Creates/updates enriched incident ticket"),
                "approval-gateway": ("🔀", "Compute → Gitea",            "Creates PR, waits for human approval"),
            }
            for ag in agents:
                name   = ag.get("name", "")
                status = ag.get("status", "idle")
                icon   = AGENT_ICONS.get(status, "⚪")
                lbl = AGENT_LABELS.get(name, ("🔧", name, ""))
                border = "border-left: 3px solid #ff9800;" if status == "running" else \
                         "border-left: 3px solid #4caf50;" if status == "completed" else \
                         "border-left: 3px solid #37474f;"
                st.markdown(
                    f'<div style="background:#111827;{border}padding:8px 12px;'
                    f'border-radius:0 6px 6px 0;margin-bottom:6px;">'
                    f'<span style="font-size:18px">{icon}</span> '
                    f'<b style="color:#e1e4e8">{lbl[1]}</b><br/>'
                    f'<span style="color:#64748b;font-size:12px">{lbl[2]}</span></div>',
                    unsafe_allow_html=True,
                )

        with right:
            st.markdown("**Platform Communication Details**")

            # AI Analysis output
            if anal and anal.get("root_cause") not in (None, "Analyzing...", ""):
                with st.expander("🧠 Obs-Intelligence → Compute Agent  (AI Analysis)", expanded=True):
                    c1, c2 = st.columns(2)
                    c1.metric("Confidence",     f"{anal.get('confidence', 0):.0%}")
                    c2.metric("Provider",       anal.get("provider", "—"))
                    c1.metric("Scenario",       anal.get("scenario_id", "—"))
                    c2.metric("LLM Verdict",    anal.get("local_validation_status", "—") or "—")
                    st.markdown(f"**Root Cause:** {anal.get('root_cause','—')}")
                    st.markdown(f"**Recommended Action:** {anal.get('recommended_action','—')}")

            # Metrics fetched from Prometheus
            if metrics:
                with st.expander("📈 Prometheus → Compute Agent  (Fetched Metrics)"):
                    mdf = pd.DataFrame(
                        [{"metric": k, "value": v} for k, v in metrics.items()]
                    )
                    if not mdf.empty:
                        st.dataframe(mdf, use_container_width=True, hide_index=True)
                    else:
                        st.info("No metric values returned yet.")

            # Log lines from Loki
            if logs_raw and logs_raw.strip():
                with st.expander("📋 Loki → Compute Agent  (Log Lines)"):
                    lines = [l for l in logs_raw.split("\n") if l.strip()]
                    st.code("\n".join(lines[-30:]), language="text")

            # Approval status
            approval_id  = mesh_session.get("approval_id", "")
            approval_num = mesh_session.get("approval_ticket_num", 0)
            if stage in ("awaiting_approval",) or approval_id:
                with st.expander("🔀 Gitea PR / Approval Gate", expanded=stage == "awaiting_approval"):
                    st.markdown(f"**Approval ID:** `{approval_id or '—'}`")
                    st.markdown(f"**xyOps Approval Ticket:** #{approval_num}" if approval_num else "**xyOps Approval Ticket:** —")
                    st.markdown(f"**Decision:** {mesh_session.get('autonomy_decision','—')}")
                    pr_url = f"{GITEA_URL.replace('gitea:3000','localhost:3002')}/aiops-org/ansible-playbooks/pulls"
                    st.markdown(f"[View open PRs →]({pr_url})")

            # Trust progress toward autonomous tier
            trust = mesh_session.get("trust_progress", "")
            tm    = mesh_session.get("trust_metrics", {})
            if tm:
                with st.expander("📊 Autonomy Engine  (Trust Metrics)"):
                    ta, tb, tc = st.columns(3)
                    ta.metric("Approvals Recorded", tm.get("approvals_recorded", 0))
                    tb.metric("Success Rate",        f"{tm.get('success_rate',0):.0%}")
                    tc.metric("Executed Runs",       tm.get("executed_runs", 0))
                    st.caption(tm.get("path_to_next_tier", ""))
    else:
        st.info("No active pipeline session. Trigger an alert and the mesh will light up.")

    # ── Cross-domain correlation panel ────────────────────────────────────────
    st.divider()
    st.subheader("🔗 Cross-Domain Correlation")

    xd = api_get(f"{OBS_INTELLIGENCE_URL}/intelligence/correlation/current")
    xd_assessment = (xd or {}).get("assessment")

    if xd_assessment:
        ctype   = xd_assessment.get("correlation_type", "UNKNOWN")
        primary = xd_assessment.get("primary_domain", "unknown")
        risk_lv = xd_assessment.get("combined_risk_level", "unknown").upper()
        risk_sc = float(xd_assessment.get("combined_risk_score", 0.0))
        urgency = xd_assessment.get("urgency", "—")
        detected_at = xd_assessment.get("detected_at", "")
        narrative   = xd_assessment.get("narrative", "")
        chain   = xd_assessment.get("causal_chain") or []
        actions = xd_assessment.get("unified_recommended_actions") or []
        evidence= xd_assessment.get("evidence") or []
        compute_svc = xd_assessment.get("compute_service", "—")
        storage_svc = xd_assessment.get("storage_service", "—")
        compute_scn = xd_assessment.get("compute_scenario", "—")
        storage_scn = xd_assessment.get("storage_scenario", "—")

        _CTYPE_COLOUR = {
            "STORAGE_ROOT":           "#ef4444",  # red    — storage caused compute
            "COMPUTE_ROOT":           "#f97316",  # orange — compute caused storage
            "SHARED_INFRASTRUCTURE":  "#eab308",  # yellow — shared fault
            "INDEPENDENT_CONCURRENT": "#64748b",  # grey   — coincidental
        }
        _RISK_COLOUR = {"CRITICAL": "#ef4444", "HIGH": "#f97316", "MEDIUM": "#eab308", "LOW": "#22c55e"}
        badge_c = _CTYPE_COLOUR.get(ctype, "#64748b")
        badge_r = _RISK_COLOUR.get(risk_lv, "#64748b")

        st.markdown(
            f'<div style="background:#1e293b;border-left:4px solid {badge_c};padding:14px 18px;'
            f'border-radius:0 8px 8px 0;margin-bottom:12px;">'
            f'<span style="background:{badge_c};color:#fff;padding:2px 8px;border-radius:4px;'
            f'font-size:12px;font-weight:700">{ctype}</span>&nbsp;&nbsp;'
            f'<span style="background:{badge_r};color:#fff;padding:2px 8px;border-radius:4px;'
            f'font-size:12px;font-weight:700">{risk_lv} {risk_sc:.2f}</span>&nbsp;&nbsp;'
            f'<span style="color:#94a3b8;font-size:12px">urgency: <b>{urgency}</b>'
            f' &nbsp;|&nbsp; primary: <b>{primary}</b>'
            f' &nbsp;|&nbsp; {detected_at[:19].replace("T"," ") if detected_at else ""}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Service / scenario row
        xa, xb = st.columns(2)
        xa.markdown(
            f"**💻 Compute**  \n`{compute_svc}`  \nScenario: `{compute_scn}`"
        )
        xb.markdown(
            f"**🗄️ Storage**  \n`{storage_svc}`  \nScenario: `{storage_scn}`"
        )

        if narrative:
            st.info(narrative)

        xc1, xc2, xc3 = st.columns(3)
        with xc1:
            with st.expander("🔗 Causal Chain", expanded=True):
                if chain:
                    for i, step in enumerate(chain, 1):
                        st.markdown(f"**{i}.** {step}")
                else:
                    st.caption("No chain data.")
        with xc2:
            with st.expander("🛠️ Recommended Actions", expanded=True):
                if actions:
                    for act in actions:
                        st.markdown(f"- {act}")
                else:
                    st.caption("No actions.")
        with xc3:
            with st.expander("🔍 Evidence", expanded=False):
                if evidence:
                    for ev in evidence:
                        st.markdown(f"- {ev}")
                else:
                    st.caption("No evidence.")

        # Raw JSON for debugging
        with st.expander("📄 Raw unified_assessment JSON", expanded=False):
            st.json(xd_assessment)
    else:
        st.success(
            "No active cross-domain correlation — compute and storage are operating independently.",
            icon="✅",
        )
        st.caption("This panel updates automatically when both agents fire simultaneously within a 2-minute window.")

    # ── Pending approvals mini-feed ────────────────────────────────────────────
    st.divider()
    st.subheader("⏳ Pending Approvals (Gitea PRs waiting)")
    pend = api_get(f"{COMPUTE_AGENT_URL}/approvals/pending")
    pend_items = (pend or {}).get("items", [])
    if pend_items:
        pdf = pd.DataFrame(pend_items)
        if "created_at" in pdf.columns:
            pdf["waiting"] = pdf["created_at"].apply(since_str)
        cols_show = [c for c in ["approval_id","service_name","alert_name","severity","waiting"] if c in pdf.columns]
        st.dataframe(pdf[cols_show], use_container_width=True, hide_index=True)
        st.caption(f"**{len(pend_items)} approvals** waiting for human sign-off in Gitea.")
    else:
        st.success("No pending approvals — pipeline queue is clear.")


# ─── Footer: auto-refresh ─────────────────────────────────────────────────────
st.divider()
fc1, fc2 = st.columns([4, 1])
with fc1:
    auto = st.toggle("⏱ Auto-refresh every 5 seconds", value=True, key="auto_refresh")
with fc2:
    if st.button("🔄 Refresh now", use_container_width=True):
        st.rerun()

if auto:
    time.sleep(POLL_INTERVAL)
    st.rerun()
