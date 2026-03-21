"""
AIOps Command Center — Overview (main page)
============================================
Multi-page Streamlit dashboard — entry point.

This page: Service Health + Pipeline KPI summary.

Sidebar navigation links to all other pages (pages/ directory):
  🔴 Live Pipeline      — 1-second fragment refresh, SSE stream on compute-agent
  🗄️ Storage Pipeline   — storage-agent pipeline mirror
  ⏳ Pending Approvals  — approve/decline from both agents + Gitea PRs
  ✅ Workflow Outcomes  — drill-down links to xyOps tickets + Grafana
  📜 Pipeline History   — filterable combined compute+storage history
  🤖 Autonomy & Trust   — tier config, trust ladder, decision distribution
  🌐 Agent Mesh         — vis.js topology with bandwidth labels + click-to-drill

Shared config + helpers: shared.py
Environment variables: see shared.py
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from shared import (
    COMPUTE_AGENT_URL, STORAGE_AGENT_URL, OBS_INTELLIGENCE_URL,
    GITEA_ORG, GITEA_REPO,
    api_get, gitea_get, since_str, sev_icon,
    page_header, page_footer,
)

st.set_page_config(
    page_title="AIOps Command Center",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

page_header("🤖 AIOps Command Center")

# ─── Fetch data ────────────────────────────────────────────────────────────────

compute_health    = api_get(f"{COMPUTE_AGENT_URL}/health")
storage_health    = api_get(f"{STORAGE_AGENT_URL}/health")
obs_health        = api_get(f"{OBS_INTELLIGENCE_URL}/health")
pipeline_history  = api_get(f"{COMPUTE_AGENT_URL}/pipeline/history")
pending_approvals = api_get(f"{COMPUTE_AGENT_URL}/approvals/pending")
autonomy_history  = api_get(f"{COMPUTE_AGENT_URL}/autonomy/history")
gitea_prs         = gitea_get(
    f"/api/v1/repos/{GITEA_ORG}/{GITEA_REPO}/pulls?state=open&limit=50&type=pulls"
)

history_items = (pipeline_history or {}).get("history", [])
pending_count = (pending_approvals or {}).get("count", 0)
open_prs      = gitea_prs if isinstance(gitea_prs, list) else []

# ─── Service Health ────────────────────────────────────────────────────────────

st.subheader("Service Health")
h = st.columns(5)
h[0].metric("Compute Agent",    "🟢 healthy" if compute_health else "🔴 offline")
h[1].metric("Storage Agent",    "🟢 healthy" if storage_health else "🔴 offline")
h[2].metric("Obs Intelligence", "🟢 healthy" if obs_health     else "🔴 offline")
h[3].metric("Gitea (Git)",      "🟢 reachable" if gitea_prs is not None else "🔴 offline")
h[4].metric("Pipeline DB",      "🟢 ok" if history_items else "⚠️ empty")

st.divider()

# ─── Pipeline KPIs ────────────────────────────────────────────────────────────

st.subheader("Pipeline KPIs (last 24 h)")
ah             = autonomy_history or {}
total_runs     = len(history_items)
completed_runs = sum(1 for x in history_items if x.get("stage") == "complete")
approved_cnt   = ah.get("approved", 0)
autonomous_cnt = ah.get("autonomous", 0)
declined_cnt   = ah.get("declined", 0)
successes      = ah.get("successes", 0)
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

# ─── Recent Runs + Decision chart ─────────────────────────────────────────────

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

page_footer("overview")
