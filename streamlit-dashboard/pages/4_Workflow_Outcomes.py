"""
AIOps Command Center — ✅ Workflow Outcomes
============================================
Per-run end-to-end results with drill-down links to xyOps tickets,
Grafana dashboards, and Ansible run logs.
Also shows the Scenario Confidence Learning (RL feedback) panel.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from shared import (
    COMPUTE_AGENT_URL, OBS_INTELLIGENCE_URL,
    XYOPS_EXT, GRAFANA_EXT,
    api_get, since_str, sev_icon,
    page_header, page_footer,
)

st.set_page_config(
    page_title="Workflow Outcomes — AIOps",
    page_icon="✅",
    layout="wide",
    initial_sidebar_state="expanded",
)

page_header("✅ Workflow Outcomes")

# ─── Fetch data ────────────────────────────────────────────────────────────────

pipeline_history  = api_get(f"{COMPUTE_AGENT_URL}/pipeline/history")
autonomy_history  = api_get(f"{COMPUTE_AGENT_URL}/autonomy/history")
history_items     = (pipeline_history or {}).get("history", [])
ah                = autonomy_history or {}

# ─── KPI row ──────────────────────────────────────────────────────────────────

st.subheader("✅ Workflow Execution Outcomes")

k = st.columns(6)
k[0].metric("Total Decisions",       ah.get("recent_records", 0))
k[1].metric("Approved",              ah.get("approved", 0))
k[2].metric("Autonomous",            ah.get("autonomous", 0))
k[3].metric("Declined",              ah.get("declined", 0))
k[4].metric("Successful Executions", ah.get("successes", 0))
k[5].metric("Failed Executions",     ah.get("failures", 0))

st.divider()

# ─── Per-run end-to-end table with drill-down links ──────────────────────────

st.subheader("End-to-End Pipeline Results")
st.caption("Click the xyOps or Grafana links to drill into the originating ticket or dashboard.")

if not history_items:
    st.info("No pipeline runs recorded yet.")
else:
    outcome_rows = []
    for x in history_items:
        ticket_num    = x.get("ticket_num", 0)
        grafana_url   = ""
        xyops_url     = f"{XYOPS_EXT}/tickets/{ticket_num}" if ticket_num else ""

        # Build Grafana drill-down: link to a pre-filtered Grafana dashboard
        # using the session start time and service name as URL parameters.
        session_id = x.get("session_id", "")
        svc        = x.get("service_name", "")
        if svc:
            grafana_url = (
                f"{GRAFANA_EXT}/d/aiops-incidents?var-service={svc}"
                f"&from=now-1h&to=now"
            )

        outcome_rows.append({
            "Session":      session_id,
            "Service":      svc,
            "Alert":        x.get("alert_name", ""),
            "Severity":     sev_icon(x.get("severity", "")) + " " + x.get("severity", "").upper(),
            "Final Stage":  x.get("stage", ""),
            "Autonomy":     x.get("autonomy_decision", ""),
            "Outcome": {
                "success": "✅ success",
                "failure": "❌ failure",
                "pending": "⏳ pending",
            }.get(x.get("outcome", ""), x.get("outcome", "-")),
            "MTTR":         f"{x.get('mttr_seconds', 0):.1f}s" if x.get("mttr_seconds") else "—",
            "Started":      since_str(x.get("created_at", 0)),
            "xyOps":        xyops_url,
            "Grafana":      grafana_url,
        })

    df_outcomes = pd.DataFrame(outcome_rows)
    st.dataframe(
        df_outcomes,
        use_container_width=True,
        hide_index=True,
        column_config={
            "xyOps":   st.column_config.LinkColumn("xyOps Ticket", display_text="Open →"),
            "Grafana": st.column_config.LinkColumn("Grafana", display_text="Dashboard →"),
        },
    )

    # ── Charts row ────────────────────────────────────────────────────────
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
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e1e4e8", showlegend=False,
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
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e1e4e8", showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ─── Scenario Confidence Learning (RL feedback) ────────────────────────────────

st.subheader("📈 Scenario Confidence Learning")
st.caption(
    "Each recorded outcome (remediation result + LLM validation signal) is fed "
    "back into the scenario correlator using exponential time decay and a "
    "dynamic evidence-tier cap.  Green = confidence boosted, Red = penalised."
)

fb           = api_get(f"{OBS_INTELLIGENCE_URL}/intelligence/feedback-stats")
fb_scenarios = (fb or {}).get("scenarios", [])
fb_global    = (fb or {}).get("global", {})

if fb_global:
    g1, g2, g3, g4, g5 = st.columns(5)
    g1.metric("Scenarios Tracked",  fb_global.get("scenarios_tracked", 0))
    g2.metric("Total Outcomes",     fb_global.get("total_outcomes_recorded", 0))
    g3.metric("Avg Weight Δ",       f"{fb_global.get('avg_weight_adjustment', 0.0):+.3f}")
    g4.metric("↑ Improving",        fb_global.get("scenarios_improving", 0),
              delta=fb_global.get("scenarios_improving", 0) or None)
    g5.metric("↓ Degrading",        fb_global.get("scenarios_degrading", 0),
              delta=(-fb_global.get("scenarios_degrading", 0)) if fb_global.get("scenarios_degrading") else None)

active_scenarios = [s for s in fb_scenarios if s.get("total_seen", 0) > 0]
if active_scenarios:
    df_fb = pd.DataFrame(active_scenarios)
    df_fb["bar_colour"] = df_fb["weight_adjustment"].apply(
        lambda v: "#22c55e" if v > 0.01 else ("#ef4444" if v < -0.01 else "#64748b")
    )
    fig_fb = px.bar(
        df_fb, x="scenario_id", y="weight_adjustment",
        title="Confidence Weight Adjustment per Scenario  (learned from outcomes)",
        color="bar_colour",
        color_discrete_map="identity",
        labels={"weight_adjustment": "Weight Δ", "scenario_id": "Scenario"},
        text="weight_adjustment",
    )
    fig_fb.update_traces(texttemplate="%{text:+.4f}", textposition="outside")
    fig_fb.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e1e4e8", showlegend=False, xaxis_tickangle=-30,
        yaxis=dict(range=[-0.30, 0.30], zeroline=True, zerolinecolor="#374151"),
    )
    st.plotly_chart(fig_fb, use_container_width=True)

    # Per-scenario summary table with xyOps drill-down link
    _TIER_ICON  = {"strong": "🟢", "high": "🟡", "medium": "🔵", "low": "⚪", "none": "—"}
    _TREND_ICON = {"improving": "↗", "degrading": "↘", "stable": "→"}
    table_rows = []
    for s in active_scenarios:
        tier  = s.get("evidence_tier", "none")
        trend = s.get("trend", "stable")
        table_rows.append({
            "Scenario":     s["scenario_id"],
            "Tier":         f"{_TIER_ICON.get(tier,'—')} {tier}",
            "Trend":        f"{_TREND_ICON.get(trend,'?')} {trend}",
            "Weight Δ":     f"{s.get('weight_adjustment', 0.0):+.4f}",
            "Decay Rate":   f"{s.get('decay_success_rate', 0.5):.1%}",
            "Raw Rate":     f"{s.get('success_rate', 0.0):.1%}",
            "Outcomes":     s.get("total_seen", 0),
            "Signals":      s.get("signal_count", 0),
            "Last Updated": (s.get("last_updated") or "")[:16],
        })
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    # Sparklines
    spark_scenarios = [
        s for s in active_scenarios
        if s.get("total_seen", 0) >= 3 and s.get("recent_outcomes")
    ][:3]
    if spark_scenarios:
        st.markdown("**Outcome History (most recent runs, newest right):**")
        sp_cols = st.columns(len(spark_scenarios))
        for i, s in enumerate(spark_scenarios):
            with sp_cols[i]:
                pts = s.get("recent_outcomes", [])
                if pts:
                    df_sp = pd.DataFrame(pts)
                    df_sp["i"] = range(len(df_sp))
                    fig_sp = px.line(
                        df_sp, x="i", y="outcome_value",
                        title=s["scenario_id"], range_y=[-0.1, 1.3], markers=True,
                    )
                    fig_sp.add_hline(y=0.5, line_dash="dot", line_color="#475569",
                                     annotation_text="neutral", annotation_font_size=10)
                    fig_sp.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#e1e4e8", height=200,
                        margin=dict(l=0, r=0, t=30, b=0), showlegend=False,
                        xaxis=dict(title="", showticklabels=False),
                        yaxis=dict(title="", tickvals=[0, 0.5, 1.0],
                                   ticktext=["fail", "partial", "pass"]),
                    )
                    st.plotly_chart(fig_sp, use_container_width=True)
else:
    st.info(
        "No outcome data yet.  Outcomes are recorded automatically when domain agents "
        "report remediation results via `POST /intelligence/record-outcome`.",
        icon="ℹ️",
    )

page_footer("outcomes")
