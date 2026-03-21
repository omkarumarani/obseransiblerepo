"""
AIOps Command Center — 🤖 Autonomy & Trust
============================================
Per-service autonomy tier configuration, trust ladder progress,
and historical decision distribution chart.
"""

import plotly.express as px
import streamlit as st

from shared import (
    COMPUTE_AGENT_URL,
    api_get, since_str,
    page_header, page_footer,
)

st.set_page_config(
    page_title="Autonomy & Trust — AIOps",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

page_header("🤖 Autonomy Engine & Trust Scores")

# ─── Fetch data ────────────────────────────────────────────────────────────────

live_session    = api_get(f"{COMPUTE_AGENT_URL}/pipeline/session/default")
autonomy_history = api_get(f"{COMPUTE_AGENT_URL}/autonomy/history")
autonomy_tiers   = api_get(f"{COMPUTE_AGENT_URL}/autonomy/tiers")

# ─── Service tier table ────────────────────────────────────────────────────────

st.subheader("Service Tier Configuration")

if autonomy_tiers:
    raw_tiers = (
        autonomy_tiers if isinstance(autonomy_tiers, list)
        else autonomy_tiers.get("tiers", autonomy_tiers.get("services", [autonomy_tiers]))
    )
    if isinstance(raw_tiers, list) and raw_tiers:
        import pandas as pd
        tier_rows = []
        for t in raw_tiers:
            tier_rows.append({
                "Service":          t.get("service_name", t.get("name", "-")),
                "Tier":             t.get("tier", "-"),
                "Risk Ceiling":     t.get("risk_ceiling", "-"),
                "Min Approvals":    t.get("min_approvals_for_autonomy", "-"),
                "Min Success Rate": f"{float(t.get('min_success_rate', 0))*100:.0f}%",
                "Auto-execute?":    "✅" if t.get("autonomous") else "❌",
            })
        st.dataframe(pd.DataFrame(tier_rows), use_container_width=True, hide_index=True)
    else:
        st.json(autonomy_tiers)
else:
    st.info("Autonomy tier configuration unavailable.")

st.divider()

# ─── Live trust progress ───────────────────────────────────────────────────────

if live_session and live_session.get("trust_metrics"):
    trust = live_session["trust_metrics"]
    svc   = live_session.get("service_name", "current service")

    st.subheader(f"Trust Ladder: {svc}")

    t = st.columns(4)
    t[0].metric("Approvals Recorded", trust.get("approvals_recorded", 0))
    t[1].metric("Successful Runs",    trust.get("successful_runs", 0))
    t[2].metric("Executed Runs",      trust.get("executed_runs", 0))
    t[3].metric("Success Rate",       f"{trust.get('success_rate', 0)*100:.1f}%")

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
            f"{trust.get('success_rate', 0)*100:.1f}% / "
            f"{next_tier.get('success_rate_needed', 0)*100:.0f}%",
        )
else:
    st.info("No active session with trust metrics. Run a pipeline to see progress.")

st.divider()

# ─── Historical decision distribution ─────────────────────────────────────────

st.subheader("Decision History (last 90 days)")

ah = autonomy_history or {}

if ah:
    import pandas as pd
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
            "Approved":   ah.get("approved", 0),
            "Autonomous": ah.get("autonomous", 0),
            "Declined":   ah.get("declined", 0),
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
                paper_bgcolor="rgba(0,0,0,0)", font_color="#e1e4e8",
                margin=dict(t=20, b=0),
            )
            fig.update_traces(textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No autonomy history data available.")

page_footer("autonomy")
