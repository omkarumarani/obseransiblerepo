"""
AIOps Command Center — 🗄️ Storage Pipeline
============================================
Real-time view of the storage-agent pipeline, mirroring the compute
agent tab but specialised for Ceph/storage incident workflows.

Stage display refreshes every 1 second via @st.fragment.
"""

import streamlit as st

from shared import (
    STORAGE_AGENT_URL, XYOPS_EXT, GRAFANA_EXT,
    api_get, api_post, since_str, sev_icon, status_icon,
    page_header, page_footer,
)

st.set_page_config(
    page_title="Storage Pipeline — AIOps",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

page_header("🗄️ Storage Pipeline  (Storage Agent)")

st.caption(
    "⚡ Agent status boxes refresh every **1 second**.  "
    f"SSE stream: `GET {STORAGE_AGENT_URL}/pipeline/events`"
)

# Storage pipeline stage ordering (maps status → display order)
_STORAGE_STAGE_ORDER = [
    "init", "storage_metrics", "logs", "analyzed",
    "ticket_enriched", "awaiting_approval", "autonomous_executing", "complete",
]

# ── Agents displayed (maps to pipeline step names) ────────────────────────────
_STORAGE_AGENTS = [
    ("init",               "1️⃣ Session\nStart"),
    ("storage_metrics",    "2️⃣ Storage\nMetrics"),
    ("logs",               "3️⃣ Log\nFetcher"),
    ("analyzed",           "4️⃣ AI\nAnalyst"),
    ("ticket_enriched",    "5️⃣ Ticket\nWriter"),
    ("complete",           "6️⃣ Approval\nGateway"),
]


def _stage_status(session_status: str, agent_stage: str) -> str:
    """Derive per-agent display status from the overall session status."""
    try:
        current_idx = _STORAGE_STAGE_ORDER.index(session_status)
        agent_idx   = _STORAGE_STAGE_ORDER.index(agent_stage)
    except ValueError:
        return "idle"
    if agent_idx < current_idx:
        return "completed"
    if agent_idx == current_idx:
        return "running"
    return "idle"


# ─── Fragment: 1-second refresh ───────────────────────────────────────────────

@st.fragment(run_every="1s")
def _storage_pipeline_view() -> None:
    live = api_get(f"{STORAGE_AGENT_URL}/pipeline/session/default")

    if not live:
        st.warning("No storage pipeline session found — storage-agent returned no data.")
        st.info(
            "**Manually trigger a storage test alert:**\n"
            "```bash\n"
            "curl -X POST http://localhost:9001/webhook \\\n"
            "  -H 'Content-Type: application/json' \\\n"
            "  -d '{\"status\":\"firing\",\"alerts\":[{\"status\":\"firing\","
            "\"labels\":{\"alertname\":\"StoragePoolNearCapacity\",\"service_name\":\"ceph-cluster\","
            "\"severity\":\"warning\",\"domain\":\"storage\"},\"annotations\":"
            "{\"summary\":\"Test storage alert\"},\"startsAt\":\"2026-03-22T10:00:00Z\"}]}'\n"
            "```"
        )
        return

    s          = live
    sev        = s.get("severity", "info")
    status_val = s.get("status", "init")

    # ── Session header ─────────────────────────────────────────────────────
    c = st.columns(4)
    c[0].metric("Session ID",  s.get("session_id", "-"))
    c[1].metric("Service",     s.get("service_name", "-"))
    c[2].metric(f"Severity {sev_icon(sev)}", sev.upper())
    c[3].metric("Status",      status_val)

    c2 = st.columns(4)
    c2[0].metric("Alert",      s.get("alert_name", "-"))
    c2[1].metric("Risk Score", f"{s.get('risk_score', 0):.2f}  ({s.get('risk_level', '-')})")
    c2[2].metric("Ticket",     f"#{s.get('ticket_num', 0)}" if s.get("ticket_num") else "—")
    c2[3].metric(
        "MTTR",
        f"{s.get('mttr_seconds', 0):.1f}s" if s.get("mttr_seconds") else "⏳ running",
    )

    st.divider()

    # ── 6-Stage Pipeline Visualisation ────────────────────────────────────
    st.subheader("Storage Agent Pipeline  ⚡ live")

    agent_cols = st.columns(6)
    completed_cnt = 0
    for i, (agent_stage, agent_label) in enumerate(_STORAGE_AGENTS):
        ast  = _stage_status(status_val, agent_stage)
        icon = status_icon(ast)
        if ast == "completed":
            completed_cnt += 1
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
        completed_cnt / 6,
        text=f"{completed_cnt} / 6 stages completed",
    )

    st.divider()

    # ── AI Analysis + Ticket details ──────────────────────────────────────
    left_col, right_col = st.columns(2)

    with left_col:
        ticket_num = s.get("ticket_num", 0)
        ticket_id  = s.get("ticket_id", "")
        if ticket_num:
            st.subheader("🎫 Incident Ticket")
            st.markdown(f"**xyOps Ticket:** [#{ticket_num}]({XYOPS_EXT}) `{ticket_id}`")

        approval_id = s.get("approval_id", "")
        if approval_id:
            st.markdown(f"**Approval ID:** `{approval_id}`")

        st.markdown(f"**Summary:** {s.get('summary', '—')}")

        # Approval action buttons
        if status_val == "awaiting_approval" and approval_id:
            st.divider()
            st.subheader("⏳ Pending Approval")
            acol1, acol2 = st.columns(2)
            if acol1.button("✅ Approve", key="stor_approve", use_container_width=True):
                result = api_post(
                    f"{STORAGE_AGENT_URL}/approval/{s.get('session_id')}/decision",
                    {"decision": "approved", "approver": "command-center-ui",
                     "notes": "Approved via Streamlit UI"},
                )
                if result and "error" not in result:
                    st.toast("✅ Storage remediation approved", icon="✅")
                else:
                    st.error(f"Approval failed: {result}")
                st.rerun()
            if acol2.button("❌ Decline", key="stor_decline", use_container_width=True):
                result = api_post(
                    f"{STORAGE_AGENT_URL}/approval/{s.get('session_id')}/decision",
                    {"decision": "declined", "approver": "command-center-ui",
                     "notes": "Declined via Streamlit UI"},
                )
                if result and "error" not in result:
                    st.toast("❌ Storage remediation declined", icon="❌")
                else:
                    st.error(f"Decline failed: {result}")
                st.rerun()

    with right_col:
        analysis = s.get("analysis", {})
        if analysis:
            st.subheader("🧠 AI Analysis")
            root_cause = analysis.get("root_cause", "Analyzing...")
            action     = analysis.get("recommended_action", "Pending...")

            if root_cause and root_cause not in ("Analyzing...", ""):
                st.info(f"**Root Cause:** {root_cause}")
                st.success(f"**Recommended Action:** {action}")
                with st.expander("Analysis details"):
                    st.json({
                        "provider":              analysis.get("provider"),
                        "confidence":            analysis.get("confidence"),
                        "scenario_id":           analysis.get("scenario_id"),
                        "local_validation":      analysis.get("local_validation_status"),
                        "local_validation_conf": analysis.get("local_validation_confidence"),
                        "local_model":           analysis.get("local_model"),
                        "knowledge_similarity":  analysis.get("knowledge_top_similarity"),
                    })
            else:
                st.caption("⏳ AI analysis in progress...")

    with st.expander("🔍 Full raw storage session JSON"):
        st.json(s)


_storage_pipeline_view()

# ─── Storage History ──────────────────────────────────────────────────────────
st.divider()
st.subheader("📜 Storage Pipeline History")

storage_history = api_get(f"{STORAGE_AGENT_URL}/pipeline/history")
hist_items = (storage_history or {}).get("history", [])

if hist_items:
    import pandas as pd
    rows = []
    for x in hist_items:
        ticket_num = x.get("ticket_num", 0)
        rows.append({
            "Session":    x.get("session_id", ""),
            "Service":    x.get("service_name", ""),
            "Alert":      x.get("alert_name", ""),
            "Sev":        x.get("severity", ""),
            "Status":     x.get("status", ""),
            "Autonomy":   x.get("autonomy_decision", "—"),
            "Outcome":    x.get("outcome", "—"),
            "MTTR":       f"{x.get('mttr_seconds', 0):.1f}s" if x.get("mttr_seconds") else "—",
            "xyOps":      f"http://localhost:5522" if ticket_num else "",
            "Started":    since_str(x.get("created_at", 0)),
        })
    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "xyOps": st.column_config.LinkColumn("xyOps Ticket", display_text="Open →"),
        },
    )
else:
    st.info("No storage pipeline history yet.")

page_footer("storage", fragment_mode=True)
