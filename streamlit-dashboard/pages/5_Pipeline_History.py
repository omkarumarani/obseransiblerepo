"""
AIOps Command Center — 📜 Pipeline History
============================================
Filterable table of all pipeline sessions with per-session analysis
drill-down.  Covers both compute and storage agent histories.
"""

import pandas as pd
import streamlit as st

from shared import (
    COMPUTE_AGENT_URL, STORAGE_AGENT_URL,
    api_get, since_str, sev_icon,
    page_header, page_footer,
)

st.set_page_config(
    page_title="Pipeline History — AIOps",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded",
)

page_header("📜 Pipeline History")

# ─── Fetch history from both agents ───────────────────────────────────────────

compute_history = api_get(f"{COMPUTE_AGENT_URL}/pipeline/history")
storage_history = api_get(f"{STORAGE_AGENT_URL}/pipeline/history")

compute_items = (compute_history or {}).get("history", [])
storage_items = (storage_history or {}).get("history", [])

# Tag agent domain for combined view
for x in compute_items:
    x.setdefault("agent", "compute")
    # Normalise: compute uses "stage", storage uses "status"
    x.setdefault("domain", "compute")

for x in storage_items:
    x["agent"] = "storage"
    x["domain"] = "storage"
    # Normalise stage field
    if "status" in x and "stage" not in x:
        x["stage"] = x["status"]

all_items = sorted(
    compute_items + storage_items,
    key=lambda x: x.get("created_at", 0),
    reverse=True,
)

# ─── Filters ──────────────────────────────────────────────────────────────────

st.subheader(f"📜 Full Pipeline History  ({len(all_items)} sessions)")

if not all_items:
    st.info("No pipeline history yet.")
    page_footer("history")
    st.stop()

f1, f2, f3, f4 = st.columns(4)
all_agents    = sorted({x.get("agent", "compute") for x in all_items})
all_services  = sorted({x.get("service_name", "") for x in all_items})
all_stages    = sorted({x.get("stage", "") for x in all_items})
all_decisions = sorted({x.get("autonomy_decision", "") for x in all_items if x.get("autonomy_decision")})

sel_agents    = f1.multiselect("Agent",             all_agents,    default=all_agents)
sel_services  = f2.multiselect("Service",           all_services,  default=all_services)
sel_stages    = f3.multiselect("Final Stage",       all_stages,    default=all_stages)
sel_decisions = f4.multiselect("Autonomy Decision", all_decisions, default=all_decisions)

filtered = [
    x for x in all_items
    if x.get("agent", "compute") in sel_agents
    and x.get("service_name", "") in sel_services
    and x.get("stage", "") in sel_stages
    and (x.get("autonomy_decision", "") in sel_decisions or not x.get("autonomy_decision"))
]

st.caption(f"Showing {len(filtered)} / {len(all_items)} sessions")

# ─── Expandable per-session cards ─────────────────────────────────────────────

for x in filtered:
    complete  = x.get("stage") in ("complete", "resolved")
    agent_tag = f"[{x.get('agent', 'compute').upper()}]"
    label = (
        f"{'✅' if complete else '⏳'}  "
        f"{sev_icon(x.get('severity', ''))}  "
        f"{agent_tag}  **{x.get('session_id', '')}** — "
        f"{x.get('service_name', '')} / {x.get('alert_name', '')} — "
        f"{since_str(x.get('created_at', 0))}"
    )

    with st.expander(label, expanded=False):
        r1 = st.columns(4)
        r1[0].metric("Stage",      x.get("stage", "-"))
        r1[1].metric("Risk Score", f"{x.get('risk_score', 0):.2f}")
        r1[2].metric("MTTR",       f"{x.get('mttr_seconds', 0):.1f}s" if x.get("mttr_seconds") else "—")
        r1[3].metric("Outcome",    x.get("outcome", "-"))

        r2 = st.columns(4)
        r2[0].metric("Severity",   x.get("severity", "-").upper())
        r2[1].metric("Autonomy",   x.get("autonomy_decision", "-"))
        r2[2].metric("Domain",     x.get("domain", "-"))
        r2[3].metric("Completed",  since_str(x.get("completed_at")) if x.get("completed_at") else "—")

        # Load full analysis on demand (compute sessions only)
        agent = x.get("agent", "compute")
        if agent == "compute":
            agent_url = COMPUTE_AGENT_URL
        else:
            agent_url = STORAGE_AGENT_URL

        if st.button("🔍 Load full analysis", key=f"detail_{x.get('session_id')}_{agent}"):
            full = api_get(f"{agent_url}/pipeline/session/{x.get('session_id')}")
            if full:
                analysis = full.get("analysis", {})
                if analysis and analysis.get("root_cause") not in (None, "Analyzing...", ""):
                    st.info(f"**Root Cause:** {analysis.get('root_cause', '-')}")
                    st.success(f"**Recommended Action:** {analysis.get('recommended_action', '-')}")
                    acols = st.columns(3)
                    acols[0].metric("Provider",   analysis.get("provider", "-"))
                    acols[1].metric("Confidence", f"{analysis.get('confidence', 0):.0%}")
                    acols[2].metric("Scenario",   analysis.get("scenario_id", "-"))
                    st.json({
                        "local_validation":      analysis.get("local_validation_status"),
                        "local_confidence":      analysis.get("local_validation_confidence"),
                        "local_model":           analysis.get("local_model"),
                        "knowledge_similarity":  analysis.get("knowledge_top_similarity"),
                    })
                else:
                    st.warning("Analysis not yet available for this session.")
            else:
                st.error("Could not fetch session details.")

page_footer("history")
