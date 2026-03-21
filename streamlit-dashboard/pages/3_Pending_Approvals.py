"""
AIOps Command Center — ⏳ Pending Approvals
============================================
Shows all pending human approvals (from both compute and storage agents)
plus open Gitea pull requests.  Approve or Decline directly from this page.
"""

import streamlit as st

from shared import (
    COMPUTE_AGENT_URL, STORAGE_AGENT_URL,
    GITEA_ORG, GITEA_REPO, GITEA_EXT,
    api_get, api_post, gitea_get,
    since_str, sev_icon,
    page_header, page_footer,
)

st.set_page_config(
    page_title="Pending Approvals — AIOps",
    page_icon="⏳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Fetch data ────────────────────────────────────────────────────────────────

compute_pending = api_get(f"{COMPUTE_AGENT_URL}/approvals/pending")
storage_pending = api_get(f"{STORAGE_AGENT_URL}/approvals/pending")
gitea_prs       = gitea_get(
    f"/api/v1/repos/{GITEA_ORG}/{GITEA_REPO}/pulls?state=open&limit=50&type=pulls"
)

compute_items = (compute_pending or {}).get("items", [])
storage_items = (storage_pending or {}).get("items", [])
total_pending = len(compute_items) + len(storage_items)
open_prs      = gitea_prs if isinstance(gitea_prs, list) else []

page_header(f"⏳ Pending Approvals  {'🔴' + str(total_pending) if total_pending else '✅ 0'}")

# ─── KPI row ──────────────────────────────────────────────────────────────────

k = st.columns(3)
k[0].metric("Compute Agent  Approvals", len(compute_items))
k[1].metric("Storage Agent  Approvals", len(storage_items))
k[2].metric("Open Gitea PRs",           len(open_prs))

st.divider()

# ─── Gitea Open PRs ───────────────────────────────────────────────────────────

st.subheader(f"📋 Open Gitea Pull Requests — {len(open_prs)} open")

if open_prs:
    import pandas as pd
    pr_rows = []
    for pr in open_prs:
        pr_rows.append({
            "PR #":    pr.get("number"),
            "Title":   pr.get("title", ""),
            "Branch":  pr.get("head", {}).get("label", pr.get("head", {}).get("ref", "")),
            "Author":  pr.get("user", {}).get("login", ""),
            "Created": since_str(pr.get("created_at", "")),
            "Link":    (pr.get("html_url", "")).replace("gitea:3000", "localhost:3002"),
        })
    pr_df = pd.DataFrame(pr_rows)
    st.dataframe(
        pr_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Link": st.column_config.LinkColumn("Open PR", display_text="View →"),
        },
    )
else:
    st.info("No open Gitea PRs.")

st.divider()


# ─── Approval Queue helper ─────────────────────────────────────────────────────

def _render_approval_queue(items: list, agent_url: str, agent_label: str) -> None:
    st.subheader(f"⏳ {agent_label} Queue — {len(items)} pending")

    if not items:
        st.success(f"✅ No pending approvals from {agent_label}.")
        return

    # Batch breakdown by service
    services = {}
    for item in items:
        svc = item.get("service_name", "unknown")
        services[svc] = services.get(svc, 0) + 1

    sw_cols = st.columns(min(len(services), 5))
    for i, (svc, cnt) in enumerate(services.items()):
        sw_cols[i % 5].metric(f"{svc}", f"{cnt} waiting")

    st.divider()

    for item in items:
        approval_id = item.get("approval_id", item.get("session_id", ""))
        svc         = item.get("service_name", "unknown")
        sev         = item.get("severity", "warning")
        alert       = item.get("alert_name", "")
        created     = since_str(item.get("created_at", ""))
        ticket      = item.get("approval_ticket_id", "")

        with st.container(border=True):
            h1, h2, h3 = st.columns([4, 2, 2])
            h1.markdown(
                f"**`{approval_id}`**  \n"
                f"Agent: **{agent_label}** · Service: **{svc}** · Alert: **{alert or 'N/A'}**  \n"
                f"Ticket: `{ticket or 'N/A'}` · {sev_icon(sev)} {sev.upper()} · 🕐 {created}"
            )

            approve_key = f"approve_{agent_label}_{approval_id}"
            reject_key  = f"reject_{agent_label}_{approval_id}"
            decision_id = item.get("session_id", approval_id)

            if h2.button("✅ Approve", key=approve_key, use_container_width=True):
                result = api_post(
                    f"{agent_url}/approval/{decision_id}/decision",
                    {"decision": "approved", "approver": "command-center-ui",
                     "notes": "Approved via Streamlit UI"},
                )
                if result and "error" not in result:
                    st.toast(f"✅ Approved {approval_id}", icon="✅")
                else:
                    st.error(f"Approval failed: {result}")
                st.rerun()

            if h3.button("❌ Decline", key=reject_key, use_container_width=True):
                result = api_post(
                    f"{agent_url}/approval/{decision_id}/decision",
                    {"decision": "declined", "approver": "command-center-ui",
                     "notes": "Declined via Streamlit UI"},
                )
                if result and "error" not in result:
                    st.toast(f"❌ Declined {approval_id}", icon="❌")
                else:
                    st.error(f"Decline failed: {result}")
                st.rerun()


_render_approval_queue(compute_items, COMPUTE_AGENT_URL, "Compute Agent")
st.divider()
_render_approval_queue(storage_items, STORAGE_AGENT_URL, "Storage Agent")

page_footer("approvals")
