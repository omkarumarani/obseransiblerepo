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
        rca         = item.get("rca_summary", "")
        playbook    = item.get("ansible_playbook", "")
        val_passed  = item.get("validation_passed", False)
        val_result  = item.get("validation_result", {})
        test_results = val_result.get("test_results", []) if val_result else []
        risk_score  = item.get("risk_score", 0.0)

        with st.container(border=True):
            h1, h2, h3 = st.columns([4, 2, 2])
            h1.markdown(
                f"**`{approval_id}`**  \n"
                f"Agent: **{agent_label}** · Service: **{svc}** · Alert: **{alert or 'N/A'}**  \n"
                f"Ticket: `{ticket or 'N/A'}` · {sev_icon(sev)} {sev.upper()} · 🕐 {created}"
            )

            # ── Validation badge ──────────────────────────────────────────
            if val_passed:
                h2.success("✅ Validation Passed")
            else:
                h2.error("❌ Validation Failed")

            if risk_score:
                h3.metric("Risk Score", f"{risk_score:.2f}")

            # ── Test case results table -----------------------------------------------
            if test_results:
                passed_count = sum(1 for t in test_results if t.get("status") == "PASSED")
                total_count  = len(test_results)
                st.markdown(
                    f"**Playbook Test Results:** {passed_count}/{total_count} passed"
                )
                import pandas as pd
                tc_rows = []
                for tc in test_results:
                    tc_rows.append({
                        "ID":     tc.get("id", "?"),
                        "Name":   tc.get("name", ""),
                        "Phase":  tc.get("phase", ""),
                        "Status": f"{'✅' if tc.get('status') == 'PASSED' else '❌'} {tc.get('status', '?')}",
                        "Detail": tc.get("output", "")[:120],
                    })
                st.dataframe(
                    pd.DataFrame(tc_rows),
                    use_container_width=True,
                    hide_index=True,
                )
            elif val_result:
                st.info("No structured test cases returned — check ansible-runner output.")

            # ── RCA summary ──────────────────────────────────────────
            if rca:
                with st.expander("📋 Root Cause Analysis"):
                    st.markdown(rca)

            # ── Playbook preview ──────────────────────────────────────
            if playbook:
                with st.expander("📄 Ansible Playbook"):
                    st.code(playbook, language="yaml")

            # ── Validation stdout ─────────────────────────────────────
            val_stdout = val_result.get("stdout", "") if val_result else ""
            if val_stdout:
                with st.expander("🔍 Validation Dry-Run Output"):
                    st.code(val_stdout, language="text")

            # ── Approve / Decline buttons (gated on validation) ─────────
            approve_key = f"approve_{agent_label}_{approval_id}"
            reject_key  = f"reject_{agent_label}_{approval_id}"
            decision_id = item.get("session_id", approval_id)

            btn_cols = st.columns([3, 3, 6])
            if val_passed:
                if btn_cols[0].button("✅ Approve", key=approve_key, use_container_width=True):
                    result = api_post(
                        f"{agent_url}/approval/{decision_id}/decision",
                        {"approved": True, "decided_by": "command-center-ui",
                         "notes": "Approved via Streamlit UI"},
                    )
                    if result and "error" not in result:
                        st.toast(f"✅ Approved {approval_id}", icon="✅")
                    else:
                        st.error(f"Approval failed: {result}")
                    st.rerun()
            else:
                btn_cols[0].button(
                    "🔒 Approve (blocked)", key=approve_key,
                    use_container_width=True, disabled=True,
                    help="Approval blocked — playbook validation must pass first",
                )

            if btn_cols[1].button("❌ Decline", key=reject_key, use_container_width=True):
                result = api_post(
                    f"{agent_url}/approval/{decision_id}/decision",
                    {"approved": False, "decided_by": "command-center-ui",
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
