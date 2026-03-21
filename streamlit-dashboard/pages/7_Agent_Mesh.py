"""
AIOps Command Center — 🌐 Agent Mesh
======================================
Live vis.js topology graph showing agent-to-agent data flows.

New in v14:
  • Edge bandwidth labels — actual data sizes (log lines, metric count,
    AI confidence, etc.) shown on each active edge.
  • Click-to-drill — clicking a node opens its service UI in a new tab.
  • Cross-domain correlation panel.
"""

import json as _json
from typing import Optional, Dict

import pandas as pd
import streamlit as st

from shared import (
    COMPUTE_AGENT_URL, OBS_INTELLIGENCE_URL, GITEA_URL,
    GITEA_EXT, XYOPS_EXT, GRAFANA_EXT, PROMETHEUS_EXT,
    ALERTMANAGER_EXT, COMPUTE_EXT, STORAGE_EXT, OBS_INTEL_EXT, ANSIBLE_EXT,
    api_get, since_str, sev_icon,
    page_header, page_footer,
)

st.set_page_config(
    page_title="Agent Mesh — AIOps",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

page_header("🌐 Agent Mesh — Live Data Flow")
st.caption(
    "Orange = active now · Green = completed · Dark = idle  |  "
    "**Click a node** to open its service UI in a new browser tab."
)


def build_agent_mesh_html(session: Optional[Dict], health: Optional[Dict]) -> str:
    """
    vis.js Network graph with:
    - Stage-coloured nodes + glow shadow for active nodes
    - Dynamic bandwidth labels on active edges (log lines, metric count, etc.)
    - JS click handler to open service UIs in new tabs
    """
    stage = (session or {}).get("stage", "")

    # ── Derive bandwidth metadata from session ─────────────────────────────────
    logs_raw   = (session or {}).get("logs", "")
    metrics    = (session or {}).get("metrics", {})
    analysis   = (session or {}).get("analysis", {})

    log_line_count  = len([l for l in logs_raw.split("\n") if l.strip()]) if logs_raw else 0
    metric_count    = len(metrics)
    ai_conf         = analysis.get("confidence", 0) or 0
    local_verdict   = analysis.get("local_validation_status", "") or ""
    local_conf      = analysis.get("local_validation_confidence", 0) or 0

    # ── Stage → active node set ────────────────────────────────────────────────
    STAGE_ACTIVE = {
        "started":              {"alertmanager", "compute"},
        "logs":                 {"compute", "loki"},
        "metrics":              {"compute", "prometheus"},
        "analyzed":             {"compute", "obs_intel", "local_llm"},
        "ticket_enriched":      {"compute", "xyops"},
        "awaiting_approval":    {"compute", "gitea"},
        "autonomous_executing": {"compute", "ansible"},
        "complete":             set(),
    }
    active_nodes = STAGE_ACTIVE.get(stage, set())

    STAGE_ORDER = [
        "started", "logs", "metrics", "analyzed", "ticket_enriched",
        "awaiting_approval", "autonomous_executing", "complete",
    ]
    stage_idx = STAGE_ORDER.index(stage) if stage in STAGE_ORDER else -1

    STAGE_COMPLETE_NODES = {
        0: set(),
        1: {"alertmanager", "compute"},
        2: {"alertmanager", "compute", "loki"},
        3: {"alertmanager", "compute", "loki", "prometheus"},
        4: {"alertmanager", "compute", "loki", "prometheus", "obs_intel", "local_llm"},
        5: {"alertmanager", "compute", "loki", "prometheus", "obs_intel", "local_llm", "xyops"},
        6: {"alertmanager", "compute", "loki", "prometheus", "obs_intel", "local_llm", "xyops", "gitea"},
        7: {"alertmanager", "compute", "loki", "prometheus", "obs_intel", "local_llm", "xyops", "gitea", "ansible"},
    }
    done_nodes = STAGE_COMPLETE_NODES.get(stage_idx, set())
    svc_h = (health or {}).get("services", {})

    def node_colors(nid: str):
        if nid in active_nodes:
            return "#ff9800", "#fff3e0", 3
        if nid in done_nodes:
            return "#1b5e20", "#4caf50", 2
        svc_map = {"compute": "compute-agent", "obs_intel": "obs-intelligence"}
        svc_key = svc_map.get(nid)
        if svc_key and svc_h.get(svc_key) == "unhealthy":
            return "#b71c1c", "#ef9a9a", 2
        return "#1a1f3a", "#37474f", 1

    def edge_active(a: str, b: str) -> bool:
        return a in active_nodes and b in active_nodes

    def make_node(nid, label, shape, size, x, y, url=""):
        bg, border_col, bw = node_colors(nid)
        glow = nid in active_nodes
        node = {
            "id": nid, "label": label, "shape": shape, "size": size,
            "x": x, "y": y, "fixed": True,
            "color": {
                "background": bg, "border": border_col,
                "highlight": {"background": bg, "border": "#ffeb3b"},
            },
            "borderWidth": bw,
            "font": {"color": "#ffffff", "size": 13, "face": "monospace"},
            "shadow": {"enabled": glow, "color": "#ff9800", "size": 15, "x": 0, "y": 0},
            "title": f"Click to open {label.splitlines()[1] if chr(10) in label else label}" if url else "",
        }
        return node

    nodes = [
        make_node("alertmanager", "🚨\nAlertmanager",       "box",     60, -500,  0,  ALERTMANAGER_EXT),
        make_node("compute",      "🤖\nCompute Agent",      "box",     80,  -50,  0,  COMPUTE_EXT),
        make_node("prometheus",   "📈\nPrometheus",         "ellipse", 55,  300, -250, PROMETHEUS_EXT),
        make_node("loki",         "📋\nLoki",               "ellipse", 55,  300,  250, GRAFANA_EXT + "/explore"),
        make_node("obs_intel",    "🧠\nObs-Intelligence",   "box",     65,  300,    0, OBS_INTEL_EXT + "/docs"),
        make_node("local_llm",    "🦙\nLocal LLM\n(qwen3)", "ellipse", 50,  600,    0, ""),
        make_node("xyops",        "🎫\nxyOps",              "box",     60, -200,  300, XYOPS_EXT),
        make_node("gitea",        "🔀\nGitea\n(PR/Approval)","box",    55, -200, -300, GITEA_EXT),
        make_node("ansible",      "⚙️\nAnsible Runner",     "box",     55, -450,  300, ANSIBLE_EXT),
    ]

    # ── Bandwidth labels per edge ─────────────────────────────────────────────
    def bw_label(src: str, dst: str, static: str) -> str:
        """Return a dynamic bandwidth label when the edge is currently active."""
        if not edge_active(src, dst):
            return static
        if src == "alertmanager" and dst == "compute":
            return "1 alert"
        if src == "compute" and dst == "loki":
            return f"{log_line_count} lines" if log_line_count else "fetch logs"
        if src == "compute" and dst == "prometheus":
            return f"{metric_count} metrics" if metric_count else "fetch metrics"
        if src == "obs_intel" and dst == "local_llm":
            return "corroborate"
        if src == "local_llm" and dst == "obs_intel":
            ll = f"{local_verdict}" if local_verdict else "verdict"
            return f"{ll} {local_conf:.0%}" if local_conf and local_verdict else ll
        if src == "obs_intel" and dst == "compute":
            return f"conf:{ai_conf:.0%}" if ai_conf else "enriched"
        return static

    def make_edge(src, dst, static_label="", dashed=False):
        active    = edge_active(src, dst)
        done_edge = src in done_nodes and dst in done_nodes
        col       = "#ff9800" if active else ("#4caf50" if done_edge else "#37474f")
        width     = 4 if active else (2 if done_edge else 1)
        label     = bw_label(src, dst, static_label)
        return {
            "from": src, "to": dst, "label": label,
            "arrows": "to",
            "color": {"color": col, "highlight": "#ffeb3b"},
            "width": width,
            "dashes": dashed or active,
            "font": {"color": "#90caf9", "size": 11, "align": "middle", "strokeWidth": 0},
            "smooth": {"type": "curvedCW", "roundness": 0.2},
        }

    edges = [
        make_edge("alertmanager", "compute",   "webhook"),
        make_edge("compute",      "loki",      "fetch logs"),
        make_edge("compute",      "prometheus","fetch metrics"),
        make_edge("compute",      "obs_intel", "AI analysis"),
        make_edge("obs_intel",    "local_llm", "corroborate"),
        make_edge("local_llm",    "obs_intel", "verdict",          dashed=True),
        make_edge("obs_intel",    "compute",   "enriched result"),
        make_edge("compute",      "xyops",     "create ticket"),
        make_edge("compute",      "gitea",     "create PR"),
        make_edge("gitea",        "compute",   "approved ✓",       dashed=True),
        make_edge("compute",      "ansible",   "run playbook"),
        make_edge("ansible",      "compute",   "result",           dashed=True),
        make_edge("ansible",      "xyops",     "update ticket"),
    ]

    nodes_json = _json.dumps(nodes)
    edges_json = _json.dumps(edges)

    # Click-to-drill URL map (JS side)
    node_urls = {
        "alertmanager": ALERTMANAGER_EXT,
        "compute":      COMPUTE_EXT + "/docs",
        "prometheus":   PROMETHEUS_EXT,
        "loki":         GRAFANA_EXT + "/explore",
        "obs_intel":    OBS_INTEL_EXT + "/docs",
        "xyops":        XYOPS_EXT,
        "gitea":        GITEA_EXT,
        "ansible":      ANSIBLE_EXT,
        "local_llm":    "",
    }
    urls_json = _json.dumps(node_urls)

    # Stage info for HUD
    alive       = session is not None
    stage_label = stage.replace("_", " ").upper() if stage else "IDLE"
    svc         = (session or {}).get("service_name", "—")
    alrt        = (session or {}).get("alert_name",   "—")
    sev         = (session or {}).get("severity",     "—")
    risk        = (session or {}).get("risk_score",   0)
    dec         = (session or {}).get("autonomy_decision", "—")
    sid         = (session or {}).get("session_id",   "—")
    stage_color = "#ff9800" if stage not in ("complete", "") else "#4caf50"

    narratives = {
        "started":              "🚨 Alert received. Compute agent opening pipeline session.",
        "logs":                 "📋 Compute agent querying Loki for recent log lines.",
        "metrics":              "📈 Compute agent fetching Prometheus metrics.",
        "analyzed":             "🧠 Obs-Intelligence running AI analysis + Local LLM corroboration.",
        "ticket_enriched":      "🎫 Enriched ticket created and updated in xyOps.",
        "awaiting_approval":    "🔀 PR created in Gitea. Waiting for human approval.",
        "autonomous_executing": "⚙️ Autonomous execution — Ansible playbook running.",
        "complete":             "✅ Pipeline complete. All agents idle.",
        "":                     "😴 No active pipeline. Waiting for next alert.",
    }
    narrative = narratives.get(stage, "Processing…")

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<script src="https://unpkg.com/vis-network@9.1.9/dist/vis-network.min.js"></script>
<style>
  body  {{ margin:0; padding:0; background:#0a0e27; color:#e1e4e8; font-family:monospace; }}
  #net  {{ width:100%; height:520px; border:1px solid #1e2a4a; border-radius:8px; cursor:pointer; }}
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
  #drill-hint {{ margin-top:6px; font-size:11px; color:#64748b; text-align:center; }}
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
    <span class="info-label">Risk&nbsp;Score</span>
    <span class="info-val">{risk:.2f}</span>
  </div>
  <div class="info-item">
    <span class="info-label">Autonomy</span>
    <span class="info-val">{dec}</span>
  </div>
  <div class="info-item">
    <span class="info-label">Log Lines</span>
    <span class="info-val">{log_line_count if log_line_count else '—'}</span>
  </div>
  <div class="info-item">
    <span class="info-label">Metrics</span>
    <span class="info-val">{metric_count if metric_count else '—'}</span>
  </div>
</div>

<div id="narrative">{narrative}</div>
<div id="drill-hint">💡 Click a node to open its service UI in a new tab</div>

<div class="legend">
  <div class="leg-item"><div class="leg-dot" style="background:#ff9800;"></div> Active now</div>
  <div class="leg-item"><div class="leg-dot" style="background:#4caf50;"></div> Completed</div>
  <div class="leg-item"><div class="leg-dot" style="background:#1a1f3a; border:1px solid #37474f;"></div> Idle</div>
  <div class="leg-item"><div class="leg-dot" style="background:#b71c1c;"></div> Unhealthy</div>
  <div class="leg-item" style="color:#90caf9">📊 Edge labels = live data size when active</div>
</div>

<script>
const NODE_URLS = {urls_json};

const nodes = new vis.DataSet({nodes_json});
const edges = new vis.DataSet({edges_json});
const container = document.getElementById("net");
const options = {{
  physics: {{ enabled: false }},
  interaction: {{ dragNodes: false, zoomView: true, dragView: true, hover: true }},
  nodes: {{ margin: 10 }},
  edges: {{
    smooth: {{ type: "curvedCW", roundness: 0.2 }},
    arrows: {{ to: {{ enabled: true, scaleFactor: 0.8 }} }},
  }},
}};

const network = new vis.Network(container, {{ nodes, edges }}, options);
network.fit({{ animation: false }});

// ── Click-to-drill: open service UI on node click ──────────────────────────
network.on("click", function(params) {{
  if (params.nodes.length > 0) {{
    const nodeId = params.nodes[0];
    const url = NODE_URLS[nodeId];
    if (url) {{
      window.open(url, "_blank", "noopener,noreferrer");
    }}
  }}
}});

// Pointer cursor on hover
network.on("hoverNode", function() {{
  container.style.cursor = "pointer";
}});
network.on("blurNode", function() {{
  container.style.cursor = "default";
}});
</script>
</body>
</html>
"""


# ─── Render mesh ──────────────────────────────────────────────────────────────

mesh_session = api_get(f"{COMPUTE_AGENT_URL}/pipeline/session/default")
mesh_health  = api_get(f"{COMPUTE_AGENT_URL}/health")

html_src = build_agent_mesh_html(mesh_session, mesh_health)
st.components.v1.html(html_src, height=730, scrolling=False)

st.divider()

# ─── Live Agent Activity Feed ─────────────────────────────────────────────────

st.subheader("📡 Live Agent Activity Feed")

if mesh_session:
    stage   = mesh_session.get("stage", "")
    agents  = mesh_session.get("agents", [])
    anal    = mesh_session.get("analysis", {})
    metrics = mesh_session.get("metrics", {})
    logs_raw = mesh_session.get("logs", "")

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
            lbl    = AGENT_LABELS.get(name, ("🔧", name, ""))
            border = (
                "border-left: 3px solid #ff9800;" if status == "running" else
                "border-left: 3px solid #4caf50;" if status == "completed" else
                "border-left: 3px solid #37474f;"
            )
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

        if anal and anal.get("root_cause") not in (None, "Analyzing...", ""):
            with st.expander("🧠 Obs-Intelligence → Compute Agent  (AI Analysis)", expanded=True):
                c1, c2 = st.columns(2)
                c1.metric("Confidence", f"{anal.get('confidence', 0):.0%}")
                c2.metric("Provider",   anal.get("provider", "—"))
                c1.metric("Scenario",   anal.get("scenario_id", "—"))
                c2.metric("LLM Verdict",anal.get("local_validation_status", "—") or "—")
                st.markdown(f"**Root Cause:** {anal.get('root_cause','—')}")
                st.markdown(f"**Recommended Action:** {anal.get('recommended_action','—')}")

        if metrics:
            with st.expander("📈 Prometheus → Compute Agent  (Fetched Metrics)"):
                mdf = pd.DataFrame([{"metric": k, "value": v} for k, v in metrics.items()])
                if not mdf.empty:
                    st.dataframe(mdf, use_container_width=True, hide_index=True)

        if logs_raw and logs_raw.strip():
            with st.expander("📋 Loki → Compute Agent  (Log Lines)"):
                lines = [l for l in logs_raw.split("\n") if l.strip()]
                st.caption(f"{len(lines)} log lines fetched")
                st.code("\n".join(lines[-30:]), language="text")

        approval_id  = mesh_session.get("approval_id", "")
        approval_num = mesh_session.get("approval_ticket_num", 0)
        if stage in ("awaiting_approval",) or approval_id:
            with st.expander("🔀 Gitea PR / Approval Gate", expanded=stage == "awaiting_approval"):
                st.markdown(f"**Approval ID:** `{approval_id or '—'}`")
                st.markdown(f"**xyOps Approval Ticket:** #{approval_num}" if approval_num else "**xyOps Approval Ticket:** —")
                st.markdown(f"**Decision:** {mesh_session.get('autonomy_decision','—')}")
                st.markdown(f"[View open PRs →]({GITEA_EXT}/{GITEA_EXT.split('/')[-1] if False else 'aiops-org'}/ansible-playbooks/pulls)")

        tm = mesh_session.get("trust_metrics", {})
        if tm:
            with st.expander("📊 Autonomy Engine  (Trust Metrics)"):
                ta, tb, tc = st.columns(3)
                ta.metric("Approvals Recorded", tm.get("approvals_recorded", 0))
                tb.metric("Success Rate",        f"{tm.get('success_rate',0):.0%}")
                tc.metric("Executed Runs",       tm.get("executed_runs", 0))
                st.caption(tm.get("path_to_next_tier", ""))
else:
    st.info("No active pipeline session. Trigger an alert and the mesh will light up.")

st.divider()

# ─── Cross-domain correlation panel ──────────────────────────────────────────

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

    _CTYPE_COLOUR = {
        "STORAGE_ROOT":           "#ef4444",
        "COMPUTE_ROOT":           "#f97316",
        "SHARED_INFRASTRUCTURE":  "#eab308",
        "INDEPENDENT_CONCURRENT": "#64748b",
    }
    _RISK_COLOUR = {
        "CRITICAL": "#ef4444", "HIGH": "#f97316",
        "MEDIUM": "#eab308", "LOW": "#22c55e",
    }
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

    xa, xb = st.columns(2)
    xa.markdown(
        f"**💻 Compute**  \n`{xd_assessment.get('compute_service','—')}`  \n"
        f"Scenario: `{xd_assessment.get('compute_scenario','—')}`"
    )
    xb.markdown(
        f"**🗄️ Storage**  \n`{xd_assessment.get('storage_service','—')}`  \n"
        f"Scenario: `{xd_assessment.get('storage_scenario','—')}`"
    )

    if narrative:
        st.info(narrative)

    xc1, xc2, xc3 = st.columns(3)
    with xc1:
        with st.expander("🔗 Causal Chain", expanded=True):
            for i, step in enumerate(chain, 1):
                st.markdown(f"**{i}.** {step}")
    with xc2:
        with st.expander("🛠️ Recommended Actions", expanded=True):
            for act in actions:
                st.markdown(f"- {act}")
    with xc3:
        with st.expander("🔍 Evidence", expanded=False):
            for ev in evidence:
                st.markdown(f"- {ev}")

    with st.expander("📄 Raw unified_assessment JSON", expanded=False):
        st.json(xd_assessment)
else:
    st.success(
        "No active cross-domain correlation — compute and storage are operating independently.",
        icon="✅",
    )
    st.caption("Updates automatically when both agents fire simultaneously within a 2-minute window.")

st.divider()

# ─── Pending approvals mini-feed ─────────────────────────────────────────────

st.subheader("⏳ Pending Approvals (quick view)")
pend       = api_get(f"{COMPUTE_AGENT_URL}/approvals/pending")
pend_items = (pend or {}).get("items", [])
if pend_items:
    pdf = pd.DataFrame(pend_items)
    if "created_at" in pdf.columns:
        pdf["waiting"] = pdf["created_at"].apply(since_str)
    cols_show = [c for c in ["approval_id", "service_name", "alert_name", "severity", "waiting"] if c in pdf.columns]
    st.dataframe(pdf[cols_show], use_container_width=True, hide_index=True)
    st.caption(f"**{len(pend_items)} approvals** waiting — go to [Pending Approvals](#) page to act.")
else:
    st.success("No pending approvals — pipeline queue is clear.")

page_footer("mesh")
