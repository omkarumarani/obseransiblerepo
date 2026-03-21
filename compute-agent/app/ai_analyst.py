"""
aiops-bridge/app/ai_analyst.py
───────────────────────────────────────────────────────────────
AI Analyst — multi-provider RCA, Ansible playbook generation,
and GitHub PR description writer.

Supports both OpenAI (GPT-4o / gpt-4o-mini) and Anthropic (Claude).
Provider is auto-detected from the API key:
  - OPENAI_API_KEY set  → uses https://api.openai.com/v1/chat/completions
  - CLAUDE_API_KEY set  → uses https://api.anthropic.com/v1/messages
  OPENAI_API_KEY takes priority if both are set.

Environment variables consumed:
  OPENAI_API_KEY        OpenAI API key (sk-proj-... or sk-...)
  AI_MODEL              Model name (default: gpt-4o-mini for OpenAI,
                                    claude-3-5-haiku-20241022 for Claude)
  CLAUDE_API_KEY        Anthropic API key (legacy / alternative to OpenAI)
  LOKI_URL              http://loki:3100
  PROMETHEUS_URL        http://prometheus:9090
  GITHUB_REPO           owner/repo (e.g. myorg/myapp) — optional
  NOTIFY_EMAIL          comma-separated addresses for email CC on tickets
"""

import json
import logging
import os
from typing import Any

import httpx
from obs_intelligence.telemetry_client import (
    fetch_instant_metric as _fetch_instant_metric,
    fetch_loki_context as _fetch_loki_context,
)

logger = logging.getLogger("aiops-bridge.ai")

# ── Config ─────────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
CLAUDE_API_KEY: str = os.getenv("CLAUDE_API_KEY", "")  # legacy / Anthropic

# Provider auto-detection: OpenAI takes priority
_USE_OPENAI: bool = bool(OPENAI_API_KEY)
_USE_CLAUDE: bool = bool(CLAUDE_API_KEY) and not _USE_OPENAI

_DEFAULT_MODEL = "gpt-4o-mini" if _USE_OPENAI else "claude-3-5-haiku-20241022"
# AI_MODEL overrides both CLAUDE_MODEL (legacy) and the default
AI_MODEL: str = (
    os.getenv("AI_MODEL")
    or os.getenv("CLAUDE_MODEL")
    or _DEFAULT_MODEL
)
LOKI_URL: str = os.getenv("LOKI_URL", "http://loki:3100")
PROMETHEUS_URL: str = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
GITHUB_REPO: str = os.getenv("GITHUB_REPO", "")
NOTIFY_EMAIL: str = os.getenv("NOTIFY_EMAIL", "")

AI_ENABLED: bool = _USE_OPENAI or _USE_CLAUDE

# ── Local LLM (Ollama) — secondary fallback ────────────────────────────────────
LOCAL_LLM_URL: str  = os.getenv("LOCAL_LLM_URL",   "http://local-llm:11434")
LOCAL_LLM_MODEL: str = os.getenv("LOCAL_LLM_MODEL", "qwen3.5")
LOCAL_LLM_ENABLED: bool = os.getenv("LOCAL_LLM_ENABLED", "false").lower() == "true"
# Ollama exposes an OpenAI-compatible endpoint at /v1/chat/completions
_LOCAL_LLM_URL = f"{LOCAL_LLM_URL.rstrip('/')}/v1/chat/completions"

# ── Provider endpoints & headers ───────────────────────────────────────────────
_OPENAI_URL = "https://api.openai.com/v1/chat/completions"
_CLAUDE_URL = "https://api.anthropic.com/v1/messages"
_CLAUDE_HEADERS = {
    "Content-Type": "application/json",
    "anthropic-version": "2023-06-01",
}

logger.info(
    "AI provider: %s  model: %s  local_llm: %s (%s)",
    "openai" if _USE_OPENAI else ("claude" if _USE_CLAUDE else "disabled"),
    AI_MODEL if AI_ENABLED else "n/a",
    "enabled" if LOCAL_LLM_ENABLED else "disabled",
    LOCAL_LLM_MODEL,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Context fetchers
# ═══════════════════════════════════════════════════════════════════════════════

async def fetch_loki_logs(
    service_name: str,
    http: httpx.AsyncClient,
    limit: int = 50,
) -> str:
    """
    Query Loki for the last `limit` log lines from the given service.
    Delegates to the shared obs_intelligence telemetry client.
    """
    return await _fetch_loki_context(
        label_query=f'{{service_name="{service_name}"}}',
        http=http,
        limit=limit,
        loki_url=LOKI_URL,
    )


async def fetch_prometheus_context(
    service_name: str,
    http: httpx.AsyncClient,
) -> dict[str, str]:
    """
    Fetch key golden-signal metrics for the service from Prometheus.
    Returns a dict of metric_name → formatted value string.
    """
    job = service_name  # Prometheus job label matches service name
    queries = {
        "error_rate_pct": (
            f'100 * sum(rate(http_server_duration_count{{job="{job}",http_response_status_code=~"5.."}}[5m]))'
            f' / sum(rate(http_server_duration_count{{job="{job}"}}[5m]))'
        ),
        "p99_latency_ms": (
            f'histogram_quantile(0.99, sum(rate(http_server_duration_bucket{{job="{job}"}}[5m])) by (le)) * 1000'
        ),
        "p50_latency_ms": (
            f'histogram_quantile(0.50, sum(rate(http_server_duration_bucket{{job="{job}"}}[5m])) by (le)) * 1000'
        ),
        "rps": (
            f'sum(rate(http_server_duration_count{{job="{job}"}}[5m]))'
        ),
        # ── Extended signals (wiring cpu/memory/connections into ObsFeatures) ──
        "cpu_usage_pct": (
            f'rate(process_cpu_seconds_total{{job="{job}"}}[5m]) * 100'
        ),
        "memory_usage_pct": (
            # Primary: divide by total RAM reported by node_exporter.
            # Fallback: if node_exporter is absent the division returns no data,
            # so we fall back to dividing by 536870912 (512 MB cap heuristic).
            f'process_resident_memory_bytes{{job="{job}"}} / on() node_memory_MemTotal_bytes * 100'
            f' or process_resident_memory_bytes{{job="{job}"}} / 536870912 * 100'
        ),
        "active_connections": (
            # Primary: explicit active-request gauge; fallback to request-count
            # rate as a proxy when the gauge metric is absent.
            f'sum(http_server_active_requests{{job="{job}"}})'
            f' or sum(http_server_duration_count{{job="{job}"}})'
        ),
    }

    results: dict[str, str] = {}
    for name, promql in queries.items():
        series = await _fetch_instant_metric(promql, http, PROMETHEUS_URL)
        if series:
            try:
                val = float(series[0]["value"][1])
                results[name] = f"{val:.2f}"
            except (KeyError, ValueError, IndexError):
                results[name] = "parse error"
        else:
            results[name] = "no data"

    logger.info("Prometheus context for %s: %s", service_name, results)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Claude AI Analysis
# ═══════════════════════════════════════════════════════════════════════════════

async def generate_ai_analysis(
    alert_name: str,
    service_name: str,
    severity: str,
    description: str,
    logs: str,
    metrics: dict[str, str],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """
    Call Claude to generate:
      - rca_summary: 2-3 sentence plain-English root cause analysis
      - rca_detail:  structured analysis (symptoms, probable cause, impact)
      - ansible_playbook: YAML playbook to remediate the issue
      - pr_description: GitHub PR description for any code/config changes
      - test_plan: step-by-step test plan for the playbook (dry-run)
      - confidence: "high" | "medium" | "low"

    Returns an empty dict if AI is not enabled or Claude call fails.
    """
    if not AI_ENABLED:
        logger.info("AI analysis skipped (no OPENAI_API_KEY or CLAUDE_API_KEY set)")
        return {}

    metrics_text = "\n".join(
        f"  {k}: {v}" for k, v in metrics.items()
    ) or "  (no metrics available)"

    logs_text = logs[:3000] if logs else "(no recent logs available)"

    system_prompt = """You are a senior SRE (Site Reliability Engineer) and DevOps expert.
You analyze production incidents and produce:
1. Clear root cause analysis (RCA)
2. Ansible playbooks to fix/mitigate the issue
3. GitHub PR descriptions for code/config changes
4. Test plans to validate fixes safely

Always respond with ONLY valid JSON matching the schema provided. No markdown fences, no extra text."""

    user_prompt = f"""Analyze this production incident and respond with a JSON object.

INCIDENT:
  Alert: {alert_name}
  Service: {service_name}
  Severity: {severity}
  Description: {description}

CURRENT METRICS (last 5 minutes):
{metrics_text}

RECENT LOGS (last 50 lines, most recent last):
{logs_text}

Respond with this exact JSON schema:
{{
  "rca_summary": "2-3 sentence plain English summary of root cause",
  "rca_detail": {{
    "symptoms": ["list of observed symptoms from metrics and logs"],
    "probable_cause": "most likely root cause with reasoning",
    "contributing_factors": ["secondary factors"],
    "blast_radius": "description of impact scope"
  }},
  "confidence": "high|medium|low",
  "ansible_playbook": "Complete YAML playbook string (use \\n for newlines). Structure: (1) Play 1 named 'Pre-validation: Assert baseline state' with pre_tasks using ansible.builtin.assert to check current service state, (2) Play 2 named 'Remediate {service_name}' with main tasks for the fix plus notify handlers, (3) post_tasks using ansible.builtin.assert to verify recovery, (4) Play 3 named 'Rollback' with tasks that revert all changes if needed. Use -i localhost, --connection=local compatible tasks only.",
  "ansible_description": "1-sentence description of what the playbook does",
  "test_cases": [
    {{"id": "TC-PRE-1", "name": "Assert service is reachable", "assertion": "HTTP GET /health returns 200", "phase": "pre"}},
    {{"id": "TC-PRE-2", "name": "Assert error rate below critical threshold", "assertion": "error_rate metric available and not zero", "phase": "pre"}},
    {{"id": "TC-POST-1", "name": "Verify error rate recovered", "assertion": "error_rate < 1% after remediation", "phase": "post"}},
    {{"id": "TC-POST-2", "name": "Verify service endpoints responding", "assertion": "HTTP 200 in < 500ms after restart", "phase": "post"}}
  ],
  "pr_description": "GitHub PR description (markdown) for the code/config change that would prevent recurrence",
  "pr_title": "Short PR title (under 72 chars)",
  "estimated_fix_time_minutes": 15,
  "rollback_steps": [
    "Step 1: Run the Rollback play in the playbook: ansible-playbook playbook.yml --tags rollback",
    "Step 2: Verify rollback by checking service health endpoint",
    "Step 3: Monitor error rate for 5 minutes to confirm stability"
  ]
}}"""

    if _USE_OPENAI:
        payload = {
            "model": AI_MODEL,
            "max_tokens": 2000,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        }
        api_url = _OPENAI_URL
        api_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        }
    else:
        payload = {
            "model": AI_MODEL,
            "max_tokens": 2000,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        api_url = _CLAUDE_URL
        api_headers = {**_CLAUDE_HEADERS, "x-api-key": CLAUDE_API_KEY}

    try:
        resp = await http.post(
            api_url,
            json=payload,
            headers=api_headers,
            timeout=30.0,
        )
        if resp.status_code != 200:
            logger.warning(
                "AI API (%s) returned HTTP %d: %s",
                "openai" if _USE_OPENAI else "claude",
                resp.status_code,
                resp.text[:300],
            )
            return {}

        if _USE_OPENAI:
            content = resp.json()["choices"][0]["message"]["content"]
        else:
            content = resp.json()["content"][0]["text"]
        analysis = json.loads(content)
        logger.info(
            "AI analysis complete  alert=%s  confidence=%s",
            alert_name,
            analysis.get("confidence", "?"),
        )
        return analysis

    except json.JSONDecodeError as exc:
        logger.warning("AI response was not valid JSON: %s", exc)
        return {}
    except Exception as exc:
        logger.warning("AI API call failed: %s", exc)
        return {}


# ═══════════════════════════════════════════════════════════════════════════════
# Deterministic analysis (scenario-catalog fallback, no LLM required)
# ═══════════════════════════════════════════════════════════════════════════════

# Module-level catalog cache — populated on first call.
_compute_catalog: list | None = None


def _get_compute_catalog() -> list:
    """Return (and lazily initialise) the compute scenario catalog."""
    global _compute_catalog
    if _compute_catalog is None:
        from obs_intelligence.scenario_correlator import load_catalog
        _compute_catalog = load_catalog(domain="compute")
    return _compute_catalog


# ═══════════════════════════════════════════════════════════════════════════════
# Local LLM analysis (Ollama — secondary fallback)
# ═══════════════════════════════════════════════════════════════════════════════

async def generate_local_llm_analysis(
    alert_name: str,
    service_name: str,
    severity: str,
    description: str,
    logs: str,
    metrics: dict[str, str],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """
    Call the local Ollama instance (llama3.2:3b or configured model) via its
    OpenAI-compatible /v1/chat/completions endpoint.

    Falls back to the same JSON schema as generate_ai_analysis() so the rest
    of the pipeline is unaffected. Returns {} on any failure so the caller can
    fall through to deterministic_analysis().
    """
    if not LOCAL_LLM_ENABLED:
        logger.info("Local LLM disabled (LOCAL_LLM_ENABLED != true)")
        return {}

    metrics_text = "\n".join(f"  {k}: {v}" for k, v in metrics.items()) or "  (no metrics)"
    logs_text = (logs or "(no logs)")[:2000]

    # Compact prompt — local models have small context windows
    prompt = f"""You are an SRE. Respond with ONLY a JSON object, no markdown fences.

Incident: alert={alert_name}  service={service_name}  severity={severity}
Description: {description}
Metrics: {metrics_text}
Logs: {logs_text}

Respond with this exact JSON (fill in real values, do not copy placeholders):
{{
  "rca_summary": "2-sentence root cause for {alert_name} on {service_name}",
  "rca_detail": {{"symptoms": ["high memory usage detected"], "probable_cause": "memory leak or unbounded cache growth", "contributing_factors": ["no memory limits set"], "blast_radius": "single service {service_name}"}},
  "confidence": "low",
  "ansible_playbook": "---\\n- name: Remediate {alert_name} on {service_name}\\n  hosts: localhost\\n  connection: local\\n  gather_facts: false\\n  tasks:\\n    - name: Log remediation start\\n      ansible.builtin.debug:\\n        msg: Starting remediation of {alert_name} on {service_name}\\n    - name: Restart service to clear memory\\n      ansible.builtin.debug:\\n        msg: Would restart {service_name} container here in live mode\\n  post_tasks:\\n    - name: Verify service recovered\\n      ansible.builtin.debug:\\n        msg: Post-validation complete",
  "ansible_description": "Restart {service_name} to clear elevated memory and restore normal operation",
  "test_cases": [
    {{"id": "TC-PRE-1", "name": "Assert service is reachable", "assertion": "HTTP /health returns 200", "phase": "pre"}},
    {{"id": "TC-POST-1", "name": "Verify memory usage dropped", "assertion": "memory_usage_pct below threshold after restart", "phase": "post"}}
  ],
  "pr_description": "Automated remediation playbook for {alert_name} on {service_name}",
  "pr_title": "fix: remediate {alert_name} on {service_name}",
  "estimated_fix_time_minutes": 10,
  "rollback_steps": ["Revert restart and monitor service health for 5 minutes"]
}}"""

    payload = {
        "model": LOCAL_LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "stream": False,
    }

    try:
        resp = await http.post(
            _LOCAL_LLM_URL,
            json=payload,
            timeout=60.0,   # local models can be slow on CPU
        )
        if resp.status_code != 200:
            logger.warning(
                "Local LLM returned HTTP %d: %s",
                resp.status_code, resp.text[:200],
            )
            return {}

        content = resp.json()["choices"][0]["message"]["content"].strip()
        # Strip any accidental markdown fences the model may add
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        analysis = json.loads(content)
        logger.info(
            "Local LLM analysis complete  alert=%s  model=%s  confidence=%s",
            alert_name, LOCAL_LLM_MODEL, analysis.get("confidence", "?"),
        )
        return analysis

    except json.JSONDecodeError as exc:
        logger.warning("Local LLM response not valid JSON: %s", exc)
        return {}
    except Exception as exc:
        logger.warning("Local LLM call failed: %s", exc)
        return {}


def deterministic_analysis(
    alert_name: str,
    service_name: str,
    severity: str = "warning",
) -> dict[str, Any]:
    """
    Rule-engine fallback for when AI is disabled.

    Matches *alert_name* against the compute scenario YAML catalog and returns
    an action recommendation in the same dict schema as generate_ai_analysis(),
    so downstream agents (ticket builder, approval workflow) work unchanged.
    """
    import datetime as _dt

    from obs_intelligence.models import ObsFeatures
    from obs_intelligence.scenario_correlator import match_best

    features = ObsFeatures(
        alert_name=alert_name,
        service_name=service_name,
        severity=severity,
        domain="compute",
        timestamp=_dt.datetime.now(_dt.timezone.utc),
    )

    catalog = _get_compute_catalog()
    best_match, best_def = match_best(features, catalog)

    if best_match and best_def:
        logger.info(
            "Scenario catalog match  alert=%s  scenario=%s  confidence=%.2f",
            alert_name, best_def.scenario_id, best_match.confidence,
        )
        return {
            "rca_summary": best_def.rca,
            "recommended_action": best_def.action,
            "autonomy_level": best_def.autonomy,
            "ansible_playbook": _build_compute_stub_playbook(alert_name, best_def.playbook_hint),
            "ansible_description": (
                f"Deterministic remediation for {best_def.display_name}"
            ),
            "test_plan": [
                f"Verify {service_name} health after {best_def.action}",
                "Monitor error_rate and latency_p99 for 5 minutes post-action",
                "Check Grafana Agentic AI Operations dashboard for confirmation",
            ],
            "confidence": f"{best_match.confidence:.2f}",
            "provider": "scenario-catalog",
        }

    logger.info("No scenario catalog match  alert=%s — using generic escalate", alert_name)
    return {
        "rca_summary": (
            f"Unrecognised compute alert '{alert_name}'. "
            "No scenario catalog match found. Escalating to on-call SRE."
        ),
        "recommended_action": "escalate",
        "autonomy_level": "human_only",
        "ansible_playbook": "",
        "ansible_description": "No matching scenario — manual investigation required",
        "test_plan": ["Manual investigation required — no scenario matched"],
        "confidence": "0.00",
        "provider": "scenario-catalog",
    }


def _build_compute_stub_playbook(alert_name: str, hint: str) -> str:
    return f"""---
# Compute Remediation Playbook — {alert_name}
# Generated by: compute-agent scenario catalog (deterministic)
# Review carefully before execution.
- name: Compute incident remediation — {alert_name}
  hosts: localhost
  connection: local
  gather_facts: false
  tasks:
    - name: Remediation step
      # {hint}
      debug:
        msg: "Review and implement: {hint}"
"""




def build_enriched_ticket_body(
    service_name: str,
    alert_name: str,
    severity: str,
    description: str,
    starts_at: str,
    dashboard_url: str,
    bridge_trace_id: str,
    metrics: dict[str, str],
    analysis: dict[str, Any],
    risk_score: float = 0.0,
    risk_level: str = "",
    evidence_lines: list | None = None,
) -> str:
    """
    Build the full xyOps ticket body (Markdown) combining raw context
    with the AI-generated RCA and remediation plan.
    """
    now_utc = __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc
    ).isoformat()

    # ── Header ──────────────────────────────────────────────
    body = (
        f"## 🚨 Automated Incident — AIOps Bridge\n\n"
        f"| Field | Value |\n"
        f"|---|---|\n"
        f"| **Service** | `{service_name}` |\n"
        f"| **Alert** | `{alert_name}` |\n"
        f"| **Severity** | `{severity.upper()}` |\n"
        f"| **Detected at** | {starts_at or now_utc} |\n"
        f"| **Dashboard** | [{dashboard_url}]({dashboard_url}) |\n"
        f"| **OTel Trace** | `{bridge_trace_id}` — paste in Grafana → Tempo |\n\n"
    )

    # ── Metrics snapshot ─────────────────────────────────────
    if metrics:
        body += "### 📊 Metrics at Time of Incident\n\n"
        body += "| Metric | Value |\n|---|---|\n"
        for k, v in metrics.items():
            label = k.replace("_", " ").title()
            body += f"| {label} | `{v}` |\n"
        body += "\n"

    # ── Risk assessment ───────────────────────────────────────
    _eff_risk_score = risk_score or (analysis or {}).get("risk_score", 0.0)
    _eff_risk_level = risk_level or (analysis or {}).get("risk_level", "")
    if _eff_risk_level:
        level_badge = {
            "critical": "🔴 CRITICAL",
            "high":     "🟠 HIGH",
            "medium":   "🟡 MEDIUM",
            "low":      "🟢 LOW",
        }.get(_eff_risk_level.lower(), f"⚪ {_eff_risk_level.upper()}")
        body += f"### ⚡ Risk Assessment\n\n"
        body += f"**Risk level:** {level_badge}  |  **Score:** `{_eff_risk_score:.3f}`\n\n"

    # ── Evidence observations ─────────────────────────────────
    _eff_evidence = evidence_lines or (analysis or {}).get("evidence_lines") or []
    if _eff_evidence:
        body += "### 🔍 Evidence Observations\n\n"
        for line in _eff_evidence:
            body += f"{line}\n"
        body += "\n"

    # ── AI RCA ───────────────────────────────────────────────
    if analysis:
        rca = analysis.get("rca_detail", {})
        confidence = analysis.get("confidence", "unknown")

        body += f"### 🤖 AI Root Cause Analysis (confidence: {confidence})\n\n"
        body += f"{analysis.get('rca_summary', description)}\n\n"

        if rca:
            if rca.get("symptoms"):
                body += "**Observed symptoms:**\n"
                for s in rca["symptoms"]:
                    body += f"- {s}\n"
                body += "\n"

            if rca.get("probable_cause"):
                body += f"**Probable cause:** {rca['probable_cause']}\n\n"

            if rca.get("blast_radius"):
                body += f"**Impact scope:** {rca['blast_radius']}\n\n"

        # ── Ansible playbook ──────────────────────────────────
        if analysis.get("ansible_playbook"):
            body += "### 🔧 Proposed Ansible Remediation\n\n"
            body += f"_{analysis.get('ansible_description', 'Auto-generated playbook')}_\n\n"
            body += "```yaml\n"
            body += analysis["ansible_playbook"]
            body += "\n```\n\n"

        # ── Test plan / test cases ────────────────────────────────────────
        if analysis.get("test_cases"):
            cases = analysis["test_cases"]
            pre_cases  = [tc for tc in cases if tc.get("phase") == "pre"]
            post_cases = [tc for tc in cases if tc.get("phase") == "post"]
            body += "### 🧪 Test Cases\n\n"
            if pre_cases:
                body += "**Pre-execution validation:**\n\n"
                for tc in pre_cases:
                    body += (
                        f"- [ ] `{tc.get('id', '')}` **{tc.get('name', '')}**"
                        f" — _{tc.get('assertion', '')}_\n"
                    )
                body += "\n"
            if post_cases:
                body += "**Post-execution verification:**\n\n"
                for tc in post_cases:
                    body += (
                        f"- [ ] `{tc.get('id', '')}` **{tc.get('name', '')}**"
                        f" — _{tc.get('assertion', '')}_\n"
                    )
                body += "\n"
            est = analysis.get("estimated_fix_time_minutes")
            if est:
                body += f"_Estimated remediation time: {est} minutes_\n\n"
        elif analysis.get("test_plan"):
            body += "### ✅ Playbook Test Plan (dry-run)\n\n"
            for step in analysis["test_plan"]:
                body += f"1. {step}\n" if not step.startswith("Step") else f"- {step}\n"
            body += "\n"
            est = analysis.get("estimated_fix_time_minutes")
            if est:
                body += f"_Estimated remediation time: {est} minutes_\n\n"

        # ── GitHub PR ─────────────────────────────────────────
        if analysis.get("pr_title"):
            repo = GITHUB_REPO or "your-org/your-repo"
            body += "### 🔀 Suggested Code/Config Fix (GitHub PR)\n\n"
            body += f"**PR Title:** `{analysis['pr_title']}`\n\n"
            if GITHUB_REPO:
                body += f"**Repo:** https://github.com/{repo}\n\n"
            body += "**PR Description:**\n\n"
            body += analysis.get("pr_description", "") + "\n\n"

        # ── Rollback ──────────────────────────────────────────
        if analysis.get("rollback_steps"):
            body += "### ⏪ Rollback Steps\n\n"
            for step in analysis["rollback_steps"]:
                body += f"- {step}\n"
            body += "\n"

    else:
        # No AI — fall back to raw description
        body += f"### Description\n\n{description}\n\n"

    body += (
        "---\n"
        "*This ticket was created automatically by the AIOps Bridge.*\n"
        f"*Trace ID: `{bridge_trace_id}` — open in Grafana → Tempo to see the detection span.*\n"
    )
    return body


# ═══════════════════════════════════════════════════════════════════════════════
# Email notify list builder
# ═══════════════════════════════════════════════════════════════════════════════

def get_notify_list(severity: str) -> list[str]:
    """
    Return list of email addresses to add to the xyOps ticket notify field.
    Critical alerts get everyone; warning only gets the primary contact.
    Reads from NOTIFY_EMAIL env var (comma-separated).
    """
    if not NOTIFY_EMAIL:
        return []
    emails = [e.strip() for e in NOTIFY_EMAIL.split(",") if e.strip()]
    if severity == "critical":
        return emails
    # warning/info: just the first address
    return emails[:1]
