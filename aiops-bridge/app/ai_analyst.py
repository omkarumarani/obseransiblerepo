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

# ── Provider endpoints & headers ───────────────────────────────────────────────
_OPENAI_URL = "https://api.openai.com/v1/chat/completions"
_CLAUDE_URL = "https://api.anthropic.com/v1/messages"
_CLAUDE_HEADERS = {
    "Content-Type": "application/json",
    "anthropic-version": "2023-06-01",
}

logger.info(
    "AI provider: %s  model: %s",
    "openai" if _USE_OPENAI else ("claude" if _USE_CLAUDE else "disabled"),
    AI_MODEL if AI_ENABLED else "n/a",
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
    Returns a plain-text string of log lines, or an empty string on failure.
    """
    try:
        query = f'{{service_name="{service_name}"}}'
        resp = await http.get(
            f"{LOKI_URL}/loki/api/v1/query_range",
            params={
                "query": query,
                "limit": limit,
                "direction": "backward",
            },
            timeout=5.0,
        )
        if resp.status_code != 200:
            logger.warning("Loki query returned HTTP %d", resp.status_code)
            return ""
        data = resp.json()
        lines: list[str] = []
        for stream in data.get("data", {}).get("result", []):
            for _ts, msg in stream.get("values", []):
                lines.append(msg)
        # Most recent last → chronological order
        lines.reverse()
        log_text = "\n".join(lines[-limit:])
        logger.info(
            "Loki: fetched %d log lines for service=%s", len(lines), service_name
        )
        return log_text
    except Exception as exc:
        logger.warning("Loki fetch failed for %s: %s", service_name, exc)
        return ""


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
    }

    results: dict[str, str] = {}
    for name, promql in queries.items():
        try:
            resp = await http.get(
                f"{PROMETHEUS_URL}/api/v1/query",
                params={"query": promql},
                timeout=5.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                result_list = data.get("data", {}).get("result", [])
                if result_list:
                    val = float(result_list[0]["value"][1])
                    results[name] = f"{val:.2f}"
                else:
                    results[name] = "no data"
            else:
                results[name] = f"HTTP {resp.status_code}"
        except Exception as exc:
            results[name] = f"error: {exc}"

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
  "ansible_playbook": "full YAML ansible playbook content as a string (use \\n for newlines)",
  "ansible_description": "1-sentence description of what the playbook does",
  "pr_description": "GitHub PR description (markdown) for the code/config change that would prevent recurrence",
  "pr_title": "Short PR title (under 72 chars)",
  "test_plan": [
    "Step 1: ...",
    "Step 2: ..."
  ],
  "estimated_fix_time_minutes": 15,
  "rollback_steps": ["Step 1: ...", "Step 2: ..."]
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
# Ticket body builder (AI-enriched)
# ═══════════════════════════════════════════════════════════════════════════════

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

        # ── Test plan ─────────────────────────────────────────
        if analysis.get("test_plan"):
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
