"""
obs_intelligence/llm_enricher.py
────────────────────────────────────────────────────────────────────────────────
LLM enrichment layer.

Wraps the raw OpenAI / Claude API call and converts an EvidenceReport +
Recommendation into a rich LLMEnrichment that agents use to write incident
tickets and approval requests.

  ┌───────────────────────────────────────────────┐
  │  EvidenceReport  ──┐                          │
  │  Recommendation  ──┤─► enrich() ─► LLMEnrichment | None
  │  RiskAssessment  ──┘                          │
  └───────────────────────────────────────────────┘

When AI is disabled (no API key), enrich() returns None — callers fall back
to the Recommendation and RiskAssessment that the deterministic pipeline
already produced.

Provider auto-detection (identical to agent-side ai_analyst.py):
  OPENAI_API_KEY set  → OpenAI (priority)
  CLAUDE_API_KEY set  → Anthropic Claude
  Neither set         → AI disabled → returns None

Environment variables
─────────────────────
  OPENAI_API_KEY    OpenAI API key
  CLAUDE_API_KEY    Anthropic API key (legacy)
  AI_MODEL          Model override (default: gpt-4o-mini / claude-3-5-haiku)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field

import httpx

from obs_intelligence.models import EvidenceReport, RiskAssessment, Recommendation
from obs_intelligence.evidence_builder import evidence_lines

logger = logging.getLogger("obs_intelligence.llm_enricher")

# ── Provider config ───────────────────────────────────────────────────────────
_OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
_CLAUDE_API_KEY: str = os.getenv("CLAUDE_API_KEY", "")
_USE_OPENAI: bool = bool(_OPENAI_API_KEY)
_USE_CLAUDE: bool = bool(_CLAUDE_API_KEY) and not _USE_OPENAI
_DEFAULT_MODEL = "gpt-4o-mini" if _USE_OPENAI else "claude-3-5-haiku-20241022"
AI_MODEL: str = os.getenv("AI_MODEL") or os.getenv("CLAUDE_MODEL") or _DEFAULT_MODEL
AI_ENABLED: bool = _USE_OPENAI or _USE_CLAUDE

_OPENAI_URL = "https://api.openai.com/v1/chat/completions"
_CLAUDE_URL = "https://api.anthropic.com/v1/messages"
_CLAUDE_HEADERS = {
    "Content-Type": "application/json",
    "anthropic-version": "2023-06-01",
}

logger.info(
    "llm_enricher provider: %s  model: %s",
    "openai" if _USE_OPENAI else ("claude" if _USE_CLAUDE else "disabled"),
    AI_MODEL if AI_ENABLED else "n/a",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Output model
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LLMEnrichment:
    """
    Structured output from the LLM enrichment step.

    Mirrors the JSON schema expected from OpenAI/Claude and used by
    build_enriched_ticket_body() in both domain agents.
    """

    rca_summary: str
    recommended_action: str
    autonomy_level: str
    confidence: str                            # "high" | "medium" | "low"
    provider: str                              # "openai" | "claude"

    ansible_playbook: str = ""
    ansible_description: str = ""
    test_plan: list[str] = field(default_factory=list)
    test_cases: list[dict] = field(default_factory=list)

    rca_detail: dict = field(default_factory=dict)
    pr_title: str | None = None
    pr_description: str | None = None
    rollback_steps: list[str] = field(default_factory=list)
    estimated_fix_time_minutes: int | None = None

    # The raw dict returned by the LLM (for downstream use / debugging)
    raw: dict = field(default_factory=dict)

    def to_analysis_dict(self) -> dict:
        """
        Convert to the 'analysis' dict schema expected by existing
        build_enriched_ticket_body() implementations.

        This preserves backward compatibility with Agent 5 (ticket scribe)
        in both domain pipelines.
        """
        result: dict = {
            "rca_summary":          self.rca_summary,
            "recommended_action":   self.recommended_action,
            "autonomy_level":       self.autonomy_level,
            "confidence":           self.confidence,
            "provider":             self.provider,
            "ansible_playbook":     self.ansible_playbook,
            "ansible_description":  self.ansible_description,
            "test_plan":            self.test_plan,
        }
        if self.test_cases:
            result["test_cases"] = self.test_cases
        if self.rca_detail:
            result["rca_detail"] = self.rca_detail
        if self.pr_title:
            result["pr_title"] = self.pr_title
        if self.pr_description:
            result["pr_description"] = self.pr_description
        if self.rollback_steps:
            result["rollback_steps"] = self.rollback_steps
        if self.estimated_fix_time_minutes is not None:
            result["estimated_fix_time_minutes"] = self.estimated_fix_time_minutes
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# Public
# ═══════════════════════════════════════════════════════════════════════════════

async def enrich(
    evidence: EvidenceReport,
    recommendation: Recommendation,
    risk: RiskAssessment,
    http: httpx.AsyncClient,
) -> LLMEnrichment | None:
    """
    Call the LLM to produce a rich incident narrative from the evidence bundle.

    Builds a prompt from the EvidenceReport (features, scenario matches, risk,
    evidence lines) and the top Recommendation, then requests:
      - A detailed RCA narrative
      - An Ansible playbook for remediation
      - Pre/post test cases
      - A GitHub PR description
      - Rollback steps

    Returns None when AI is disabled (caller uses deterministic output instead).
    Returns LLMEnrichment on success.
    Logs warnings and returns None on LLM API failure (graceful degradation).
    """
    if not AI_ENABLED:
        logger.debug("AI disabled — skipping LLM enrichment")
        return None

    prompt = _build_prompt(evidence, recommendation, risk)

    try:
        if _USE_OPENAI:
            raw = await _call_openai(prompt, http)
        else:
            raw = await _call_claude(prompt, http)
    except Exception as exc:
        logger.warning("LLM enrichment API call failed: %s", exc)
        return None

    if not raw:
        return None

    return _parse_enrichment(raw, recommendation)


# ─────────────────────────────────────────────────────────────────────────────
# Internal: prompt builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_prompt(
    evidence: EvidenceReport,
    recommendation: Recommendation,
    risk: RiskAssessment,
) -> str:
    """
    Build the LLM user prompt from the EvidenceReport and top Recommendation.
    """
    f = evidence.features
    ev_lines = "\n".join(evidence_lines(evidence))

    scenario_context = ""
    if evidence.scenario_matches:
        top = evidence.scenario_matches[0]
        scenario_context = (
            f"\nMatched scenario: {top.display_name} (confidence: {top.confidence:.0%})"
        )
        if top.matched_features:
            scenario_context += f"\nMatched conditions: {', '.join(top.matched_features)}"

    playbook_name = recommendation.ansible_playbook or "(none identified)"
    rollback_hint = recommendation.rollback_plan or "(none)"

    return f"""Analyze this production incident and produce a structured JSON response.

INCIDENT SUMMARY
━━━━━━━━━━━━━━━━
Alert:     {f.alert_name}
Service:   {f.service_name}
Severity:  {f.severity.upper()}
Domain:    {f.domain}
{scenario_context}

EVIDENCE OBSERVATIONS
━━━━━━━━━━━━━━━━━━━━━
{ev_lines}

RISK ASSESSMENT
━━━━━━━━━━━━━━━
Risk level: {risk.risk_level.upper()} (score: {risk.risk_score:.2f})
Contributing factors: {'; '.join(risk.contributing_factors[:5])}
Blast radius: {risk.blast_radius}
Time to impact: {risk.time_to_impact or 'unknown'}

RECOMMENDED ACTION
━━━━━━━━━━━━━━━━━━
Action type:   {recommendation.action_type}
Display name:  {recommendation.display_name}
Playbook:      {playbook_name}
Rollback hint: {rollback_hint}

Respond with ONLY valid JSON (no markdown fences) matching this schema:
{{
  "rca_summary": "2-3 sentence root cause analysis",
  "rca_detail": {{
    "symptoms": ["observed symptoms"],
    "probable_cause": "most likely root cause",
    "contributing_factors": ["secondary factors"],
    "blast_radius": "impact scope description"
  }},
  "confidence": "high|medium|low",
  "ansible_playbook": "Complete YAML playbook string",
  "ansible_description": "1-sentence playbook description",
  "test_cases": [
    {{"id": "TC-PRE-1", "name": "Check name", "assertion": "What to verify", "phase": "pre"}},
    {{"id": "TC-POST-1", "name": "Check name", "assertion": "What to verify", "phase": "post"}}
  ],
  "pr_title": "Short PR title under 72 chars (or null)",
  "pr_description": "Markdown PR description (or null)",
  "rollback_steps": ["Step 1", "Step 2"],
  "estimated_fix_time_minutes": 10
}}"""


_SYSTEM_PROMPT = (
    "You are a senior SRE and DevOps expert specializing in production incident analysis. "
    "Respond ONLY with valid JSON matching the provided schema. No markdown, no extra text."
)


# ─────────────────────────────────────────────────────────────────────────────
# Internal: API callers
# ─────────────────────────────────────────────────────────────────────────────

async def _call_openai(prompt: str, http: httpx.AsyncClient) -> dict | None:
    resp = await http.post(
        _OPENAI_URL,
        headers={
            "Authorization": f"Bearer {_OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": AI_MODEL,
            "max_tokens": 2000,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
        },
        timeout=30.0,
    )
    if resp.status_code != 200:
        logger.warning("OpenAI returned HTTP %d: %s", resp.status_code, resp.text[:200])
        return None
    content = resp.json()["choices"][0]["message"]["content"]
    return json.loads(content)


async def _call_claude(prompt: str, http: httpx.AsyncClient) -> dict | None:
    resp = await http.post(
        _CLAUDE_URL,
        headers={**_CLAUDE_HEADERS, "x-api-key": _CLAUDE_API_KEY},
        json={
            "model": AI_MODEL,
            "max_tokens": 2000,
            "system": _SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=30.0,
    )
    if resp.status_code != 200:
        logger.warning("Claude returned HTTP %d: %s", resp.status_code, resp.text[:200])
        return None
    content = resp.json()["content"][0]["text"]
    clean = content.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(clean)


# ─────────────────────────────────────────────────────────────────────────────
# Internal: response parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse_enrichment(raw: dict, recommendation: Recommendation) -> LLMEnrichment:
    """
    Parse the raw LLM JSON into a typed LLMEnrichment.

    Uses the Recommendation as a fallback for fields the LLM may have omitted.
    """
    provider = "openai" if _USE_OPENAI else "claude"

    return LLMEnrichment(
        rca_summary         = raw.get("rca_summary", "No RCA generated."),
        recommended_action  = recommendation.action_type,
        autonomy_level      = "autonomous" if recommendation.autonomous else "approval_gated",
        confidence          = raw.get("confidence", "low"),
        provider            = provider,
        ansible_playbook    = raw.get("ansible_playbook", ""),
        ansible_description = raw.get("ansible_description", ""),
        test_plan           = raw.get("test_plan", []),
        test_cases          = raw.get("test_cases", []),
        rca_detail          = raw.get("rca_detail", {}),
        pr_title            = raw.get("pr_title"),
        pr_description      = raw.get("pr_description"),
        rollback_steps      = raw.get("rollback_steps", []),
        estimated_fix_time_minutes = raw.get("estimated_fix_time_minutes"),
        raw                 = raw,
    )
