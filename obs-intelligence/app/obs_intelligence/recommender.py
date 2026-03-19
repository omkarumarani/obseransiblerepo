"""
obs_intelligence/recommender.py
────────────────────────────────────────────────────────────────────────────────
Recommendation engine.

Converts a (ScenarioMatch, ScenarioDef) pair + RiskAssessment into a typed
Recommendation, applying domain-specific autonomy rules to constrain which
actions may be executed without human approval.

Public API
──────────
    recommend(
        best_match, best_def, risk, domain, autonomy_rules
    ) -> Recommendation

    recommend_all(
        matches_and_defs, risk, domain, autonomy_rules
    ) -> list[Recommendation]

autonomy_rules is expected to be an autonomy_rules module (or any object with
APPROVAL_REQUIRED, AUTONOMOUS_ALLOWED, HUMAN_ONLY, FORCE_APPROVAL_ABOVE_RISK).
"""

from __future__ import annotations

import logging
from types import ModuleType
from typing import Any

from obs_intelligence.models import RiskAssessment, Recommendation, ScenarioMatch
from obs_intelligence.scenario_loader import ScenarioDef

logger = logging.getLogger("obs_intelligence.recommender")

# ── Playbook name map ─────────────────────────────────────────────────────────
# Maps action_type → Ansible playbook filename (best-effort).
_PLAYBOOK_MAP: dict[str, str] = {
    "restart_service":            "restart_service.yml",
    "cpu_scale_out":              "scale_compute.yml",
    "rollback_deploy":            "rollback_deploy.yml",
    "osd_reweight":               "ceph_osd_reweight.yml",
    "pvc_throttle":               "pvc_qos_throttle.yml",
    "pool_expand_advisory":       "ceph_pool_expand.yml",
    "pool_critical_action":       "ceph_pool_critical.yml",
    "investigate_io":             "investigate_storage_io.yml",
    "cluster_assessment":         "ceph_cluster_assess.yml",
    "investigate_errors":         "investigate_service_errors.yml",
    "investigate_latency":        "investigate_latency.yml",
    "throttle_noisy_neighbour":   "throttle_noisy_neighbour.yml",
    "scale_workers":              "scale_workers.yml",
    "reduce_otel_sampling":       "reduce_otel_sampling.yml",
    "circuit_break_dependency":   "circuit_breaker_enable.yml",
    "deep_dive_investigation":    "deep_dive_sre.yml",
    "escalate":                   None,
}

# ── Duration estimates ────────────────────────────────────────────────────────
_DURATION_MAP: dict[str, str] = {
    "restart_service":            "30s–2 min",
    "cpu_scale_out":              "1–3 min",
    "rollback_deploy":            "3–10 min",
    "osd_reweight":               "2–5 min",
    "pvc_throttle":               "< 30s",
    "pool_expand_advisory":       "5–15 min",
    "pool_critical_action":       "5–15 min",
    "investigate_io":             "5 min (review)",
    "cluster_assessment":         "10 min (review)",
    "reduce_otel_sampling":       "< 30s",
    "throttle_noisy_neighbour":   "< 30s",
    "scale_workers":              "1–2 min",
    "circuit_break_dependency":   "< 30s",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Public
# ═══════════════════════════════════════════════════════════════════════════════

def recommend(
    best_match: ScenarioMatch | None,
    best_def: ScenarioDef | None,
    risk: RiskAssessment,
    domain: str,
    autonomy_rules: Any,
) -> Recommendation:
    """
    Build a single Recommendation for the top-scoring scenario.

    If no scenario matched, returns a safe "escalate" recommendation.
    The autonomy level from the scenario YAML is clamped by the domain's
    autonomy_rules (never permit more autonomy than the rules allow).

    Parameters
    ----------
    best_match :       The top ScenarioMatch, or None.
    best_def :         The corresponding ScenarioDef, or None.
    risk :             The RiskAssessment for this incident.
    domain :           "compute" | "storage"
    autonomy_rules :   Module exposing APPROVAL_REQUIRED, AUTONOMOUS_ALLOWED,
                       HUMAN_ONLY, and FORCE_APPROVAL_ABOVE_RISK.
    """
    if not best_match or not best_def:
        return _escalate_recommendation(domain)

    action_type  = best_def.action
    autonomy     = _clamp_autonomy(action_type, best_def.autonomy, risk, autonomy_rules)
    autonomous   = (autonomy == "autonomous")
    playbook     = _PLAYBOOK_MAP.get(action_type)
    duration     = _DURATION_MAP.get(action_type)
    rollback     = _rollback_hint(action_type, best_def)

    return Recommendation(
        action_type=action_type,
        display_name=best_def.display_name,
        description=(
            best_def.rca or
            f"Detected scenario: {best_def.display_name}. "
            f"Recommended action: {action_type}."
        ),
        confidence=best_match.confidence,
        autonomous=autonomous,
        ansible_playbook=playbook,
        xyops_workflow=f"{domain.title()} AIOps Agent Pipeline",
        estimated_duration=duration,
        rollback_plan=rollback,
    )


def recommend_all(
    matches_and_defs: list[tuple[ScenarioMatch, ScenarioDef]],
    risk: RiskAssessment,
    domain: str,
    autonomy_rules: Any,
) -> list[Recommendation]:
    """
    Build a Recommendation for every (ScenarioMatch, ScenarioDef) pair.

    Results are ordered by match confidence descending.
    """
    results: list[Recommendation] = []
    for match, defn in matches_and_defs:
        action_type  = defn.action
        autonomy     = _clamp_autonomy(action_type, defn.autonomy, risk, autonomy_rules)
        autonomous   = (autonomy == "autonomous")
        playbook     = _PLAYBOOK_MAP.get(action_type)
        duration     = _DURATION_MAP.get(action_type)
        rollback     = _rollback_hint(action_type, defn)
        results.append(Recommendation(
            action_type=action_type,
            display_name=defn.display_name,
            description=defn.rca or f"Scenario: {defn.display_name}",
            confidence=match.confidence,
            autonomous=autonomous,
            ansible_playbook=playbook,
            xyops_workflow=f"{domain.title()} AIOps Agent Pipeline",
            estimated_duration=duration,
            rollback_plan=rollback,
        ))
    return sorted(results, key=lambda r: r.confidence, reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clamp_autonomy(
    action_type: str,
    scenario_autonomy: str,
    risk: RiskAssessment,
    rules: Any,
) -> str:
    """
    Apply domain autonomy rules on top of the scenario's declared autonomy.

    Priority (most restrictive wins):
      1. HUMAN_ONLY set → always human_only
      2. APPROVAL_REQUIRED set → at least approval_gated
      3. High risk score → upgrade autonomous → approval_gated
      4. Scenario value otherwise
    """
    human_only_set   = getattr(rules, "HUMAN_ONLY",         set())
    approval_set     = getattr(rules, "APPROVAL_REQUIRED",  set())
    force_threshold  = getattr(rules, "FORCE_APPROVAL_ABOVE_RISK", 0.70)

    if action_type in human_only_set:
        return "human_only"

    if action_type in approval_set:
        # Cannot be more lenient than approval_gated
        if scenario_autonomy == "autonomous":
            logger.debug(
                "Clamped autonomy for '%s': autonomous → approval_gated (in APPROVAL_REQUIRED)",
                action_type,
            )
            return "approval_gated"

    # High risk override: never execute autonomously if risk is too high
    if (
        scenario_autonomy == "autonomous"
        and risk.risk_score >= force_threshold
    ):
        logger.info(
            "Risk %.3f >= %.3f — upgrading autonomy for '%s': autonomous → approval_gated",
            risk.risk_score, force_threshold, action_type,
        )
        return "approval_gated"

    return scenario_autonomy


def _escalate_recommendation(domain: str) -> Recommendation:
    return Recommendation(
        action_type="escalate",
        display_name="Manual Escalation",
        description=(
            "No scenario catalog match found. "
            "Escalating to on-call SRE for manual investigation."
        ),
        confidence=0.0,
        autonomous=False,
        ansible_playbook=None,
        xyops_workflow=f"{domain.title()} AIOps Agent Pipeline",
        estimated_duration=None,
        rollback_plan=None,
    )


def _rollback_hint(action_type: str, defn: ScenarioDef) -> str | None:
    hints = {
        "restart_service":          "Re-deploy the previous container image version.",
        "cpu_scale_out":            "Scale back in: remove extra replicas after load subsides.",
        "rollback_deploy":          "Re-apply the reverted commit and monitor for 10 minutes.",
        "osd_reweight":             "Restore OSD weight to 1.0: ceph osd reweight <id> 1.0",
        "pvc_throttle":             "Remove StorageClass QoS annotation to restore full IOPS.",
        "pool_expand_advisory":     "No automatic rollback — advisory only, no changes made.",
        "circuit_break_dependency": "Disable circuit breaker flag to restore traffic flow.",
    }
    return hints.get(action_type, defn.playbook_hint or None)
