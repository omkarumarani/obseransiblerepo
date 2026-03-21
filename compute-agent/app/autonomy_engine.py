"""
compute-agent/app/autonomy_engine.py
────────────────────────────────────────────────────────────────────────────────
Autonomy Decision Engine — determines whether a proposed remediation action
may execute autonomously or must wait for human approval.

Decision hierarchy (most-restrictive wins):
  1. Hard HUMAN_ONLY rules (autonomy_rules.py)   → never autonomous
  2. APPROVAL_REQUIRED rules                     → always gate human
  3. Risk score above tier risk_ceiling           → gate human
  4. Trust score: insufficient approval history   → gate human
  5. Trust score: below min_success_rate          → gate human
  6. Recent declined decisions                   → gate human
  7. All checks pass                             → autonomous allowed

Call check_autonomy() before creating an approval gate in the pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from . import autonomy_rules as _rules
from .approval_history import ApprovalRecord, TrustScore, history_store
from .tier_registry import ServiceTier, TierPolicy, get_service_tier, get_tier_policy

logger = logging.getLogger("aiops-bridge.autonomy_engine")


# ═══════════════════════════════════════════════════════════════════════════════
# Data model
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AutonomyDecision:
    """Result of an autonomy eligibility check."""
    autonomous: bool          # True → run without human approval
    gate_reason: str          # Human-readable reason for the decision
    tier: ServiceTier
    tier_policy: TierPolicy
    trust_score: TrustScore | None
    auto_merge_pr: bool       # True → agent should auto-merge Gitea PR
    action_type: str
    service_name: str
    risk_score: float

    @property
    def mode(self) -> str:
        return "autonomous" if self.autonomous else "approval_required"

    def as_dict(self) -> dict:
        return {
            "autonomous": self.autonomous,
            "mode": self.mode,
            "gate_reason": self.gate_reason,
            "tier": self.tier.value,
            "auto_merge_pr": self.auto_merge_pr,
            "action_type": self.action_type,
            "service_name": self.service_name,
            "risk_score": round(self.risk_score, 3),
            "trust": (
                {
                    "total_decisions": self.trust_score.total_decisions,
                    "approved_count": self.trust_score.approved_count,
                    "autonomous_count": self.trust_score.autonomous_count,
                    "declined_count": self.trust_score.declined_count,
                    "success_rate": round(self.trust_score.success_rate, 3),
                    "autonomy_eligible": self.trust_score.autonomy_eligible,
                    "reason": self.trust_score.reason,
                }
                if self.trust_score
                else None
            ),
            "tier_policy": {
                "min_approvals_for_autonomy": self.tier_policy.min_approvals_for_autonomy,
                "min_success_rate": self.tier_policy.min_success_rate,
                "risk_ceiling": self.tier_policy.risk_ceiling,
                "auto_merge_pr": self.tier_policy.auto_merge_pr,
            },
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def check_autonomy(
    *,
    service_name: str,
    action_type: str,
    risk_score: float,
) -> AutonomyDecision:
    """
    Evaluate whether a (service, action, risk) triple may execute autonomously.

    This is the single authoritative gate before creating an approval ticket.
    Call it from pipeline.py Agent 6.

    Returns an AutonomyDecision.  If .autonomous is True the pipeline should:
      • auto-merge the Gitea PR (if decision.auto_merge_pr is True)
      • invoke ansible-runner /run directly (skip the approval gate ticket)
      • record the decision in history_store with decision="autonomous"

    If .autonomous is False the pipeline should create the approval gate ticket
    as normal and record the human decision later via record_human_decision().
    """
    tier = get_service_tier(service_name)
    policy = get_tier_policy(tier)

    # ── 1. Hard HUMAN_ONLY (domain rules override everything) ────────────────
    if action_type in _rules.HUMAN_ONLY:
        return _gate(
            reason=f"Action '{action_type}' is classified HUMAN_ONLY — automation permanently disabled",
            tier=tier, policy=policy, action_type=action_type,
            service_name=service_name, risk_score=risk_score,
        )

    # ── 2. APPROVAL_REQUIRED (domain rules) ──────────────────────────────────
    if action_type in _rules.APPROVAL_REQUIRED:
        # Still check trust — if trust threshold is met AND risk is low we can
        # downgrade APPROVAL_REQUIRED to autonomous for non-HUMAN_ONLY actions.
        # This is the "graduated autonomy" path.
        pass  # fall through to trust check below

    # ── 3. Risk ceiling check ─────────────────────────────────────────────────
    if risk_score > policy.risk_ceiling:
        return _gate(
            reason=(
                f"Risk score {risk_score:.3f} exceeds tier {tier.value} ceiling "
                f"{policy.risk_ceiling:.2f} — human approval required"
            ),
            tier=tier, policy=policy, action_type=action_type,
            service_name=service_name, risk_score=risk_score,
        )

    # ── 4. Domain-level force-approval-above-risk ────────────────────────────
    if risk_score > _rules.FORCE_APPROVAL_ABOVE_RISK:
        return _gate(
            reason=(
                f"Risk score {risk_score:.3f} above domain FORCE_APPROVAL_ABOVE_RISK "
                f"threshold {_rules.FORCE_APPROVAL_ABOVE_RISK}"
            ),
            tier=tier, policy=policy, action_type=action_type,
            service_name=service_name, risk_score=risk_score,
        )

    # ── 5. Trust / approval history check ────────────────────────────────────
    trust = history_store.compute_trust_score(
        service_name=service_name,
        action_type=action_type,
        env_tier=tier.value,
        min_approvals=policy.min_approvals_for_autonomy,
        min_success_rate=policy.min_success_rate,
        window_days=policy.history_window_days,
    )

    if not trust.autonomy_eligible:
        return _gate(
            reason=trust.reason,
            tier=tier, policy=policy, action_type=action_type,
            service_name=service_name, risk_score=risk_score,
            trust=trust,
        )

    # ── 6. For APPROVAL_REQUIRED actions, final hard-gate unless in
    #       AUTONOMOUS_ALLOWED and trust threshold is fully met ───────────────
    if action_type in _rules.APPROVAL_REQUIRED and action_type not in _rules.AUTONOMOUS_ALLOWED:
        # Action is in APPROVAL_REQUIRED but NOT in AUTONOMOUS_ALLOWED.
        # Even with trust met, require approval for these high-impact actions
        # unless the tier explicitly allows full autonomy (sandbox/dev).
        if tier not in (ServiceTier.DEVELOPMENT, ServiceTier.SANDBOX):
            return _gate(
                reason=(
                    f"Action '{action_type}' is APPROVAL_REQUIRED — autonomous execution "
                    f"is only allowed for DEVELOPMENT/SANDBOX tiers regardless of trust history"
                ),
                tier=tier, policy=policy, action_type=action_type,
                service_name=service_name, risk_score=risk_score,
                trust=trust,
            )

    # ── All checks passed → AUTONOMOUS ───────────────────────────────────────
    logger.info(
        "Autonomy GRANTED  service=%s  action=%s  tier=%s  risk=%.3f  trust=%s",
        service_name, action_type, tier.value, risk_score, trust.reason,
    )
    return AutonomyDecision(
        autonomous=True,
        gate_reason=f"Autonomous execution granted — {trust.reason}",
        tier=tier,
        tier_policy=policy,
        trust_score=trust,
        auto_merge_pr=policy.auto_merge_pr,
        action_type=action_type,
        service_name=service_name,
        risk_score=risk_score,
    )


def check_autonomy_for_new_service(
    service_name: str,
    action_type: str,
    risk_score: float,
) -> AutonomyDecision:
    """
    Convenience wrapper — same as check_autonomy() but logs a prominent message
    for services with zero history, making the onboarding path clear in logs.
    """
    tier = get_service_tier(service_name)
    policy = get_tier_policy(tier)
    trust = history_store.compute_trust_score(
        service_name=service_name,
        action_type=action_type,
        env_tier=tier.value,
        min_approvals=policy.min_approvals_for_autonomy,
        min_success_rate=policy.min_success_rate,
        window_days=policy.history_window_days,
    )
    if trust.total_decisions == 0:
        logger.info(
            "New service '%s' / action '%s' — zero history, requiring human approval "
            "(need %d approved execution(s) to unlock autonomous mode for tier=%s)",
            service_name, action_type, policy.min_approvals_for_autonomy, tier.value,
        )
    return check_autonomy(
        service_name=service_name,
        action_type=action_type,
        risk_score=risk_score,
    )


# ── Internal helper ────────────────────────────────────────────────────────────

def _gate(
    *,
    reason: str,
    tier: ServiceTier,
    policy: TierPolicy,
    action_type: str,
    service_name: str,
    risk_score: float,
    trust: TrustScore | None = None,
) -> AutonomyDecision:
    logger.info(
        "Autonomy GATED  service=%s  action=%s  tier=%s  risk=%.3f  reason=%s",
        service_name, action_type, tier.value, risk_score, reason,
    )
    return AutonomyDecision(
        autonomous=False,
        gate_reason=reason,
        tier=tier,
        tier_policy=policy,
        trust_score=trust,
        auto_merge_pr=False,   # never auto-merge when gated
        action_type=action_type,
        service_name=service_name,
        risk_score=risk_score,
    )
