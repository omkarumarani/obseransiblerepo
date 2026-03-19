"""
storage-agent/app/autonomy_rules.py
────────────────────────────────────────────────────────────────────────────────
Storage domain autonomy rules.

Imported by the recommender to constrain scenario-driven actions.  These rules
override the autonomy level declared in the scenario YAML to prevent over-
automation of high-impact Ceph operations.

Rule semantics (most restrictive wins):
  • HUMAN_ONLY            → action is never automated, ever
  • APPROVAL_REQUIRED     → action needs human sign-off (at least approval_gated)
  • AUTONOMOUS_ALLOWED    → action may execute without approval when risk is low
  • FORCE_APPROVAL_ABOVE_RISK  → even autonomous actions get gated above this score
"""

from __future__ import annotations

DOMAIN = "storage"

# Actions that must NEVER be executed autonomously.
# Any scenario declaring "autonomous" or "approval_gated" for these
# will be hard-clamped to "human_only".
HUMAN_ONLY: set[str] = {
    "multi_osd_escalate",          # multiple simultaneous OSD failures — data loss risk
}

# Actions that require human sign-off before execution.
APPROVAL_REQUIRED: set[str] = {
    "osd_reweight",                # OSD weight change affects data distribution
    "pool_expand_advisory",        # capacity expansion — cost + topology change
    "pool_critical_action",        # pool-full mitigation — data-loss adjacent
    "investigate_io",              # advisory — human follow-up needed
    "cluster_assessment",          # broad cluster investigation
    "escalate",                    # no automation path
}

# Actions that are safe to run without any approval gate.
AUTONOMOUS_ALLOWED: set[str] = {
    "pvc_throttle",                # QoS annotation change — reversible, single-PVC scope
}

# Risk score above which even AUTONOMOUS_ALLOWED actions get approval-gated.
FORCE_APPROVAL_ABOVE_RISK: float = 0.65
