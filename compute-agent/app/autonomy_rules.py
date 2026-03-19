"""
compute-agent/app/autonomy_rules.py
────────────────────────────────────────────────────────────────────────────────
Compute domain autonomy rules.

Imported by the recommender to constrain scenario-driven actions.  These rules
are the *last line of defence* — they override the autonomy level declared in
the scenario YAML to prevent over-automation of high-impact operations.

Rule semantics (most restrictive wins):
  • HUMAN_ONLY            → action is never automated, ever
  • APPROVAL_REQUIRED     → action needs human sign-off (at least approval_gated)
  • AUTONOMOUS_ALLOWED    → action may execute without approval when risk is low
  • FORCE_APPROVAL_ABOVE_RISK  → even autonomous actions get gated above this score
"""

from __future__ import annotations

DOMAIN = "compute"

# Actions that must NEVER be executed autonomously — require explicit human decision.
HUMAN_ONLY: set[str] = set()
# (No compute action is hard-blocked at the human_only tier by default.
#  Add actions here if they become too risky for approval-gate alone,
#  e.g. "force_shutdown_cluster".)

# Actions that require human sign-off before execution.
# Scenarios declaring "autonomous" for these will be clamped to "approval_gated".
APPROVAL_REQUIRED: set[str] = {
    "rollback_deploy",             # reverting a deployment is high-impact
    "restart_service",             # process restart may cause brief downtime
    "circuit_break_dependency",    # disabling a dependency affects all callers
    "scale_workers",               # capacity changes have cost implications
    "deep_dive_investigation",     # human review required before acting
    "investigate_errors",          # advisory — needs human follow-up
    "investigate_latency",         # advisory — needs human follow-up
    "cpu_scale_out",               # infrastructure cost change
}

# Actions that are safe to run without any approval gate (low blast radius).
AUTONOMOUS_ALLOWED: set[str] = {
    "reduce_otel_sampling",        # telemetry config — zero production impact
    "throttle_noisy_neighbour",    # QoS tweak affecting only the noisy tenant
}

# Risk score above which even AUTONOMOUS_ALLOWED actions get approval-gated.
# This prevents autonomous execution during high-blast-radius incidents.
FORCE_APPROVAL_ABOVE_RISK: float = 0.70
