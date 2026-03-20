"""
obs_intelligence/sre_reasoning_agent.py
────────────────────────────────────────────────────────────────────────────────
SRE Reasoning Agent — deterministic structured reasoning layer (Phase 5).

Produces an SREAssessment from raw observability features + scenario matches
+ risk assessment.  No LLM calls are made here — this is fully deterministic
and unit-testable.

                 ObsFeatures
                 ScenarioMatch[]   ──►  SREReasoningAgent.assess()  ──►  SREAssessment
                 RiskAssessment                                                │
                                                                              ▼
                                                              llm_enricher injects this
                                                              as structured context so
                                                              the LLM writes narrative
                                                              from pre-computed facts,
                                                              not from raw signals.

Usage
─────
    from obs_intelligence.sre_reasoning_agent import SREReasoningAgent, SREAssessment

    agent = SREReasoningAgent()
    sre = agent.assess(features, scenario_matches, risk)

    # Embed in LLM prompt:
    prompt_section = sre.to_prompt_block()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from obs_intelligence.models import ObsFeatures, RiskAssessment, ScenarioMatch

logger = logging.getLogger("obs_intelligence.sre_reasoning_agent")


# ═══════════════════════════════════════════════════════════════════════════════
# Output model
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SREAssessment:
    """
    Structured SRE reasoning produced deterministically from observability signals.

    All fields are populated before the LLM is invoked so the model only has to
    write a clear narrative from pre-validated structured data.  This prevents
    the LLM from hallucinating causal chains that are not supported by the
    actual metrics.
    """

    degradation_summary: str
    """One-sentence technical summary of the observed service condition."""

    causal_chain: list[str]
    """
    Ordered sequence of probable causes from most-likely to least-likely.
    Each entry is a complete sentence starting with a step number, e.g.
    "1. Upstream dependency timeout causing connection pool exhaustion."
    """

    predicted_impact: str
    """Predicted user / business impact if the incident is left unresolved."""

    recommended_actions: list[str]
    """
    Ordered list of concrete SRE actions.  Each entry starts with a step
    number, e.g. "1. Check upstream service health via /health endpoint."
    """

    autonomy_recommendation: str = "approval_gated"
    """
    SRE layer's view of which autonomy level is safe for this incident.
    "autonomous" | "approval_gated" | "human_only"
    This is an *advisory* — the recommender's autonomy_rules take precedence.
    """

    urgency: str = "medium"
    """Operational urgency: "low" | "medium" | "high" | "critical"."""

    evidence_strength: str = "moderate"
    """How much evidence supports the causal chain: "weak" | "moderate" | "strong"."""

    def to_prompt_block(self) -> str:
        """
        Render this assessment as a formatted block for insertion into the LLM prompt.

        The LLM should treat this as verified structured facts and write a
        narrative FROM them — it must NOT re-derive or contradict the reasoning
        below.
        """
        chain_text = "\n".join(f"  {item}" for item in self.causal_chain)
        actions_text = "\n".join(f"  {item}" for item in self.recommended_actions)
        return (
            "SRE REASONING LAYER (pre-computed — write your narrative from these "
            "verified facts; do NOT re-derive or contradict them)\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Degradation:       {self.degradation_summary}\n"
            f"Urgency:           {self.urgency.upper()}\n"
            f"Autonomy advised:  {self.autonomy_recommendation}\n"
            f"Evidence strength: {self.evidence_strength}\n\n"
            f"Causal Chain (most likely → least likely):\n{chain_text}\n\n"
            f"Predicted Impact:\n  {self.predicted_impact}\n\n"
            f"Recommended Actions:\n{actions_text}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Reasoning agent
# ═══════════════════════════════════════════════════════════════════════════════

class SREReasoningAgent:
    """
    Deterministic SRE reasoning engine.

    Stateless — safe to instantiate once at module level and reuse.
    Call assess() once per incident; no internal mutable state is modified.
    """

    def assess(
        self,
        features: ObsFeatures,
        matches: list[ScenarioMatch],
        risk: RiskAssessment,
    ) -> SREAssessment:
        """
        Produce a structured SRE assessment for the current incident.

        Parameters
        ----------
        features :  Raw observability feature snapshot from the current cycle.
        matches  :  Scenario matches sorted by confidence descending.
        risk     :  RiskAssessment for this incident.

        Returns
        -------
        SREAssessment ready to embed in the LLM prompt via to_prompt_block().
        """
        top_match = matches[0] if matches else None

        assessment = SREAssessment(
            degradation_summary    = self._build_degradation_summary(features, top_match),
            causal_chain           = self._build_causal_chain(features, matches, risk),
            predicted_impact       = self._build_predicted_impact(features, risk),
            recommended_actions    = self._build_recommended_actions(features, matches, risk),
            autonomy_recommendation = self._compute_autonomy(risk, top_match),
            urgency                = self._compute_urgency(risk),
            evidence_strength      = self._compute_evidence_strength(features, matches),
        )

        logger.debug(
            "SREAssessment produced  service=%s  urgency=%s  autonomy=%s  chain_depth=%d",
            features.service_name,
            assessment.urgency,
            assessment.autonomy_recommendation,
            len(assessment.causal_chain),
        )
        return assessment

    # ── Private: degradation summary ─────────────────────────────────────────

    def _build_degradation_summary(
        self,
        f: ObsFeatures,
        top_match: ScenarioMatch | None,
    ) -> str:
        signals: list[str] = []

        if f.domain == "compute":
            if f.error_rate > 0.05:
                signals.append(f"{f.error_rate:.0%} error rate")
            if f.latency_p99 > 1.0:
                signals.append(f"p99 latency {f.latency_p99:.2f}s")
            elif f.latency_p99 > 0.5:
                signals.append(f"elevated p99 {f.latency_p99:.2f}s")
            if f.cpu_usage > 0.80:
                signals.append(f"{f.cpu_usage:.0%} CPU")
            if f.memory_usage > 0.85:
                signals.append(f"{f.memory_usage:.0%} memory")
        else:  # storage
            if f.osd_up_count < f.osd_total_count and f.osd_total_count > 0:
                down = f.osd_total_count - f.osd_up_count
                signals.append(f"{down} OSD(s) down")
            if f.pool_usage_pct > 0.80:
                signals.append(f"{f.pool_usage_pct:.0%} pool fill")
            if f.degraded_pgs > 0:
                signals.append(f"{f.degraded_pgs} degraded PGs")
            if f.io_latency > 0.1:
                signals.append(f"IO latency {f.io_latency:.2f}s")

        if f.log_anomaly_detected and not signals:
            signals.append("log error anomaly")

        if not signals:
            signals.append("metric anomaly detected")

        signal_str = ", ".join(signals)
        scenario_part = (
            f" — matches '{top_match.display_name}'"
            if top_match and top_match.confidence > 0.5
            else ""
        )
        return (
            f"`{f.service_name}` ({f.domain}): {signal_str}"
            f" [{f.severity.upper()}]{scenario_part}."
        )

    # ── Private: causal chain ─────────────────────────────────────────────────

    def _build_causal_chain(
        self,
        f: ObsFeatures,
        matches: list[ScenarioMatch],
        risk: RiskAssessment,
    ) -> list[str]:
        chain: list[str] = []
        step = 1

        # Top scenario match (highest confidence)
        if matches and matches[0].confidence > 0.4:
            top = matches[0]
            features_note = (
                f" — matched on: {', '.join(top.matched_features[:4])}"
                if top.matched_features
                else ""
            )
            chain.append(
                f"{step}. Scenario match: '{top.display_name}' "
                f"(confidence {top.confidence:.0%}){features_note}."
            )
            step += 1

        # Contributing risk factors from the risk scorer
        for factor in risk.contributing_factors[:4]:
            chain.append(f"{step}. {factor}")
            step += 1

        # Domain-specific signal-driven inferences
        if f.domain == "compute":
            if f.error_rate > 0.3 and f.latency_p99 > 2.0:
                chain.append(
                    f"{step}. Combined high error rate ({f.error_rate:.0%}) and "
                    f"extreme latency ({f.latency_p99:.2f}s) — upstream dependency "
                    "failure or thread pool saturation is likely."
                )
                step += 1
            elif f.error_rate > 0.1:
                chain.append(
                    f"{step}. Error rate {f.error_rate:.1%} above warning threshold "
                    "— likely application-level failure, not infrastructure."
                )
                step += 1
            if f.cpu_usage > 0.90:
                chain.append(
                    f"{step}. CPU saturation ({f.cpu_usage:.0%}) causing request "
                    "queuing and latency amplification."
                )
                step += 1
            if f.memory_usage > 0.90:
                chain.append(
                    f"{step}. Memory pressure ({f.memory_usage:.0%}) may indicate "
                    "a leak or under-provisioning triggering GC pauses."
                )
                step += 1
        else:  # storage
            if f.pool_usage_pct > 0.85:
                chain.append(
                    f"{step}. Pool fill at {f.pool_usage_pct:.0%} — approaching "
                    "capacity limit; new writes will fail when full."
                )
                step += 1
            if f.degraded_pgs > 5:
                chain.append(
                    f"{step}. {f.degraded_pgs} degraded Placement Groups indicate "
                    "active replication recovery or OSD failure."
                )
                step += 1
            if f.osd_up_count < f.osd_total_count and f.osd_total_count > 0:
                down = f.osd_total_count - f.osd_up_count
                chain.append(
                    f"{step}. {down}/{f.osd_total_count} OSD(s) down — check hardware "
                    "and network connectivity."
                )
                step += 1

        if f.log_anomaly_detected:
            chain.append(
                f"{step}. Log error anomaly spike — error pattern deviated "
                "significantly from rolling baseline."
            )
            step += 1

        if not chain:
            chain.append(
                "1. Insufficient signal for confident root cause analysis — "
                "manual investigation required."
            )

        return chain

    # ── Private: predicted impact ─────────────────────────────────────────────

    def _build_predicted_impact(self, f: ObsFeatures, risk: RiskAssessment) -> str:
        impact_map = {
            "critical": (
                "Severe user-facing degradation. Multiple services or the entire "
                "cluster affected. SLA breach imminent without immediate action."
            ),
            "high": (
                "Significant user-facing degradation. Error rate or latency above SLO. "
                "On-call escalation warranted."
            ),
            "medium": (
                "Partial degradation detected. Subset of users or operations affected. "
                "Monitor closely and prepare remediations."
            ),
            "low": (
                "Minor anomaly — no confirmed user impact yet. "
                "Watch for escalation before committing remediation."
            ),
        }
        base = impact_map.get(risk.risk_level, impact_map["medium"])
        extras: list[str] = []
        if risk.time_to_impact and risk.time_to_impact not in ("unknown", ""):
            extras.append(f"Estimated time to user-visible impact: {risk.time_to_impact}.")
        if risk.blast_radius and risk.blast_radius not in ("unknown", ""):
            extras.append(f"Blast radius: {risk.blast_radius}.")
        return " ".join([base] + extras)

    # ── Private: recommended actions ─────────────────────────────────────────

    def _build_recommended_actions(
        self,
        f: ObsFeatures,
        matches: list[ScenarioMatch],
        risk: RiskAssessment,
    ) -> list[str]:
        actions: list[str] = []
        step = 1

        if matches:
            top = matches[0]
            actions.append(
                f"{step}. Execute '{top.scenario_id}' playbook from the scenario catalog."
            )
            step += 1

        if f.domain == "compute":
            if f.error_rate > 0.2:
                actions.append(
                    f"{step}. Check downstream dependency health; enable circuit breaker "
                    "if an upstream service is failing."
                )
                step += 1
            if f.cpu_usage > 0.85:
                actions.append(
                    f"{step}. Trigger horizontal scale-out to relieve CPU saturation."
                )
                step += 1
            if f.memory_usage > 0.90:
                actions.append(
                    f"{step}. Rolling restart to clear memory pressure — capture heap "
                    "dump before restarting."
                )
                step += 1
            if f.latency_p99 > 2.0:
                actions.append(
                    f"{step}. Review connection pool size and upstream timeout "
                    "configurations to prevent cascading queuing."
                )
                step += 1
        else:  # storage
            if f.pool_usage_pct > 0.85:
                actions.append(
                    f"{step}. Emergency pool expansion or workload redistribution "
                    "to prevent write failures."
                )
                step += 1
            if f.osd_up_count < f.osd_total_count and f.osd_total_count > 0:
                actions.append(
                    f"{step}. Investigate OSD failures: check SMART data, dmesg, "
                    "and network connectivity."
                )
                step += 1
            if f.degraded_pgs > 0:
                actions.append(
                    f"{step}. Monitor PG recovery: do not add write load until cluster "
                    "returns to HEALTH_OK."
                )
                step += 1

        if risk.risk_level in ("high", "critical"):
            actions.append(
                f"{step}. Page on-call SRE immediately — do not wait for "
                "auto-remediation to confirm success."
            )
            step += 1

        actions.append(
            f"{step}. Post resolution comment to xyOps ticket with confirmed "
            "root cause and timeline for post-mortem."
        )

        return actions

    # ── Private: computed fields ──────────────────────────────────────────────

    def _compute_urgency(self, risk: RiskAssessment) -> str:
        if risk.risk_level == "critical":
            return "critical"
        if risk.risk_level == "high":
            return "high"
        if risk.time_to_impact in ("immediate", "~5 min"):
            return "high"
        if risk.risk_level == "medium":
            return "medium"
        return "low"

    def _compute_evidence_strength(
        self,
        f: ObsFeatures,
        matches: list[ScenarioMatch],
    ) -> str:
        score = 0
        if matches and matches[0].confidence > 0.7:
            score += 2
        elif matches and matches[0].confidence > 0.4:
            score += 1
        if f.log_anomaly_detected:
            score += 1
        if f.recent_error_count > 10:
            score += 1
        if f.error_rate > 0.1 or f.pool_usage_pct > 0.7:
            score += 1
        if score >= 4:
            return "strong"
        if score >= 2:
            return "moderate"
        return "weak"

    def _compute_autonomy(
        self,
        risk: RiskAssessment,
        top_match: ScenarioMatch | None,
    ) -> str:
        if risk.risk_level == "critical":
            return "human_only"
        if not risk.requires_approval:
            return "autonomous"
        if risk.risk_score >= 0.7:
            return "approval_gated"
        if risk.risk_level == "low":
            return "autonomous"
        return "approval_gated"
