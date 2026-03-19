"""
obs_intelligence/evidence_builder.py
────────────────────────────────────────────────────────────────────────────────
Evidence builder.

Assembles a complete EvidenceReport from the intelligence pipeline outputs
and generates human-readable evidence lines suitable for inclusion in xyOps
ticket bodies and OTel span attributes.

Public API
──────────
    build_evidence(
        trace_id, incident_id, features, matches, risk, recommendations
    ) -> EvidenceReport

    evidence_lines(report) -> list[str]
        Returns a flat list of plain-English evidence observations for use
        in ticket body builders.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from obs_intelligence.models import (
    EvidenceReport,
    ObsFeatures,
    RiskAssessment,
    Recommendation,
    ScenarioMatch,
)

logger = logging.getLogger("obs_intelligence.evidence_builder")


# ═══════════════════════════════════════════════════════════════════════════════
# Public
# ═══════════════════════════════════════════════════════════════════════════════

def build_evidence(
    trace_id: str,
    incident_id: str,
    features: ObsFeatures,
    matches: list[ScenarioMatch],
    risk: RiskAssessment,
    recommendations: list[Recommendation],
) -> EvidenceReport:
    """
    Compile an EvidenceReport from all intelligence pipeline outputs.

    Parameters
    ----------
    trace_id :       OTel trace ID from the originating pipeline span.
    incident_id :    xyOps ticket ID (may be empty string before ticket creation).
    features :       Extracted ObsFeatures for this incident.
    matches :        All ScenarioMatches above threshold, sorted by confidence desc.
    risk :           RiskAssessment produced by risk_scorer.
    recommendations: Ranked Recommendations from recommender.

    Returns
    -------
    EvidenceReport
        Fully-assembled evidence bundle.
    """
    return EvidenceReport(
        trace_id=trace_id,
        incident_id=incident_id,
        features=features,
        scenario_matches=matches,
        risk=risk,
        recommendations=recommendations,
        generated_at=datetime.now(timezone.utc),
    )


def evidence_lines(report: EvidenceReport) -> list[str]:
    """
    Generate human-readable evidence observation lines from an EvidenceReport.

    These lines are written into the xyOps ticket body to give an SRE
    an at-a-glance view of what signals drove the risk score and recommendation.

    Returns
    -------
    list[str]
        Ordered, deduplicated observations.  Each line is a self-contained
        plain-English statement, e.g.
          "• error_rate = 4.2% (threshold: 1.0%)"
          "• Scenario match: Error Spike (confidence: 0.82)"
          "• Risk level: HIGH (score: 0.71)"
    """
    lines: list[str] = []
    f = report.features

    # ── Alert identity ────────────────────────────────────────────────────────
    lines.append(f"• Alert: {f.alert_name}  |  Service: {f.service_name}  |  Severity: {f.severity.upper()}")
    lines.append(f"• Domain: {f.domain}  |  Captured at: {f.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    # ── Compute signals ───────────────────────────────────────────────────────
    if f.domain == "compute":
        if f.error_rate > 0:
            badge = " ⚠" if f.error_rate > 0.01 else ""
            lines.append(f"• Error rate: {f.error_rate * 100:.2f}%{badge}")
        if f.latency_p99 > 0:
            badge = " ⚠" if f.latency_p99 > 0.5 else ""
            lines.append(f"• Latency p99: {f.latency_p99 * 1000:.1f} ms{badge}")
        if f.latency_p95 > 0:
            lines.append(f"• Latency p50: {f.latency_p95 * 1000:.1f} ms")
        if f.request_rate > 0:
            lines.append(f"• Request rate: {f.request_rate:.1f} rps")
        if f.cpu_usage > 0:
            lines.append(f"• CPU usage: {f.cpu_usage * 100:.1f}%")
        if f.memory_usage > 0:
            lines.append(f"• Memory usage: {f.memory_usage * 100:.1f}%")

    # ── Storage signals ───────────────────────────────────────────────────────
    elif f.domain == "storage":
        if f.osd_total_count > 0:
            badge = " ⚠" if f.osd_up_count < f.osd_total_count else ""
            lines.append(f"• OSD status: {f.osd_up_count}/{f.osd_total_count} up{badge}")
        if f.pool_usage_pct > 0:
            badge = " ⚠" if f.pool_usage_pct > 0.75 else ""
            lines.append(f"• Pool usage: {f.pool_usage_pct * 100:.1f}%{badge}")
        if f.cluster_health_score < 2:
            score_label = {0: "ERROR", 1: "WARN"}.get(f.cluster_health_score, "?")
            lines.append(f"• Cluster health: {score_label} (score={f.cluster_health_score}) ⚠")
        if f.degraded_pgs > 0:
            lines.append(f"• Degraded placement groups: {f.degraded_pgs} ⚠")
        if f.io_latency > 0:
            lines.append(f"• PVC IO latency: {f.io_latency * 1000:.1f} ms")
        if f.pvc_iops > 0:
            lines.append(f"• PVC IOPS: {f.pvc_iops:.0f}")

    # ── Log signals ───────────────────────────────────────────────────────────
    if f.recent_error_count > 0:
        lines.append(f"• Log errors (last 5 min): {f.recent_error_count}")
    if f.recent_warning_count > 0:
        lines.append(f"• Log warnings (last 5 min): {f.recent_warning_count}")
    if f.log_anomaly_detected:
        lines.append("• Log anomaly: elevated error rate detected ⚠")

    # ── Scenario matches ──────────────────────────────────────────────────────
    if report.scenario_matches:
        top = report.scenario_matches[0]
        lines.append(
            f"• Best scenario match: {top.display_name} "
            f"(confidence: {top.confidence:.0%})"
        )
        if len(report.scenario_matches) > 1:
            others = ", ".join(
                f"{m.display_name} ({m.confidence:.0%})"
                for m in report.scenario_matches[1:3]
            )
            lines.append(f"• Other candidates: {others}")
        if top.matched_features:
            lines.append(f"• Matched conditions: {', '.join(top.matched_features)}")

    # ── Risk assessment ───────────────────────────────────────────────────────
    if report.risk:
        r = report.risk
        lines.append(
            f"• Risk level: {r.risk_level.upper()} "
            f"(score: {r.risk_score:.2f})"
        )
        if r.blast_radius:
            lines.append(f"• Blast radius: {r.blast_radius}")
        if r.time_to_impact:
            lines.append(f"• Time to impact: {r.time_to_impact}")

    # ── Top recommendation ────────────────────────────────────────────────────
    if report.recommendations:
        rec = report.recommendations[0]
        autonomy_tag = "autonomous" if rec.autonomous else "approval required"
        lines.append(
            f"• Recommended action: {rec.action_type} "
            f"[{autonomy_tag}] (confidence: {rec.confidence:.0%})"
        )
        if rec.estimated_duration:
            lines.append(f"• Estimated remediation time: {rec.estimated_duration}")

    return lines
