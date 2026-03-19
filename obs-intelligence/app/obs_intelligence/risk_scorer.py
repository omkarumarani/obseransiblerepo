"""
obs_intelligence/risk_scorer.py
────────────────────────────────────────────────────────────────────────────────
Deterministic risk scorer.

Computes a normalised 0.0–1.0 risk score from four evidence signals:

  Component          Weight   Source
  ─────────────────  ──────   ──────────────────────────────────────────────
  severity_score      30 %   alert severity label (warning=0.4, critical=0.85)
  confidence_score    40 %   best scenario-match confidence (0.0–1.0)
  log_anomaly_score   15 %   error count + log_anomaly_detected flag
  forecast_urgency    15 %   domain-specific proximity to critical thresholds

Risk levels:
  0.00 – 0.29  →  low
  0.30 – 0.59  →  medium
  0.60 – 0.79  →  high
  0.80 – 1.00  →  critical

Public API
──────────
    score_risk(features, best_match, domain) -> RiskAssessment
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from obs_intelligence.models import ObsFeatures, RiskAssessment, ScenarioMatch

if TYPE_CHECKING:
    pass

logger = logging.getLogger("obs_intelligence.risk_scorer")

# ── Weighting constants ───────────────────────────────────────────────────────
_W_SEVERITY   = 0.30
_W_CONFIDENCE = 0.40
_W_LOG        = 0.15
_W_FORECAST   = 0.15

# ── Severity base scores ─────────────────────────────────────────────────────
_SEVERITY_SCORES: dict[str, float] = {
    "info":     0.1,
    "warning":  0.4,
    "critical": 0.85,
    "page":     1.0,
}

# ── Risk level thresholds ─────────────────────────────────────────────────────
_LEVEL_THRESHOLDS = [
    (0.80, "critical"),
    (0.60, "high"),
    (0.30, "medium"),
    (0.00, "low"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Public
# ═══════════════════════════════════════════════════════════════════════════════

def score_risk(
    features: ObsFeatures,
    best_match: ScenarioMatch | None,
    domain: str,
) -> RiskAssessment:
    """
    Compute a deterministic RiskAssessment from features + scenario match.

    Parameters
    ----------
    features :
        The ObsFeatures snapshot for this incident.
    best_match :
        The highest-confidence ScenarioMatch from the correlator, or None
        if no scenario matched.
    domain :
        "compute" | "storage" — used to select domain-specific forecast signals.

    Returns
    -------
    RiskAssessment
        Fully populated risk assessment including contributing_factors, blast_radius,
        and time_to_impact estimates.
    """
    contributing: list[str] = []

    # Component 1: severity
    sev = _SEVERITY_SCORES.get(features.severity.lower(), 0.3)
    contributing.append(f"severity={features.severity} (score={sev:.2f})")

    # Component 2: scenario confidence
    conf = best_match.confidence if best_match else 0.0
    if best_match:
        contributing.append(
            f"scenario_confidence={conf:.2f} ({best_match.display_name})"
        )
    else:
        contributing.append("no_scenario_match (confidence=0.00)")

    # Component 3: log anomaly
    log_score = _log_anomaly_score(features, contributing)

    # Component 4: forecast urgency
    forecast_score = _forecast_urgency(features, domain, contributing)

    raw = (
        _W_SEVERITY   * sev
        + _W_CONFIDENCE * conf
        + _W_LOG        * log_score
        + _W_FORECAST   * forecast_score
    )
    risk_score = min(1.0, max(0.0, raw))
    risk_level = _to_level(risk_score)

    blast_radius = _blast_radius(features, domain, risk_level)
    time_to_impact = _time_to_impact(features, domain, risk_score)
    requires_approval = risk_level in ("high", "critical")

    logger.info(
        "Risk scored  alert=%s  domain=%s  score=%.3f  level=%s",
        features.alert_name, domain, risk_score, risk_level,
    )

    return RiskAssessment(
        risk_score=risk_score,
        risk_level=risk_level,
        contributing_factors=contributing,
        blast_radius=blast_radius,
        time_to_impact=time_to_impact,
        requires_approval=requires_approval,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _log_anomaly_score(features: ObsFeatures, contributing: list[str]) -> float:
    """
    Score based on log error counts and anomaly flag.

    Returns a value in [0.0, 1.0].
    """
    score = 0.0

    if features.log_anomaly_detected:
        score = min(1.0, score + 0.6)
        contributing.append("log_anomaly_detected=True")

    # Scale error count: each error adds 0.05, capped at 1.0
    error_contribution = min(1.0, features.recent_error_count * 0.05)
    score = min(1.0, score + error_contribution * 0.4)

    if features.recent_error_count > 0:
        contributing.append(f"recent_error_count={features.recent_error_count}")

    return score


def _forecast_urgency(
    features: ObsFeatures, domain: str, contributing: list[str]
) -> float:
    """
    Domain-specific proximity to critical thresholds.

    Compute: high error_rate, latency_p99 breaches
    Storage: pool fill level, OSD availability, degraded PGs

    Returns a value in [0.0, 1.0].
    """
    urgency = 0.0

    if domain == "compute":
        # Error rate urgency: 0% → 0.0, 10%+ → 1.0
        if features.error_rate > 0:
            urgency = max(urgency, min(1.0, features.error_rate / 0.10))
            contributing.append(f"error_rate={features.error_rate:.4f}")

        # Latency urgency: >1s p99 → escalating urgency
        if features.latency_p99 > 0:
            lat_urgency = min(1.0, features.latency_p99 / 3.0)
            urgency = max(urgency, lat_urgency)
            if features.latency_p99 > 0.5:
                contributing.append(f"latency_p99={features.latency_p99:.3f}s")

    elif domain == "storage":
        # Pool fill urgency: 0.70 → 0.0, 0.95+ → 1.0
        if features.pool_usage_pct > 0.70:
            pool_urgency = min(1.0, (features.pool_usage_pct - 0.70) / 0.25)
            urgency = max(urgency, pool_urgency)
            contributing.append(f"pool_usage_pct={features.pool_usage_pct:.2f}")

        # OSD availability urgency
        if features.osd_total_count > 0:
            osd_ratio = features.osd_up_count / features.osd_total_count
            if osd_ratio < 1.0:
                osd_urgency = min(1.0, (1.0 - osd_ratio) * 4.0)
                urgency = max(urgency, osd_urgency)
                contributing.append(
                    f"osd_availability={features.osd_up_count}/{features.osd_total_count}"
                )

        # Degraded PGs urgency: 0 → 0.0, 100+ → 1.0
        if features.degraded_pgs > 0:
            pg_urgency = min(1.0, features.degraded_pgs / 100.0)
            urgency = max(urgency, pg_urgency)
            contributing.append(f"degraded_pgs={features.degraded_pgs}")

    return urgency


def _to_level(score: float) -> str:
    for threshold, level in _LEVEL_THRESHOLDS:
        if score >= threshold:
            return level
    return "low"


def _blast_radius(features: ObsFeatures, domain: str, risk_level: str) -> str:
    if domain == "compute":
        if features.request_rate > 100:
            return "platform-wide (high-traffic service)"
        if risk_level == "critical":
            return "all downstream consumers"
        if risk_level == "high":
            return "single service + dependents"
        return "single service"

    elif domain == "storage":
        if features.osd_total_count > 0 and features.osd_up_count < features.osd_total_count:
            return "storage cluster + all PVCs"
        if features.pool_usage_pct > 0.90:
            return "all workloads writing to pool"
        if features.degraded_pgs > 50:
            return "storage cluster health"
        return "storage service"

    return "unknown"


def _time_to_impact(features: ObsFeatures, domain: str, risk_score: float) -> str | None:
    if risk_score >= 0.80:
        return "immediate"
    if domain == "storage" and features.pool_usage_pct > 0.85:
        return "~10–30 min"
    if domain == "compute" and features.error_rate > 0.05:
        return "immediate"
    if risk_score >= 0.60:
        return "~5 min"
    if risk_score >= 0.30:
        return "~15–30 min"
    return None
