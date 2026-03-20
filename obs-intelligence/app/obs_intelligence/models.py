"""
obs-intelligence/app/obs_intelligence/models.py
────────────────────────────────────────────────────────────────────────────
Shared data models for the multi-agent AIOps platform.

These dataclasses flow through the entire intelligence pipeline:

    ObsFeatures        Raw signals from Prometheus + Loki
         │
         ▼
    ScenarioMatch      Which pre-defined pattern this looks like
         │
         ▼
    RiskAssessment     How dangerous is this incident right now
         │
         ▼
    Recommendation     What action(s) to take
         │
         ▼
    EvidenceReport     Complete bundle shipped to the agent pipeline

    AnomalySignal      Statistical deviation in a single metric series
    ForecastResult     Predictive output — when will a threshold be breached
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# Input layer
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ObsFeatures:
    """
    Raw observability features extracted from Prometheus metrics and Loki logs.

    Produced by the domain agents during the 'fetch-metrics' and 'fetch-logs'
    steps of their 6-step pipeline.  Passed into the intelligence engine as
    the primary input for scenario matching, risk scoring, and anomaly detection.
    """

    # ── Alert identity ────────────────────────────────────────────────────────
    alert_name: str
    service_name: str
    severity: str                       # "warning" | "critical"
    domain: str                         # "compute" | "storage"
    timestamp: datetime

    # ── Compute metrics ───────────────────────────────────────────────────────
    error_rate: float = 0.0             # proportion of 5xx responses (0.0 – 1.0)
    latency_p95: float = 0.0            # 95th-percentile request latency (seconds)
    latency_p99: float = 0.0            # 99th-percentile request latency (seconds)
    cpu_usage: float = 0.0              # CPU utilisation fraction (0.0 – 1.0)
    memory_usage: float = 0.0          # Memory utilisation fraction (0.0 – 1.0)
    request_rate: float = 0.0           # requests per second
    active_connections: int = 0

    # ── Storage metrics ───────────────────────────────────────────────────────
    osd_up_count: int = 0               # number of OSDs with status=up
    osd_total_count: int = 0            # total number of OSDs
    pool_usage_pct: float = 0.0         # pool fill level (0.0 – 1.0)
    cluster_health_score: int = 2       # 0 = ERROR, 1 = WARN, 2 = OK
    degraded_pgs: int = 0               # number of degraded placement groups
    io_latency: float = 0.0             # PVC IO latency (seconds)
    pvc_iops: float = 0.0               # PVC IOPS

    # ── Log-derived signals ───────────────────────────────────────────────────
    recent_error_count: int = 0         # ERROR lines in last 5 min
    recent_warning_count: int = 0       # WARN  lines in last 5 min
    log_anomaly_detected: bool = False  # heuristic spike in error log rate
    # ── Recurrence tracking ─────────────────────────────────────────────
    recurrence_count: int = 0
    # Number of times the same alert_name has fired in the last 6 hours.
    # Populated by the background loop alert-recurrence tracker before
    # features are passed to scenario correlation.  Drives the
    # recurring_failure_signature scenario (human_only autonomy when >= 3).
    # ── Raw alert metadata ────────────────────────────────────────────────────
    labels: dict = field(default_factory=dict)
    annotations: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Correlation layer
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScenarioMatch:
    """
    A matched scenario pattern with confidence score.

    Scenario definitions live in obs-intelligence/scenarios/{domain}/*.yaml.
    The scenario correlator loads them at startup and scores each one against
    the incoming ObsFeatures.  The top-scoring match drives the risk assessment.
    """

    scenario_id: str                    # e.g. "HIGH_ERROR_RATE", "OSD_DOWN"
    display_name: str                   # human-readable label
    confidence: float                   # 0.0 – 1.0 (how strongly features match)
    domain: str                         # "compute" | "storage"

    matched_features: list[str] = field(default_factory=list)
    # Names of the ObsFeatures fields that contributed to this match.

    scenario_file: str | None = None
    # Relative path to the YAML file that defines this scenario,
    # e.g. "scenarios/compute/high_error_rate.yaml"


# ─────────────────────────────────────────────────────────────────────────────
# Risk layer
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RiskAssessment:
    """
    Risk scoring result produced by the intelligence engine.

    risk_score is normalised 0.0 – 1.0.  The mapping to risk_level is:
        0.00 – 0.29  →  low
        0.30 – 0.59  →  medium
        0.60 – 0.79  →  high
        0.80 – 1.00  →  critical
    """

    risk_score: float                   # 0.0 – 1.0
    risk_level: str                     # "low" | "medium" | "high" | "critical"

    contributing_factors: list[str] = field(default_factory=list)
    # Plain-English explanation of what drove the score up,
    # e.g. ["error_rate > 0.3", "log_anomaly_detected", "latency_p99 > 2s"]

    blast_radius: str = "unknown"
    # Estimated scope of user / system impact,
    # e.g. "single service", "storage cluster", "platform-wide"

    time_to_impact: str | None = None
    # How quickly this becomes user-visible if unaddressed,
    # e.g. "immediate", "~5 min", "~2 hours"

    requires_approval: bool = True
    # Whether the recommended action must pass through a human approval gate.


# ─────────────────────────────────────────────────────────────────────────────
# Action layer
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Recommendation:
    """
    A recommended remediation action produced by the intelligence engine.

    Multiple recommendations may be returned, sorted by confidence descending.
    The agent pipeline carries out the top recommendation unless overridden.
    """

    action_type: str
    # Machine-readable action key understood by the approval workflow,
    # e.g. "restart_service", "osd_reweight", "scale_up", "pvc_throttle"

    display_name: str                   # human-readable title for the xyOps ticket
    description: str                    # fuller explanation shown in ticket body
    confidence: float                   # 0.0 – 1.0

    autonomous: bool = False
    # True = this action is safe to execute without human sign-off
    # (low-blast-radius actions such as cache flushes or metric-only responses).

    ansible_playbook: str | None = None
    # Name of the Ansible playbook to run, e.g. "restart_service.yml".
    # None if the action is handled by xyOps workflow directly.

    xyops_workflow: str | None = None
    # xyOps workflow name to trigger, e.g. "Storage AIOps Agent Pipeline".

    estimated_duration: str | None = None
    # Rough wall-clock expectation, e.g. "30s", "2 min", "~10 min".

    rollback_plan: str | None = None
    # What to do if the action makes things worse.


# ─────────────────────────────────────────────────────────────────────────────
# Output layer
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvidenceReport:
    """
    Complete evidence bundle produced by the obs-intelligence engine.

    Returned to the domain agent after the 'analyse' step of the pipeline.
    Serialised into the xyOps ticket body and attached as OTel span attributes
    so it is discoverable from the Tempo trace.
    """

    trace_id: str                       # OTel trace ID from the originating span
    incident_id: str                    # xyOps ticket ID (set after ticket creation)

    features: ObsFeatures               # the raw signal snapshot that was analysed

    scenario_matches: list[ScenarioMatch] = field(default_factory=list)
    # Sorted by confidence descending.  First entry is the primary scenario.

    risk: RiskAssessment | None = None
    # None only if the engine failed and fell back to a default assessment.

    recommendations: list[Recommendation] = field(default_factory=list)
    # Sorted by confidence descending.  Agent pipeline acts on [0].

    ai_summary: str | None = None
    # LLM-generated natural-language narrative of the incident.
    # Populated by the AI analyst step; None if LLM is unavailable.

    generated_at: datetime | None = None
    # Wall-clock time when this report was produced by the engine.

    engine_version: str = "1.0.0"


# ─────────────────────────────────────────────────────────────────────────────
# Anomaly detection layer
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AnomalySignal:
    """
    A statistical anomaly detected in a single metric time series.

    Produced by the anomaly detector in the intelligence engine.  Multiple
    AnomalySignals may be emitted per incident — one per suspicious metric.
    Attached to an EvidenceReport or used independently for proactive alerting.
    """

    metric_name: str                    # Prometheus metric name, e.g. "error_rate"
    current_value: float                # the most recent sampled value
    baseline_mean: float                # rolling mean over the lookback window
    baseline_stddev: float              # rolling std-dev over the lookback window
    z_score: float                      # (current_value - baseline_mean) / baseline_stddev

    anomaly_type: str = "spike"
    # "spike"  — sudden upward deviation  (z > +threshold)
    # "drop"   — sudden downward deviation (z < -threshold)
    # "trend"  — monotonic drift over time (separate trend detector)

    detected_at: datetime | None = None
    confidence: float = 0.0             # normalised certainty of the anomaly call


# ─────────────────────────────────────────────────────────────────────────────
# Forecasting layer
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ForecastResult:
    """
    Predictive forecast for a metric — used for proactive / pre-fire alerting.

    When predicted_breach is not None the platform can open a xyOps ticket
    *before* Prometheus fires, giving the on-call engineer lead time.
    """

    metric_name: str

    forecast_values: list[float] = field(default_factory=list)
    forecast_timestamps: list[datetime] = field(default_factory=list)
    # Parallel lists — forecast_values[i] is predicted at forecast_timestamps[i].

    predicted_breach: datetime | None = None
    # Time at which the metric is predicted to cross `threshold`.
    # None if no breach is expected within the forecast horizon.

    threshold: float | None = None
    # The alerting threshold the forecast is comparing against.

    confidence_interval_lower: list[float] = field(default_factory=list)
    confidence_interval_upper: list[float] = field(default_factory=list)
    # 95 % confidence bands — same length as forecast_values.

    model_used: str = "linear"
    # "linear"       — simple linear regression (fast, low data requirement)
    # "holt_winters" — triple exponential smoothing (seasonal patterns)
    # "arima"        — ARIMA via statsmodels (Phase 4+, requires more history)

    horizon_minutes: int = 60
    # How far ahead this forecast looks.
