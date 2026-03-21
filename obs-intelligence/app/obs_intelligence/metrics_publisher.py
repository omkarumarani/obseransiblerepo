"""
obs_intelligence/metrics_publisher.py
──────────────────────────────────────
Prometheus-client gauges and counters for the obs-intelligence engine.

All metrics use the obs_intelligence_ prefix and are served via
GET /metrics on the obs-intelligence FastAPI service (port 9100).
Prometheus scrapes them from obs-intelligence:9100/metrics.
"""

from prometheus_client import Counter as PromCounter, Gauge as PromGauge, Histogram as PromHistogram

# ── Scenario correlation metrics ──────────────────────────────────────────────
obs_intelligence_scenario_match_confidence = PromGauge(
    "obs_intelligence_scenario_match_confidence",
    "Confidence score (0.0–1.0) of the most recent scenario match",
    ["domain", "scenario_id"],
)
obs_intelligence_scenario_matches_total = PromCounter(
    "obs_intelligence_scenario_matches_total",
    "Total scenario matches identified by the intelligence engine",
    ["domain", "scenario_id"],
)

# ── Risk assessment metrics ───────────────────────────────────────────────────
obs_intelligence_risk_score = PromGauge(
    "obs_intelligence_risk_score",
    "Current risk score computed by the intelligence engine (0.0–1.0)",
    ["domain", "risk_level"],
)

# ── Anomaly detection metrics ─────────────────────────────────────────────────
obs_intelligence_anomaly_z_score = PromGauge(
    "obs_intelligence_anomaly_z_score",
    "Z-score for a monitored metric relative to its rolling baseline",
    ["metric_name", "domain"],
)
obs_intelligence_anomaly_detected_total = PromCounter(
    "obs_intelligence_anomaly_detected_total",
    "Total anomalies detected by the intelligence engine",
    ["metric_name", "anomaly_type", "domain"],
)

# ── Forecast metrics ──────────────────────────────────────────────────────────
obs_intelligence_forecast_breach_minutes = PromGauge(
    "obs_intelligence_forecast_breach_minutes",
    "Predicted minutes until threshold breach (0 if no breach predicted)",
    ["metric_name"],
)

# ── Predictive alert metrics ──────────────────────────────────────────────
obs_intelligence_predictive_alerts_sent_total = PromCounter(
    "obs_intelligence_predictive_alerts_sent_total",
    "Total predictive alerts sent to domain agents (high risk, no active Prometheus alert)",
    ["domain"],
)

# ── Scenario outcome metrics ───────────────────────────────────────────────
obs_intelligence_scenario_outcome_total = PromCounter(
    "obs_intelligence_scenario_outcome_total",
    "Scenario outcomes recorded after alert resolution or disposition",
    ["scenario_id", "outcome"],  # outcome: resolved | escalated | declined | timedout
)

# ── Alertmanager webhook metrics ─────────────────────────────────────────────
obs_intelligence_webhook_alerts_total = PromCounter(
    "obs_intelligence_webhook_alerts_total",
    "Total Alertmanager alerts received via the /webhook endpoint",
    ["status", "severity"],  # status: firing | resolved
)

# ── Background loop performance ───────────────────────────────────────────────
obs_intelligence_analysis_loop_duration_seconds = PromHistogram(
    "obs_intelligence_analysis_loop_duration_seconds",
    "Wall-clock duration of each periodic analysis loop run",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)
obs_intelligence_analysis_loop_runs_total = PromCounter(
    "obs_intelligence_analysis_loop_runs_total",
    "Total analysis loop iterations completed",
    ["status"],   # success | error
)
obs_intelligence_forecast_loop_runs_total = PromCounter(
    "obs_intelligence_forecast_loop_runs_total",
    "Total forecasting loop iterations completed",
    ["status"],   # success | error
)

# -- Learning layer metrics ---------------------------------------------------
obs_intelligence_external_validation_total = PromCounter(
    "obs_intelligence_external_validation_total",
    "Total externally-authored analyses recorded by the learning layer",
    ["domain", "provider", "status"],
)
obs_intelligence_local_validation_total = PromCounter(
    "obs_intelligence_local_validation_total",
    "Total local validation attempts by verdict",
    ["domain", "verdict"],
)
obs_intelligence_local_validation_duration_seconds = PromHistogram(
    "obs_intelligence_local_validation_duration_seconds",
    "Latency of local corroboration requests",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
)
