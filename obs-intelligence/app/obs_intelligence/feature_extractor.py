"""
obs_intelligence/feature_extractor.py
────────────────────────────────────────────────────────────────────────────────
Feature extraction — maps raw agent session data (Prometheus metrics + Loki
logs) into a typed ObsFeatures snapshot ready for scenario correlation and
risk scoring.

Public API
──────────
    extract_features(
        alert_name, service_name, severity, domain, metrics, logs
    ) -> ObsFeatures

The *metrics* parameter accepts either format produced by the domain agents:

  Compute (flat str→str dict from fetch_prometheus_context):
    {"error_rate_pct": "1.23", "p99_latency_ms": "456.0",
     "p50_latency_ms": "123.0", "rps": "12.5"}

  Storage (nested dict from fetch_storage_metrics):
    {"raw": {"osd_status": [...], "pool_fill_pct": [...], ...},
     "summary": "Storage Metrics Snapshot:\\n  ..."}
    — or a flat dict when called from the deterministic path.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

from obs_intelligence.models import ObsFeatures

logger = logging.getLogger("obs_intelligence.feature_extractor")

# ─────────────────────────────────────────────────────────────────────────────
# Public
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(
    alert_name: str,
    service_name: str,
    severity: str,
    domain: str,
    metrics: dict[str, Any],
    logs: str,
) -> ObsFeatures:
    """
    Convert raw pipeline session data into a typed ObsFeatures snapshot.

    Works for both compute and storage domains.  Unknown or unparseable
    metric values are silently substituted with zero/default rather than
    raising, so the pipeline degrades gracefully if Prometheus is unavailable.
    """
    f = ObsFeatures(
        alert_name=alert_name,
        service_name=service_name,
        severity=severity,
        domain=domain,
        timestamp=datetime.now(timezone.utc),
    )

    if domain == "compute":
        _fill_compute(f, metrics)
    elif domain == "storage":
        _fill_storage(f, metrics)
    else:
        logger.warning("Unknown domain '%s' — metrics will not be parsed", domain)

    _fill_log_signals(f, logs)
    return f


# ─────────────────────────────────────────────────────────────────────────────
# Internal: compute metrics mapping
# ─────────────────────────────────────────────────────────────────────────────

def _fill_compute(f: ObsFeatures, metrics: dict[str, Any]) -> None:
    """
    Parse compute golden-signal dict as returned by fetch_prometheus_context().

    Keys are str → str (formatted float strings or "no data" / "parse error").
    """
    f.error_rate   = _safe_float(metrics.get("error_rate_pct"), scale=0.01)
    # error_rate_pct is percentage (0–100) → ObsFeatures wants fraction (0.0–1.0)

    f.latency_p99  = _safe_float(metrics.get("p99_latency_ms"), scale=0.001)
    # milliseconds → seconds

    f.latency_p95  = _safe_float(metrics.get("p50_latency_ms"), scale=0.001)
    # Use p50 as a stand-in for p95 when only p50 is available; the scenario
    # conditions use latency_p95 / latency_p99 field names.

    f.request_rate = _safe_float(metrics.get("rps"))
    # requests per second — no unit conversion needed

    logger.debug(
        "Compute features  error_rate=%.4f  latency_p99=%.4f  rps=%.2f",
        f.error_rate, f.latency_p99, f.request_rate,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internal: storage metrics mapping
# ─────────────────────────────────────────────────────────────────────────────

def _fill_storage(f: ObsFeatures, metrics: dict[str, Any]) -> None:
    """
    Parse storage metrics dict.

    Accepts either:
      • The full dict from fetch_storage_metrics()  → {"raw": {...}, "summary": "..."}
      • A flat dict with Prometheus result lists per metric name

    Prometheus raw results are lists of {"metric": {...}, "value": [ts, "value_str"]}.
    We take the *first* result for each query (matching storage-simulator topology).
    """
    raw: dict[str, list] = metrics.get("raw", metrics)
    # Fallback: treat `metrics` itself as the raw dict when "raw" is absent.

    # ── OSD counts ───────────────────────────────────────────────────────────
    osd_results = raw.get("osd_status", [])
    f.osd_up_count = sum(
        1 for r in osd_results if _prom_float(r) >= 1.0
    )
    f.osd_total_count = len(osd_results) if osd_results else 0

    # ── Pool fill level ───────────────────────────────────────────────────────
    pool_results = raw.get("pool_fill_pct", [])
    if pool_results:
        f.pool_usage_pct = _prom_float(pool_results[0])

    # ── Cluster health ───────────────────────────────────────────────────────
    health_results = raw.get("cluster_health", [])
    if health_results:
        f.cluster_health_score = int(_prom_float(health_results[0]))

    # ── Degraded PGs ─────────────────────────────────────────────────────────
    dpg_results = raw.get("degraded_pgs", [])
    if dpg_results:
        f.degraded_pgs = int(_prom_float(dpg_results[0]))

    # ── IO latency ───────────────────────────────────────────────────────────
    lat_results = raw.get("io_latency_ms", [])
    if lat_results:
        f.io_latency = _prom_float(lat_results[0]) * 0.001  # ms → s

    # ── PVC IOPS ─────────────────────────────────────────────────────────────
    iops_r = sum(_prom_float(r) for r in raw.get("pvc_iops_read",  []))
    iops_w = sum(_prom_float(r) for r in raw.get("pvc_iops_write", []))
    f.pvc_iops = iops_r + iops_w

    logger.debug(
        "Storage features  osd_up=%d/%d  pool=%.2f  health=%d  dpgs=%d",
        f.osd_up_count, f.osd_total_count,
        f.pool_usage_pct, f.cluster_health_score, f.degraded_pgs,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internal: log signal extraction
# ─────────────────────────────────────────────────────────────────────────────

def _fill_log_signals(f: ObsFeatures, logs: str) -> None:
    """
    Scan log text for error/warning counts and set log_anomaly_detected.

    An anomaly is flagged when:
      • recent_error_count >= 5, OR
      • recent_error_count >= 2 AND severity is "critical"
    """
    if not logs:
        return

    upper = logs.upper()
    f.recent_error_count   = upper.count("ERROR")
    f.recent_warning_count = upper.count("WARN")

    anomaly_threshold = 2 if f.severity == "critical" else 5
    f.log_anomaly_detected = f.recent_error_count >= anomaly_threshold

    logger.debug(
        "Log signals  errors=%d  warnings=%d  anomaly=%s",
        f.recent_error_count, f.recent_warning_count, f.log_anomaly_detected,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(value: Any, scale: float = 1.0) -> float:
    """
    Convert a metric string value to float, applying *scale*.

    Returns 0.0 for None, "no data", "parse error", and any non-numeric string.
    """
    if value is None:
        return 0.0
    try:
        return float(value) * scale
    except (ValueError, TypeError):
        return 0.0


def _prom_float(result: dict | Any) -> float:
    """
    Extract the numeric value from a single Prometheus instant result dict:
    {"metric": {...}, "value": [timestamp, "value_str"]}

    Returns 0.0 if the result is not in the expected format.
    """
    if not isinstance(result, dict):
        return 0.0
    try:
        return float(result["value"][1])
    except (KeyError, IndexError, ValueError, TypeError):
        return 0.0
