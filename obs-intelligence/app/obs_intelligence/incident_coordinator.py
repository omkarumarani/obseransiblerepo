"""
obs_intelligence/incident_coordinator.py
────────────────────────────────────────────────────────────────────────────────
Cross-domain incident coordinator.

When both compute-agent and storage-agent report incidents within a short
time window, this module detects the co-occurrence and returns a composite
CrossDomainEvent that callers can attach to their incident tickets.

Design
──────
- In-memory ring buffer (max 50 entries) — intentionally lightweight.
  No persistence needed: if the container restarts mid-window the worst
  outcome is a missed correlation, not data corruption.
- Thread-safe via a simple lock (FastAPI runs handlers in asyncio but
  background tasks may call from threads).
- Best-effort: callers fire-and-forget; exceptions are silenced upstream.
- Stores the last UnifiedSREAssessment (from CrossDomainCorrelator) so
  the Streamlit dashboard can poll it via GET /intelligence/correlation/current.

Usage
─────
    from obs_intelligence.incident_coordinator import IncidentCoordinator
    coordinator = IncidentCoordinator()

    event = coordinator.record_incident(
        domain="compute",
        service_name="frontend-api",
        alert_name="HighErrorRate",
        risk_score=0.72,
        scenario_id="high_error_rate",
        run_id="run-abc123",
        signals={"error_rate": 0.15, "latency_p99_ms": 620, "cpu_usage_pct": 0.71},
    )
    if event:
        # Cross-domain correlation detected — attach to ticket
        print(event["message"])
        # event["_compute_entry"] and event["_storage_entry"] carry full ring-buffer
        # entries for CrossDomainCorrelator.assess()
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any

COOCCURRENCE_WINDOW_SECONDS: int = 120   # two incidents within 2 min = correlated
_PRUNE_WINDOW_SECONDS: int = 600         # discard entries older than 10 minutes
_RING_BUFFER_MAX: int = 50
_UNIFIED_ASSESSMENT_TTL: int = 600       # keep last unified assessment for 10 min

_recent_incidents: list[dict] = []
_lock = threading.Lock()

# Last unified SREAssessment stored by store_unified_assessment()
_last_unified_assessment: dict[str, Any] | None = None
_last_unified_at: float = 0.0


class IncidentCoordinator:
    """Detect cross-domain incident co-occurrence in a sliding time window."""

    def record_incident(
        self,
        domain: str,
        service_name: str,
        alert_name: str,
        risk_score: float,
        scenario_id: str,
        run_id: str,
        signals: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """
        Add *domain* incident to the ring buffer.

        Returns a CrossDomainEvent dict if another domain reported an incident
        within ``COOCCURRENCE_WINDOW_SECONDS``, otherwise returns ``None``.

        The returned dict contains ``_compute_entry`` and ``_storage_entry`` keys
        with the full ring-buffer payloads (including signals) so that the caller
        can pass them directly to CrossDomainCorrelator.assess().

        Side-effects:
          - Prunes ring buffer entries older than 10 minutes.
          - Caps ring buffer at ``_RING_BUFFER_MAX`` entries (oldest dropped).
        """
        now = time.time()
        new_entry: dict[str, Any] = {
            "domain":       domain,
            "service_name": service_name,
            "alert_name":   alert_name,
            "risk_score":   risk_score,
            "scenario_id":  scenario_id,
            "run_id":       run_id,
            "signals":      signals or {},
            "recorded_at":  now,
        }

        with _lock:
            # Prune stale entries (> 10 min)
            cutoff_prune = now - _PRUNE_WINDOW_SECONDS
            _recent_incidents[:] = [
                e for e in _recent_incidents if e["recorded_at"] > cutoff_prune
            ]

            # Find a co-occurrence: same window, different domain
            cutoff_window = now - COOCCURRENCE_WINDOW_SECONDS
            correlated = next(
                (
                    e for e in _recent_incidents
                    if e["recorded_at"] > cutoff_window and e["domain"] != domain
                ),
                None,
            )

            # Append new entry and enforce ring-buffer cap
            _recent_incidents.append(new_entry)
            if len(_recent_incidents) > _RING_BUFFER_MAX:
                _recent_incidents.pop(0)

        if correlated is None:
            return None

        # Build CrossDomainEvent — identify compute vs storage entries
        if domain == "compute":
            compute_entry = new_entry
            storage_entry = correlated
        else:
            compute_entry = correlated
            storage_entry = new_entry

        combined_risk = min(1.0, max(compute_entry["risk_score"], storage_entry["risk_score"]) * 1.25)

        return {
            "event_type": "cross_domain_correlation",
            "domains":    [compute_entry["domain"], storage_entry["domain"]],
            "services":   [compute_entry["service_name"], storage_entry["service_name"]],
            "scenarios":  [compute_entry["scenario_id"], storage_entry["scenario_id"]],
            "combined_risk_score": round(combined_risk, 3),
            "message": (
                "Simultaneous compute+storage incident detected. "
                "Likely shared root cause (network, NFS mount, "
                "or storage backend degradation)."
            ),
            "detected_at": datetime.now(timezone.utc).isoformat(),
            # Extra context for ticket comments
            "alert_a":  compute_entry["alert_name"],
            "alert_b":  storage_entry["alert_name"],
            "run_id_a": compute_entry["run_id"],
            "run_id_b": storage_entry["run_id"],
            # Full entries for CrossDomainCorrelator (prefixed _ to signal internal use)
            "_compute_entry": compute_entry,
            "_storage_entry": storage_entry,
        }

    # ── Unified assessment storage ────────────────────────────────────────────

    def store_unified_assessment(self, assessment_dict: dict[str, Any]) -> None:
        """
        Store the most recent UnifiedSREAssessment produced by CrossDomainCorrelator.

        Called by the /intelligence/record-incident endpoint after it runs the
        correlator.  The stored value is returned by get_active_correlation().
        """
        global _last_unified_assessment, _last_unified_at
        with _lock:
            _last_unified_assessment = assessment_dict
            _last_unified_at = time.time()

    def get_active_correlation(self) -> dict[str, Any] | None:
        """
        Return the last stored UnifiedSREAssessment if it is within the TTL window.

        Returns None if no correlation has been detected recently.
        """
        with _lock:
            if _last_unified_assessment is None:
                return None
            if time.time() - _last_unified_at > _UNIFIED_ASSESSMENT_TTL:
                return None
            return _last_unified_assessment
