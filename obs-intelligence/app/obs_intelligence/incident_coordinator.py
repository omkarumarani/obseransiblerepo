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
    )
    if event:
        # Cross-domain correlation detected — attach to ticket
        print(event["message"])
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any

COOCCURRENCE_WINDOW_SECONDS: int = 120   # two incidents within 2 min = correlated
_PRUNE_WINDOW_SECONDS: int = 600         # discard entries older than 10 minutes
_RING_BUFFER_MAX: int = 50

_recent_incidents: list[dict] = []
_lock = threading.Lock()


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
    ) -> dict[str, Any] | None:
        """
        Add *domain* incident to the ring buffer.

        Returns a CrossDomainEvent dict if another domain reported an incident
        within ``COOCCURRENCE_WINDOW_SECONDS``, otherwise returns ``None``.

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

        # Build CrossDomainEvent
        entry_a = correlated
        entry_b = new_entry
        combined_risk = min(1.0, max(entry_a["risk_score"], entry_b["risk_score"]) * 1.25)

        return {
            "event_type": "cross_domain_correlation",
            "domains":    [entry_a["domain"], entry_b["domain"]],
            "services":   [entry_a["service_name"], entry_b["service_name"]],
            "scenarios":  [entry_a["scenario_id"], entry_b["scenario_id"]],
            "combined_risk_score": round(combined_risk, 3),
            "message": (
                "Simultaneous compute+storage incident detected. "
                "Likely shared root cause (network, NFS mount, "
                "or storage backend degradation)."
            ),
            "detected_at": datetime.now(timezone.utc).isoformat(),
            # Extra context for ticket comments
            "alert_a":  entry_a["alert_name"],
            "alert_b":  entry_b["alert_name"],
            "run_id_a": entry_a["run_id"],
            "run_id_b": entry_b["run_id"],
        }
