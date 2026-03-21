"""
obs_intelligence/cold_start_seeder.py
─────────────────────────────────────────────────────────────────────────────
ChromaDB cold-start seeder.

On day one the ``aiops_incidents`` ChromaDB collection is empty, so every local
LLM validation call returns ``insufficient_context`` — the local validator has
nothing to compare against.  This seeder pre-populates the collection with
synthetic past incidents derived from the known scenario catalogue so that
meaningful similarity search and corroboration is possible from the very first
real incident.

Design
──────
* **Idempotent** — checks the current entry count first; only seeds when the
  collection has fewer than ``CHROMA_SEED_MIN_ENTRIES`` entries (default 5) so
  it never overwrites existing knowledge.
* **10 diverse templates** covering both domains (compute + storage), a variety
  of scenario IDs, and both positive (resolved) and escalated outcomes.
* Each entry is stored via ``local_llm_enricher.store_incident_resolution()``,
  which embeds the text with ``nomic-embed-text`` and upserts into ChromaDB.
* Marked as ``validation_status="corroborated"`` with ``provider="synthetic_seed"``
  so the LLM can meaningfully use them as references on the first real call.
* Failures are logged as warnings and never raise — ChromaDB / Ollama being
  unavailable at startup should not block the engine from starting.

Called from ``obs-intelligence/app/main.py`` lifespan as an
``asyncio.create_task()`` so it runs in the background after startup.

Environment variables
─────────────────────
  CHROMA_SEED_MIN_ENTRIES   Skip seeding if collection already has this many
                            entries or more.  Default: 5.
  CHROMA_SEED_DELAY_S       Seconds to wait between individual embed calls to
                            avoid hammering the Ollama server.  Default: 0.5.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger("obs-intelligence.seeder")

SEED_MIN_ENTRIES: int = int(os.getenv("CHROMA_SEED_MIN_ENTRIES", "5"))
_SEED_DELAY_S: float = float(os.getenv("CHROMA_SEED_DELAY_S", "0.5"))

# ── Synthetic incident templates ──────────────────────────────────────────────
# One row per synthetic past incident.  Fields are used directly as metadata
# in the ChromaDB upsert and as the incident text that gets embedded.
_SEED_TEMPLATES: list[dict[str, Any]] = [
    {
        "scenario_id":  "high_cpu_saturation",
        "domain":       "compute",
        "service_name": "backend-api",
        "alert_name":   "HighCPUSaturation",
        "root_cause":   (
            "CPU saturation caused by an unbounded thread pool under high request "
            "volume. All available cores pegged above 90% utilisation for >5 minutes."
        ),
        "recommended_action": (
            "Scale horizontally by adding two replicas, throttle inbound request "
            "rate at the ingress layer, profile hot code paths with async-profiler."
        ),
        "outcome":               "resolved",
        "resolution_time_seconds": 420.0,
        "risk_score":            0.78,
    },
    {
        "scenario_id":  "CriticalErrorRate",
        "domain":       "compute",
        "service_name": "frontend-api",
        "alert_name":   "CriticalErrorRate",
        "root_cause":   (
            "HTTP 5xx error rate exceeded 10% SLO due to upstream database "
            "connection pool exhaustion.  Pool size was capped at 10 connections "
            "while request concurrency was 80."
        ),
        "recommended_action": (
            "Increase DB connection pool size to 50, add a circuit breaker on the "
            "upstream DB calls, rolling-restart unhealthy pods."
        ),
        "outcome":               "resolved",
        "resolution_time_seconds": 180.0,
        "risk_score":            0.85,
    },
    {
        "scenario_id":  "memory_leak_detected",
        "domain":       "compute",
        "service_name": "backend-api",
        "alert_name":   "HighMemoryUsage",
        "root_cause":   (
            "Memory leak in the async request handler retaining large JSON payload "
            "references across requests.  Heap grew from 512 MB to 4 GB over 2 hours."
        ),
        "recommended_action": (
            "Rolling restart of all affected pods.  Patch the async request handler "
            "to explicitly release payload references after processing."
        ),
        "outcome":               "resolved",
        "resolution_time_seconds": 300.0,
        "risk_score":            0.72,
    },
    {
        "scenario_id":  "latency_spike_p99",
        "domain":       "compute",
        "service_name": "frontend-api",
        "alert_name":   "HighLatencyP99",
        "root_cause":   (
            "P99 latency rose to 850ms due to an N+1 query pattern introduced in "
            "a recent deployment.  Per-request DB query count increased 40x."
        ),
        "recommended_action": (
            "Roll back the last deployment, add per-route DB query count logging, "
            "add integration test asserting query budgets."
        ),
        "outcome":               "resolved",
        "resolution_time_seconds": 240.0,
        "risk_score":            0.65,
    },
    {
        "scenario_id":  "restart_service",
        "domain":       "compute",
        "service_name": "backend-api",
        "alert_name":   "ServiceUnhealthy",
        "root_cause":   (
            "Service health check failures due to OOM kill by the Linux kernel. "
            "Container memory limit set to 256 MB was insufficient for peak load."
        ),
        "recommended_action": (
            "Restart service, increase container memory limit to 512 MB, add a "
            "memory-usage alert at 80% of the new limit."
        ),
        "outcome":               "auto_resolved",
        "resolution_time_seconds": 90.0,
        "risk_score":            0.60,
    },
    {
        "scenario_id":  "recurring_failure_signature",
        "domain":       "compute",
        "service_name": "backend-api",
        "alert_name":   "RecurringHighErrorRate",
        "root_cause":   (
            "Alert fired 4 times within 6 hours indicating a systemic fault not "
            "resolved by restarts.  Root cause: a cron job generates a traffic "
            "spike every 90 minutes that exceeds upstream capacity."
        ),
        "recommended_action": (
            "Human review required.  Escalate to on-call SRE.  Schedule the cron "
            "job during off-peak hours and add rate limiting on the job's egress."
        ),
        "outcome":               "escalated",
        "resolution_time_seconds": 3600.0,
        "risk_score":            0.88,
    },
    {
        "scenario_id":  "storage_pool_near_capacity",
        "domain":       "storage",
        "service_name": "ceph-cluster",
        "alert_name":   "StoragePoolNearCapacity",
        "root_cause":   (
            "Write-intensive batch job filled the primary pool to 87% utilisation "
            "overnight.  No automatic cold-data tiering was configured."
        ),
        "recommended_action": (
            "Move cold data to the archive tier, provision two additional OSDs, "
            "lower the capacity alert threshold to 75%."
        ),
        "outcome":               "resolved",
        "resolution_time_seconds": 900.0,
        "risk_score":            0.68,
    },
    {
        "scenario_id":  "osd_down_degraded_cluster",
        "domain":       "storage",
        "service_name": "ceph-cluster",
        "alert_name":   "OSDDownDegradedCluster",
        "root_cause":   (
            "Two OSDs went offline simultaneously due to a host network interface "
            "failure.  Cluster entered a degraded state with 12% unclean PGs."
        ),
        "recommended_action": (
            "Restore both OSD hosts, allow rebalancing to complete, verify PG count "
            "returns to fully-clean state before closing the incident."
        ),
        "outcome":               "resolved",
        "resolution_time_seconds": 1800.0,
        "risk_score":            0.91,
    },
    {
        "scenario_id":  "high_io_latency_storage",
        "domain":       "storage",
        "service_name": "ceph-cluster",
        "alert_name":   "HighIOLatency",
        "root_cause":   (
            "IO latency spiked to 350ms because a single OSD journal disk was "
            "serving 6 OSDs, saturating the device queue."
        ),
        "recommended_action": (
            "Migrate OSD journals to dedicated NVMe devices, enable BlueStore on "
            "the affected OSDs to eliminate the journal bottleneck entirely."
        ),
        "outcome":               "resolved",
        "resolution_time_seconds": 2400.0,
        "risk_score":            0.75,
    },
    {
        "scenario_id":  "degraded_pgs_high",
        "domain":       "storage",
        "service_name": "ceph-cluster",
        "alert_name":   "DegradedPGsHigh",
        "root_cause":   (
            "High number of degraded placement groups following a rolling OSD "
            "replacement during a maintenance window.  Rebalancing was still in "
            "progress when the alert fired."
        ),
        "recommended_action": (
            "Monitor rebalancing progress.  If not completing within 30 minutes, "
            "check OSD health and increase ``osd_recovery_op_priority``."
        ),
        "outcome":               "resolved",
        "resolution_time_seconds": 1200.0,
        "risk_score":            0.55,
    },
]


async def seed_chromadb_if_empty() -> None:
    """
    Seed the ``aiops_incidents`` ChromaDB collection with synthetic past incidents
    if the collection has fewer than ``SEED_MIN_ENTRIES`` entries.

    Safe to call on every startup — checks entry count before inserting.
    All errors are caught and logged; never raises.
    """
    from obs_intelligence.local_llm_enricher import (
        LocalValidationResult,
        local_llm_enricher,
    )

    try:
        stats = await local_llm_enricher.knowledge_stats()
        existing = int(stats.get("knowledge_entries_total", 0))

        if existing >= SEED_MIN_ENTRIES:
            logger.info(
                "ChromaDB cold-start seed skipped — collection already has %d entries "
                "(threshold=%d)",
                existing, SEED_MIN_ENTRIES,
            )
            return

        logger.info(
            "ChromaDB cold-start seeding started — %d existing entries < threshold=%d; "
            "seeding %d synthetic incidents",
            existing, SEED_MIN_ENTRIES, len(_SEED_TEMPLATES),
        )

        base_dt = datetime.now(timezone.utc) - timedelta(days=30)
        seeded = 0
        for i, tmpl in enumerate(_SEED_TEMPLATES):
            run_id = f"seed-{tmpl['scenario_id']}-{uuid.uuid4().hex[:8]}"
            incident_dt = base_dt + timedelta(days=i * 2, hours=i % 8)

            incident_context: dict[str, Any] = {
                "service_name":  tmpl["service_name"],
                "alert_name":    tmpl["alert_name"],
                "domain":        tmpl["domain"],
                "scenario_id":   tmpl["scenario_id"],
                "risk_score":    tmpl["risk_score"],
                "description": (
                    f"service={tmpl['service_name']} alert={tmpl['alert_name']} "
                    f"domain={tmpl['domain']} scenario={tmpl['scenario_id']} "
                    f"root_cause={tmpl['root_cause']} "
                    f"action={tmpl['recommended_action']}"
                ),
                "run_id":        run_id,
                "occurred_at":   incident_dt.isoformat(),
            }
            external_result = {
                "provider":           "synthetic_seed",
                "model":              "seed_v1",
                "confidence":         "high",
                "root_cause":         tmpl["root_cause"],
                "recommended_action": tmpl["recommended_action"],
            }
            # Mark seeds as corroborated so they carry positive similarity weight
            fake_validation = LocalValidationResult(
                validation_status="corroborated",
                confidence=0.85,
                reasoning_summary=f"Synthetic seed entry for {tmpl['scenario_id']}.",
                rca_alignment="aligned",
                action_alignment="aligned",
                suggested_adjustment="none",
                top_similarity=0.0,
                similar_count=0,
            )

            try:
                entry_id = await local_llm_enricher.store_incident_resolution(
                    incident_context=incident_context,
                    external_result=external_result,
                    local_validation=fake_validation,
                    similar=[],
                    outcome=tmpl["outcome"],
                    run_id=run_id,
                )
                if entry_id:
                    seeded += 1
                    logger.debug(
                        "Seeded  scenario=%s  run_id=%s",
                        tmpl["scenario_id"], run_id,
                    )
            except Exception as exc:
                logger.warning("Seed failed for %s: %s", tmpl["scenario_id"], exc)

            # Avoid hammering Ollama with embedding requests
            if _SEED_DELAY_S > 0:
                await asyncio.sleep(_SEED_DELAY_S)

        logger.info(
            "ChromaDB cold-start seed complete — %d/%d entries inserted",
            seeded, len(_SEED_TEMPLATES),
        )

    except Exception as exc:
        logger.warning("ChromaDB cold-start seeder error (non-fatal): %s", exc)
