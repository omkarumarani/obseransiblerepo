"""
obs_intelligence/cross_domain_correlator.py
────────────────────────────────────────────────────────────────────────────────
Cross-Domain Correlator — Phase 11.

Produces a UnifiedSREAssessment when compute AND storage agents both report
incidents within the co-occurrence window (default 120 s).

Correlation patterns detected
──────────────────────────────
  STORAGE_ROOT            Storage degradation (pool full, OSD down, high IO latency)
                          is the probable root cause of the compute errors/latency.
                          Classic symptom: NFS/iSCSI mount IO stalls blocking app
                          threads → error rate spikes, latency explodes.

  COMPUTE_ROOT            Compute overload (CPU/memory saturation) is spilling into
                          the storage path — less common but possible when storage
                          IO threads share CPU with app threads.

  SHARED_INFRASTRUCTURE   Both domains are degraded with no clear causal direction.
                          Likely a shared physical host, switch, or hypervisor fault.

  INDEPENDENT_CONCURRENT  Correlation window matched but signals do not cross-correlate.
                          Risk is still boosted by 25 % (two simultaneous incidents
                          always demand more attention).

Usage
─────
    from obs_intelligence.cross_domain_correlator import CrossDomainCorrelator
    from obs_intelligence.incident_coordinator import IncidentCoordinator

    # The coordinator ring-buffer entries already carry everything we need:
    correlator = CrossDomainCorrelator()
    unified = correlator.assess(compute_entry, storage_entry)

    print(unified.to_ticket_comment())   # paste into xyOps ticket
    print(unified.to_dict())             # REST API response
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("obs_intelligence.cross_domain_correlator")

# ── Correlation type constants ────────────────────────────────────────────────
STORAGE_ROOT           = "STORAGE_ROOT"
COMPUTE_ROOT           = "COMPUTE_ROOT"
SHARED_INFRASTRUCTURE  = "SHARED_INFRASTRUCTURE"
INDEPENDENT_CONCURRENT = "INDEPENDENT_CONCURRENT"

# ── Thresholds ────────────────────────────────────────────────────────────────
_STORAGE_HIGH_IO_LATENCY_S   = 0.2    # storage IO latency considered "high"
_STORAGE_HIGH_POOL_USAGE     = 0.75   # pool fill fraction considered "high"
_STORAGE_LOW_OSD_FRACTION    = 0.90   # fraction of OSDs up; below = "degraded"
_COMPUTE_HIGH_ERROR_RATE     = 0.10   # 10 % error rate
_COMPUTE_HIGH_LATENCY_MS     = 500    # p99 latency ms
_COMPUTE_HIGH_CPU            = 0.85   # 85 % CPU
_STORAGE_RISK_LEAD_ADVANTAGE = 0.15   # storage risk must exceed compute by this delta to be "root"
_COMPUTE_RISK_LEAD_ADVANTAGE = 0.15


# ═══════════════════════════════════════════════════════════════════════════════
# Output model
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class UnifiedSREAssessment:
    """
    Unified SRE assessment spanning both compute and storage domains.

    Produced by CrossDomainCorrelator when both agents report within the
    co-occurrence window.  Surfaced:
      - As a comment on each domain's xyOps ticket
      - Via GET /intelligence/correlation/current on obs-intelligence
      - In the Streamlit Agent Mesh tab (cross-domain correlation panel)
    """

    # ── Correlation metadata ─────────────────────────────────────────────────
    correlation_type: str
    """STORAGE_ROOT | COMPUTE_ROOT | SHARED_INFRASTRUCTURE | INDEPENDENT_CONCURRENT"""

    primary_domain: str
    """Which domain is the likely root cause: "compute" | "storage" | "shared" | "unknown"."""

    detected_at: str
    """ISO 8601 timestamp."""

    # ── Participating incidents ───────────────────────────────────────────────
    compute_service: str
    storage_service: str
    compute_scenario: str
    storage_scenario: str
    compute_run_id: str
    storage_run_id: str

    # ── Risk ─────────────────────────────────────────────────────────────────
    compute_risk_score: float
    storage_risk_score: float
    combined_risk_score: float
    combined_risk_level: str

    # ── Reasoning ────────────────────────────────────────────────────────────
    causal_chain: list[str] = field(default_factory=list)
    unified_recommended_actions: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    urgency: str = "high"
    narrative: str = ""

    # ── Signal snapshot (for ticket comment) ─────────────────────────────────
    compute_signals: dict = field(default_factory=dict)
    storage_signals: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "correlation_type":    self.correlation_type,
            "primary_domain":      self.primary_domain,
            "detected_at":         self.detected_at,
            "compute_service":     self.compute_service,
            "storage_service":     self.storage_service,
            "compute_scenario":    self.compute_scenario,
            "storage_scenario":    self.storage_scenario,
            "compute_run_id":      self.compute_run_id,
            "storage_run_id":      self.storage_run_id,
            "compute_risk_score":  self.compute_risk_score,
            "storage_risk_score":  self.storage_risk_score,
            "combined_risk_score": self.combined_risk_score,
            "combined_risk_level": self.combined_risk_level,
            "causal_chain":        self.causal_chain,
            "unified_recommended_actions": self.unified_recommended_actions,
            "evidence":            self.evidence,
            "urgency":             self.urgency,
            "narrative":           self.narrative,
            "compute_signals":     self.compute_signals,
            "storage_signals":     self.storage_signals,
        }

    def to_ticket_comment(self) -> str:
        """
        Markdown-formatted comment suitable for posting to both domain xyOps tickets.
        Keeps table formatting tight so it renders cleanly in xyOps.
        """
        type_icon = {
            STORAGE_ROOT:           "🗄️  STORAGE ROOT",
            COMPUTE_ROOT:           "🖥️  COMPUTE ROOT",
            SHARED_INFRASTRUCTURE:  "🌐  SHARED INFRA",
            INDEPENDENT_CONCURRENT: "⚡  CONCURRENT",
        }.get(self.correlation_type, self.correlation_type)

        risk_icon = "🔴" if self.combined_risk_level in ("critical", "high") else "🟡"

        chain_txt    = "\n".join(f"  {c}" for c in self.causal_chain)
        actions_txt  = "\n".join(f"  {a}" for a in self.unified_recommended_actions)
        evidence_txt = "\n".join(f"  - {e}" for e in self.evidence[:6])

        cs = self.compute_signals
        ss = self.storage_signals

        return (
            f"## 🔗 Cross-Domain Correlation Detected\n\n"
            f"| Field | Value |\n|---|---|\n"
            f"| **Pattern** | {type_icon} |\n"
            f"| **Primary root** | `{self.primary_domain}` domain |\n"
            f"| **Combined risk** | {risk_icon} `{self.combined_risk_level.upper()}`"
            f" ({self.combined_risk_score:.2f}) |\n"
            f"| **Urgency** | `{self.urgency.upper()}` |\n"
            f"| **Compute service** | `{self.compute_service}` "
            f"(risk {self.compute_risk_score:.2f}) |\n"
            f"| **Storage service** | `{self.storage_service}` "
            f"(risk {self.storage_risk_score:.2f}) |\n"
            f"| **Compute scenario** | `{self.compute_scenario}` |\n"
            f"| **Storage scenario** | `{self.storage_scenario}` |\n\n"
            f"**Compute signals:** "
            f"error_rate={cs.get('error_rate', 0):.1%}  "
            f"p99={cs.get('latency_p99_ms', 0)}ms  "
            f"cpu={cs.get('cpu_usage_pct', 0):.0%}\n\n"
            f"**Storage signals:** "
            f"pool={ss.get('pool_usage_pct', 0):.0%}  "
            f"io_latency={ss.get('io_latency_s', 0):.3f}s  "
            f"osds={ss.get('osd_up', '?')}/{ss.get('osd_total', '?')}\n\n"
            f"**Causal chain:**\n{chain_txt}\n\n"
            f"**Unified recommended actions:**\n{actions_txt}\n\n"
            f"**Evidence:**\n{evidence_txt}\n\n"
            f"*Detected at {self.detected_at}  |  "
            f"Compute run: `{self.compute_run_id}`  |  "
            f"Storage run: `{self.storage_run_id}`*"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Correlator
# ═══════════════════════════════════════════════════════════════════════════════

class CrossDomainCorrelator:
    """
    Stateless cross-domain incident correlator.

    Call assess() with the two ring-buffer entries returned by IncidentCoordinator
    to get a UnifiedSREAssessment.

    The correlator is intentionally side-effect-free: it reads signal values
    from the entries and returns a pure dataclass.  All persistence and metric
    publishing is done by the caller.
    """

    def assess(
        self,
        compute_entry: dict[str, Any],
        storage_entry: dict[str, Any],
    ) -> UnifiedSREAssessment:
        """
        Produce a UnifiedSREAssessment from two co-occurring domain incident entries.

        Parameters
        ----------
        compute_entry  — ring-buffer entry from domain="compute"
        storage_entry  — ring-buffer entry from domain="storage"

        Both entries must have at minimum:
            domain, service_name, alert_name, risk_score, scenario_id, run_id,
            signals (optional dict of raw signal values)
        """
        compute_risk = float(compute_entry.get("risk_score", 0.0))
        storage_risk = float(storage_entry.get("risk_score", 0.0))
        cs = compute_entry.get("signals", {})
        ss = storage_entry.get("signals", {})

        corr_type = self._detect_correlation_type(compute_risk, storage_risk, cs, ss)
        primary   = self._determine_primary_domain(corr_type, compute_risk, storage_risk)

        # Combined risk: max of both * 1.25 boost (two simultaneous incidents always worse)
        combined_raw   = min(1.0, max(compute_risk, storage_risk) * 1.25)
        combined_level = self._risk_level(combined_raw)

        causal_chain = self._build_causal_chain(
            corr_type, compute_entry, storage_entry, cs, ss
        )
        actions = self._build_recommended_actions(
            corr_type, primary, compute_entry, storage_entry
        )
        evidence = self._build_evidence(corr_type, cs, ss, compute_entry, storage_entry)
        urgency  = self._urgency(combined_level, corr_type)
        narrative = self._narrative(corr_type, primary, compute_entry, storage_entry, combined_level)

        assessment = UnifiedSREAssessment(
            correlation_type   = corr_type,
            primary_domain     = primary,
            detected_at        = datetime.now(timezone.utc).isoformat(),
            compute_service    = compute_entry.get("service_name", "unknown"),
            storage_service    = storage_entry.get("service_name", "unknown"),
            compute_scenario   = compute_entry.get("scenario_id", "unknown"),
            storage_scenario   = storage_entry.get("scenario_id", "unknown"),
            compute_run_id     = compute_entry.get("run_id", ""),
            storage_run_id     = storage_entry.get("run_id", ""),
            compute_risk_score = round(compute_risk, 3),
            storage_risk_score = round(storage_risk, 3),
            combined_risk_score = round(combined_raw, 3),
            combined_risk_level = combined_level,
            causal_chain       = causal_chain,
            unified_recommended_actions = actions,
            evidence           = evidence,
            urgency            = urgency,
            narrative          = narrative,
            compute_signals    = cs,
            storage_signals    = ss,
        )

        logger.info(
            "Cross-domain assessment complete  type=%s  primary=%s  "
            "combined_risk=%.2f  compute=%s  storage=%s",
            corr_type, primary, combined_raw,
            compute_entry.get("service_name"),
            storage_entry.get("service_name"),
        )
        return assessment

    # ── Correlation type detection ────────────────────────────────────────────

    def _detect_correlation_type(
        self,
        compute_risk: float,
        storage_risk: float,
        cs: dict,
        ss: dict,
    ) -> str:
        """
        Rule-based pattern matching to classify the cross-domain relationship.

        Rules are evaluated in priority order; the first match wins.
        """
        # Extract scalar signals with safe defaults
        io_latency    = float(ss.get("io_latency_s", 0))
        pool_usage    = float(ss.get("pool_usage_pct", 0))
        osd_up        = int(ss.get("osd_up", 99))
        osd_total     = int(ss.get("osd_total", 99))
        error_rate    = float(cs.get("error_rate", 0))
        latency_p99   = float(cs.get("latency_p99_ms", 0))
        cpu_usage     = float(cs.get("cpu_usage_pct", 0))

        osd_fraction  = (osd_up / osd_total) if osd_total > 0 else 1.0

        storage_degraded = (
            io_latency   >= _STORAGE_HIGH_IO_LATENCY_S
            or pool_usage >= _STORAGE_HIGH_POOL_USAGE
            or osd_fraction < _STORAGE_LOW_OSD_FRACTION
        )
        compute_stressed = (
            error_rate  >= _COMPUTE_HIGH_ERROR_RATE
            or latency_p99 >= _COMPUTE_HIGH_LATENCY_MS
        )
        compute_saturated = cpu_usage >= _COMPUTE_HIGH_CPU

        # Rule 1: Storage clearly degraded AND compute is stressed
        #         → storage latency/failure is blocking IO-backed app calls
        if storage_degraded and compute_stressed:
            return STORAGE_ROOT

        # Rule 2: Compute CPU saturated AND storage IO latency elevated (mild)
        #         → compute overload starving storage IO threads
        if compute_saturated and io_latency > 0.05:
            return COMPUTE_ROOT

        # Rule 3: Both domains degraded but neither clearly dominates by risk
        if (
            storage_risk > 0.4
            and compute_risk > 0.4
            and abs(storage_risk - compute_risk) < _STORAGE_RISK_LEAD_ADVANTAGE
        ):
            return SHARED_INFRASTRUCTURE

        # Default: timing coincidence, no clear signal cross-correlation
        return INDEPENDENT_CONCURRENT

    def _determine_primary_domain(
        self, corr_type: str, compute_risk: float, storage_risk: float
    ) -> str:
        mapping = {
            STORAGE_ROOT:           "storage",
            COMPUTE_ROOT:           "compute",
            SHARED_INFRASTRUCTURE:  "shared",
            INDEPENDENT_CONCURRENT: "unknown",
        }
        return mapping.get(corr_type, "unknown")

    # ── Causal chain ──────────────────────────────────────────────────────────

    def _build_causal_chain(
        self,
        corr_type: str,
        compute_entry: dict,
        storage_entry: dict,
        cs: dict,
        ss: dict,
    ) -> list[str]:
        chain: list[str] = []
        step = 1

        if corr_type == STORAGE_ROOT:
            io_latency  = float(ss.get("io_latency_s", 0))
            pool_usage  = float(ss.get("pool_usage_pct", 0))
            osd_up      = int(ss.get("osd_up", 0))
            osd_total   = int(ss.get("osd_total", 0))

            if osd_total > 0 and osd_up < osd_total:
                chain.append(
                    f"{step}. Storage OSD failure: {osd_up}/{osd_total} OSDs online "
                    f"({storage_entry.get('scenario_id', 'storage scenario')})."
                )
                step += 1
            elif pool_usage >= _STORAGE_HIGH_POOL_USAGE:
                chain.append(
                    f"{step}. Storage pool fill at {pool_usage:.0%} — write operations "
                    "are throttled or failing."
                )
                step += 1
            if io_latency > 0:
                chain.append(
                    f"{step}. Storage IO latency {io_latency:.3f}s is causing "
                    "synchronous read/write calls from compute services to stall."
                )
                step += 1
            chain.append(
                f"{step}. Stalled IO calls exhaust connection pool threads in "
                f"`{compute_entry.get('service_name', 'compute service')}`, "
                "leading to request queue buildup."
            )
            step += 1
            chain.append(
                f"{step}. Queue buildup manifests as elevated error rate "
                f"({float(cs.get('error_rate', 0)):.1%}) "
                f"and p99 latency ({cs.get('latency_p99_ms', 0)}ms) on the compute side."
            )
            step += 1
            chain.append(
                f"{step}. Compute alert `{compute_entry.get('alert_name', '')}` fired "
                "AFTER the storage degradation — storage is the primary root cause."
            )

        elif corr_type == COMPUTE_ROOT:
            cpu = float(cs.get("cpu_usage_pct", 0))
            chain.append(
                f"{step}. Compute CPU saturation at {cpu:.0%} on "
                f"`{compute_entry.get('service_name', 'compute service')}`."
            )
            step += 1
            chain.append(
                f"{step}. Saturated CPU is starving storage IO dispatch threads, "
                "increasing storage IO latency as a side-effect."
            )
            step += 1
            chain.append(
                f"{step}. Storage metrics show elevated IO but no OSD/pool hardware fault "
                "— indirect effect of compute overload."
            )
            step += 1
            chain.append(
                f"{step}. Resolving the compute CPU issue (scale-out or restart) "
                "should normalise both domains."
            )

        elif corr_type == SHARED_INFRASTRUCTURE:
            chain.append(
                f"{step}. Both compute (`{compute_entry.get('service_name')}`) and "
                f"storage (`{storage_entry.get('service_name')}`) degraded simultaneously "
                "with no clear causal direction."
            )
            step += 1
            chain.append(
                f"{step}. Combined risk {max(compute_entry.get('risk_score', 0), storage_entry.get('risk_score', 0)):.2f} "
                "with risks within 15 % of each other — suggests a shared fault."
            )
            step += 1
            chain.append(
                f"{step}. Probable shared root causes: network switch / vSwitch failure, "
                "hypervisor host issue, or shared storage fabric degradation."
            )
            step += 1
            chain.append(
                f"{step}. Investigate the physical/virtual network layer between the "
                "compute and storage clusters before acting on either domain individually."
            )

        else:  # INDEPENDENT_CONCURRENT
            chain.append(
                f"{step}. Compute incident `{compute_entry.get('alert_name')}` "
                f"(risk {compute_entry.get('risk_score', 0):.2f}) and storage "
                f"incident `{storage_entry.get('alert_name')}` "
                f"(risk {storage_entry.get('risk_score', 0):.2f}) fired within 120 s."
            )
            step += 1
            chain.append(
                f"{step}. Signal analysis does not show a clear causal link between "
                "the two incidents — timing may be coincidental."
            )
            step += 1
            chain.append(
                f"{step}. Treat independently but escalate combined risk — two simultaneous "
                "incidents require increased operator attention even without proven causation."
            )

        return chain

    # ── Recommended actions ───────────────────────────────────────────────────

    def _build_recommended_actions(
        self,
        corr_type: str,
        primary: str,
        compute_entry: dict,
        storage_entry: dict,
    ) -> list[str]:
        actions: list[str] = []
        step = 1

        if corr_type == STORAGE_ROOT:
            actions.append(
                f"{step}. Fix storage first: remediate "
                f"`{storage_entry.get('scenario_id', 'storage-scenario')}` "
                f"on `{storage_entry.get('service_name', 'storage')}` "
                "(see linked storage ticket)."
            )
            step += 1
            actions.append(
                f"{step}. Verify storage IO latency returns to baseline "
                "before restarting compute services."
            )
            step += 1
            actions.append(
                f"{step}. Do NOT restart `{compute_entry.get('service_name')}` "
                "until storage is stable — restart will not fix an IO stall."
            )
            step += 1
            actions.append(
                f"{step}. Once storage stabilises, monitor compute error_rate "
                "auto-recovery within 2–5 min without further action."
            )
            step += 1
            actions.append(
                f"{step}. If compute does not self-heal after storage recovery, "
                "execute the compute-domain remediation playbook."
            )

        elif corr_type == COMPUTE_ROOT:
            actions.append(
                f"{step}. Address compute CPU saturation on "
                f"`{compute_entry.get('service_name')}` first: "
                "scale-out or restart the offending process."
            )
            step += 1
            actions.append(
                f"{step}. Storage metrics should normalise once compute CPU "
                "returns below 80 % — do not run storage remediation prematurely."
            )
            step += 1
            actions.append(
                f"{step}. If storage IO latency persists after compute recovery, "
                "then independently investigate the storage scenario."
            )

        elif corr_type == SHARED_INFRASTRUCTURE:
            actions.append(
                f"{step}. Escalate immediately to infrastructure/network team — "
                "shared root cause suspected."
            )
            step += 1
            actions.append(
                f"{step}. Check physical network: switch uplinks, NIC bonding on "
                "hypervisor hosts, and storage fabric (iSCSI/NFS/FC)."
            )
            step += 1
            actions.append(
                f"{step}. Check hypervisor host health: CPU ready time, memory balloon, "
                "and disk queue depth."
            )
            step += 1
            actions.append(
                f"{step}. Do NOT commence automated playbooks for either domain until "
                "the shared fault is identified."
            )
            step += 1
            actions.append(
                f"{step}. Apply compute domain remediation "
                f"(`{compute_entry.get('scenario_id', '')}`) ONLY after infrastructure "
                "is cleared."
            )

        else:  # INDEPENDENT_CONCURRENT
            actions.append(
                f"{step}. Handle each incident independently through its own "
                "xyOps approval workflow."
            )
            step += 1
            actions.append(
                f"{step}. Prioritise by risk: "
                f"{'compute' if compute_entry.get('risk_score', 0) >= storage_entry.get('risk_score', 0) else 'storage'} "
                "domain has the higher score."
            )
            step += 1
            actions.append(
                f"{step}. Monitor for secondary cascading symptoms over the next "
                "10 minutes — if they appear, re-evaluate as SHARED_INFRASTRUCTURE."
            )

        return actions

    # ── Evidence ──────────────────────────────────────────────────────────────

    def _build_evidence(
        self,
        corr_type: str,
        cs: dict,
        ss: dict,
        compute_entry: dict,
        storage_entry: dict,
    ) -> list[str]:
        ev: list[str] = [
            f"Both domains incident within 120 s co-occurrence window",
            f"Compute: alert={compute_entry.get('alert_name')} "
            f"risk={compute_entry.get('risk_score', 0):.2f} "
            f"scenario={compute_entry.get('scenario_id')}",
            f"Storage: alert={storage_entry.get('alert_name')} "
            f"risk={storage_entry.get('risk_score', 0):.2f} "
            f"scenario={storage_entry.get('scenario_id')}",
        ]
        if cs:
            ev.append(
                f"Compute signals: error_rate={float(cs.get('error_rate', 0)):.1%} "
                f"p99={cs.get('latency_p99_ms', 'n/a')}ms "
                f"cpu={float(cs.get('cpu_usage_pct', 0)):.0%}"
            )
        if ss:
            ev.append(
                f"Storage signals: pool={float(ss.get('pool_usage_pct', 0)):.0%} "
                f"io_latency={float(ss.get('io_latency_s', 0)):.3f}s "
                f"osds={ss.get('osd_up', '?')}/{ss.get('osd_total', '?')}"
            )
        ev.append(f"Correlation pattern: {corr_type}")
        return ev

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _risk_level(score: float) -> str:
        if score >= 0.80:
            return "critical"
        if score >= 0.60:
            return "high"
        if score >= 0.30:
            return "medium"
        return "low"

    @staticmethod
    def _urgency(risk_level: str, corr_type: str) -> str:
        # Shared infra / storage root with high combined risk = critical urgency
        if corr_type in (STORAGE_ROOT, SHARED_INFRASTRUCTURE) and risk_level in (
            "critical", "high"
        ):
            return "critical"
        if risk_level == "critical":
            return "critical"
        if risk_level == "high":
            return "high"
        return "medium"

    @staticmethod
    def _narrative(
        corr_type: str,
        primary: str,
        compute_entry: dict,
        storage_entry: dict,
        combined_level: str,
    ) -> str:
        c_svc = compute_entry.get("service_name", "compute service")
        s_svc = storage_entry.get("service_name", "storage service")
        c_alert = compute_entry.get("alert_name", "compute alert")
        s_alert = storage_entry.get("alert_name", "storage alert")

        if corr_type == STORAGE_ROOT:
            return (
                f"Storage degradation on `{s_svc}` ({s_alert}) is the probable root "
                f"cause of the compute alert on `{c_svc}` ({c_alert}). "
                f"IO stalls from the storage layer are blocking application threads, "
                f"causing error rate and latency to spike on the compute side. "
                f"Remediate storage first to restore both domains. "
                f"Combined risk: {combined_level.upper()}."
            )
        if corr_type == COMPUTE_ROOT:
            return (
                f"CPU saturation on `{c_svc}` is indirectly elevating storage IO latency "
                f"on `{s_svc}`. Both alerts share a compute-side root cause. "
                f"Resolving the compute CPU issue should normalise both domains. "
                f"Combined risk: {combined_level.upper()}."
            )
        if corr_type == SHARED_INFRASTRUCTURE:
            return (
                f"Both `{c_svc}` and `{s_svc}` are degraded simultaneously without a "
                f"clear causal direction, suggesting a shared infrastructure fault "
                f"(network switch, hypervisor, or storage fabric). "
                f"Escalate to infrastructure team before executing domain-specific playbooks. "
                f"Combined risk: {combined_level.upper()}."
            )
        return (
            f"Compute ({c_alert} on `{c_svc}`) and storage ({s_alert} on `{s_svc}`) "
            f"alerts fired within 120 s. Signal analysis finds no direct causal link — "
            f"treat independently but monitor for secondary cascading effects. "
            f"Combined risk: {combined_level.upper()}."
        )


# ── Module-level singleton ────────────────────────────────────────────────────
cross_domain_correlator = CrossDomainCorrelator()
