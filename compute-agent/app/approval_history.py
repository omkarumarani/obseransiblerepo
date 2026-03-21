"""
compute-agent/app/approval_history.py
────────────────────────────────────────────────────────────────────────────────
Persistent approval history store.

Records every human decision (approve / decline) and each autonomous execution
outcome so the Autonomy Engine can compute trust scores over time.

Storage: JSON-line file written to APPROVAL_HISTORY_PATH (default
/data/approval_history.jsonl).  Mount a persistent Docker volume at /data
to survive container restarts; falls back to /tmp if the path is not writable.

Schema (one JSON object per line):
  {
    "record_id":          str,      # UUID
    "approval_id":        str,
    "service_name":       str,
    "alert_name":         str,
    "action_type":        str,      # e.g. "restart_service", "scale_workers"
    "env_tier":           str,      # "production" | "staging" | "development" | "sandbox"
    "decided_by":         str,      # username or "autonomous"
    "decision":           str,      # "approved" | "declined" | "autonomous"
    "execution_outcome":  str,      # "success" | "failure" | "skipped" | "pending"
    "risk_score":         float,
    "decided_at":         str,      # ISO-8601 UTC
    "executed_at":        str,      # ISO-8601 UTC or ""
    "notes":              str
  }

Thread safety: append-only writes + in-memory cache built on load — safe for
single-process FastAPI (asyncio event loop, no concurrent writes).
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Sequence

logger = logging.getLogger("aiops-bridge.approval_history")

# ── Persistence path ───────────────────────────────────────────────────────────
_DEFAULT_PATH = "/data/approval_history.jsonl"
_FALLBACK_PATH = "/tmp/approval_history.jsonl"


def _resolve_path() -> str:
    """Return the first writable path for history storage."""
    configured = os.getenv("APPROVAL_HISTORY_PATH", _DEFAULT_PATH)
    parent = os.path.dirname(configured)
    try:
        os.makedirs(parent, exist_ok=True)
        # test write
        test_file = os.path.join(parent, ".write_test")
        with open(test_file, "w") as f:
            f.write("")
        os.unlink(test_file)
        return configured
    except OSError:
        logger.warning(
            "Cannot write to %s — falling back to %s", configured, _FALLBACK_PATH
        )
        return _FALLBACK_PATH


# ═══════════════════════════════════════════════════════════════════════════════
# Data model
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ApprovalRecord:
    record_id: str
    approval_id: str
    service_name: str
    alert_name: str
    action_type: str
    env_tier: str
    decided_by: str
    decision: str              # "approved" | "declined" | "autonomous"
    execution_outcome: str     # "success" | "failure" | "skipped" | "pending"
    risk_score: float
    decided_at: str
    executed_at: str = ""
    notes: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "ApprovalRecord":
        return cls(
            record_id=d.get("record_id", str(uuid.uuid4())),
            approval_id=d.get("approval_id", ""),
            service_name=d.get("service_name", ""),
            alert_name=d.get("alert_name", ""),
            action_type=d.get("action_type", ""),
            env_tier=d.get("env_tier", "production"),
            decided_by=d.get("decided_by", ""),
            decision=d.get("decision", ""),
            execution_outcome=d.get("execution_outcome", "pending"),
            risk_score=d.get("risk_score", 0.0),
            decided_at=d.get("decided_at", ""),
            executed_at=d.get("executed_at", ""),
            notes=d.get("notes", ""),
        )


@dataclass
class TrustScore:
    """Summarised trust data for a (service, action_type) pair."""
    service_name: str
    action_type: str
    env_tier: str
    total_decisions: int
    approved_count: int
    declined_count: int
    autonomous_count: int
    success_count: int        # approved/autonomous + outcome success
    failure_count: int
    success_rate: float       # success / (success + failure), 0.0 if no executions
    window_days: int
    autonomy_eligible: bool
    reason: str               # human-readable explanation


# ═══════════════════════════════════════════════════════════════════════════════
# Store
# ═══════════════════════════════════════════════════════════════════════════════

class ApprovalHistoryStore:
    """
    Append-only JSONL-backed store for approval decisions and outcomes.

    Usage (singleton, instantiated once at module level):
      from .approval_history import history_store

      # record a human decision
      history_store.record_decision(...)

      # update execution outcome after ansible run
      history_store.update_outcome(approval_id, "success")

      # query trust
      trust = history_store.compute_trust_score(service, action, tier)
    """

    def __init__(self) -> None:
        self._path = _resolve_path()
        self._records: list[ApprovalRecord] = []
        self._loaded = False

    # ── Load ──────────────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._records = []
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self._records.append(ApprovalRecord.from_dict(json.loads(line)))
                        except (json.JSONDecodeError, TypeError):
                            pass  # skip malformed lines
            logger.info(
                "Loaded %d approval history records from %s",
                len(self._records), self._path,
            )
        except FileNotFoundError:
            logger.info("No approval history file yet — starting fresh")
        self._loaded = True

    # ── Write ──────────────────────────────────────────────────────────────────

    def _append(self, record: ApprovalRecord) -> None:
        """Append a single record to the JSONL file and in-memory list."""
        line = json.dumps(asdict(record)) + "\n"
        try:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line)
        except OSError as exc:
            logger.warning("Could not persist approval record: %s", exc)
        self._records.append(record)

    # ── Public API ─────────────────────────────────────────────────────────────

    def record_decision(
        self,
        *,
        approval_id: str,
        service_name: str,
        alert_name: str,
        action_type: str,
        env_tier: str,
        decided_by: str,
        decision: str,           # "approved" | "declined" | "autonomous"
        risk_score: float = 0.0,
        notes: str = "",
    ) -> ApprovalRecord:
        """Record a new approval decision (outcome defaults to 'pending')."""
        self._ensure_loaded()
        record = ApprovalRecord(
            record_id=str(uuid.uuid4()),
            approval_id=approval_id,
            service_name=service_name,
            alert_name=alert_name,
            action_type=action_type,
            env_tier=env_tier,
            decided_by=decided_by,
            decision=decision,
            execution_outcome="pending",
            risk_score=risk_score,
            decided_at=datetime.now(timezone.utc).isoformat(),
            notes=notes,
        )
        self._append(record)
        logger.info(
            "History: recorded %s decision  service=%s  action=%s  by=%s",
            decision, service_name, action_type, decided_by,
        )
        return record

    def update_outcome(self, approval_id: str, outcome: str) -> None:
        """
        Update execution_outcome for all records matching approval_id.
        outcome: "success" | "failure" | "skipped"
        """
        self._ensure_loaded()
        now = datetime.now(timezone.utc).isoformat()
        updated = False
        for rec in reversed(self._records):
            if rec.approval_id == approval_id and rec.execution_outcome == "pending":
                rec.execution_outcome = outcome
                rec.executed_at = now
                updated = True
                break

        if updated:
            # Rewrite entire file to reflect the update (records are few)
            self._rewrite()
            logger.info(
                "History: updated outcome approval_id=%s  outcome=%s",
                approval_id, outcome,
            )
        else:
            logger.debug("update_outcome: no pending record for approval_id=%s", approval_id)

    def _rewrite(self) -> None:
        """Rewrite the JSONL file with in-memory state (used after updates)."""
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                for rec in self._records:
                    f.write(json.dumps(asdict(rec)) + "\n")
        except OSError as exc:
            logger.warning("Could not rewrite approval history: %s", exc)

    def get_history(
        self,
        service_name: str,
        action_type: str,
        window_days: int = 90,
    ) -> list[ApprovalRecord]:
        """
        Return records for (service, action_type) within window_days.
        Sorted oldest-first.
        """
        self._ensure_loaded()
        cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
        result: list[ApprovalRecord] = []
        for rec in self._records:
            if rec.service_name != service_name:
                continue
            if rec.action_type != action_type:
                continue
            try:
                ts = datetime.fromisoformat(rec.decided_at)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts >= cutoff:
                    result.append(rec)
            except ValueError:
                result.append(rec)  # include records with unparseable dates
        return sorted(result, key=lambda r: r.decided_at)

    def compute_trust_score(
        self,
        service_name: str,
        action_type: str,
        env_tier: str,
        *,
        min_approvals: int,
        min_success_rate: float,
        window_days: int = 90,
    ) -> TrustScore:
        """
        Compute a TrustScore for a (service, action_type) pair.

        Autonomy is eligible when:
          • approved_count + autonomous_count >= min_approvals
          • success_rate >= min_success_rate
          • No declined decisions in the last window
        """
        records = self.get_history(service_name, action_type, window_days)

        approved     = [r for r in records if r.decision == "approved"]
        declined     = [r for r in records if r.decision == "declined"]
        autonomous   = [r for r in records if r.decision == "autonomous"]
        executed     = [r for r in records if r.execution_outcome in ("success", "failure")]
        successes    = [r for r in executed if r.execution_outcome == "success"]
        failures     = [r for r in executed if r.execution_outcome == "failure"]

        approved_count   = len(approved)
        declined_count   = len(declined)
        autonomous_count = len(autonomous)
        success_count    = len(successes)
        failure_count    = len(failures)
        total_exec       = success_count + failure_count
        success_rate     = (success_count / total_exec) if total_exec > 0 else 0.0

        qualified_count  = approved_count + autonomous_count

        # Eligibility checks
        if qualified_count < min_approvals:
            eligible = False
            reason = (
                f"Needs {min_approvals - qualified_count} more successful approval(s) "
                f"before autonomous execution is unlocked "
                f"({qualified_count}/{min_approvals} so far in last {window_days}d)"
            )
        elif success_rate < min_success_rate and total_exec > 0:
            eligible = False
            reason = (
                f"Success rate {success_rate:.0%} is below required {min_success_rate:.0%} "
                f"({success_count}/{total_exec} executions succeeded)"
            )
        elif declined_count > 0:
            # Recent declines signal human concern — stay in approval mode
            eligible = False
            reason = (
                f"Autonomous mode paused: {declined_count} declined decision(s) found "
                f"in the last {window_days}d — human review recommended"
            )
        else:
            eligible = True
            reason = (
                f"Trust threshold met: {qualified_count} approvals, "
                f"{success_rate:.0%} success rate over {window_days}d"
            )

        return TrustScore(
            service_name=service_name,
            action_type=action_type,
            env_tier=env_tier,
            total_decisions=len(records),
            approved_count=approved_count,
            declined_count=declined_count,
            autonomous_count=autonomous_count,
            success_count=success_count,
            failure_count=failure_count,
            success_rate=success_rate,
            window_days=window_days,
            autonomy_eligible=eligible,
            reason=reason,
        )

    def get_all_services(self) -> list[str]:
        """Return unique service names in the history store."""
        self._ensure_loaded()
        return sorted({r.service_name for r in self._records})

    def get_summary(self, window_days: int = 90) -> dict:
        """Return a compact summary suitable for the /autonomy/history endpoint."""
        self._ensure_loaded()
        cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
        recent = []
        for rec in self._records:
            try:
                ts = datetime.fromisoformat(rec.decided_at)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts >= cutoff:
                    recent.append(rec)
            except ValueError:
                recent.append(rec)

        return {
            "total_records": len(self._records),
            "window_days": window_days,
            "recent_records": len(recent),
            "approved": sum(1 for r in recent if r.decision == "approved"),
            "declined": sum(1 for r in recent if r.decision == "declined"),
            "autonomous": sum(1 for r in recent if r.decision == "autonomous"),
            "successes": sum(1 for r in recent if r.execution_outcome == "success"),
            "failures": sum(1 for r in recent if r.execution_outcome == "failure"),
            "services": sorted({r.service_name for r in recent}),
            "storage_path": self._path,
        }


# ── Module-level singleton ─────────────────────────────────────────────────────
history_store = ApprovalHistoryStore()
