"""
obs_intelligence/outcome_store.py
────────────────────────────────────────────────────────────────────────────────
OutcomeStore — SQLite-backed feedback loop for per-scenario weight adjustments.

Each time a remediation completes (success / failure / timedout) the domain
agent POSTs to /intelligence/record-outcome.  This module persists those
outcomes and continuously recalculates a weight_adjustment that is applied to
every future confidence score for that scenario, creating a closed learning
loop without any external service.

Reinforcement learning lite
───────────────────────────
Three improvements over a naive success-rate formula:

1. Exponential time decay
   Recent outcomes count more than old ones.  Weight of an outcome recorded
   *d* days ago = exp(−λ·d) where λ = ln(2) / half_life_days.
   Default: 30-day half-life (an outcome loses 50% of its weight after 30 days).

2. Dynamic adjustment range
   The ±cap scales with the number of recorded outcomes (evidence tier):
     none  (0)      →  ±0.00
     low   (1–4)    →  ±0.10
     medium (5–14)  →  ±0.15
     high  (15–29)  →  ±0.20
     strong (30+)   →  ±0.25

3. Multi-signal inputs
   Real remediation outcomes carry signal_strength = 1.0.
   Implicit validation signals carry signal_strength = 0.3:
     "corroborated" → validation_positive  (+0.3 weight)
     "divergent"    → validation_negative  (−0.3 weight)
   Partial outcomes (timedout, declined) count as 0.5-positive.

4. Fast resolution bonus
   If resolution_time_seconds < 300 s and the outcome is positive, the
   outcome value is boosted to min(1.0, value + 0.1).

Schema
──────
  scenario_outcomes  — append-only ledger of every recorded outcome.
  scenario_weights   — one row per scenario_id: rolling statistics + adj.

Outcome normalisation
─────────────────────
  Positive (1.0) : success, resolved, auto_resolved, autonomous_executed,
                   validation_positive
  Partial  (0.5) : timedout, declined, human_escalated, timeout
  Negative (0.0) : everything else  (failure, failed, escalated, …)
"""

from __future__ import annotations

import logging
import math
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("obs_intelligence.outcome_store")

_DB_PATH_DEFAULT = "/data/outcomes.db"

# ── Decay constants ────────────────────────────────────────────────────────────
_DECAY_HALF_LIFE_DAYS: float = float(os.getenv("OUTCOME_DECAY_HALF_LIFE_DAYS", "30"))
_DECAY_LAMBDA: float          = math.log(2.0) / max(1.0, _DECAY_HALF_LIFE_DAYS)

# ── Outcome normalisation ──────────────────────────────────────────────────────
_POSITIVE_OUTCOMES = frozenset({
    "success", "resolved", "auto_resolved", "autonomous_executed",
    "validation_positive",
})
_PARTIAL_OUTCOMES = frozenset({
    "timedout", "declined", "human_escalated", "timeout",
})


def _outcome_value(outcome: str) -> float:
    """Map an outcome string to a numeric signal in [0.0, 1.0]."""
    if outcome in _POSITIVE_OUTCOMES:
        return 1.0
    if outcome in _PARTIAL_OUTCOMES:
        return 0.5
    return 0.0   # negative (failure, escalated, failed, …)


def _decay_weight(days_ago: float) -> float:
    """Exponential decay: returns exp(−λ·d), always in (0, 1]."""
    return math.exp(-_DECAY_LAMBDA * max(0.0, days_ago))


def _evidence_tier(total_seen: int) -> str:
    if total_seen >= 30: return "strong"
    if total_seen >= 15: return "high"
    if total_seen >= 5:  return "medium"
    if total_seen >= 1:  return "low"
    return "none"


def _max_adjustment(total_seen: int) -> float:
    if total_seen >= 30: return 0.25
    if total_seen >= 15: return 0.20
    if total_seen >= 5:  return 0.15
    if total_seen >= 1:  return 0.10
    return 0.0


class OutcomeStore:
    """Thread-safe, write-through SQLite store for scenario outcome-feedback weights."""

    def __init__(self, db_path: str = _DB_PATH_DEFAULT) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._init_schema()
        self._refresh_stale_weights()
        logger.info("OutcomeStore initialised  db=%s", db_path)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def record(
        self,
        scenario_id: str,
        outcome: str,
        run_id: str = "",
        domain: str = "compute",
        signal_strength: float = 1.0,
        resolution_time_seconds: float | None = None,
    ) -> None:
        """
        Persist one outcome event and immediately recalculate the weight for
        this scenario.

        Parameters
        ----------
        scenario_id :
            Canonical scenario identifier (e.g. "high_cpu_saturation").
        outcome :
            Outcome string.  Normalised via _outcome_value():
            "success"/"resolved"/"auto_resolved" → positive;
            "timedout"/"declined" → partial; everything else → negative.
        run_id :
            Optional pipeline run UUID for audit purposes.
        domain :
            "compute" | "storage" (informational only).
        signal_strength :
            Weight multiplier for this signal.  1.0 for real remediation
            outcomes; 0.3 for implicit validation signals.
        resolution_time_seconds :
            Optional time-to-resolution.  Fast resolutions (< 300 s) earn
            an extra +0.1 boost on the outcome value.
        """
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO scenario_outcomes
                    (scenario_id, outcome, run_id, domain,
                     signal_strength, resolution_time_seconds, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (scenario_id, outcome, run_id, domain,
                 float(signal_strength), resolution_time_seconds, now),
            )
        self._recalculate(scenario_id)
        logger.debug(
            "Outcome recorded  scenario=%s  outcome=%s  strength=%.2f  run_id=%s",
            scenario_id, outcome, signal_strength, run_id,
        )

    def record_validation_signal(
        self,
        scenario_id: str,
        validation_status: str,
        run_id: str = "",
    ) -> None:
        """
        Record an implicit 0.3-weight signal derived from LLM validation.

        Called after POST /intelligence/validate-external-analysis completes
        when the local LLM returns "corroborated" or "divergent".

        "corroborated" → "validation_positive"  (signal_strength 0.3)
        "divergent"    → "validation_negative"  (signal_strength 0.3)
        "weak_support" is ignored (insufficient signal information).
        """
        if not scenario_id:
            return
        if validation_status not in ("corroborated", "divergent"):
            return
        outcome = (
            "validation_positive"
            if validation_status == "corroborated"
            else "validation_negative"
        )
        self.record(
            scenario_id=scenario_id,
            outcome=outcome,
            run_id=run_id,
            domain="obs-intelligence",
            signal_strength=0.3,
        )
        logger.debug(
            "Validation signal recorded  scenario=%s  status=%s",
            scenario_id, validation_status,
        )

    def get_weight_adjustment(self, scenario_id: str) -> float:
        """
        Return the current decay-weighted weight adjustment for *scenario_id*.

        Returns 0.0 if there is no recorded history yet (neutral prior).
        The value is clamped to [−max_adj, +max_adj] at write time so callers
        get a guaranteed bounded delta to add to a confidence score.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT weight_adjustment FROM scenario_weights WHERE scenario_id = ?",
                (scenario_id,),
            ).fetchone()
        return float(row[0]) if row else 0.0

    def stats_all(self) -> list[dict[str, Any]]:
        """
        Return per-scenario statistics suitable for JSON serialisation.

        Fields: scenario_id, weight_adjustment, total_seen, success_count,
        failure_count, signal_count, evidence_tier, trend, decay_success_rate,
        success_rate, last_updated.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    scenario_id, weight_adjustment, total_seen, success_count,
                    failure_count, signal_count, evidence_tier, trend,
                    decay_success_rate, last_updated
                FROM scenario_weights
                ORDER BY total_seen DESC
                """
            ).fetchall()
        return [
            {
                "scenario_id":        row[0],
                "weight_adjustment":  round(float(row[1] or 0.0), 4),
                "total_seen":         int(row[2] or 0),
                "success_count":      int(row[3] or 0),
                "failure_count":      int(row[4] or 0),
                "signal_count":       int(row[5] or 0),
                "evidence_tier":      str(row[6] or "none"),
                "trend":              str(row[7] or "stable"),
                "decay_success_rate": round(float(row[8] or 0.5), 4),
                "success_rate":       round((row[3] or 0) / max(1, (row[2] or 1)), 4),
                "last_updated":       str(row[9] or ""),
            }
            for row in rows
        ]

    def trend_data(self, scenario_id: str, limit: int = 20) -> list[dict[str, Any]]:
        """
        Return the last *limit* raw outcome rows for a scenario, oldest first.

        Intended for sparkline rendering in the Streamlit dashboard.
        Each row contains: outcome, signal_strength, outcome_value, recorded_at.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT outcome, signal_strength, recorded_at
                FROM scenario_outcomes
                WHERE scenario_id = ?
                ORDER BY recorded_at DESC
                LIMIT ?
                """,
                (scenario_id, limit),
            ).fetchall()
        result = []
        for outcome, sig, ts in reversed(rows):
            result.append({
                "outcome":         outcome,
                "signal_strength": float(sig or 1.0),
                "outcome_value":   _outcome_value(outcome),
                "recorded_at":     str(ts),
            })
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scenario_outcomes (
                    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                    scenario_id             TEXT    NOT NULL,
                    outcome                 TEXT    NOT NULL,
                    run_id                  TEXT,
                    domain                  TEXT,
                    signal_strength         REAL    DEFAULT 1.0,
                    resolution_time_seconds REAL,
                    recorded_at             TEXT    DEFAULT (datetime('now'))
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scenario_weights (
                    scenario_id        TEXT PRIMARY KEY,
                    weight_adjustment  REAL    DEFAULT 0.0,
                    total_seen         INTEGER DEFAULT 0,
                    success_count      INTEGER DEFAULT 0,
                    failure_count      INTEGER DEFAULT 0,
                    signal_count       INTEGER DEFAULT 0,
                    evidence_tier      TEXT    DEFAULT 'none',
                    trend              TEXT    DEFAULT 'stable',
                    decay_success_rate REAL    DEFAULT 0.5,
                    last_updated       TEXT
                )
                """
            )
            self._migrate_schema(conn)

    def _refresh_stale_weights(self) -> None:
        """
        On startup, recalculate weights for any scenario whose scenario_weights
        row pre-dates the RL upgrade (indicated by evidence_tier='none' while
        total_seen > 0, or whose success_count is 0 despite having outcomes).

        This ensures that existing "resolved" outcomes — which the old formula
        never counted as positive — are immediately reflected in the live weights.
        """
        with self._connect() as conn:
            # Scenarios recorded in the ledger but with stale / missing weights
            rows = conn.execute(
                """
                SELECT DISTINCT o.scenario_id
                FROM scenario_outcomes o
                LEFT JOIN scenario_weights w ON w.scenario_id = o.scenario_id
                WHERE w.scenario_id IS NULL
                   OR (w.total_seen > 0 AND w.evidence_tier = 'none')
                   OR (w.total_seen > 0 AND w.success_count = 0
                       AND w.signal_count = 0 AND w.failure_count = 0)
                """
            ).fetchall()
        stale = [row[0] for row in rows]
        if stale:
            logger.info(
                "OutcomeStore: recalculating %d stale scenario weights: %s",
                len(stale), stale,
            )
            for scenario_id in stale:
                self._recalculate(scenario_id)

    def _migrate_schema(self, conn: sqlite3.Connection) -> None:
        """Add new columns to existing tables without dropping data."""
        new_outcome_cols = [
            ("signal_strength",          "REAL    DEFAULT 1.0"),
            ("resolution_time_seconds",  "REAL"),
        ]
        new_weight_cols = [
            ("failure_count",        "INTEGER DEFAULT 0"),
            ("signal_count",         "INTEGER DEFAULT 0"),
            ("evidence_tier",        "TEXT    DEFAULT 'none'"),
            ("trend",                "TEXT    DEFAULT 'stable'"),
            ("decay_success_rate",   "REAL    DEFAULT 0.5"),
        ]
        for col, definition in new_outcome_cols:
            try:
                conn.execute(
                    f"ALTER TABLE scenario_outcomes ADD COLUMN {col} {definition}"
                )
                logger.info("OutcomeStore: migrated scenario_outcomes — added %s", col)
            except sqlite3.OperationalError:
                pass  # column already exists
        for col, definition in new_weight_cols:
            try:
                conn.execute(
                    f"ALTER TABLE scenario_weights ADD COLUMN {col} {definition}"
                )
                logger.info("OutcomeStore: migrated scenario_weights — added %s", col)
            except sqlite3.OperationalError:
                pass  # column already exists

    def _recalculate(self, scenario_id: str) -> None:
        """
        Recompute the decay-weighted weight_adjustment and upsert into
        scenario_weights.

        Algorithm:
          1. Load all outcome rows (outcome, signal_strength, resolution_time_s, ts)
          2. For each row compute decay_weight = exp(−λ · days_ago)
          3. effective_weight = decay_weight × signal_strength
          4. decay_success_rate = Σ(w · outcome_value) / Σ(w)
          5. weight_adjustment  = clamp(
                 (decay_success_rate − 0.5) × (max_adj × 2),
                 −max_adj, +max_adj)
        """
        now_ts = datetime.now(timezone.utc)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT outcome, signal_strength, resolution_time_seconds, recorded_at
                FROM scenario_outcomes
                WHERE scenario_id = ?
                ORDER BY recorded_at ASC
                """,
                (scenario_id,),
            ).fetchall()

        total_seen    = len(rows)
        # Count only full-strength real outcomes for human-readable stats
        success_count = sum(
            1 for r in rows
            if _outcome_value(r[0]) > 0.5 and float(r[1] or 1.0) >= 1.0
        )
        failure_count = sum(
            1 for r in rows
            if _outcome_value(r[0]) < 0.5 and float(r[1] or 1.0) >= 1.0
        )
        signal_count  = sum(1 for r in rows if float(r[1] or 1.0) < 1.0)

        weighted_positives = 0.0
        weighted_total     = 0.0

        for outcome, sig_raw, res_time, recorded_at_str in rows:
            try:
                recorded_at = datetime.fromisoformat(
                    str(recorded_at_str).replace(" ", "T")
                )
                if recorded_at.tzinfo is None:
                    recorded_at = recorded_at.replace(tzinfo=timezone.utc)
                days_ago = max(0.0, (now_ts - recorded_at).total_seconds() / 86400.0)
            except Exception:
                days_ago = 0.0

            decay_w = _decay_weight(days_ago)
            sig_w   = float(sig_raw or 1.0)
            w       = decay_w * sig_w

            outcome_val = _outcome_value(outcome)
            # Fast resolution bonus — only applies to real full-strength positives
            if (outcome_val > 0.0 and sig_w >= 1.0
                    and res_time is not None and float(res_time) < 300.0):
                outcome_val = min(1.0, outcome_val + 0.1)

            weighted_positives += w * outcome_val
            weighted_total     += w

        if weighted_total <= 0.0:
            decay_success_rate = 0.5   # no data → neutral
        else:
            decay_success_rate = weighted_positives / weighted_total

        max_a             = _max_adjustment(total_seen)
        weight_adjustment = (decay_success_rate - 0.5) * (max_a * 2.0)
        weight_adjustment = round(max(-max_a, min(max_a, weight_adjustment)), 4)
        tier              = _evidence_tier(total_seen)

        # Trend — compare last-5 vs prior-5 real outcomes only
        real_rows = [r for r in rows if float(r[1] or 1.0) >= 1.0]
        if len(real_rows) >= 10:
            last5  = [_outcome_value(r[0]) for r in real_rows[-5:]]
            prior5 = [_outcome_value(r[0]) for r in real_rows[-10:-5]]
            delta  = sum(last5) / 5.0 - sum(prior5) / 5.0
            trend  = "improving" if delta > 0.15 else ("degrading" if delta < -0.15 else "stable")
        elif len(real_rows) >= 3:
            recent_pos = sum(_outcome_value(r[0]) > 0.5 for r in real_rows[-3:])
            trend = "improving" if recent_pos == 3 else ("degrading" if recent_pos == 0 else "stable")
        else:
            trend = "stable"

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO scenario_weights (
                    scenario_id, weight_adjustment, total_seen, success_count,
                    failure_count, signal_count, evidence_tier, trend,
                    decay_success_rate, last_updated
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(scenario_id) DO UPDATE SET
                    weight_adjustment  = excluded.weight_adjustment,
                    total_seen         = excluded.total_seen,
                    success_count      = excluded.success_count,
                    failure_count      = excluded.failure_count,
                    signal_count       = excluded.signal_count,
                    evidence_tier      = excluded.evidence_tier,
                    trend              = excluded.trend,
                    decay_success_rate = excluded.decay_success_rate,
                    last_updated       = excluded.last_updated
                """,
                (
                    scenario_id, weight_adjustment, total_seen, success_count,
                    failure_count, signal_count, tier, trend,
                    round(decay_success_rate, 4), now,
                ),
            )

        logger.debug(
            "Weight recalculated  scenario=%s  total=%d  decay_rate=%.3f  "
            "adj=%.4f  tier=%s  trend=%s",
            scenario_id, total_seen, decay_success_rate,
            weight_adjustment, tier, trend,
        )

    def _connect(self) -> sqlite3.Connection:
        """Return an auto-committing connection."""
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.isolation_level = None   # autocommit
        return conn
