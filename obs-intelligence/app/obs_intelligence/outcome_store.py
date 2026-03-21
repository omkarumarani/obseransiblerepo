"""
obs_intelligence/outcome_store.py
────────────────────────────────────────────────────────────────────────────────
OutcomeStore — SQLite-backed feedback loop for per-scenario weight adjustments.

Each time a remediation completes (success / failure / timedout) the domain
agent POSTs to /intelligence/record-outcome.  This module persists those
outcomes and continuously recalculates a small weight_adjustment
(-0.10 … +0.10) that is applied to every future confidence score for that
scenario, creating a closed learning loop without any external service.

Schema
──────
  scenario_outcomes   — append-only ledger of every recorded outcome.
  scenario_weights    — one row per scenario_id: rolling statistics + adj.

Weight formula
──────────────
  success_rate       = success_count / total_seen
  weight_adjustment  = (success_rate - 0.5) * 0.2

  Maps:
    100% success  →  +0.10   (scenario is trustworthy, boost its confidence)
     50% success  →   0.00   (no opinion yet)
      0% success  →  -0.10   (scenario keeps failing, reduce its confidence)
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("obs_intelligence.outcome_store")

_DB_PATH_DEFAULT = "/data/outcomes.db"


class OutcomeStore:
    """Thread-safe, write-through SQLite cache for scenario outcome weights."""

    def __init__(self, db_path: str = _DB_PATH_DEFAULT) -> None:
        # Ensure parent directory exists (important inside Docker volumes)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._db_path = db_path
        self._init_schema()
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
    ) -> None:
        """
        Persist one outcome event and immediately recalculate the weight for
        this scenario.

        Parameters
        ----------
        scenario_id :
            Canonical scenario identifier (e.g. "high_cpu_saturation").
        outcome :
            One of ``"success"``, ``"failure"``, or ``"timedout"``.
        run_id :
            Optional pipeline run UUID for audit purposes.
        domain :
            "compute" | "storage" (informational only).
        """
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO scenario_outcomes (scenario_id, outcome, run_id, domain, recorded_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (scenario_id, outcome, run_id, domain, now),
            )
        self._recalculate(scenario_id)
        logger.debug(
            "Outcome recorded  scenario=%s  outcome=%s  run_id=%s",
            scenario_id, outcome, run_id,
        )

    def get_weight_adjustment(self, scenario_id: str) -> float:
        """
        Return the current weight adjustment for *scenario_id*.

        Returns 0.0 if there is no recorded history yet (neutral prior).
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT weight_adjustment FROM scenario_weights WHERE scenario_id = ?",
                (scenario_id,),
            ).fetchone()
        return float(row[0]) if row else 0.0

    def stats_all(self) -> list[dict[str, Any]]:
        """
        Return a list of per-scenario statistics suitable for JSON serialisation.

        Each item contains:
          scenario_id, weight_adjustment, total_seen, success_count,
          success_rate, last_updated.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    w.scenario_id,
                    w.weight_adjustment,
                    w.total_seen,
                    w.success_count,
                    CASE WHEN w.total_seen > 0
                         THEN ROUND(CAST(w.success_count AS REAL) / w.total_seen, 4)
                         ELSE 0.0 END AS success_rate,
                    w.last_updated
                FROM scenario_weights w
                ORDER BY w.total_seen DESC
                """
            ).fetchall()

        return [
            {
                "scenario_id":       row[0],
                "weight_adjustment": row[1],
                "total_seen":        row[2],
                "success_count":     row[3],
                "success_rate":      row[4],
                "last_updated":      row[5],
            }
            for row in rows
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scenario_outcomes (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    scenario_id TEXT NOT NULL,
                    outcome     TEXT NOT NULL,
                    run_id      TEXT,
                    domain      TEXT,
                    recorded_at TEXT DEFAULT (datetime('now'))
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scenario_weights (
                    scenario_id       TEXT PRIMARY KEY,
                    weight_adjustment REAL    DEFAULT 0.0,
                    total_seen        INTEGER DEFAULT 0,
                    success_count     INTEGER DEFAULT 0,
                    last_updated      TEXT
                )
                """
            )

    def _recalculate(self, scenario_id: str) -> None:
        """Recompute weight_adjustment from the full outcome history and upsert."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*)                                          AS total_seen,
                    SUM(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) AS success_count
                FROM scenario_outcomes
                WHERE scenario_id = ?
                """,
                (scenario_id,),
            ).fetchone()

        total_seen    = row[0] if row else 0
        success_count = row[1] if row else 0

        if total_seen == 0:
            weight_adjustment = 0.0
        else:
            success_rate = success_count / total_seen
            # Maps: 100% → +0.10,  50% → 0.0,  0% → -0.10
            weight_adjustment = (success_rate - 0.5) * 0.2

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO scenario_weights
                    (scenario_id, weight_adjustment, total_seen, success_count, last_updated)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(scenario_id) DO UPDATE SET
                    weight_adjustment = excluded.weight_adjustment,
                    total_seen        = excluded.total_seen,
                    success_count     = excluded.success_count,
                    last_updated      = excluded.last_updated
                """,
                (scenario_id, weight_adjustment, total_seen, success_count, now),
            )

        logger.debug(
            "Weight recalculated  scenario=%s  total=%d  successes=%d  adj=%.4f",
            scenario_id, total_seen, success_count, weight_adjustment,
        )

    def _connect(self) -> sqlite3.Connection:
        """Return an auto-committing connection (used as context manager)."""
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.isolation_level = None   # autocommit
        return conn
