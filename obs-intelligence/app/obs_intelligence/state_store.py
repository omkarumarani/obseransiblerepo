"""
obs_intelligence/state_store.py
─────────────────────────────────────────────────────────────────────────────
Persistent SQLite store for two pieces of state previously lost on every
obs-intelligence restart:

  1. Intelligence state  — the ``current_intelligence`` dict (anomaly list,
     forecast list, loop counters, last-run timestamps).  Written after every
     background loop iteration; loaded at startup so the dashboard always has
     something to display even if Prometheus is briefly unreachable.

  2. Alert recurrence counters  — append-only log of alert firing timestamps.
     Used to populate ``ObsFeatures.recurrence_count`` for scenario correlation.
     The ``recurring_failure_signature`` scenario triggers ``human_only`` autonomy
     when the same alert fires ≥ 3 times within the recurrence window (default 6h).
     Rows older than the window are pruned on each insert to keep the table small.

SQLite path:  /data/state.db   (override via STATE_DB_PATH env var)
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger("obs-intelligence.state_store")

_DB_PATH_DEFAULT: str = os.getenv("STATE_DB_PATH", "/data/state.db")

#: Window (hours) used for recurrence counting.  Configurable via env var.
RECURRENCE_WINDOW_HOURS: float = float(os.getenv("RECURRENCE_WINDOW_HOURS", "6"))


class StateStore:
    """
    Lightweight SQLite persistence for intelligence state and alert recurrence.

    Thread-safe: each public method opens a fresh connection inside a
    commit/rollback context manager so concurrent FastAPI request handlers
    and APScheduler background jobs cannot deadlock each other.
    """

    def __init__(self, db_path: str = _DB_PATH_DEFAULT) -> None:
        self._db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_schema()
        logger.info("StateStore initialised  db=%s", db_path)

    # ── Connection helper ─────────────────────────────────────────────────────

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        con = sqlite3.connect(self._db_path, timeout=10.0)
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def _init_schema(self) -> None:
        with self._conn() as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS intelligence_state (
                    key        TEXT PRIMARY KEY,
                    value      TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS alert_recurrence (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_name TEXT    NOT NULL,
                    fired_at   REAL    NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_recurrence_alert_time
                    ON alert_recurrence(alert_name, fired_at);
            """)

    # ── Intelligence state ────────────────────────────────────────────────────

    def save_intelligence(self, data: dict[str, Any]) -> None:
        """
        Persist a snapshot of *data* (the full ``current_intelligence`` dict) to
        SQLite.  Each key is stored as a separate row so partial reads are safe.
        Non-serialisable values (e.g. datetime objects) are coerced to strings.
        """
        now = time.time()
        with self._conn() as con:
            for key, value in data.items():
                serialised = json.dumps(value, default=str)
                con.execute(
                    """
                    INSERT INTO intelligence_state(key, value, updated_at)
                        VALUES(?, ?, ?)
                    ON CONFLICT(key) DO UPDATE
                        SET value = excluded.value,
                            updated_at = excluded.updated_at
                    """,
                    (key, serialised, now),
                )

    def load_intelligence(self) -> dict[str, Any]:
        """
        Load the persisted intelligence state snapshot.

        Returns the full ``current_intelligence``-shaped dict with safe defaults
        for any missing keys so callers can rely on the complete schema even on
        the very first start or after a DB wipe.
        """
        defaults: dict[str, Any] = {
            "anomalies": [],
            "forecasts": [],
            "last_analysis_at": None,
            "last_forecast_at": None,
            "analysis_loop_count": 0,
            "forecast_loop_count": 0,
        }
        try:
            with self._conn() as con:
                rows = con.execute(
                    "SELECT key, value FROM intelligence_state"
                ).fetchall()
            for row in rows:
                try:
                    defaults[row["key"]] = json.loads(row["value"])
                except (json.JSONDecodeError, KeyError):
                    pass
            if any(defaults[k] for k in ("anomalies", "forecasts", "analysis_loop_count")):
                logger.info(
                    "Intelligence state restored  anomalies=%d  forecasts=%d  "
                    "analysis_loops=%d  forecast_loops=%d",
                    len(defaults["anomalies"]),
                    len(defaults["forecasts"]),
                    defaults["analysis_loop_count"],
                    defaults["forecast_loop_count"],
                )
        except Exception as exc:
            logger.warning("Could not load intelligence state (using defaults): %s", exc)
        return defaults

    # ── Alert recurrence counters ─────────────────────────────────────────────

    def record_alert_fired(self, alert_name: str) -> int:
        """
        Record one new firing event for *alert_name*.

        Steps:
          1. Delete rows for this alert older than ``RECURRENCE_WINDOW_HOURS``.
          2. Insert the new firing timestamp.
          3. Count and return all rows for this alert within the window.

        The returned count is the value that should be set on
        ``ObsFeatures.recurrence_count`` for scenario correlation.
        """
        now = time.time()
        cutoff = now - RECURRENCE_WINDOW_HOURS * 3600.0
        try:
            with self._conn() as con:
                con.execute(
                    "DELETE FROM alert_recurrence "
                    "WHERE alert_name = ? AND fired_at < ?",
                    (alert_name, cutoff),
                )
                con.execute(
                    "INSERT INTO alert_recurrence(alert_name, fired_at) VALUES(?, ?)",
                    (alert_name, now),
                )
                count: int = con.execute(
                    "SELECT COUNT(*) FROM alert_recurrence "
                    "WHERE alert_name = ? AND fired_at >= ?",
                    (alert_name, cutoff),
                ).fetchone()[0]
        except Exception as exc:
            logger.warning("Could not record alert firing  alert=%s: %s", alert_name, exc)
            return 1
        logger.debug(
            "Alert recurrence recorded  alert=%s  count_in_%.0fh=%d",
            alert_name, RECURRENCE_WINDOW_HOURS, count,
        )
        return count

    def get_recurrence_count(self, alert_name: str) -> int:
        """
        Return the number of times *alert_name* has fired within
        ``RECURRENCE_WINDOW_HOURS`` (does NOT insert a new row).
        """
        cutoff = time.time() - RECURRENCE_WINDOW_HOURS * 3600.0
        try:
            with self._conn() as con:
                row = con.execute(
                    "SELECT COUNT(*) FROM alert_recurrence "
                    "WHERE alert_name = ? AND fired_at >= ?",
                    (alert_name, cutoff),
                ).fetchone()
            return int(row[0]) if row else 0
        except Exception as exc:
            logger.warning("Could not read recurrence count  alert=%s: %s", alert_name, exc)
            return 0

    def cleanup_old_firings(self) -> int:
        """
        Prune ALL alert_recurrence rows older than ``RECURRENCE_WINDOW_HOURS``.
        Called by the hourly scheduler job to keep the table small.
        Returns the number of deleted rows.
        """
        cutoff = time.time() - RECURRENCE_WINDOW_HOURS * 3600.0
        try:
            with self._conn() as con:
                cur = con.execute(
                    "DELETE FROM alert_recurrence WHERE fired_at < ?", (cutoff,)
                )
                deleted = cur.rowcount
            if deleted:
                logger.debug("Pruned %d stale alert recurrence rows", deleted)
            return deleted
        except Exception as exc:
            logger.warning("Could not prune old alert firings: %s", exc)
            return 0
