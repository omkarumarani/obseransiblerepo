"""
SQLite storage for pipeline history and scenario statistics.

This module handles persistent storage of pipeline execution snapshots
for the playback feature (Section 1.2) and scenario statistics aggregation
for the knowledge map (Section 1.3).
"""
import os
import json
import aiosqlite
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from .models import (
    PipelineRunSummary,
    PipelineRunFull,
    HistoryFilters,
    ScenarioCard,
)

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("DB_PATH", "/data/pipeline_history.db")


async def init_db():
    """Initialize SQLite database with required tables"""
    async with aiosqlite.connect(DB_PATH) as db:
        # Pipeline runs summary table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                session_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                service_name TEXT NOT NULL,
                alert_name TEXT NOT NULL,
                scenario_matched TEXT,
                risk_score REAL NOT NULL,
                autonomy_decision TEXT NOT NULL,
                outcome TEXT NOT NULL,
                duration REAL,
                domain TEXT NOT NULL
            )
        """)
        
        # Full pipeline run data (JSON blob for complete snapshot)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_snapshots (
                session_id TEXT PRIMARY KEY,
                snapshot_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES pipeline_runs(session_id)
            )
        """)
        
        # Scenario statistics cache
        await db.execute("""
            CREATE TABLE IF NOT EXISTS scenario_stats (
                scenario_id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                domain TEXT NOT NULL,
                autonomy_badge TEXT NOT NULL,
                action TEXT NOT NULL,
                confidence_threshold REAL NOT NULL,
                times_seen INTEGER DEFAULT 0,
                last_seen TEXT,
                avg_resolution_time REAL,
                success_rate REAL,
                updated_at TEXT NOT NULL
            )
        """)
        
        # Indexes for faster queries
        await db.execute("CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON pipeline_runs(timestamp)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_runs_domain ON pipeline_runs(domain)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_runs_scenario ON pipeline_runs(scenario_matched)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_runs_outcome ON pipeline_runs(outcome)")
        
        await db.commit()
        logger.info(f"Database initialized at {DB_PATH}")


async def save_pipeline_run(summary: PipelineRunSummary, full_snapshot: PipelineRunFull):
    """Save a completed pipeline run with full snapshot"""
    async with aiosqlite.connect(DB_PATH) as db:
        # Insert summary
        await db.execute("""
            INSERT OR REPLACE INTO pipeline_runs 
            (session_id, timestamp, service_name, alert_name, scenario_matched, 
             risk_score, autonomy_decision, outcome, duration, domain)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            summary.session_id,
            summary.timestamp,
            summary.service_name,
            summary.alert_name,
            summary.scenario_matched,
            summary.risk_score,
            summary.autonomy_decision,
            summary.outcome,
            summary.duration,
            summary.domain,
        ))
        
        # Insert full snapshot as JSON
        snapshot_json = full_snapshot.model_dump_json()
        await db.execute("""
            INSERT OR REPLACE INTO pipeline_snapshots 
            (session_id, snapshot_json, created_at)
            VALUES (?, ?, ?)
        """, (
            summary.session_id,
            snapshot_json,
            datetime.utcnow().isoformat(),
        ))
        
        await db.commit()
        logger.info(f"Saved pipeline run {summary.session_id}")
        
        # Update scenario statistics if scenario was matched
        if summary.scenario_matched:
            await update_scenario_stats(db, summary)


async def update_scenario_stats(db: aiosqlite.Connection, summary: PipelineRunSummary):
    """Update scenario statistics after a new run"""
    scenario_id = summary.scenario_matched
    if not scenario_id:
        return
    
    # Calculate new statistics
    cursor = await db.execute("""
        SELECT 
            COUNT(*) as times_seen,
            MAX(timestamp) as last_seen,
            AVG(duration) as avg_resolution_time,
            SUM(CASE WHEN outcome = 'success' THEN 1.0 ELSE 0.0 END) / COUNT(*) as success_rate
        FROM pipeline_runs
        WHERE scenario_matched = ?
    """, (scenario_id,))
    
    row = await cursor.fetchone()
    if row:
        times_seen, last_seen, avg_resolution_time, success_rate = row
        
        # Update stats table (will need scenario metadata from config)
        await db.execute("""
            UPDATE scenario_stats
            SET times_seen = ?,
                last_seen = ?,
                avg_resolution_time = ?,
                success_rate = ?,
                updated_at = ?
            WHERE scenario_id = ?
        """, (
            times_seen,
            last_seen,
            avg_resolution_time,
            success_rate,
            datetime.utcnow().isoformat(),
            scenario_id,
        ))
        
        await db.commit()


async def get_pipeline_history(filters: HistoryFilters) -> List[PipelineRunSummary]:
    """Query pipeline history with optional filters"""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        
        query = "SELECT * FROM pipeline_runs WHERE 1=1"
        params = []
        
        if filters.domain:
            query += " AND domain = ?"
            params.append(filters.domain)
        
        if filters.scenario:
            query += " AND scenario_matched = ?"
            params.append(filters.scenario)
        
        if filters.start_date:
            query += " AND timestamp >= ?"
            params.append(filters.start_date)
        
        if filters.end_date:
            query += " AND timestamp <= ?"
            params.append(filters.end_date)
        
        if filters.autonomy_decision:
            query += " AND autonomy_decision = ?"
            params.append(filters.autonomy_decision)
        
        if filters.outcome:
            query += " AND outcome = ?"
            params.append(filters.outcome)
        
        query += " ORDER BY timestamp DESC LIMIT 200"
        
        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()
        
        return [PipelineRunSummary(**dict(row)) for row in rows]


async def get_pipeline_snapshot(session_id: str) -> Optional[PipelineRunFull]:
    """Retrieve full pipeline snapshot for playback"""
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute("""
            SELECT snapshot_json FROM pipeline_snapshots WHERE session_id = ?
        """, (session_id,))
        
        row = await cursor.fetchone()
        if row:
            return PipelineRunFull.model_validate_json(row[0])
        return None


async def seed_scenario_metadata(scenarios: List[Dict[str, Any]]):
    """Seed scenario metadata from configuration files"""
    async with aiosqlite.connect(DB_PATH) as db:
        for scenario in scenarios:
            await db.execute("""
                INSERT OR IGNORE INTO scenario_stats 
                (scenario_id, display_name, domain, autonomy_badge, action, 
                 confidence_threshold, times_seen, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, 0, ?)
            """, (
                scenario["scenario_id"],
                scenario["display_name"],
                scenario["domain"],
                scenario["autonomy_badge"],
                scenario["action"],
                scenario["confidence_threshold"],
                datetime.utcnow().isoformat(),
            ))
        
        await db.commit()
        logger.info(f"Seeded {len(scenarios)} scenario definitions")


async def get_all_scenarios() -> List[ScenarioCard]:
    """Get all scenarios with aggregated statistics"""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        
        cursor = await db.execute("""
            SELECT * FROM scenario_stats ORDER BY domain, display_name
        """)
        
        rows = await cursor.fetchall()
        return [ScenarioCard(**dict(row)) for row in rows]


async def get_scenario_runs(scenario_id: str, limit: int = 20) -> List[PipelineRunSummary]:
    """Get historical runs for a specific scenario"""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        
        cursor = await db.execute("""
            SELECT * FROM pipeline_runs 
            WHERE scenario_matched = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (scenario_id, limit))
        
        rows = await cursor.fetchall()
        return [PipelineRunSummary(**dict(row)) for row in rows]
