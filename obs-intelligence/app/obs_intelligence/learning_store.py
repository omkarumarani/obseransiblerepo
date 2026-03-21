from __future__ import annotations

import json
import logging
import re
import sqlite3
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

logger = logging.getLogger("obs_intelligence.learning_store")

_DB_PATH_DEFAULT = "/data/learning.db"
_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_json(value: Any) -> str:
    return json.dumps(value or {}, sort_keys=True)


def _normalise_text(value: str) -> str:
    return " ".join(value.lower().split())


def _tokenise(value: str) -> set[str]:
    return set(_TOKEN_RE.findall(value.lower()))


class LearningStore:
    """SQLite-backed incident knowledge store for Block F learning metadata."""

    def __init__(self, db_path: str = _DB_PATH_DEFAULT) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._init_schema()
        logger.info("LearningStore initialised  db=%s", db_path)

    def record_validation(
        self,
        *,
        domain: str,
        service_name: str,
        alert_name: str,
        scenario_id: str,
        run_id: str,
        ticket_id: str,
        trace_id: str,
        evidence_summary: str,
        evidence_lines: list[str],
        external_analysis: dict[str, Any],
        validation: dict[str, Any],
        similar_entries: list[dict[str, Any]],
    ) -> dict[str, Any]:
        entry_id = str(uuid.uuid4())
        now = _utc_now()
        root_cause = str(
            external_analysis.get("root_cause")
            or external_analysis.get("rca_detail", {}).get("probable_cause")
            or external_analysis.get("rca_summary", "")
        )
        recommended_action = str(external_analysis.get("recommended_action", ""))
        remediation_summary = str(
            external_analysis.get("ansible_description")
            or external_analysis.get("pr_title")
            or external_analysis.get("rca_summary", "")
        )
        metadata = {
            "domain": domain,
            "service_name": service_name,
            "alert_name": alert_name,
            "scenario_id": scenario_id,
            "run_id": run_id,
            "ticket_id": ticket_id,
            "trace_id": trace_id,
            "timestamp": now,
            "external_source": external_analysis.get("provider", "external"),
            "external_model": external_analysis.get("model", ""),
            "external_confidence": external_analysis.get("confidence"),
            "recommended_action": recommended_action,
            "root_cause": root_cause,
            "remediation_summary": remediation_summary,
            "validation_status": validation.get("status", "unavailable"),
            "validation_confidence": validation.get("confidence"),
            "validation_reason": validation.get("reason", ""),
            "top_similarity": validation.get("top_similarity"),
            "local_model": validation.get("local_model", ""),
            "local_validation_completed": validation.get("completed", False),
            "supporting_entry_ids": validation.get("supporting_entry_ids", []),
            "similar_match_count": len(similar_entries),
            "outcome": "pending",
        }
        document = self._build_document(
            service_name=service_name,
            alert_name=alert_name,
            scenario_id=scenario_id,
            root_cause=root_cause,
            recommended_action=recommended_action,
            remediation_summary=remediation_summary,
            evidence_summary=evidence_summary,
            evidence_lines=evidence_lines,
        )

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO knowledge_entries (
                    entry_id, domain, service_name, alert_name, scenario_id,
                    run_id, ticket_id, trace_id, document, evidence_summary,
                    evidence_lines_json, root_cause, recommended_action,
                    remediation_summary, external_provider, external_model,
                    external_confidence, validation_status, validation_confidence,
                    validation_reason, top_similarity, local_model,
                    local_validation_completed, similar_entries_json,
                    supporting_entry_ids_json, outcome, resolution_time_seconds,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry_id,
                    domain,
                    service_name,
                    alert_name,
                    scenario_id,
                    run_id,
                    ticket_id,
                    trace_id,
                    document,
                    evidence_summary,
                    _safe_json(evidence_lines),
                    root_cause,
                    recommended_action,
                    remediation_summary,
                    str(external_analysis.get("provider", "external")),
                    str(external_analysis.get("model", "")),
                    str(external_analysis.get("confidence", "")),
                    str(validation.get("status", "unavailable")),
                    float(validation.get("confidence") or 0.0),
                    str(validation.get("reason", "")),
                    float(validation.get("top_similarity") or 0.0),
                    str(validation.get("local_model", "")),
                    1 if validation.get("completed") else 0,
                    _safe_json(similar_entries),
                    _safe_json(validation.get("supporting_entry_ids", [])),
                    "pending",
                    None,
                    now,
                    now,
                ),
            )

        return {
            "id": entry_id,
            "document": document,
            "metadata": metadata,
        }

    def update_outcome(
        self,
        *,
        scenario_id: str,
        service_name: str = "",
        run_id: str = "",
        outcome: str,
        resolution_time_seconds: float | None = None,
    ) -> int:
        if not scenario_id and not run_id:
            return 0

        clauses: list[str] = []
        params: list[Any] = []
        if run_id:
            clauses.append("run_id = ?")
            params.append(run_id)
        if scenario_id:
            clauses.append("(scenario_id = ? OR alert_name = ?)")
            params.extend([scenario_id, scenario_id])
        if service_name:
            clauses.append("service_name = ?")
            params.append(service_name)
        where_sql = " AND ".join(clauses) if clauses else "1=1"

        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT entry_id FROM knowledge_entries
                WHERE {where_sql}
                ORDER BY created_at DESC
                LIMIT 1
                """,
                tuple(params),
            ).fetchone()
            if not row:
                return 0
            conn.execute(
                """
                UPDATE knowledge_entries
                SET outcome = ?, resolution_time_seconds = ?, updated_at = ?
                WHERE entry_id = ?
                """,
                (
                    outcome,
                    resolution_time_seconds,
                    _utc_now(),
                    row[0],
                ),
            )
        return 1

    def list_entries(
        self,
        *,
        service_name: str = "",
        scenario_id: str = "",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM knowledge_entries WHERE 1=1"
        params: list[Any] = []
        if service_name:
            query += " AND service_name = ?"
            params.append(service_name)
        if scenario_id:
            query += " AND scenario_id = ?"
            params.append(scenario_id)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(max(1, min(limit, 500)))

        with self._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()

        return [self._row_to_entry(row) for row in rows]

    def find_similar(
        self,
        *,
        query_text: str,
        service_name: str = "",
        scenario_id: str = "",
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM knowledge_entries WHERE 1=1"
        params: list[Any] = []
        if scenario_id:
            sql += " AND scenario_id = ?"
            params.append(scenario_id)
        elif service_name:
            sql += " AND service_name = ?"
            params.append(service_name)
        sql += " ORDER BY created_at DESC LIMIT 200"

        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()

        ranked: list[dict[str, Any]] = []
        for row in rows:
            document = str(row["document"] or "")
            score = self._similarity_score(query_text, document)
            if score <= 0.0:
                continue
            entry = self._row_to_entry(row)
            entry["similarity"] = round(score, 4)
            ranked.append(entry)

        ranked.sort(key=lambda item: item.get("similarity", 0.0), reverse=True)
        return ranked[: max(1, min(limit, 20))]

    def learning_stats(self, *, window_days: int = 30) -> dict[str, Any]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).replace(microsecond=0).isoformat()
        with self._connect() as conn:
            recent = conn.execute(
                "SELECT * FROM knowledge_entries WHERE created_at >= ? ORDER BY created_at DESC",
                (cutoff,),
            ).fetchall()
            total_entries = conn.execute(
                "SELECT COUNT(*) FROM knowledge_entries"
            ).fetchone()[0]
            successful_entries = conn.execute(
                "SELECT COUNT(*) FROM knowledge_entries WHERE outcome IN ('success', 'resolved')"
            ).fetchone()[0]
            scenario_count = conn.execute(
                "SELECT COUNT(DISTINCT scenario_id) FROM knowledge_entries WHERE scenario_id != ''"
            ).fetchone()[0]

        recent_entries = [self._row_to_entry(row) for row in recent]
        completed = [
            entry for entry in recent_entries
            if bool(entry["metadata"].get("local_validation_completed"))
        ]

        def _status_count(status: str) -> int:
            return sum(1 for entry in recent_entries if entry["metadata"].get("validation_status") == status)

        avg_similarity = 0.0
        similarity_values = [
            float(entry["metadata"].get("top_similarity") or 0.0)
            for entry in completed
            if entry["metadata"].get("top_similarity") is not None
        ]
        if similarity_values:
            avg_similarity = sum(similarity_values) / len(similarity_values)

        weekly_hit_rate = self._weekly_hit_rate(recent_entries)

        return {
            "external_llm_calls_30d": len(recent_entries),
            "local_validation_attempts_30d": len(recent_entries),
            "local_validation_completed_30d": len(completed),
            "corroborated_count_30d": _status_count("corroborated"),
            "weak_support_count_30d": _status_count("weak_support"),
            "divergent_count_30d": _status_count("divergent"),
            "insufficient_context_count_30d": _status_count("insufficient_context"),
            "avg_top_similarity_30d": round(avg_similarity, 4),
            "knowledge_entries_total": total_entries,
            "knowledge_entries_with_success_outcome": successful_entries,
            "local_validation_coverage_pct": round((len(completed) / len(recent_entries)), 4) if recent_entries else 0.0,
            "corroboration_rate_pct": round((_status_count("corroborated") / len(completed)), 4) if completed else 0.0,
            "weekly_hit_rate": weekly_hit_rate,
            "scenario_count": scenario_count,
        }

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_entries (
                    entry_id TEXT PRIMARY KEY,
                    domain TEXT,
                    service_name TEXT,
                    alert_name TEXT,
                    scenario_id TEXT,
                    run_id TEXT,
                    ticket_id TEXT,
                    trace_id TEXT,
                    document TEXT,
                    evidence_summary TEXT,
                    evidence_lines_json TEXT,
                    root_cause TEXT,
                    recommended_action TEXT,
                    remediation_summary TEXT,
                    external_provider TEXT,
                    external_model TEXT,
                    external_confidence TEXT,
                    validation_status TEXT,
                    validation_confidence REAL,
                    validation_reason TEXT,
                    top_similarity REAL,
                    local_model TEXT,
                    local_validation_completed INTEGER DEFAULT 0,
                    similar_entries_json TEXT,
                    supporting_entry_ids_json TEXT,
                    outcome TEXT DEFAULT 'pending',
                    resolution_time_seconds REAL,
                    created_at TEXT,
                    updated_at TEXT
                )
                """
            )

    def _build_document(
        self,
        *,
        service_name: str,
        alert_name: str,
        scenario_id: str,
        root_cause: str,
        recommended_action: str,
        remediation_summary: str,
        evidence_summary: str,
        evidence_lines: list[str],
    ) -> str:
        joined_lines = "\n".join(f"- {line}" for line in evidence_lines[:10])
        return (
            f"Service: {service_name}\n"
            f"Alert: {alert_name}\n"
            f"Scenario: {scenario_id or 'unknown'}\n"
            f"Root Cause: {root_cause}\n"
            f"Recommended Action: {recommended_action}\n"
            f"Remediation Summary: {remediation_summary}\n"
            f"Evidence Summary: {evidence_summary}\n"
            f"Evidence Lines:\n{joined_lines}"
        )

    def _weekly_hit_rate(self, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        now = datetime.now(timezone.utc)
        buckets: list[dict[str, Any]] = []
        for offset in range(3, -1, -1):
            start = now - timedelta(days=(offset + 1) * 7)
            end = now - timedelta(days=offset * 7)
            window = []
            for entry in entries:
                created_raw = entry["metadata"].get("timestamp")
                if not created_raw:
                    continue
                created = datetime.fromisoformat(str(created_raw))
                if start <= created < end:
                    window.append(entry)
            completed = [entry for entry in window if entry["metadata"].get("local_validation_completed")]
            corroborated = [entry for entry in window if entry["metadata"].get("validation_status") == "corroborated"]
            hit_rate = (len(corroborated) / len(completed)) if completed else 0.0
            buckets.append({
                "week": f"W{4 - offset}",
                "hit_rate": round(hit_rate, 4),
            })
        return buckets

    def _row_to_entry(self, row: sqlite3.Row) -> dict[str, Any]:
        metadata = {
            "domain": row["domain"],
            "service_name": row["service_name"],
            "alert_name": row["alert_name"],
            "scenario_id": row["scenario_id"],
            "run_id": row["run_id"],
            "ticket_id": row["ticket_id"],
            "trace_id": row["trace_id"],
            "timestamp": row["created_at"],
            "external_source": row["external_provider"],
            "external_model": row["external_model"],
            "external_confidence": row["external_confidence"],
            "action_taken": row["recommended_action"],
            "root_cause": row["root_cause"],
            "remediation_summary": row["remediation_summary"],
            "validation_status": row["validation_status"],
            "validation_confidence": row["validation_confidence"],
            "validation_reason": row["validation_reason"],
            "top_similarity": row["top_similarity"],
            "local_model": row["local_model"],
            "local_validation_completed": bool(row["local_validation_completed"]),
            "supporting_entry_ids": json.loads(row["supporting_entry_ids_json"] or "[]"),
            "similar_entries": json.loads(row["similar_entries_json"] or "[]"),
            "outcome": row["outcome"],
            "resolution_time_seconds": row["resolution_time_seconds"],
        }
        return {
            "id": row["entry_id"],
            "document": row["document"],
            "metadata": metadata,
        }

    def _similarity_score(self, query_text: str, candidate_text: str) -> float:
        left = _normalise_text(query_text)
        right = _normalise_text(candidate_text)
        if not left or not right:
            return 0.0

        sequence = SequenceMatcher(None, left, right).ratio()
        left_tokens = _tokenise(left)
        right_tokens = _tokenise(right)
        overlap = left_tokens & right_tokens
        union = left_tokens | right_tokens
        jaccard = (len(overlap) / len(union)) if union else 0.0

        left_counts = Counter(_TOKEN_RE.findall(left))
        right_counts = Counter(_TOKEN_RE.findall(right))
        weighted_overlap = sum(min(left_counts[token], right_counts[token]) for token in overlap)
        weighted_total = max(sum(left_counts.values()), 1)
        weighted = weighted_overlap / weighted_total

        return round((sequence * 0.45) + (jaccard * 0.35) + (weighted * 0.20), 4)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.isolation_level = None
        return conn