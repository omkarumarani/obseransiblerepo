from obs_intelligence.learning_store import LearningStore


def _external_analysis(provider: str = "openai") -> dict:
    return {
        "provider": provider,
        "model": "gpt-4o-mini",
        "confidence": "high",
        "root_cause": "Database connection pool exhaustion",
        "recommended_action": "restart_service",
        "ansible_description": "Restart the service to recycle exhausted workers",
    }


def _validation(status: str = "corroborated", confidence: float = 0.91) -> dict:
    return {
        "status": status,
        "confidence": confidence,
        "reason": "Previous incidents support the same RCA and remediation.",
        "top_similarity": 0.93,
        "local_model": "llama3:8b",
        "completed": True,
        "supporting_entry_ids": [],
    }


def test_record_validation_updates_learning_stats(tmp_path):
    store = LearningStore(str(tmp_path / "learning.db"))

    entry = store.record_validation(
        domain="compute",
        service_name="frontend-api",
        alert_name="HighErrorRate",
        scenario_id="high_error_rate",
        run_id="run-001",
        ticket_id="ticket-001",
        trace_id="trace-001",
        evidence_summary="frontend-api HighErrorRate database pool exhaustion",
        evidence_lines=["error_rate > 25%", "DatabaseConnectionError in logs"],
        external_analysis=_external_analysis(),
        validation=_validation(),
        similar_entries=[],
    )

    assert entry["id"]
    stats = store.learning_stats()
    assert stats["external_llm_calls_30d"] == 1
    assert stats["local_validation_completed_30d"] == 1
    assert stats["corroborated_count_30d"] == 1
    assert stats["knowledge_entries_total"] == 1


def test_find_similar_returns_best_match(tmp_path):
    store = LearningStore(str(tmp_path / "learning.db"))

    store.record_validation(
        domain="compute",
        service_name="frontend-api",
        alert_name="HighErrorRate",
        scenario_id="high_error_rate",
        run_id="run-001",
        ticket_id="ticket-001",
        trace_id="trace-001",
        evidence_summary="frontend-api database connection pool exhaustion and restart_service",
        evidence_lines=["pool exhausted", "500 errors increased"],
        external_analysis=_external_analysis(),
        validation=_validation(),
        similar_entries=[],
    )
    store.record_validation(
        domain="compute",
        service_name="frontend-api",
        alert_name="LatencySpike",
        scenario_id="latency_spike",
        run_id="run-002",
        ticket_id="ticket-002",
        trace_id="trace-002",
        evidence_summary="frontend-api latency spike caused by cache saturation",
        evidence_lines=["p99 latency > 2s", "cache miss storm"],
        external_analysis={
            **_external_analysis(),
            "root_cause": "Cache saturation",
            "recommended_action": "scale_cache",
        },
        validation=_validation(status="weak_support", confidence=0.62),
        similar_entries=[],
    )

    matches = store.find_similar(
        query_text="frontend-api database pool exhaustion restart_service 500 errors",
        service_name="frontend-api",
        scenario_id="high_error_rate",
    )

    assert matches
    assert matches[0]["metadata"]["scenario_id"] == "high_error_rate"
    assert matches[0]["similarity"] > 0.4


def test_update_outcome_marks_latest_matching_entry(tmp_path):
    store = LearningStore(str(tmp_path / "learning.db"))

    store.record_validation(
        domain="compute",
        service_name="frontend-api",
        alert_name="HighErrorRate",
        scenario_id="high_error_rate",
        run_id="run-003",
        ticket_id="ticket-003",
        trace_id="trace-003",
        evidence_summary="frontend-api database pool exhaustion",
        evidence_lines=["error rate rising"],
        external_analysis=_external_analysis(),
        validation=_validation(),
        similar_entries=[],
    )

    updated = store.update_outcome(
        scenario_id="high_error_rate",
        service_name="frontend-api",
        run_id="run-003",
        outcome="success",
        resolution_time_seconds=42.0,
    )

    assert updated == 1
    rows = store.list_entries(service_name="frontend-api", scenario_id="high_error_rate", limit=5)
    assert rows[0]["metadata"]["outcome"] == "success"
    assert rows[0]["metadata"]["resolution_time_seconds"] == 42.0