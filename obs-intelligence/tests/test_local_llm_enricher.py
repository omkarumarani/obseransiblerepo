"""
tests/test_local_llm_enricher.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for obs_intelligence.local_llm_enricher.

All external I/O is mocked:
  - chromadb.HttpClient / collection methods → MagicMock
  - ollama.Client.embeddings / generate      → MagicMock

No running ChromaDB, Ollama, or network connections are required.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

# ── Add obs-intelligence/app to path so obs_intelligence is importable ───────
_APP_DIR = pathlib.Path(__file__).parents[1] / "app"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

# Disable LOCAL_LLM by default; individual tests override via env patching
os.environ.setdefault("LOCAL_LLM_ENABLED", "true")
os.environ.setdefault("LOCAL_LLM_MIN_SIMILARITY", "0.7")
os.environ.setdefault("LOCAL_LLM_TOP_K", "3")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_chroma_query_result(
    ids: list[str],
    docs: list[str],
    metas: list[dict],
    distances: list[float],
) -> dict:
    """Build a mock ChromaDB collection.query() return value."""
    return {
        "ids":       [ids],
        "documents": [docs],
        "metadatas": [metas],
        "distances": [distances],
    }


def _make_chroma_get_result(
    ids: list[str],
    docs: list[str] | None = None,
    metas: list[dict] | None = None,
) -> dict:
    return {
        "ids":       ids,
        "documents": docs or [""] * len(ids),
        "metadatas": metas or [{}] * len(ids),
    }


def _mock_collection(
    query_result: dict | None = None,
    get_result: dict | None = None,
) -> MagicMock:
    col = MagicMock()
    col.query.return_value  = query_result or _make_chroma_query_result([], [], [], [])
    col.get.return_value    = get_result   or _make_chroma_get_result([])
    col.upsert.return_value = None
    col.update.return_value = None
    return col


def _make_enricher(collection: MagicMock | None = None):
    """
    Return a fresh LocalLLMEnricher with ChromaDB pre-connected to the
    supplied mock collection (bypasses _get_collection's real network call).
    """
    # Import fresh to avoid module-level singleton caching
    from obs_intelligence.local_llm_enricher import LocalLLMEnricher
    e = LocalLLMEnricher()
    e._collection = collection or _mock_collection()
    return e


# ─────────────────────────────────────────────────────────────────────────────
# KnowledgeEntry
# ─────────────────────────────────────────────────────────────────────────────

class TestKnowledgeEntry:
    def test_similarity_reads_metadata(self):
        from obs_intelligence.local_llm_enricher import KnowledgeEntry
        e = KnowledgeEntry(id="x", document="doc", metadata={"similarity": 0.91})
        assert e.similarity() == 0.91

    def test_similarity_defaults_to_zero(self):
        from obs_intelligence.local_llm_enricher import KnowledgeEntry
        e = KnowledgeEntry(id="x", document="doc")
        assert e.similarity() == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# LocalValidationResult
# ─────────────────────────────────────────────────────────────────────────────

class TestLocalValidationResult:
    def test_to_dict_contains_all_fields(self):
        from obs_intelligence.local_llm_enricher import LocalValidationResult
        r = LocalValidationResult(
            validation_status="corroborated",
            confidence=0.92,
            rca_alignment="Matches DB pool pattern",
            action_alignment="restart_service aligns",
            suggested_adjustment="",
            reasoning_summary="Strong historical support",
            top_similarity=0.95,
            similar_count=3,
        )
        d = r.to_dict()
        assert d["validation_status"] == "corroborated"
        assert d["confidence"] == 0.92
        assert d["similar_count"] == 3

    def test_default_status_is_unavailable(self):
        from obs_intelligence.local_llm_enricher import LocalValidationResult
        r = LocalValidationResult()
        assert r.validation_status == "unavailable"


# ─────────────────────────────────────────────────────────────────────────────
# query_similar_incidents
# ─────────────────────────────────────────────────────────────────────────────

class TestQuerySimilarIncidents:

    @pytest.mark.asyncio
    async def test_returns_entries_above_min_similarity(self):
        """Entries with cosine distance ≤ 0.3 (similarity ≥ 0.7) are returned."""
        col = _mock_collection(
            query_result=_make_chroma_query_result(
                ids=["id1", "id2", "id3"],
                docs=["doc1", "doc2", "doc3"],
                metas=[
                    {"scenario_id": "high_error_rate"},
                    {"scenario_id": "latency_spike"},
                    {"scenario_id": "oom"},
                ],
                distances=[0.05, 0.25, 0.50],   # similarities: 0.95, 0.75, 0.50
            )
        )
        enricher = _make_enricher(col)

        mock_embed = [0.1, 0.2, 0.3]
        with patch.object(enricher, "_embed", return_value=mock_embed):
            results = await enricher.query_similar_incidents("frontend latency spike")

        # Only id1 (0.95) and id2 (0.75) clear the 0.7 threshold; id3 (0.50) filtered
        assert len(results) == 2
        assert results[0].id == "id1"   # highest similarity first
        assert results[1].id == "id2"
        assert results[0].similarity() == pytest.approx(0.95, abs=0.001)

    @pytest.mark.asyncio
    async def test_returns_empty_when_chromadb_unavailable(self):
        enricher = _make_enricher()
        enricher._collection = None

        with patch.object(enricher, "_get_collection", return_value=None):
            results = await enricher.query_similar_incidents("some alert")

        assert results == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_embedding_fails(self):
        col = _mock_collection()
        enricher = _make_enricher(col)

        with patch.object(enricher, "_embed", return_value=None):
            results = await enricher.query_similar_incidents("some alert")

        assert results == []
        col.query.assert_not_called()

    @pytest.mark.asyncio
    async def test_applies_domain_filter(self):
        col = _mock_collection(query_result=_make_chroma_query_result([], [], [], []))
        enricher = _make_enricher(col)

        with patch.object(enricher, "_embed", return_value=[0.1, 0.2]):
            await enricher.query_similar_incidents("text", domain="storage")

        call_kwargs = col.query.call_args[1]
        assert call_kwargs["where"] == {"domain": "storage"}

    @pytest.mark.asyncio
    async def test_no_domain_filter_when_domain_empty(self):
        col = _mock_collection(query_result=_make_chroma_query_result([], [], [], []))
        enricher = _make_enricher(col)

        with patch.object(enricher, "_embed", return_value=[0.1, 0.2]):
            await enricher.query_similar_incidents("text", domain="")

        call_kwargs = col.query.call_args[1]
        assert "where" not in call_kwargs

    @pytest.mark.asyncio
    async def test_resets_collection_on_chroma_error(self):
        col = MagicMock()
        col.query.side_effect = Exception("network error")
        enricher = _make_enricher(col)

        with patch.object(enricher, "_embed", return_value=[0.1, 0.2]):
            results = await enricher.query_similar_incidents("text")

        assert results == []
        assert enricher._collection is None   # reset_collection called


# ─────────────────────────────────────────────────────────────────────────────
# validate_external_result
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateExternalResult:

    def _make_similar(self, n: int = 2):
        from obs_intelligence.local_llm_enricher import KnowledgeEntry
        return [
            KnowledgeEntry(
                id=f"e{i}",
                document=f"doc{i}",
                metadata={"scenario_id": "high_error_rate",
                           "action_taken": "restart_service",
                           "outcome": "success",
                           "similarity": 0.9 - i * 0.05},
            )
            for i in range(n)
        ]

    def _ollama_response(self, payload: dict) -> MagicMock:
        r = MagicMock()
        r.response = json.dumps(payload)
        return r

    @pytest.mark.asyncio
    async def test_corroborated_response(self):
        enricher = _make_enricher()
        similar  = self._make_similar()
        payload  = {
            "validation_status": "corroborated",
            "confidence": 0.93,
            "rca_alignment": "Matches DB pattern",
            "action_alignment": "restart_service confirmed",
            "suggested_adjustment": "",
            "reasoning_summary": "Strong history support",
        }
        mock_client = MagicMock()
        mock_client.generate.return_value = self._ollama_response(payload)
        mock_ollama = MagicMock()
        mock_ollama.Client.return_value = mock_client

        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            result = await enricher.validate_external_result(
                incident_context={"service_name": "frontend-api", "alert_name": "HighErrorRate",
                                  "domain": "compute", "scenario_id": "high_error_rate"},
                external_result={"root_cause": "DB pool exhausted",
                                  "recommended_action": "restart_service",
                                  "confidence": "high", "ansible_description": "restart"},
                similar=similar,
            )

        assert result is not None
        assert result.validation_status == "corroborated"
        assert result.confidence == pytest.approx(0.93)
        assert result.similar_count == 2
        assert result.top_similarity == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_returns_insufficient_context_when_no_similar(self):
        enricher = _make_enricher()

        result = await enricher.validate_external_result(
            incident_context={"service_name": "svc", "alert_name": "Alert",
                              "domain": "compute", "scenario_id": "unknown"},
            external_result={"root_cause": "unknown", "recommended_action": "investigate"},
            similar=[],
        )

        assert result is not None
        assert result.validation_status == "insufficient_context"
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_returns_none_when_local_llm_disabled(self):
        import obs_intelligence.local_llm_enricher as mod
        original = mod._LOCAL_LLM_ENABLED
        mod._LOCAL_LLM_ENABLED = False
        try:
            enricher = _make_enricher()
            result = await enricher.validate_external_result(
                incident_context={},
                external_result={},
                similar=self._make_similar(),
            )
            assert result is None
        finally:
            mod._LOCAL_LLM_ENABLED = original

    @pytest.mark.asyncio
    async def test_invalid_verdict_falls_back_to_insufficient_context(self):
        enricher = _make_enricher()
        similar  = self._make_similar()
        payload  = {"validation_status": "TOTALLY_INVALID", "confidence": 0.5,
                    "rca_alignment": "", "action_alignment": "",
                    "suggested_adjustment": "", "reasoning_summary": ""}
        mock_client = MagicMock()
        mock_client.generate.return_value = self._ollama_response(payload)
        mock_ollama = MagicMock()
        mock_ollama.Client.return_value = mock_client

        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            result = await enricher.validate_external_result(
                incident_context={"service_name": "s", "alert_name": "a",
                                  "domain": "compute", "scenario_id": "x"},
                external_result={"root_cause": "r", "recommended_action": "a"},
                similar=similar,
            )

        assert result is not None
        assert result.validation_status == "insufficient_context"

    @pytest.mark.asyncio
    async def test_returns_none_on_json_decode_error(self):
        enricher = _make_enricher()
        similar  = self._make_similar()
        mock_client = MagicMock()
        bad_resp = MagicMock()
        bad_resp.response = "not json {{ bad"
        mock_client.generate.return_value = bad_resp
        mock_ollama = MagicMock()
        mock_ollama.Client.return_value = mock_client

        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            result = await enricher.validate_external_result(
                incident_context={"service_name": "s", "alert_name": "a",
                                  "domain": "compute", "scenario_id": "x"},
                external_result={"root_cause": "r", "recommended_action": "a"},
                similar=similar,
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_ollama_error(self):
        enricher = _make_enricher()
        similar  = self._make_similar()
        mock_client = MagicMock()
        mock_client.generate.side_effect = ConnectionError("Ollama unreachable")
        mock_ollama = MagicMock()
        mock_ollama.Client.return_value = mock_client

        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            result = await enricher.validate_external_result(
                incident_context={"service_name": "s", "alert_name": "a",
                                  "domain": "compute", "scenario_id": "x"},
                external_result={"root_cause": "r", "recommended_action": "a"},
                similar=similar,
            )

        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# store_incident_resolution
# ─────────────────────────────────────────────────────────────────────────────

class TestStoreIncidentResolution:

    def _make_local_validation(self, status: str = "corroborated"):
        from obs_intelligence.local_llm_enricher import LocalValidationResult
        return LocalValidationResult(
            validation_status=status,
            confidence=0.88,
            reasoning_summary="Supported by history",
            top_similarity=0.91,
            similar_count=2,
        )

    @pytest.mark.asyncio
    async def test_stores_entry_and_returns_run_id(self):
        col      = _mock_collection()
        enricher = _make_enricher(col)

        with patch.object(enricher, "_embed", return_value=[0.1, 0.2]):
            entry_id = await enricher.store_incident_resolution(
                incident_context={
                    "service_name": "frontend-api",
                    "alert_name":   "HighErrorRate",
                    "domain":       "compute",
                    "scenario_id":  "high_error_rate",
                    "description":  "Error rate over 25%",
                    "run_id":       "run-001",
                },
                external_result={
                    "recommended_action": "restart_service",
                    "ansible_description": "Restart service",
                    "provider": "openai",
                },
                local_validation=self._make_local_validation(),
                similar=[],
                outcome="pending",
                run_id="run-001",
            )

        assert entry_id == "run-001"
        col.upsert.assert_called_once()
        upsert_kwargs = col.upsert.call_args[1]
        assert upsert_kwargs["ids"] == ["run-001"]
        meta = upsert_kwargs["metadatas"][0]
        assert meta["service_name"] == "frontend-api"
        assert meta["validation_status"] == "corroborated"
        assert meta["outcome"] == "pending"

    @pytest.mark.asyncio
    async def test_stores_without_embedding_when_embed_returns_none(self):
        col      = _mock_collection()
        enricher = _make_enricher(col)

        with patch.object(enricher, "_embed", return_value=None):
            entry_id = await enricher.store_incident_resolution(
                incident_context={"service_name": "svc", "alert_name": "Alert",
                                  "domain": "compute", "scenario_id": "x"},
                external_result={},
                local_validation=None,
                similar=[],
                run_id="run-002",
            )

        assert entry_id == "run-002"
        upsert_kwargs = col.upsert.call_args[1]
        assert "embeddings" not in upsert_kwargs   # no embedding when Ollama unavailable

    @pytest.mark.asyncio
    async def test_strips_sensitive_keys_from_metadata(self):
        """ansible_playbook, pr_description, etc. must not appear in ChromaDB metadata."""
        col      = _mock_collection()
        enricher = _make_enricher(col)

        with patch.object(enricher, "_embed", return_value=None):
            await enricher.store_incident_resolution(
                incident_context={"service_name": "svc", "alert_name": "Alert",
                                  "domain": "compute", "scenario_id": "x",
                                  "ansible_playbook": "SECRET_PLAYBOOK",
                                  "pr_description": "SECRET_PR"},
                external_result={},
                local_validation=None,
                similar=[],
                run_id="run-strip",
            )

        meta = col.upsert.call_args[1]["metadatas"][0]
        assert "ansible_playbook" not in meta
        assert "pr_description"   not in meta

    @pytest.mark.asyncio
    async def test_returns_none_when_chromadb_unavailable(self):
        enricher = _make_enricher()
        enricher._collection = None

        with patch.object(enricher, "_get_collection", return_value=None):
            result = await enricher.store_incident_resolution(
                incident_context={},
                external_result={},
                local_validation=None,
                similar=[],
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_upsert_error(self):
        col = MagicMock()
        col.upsert.side_effect = Exception("ChromaDB write error")
        enricher = _make_enricher(col)

        with patch.object(enricher, "_embed", return_value=[0.1]):
            result = await enricher.store_incident_resolution(
                incident_context={"service_name": "svc", "alert_name": "a",
                                  "domain": "compute", "scenario_id": "x"},
                external_result={},
                local_validation=None,
                similar=[],
                run_id="run-err",
            )

        assert result is None
        assert enricher._collection is None   # reset on error


# ─────────────────────────────────────────────────────────────────────────────
# update_incident_outcome
# ─────────────────────────────────────────────────────────────────────────────

class TestUpdateIncidentOutcome:

    @pytest.mark.asyncio
    async def test_updates_via_direct_id_lookup(self):
        existing_meta = {"service_name": "svc", "outcome": "pending", "run_id": "run-001"}
        col = _mock_collection(
            get_result=_make_chroma_get_result(["run-001"], metas=[existing_meta])
        )
        enricher = _make_enricher(col)

        ok = await enricher.update_incident_outcome(
            run_id="run-001", outcome="success"
        )

        assert ok is True
        col.update.assert_called_once()
        updated_meta = col.update.call_args[1]["metadatas"][0]
        assert updated_meta["outcome"] == "success"
        assert "outcome_recorded_at" in updated_meta

    @pytest.mark.asyncio
    async def test_falls_back_to_metadata_search_when_id_not_found(self):
        """First get() returns empty → falls back to get(where=run_id)."""
        col = MagicMock()
        col.get.side_effect = [
            {"ids": [], "metadatas": []},                                     # direct ID miss
            _make_chroma_get_result(["other-id"], metas=[{"run_id": "run-002", "outcome": "pending"}]),
        ]
        col.update.return_value = None
        enricher = _make_enricher(col)

        ok = await enricher.update_incident_outcome(run_id="run-002", outcome="failure")

        assert ok is True
        updated_meta = col.update.call_args[1]["metadatas"][0]
        assert updated_meta["outcome"] == "failure"

    @pytest.mark.asyncio
    async def test_returns_false_when_not_found_anywhere(self):
        col = MagicMock()
        col.get.side_effect = [
            {"ids": [], "metadatas": []},   # direct miss
            {"ids": [], "metadatas": []},   # metadata miss
        ]
        enricher = _make_enricher(col)

        ok = await enricher.update_incident_outcome(run_id="ghost-run", outcome="success")

        assert ok is False

    @pytest.mark.asyncio
    async def test_returns_false_when_chromadb_unavailable(self):
        enricher = _make_enricher()
        with patch.object(enricher, "_get_collection", return_value=None):
            ok = await enricher.update_incident_outcome(run_id="r", outcome="success")
        assert ok is False


# ─────────────────────────────────────────────────────────────────────────────
# list_entries
# ─────────────────────────────────────────────────────────────────────────────

class TestListEntries:

    @pytest.mark.asyncio
    async def test_returns_all_entries_without_filters(self):
        ids   = ["a", "b"]
        docs  = ["doc a", "doc b"]
        metas = [{"service_name": "svc1"}, {"service_name": "svc2"}]
        col   = _mock_collection(get_result={"ids": ids, "documents": docs, "metadatas": metas})
        enricher = _make_enricher(col)

        entries = await enricher.list_entries()

        assert len(entries) == 2
        assert entries[0]["id"] == "a"
        assert entries[1]["metadata"]["service_name"] == "svc2"

    @pytest.mark.asyncio
    async def test_applies_service_name_filter(self):
        col = _mock_collection(get_result={"ids": [], "documents": [], "metadatas": []})
        enricher = _make_enricher(col)

        await enricher.list_entries(service_name="frontend-api")

        call_kwargs = col.get.call_args[1]
        assert call_kwargs["where"] == {"service_name": "frontend-api"}

    @pytest.mark.asyncio
    async def test_applies_combined_filters(self):
        col = _mock_collection(get_result={"ids": [], "documents": [], "metadatas": []})
        enricher = _make_enricher(col)

        await enricher.list_entries(service_name="svc", scenario_id="high_error_rate")

        call_kwargs = col.get.call_args[1]
        assert call_kwargs["where"] == {
            "$and": [{"service_name": "svc"}, {"scenario_id": "high_error_rate"}]
        }

    @pytest.mark.asyncio
    async def test_returns_empty_when_chromadb_unavailable(self):
        enricher = _make_enricher()
        with patch.object(enricher, "_get_collection", return_value=None):
            entries = await enricher.list_entries()
        assert entries == []


# ─────────────────────────────────────────────────────────────────────────────
# knowledge_stats
# ─────────────────────────────────────────────────────────────────────────────

class TestKnowledgeStats:

    def _make_entries(self) -> list[dict]:
        return [
            {"validation_status": "corroborated", "outcome": "success", "top_similarity": 0.92},
            {"validation_status": "corroborated", "outcome": "success", "top_similarity": 0.88},
            {"validation_status": "weak_support",  "outcome": "failure", "top_similarity": 0.75},
            {"validation_status": "divergent",     "outcome": "pending", "top_similarity": 0.0},
            {"validation_status": "unavailable",   "outcome": "pending", "top_similarity": 0.0},
        ]

    @pytest.mark.asyncio
    async def test_returns_correct_counts(self):
        metas = self._make_entries()
        col = _mock_collection(
            get_result={"ids": [str(i) for i in range(5)],
                        "documents": [""] * 5,
                        "metadatas": metas}
        )
        enricher = _make_enricher(col)

        stats = await enricher.knowledge_stats()

        assert stats["chroma_available"] is True
        assert stats["knowledge_entries_total"] == 5
        assert stats["corroborated_count"] == 2
        assert stats["weak_support_count"] == 1
        assert stats["divergent_count"] == 1
        assert stats["unavailable_count"] == 1
        assert stats["success_outcome_count"] == 2
        assert stats["failure_outcome_count"] == 1
        assert stats["pending_outcome_count"] == 2

    @pytest.mark.asyncio
    async def test_corroboration_rate(self):
        metas = self._make_entries()
        col = _mock_collection(
            get_result={"ids": [str(i) for i in range(5)],
                        "documents": [""] * 5,
                        "metadatas": metas}
        )
        enricher = _make_enricher(col)

        stats = await enricher.knowledge_stats()

        # 2 corroborated out of 4 validated (unavailable not counted as validated)
        assert stats["corroboration_rate_pct"] == pytest.approx(50.0, abs=0.1)

    @pytest.mark.asyncio
    async def test_returns_empty_stats_when_no_entries(self):
        col = _mock_collection(get_result={"ids": [], "documents": [], "metadatas": []})
        enricher = _make_enricher(col)

        stats = await enricher.knowledge_stats()

        assert stats["knowledge_entries_total"] == 0
        assert stats["corroboration_rate_pct"] == 0.0
        assert stats["chroma_available"] is True

    @pytest.mark.asyncio
    async def test_reports_chroma_unavailable(self):
        enricher = _make_enricher()
        with patch.object(enricher, "_get_collection", return_value=None):
            stats = await enricher.knowledge_stats()

        assert stats["chroma_available"] is False
        assert stats["knowledge_entries_total"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# _get_collection — lazy connection behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestGetCollection:

    def test_returns_none_when_chromadb_import_fails(self):
        from obs_intelligence.local_llm_enricher import LocalLLMEnricher
        enricher = LocalLLMEnricher()   # fresh, no pre-set collection

        with patch.dict("sys.modules", {"chromadb": None}):
            result = enricher._get_collection()

        assert result is None

    def test_caches_collection_after_first_connect(self):
        """Second call should return the cached collection without reconnecting."""
        from obs_intelligence.local_llm_enricher import LocalLLMEnricher
        enricher = LocalLLMEnricher()

        mock_col    = MagicMock()
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_col

        mock_chroma = MagicMock()
        mock_chroma.HttpClient.return_value = mock_client

        with patch.dict("sys.modules", {"chromadb": mock_chroma}):
            c1 = enricher._get_collection()
            c2 = enricher._get_collection()

        assert c1 is c2
        # HttpClient only called once (cached after first)
        mock_chroma.HttpClient.assert_called_once()
