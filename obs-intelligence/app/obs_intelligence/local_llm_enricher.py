"""
obs_intelligence/local_llm_enricher.py
────────────────────────────────────────────────────────────────────────────────
Block F — Local LLM validator and ChromaDB knowledge-store adapter.

Architecture
────────────
  External LLM always runs first and its result is AUTHORITATIVE.

  After the external result is available, this module:
    1. Queries ChromaDB for the top-K most similar historical incidents
       (using nomic-embed-text embeddings via Ollama).
    2. Asks the local LLM (qwen3.5) whether the external result is
       corroborated, weakly supported, or divergent from history.
    3. Stores the incident + validation metadata in ChromaDB for future
       similarity retrieval.

  All calls are best-effort and bounded by caller-supplied timeouts.
  ChromaDB / Ollama failures NEVER block the pipeline — the external
  result is returned unchanged.

Security
────────
  Only operational facts are stored.  The following are never stored:
    ansible_playbook, pr_description, pr_title, rollback_steps,
    test_cases, test_plan, raw LLM output, API keys, tokens, PII.

Environment variables
─────────────────────
  CHROMA_URL              ChromaDB server URL    (default: http://knowledge-store:8000)
  LOCAL_LLM_URL           Ollama server URL      (default: http://local-llm:11434)
  LOCAL_LLM_MODEL         Ollama model name      (default: qwen3.5)
  LOCAL_LLM_ENABLED       true/false             (default: true)
  LOCAL_LLM_MIN_SIMILARITY  cosine sim threshold (default: 0.82)
  LOCAL_LLM_TOP_K         max similar entries    (default: 5)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger("obs_intelligence.local_llm_enricher")

_CHROMA_URL          = os.getenv("CHROMA_URL",       "http://knowledge-store:8000")
_LOCAL_LLM_URL       = os.getenv("LOCAL_LLM_URL",    "http://local-llm:11434")
_LOCAL_LLM_MODEL     = os.getenv("LOCAL_LLM_MODEL",  "qwen3.5")
_LOCAL_LLM_ENABLED   = os.getenv("LOCAL_LLM_ENABLED", "true").lower() == "true"
_LOCAL_LLM_MIN_SIM   = float(os.getenv("LOCAL_LLM_MIN_SIMILARITY", "0.82"))
_LOCAL_LLM_TOP_K     = int(os.getenv("LOCAL_LLM_TOP_K", "5"))
_EMBED_MODEL         = "nomic-embed-text"
_COLLECTION_NAME     = "aiops_incidents"

_ALLOWED_VERDICTS = frozenset({
    "corroborated",
    "weak_support",
    "divergent",
    "insufficient_context",
    "unavailable",
})

# Fields that must never be persisted (may contain secrets or large payloads)
_STRIP_KEYS = frozenset({
    "ansible_playbook", "pr_description", "pr_title",
    "rollback_steps", "test_cases", "test_plan", "raw",
})


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KnowledgeEntry:
    """A single ChromaDB incident record with its similarity score."""
    id: str
    document: str
    metadata: dict = field(default_factory=dict)

    def similarity(self) -> float:
        return float(self.metadata.get("similarity", 0.0))


@dataclass
class LocalValidationResult:
    """
    Advisory validation result produced by the local LLM.

    The external LLM result is always authoritative.  This dataclass records
    whether the local LLM agrees with (corroborates), weakly supports, or
    diverges from the external result given similar historical incidents.
    """
    validation_status: str = "unavailable"
    confidence: float = 0.0
    rca_alignment: str = ""
    action_alignment: str = ""
    suggested_adjustment: str = ""
    reasoning_summary: str = ""
    top_similarity: float = 0.0
    similar_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class LocalLLMEnricher:
    """
    ChromaDB-backed incident knowledge store + Ollama local LLM validator.

    Provides four public async methods used by llm_enricher.py after each
    external LLM call and by obs-intelligence/main.py for the Learning tab:

      query_similar_incidents(...)   → list[KnowledgeEntry]
      validate_external_result(...)  → LocalValidationResult | None
      store_incident_resolution(...) → entry_id | None
      update_incident_outcome(...)   → bool
      list_entries(...)              → list[dict]
      knowledge_stats()              → dict
    """

    def __init__(self) -> None:
        self._collection = None   # lazy ChromaDB connection

    # ── ChromaDB helpers ──────────────────────────────────────────────────────

    def _get_collection(self):
        """Lazily connect to ChromaDB.  Returns None if unavailable."""
        if self._collection is not None:
            return self._collection
        try:
            import chromadb  # type: ignore[import]
            p = urlparse(_CHROMA_URL)
            client = chromadb.HttpClient(
                host=p.hostname or "knowledge-store",
                port=p.port or 8000,
            )
            self._collection = client.get_or_create_collection(
                _COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("Connected to ChromaDB collection '%s'", _COLLECTION_NAME)
        except Exception as exc:
            logger.warning("ChromaDB connection failed: %s", exc)
            return None
        return self._collection

    def _reset_collection(self) -> None:
        """Force reconnect on next access (e.g. after ChromaDB restart)."""
        self._collection = None

    # ── Ollama embedding ──────────────────────────────────────────────────────

    def _embed(self, text: str) -> list[float] | None:
        """
        Compute an embedding vector using nomic-embed-text via Ollama (sync).
        Returns None if Ollama is unavailable or disabled.
        """
        if not _LOCAL_LLM_ENABLED:
            return None
        try:
            import ollama  # type: ignore[import]
            resp = ollama.Client(host=_LOCAL_LLM_URL).embeddings(
                model=_EMBED_MODEL, prompt=text
            )
            # ollama SDK v0.1 returns dict; v0.2+ returns a typed object
            return resp.get("embedding") if isinstance(resp, dict) else list(resp.embedding)
        except Exception as exc:
            logger.warning("Ollama embedding failed: %s", exc)
            return None

    # ── Public API ────────────────────────────────────────────────────────────

    async def query_similar_incidents(
        self,
        incident_text: str,
        domain: str = "",
        limit: int | None = None,
    ) -> list[KnowledgeEntry]:
        """
        Retrieve the most similar historical incidents from ChromaDB.

        Uses nomic-embed-text embeddings and cosine distance.
        Entries below LOCAL_LLM_MIN_SIMILARITY are filtered out.
        Returns [] when ChromaDB or Ollama is unavailable.
        """
        collection = self._get_collection()
        if collection is None:
            return []

        max_k = limit or _LOCAL_LLM_TOP_K
        loop  = asyncio.get_event_loop()
        try:
            embedding = await loop.run_in_executor(None, lambda: self._embed(incident_text))
            if embedding is None:
                return []

            query_kwargs: dict[str, Any] = {
                "query_embeddings": [embedding],
                "n_results": max_k,
                "include": ["documents", "metadatas", "distances"],
            }
            if domain:
                query_kwargs["where"] = {"domain": domain}

            results = await loop.run_in_executor(
                None, lambda: collection.query(**query_kwargs)
            )
            ids   = (results.get("ids")       or [[]])[0]
            docs  = (results.get("documents") or [[]])[0]
            metas = (results.get("metadatas") or [[]])[0]
            dists = (results.get("distances") or [[]])[0]

            entries: list[KnowledgeEntry] = []
            for eid, doc, meta, dist in zip(ids, docs, metas, dists):
                similarity = max(0.0, 1.0 - float(dist))  # cosine dist → similarity
                if similarity < _LOCAL_LLM_MIN_SIM:
                    continue
                entries.append(KnowledgeEntry(
                    id=eid,
                    document=doc or "",
                    metadata={**meta, "similarity": round(similarity, 4)},
                ))
            entries.sort(key=lambda e: e.similarity(), reverse=True)
            return entries

        except Exception as exc:
            self._reset_collection()
            logger.warning("ChromaDB query failed (will retry on next call): %s", exc)
            return []

    async def validate_external_result(
        self,
        *,
        incident_context: dict[str, Any],
        external_result: Any,
        similar: list[KnowledgeEntry],
    ) -> LocalValidationResult | None:
        """
        Ask the local LLM to validate/corroborate the external LLM result.

        Returns None if the local LLM is disabled or unavailable.
        The external result is ALWAYS authoritative — this is purely advisory.
        """
        if not _LOCAL_LLM_ENABLED:
            return None

        if not similar:
            return LocalValidationResult(
                validation_status="insufficient_context",
                confidence=0.0,
                reasoning_summary="No similar historical incidents found to validate against",
            )

        similar_summaries = "\n".join(
            f"  [{i + 1}] scenario={e.metadata.get('scenario_id', '?')}  "
            f"action={e.metadata.get('action_taken', '?')}  "
            f"outcome={e.metadata.get('outcome', '?')}  "
            f"similarity={e.similarity():.2f}"
            for i, e in enumerate(similar[:5])
        )

        def _get(obj: Any, key: str, default: str = "") -> str:
            if hasattr(obj, key):
                return str(getattr(obj, key) or default)
            if isinstance(obj, dict):
                return str(obj.get(key) or default)
            return default

        root_cause = _get(external_result, "root_cause") or _get(external_result, "rca_summary")

        prompt = (
            "You are a senior SRE validating an AI-generated incident analysis.\n\n"
            "INCIDENT CONTEXT\n"
            + json.dumps(
                {k: v for k, v in incident_context.items() if k != "description"},
                indent=2,
            )
            + "\n\nEXTERNAL LLM ANALYSIS (authoritative — do not override)\n"
            f"  Root cause:          {root_cause}\n"
            f"  Recommended action:  {_get(external_result, 'recommended_action')}\n"
            f"  Confidence:          {_get(external_result, 'confidence')}\n"
            f"  Ansible description: {_get(external_result, 'ansible_description')}\n\n"
            f"SIMILAR HISTORICAL INCIDENTS (top {len(similar)})\n"
            f"{similar_summaries}\n\n"
            "TASK\n"
            "Compare the external analysis against the historical incidents.\n"
            "The external LLM result is authoritative. Your job is only to\n"
            "assess whether history corroborates or diverges from it.\n\n"
            "Return ONLY valid JSON with no markdown fences:\n"
            "{\n"
            '  "validation_status": "corroborated|weak_support|divergent|insufficient_context",\n'
            '  "confidence": 0.0,\n'
            '  "rca_alignment": "brief alignment assessment",\n'
            '  "action_alignment": "brief action alignment assessment",\n'
            '  "suggested_adjustment": "optional adjustment or empty string",\n'
            '  "reasoning_summary": "1-2 sentence reasoning"\n'
            "}\n"
        )

        loop = asyncio.get_event_loop()
        try:
            import ollama  # type: ignore[import]
            client = ollama.Client(host=_LOCAL_LLM_URL)
            resp = await loop.run_in_executor(
                None,
                lambda: client.generate(
                    model=_LOCAL_LLM_MODEL,
                    prompt=prompt,
                    format="json",
                    options={"temperature": 0.1},
                ),
            )
            raw_text = (
                resp.get("response", "{}") if isinstance(resp, dict) else (resp.response or "{}")
            )
            parsed = json.loads(raw_text)
            status = str(parsed.get("validation_status", "insufficient_context"))
            if status not in _ALLOWED_VERDICTS:
                status = "insufficient_context"
            top_sim = similar[0].similarity() if similar else 0.0
            return LocalValidationResult(
                validation_status=status,
                confidence=float(parsed.get("confidence") or 0.0),
                rca_alignment=str(parsed.get("rca_alignment", "")),
                action_alignment=str(parsed.get("action_alignment", "")),
                suggested_adjustment=str(parsed.get("suggested_adjustment", "")),
                reasoning_summary=str(parsed.get("reasoning_summary", "")),
                top_similarity=top_sim,
                similar_count=len(similar),
            )
        except json.JSONDecodeError as exc:
            logger.warning("Local LLM returned invalid JSON: %s", exc)
            return None
        except Exception as exc:
            logger.warning("Local LLM generate failed: %s", exc)
            return None

    async def store_incident_resolution(
        self,
        *,
        incident_context: dict[str, Any],
        external_result: Any,
        local_validation: LocalValidationResult | None,
        similar: list[KnowledgeEntry],
        outcome: str = "pending",
        run_id: str = "",
    ) -> str | None:
        """
        Persist an incident resolution to ChromaDB.

        Only operational facts are stored — never secrets, playbooks, or PII.
        Returns the ChromaDB entry ID on success, None on failure.
        """
        collection = self._get_collection()
        if collection is None:
            return None

        service_name = str(incident_context.get("service_name", ""))
        alert_name   = str(incident_context.get("alert_name", ""))
        domain       = str(incident_context.get("domain", ""))
        scenario_id  = str(incident_context.get("scenario_id", ""))
        description  = str(incident_context.get("description", ""))

        doc_text = description or (
            f"Service: {service_name} Alert: {alert_name} "
            f"Domain: {domain} Scenario: {scenario_id}"
        )

        def _get(obj: Any, key: str, default: Any = "") -> Any:
            if hasattr(obj, key):
                return getattr(obj, key) or default
            if isinstance(obj, dict):
                return obj.get(key) or default
            return default

        action_taken   = str(_get(external_result, "recommended_action"))
        ansible_desc   = str(_get(external_result, "ansible_description"))
        external_model = str(_get(external_result, "external_model"))
        provider       = str(_get(external_result, "provider", "external_llm"))

        metadata: dict[str, Any] = {
            "service_name":      service_name,
            "alert_name":        alert_name,
            "domain":            domain,
            "scenario_id":       scenario_id,
            "action_taken":      action_taken or ansible_desc,
            "outcome":           outcome,
            "autonomy_decision": str(incident_context.get("autonomy_decision", "")),
            "risk_score":        float(incident_context.get("risk_score") or 0.0),
            "validation_status": (
                local_validation.validation_status if local_validation else "unavailable"
            ),
            "external_source":   provider,
            "external_model":    external_model,
            "local_model":       _LOCAL_LLM_MODEL if _LOCAL_LLM_ENABLED else "",
            "top_similarity":    float(similar[0].similarity() if similar else 0.0),
            "run_id":            run_id or str(incident_context.get("run_id", "")),
            "timestamp":         datetime.now(timezone.utc).isoformat(),
        }

        entry_id = run_id or str(uuid.uuid4())
        loop = asyncio.get_event_loop()
        try:
            embedding = await loop.run_in_executor(None, lambda: self._embed(doc_text))

            upsert_kwargs: dict[str, Any] = {
                "ids":       [entry_id],
                "documents": [doc_text],
                "metadatas": [metadata],
            }
            if embedding is not None:
                upsert_kwargs["embeddings"] = [embedding]

            await loop.run_in_executor(None, lambda: collection.upsert(**upsert_kwargs))
            logger.info(
                "Stored ChromaDB entry  id=%s  service=%s  alert=%s  validation=%s",
                entry_id, service_name, alert_name, metadata["validation_status"],
            )
            return entry_id

        except Exception as exc:
            self._reset_collection()
            logger.warning("ChromaDB store failed (will retry on next call): %s", exc)
            return None

    async def update_incident_outcome(
        self,
        *,
        run_id: str,
        outcome: str,
        service_name: str = "",
        alert_name: str = "",
    ) -> bool:
        """
        Update the outcome field of an existing ChromaDB incident entry.

        Tries direct ID lookup first, then falls back to metadata run_id search.
        Returns True if an entry was updated, False otherwise.
        """
        collection = self._get_collection()
        if collection is None:
            return False

        loop = asyncio.get_event_loop()
        now  = datetime.now(timezone.utc).isoformat()
        try:
            # Direct ID lookup (run_id is used as the entry ID on store)
            result = await loop.run_in_executor(
                None, lambda: collection.get(ids=[run_id], include=["metadatas"])
            )
            if result and result.get("ids"):
                meta = dict((result["metadatas"] or [{}])[0])
                meta["outcome"] = outcome
                meta["outcome_recorded_at"] = now
                await loop.run_in_executor(
                    None, lambda: collection.update(ids=[run_id], metadatas=[meta])
                )
                logger.info("Updated outcome  id=%s  outcome=%s", run_id, outcome)
                return True

            # Fallback: search by run_id metadata field
            results = await loop.run_in_executor(
                None,
                lambda: collection.get(
                    where={"run_id": run_id}, include=["metadatas"]
                ),
            )
            if not results or not results.get("ids"):
                return False

            entry_id = results["ids"][0]
            meta = dict((results["metadatas"] or [{}])[0])
            meta["outcome"] = outcome
            meta["outcome_recorded_at"] = now
            await loop.run_in_executor(
                None, lambda: collection.update(ids=[entry_id], metadatas=[meta])
            )
            logger.info("Updated outcome  id=%s  run_id=%s  outcome=%s", entry_id, run_id, outcome)
            return True

        except Exception as exc:
            self._reset_collection()
            logger.warning("ChromaDB update_outcome failed: %s", exc)
            return False

    async def list_entries(
        self,
        *,
        service_name: str = "",
        scenario_id: str = "",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        List knowledge entries from ChromaDB with optional filters.
        Used by GET /intelligence/knowledge-entries.
        """
        collection = self._get_collection()
        if collection is None:
            return []

        loop = asyncio.get_event_loop()
        try:
            where: dict[str, Any] = {}
            if service_name and scenario_id:
                where = {"$and": [
                    {"service_name": service_name},
                    {"scenario_id": scenario_id},
                ]}
            elif service_name:
                where = {"service_name": service_name}
            elif scenario_id:
                where = {"scenario_id": scenario_id}

            get_kwargs: dict[str, Any] = {
                "include": ["documents", "metadatas"],
                "limit":   limit,
            }
            if where:
                get_kwargs["where"] = where

            results = await loop.run_in_executor(
                None, lambda: collection.get(**get_kwargs)
            )
            ids   = results.get("ids")       or []
            docs  = results.get("documents") or []
            metas = results.get("metadatas") or []
            return [
                {"id": eid, "document": doc, "metadata": meta}
                for eid, doc, meta in zip(ids, docs, metas)
            ]

        except Exception as exc:
            self._reset_collection()
            logger.warning("ChromaDB list_entries failed: %s", exc)
            return []

    async def knowledge_stats(self) -> dict[str, Any]:
        """
        Compute learning statistics from ChromaDB.
        Used by GET /intelligence/learning-stats.
        """
        collection = self._get_collection()
        if collection is None:
            return {"chroma_available": False, "knowledge_entries_total": 0}

        loop = asyncio.get_event_loop()
        try:
            all_entries = await loop.run_in_executor(
                None, lambda: collection.get(include=["metadatas"])
            )
            metas: list[dict] = all_entries.get("metadatas") or []
            total = len(metas)
            if total == 0:
                return {
                    "chroma_available": True,
                    "knowledge_entries_total": 0,
                    "corroborated_count": 0,
                    "weak_support_count": 0,
                    "divergent_count": 0,
                    "insufficient_context_count": 0,
                    "unavailable_count": 0,
                    "success_outcome_count": 0,
                    "failure_outcome_count": 0,
                    "pending_outcome_count": 0,
                    "avg_top_similarity": 0.0,
                    "corroboration_rate_pct": 0.0,
                    "local_validation_coverage_pct": 0.0,
                }

            status_counts:  dict[str, int] = {}
            outcome_counts: dict[str, int] = {}
            similarities:   list[float]    = []

            for m in metas:
                vs = str(m.get("validation_status", "unavailable"))
                status_counts[vs] = status_counts.get(vs, 0) + 1
                oc = str(m.get("outcome", "pending"))
                outcome_counts[oc] = outcome_counts.get(oc, 0) + 1
                ts = float(m.get("top_similarity", 0.0))
                if ts > 0:
                    similarities.append(ts)

            corroborated = status_counts.get("corroborated", 0)
            weak_support = status_counts.get("weak_support", 0)
            divergent    = status_counts.get("divergent", 0)
            insuff_ctx   = status_counts.get("insufficient_context", 0)
            unavailable  = status_counts.get("unavailable", 0)
            validated    = corroborated + weak_support + divergent + insuff_ctx
            coverage_pct = round((validated / total) * 100, 1) if total else 0.0
            corr_pct     = round((corroborated / validated) * 100, 1) if validated else 0.0
            avg_sim      = round(sum(similarities) / len(similarities), 4) if similarities else 0.0

            return {
                "chroma_available":               True,
                "knowledge_entries_total":         total,
                "corroborated_count":              corroborated,
                "weak_support_count":              weak_support,
                "divergent_count":                 divergent,
                "insufficient_context_count":      insuff_ctx,
                "unavailable_count":               unavailable,
                "success_outcome_count":           outcome_counts.get("success", 0),
                "failure_outcome_count":           outcome_counts.get("failure", 0),
                "pending_outcome_count":           outcome_counts.get("pending", 0),
                "avg_top_similarity":              avg_sim,
                "corroboration_rate_pct":          corr_pct,
                "local_validation_coverage_pct":   coverage_pct,
            }

        except Exception as exc:
            self._reset_collection()
            logger.warning("ChromaDB knowledge_stats failed: %s", exc)
            return {"chroma_available": False, "knowledge_entries_total": 0}


# ─────────────────────────────────────────────────────────────────────────────
# Module singleton
# ─────────────────────────────────────────────────────────────────────────────

local_llm_enricher = LocalLLMEnricher()
