# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: LOCAL LLM LEARNING LAYER — Architecture & Implementation Prompt
# ═══════════════════════════════════════════════════════════════════════════════

## Executive Summary

Currently, the AIOps agents are **NOT learning**. Each incident triggers a fresh
GPT-4 API call with no memory of previous incidents. This means:

- ❌ **No knowledge retention** — Same error seen 100 times = 100 GPT-4 API calls
- ❌ **No cost optimization** — Every incident incurs API costs
- ❌ **No speed improvement** — Every analysis takes 3-10 seconds
- ❌ **No offline capability** — Internet/API outage = system degraded

This document defines a **Local LLM Learning Layer** that enables agents to:

- ✅ **Learn from history** — Store incident → root cause → remediation knowledge
- ✅ **Reduce API costs** — Only call GPT-4 for NEW problems
- ✅ **Accelerate response** — Known issues resolve in <500ms (vs 3-10s)
- ✅ **Work offline** — Local inference when external LLM unavailable

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  INCIDENT ALERT (Prometheus → Alertmanager → Compute Agent) │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  Agent 1-3: Data Collection │
        │  (Logs, Metrics, Traces)    │
        └────────────┬─────────────────┘
                     │
                     ▼
        ┌────────────────────────────────────────┐
        │  Agent 4: Root Cause Analysis          │
        │  ┌──────────────────────────────────┐  │
        │  │ NEW: Knowledge Agent (Intercept) │  │
        │  └──────────────┬───────────────────┘  │
        └─────────────────┼──────────────────────┘
                          │
         ┌────────────────┴────────────────┐
         │                                  │
         ▼                                  ▼
┌──────────────────────┐        ┌───────────────────────┐
│  LOCAL LLM LAYER     │        │  EXTERNAL LLM (GPT-4) │
│  (Ollama + Vector DB)│        │  (Fallback for NEW)   │
└──────────┬───────────┘        └───────────┬───────────┘
           │                                 │
           │ ┌───────────────────────────────┘
           ▼ ▼
    ┌──────────────────┐
    │  Unified Response │
    │  (RCA + Action)   │
    └─────────┬─────────┘
              │
              ▼
    ┌─────────────────────────┐
    │ Agent 5-6: Remediation  │
    │ + Approval/Execution    │
    └─────────────────────────┘
              │
              ▼
    ┌─────────────────────────┐
    │ FEEDBACK LOOP           │
    │ (Store outcome → Learn) │
    └─────────────────────────┘
```

---

## Component Breakdown

### 1. Knowledge Agent Service (NEW Microservice)

**Service Name:** `knowledge-agent`  
**Port:** `:9002`  
**Tech Stack:** FastAPI + Ollama + ChromaDB/Qdrant  

**Responsibilities:**
- Intercept Agent 4 (analyze) requests BEFORE external LLM
- Query vector database for similar past incidents
- If match found (similarity > 0.85) → return cached knowledge
- If no match → forward to GPT-4, then store response
- Continuous learning via feedback loop

**Endpoints:**

```python
POST /knowledge/query
  Request:
    {
      "service_name": "frontend-api",
      "alert_name": "HighErrorRate",
      "symptoms": {
        "error_rate": 0.45,
        "status_code": 500,
        "logs": "DatabaseConnectionError: connection pool exhausted"
      },
      "metrics": {...},
      "context": "..."
    }
  
  Response (from local knowledge):
    {
      "source": "local",
      "confidence": 0.92,
      "match_count": 7,
      "root_cause": "Database connection pool exhaustion",
      "recommended_action": "restart_service",
      "remediation": "curl -X POST ansible-runner:8090/playbook/restart-db-pool",
      "avg_resolution_time_seconds": 45,
      "success_rate": 0.95
    }
  
  Response (no match, called external LLM):
    {
      "source": "external",
      "provider": "gpt-4o",
      "root_cause": "...",
      "recommended_action": "...",
      "remediation": "...",
      "stored": true  // saved to local DB for future
    }

POST /knowledge/learn
  Request:
    {
      "incident_id": "uuid",
      "service_name": "frontend-api",
      "alert_name": "HighErrorRate",
      "symptoms": {...},
      "root_cause": "...",
      "action_taken": "restart_service",
      "outcome": "success",
      "resolution_time_seconds": 42
    }
  
  Response:
    {
      "status": "learned",
      "incident_id": "uuid",
      "embeddings_updated": true,
      "similar_incidents_count": 8
    }

GET /knowledge/stats
  Response:
    {
      "total_incidents": 1547,
      "unique_scenarios": 34,
      "local_hit_rate": 0.73,  // 73% served from local DB
      "avg_local_response_ms": 320,
      "avg_external_response_ms": 4200,
      "cost_saved_usd": 247.50  // based on GPT-4 API pricing
    }

GET /knowledge/search?query=database+connection+pool
  Response:
    {
      "results": [
        {
          "incident_id": "...",
          "service_name": "...",
          "alert_name": "...",
          "root_cause": "...",
          "similarity": 0.94,
          "occurrences": 12,
          "last_seen": "2026-03-20T10:15:00Z"
        }
      ]
    }
```

### 2. Local LLM (Ollama)

**Service Name:** `ollama`  
**Image:** `ollama/ollama:latest`  
**Model:** `llama3:8b` or `mistral:7b`  
**Port:** `:11434`

**Purpose:**
- Run lightweight LLM locally for fast inference
- Handle structured knowledge queries (known patterns)
- Augment vector DB results with contextual reasoning

**Use Cases:**
- Reformatting historical RCAs for current context
- Explaining stored remediations in plain English
- Generating ticket summaries from cached knowledge

### 3. Vector Database (ChromaDB or Qdrant)

**Service Name:** `vector-db`  
**Tech:** ChromaDB (simpler) or Qdrant (production-grade)  
**Port:** `:6333` (Qdrant) or embedded (ChromaDB)

**Schema:**

```python
Collection: "incidents"

Document:
  - id: uuid
  - embedding: [768-dim vector from sentence-transformers]
  - metadata:
      - service_name: str
      - alert_name: str
      - severity: str
      - symptoms_text: str  // concatenated logs + metrics summary
      - root_cause: str
      - recommended_action: str
      - remediation_cmd: str
      - outcome: str  // success | failed | partial
      - resolution_time_seconds: int
      - timestamp: datetime
      - tier: str  // production | staging | dev
```

**Embedding Model:**
- `all-MiniLM-L6-v2` (384-dim, fast, good for short texts)
- OR `all-mpnet-base-v2` (768-dim, better quality)

**Similarity Search:**
```python
# Query embedding
query_text = f"{alert_name} {symptoms_text}"
query_embedding = embedding_model.encode(query_text)

# Search vector DB
results = vector_db.query(
    query_embeddings=[query_embedding],
    n_results=5,
    where={"service_name": service_name}  # filter by service
)

# Threshold: only use if similarity > 0.85
if results[0]['distance'] < 0.15:  # cosine distance
    return cached_knowledge
else:
    call_external_llm()
```

### 4. Integration into Existing Pipeline

**Modified `compute-agent/app/ai_analyst.py`:**

```python
async def analyze_with_ai(
    service_name: str,
    alert_name: str,
    symptoms: dict,
    metrics: dict,
    logs: str,
) -> dict:
    """
    NEW FLOW:
    1. Query knowledge-agent
    2. If local knowledge found → use it (fast path)
    3. If not found → call GPT-4 (slow path) → store result
    """
    
    # Step 1: Query local knowledge
    knowledge_resp = await httpx.post(
        "http://knowledge-agent:9002/knowledge/query",
        json={
            "service_name": service_name,
            "alert_name": alert_name,
            "symptoms": symptoms,
            "metrics": metrics,
            "context": logs[:2000],  # truncate
        },
        timeout=2.0,  // fast timeout — local should be <500ms
    )
    
    if knowledge_resp.status_code == 200:
        result = knowledge_resp.json()
        
        if result["source"] == "local":
            logger.info(
                "Local knowledge hit  service=%s  alert=%s  confidence=%.2f",
                service_name, alert_name, result["confidence"]
            )
            return {
                "source": "local",
                "root_cause": result["root_cause"],
                "action": result["recommended_action"],
                "remediation": result["remediation"],
                "confidence": result["confidence"],
                "match_count": result["match_count"],
            }
    
    # Step 2: Fallback to external LLM (GPT-4)
    logger.info("Local knowledge miss — calling external LLM")
    external_result = await call_openai_gpt4(
        service_name, alert_name, symptoms, metrics, logs
    )
    
    # Step 3: Store for future learning
    await httpx.post(
        "http://knowledge-agent:9002/knowledge/learn",
        json={
            "incident_id": uuid.uuid4().hex,
            "service_name": service_name,
            "alert_name": alert_name,
            "symptoms": symptoms,
            "root_cause": external_result["root_cause"],
            "action_taken": external_result["action"],
            "outcome": "pending",  # updated after execution
        },
        timeout=1.0,
    )
    
    return {
        "source": "external",
        "provider": "gpt-4o",
        **external_result,
    }
```

**Modified `compute-agent/app/approval_workflow.py`:**

After execution, send feedback:

```python
async def execute_autonomous(session: PipelineSession):
    # ... existing execution logic ...
    
    outcome = "success" if execution_succeeded else "failed"
    
    # Send feedback to knowledge agent
    await httpx.post(
        "http://knowledge-agent:9002/knowledge/learn",
        json={
            "incident_id": session.session_id,
            "service_name": session.service_name,
            "alert_name": session.alert_name,
            "outcome": outcome,
            "resolution_time_seconds": time.time() - session.created_at,
        },
        timeout=1.0,
    )
```

---

## Deployment — Docker Compose

Add to `docker-compose.yml`:

```yaml
  # ──────────────────────────────────────────────────────────
  # Knowledge Agent — Local LLM Learning Layer
  # ──────────────────────────────────────────────────────────
  knowledge-agent:
    build:
      context: ./knowledge-agent
      dockerfile: Dockerfile
    container_name: knowledge-agent
    hostname: knowledge-agent
    environment:
      OLLAMA_URL: "http://ollama:11434"
      VECTOR_DB_URL: "http://vector-db:6333"
      EXTERNAL_LLM_URL: "http://compute-agent:9000"
      SIMILARITY_THRESHOLD: "0.85"
      EMBEDDING_MODEL: "all-MiniLM-L6-v2"
    ports:
      - "9002:9002"
    volumes:
      - knowledge-data:/data
    depends_on:
      - ollama
      - vector-db
    networks:
      - obs-net
    restart: unless-stopped

  # ──────────────────────────────────────────────────────────
  # Ollama — Local LLM Runtime
  # ──────────────────────────────────────────────────────────
  ollama:
    image: ollama/ollama:latest
    container_name: ollama
    hostname: ollama
    volumes:
      - ollama-models:/root/.ollama
    environment:
      OLLAMA_KEEP_ALIVE: "24h"
    ports:
      - "11434:11434"
    networks:
      - obs-net
    restart: unless-stopped
    # Post-startup: docker exec ollama ollama pull llama3:8b

  # ──────────────────────────────────────────────────────────
  # Vector Database — Qdrant (or use ChromaDB embedded)
  # ──────────────────────────────────────────────────────────
  vector-db:
    image: qdrant/qdrant:latest
    container_name: vector-db
    hostname: vector-db
    volumes:
      - vector-db-data:/qdrant/storage
    ports:
      - "6333:6333"    # REST API
      - "6334:6334"    # gRPC (optional)
    networks:
      - obs-net
    restart: unless-stopped

volumes:
  knowledge-data:
    driver: local
  ollama-models:
    driver: local
  vector-db-data:
    driver: local
```

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1)
- ✅ Set up Ollama container + pull `llama3:8b`
- ✅ Set up Qdrant/ChromaDB container
- ✅ Create `knowledge-agent` FastAPI skeleton
- ✅ Implement `/knowledge/query` with passthrough to external LLM
- ✅ Test end-to-end: Alert → Knowledge Agent → GPT-4 → Response

### Phase 2: Learning Loop (Week 2)
- ✅ Implement vector embedding pipeline (sentence-transformers)
- ✅ Store incident → RCA in vector DB with metadata
- ✅ Implement similarity search (cosine distance < 0.15)
- ✅ Return cached knowledge when match found
- ✅ Add `/knowledge/learn` endpoint
- ✅ Integrate feedback loop from approval workflow

### Phase 3: Local LLM Augmentation (Week 3)
- ✅ Use Ollama for contextual reformatting of cached knowledge
- ✅ Generate natural language summaries from vector DB results
- ✅ Add confidence scoring (match quality + recency + success rate)
- ✅ Implement auto-retraining: re-embed after 100 new incidents

### Phase 4: Observability & Tuning (Week 4)
- ✅ Add `/knowledge/stats` dashboard
- ✅ Prometheus metrics:
    - `knowledge_queries_total{source="local|external"}`
    - `knowledge_query_duration_seconds{source}`
    - `knowledge_hit_rate` (rolling 1h)
- ✅ Grafana dashboard: "AI Learning Layer Performance"
- ✅ A/B test threshold tuning (0.80 vs 0.85 vs 0.90)

### Phase 5: React UI Integration (Week 5)
- ✅ Add "Knowledge Source" badge to Command Center pipeline view
- ✅ Show "Local Knowledge Hit Rate" gauge on dashboard
- ✅ "Similar Incidents" side panel (click agent 4 → see matched history)
- ✅ Cost savings counter (API calls avoided × $0.03/call)

---

## React UI Enhancements

### New Dashboard Widgets

1. **Knowledge Source Badge** (in Agent 4 card):
   ```tsx
   {analysis.source === 'local' ? (
     <Chip 
       icon={<Memory />} 
       label={`LOCAL (${analysis.match_count} similar)`}
       sx={{ background: '#4caf50' }}
     />
   ) : (
     <Chip 
       icon={<Cloud />} 
       label="EXTERNAL (GPT-4)"
       sx={{ background: '#ff9800' }}
     />
   )}
   ```

2. **Learning Stats Panel**:
   ```tsx
   <Card>
     <CardContent>
       <Typography variant="h6">AI Learning Layer</Typography>
       <Grid container spacing={2}>
         <Grid item xs={4}>
           <Metric 
             label="Hit Rate" 
             value={`${stats.hit_rate * 100}%`}
             color={stats.hit_rate > 0.7 ? 'green' : 'amber'}
           />
         </Grid>
         <Grid item xs={4}>
           <Metric 
             label="Avg Response" 
             value={`${stats.avg_response_ms}ms`}
           />
         </Grid>
         <Grid item xs={4}>
           <Metric 
             label="Cost Saved" 
             value={`$${stats.cost_saved_usd}`}
           />
         </Grid>
       </Grid>
     </CardContent>
   </Card>
   ```

3. **Similar Incidents Drawer**:
   ```tsx
   <Drawer>
     <Typography variant="h6">Similar Past Incidents</Typography>
     {similarIncidents.map(incident => (
       <Card key={incident.id}>
         <Typography>{incident.alert_name}</Typography>
         <Chip label={`Similarity: ${incident.similarity * 100}%`} />
         <Typography variant="caption">
           Occurred {incident.occurrences}× | 
           Last: {incident.last_seen} |
           Avg resolution: {incident.avg_resolution_time}s
         </Typography>
       </Card>
     ))}
   </Drawer>
   ```

---

## Success Metrics

### Target KPIs (after 3 months)

| Metric | Baseline (No Learning) | Target (With Learning) |
|--------|------------------------|------------------------|
| Local Hit Rate | 0% | 70-80% |
| Avg Analysis Time | 4.2s | 0.8s (local) / 4.5s (external) |
| GPT-4 API Calls/Day | 250 | 50-75 |
| Monthly API Cost | $225 | $56 (75% reduction) |
| Incident Resolution Time | 3m 45s | 1m 20s |
| Agent Confidence Score | N/A | 0.92 (local known issues) |

---

## Prompt for AI Assistant to Implement This

```
You are building an AI Learning Layer for an AIOps platform.

CONTEXT:
- Existing: 6-agent pipeline (start → logs → metrics → analyze → ticket → approval)
- Agent 4 (analyze) currently calls GPT-4 API for every incident (no memory)
- Need: Local LLM layer that learns from past incidents and reduces external API calls

YOUR TASK:
Build a "knowledge-agent" microservice (FastAPI) that:

1. INTERCEPTS Agent 4 requests BEFORE calling GPT-4
2. QUERIES a vector database (Qdrant) for similar past incidents:
   - Embed incident symptoms using sentence-transformers
   - Similarity search (cosine distance < 0.15)
   - If match found → return cached root cause + remediation
   - If no match → forward to GPT-4, store result for future
3. USES Ollama (llama3:8b) to reformat cached knowledge contextually
4. LEARNS from execution feedback (success/failure) via POST /knowledge/learn
5. EXPOSES:
   - POST /knowledge/query — main analysis endpoint
   - POST /knowledge/learn — feedback loop
   - GET /knowledge/stats — hit rate, cost savings, avg response time
   - GET /knowledge/search — semantic search UI

DELIVERABLES:
- knowledge-agent/Dockerfile
- knowledge-agent/app/main.py (FastAPI service)
- knowledge-agent/app/embeddings.py (sentence-transformers)
- knowledge-agent/app/vector_store.py (Qdrant client)
- knowledge-agent/app/local_llm.py (Ollama client)
- knowledge-agent/requirements.txt
- docker-compose.yml updates (add knowledge-agent, ollama, vector-db)
- Modified compute-agent/app/ai_analyst.py (integrate knowledge-agent)
- README-LEARNING-LAYER.md (architecture docs)

TECH STACK:
- FastAPI 0.110+
- Qdrant client 1.7+
- sentence-transformers 2.5+
- httpx 0.27+
- Ollama HTTP client

START WITH:
Create the knowledge-agent service structure and implement the /knowledge/query endpoint with vector similarity search.
```

---

## Additional Considerations

### Data Privacy
- Store only aggregated symptoms, not sensitive logs
- Hash user IDs / account numbers before embedding
- Option: On-premise only mode (no external LLM calls)

### Model Updates
- Periodically refresh embeddings (monthly)
- Re-rank stored knowledge by recency + success rate
- Prune low-confidence entries (confidence < 0.5 after 10 attempts)

### Multi-Service Learning
- Share knowledge across services (e.g., frontend-api learns from backend-api)
- Weight similarity by service type (API → API higher than API → DB)

### Explainability
- Store "reasoning chain" from Ollama/GPT-4
- Show in UI: "Why was this remediation chosen?"
- Link to past incident tickets (xyOps #1234, #2567, #3890)

---

## Conclusion

This Local LLM Learning Layer transforms the AIOps platform from a **stateless
reactive system** into a **stateful learning system** that:

- ✅ Learns from every incident
- ✅ Accelerates response time (5x faster for known issues)
- ✅ Reduces operational costs (75% fewer API calls)
- ✅ Improves reliability (offline capability)
- ✅ Enhances explainability (show similar past incidents)

**Next Steps:**
1. Review this architecture with the team
2. Approve tech stack (Ollama + Qdrant + sentence-transformers)
3. Assign implementation to engineering team
4. Set up Phase 1 (Week 1) deliverables
5. Track KPIs weekly in Grafana dashboard

---

**Document Version:** 1.0  
**Last Updated:** 2026-03-20  
**Author:** AIOps Platform Team  
**Status:** Ready for Implementation
