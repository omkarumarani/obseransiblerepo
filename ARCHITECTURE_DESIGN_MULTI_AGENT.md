# Multi-Agent AIOps Architecture: Comprehensive Design Document

**Author**: Principal Architect (SRE + Observability + AIOps)  
**Date**: March 2026  
**Status**: Architecture Design (Ready for Implementation)

---

## EXECUTIVE SUMMARY

### The Problem
You have a working observability + AIOps system (alert → analysis → ticket → execution) but it's compute-centric. The planned storage expansion and new intelligence layers risk creating:
- Duplicate code between compute and storage agents
- Two separate "brains" doing similar correlation/anomaly/forecasting
- No shared knowledge of degradation patterns
- LLM calls scattered across services
- Metrics and observability of agent decisions fragmented

### The Solution: Option B with Meta-Coordination
Create **ONE Observability Intelligence Engine (OIE)** used by both Compute Agent and Storage Agent.

**Architecture**:
```
Alert → Compute Agent ──┐
                        ├─→ Shared Intelligence Engine (OIE)
Alert → Storage Agent ──┤   - scenario matching
                        ├─→ - anomaly detection
Background Analysis ────┘   - forecasting
                            - risk scoring
                        ↓
                    xyOps (ticket + execution)
                        ↓
                    Ansible Runner
```

### Key Decisions
1. **Preserve aiops-bridge** — evolve it into Compute Agent, don't rewrite
2. **Extract intelligence** — move scenario/anomaly/forecasting to shared OIE
3. **Sibling agents** — Storage Agent is lightweight wrapper around OIE, not copy-paste
4. **Stateless intelligence** — OIE can be scaled, agents call it via gRPC/REST
5. **LLM at edges** — optional SRE Reasoning Agent for complex interpretation
6. **Observability first** — every decision is a metric
7. **Safe autonomy** — approval gates for risky actions, fully autonomous for safe ones

### Expected Outcome
After implementation, you'll have:
- Two specialized agents (compute + storage) sharing ONE intelligent core
- Scenario engine that knows 20+ degradation patterns
- Continuous proactive anomaly detection (not just alert-driven)
- Forecasting for storage/memory trends
- Comprehensive metrics dashboard showing agent effectiveness
- Enterprise-grade explainability (evidence + confidence scores)
- Path to add network/database/security agents without duplication

**Build time**: 10 days for Phase 1-4 (core system), optional Phase 5 (SRE reasoning) adds 3-5 days

---

## 1. RECOMMENDED FINAL ARCHITECTURE

### Services After Merge

| Service | Purpose | New/Existing | Status |
|---------|---------|--------------|--------|
| **otel-collector** | Telemetry hub | Existing | unchanged |
| **prometheus** | Metrics TSDB | Existing | updated (new scrape targets) |
| **tempo** | Trace storage | Existing | unchanged |
| **loki** | Log aggregation | Existing | unchanged |
| **grafana** | Visualization | Existing | updated (new dashboards) |
| **Shared Intelligence Engine** | Reusable ML/analysis core | NEW | core component |
| **compute-agent** | Compute incident handler (renamed from aiops-bridge) | Refactored | calls OIE |
| **storage-agent** | Storage incident handler | NEW | calls OIE |
| **storage-simulator** | Storage chaos (demo) | NEW | scenarios |
| **xyops** | Central incident platform | Existing | unchanged |
| **alertmanager** | Alert routing | Existing | updated (routes to both agents) |
| **ansible-runner** | Playbook execution | Existing | unchanged |
| **gitea** | Git + audit trail | Existing | unchanged |
| **troublemaker** | Compute chaos (demo) | Existing | unchanged |
| **sre-reasoning-agent** | LLM-powered hypothesis (optional) | OPTIONAL | phase 5 |

### What Stays, What Changes, What's New

| Component | Action | Reason |
|-----------|--------|--------|
| aiops-bridge webhook entry | **Keep** | Core alert ingestion is stable |
| aiops-bridge internal analysis | **Extract to OIE** | Duplication risk with storage |
| aiops-bridge approval workflow | **Keep** | Works well, reuse for both agents |
| aiops-bridge xyops integration | **Keep** | Stable, documented |
| alertmanager routing | **Extend** | Add storage-agent receiver |
| prometheus scrape config | **Extend** | Add storage-agent, OIE, storage-sim targets |
| grafana dashboards | **Create new** | Agentic AI Operations + Scenario Intelligence |
| xyops workflows | **Create new** | Storage Agent pipeline workflow |
| ansible runner | **No change** | Both agents use same executor |
| gitea | **No change** | Both agents use same audit repo |

---

## 2. MULTI-AGENT MODEL

### Agent Roles & Responsibilities

#### **Compute Agent** (evolved from aiops-bridge)
**Responsibility**: Handle CPU, memory, restart, latency, error rate incidents

**Flow**:
1. Receive Prometheus alert (webhook)
2. Call OIE `/analyze` with domain="compute"
3. Get scenario matches, anomaly scores, recommendations
4. Create/update xyOps ticket with OIE insights
5. Route through approval workflow (if risky)
6. Execute via Ansible Runner
7. Emit metrics: `compute_agent_*`

**Can autonomously**:
- Restart single pod
- Scale replicas
- Clear cache

**Needs approval for**:
- Restart critical service
- Migrate pod off node

#### **Storage Agent** (new)
**Responsibility**: Handle OSD, PVC, pool, backfill, latency incidents

**Flow**:
1. Receive alert (webhook from alertmanager)
2. Call OIE `/analyze` with domain="storage"
3. Get scenario matches, recommendations
4. Create/update xyOps ticket
5. Execute via xyOps jobs (OSD reweight, etc.)
6. Emit metrics: `storage_agent_*`

**Can autonomously**:
- Apply IOPS throttle
- Reweight OSD

**Needs approval for**:
- Add/remove OSD
- Trigger rebalance
- Change replica factor

#### **Shared Intelligence Engine (OIE)** — NEW CORE
**Responsibility**: Provide reusable telemetry analysis for all agents

**Exposes**:
- `POST /analyze` — given alert, return scenario matches + recommendations
- `POST /forecast` — predict degradation trajectory
- `POST /anomaly_score` — compute novelty/anomaly
- `POST /risk_score` — estimate impact * likelihood
- `GET /scenarios` — list known patterns
- `GET /metrics` — Prometheus metrics

**Modules**:
1. **Telemetry Access Layer** — query Prometheus + Loki
2. **Feature Extraction** — compute signals from raw metrics
3. **Scenario Correlation Engine** — match against known patterns
4. **Anomaly Detection Engine** — detect novel behavior
5. **Forecasting Engine** — predict time-to-critical
6. **Risk Scoring Engine** — estimate urgency
7. **Recommendation Engine** — suggest actions
8. **Evidence Generation** — explain why with metrics + confidence

**Background Jobs** (runs independently):
- Every 5 min: scan all services for early anomalies
- Every 15 min: update forecasts (storage fill, memory trends)
- Every hour: compute scenario trend metrics

**Does NOT**:
- Make decisions
- Execute actions
- Create tickets directly (agents do)
- Hold state between requests

#### **SRE Reasoning Agent** (optional, phase 5)
**Responsibility**: Enhanced interpretation for unknowns and complex scenarios

**When called**:
- Anomaly detected with no scenario match
- Unknown pattern needs human explanation
- Complex root cause chain needs hypothesis

**Provides**:
- LLM-powered interpretation
- Hypothesis generation
- Investigation steps
- Risk assessment narrative

**Runs as**:
- Sidecar or background service
- Called by agents when confidence is low
- Results feed back to ticket (not direct action)

---

## 3. MERGE STRATEGY FOR EXISTING COMPONENTS

### How aiops-bridge Evolves into Compute Agent

**Phase 1: Minimum Cut**
```
Current state:
  aiops-bridge/app/main.py
    └── webhook handler
    └── inline scenario checking
    └── inline anomaly detection
    └── calls ai_analyst.py (LLM)

Target state:
  compute-agent/app/main.py
    └── webhook handler (unchanged)
    └── calls oie_client.py (OIE service)
    └── routes to xyops (unchanged)
    └── emit metrics (NEW)
```

**What stays in compute-agent**:
- Webhook reception (`POST /webhook`)
- Alert parsing
- xyOps ticket creation
- Approval workflow
- Ansible Runner calling
- OTel instrumentation

**What moves to OIE**:
- Scenario correlation logic
- Anomaly computation
- Feature extraction
- Forecasting
- Risk scoring
- Evidence gathering

**What's new in compute-agent**:
- `oie_client.py` — HTTP/gRPC client to OIE
- `metrics_exporter.py` — expose Prometheus metrics
- `GET /metrics` endpoint
- Configuration for OIE endpoint

**Code change**: ~400 lines removed from main.py, ~150 lines of OIE client added. Net: -250 lines.

### Storage Agent: Lightweight Design

```
storage-agent/app/
  ├── main.py              # FastAPI + webhook
  ├── webhook_handler.py   # alert parsing (similar to compute)
  ├── pipeline.py          # 6-step workflow
  ├── oie_client.py        # calls shared OIE
  ├── xyops_client.py      # tickets + execution (reuse patterns)
  ├── metrics_exporter.py  # storage_agent_* metrics
  └── models.py            # Pydantic schemas
```

**Code pattern**: ~30% is shared with compute-agent (xyops calling, metrics export, OIE client). Storage-specific: action execution (OSD reweight vs pod restart).

### Storage Simulator Integration

**Role**: Inject realistic Ceph metrics for testing

**Design**:
- Exposes `/metrics` endpoint (Prometheus format)
- Scenarios: healthy, osd_down, pool_full, latency_spike, noisy_pvc, multi_osd_failure
- Prometheus scrapes it every 15s
- Scenario can be triggered via `POST /scenario/{name}` or `GET /scenario?trigger=pool_full`

**Discovery**:
- Runs with profile: `docker compose --profile storage up`
- Prometheus config updated to scrape `storage-simulator:9100/metrics`

### How xyOps Workflows Are Reused

**Current compute workflow**:
- 6 nodes calling aiops-bridge endpoints
- Visible on canvas
- Approved actions trigger ansible-runner

**New storage workflow**:
- Same 6-node pattern
- Nodes call storage-agent endpoints instead
- Same approval flow
- Actions trigger different ansible jobs (ceph-rebalance, pvc-quota, etc.)

**Implementation**:
- Create new workflow JSON: `storage-agent-pipeline.json`
- Auto-provisioned by storage-agent on startup
- Reuses xyOps canvas, approval, job execution

**No changes needed** to xyOps core — it's just routing to different webhook targets.

### Alertmanager Routing

**Current**: All alerts → aiops-bridge (compute)

**New**:
```yaml
global:
  resolve_timeout: 5m

route:
  receiver: 'default'
  group_by: ['alertname', 'service_name']
  routes:
    # Compute alerts → compute-agent
    - match:
        domain: 'compute'
      receiver: 'compute-agent'
      continue: true
    
    # Storage alerts → storage-agent
    - match:
        domain: 'storage'
      receiver: 'storage-agent'
      continue: true

receivers:
  - name: 'compute-agent'
    webhook_configs:
      - url: 'http://compute-agent:9000/webhook'
  
  - name: 'storage-agent'
    webhook_configs:
      - url: 'http://storage-agent:9001/webhook'
```

**Alert labels** must include `domain: compute` or `domain: storage` for routing to work.

---

## 4. SHARED INTELLIGENCE CORE DESIGN

### Architecture Diagram
```
Agents (Compute, Storage)
    ↓
    ├─→ OIE: /analyze?domain=compute&alert=HighMemory
    ├─→ OIE: /analyze?domain=storage&alert=CephPoolFull
    ├─→ OIE: /forecast?domain=storage&resource=pool_utilization
    └─→ OIE: /anomalies?domain=compute&lookback=24h

         ┌─────────────────────────────────────────┐
         │  Shared Intelligence Engine (OIE)        │
         ├─────────────────────────────────────────┤
         │ 1. Telemetry Access (Prom + Loki)       │
         │ 2. Feature Extraction                   │
         │ 3. Scenario Correlation Engine          │
         │ 4. Anomaly Detection                    │
         │ 5. Forecasting                          │
         │ 6. Risk Scoring                         │
         │ 7. Recommendation Generation            │
         │ 8. Evidence + Confidence               │
         │ 9. Metrics Export                       │
         └─────────────────────────────────────────┘
                    ↓
            Prometheus Metrics
            Evidence Logs
            Recommendations
```

### Module 1: Telemetry Access Layer

**Purpose**: Centralized, cached access to Prometheus + Loki

**Interfaces**:
```python
class TelemetryAccess:
    async def query_metrics(
        self,
        query: str,  # PromQL
        start: datetime,
        end: datetime,
        step: str = "15s"
    ) -> List[TimeSeries]
    
    async def query_logs(
        self,
        query: str,  # LogQL
        limit: int = 100
    ) -> List[LogLine]
    
    async def get_metric_value(
        self,
        metric_name: str,
        labels: Dict[str, str],
        lookback: str = "5m"
    ) -> Optional[float]  # latest value
```

**Implementation Details**:
- Cache metric queries for 30s (reduce Prometheus load)
- Cache log queries for 10s
- Return None if datasource unavailable (circuit breaker)
- Log query latency to Prometheus

---

### Module 2: Feature Extraction Engine

**Purpose**: Transform raw metrics into meaningful signals

**Example: Compute Domain**
```python
class ComputeFeatures:
    # Signals extracted from metrics
    memory_growth_rate: float  # % per minute
    gc_pause_duration: float   # ms, p99
    error_rate: float          # errors/sec
    latency_p99: float         # ms
    cpu_throttle_events: int   # count/min
    restart_count_5m: int      # restarts in last 5 min
    connection_pool_utilization: float  # %
```

**Example: Storage Domain**
```python
class StorageFeatures:
    osd_count_up: int
    osd_count_down: int
    pool_utilization: float  # %
    backfill_objects: int
    pvc_latency_p99: float  # ms
    single_pvc_iops_percent: float  # % of pool
    recovery_bytes_per_sec: float
```

**Implementation**:
```python
async def extract_compute_features(
    service_name: str,
    telemetry: TelemetryAccess
) -> ComputeFeatures:
    # Query last 5 min of metrics
    # Compute growth rates, percentiles, counts
    # Return feature dict
    pass
```

---

### Module 3: Scenario Correlation Engine

**Purpose**: Match extracted features to known degradation patterns

**Input**: Features + alert (optional)  
**Output**: List of matching scenarios ranked by confidence

```python
class ScenarioMatch:
    scenario_name: str      # e.g., "memory_leak"
    confidence: float       # 0.0-1.0
    matched_signals: List[str]  # which features matched
    evidence_metric_count: int  # how many evidence items present
    missing_signals: List[str]  # which features wanted but unavailable
    estimated_time_to_critical: str  # e.g., "30 minutes"
```

**Algorithm**:
```
For each scenario in scenarios/{domain}/:
  1. Check prerequisites (e.g., "service must have restarted recently")
  2. For each symptom:
     - Match metric/feature against threshold
     - Assign confidence weight
  3. Sum confidence weights, normalize
  4. If total_confidence > min_threshold: add to results
  5. Sort by confidence descending
  6. Return top 3-5
```

**Example: Memory Leak Scenario**
```yaml
# scenarios/compute/memory_leak.yaml
name: memory_leak_emergence
confidence_factors:
  - metric: memory_growth_rate > 1.0  # % per min
    weight: 0.3
  - metric: gc_pause_duration_p99 > 100  # ms
    weight: 0.2
  - metric: error_rate_spike == true
    weight: 0.2
  - duration: sustained for 30+ minutes
    weight: 0.2
  - metric: restart_count == 0  # not restarted yet
    weight: 0.1

total_confidence_threshold: 0.65

evidence_manifest:
  - query: 'rate(heap_memory[5m])'
    describes: memory_growth_rate
  - query: 'histogram_quantile(0.99, gc_pause_duration)'
    describes: gc latency spike
```

**Confidence Scoring**:
- Each matched symptom contributes weight
- Missing prerequisite → confidence *= 0.5
- Metric unavailable → confidence *= 0.8
- Recently restarted → reduce confidence (already tried)
- Final: sum of weights, normalized to 0-1

---

### Module 4: Anomaly Detection Engine

**Purpose**: Detect novel, previously-unseen patterns

**Methods**:
1. **Baseline Comparison**: Compare current metrics against normal range
2. **Novelty Detection**: Isolated Forest or PCA on feature vectors
3. **Sudden Change Detection**: Compare short-term vs long-term averages

```python
class AnomalyDetection:
    async def compute_anomaly_score(
        self,
        domain: str,  # "compute" or "storage"
        lookback: str = "24h"
    ) -> float:  # 0.0-1.0
        # Use baseline + statistical tests
        # Return anomaly_score
        pass
    
    async def get_baseline(
        self,
        metric_name: str
    ) -> Dict[str, float]:  # {p5, p50, p95, stdev}
        # Historical percentiles for metric
        pass
```

**Implementation**:
- Use Prometheus raw metrics directly
- Compute baseline from last 7 days (exclude outliers)
- Compare current 5-min avg vs baseline
- Apply z-score: (current - mean) / stdev
- If |z-score| > 2.5: anomaly_score increases

**Novelty Detection**:
- Feature vector = [memory_growth, cpu%, latency, error_rate, ...]
- Fit Isolation Forest on last 100 samples
- Score current sample
- novelty_score in [0, 1] where 1 = completely new

---

### Module 5: Forecasting Engine

**Purpose**: Predict degradation trajectories

```python
class Forecasting:
    async def forecast_degradation(
        self,
        metric_name: str,
        domain: str,
        days_ahead: int = 7
    ) -> ForecastResult:
        """Predict when metric will hit critical threshold."""
        pass
    
    async def forecast_time_to_critical(
        self,
        domain: str,
        resource: str  # "compute_memory", "storage_pool", etc.
    ) -> Dict[str, Any]:
        """How many days until this resource is exhausted?"""
        return {
            "days_remaining": 15,
            "confidence": 0.85,
            "current_utilization": 0.72,
            "daily_growth_rate": 0.02,  # 2% per day
            "critical_utilization": 0.95
        }
```

**Implementation**:
- Simple linear regression on 7-day history
- Query historical data from Prometheus
- Fit line: utilization = a * time + b
- Solve for: utilization = critical_threshold
- Return days remaining + confidence interval

**Example**:
```
Storage pool utilization:
  Day 1: 55%
  Day 2: 57%
  Day 3: 59%
  Day 4: 61%
  Day 5: 63%
  Day 6: 65%
  Day 7: 67%

Trend: +2% per day
Days to 95%: (95-67) / 2 = 14 days
Confidence: 0.78 (high variability reduces confidence)
```

---

### Module 6: Risk Scoring Engine

**Purpose**: Quantify urgency and impact

```python
class RiskScoring:
    async def compute_risk_score(
        self,
        domain: str,
        scenario_name: str,
        current_state: Dict[str, Any]
    ) -> RiskScore:
        return {
            "overall_risk": 0.85,  # 0-1
            "severity": "high",    # low/medium/high/critical
            "impact": 0.90,
            "likelihood": 0.75,
            "urgency": 0.80,
            "slo_at_risk": True,
            "estimated_mttr": "30 minutes"
        }
```

**Scoring Formula**:
```
Impact = how_many_services_affected * criticality_weight
       = 3 services * 0.8 = 2.4 (normalized to 0-1: 0.9)

Likelihood = current_anomaly_score * scenario_match_confidence
           = 0.75 * 0.85 = 0.64

Urgency = (1 - time_to_critical_ratio) * scenario_severity
        = (1 - 14_days_remaining/30_day_window) * 0.9
        = 0.53 * 0.9 = 0.48

Overall_Risk = (Impact * Likelihood * 0.5) + (Urgency * 0.5)
             = (0.9 * 0.64 * 0.5) + (0.48 * 0.5)
             = 0.29 + 0.24 = 0.53

Severity = "high" if overall_risk > 0.75 else "medium" if > 0.5 else "low"
```

---

### Module 7: Recommendation Engine

**Purpose**: Suggest actions based on scenario + risk

```python
class Recommendation:
    action: str              # "restart_service", "reweight_osd", etc.
    reasoning: str           # Why this action
    approval_required: bool
    estimated_resolution_time: str
    confidence: float
    prerequisites: List[str]  # must be true before execution
```

**Implementation**:
```python
async def generate_recommendations(
    domain: str,
    scenario_matches: List[ScenarioMatch],
    risk_score: float
) -> List[Recommendation]:
    recommendations = []
    for scenario in scenario_matches:
        for rec in scenario.yaml['recommendations']:
            recommendations.append({
                'action': rec['action'],
                'reasoning': f"{scenario.name} ({scenario.confidence:.0%} confidence)",
                'approval_required': rec.get('approval_required', False),
                'estimated_resolution_time': rec.get('estimated_resolution_time', '15m'),
                'confidence': scenario.confidence
            })
    return recommendations
```

---

### Module 8: Evidence Generation

**Purpose**: Explain WHY the agent recommends an action

```python
class AnalysisResult:
    scenario_matches: List[ScenarioMatch]
    anomaly_score: float
    risk_score: RiskScore
    evidence: List[Evidence]
    recommendations: List[Recommendation]
    confidence_factors: Dict[str, float]
    
class Evidence:
    metric_name: str
    current_value: float
    expected_range: Tuple[float, float]
    query: str  # PromQL or LogQL
    describes: str  # what this tells us
```

**Example**:
```json
{
  "scenario_matches": [
    {
      "name": "memory_leak",
      "confidence": 0.92,
      "matched_signals": [
        "memory_growth_rate > 1.0%/min",
        "gc_pause_duration_p99 spike"
      ]
    }
  ],
  "evidence": [
    {
      "metric": "heap_memory_bytes",
      "current": 5_200_000_000,
      "expected_range": [4_500_000_000, 5_000_000_000],
      "describes": "Heap is 4% over normal max"
    },
    {
      "metric": "gc_pause_duration_p99",
      "current": 250,
      "expected_range": [50, 150],
      "describes": "GC pauses 67% longer than baseline, indicates large heap"
    }
  ],
  "confidence_factors": {
    "duration_30min": 1.0,
    "all_symptoms_match": 0.9,
    "service_not_restarted_recently": 1.0,
    "metric_query_success": 1.0
  }
}
```

---

### Module 9: Metrics Export Layer

**Purpose**: Expose OIE's own decisions as Prometheus metrics

```python
# In oie/app/metrics_exporter.py
from prometheus_client import Counter, Gauge, Histogram

scenario_match_confidence = Gauge(
    'oie_scenario_match_confidence_minutes',
    'How long (minutes) has this scenario been matching',
    ['domain', 'scenario_name']
)

anomaly_score = Gauge(
    'oie_anomaly_score',
    'Current anomaly score',
    ['domain', 'metric_name']
)

recommendation_count = Counter(
    'oie_recommendation_generated_total',
    'Recommendations generated',
    ['domain', 'action_type', 'approved']
)

risk_score = Gauge(
    'oie_risk_score',
    'Current risk score',
    ['domain', 'scenario_name']
)

forecast_days_to_critical = Gauge(
    'oie_forecast_degradation_days',
    'Days until resource exhaustion',
    ['domain', 'component', 'resource_type']
)

# Background job exports these periodically
def export_intelligence_metrics():
    # Every 5 minutes:
    #   - compute scenario confidence metrics
    #   - storage scenario confidence metrics
    #   - anomaly scores
    #   - forecast predictions
    #   - risk scores
    pass
```

**Exposed at**: `GET http://shared-intelligence-engine:9100/metrics`

---

## 5. KNOWN SCENARIO ENGINE

### Scenario Schema

```yaml
# scenarios/schema.yaml
type: object
properties:
  apiVersion:
    type: string
    enum: ['v1']
  
  kind:
    type: string
    enum: ['ComputeDegradation', 'StorageDegradation']
  
  metadata:
    type: object
    properties:
      name: string         # e.g., memory_leak_emergence
      description: string
      domain: string       # compute | storage
      severity: string     # low | medium | high | critical
      version: string      # 0.1.0
      last_updated: string # ISO timestamp
  
  detection:
    type: object
    properties:
      symptoms:
        type: array
        items:
          type: object
          properties:
            metric_name: string      # Prometheus metric
            operator: string         # >, <, ==, sudden_change
            threshold: number
            duration: string         # 5m, 10m, 30m, etc.
            confidence_weight: number  # 0.0-1.0
      
      prerequisites:
        type: array
        items:
          type: object
          properties:
            check: string   # function name or metric query
            reason: string  # why this must be true
            fallback: boolean  # if check fails, reduce confidence instead of exclude
  
  impact:
    type: object
    properties:
      affected_domain: string  # compute | storage
      affected_services: array
        items: string
      estimated_mttr: string   # 5m | 30m | 1h
      slo_at_risk: boolean
      cascading_risk: string   # describes potential cascade
  
  recommendations:
    type: array
    items:
      type: object
      properties:
        action: string          # e.g., "restart_service"
        reasoning: string
        approval_required: boolean
        estimated_resolution_time: string
        prerequisites:
          type: array
          items: string        # must be true before action
        rollback_plan: string   # optional
  
  evidence_manifest:
    type: array
    items:
      type: object
      properties:
        metric: string          # PromQL query
        label: string           # human-readable name
        describes: string       # what it tells us
        confidence_weight: number  # how much it proves scenario
  
  disable_conditions:
    type: array
    items:
      type: object
      properties:
        when: string            # condition (e.g., recent_deployment)
        reason: string
        confidence_reduction: number  # multiply confidence by this factor
```

---

### Example Compute Scenarios

#### Scenario 1: Memory Leak Emergence

```yaml
apiVersion: v1
kind: ComputeDegradation
metadata:
  name: memory_leak_emergence
  description: >
    Heap memory growing >1% per minute for 30+ minutes indicates
    unreleased objects or memory accumulat. GC pause times increase.
    Risk: OOM crash, cascading service failure.
  domain: compute
  severity: high
  version: "1.0.0"

detection:
  symptoms:
    - metric_name: heap_memory_growth_rate
      operator: ">"
      threshold: 1.0        # % per minute
      duration: 30m
      confidence_weight: 0.4
    
    - metric_name: gc_pause_duration_p99
      operator: ">"
      threshold: 100        # milliseconds
      duration: 10m
      confidence_weight: 0.3
    
    - metric_name: error_rate
      operator: ">"
      threshold: 1.0        # errors/sec
      duration: 15m
      confidence_weight: 0.2
  
  prerequisites:
    - check: "service_not_restarted_in_last_5m"
      reason: "restart will reset memory, masking leak"
      fallback: true        # reduce confidence instead of reject

impact:
  affected_domain: compute
  affected_services: ["frontend-api", "backend-api"]
  estimated_mttr: "10 minutes"  # restart
  slo_at_risk: true
  cascading_risk: "High heap pressure can trigger cascading GC pauses across cluster"

recommendations:
  - action: restart_service
    reasoning: >
      Restart clears heap. Monitor for recurrence.
      If recurs within 1 hour, likely true leak.
    approval_required: false
    estimated_resolution_time: "5 minutes"
    prerequisites:
      - "service_not_critical_path"
      - "replica_count > 1"
    rollback_plan: "No rollback needed; restart is idempotent"
  
  - action: increase_heap_allocation
    reasoning: >
      Temporary measure while investigating leak root cause.
      Does not fix but buys time for proper debugging.
    approval_required: true
    estimated_resolution_time: "0 minutes (config only)"
    prerequisites:
      - "node_has_free_memory"

evidence_manifest:
  - metric: 'rate(jvm_memory_used_bytes{area="heap"}[5m])'
    label: heap_memory_growth
    describes: "Memory allocation rate (bytes/sec) → convert to %/min"
    confidence_weight: 0.4
  
  - metric: 'histogram_quantile(0.99, gc_pause_duration_seconds)'
    label: gc_p99_latency
    describes: "99th percentile GC pause; >100ms indicates heap pressure"
    confidence_weight: 0.3
  
  - metric: 'rate(errors_total[5m])'
    label: error_spike
    describes: "Error rate spike suggests service degradation"
    confidence_weight: 0.2

disable_conditions:
  - when: "deployment_rollout_in_progress"
    reason: "New code may trigger transient memory growth"
    confidence_reduction: 0.5
  
  - when: "recent_cache_clear"
    reason: "Memory growth after cache clear is normal rebuilding"
    confidence_reduction: 0.3
```

#### Scenario 2: Noisy Neighbor (CPU)

```yaml
apiVersion: v1
kind: ComputeDegradation
metadata:
  name: noisy_neighbor_cpu
  description: >
    One pod consuming >80% of node CPU, starving neighbors.
    Manifests as latency spike in other services on same node.
  domain: compute
  severity: high

detection:
  symptoms:
    - metric_name: pod_cpu_percent
      operator: ">"
      threshold: 80
      duration: 5m
      confidence_weight: 0.5
    
    - metric_name: latency_p99_neighbors
      operator: ">"
      threshold: 2.0  # 2x baseline
      duration: 5m
      confidence_weight: 0.4
    
    - metric_name: cpu_throttle_periods
      operator: ">"
      threshold: 100  # throttle events/min
      duration: 5m
      confidence_weight: 0.3

impact:
  affected_domain: compute
  affected_services: ["frontend-api", "backend-api", "cache-service"]
  estimated_mttr: "5 minutes"  # migration
  slo_at_risk: true
  cascading_risk: "Can trigger cascading latency spike across all co-located services"

recommendations:
  - action: migrate_pod_to_isolated_node
    reasoning: "Move noisy pod to dedicated node, restore CPU isolation"
    approval_required: true
    estimated_resolution_time: "2 minutes"
    prerequisites:
      - "isolated_node_available"
      - "pod_not_critical_path"
  
  - action: apply_cpu_limit
    reasoning: "Limit CPU to 50% of node to prevent starvation"
    approval_required: false
    estimated_resolution_time: "1 minute"

evidence_manifest:
  - metric: 'sum by (pod) (rate(container_cpu_usage_seconds_total[5m]))'
    label: pod_cpu_usage
    describes: "CPU % per pod; identify which pod is noisy"
    confidence_weight: 0.5
  
  - metric: 'histogram_quantile(0.99, http_request_duration_seconds)'
    label: latency_spike
    describes: "Latency spike correlates with CPU contention"
    confidence_weight: 0.4

disable_conditions:
  - when: "expected_batch_job_runtime"
    reason: "Legitimate high CPU from batch processing"
    confidence_reduction: 0.7
```

#### Scenario 3: Connection Pool Exhaustion

```yaml
apiVersion: v1
kind: ComputeDegradation
metadata:
  name: connection_pool_exhaustion
  description: >
    Database or external service connection pool is full.
    New requests wait in queue, latency spikes, eventually timeout.
  domain: compute
  severity: critical

detection:
  symptoms:
    - metric_name: connections_active_percent
      operator: ">"
      threshold: 95
      duration: 2m
      confidence_weight: 0.6
    
    - metric_name: connection_wait_queue_size
      operator: ">"
      threshold: 50
      duration: 2m
      confidence_weight: 0.4
    
    - metric_name: request_timeout_rate
      operator: ">"
      threshold: 0.5  # 50% of requests timing out
      duration: 1m
      confidence_weight: 0.5

impact:
  affected_domain: compute
  affected_services: ["backend-api"]
  estimated_mttr: "5 minutes"
  slo_at_risk: true
  cascading_risk: "Can cascade to frontend-api if backend is essential"

recommendations:
  - action: increase_pool_size
    reasoning: "Increase max connections to handle spike"
    approval_required: false
    estimated_resolution_time: "1 minute"
    prerequisites:
      - "database_has_connection_headroom"
  
  - action: restart_service
    reasoning: "Restart flushes stuck connections, resets pool"
    approval_required: false
    estimated_resolution_time: "2 minutes"
    prerequisites:
      - "replica_count > 1"
```

---

### Example Storage Scenarios

#### Scenario 1: OSD Down Cascade

```yaml
apiVersion: v1
kind: StorageDegradation
metadata:
  name: osd_down_cascade
  description: >
    One OSD goes down. Ceph begins rebalancing data.
    If another OSD fails during rebalance, pool enters degraded state.
    Data at risk if 3rd OSD fails (assumes 3-replica).
  domain: storage
  severity: critical

detection:
  symptoms:
    - metric_name: ceph_osd_up
      operator: "=="
      threshold: 0  # at least 1 OSD down
      duration: 1m
      confidence_weight: 0.9
    
    - metric_name: ceph_backfill_objects
      operator: ">"
      threshold: 1_000_000
      duration: 5m
      confidence_weight: 0.7
    
    - metric_name: ceph_pg_down_count
      operator: ">"
      threshold: 0
      duration: 1m
      confidence_weight: 0.8

impact:
  affected_domain: storage
  affected_services: ["all-services"]
  estimated_mttr: "30 minutes"
  slo_at_risk: true
  cascading_risk: "CRITICAL: If 2nd OSD fails during rebalance, data loss risk"

recommendations:
  - action: reweight_osds
    reasoning: "Rebalance load away from failing OSD, accelerate recovery"
    approval_required: true
    estimated_resolution_time: "30 minutes"
  
  - action: disable_balancer
    reasoning: "Pause automatic balancer to prevent interference"
    approval_required: false
    estimated_resolution_time: "1 minute"
  
  - action: escalate_to_ops
    reasoning: "Hardware failure warrants immediate ops attention"
    approval_required: false
    estimated_resolution_time: "N/A (escalation only)"

evidence_manifest:
  - metric: 'count(ceph_osd_up == 0)'
    label: osds_down
    describes: "Number of OSDs currently down"
    confidence_weight: 0.9
  
  - metric: 'ceph_pg_down'
    label: degraded_pgs
    describes: "Placement groups without full replication"
    confidence_weight: 0.8
```

#### Scenario 2: Noisy PVC (IOPS Hog)

```yaml
apiVersion: v1
kind: StorageDegradation
metadata:
  name: noisy_pvc_iops
  description: >
    One PVC consuming >60% of pool IOPS.
    Other PVCs see latency spike (p99 > 500ms).
    Indicates noisy neighbor at storage layer.
  domain: storage
  severity: high

detection:
  symptoms:
    - metric_name: single_pvc_iops_percent
      operator: ">"
      threshold: 60
      duration: 5m
      confidence_weight: 0.6
    
    - metric_name: pvc_latency_p99
      operator: ">"
      threshold: 500  # milliseconds
      duration: 5m
      confidence_weight: 0.5  # for non-noisy PVCs
    
    - metric_name: pool_io_rate
      operator: ">"
      threshold: 10000  # iops
      duration: 5m
      confidence_weight: 0.4

impact:
  affected_domain: storage
  affected_services: ["all-services-using-pvcs"]
  estimated_mttr: "5 minutes"
  slo_at_risk: true
  cascading_risk: "Can cause latency cascade across all apps using storage"

recommendations:
  - action: apply_iops_throttle
    reasoning: "Limit noisy PVC to 50% of its current IOPS"
    approval_required: false
    estimated_resolution_time: "2 minutes"
    prerequisites:
      - "pvc_owner_identified"
  
  - action: migrate_pvc_to_dedicated_pool
    reasoning: "Separate noisy workload into its own storage pool"
    approval_required: true
    estimated_resolution_time: "30 minutes"

evidence_manifest:
  - metric: 'pvc_iops_percent'
    label: noisy_pvc_iops
    describes: "Which PVC is consuming most IOPS"
    confidence_weight: 0.6
  
  - metric: 'rate(pvc_latency_ms[5m])'
    label: latency_spike
    describes: "Latency p99 spike in other PVCs"
    confidence_weight: 0.5
```

#### Scenario 3: Pool Fill Forecast

```yaml
apiVersion: v1
kind: StorageDegradation
metadata:
  name: pool_fill_forecast
  description: >
    Storage pool utilization trending toward 100%.
    Forecast predicts critical state (95%+ full) in N days.
    Requires proactive capacity planning action.
  domain: storage
  severity: medium  # becomes critical if < 7 days to full

detection:
  symptoms:
    - metric_name: pool_utilization_percent
      operator: ">"
      threshold: 70
      duration: 0  # no duration check, instantaneous
      confidence_weight: 0.5
    
    - metric_name: days_to_pool_full_forecast
      operator: "<"
      threshold: 14
      duration: 0
      confidence_weight: 0.8

impact:
  affected_domain: storage
  affected_services: ["all-services"]
  estimated_mttr: "N/A (planning required)"
  slo_at_risk: false  # not immediate but will become true
  cascading_risk: "Pool full causes all writes to fail"

recommendations:
  - action: request_capacity_expansion
    reasoning: "Proactively order storage hardware"
    approval_required: true
    estimated_resolution_time: "varies (procurement)"
  
  - action: implement_retention_policy
    reasoning: "Automatically delete old snapshots/backups"
    approval_required: false
    estimated_resolution_time: "1 day (after policy activation)"

evidence_manifest:
  - metric: 'ceph_pool_utilization_percent'
    label: pool_utilization
    describes: "Current pool usage %"
    confidence_weight: 0.5
  
  - metric: 'forecast_days_until_critical'
    label: forecast_trend
    describes: "Linear regression: days until pool reaches 95%"
    confidence_weight: 0.8
```

---

## 6. AGENT DECISION PIPELINE (Step-by-Step)

### Pipeline 1: Alert-Driven Incident (Compute Example)

**Trigger**: Prometheus fires alert "HighMemory"

**Step 1: Alert Reception**
```
Prometheus Alert:
{
  "status": "firing",
  "alerts": [{
    "labels": {
      "alertname": "HighMemory",
      "service_name": "backend-api",
      "severity": "warning",
      "domain": "compute"
    },
    "annotations": {
      "summary": "Memory usage > 80%",
      "runbook_url": "..."
    }
  }]
}

↓ Alertmanager routes to compute-agent ↓

POST http://compute-agent:9000/webhook
Header: Content-Type: application/json
Body: {...alert payload...}
```

**Step 2: Agent Processes Alert**
```python
# compute-agent/app/main.py
@app.post("/webhook")
async def receive_alert(payload: AlertmanagerPayload):
    alert = payload.alerts[0]
    service_name = alert.labels['service_name']
    
    # Call OIE for analysis
    analysis = await oie_client.analyze(
        domain="compute",
        alert_name="HighMemory",
        service_name=service_name,
        timestamp=alert.pop('startsAt')
    )
    
    return analysis
```

**Step 3: OIE Analyzes**
```python
# shared-intelligence-engine/app/main.py
@app.post("/analyze")
async def analyze(request: AnalysisRequest):
    domain = request.domain  # "compute"
    alert_name = request.alert_name  # "HighMemory"
    service_name = request.service_name  # "backend-api"
    
    # 1. Extract features
    features = await feature_extractor.extract_compute_features(
        service_name=service_name,
        telemetry=telemetry_access
    )
    
    # 2. Match scenarios
    scenario_matches = await scenario_engine.correlate(
        domain=domain,
        features=features,
        alert_name=alert_name
    )
    # Returns: [
    #   {name: "memory_leak", confidence: 0.92},
    #   {name: "gc_pause_spike", confidence: 0.71},
    # ]
    
    # 3. Compute anomaly
    anomaly_score = await anomaly_detector.compute_anomaly(
        domain=domain,
        features=features
    )  # Returns: 0.65
    
    # 4. Compute risk
    risk_score = await risk_scorer.score(
        domain=domain,
        scenario_matches=scenario_matches,
        anomaly_score=anomaly_score
    )  # Returns: {overall: 0.82, severity: "high"}
    
    # 5. Generate recommendations
    recommendations = await recommendation_engine.generate(
        domain=domain,
        scenario_matches=scenario_matches,
        risk_score=risk_score
    )
    # Returns: [{action: "restart", approval_required: false}, ...]
    
    # 6. Gather evidence
    evidence = await evidence_generator.explain(
        scenario_matches=scenario_matches,
        features=features
    )
    
    # 7. Export metrics
    await metrics_exporter.update(
        domain=domain,
        scenario_matches=scenario_matches,
        anomaly_score=anomaly_score,
        risk_score=risk_score
    )
    
    return {
        "scenario_matches": scenario_matches,
        "anomaly_score": anomaly_score,
        "risk_score": risk_score,
        "recommendations": recommendations,
        "evidence": evidence
    }
```

**Step 4: Agent Creates Ticket**
```python
# compute-agent/app/main.py (after OIE response)
ticket_body = f"""
## Alert: HighMemory

**Service**: {service_name}
**Severity**: {analysis.risk_score.severity}

### OIE Analysis

**Scenario Matches** (ranked by confidence):
- Memory Leak (92%) — Heap growing >1%/min, GC pauses spike
- GC Pause Spike (71%) — High GC pause latency indicates heap pressure

**Anomaly Score**: {analysis.anomaly_score:.2%} (novelty detection)

**Risk Assessment**:
- Overall Risk: {analysis.risk_score.overall:.0%}
- SLO at Risk: {analysis.risk_score.slo_at_risk}
- Estimated MTTR: {analysis.risk_score.estimated_mttr}

### Evidence
{format_evidence(analysis.evidence)}

### Recommended Actions
{format_recommendations(analysis.recommendations)}

**Trace**: [View in Tempo](http://localhost:3200/api/traces/{trace_id})
"""

ticket = await xyops_client.create_ticket(
    title=f"[COMPUTE] {alert_name} on {service_name}",
    body=ticket_body,
    severity=analysis.risk_score.severity,
    domain="compute"
)

return ticket
```

**Step 5: Approval Routing**
```python
# compute-agent/app/approval_workflow.py
if is_risky_action(recommendation):
    # Create approval ticket
    await xyops_client.request_approval(
        incident_ticket=ticket,
        recommended_action=recommendation,
        evidence=analysis.evidence
    )
else:
    # Execute autonomously
    await execute_action(
        action=recommendation.action,
        service_name=service_name
    )
```

**Step 6: Execution & Results**
```
Action executed (restart service)
    ↓
Results posted to xyOps ticket
    ↓
Metrics updated: compute_agent_actions_total{action="restart", result="success"}
    ↓
Traces emitted (OTel instrumentation)
    ↓
Agent completes
```

---

## 7. OBSERVABILITY DESIGN — PROMETHEUS METRICS

### Core Agent Metrics

#### Compute Agent
```prometheus
# Counter: total incidents processed
compute_agent_webhook_received_total{severity, service} 

# Histogram: time to analyze incident
compute_agent_analysis_latency_ms{quantile, service}

# Counter: tickets created
compute_agent_ticket_created_total{severity, scenario_match}

# Counter: actions recommended
compute_agent_recommendations_total{action, approval_required, approved}

# Counter: autonomous actions executed
compute_agent_autonomous_actions_total{action, result}  # result=success|failure

# Counter: specific actions (service restarts)
compute_agent_restarts_executed_total{service, result}

# Counter: pod migrations
compute_agent_migrations_executed_total{reason, result}

# Counter: escalations
compute_agent_escalations_total{reason, domain}

# Gauge: active incidents
compute_agent_active_incidents_gauge{severity}
```

#### Storage Agent (same pattern)
```prometheus
storage_agent_webhook_received_total
storage_agent_actions_total{action, result}
storage_agent_escalations_total
storage_agent_osd_reweights_executed_total
storage_agent_noisy_pvc_throttles_applied_total
storage_agent_pool_rebalances_triggered_total
```

#### Shared Intelligence Engine
```prometheus
# Scenario matching confidence over time
oie_scenario_match_confidence_minutes{domain, scenario}

# Current anomaly scores
oie_anomaly_score{domain, metric_name}

# Forecast results
oie_forecast_days_to_critical{domain, resource}

# Risk scores
oie_risk_score{domain, scenario}

# Recommendations generated
oie_recommendation_generated_total{domain, action, approved}

# Query performance
oie_query_latency_ms{query_type} # compute_features, storage_features, scenario_match, etc
oie_cache_hit_ratio{query_type}

# Anomaly detection
oie_novelty_detection_score{domain}
oie_baseline_deviation{domain, metric}
```

#### Forecasting Metrics
```prometheus
forecast_storage_pool_days_to_full{pool, confidence}
forecast_compute_heap_oom_days{service, confidence}
forecast_trend_up{domain, metric, days_ahead}
forecast_degradation_velocity{domain, resource}  # % per day
```

### Label Strategy

**Domain**:
- `domain="compute"` — CPU, memory, restarts, latency
- `domain="storage"` — OSD, PVC, pool, backfill
- `domain="observability"` — telemetry health

**Cardinality Warnings**:
- Service names OK (10-50 services)
- PVC names OK (100s PVCs)
- Metric names OK (100 unique metrics)
- **AVOID**: per-pod labels (cardinality explosion)
- **AVOID**: high-cardinality resource UUIDs

**Example**: ❌ `metric{pod="abc-12345", node="node-8", replica="3"}` (too specific)  
✅ `metric{service="backend", domain="compute"}` (good)

---

## 8. GRAFANA DASHBOARD ARCHITECTURE

### Dashboard 1: "Agentic AI Operations" (Main)

**Layout**: 4 rows × 3 panels each

#### Row 1: Agent Activity Summary

**Panel 1.1**: Total Autonomous Actions (Stat)
```
Query: sum(rate(compute_agent_autonomous_actions_total[5m]) + 
           rate(storage_agent_autonomous_actions_total[5m]))
Unit: actions/sec
Threshold: >0.5/sec = green, >2/sec = yellow
Shows: how busy agents are
```

**Panel 1.2**: Actions by Type (Stacked Bar)
```
Query: sum by (action_type) 
       (rate(compute_agent_autonomous_actions_total[5m]) + 
        rate(storage_agent_autonomous_actions_total[5m]))
Stacks: restart, reweight, throttle, migrate, rebalance
Time: last 24h
```

**Panel 1.3**: Agent Effectiveness (Gauge)
```
Query: (compute_agent_autonomous_actions_total{result="success"} / 
        compute_agent_autonomous_actions_total) * 100
Unit: %
Threshold: >95% = green (healthy success rate)
```

#### Row 2: Noisy Neighbour Reductions

**Panel 2.1**: CPU Migration Count (Stat)
```
Query: compute_agent_migrations_executed_total{reason="noisy_neighbor", result="success"}
Unit: migrations
Shows: how many noisy CPU pods moved to isolated nodes
```

**Panel 2.2**: PVC Throttles Applied (Stat)
```
Query: storage_agent_noisy_pvc_throttles_applied_total
Unit: throttles
Shows: how many noisy PVCs were rate-limited
```

**Panel 2.3**: Noisy Neighbour Events Timeline (Time Series)
```
Query: rate(compute_agent_migrations_executed_total[5m]) +
       rate(storage_agent_noisy_pvc_throttles_applied_total[5m])
Time: last 7 days
Shows: trend of noisy neighbor incidents
```

#### Row 3: Restarts & Autonomous Remediations

**Panel 3.1**: Pod Restarts (Stat)
```
Query: compute_agent_restarts_executed_total{result="success"}
Unit: restarts
Shows: total service restarts by agent
```

**Panel 3.2**: OSD Reweights (Stat)
```
Query: storage_agent_osd_reweights_executed_total
Unit: reweights
Shows: storage rebalance actions
```

**Panel 3.3**: Last 10 Actions (Table)
```
Query: recent compute + storage agent action logs
Columns: timestamp | agent | service | action | result | duration
Shows: detailed action history with durations
```

#### Row 4: Escalations & Human Intervention

**Panel 4.1**: Open Escalations (Stat)
```
Query: compute_agent_active_incidents_gauge + 
       storage_agent_active_incidents_gauge
Unit: incidents
Threshold: >0 = red (requires attention)
```

**Panel 4.2**: Escalations by Reason (Pie Chart)
```
Query: sum by (reason) (compute_agent_escalations_total + 
                        storage_agent_escalations_total)
Segments: "unknown_anomaly", "approval_denied", "action_failed", "slo_at_risk"
Shows: what causes escalations
```

**Panel 4.3**: Escalation Trend (Time Series)
```
Query: rate(compute_agent_escalations_total[5m]) +
       rate(storage_agent_escalations_total[5m])
Time: last 7 days
Shows: escalation rate over time
```

---

### Dashboard 2: "Scenario Intelligence" (Secondary)

#### Row 1: Scenario Match Confidence

**Panel**: Heatmap of Scenario Matches
```
X-axis: Time
Y-axis: Scenario name (memory_leak, noisy_neighbor, osd_down, pool_full, etc.)
Color: Confidence (0.0-1.0)
Query: oie_scenario_match_confidence_minutes
Shows: which scenarios are active right now, and how confident the engine is
```

#### Row 2: Anomaly Scores

**Panel**: Anomaly Trends
```
Time series: oie_anomaly_score by (domain, metric_name)
Color: red if > 0.7
Shows: which metrics are behaving abnormally
```

#### Row 3: Forecasts

**Panel**: Days to Critical Resources
```
Query: oie_forecast_days_to_critical by (domain, resource)
Bar chart:
  - storage_pool_days_to_full: 14 days
  - compute_memory_oom_days: 30 days
  - disk_fill_days: 21 days
Shows: proactive capacity warnings
```

---

### Dashboard 3: "Agent Decision Deep-Dive" (Optional)

For debugging why agent made a decision

**Panel**: Last Analysis (JSON)
```
Displays JSON response from last OIE analysis:
- scenario matches
- confidence scores
- evidence
- recommendations
```

---

## 9. XYOPS WORKFLOW INTEGRATION

### Compute Agent Workflow (evolved)

Already exists but updated to show OIE insights

```
xyOps Canvas: "Compute Incident Pipeline"

[1] Receive Alert 
    ↓ (webhook → compute-agent)
[2] Fetch Compute Context
    ↓ (OIE: scenario match + anomaly)
[3] AI Analysis & RCA
    ↓ (OIE response)
[4] Generate Recommendation  
    ↓ (OIE: action suggestion)
[5] Approval Gate (if risky)
    ↓ (human reviews evidence)
[6] Execute & Report
    ↓ (ansible runner + metrics)
```

### Storage Agent Workflow (new)

```
xyOps Canvas: "Storage Incident Pipeline"

[1] Receive Storage Alert
    ↓ (webhook → storage-agent)
[2] Fetch Storage Context
    ↓ (OIE: Ceph metrics + forecast)
[3] AI Analysis & Root Cause
    ↓ (OIE: scenario match confidence)
[4] Recommend Storage Action
    ↓ (OIE: reweight, throttle, rebalance, escalate)
[5] Approval Gate (for destructive ops)
    ↓ (human approves OSD reweight, etc.)
[6] Execute & Report
    ↓ (xyOps job runner + Gitea audit)
```

### Knowledge Feedback Loop (optional enhancement)

```
After Action Execution:
  ↓
Measure Outcome:
  - Did service recover?
  - How long did it take?
  - Were there side effects?
  ↓
Update Scenario Engine:
  - Confidence adjustments
  - Evidence strength re-calibration
  - Action success rates
  ↓
Next similar incident:
  - Scenario matches with higher confidence
  - Recommendation confidence improves
```

---

## 10. SAFE AUTONOMY MODEL

### Access Control Matrix

| Action | Compute | Storage | Approval Needed | Rollback Available |
|--------|---------|---------|-----------------|-------------------|
| Restart single pod | ✅ Auto | — | Only if critical | Yes |
| Scale +1 replica | ✅ Auto | — | No | Yes |
| Clear cache | ✅ Auto | — | No | Yes |
| Migrate pod off node | ✅ Gate | — | Always | Yes |
| Restart critical service | ❌ Gate | — | Always | Yes |
| Reweight OSD | — | ✅ Auto | No | Limited |
|Apply IOPS throttle | — | ✅ Auto | No | Yes |
| Trigger pool rebalance | — | ⚠️ Gate | Always | No |
| Add new OSD | — | ❌ Gate | Always | Limited |
| Delete PVC | — | ❌ Gate | Always | No |
| Scale down nodes | ❌ Gate | ❌ Gate | Always | Limited |

### Rules for Autonomy

**Compute Domain — Fully Autonomous** (no approval):
- Service restart (if replica_count ≥ 2)
- Add container replica (scale UP only)
- Clear in-memory cache
- Kill goroutine leak process
- Prerequisite: not critical path service

**Compute — Approval-Gated**:
- Migrate pod off node
- Restart critical service (single replica)
- Modify resource limits
- Drain node
- Prerequisite: human reviews AND alternative ready

**Compute — Human-Only**:
- Delete PVC (data loss risk)
- Modify security groups
- Change TLS certificates
- Scale DOWN nodes (capacity reduction)

**Storage — Fully Autonomous**:
- Apply IOPS throttle to noisy PVC (rate limiting only)
- Reweight OSD (load balancing)
- Prerequisite: pool health check passes

**Storage — Approval-Gated**:
- Trigger full pool rebalance (heavy operation)
- Add new OSD (capacity change)
- Change replica factor
- Retire failing OSD
- Prerequisite: human reviews degradation impact

**Storage — Human-Only**:
- Delete PVC (data loss)
- Shrink pool (capacity reduction)
- Change ceph.conf globally

### Approval Workflow

```
Decision Tree:

1. Agent proposes action
2. Is action in "Fully Autonomous" list?
   YES → Execute immediately, emit metric
   NO → Proceed to 3
3. Is risk_score > threshold?
   YES → Require human approval (create approval ticket)
   NO → Proceed to 4
4. Are all prerequisites met?
   YES → Execute autonomously
   NO → Create ticket for human investigation

Approval Ticket Contents:
  - What action? (exact command)
  - Why? (scenario name + confidence)
  - Evidence (metrics + logs)
  - Estimated resolution time
  - Rollback plan (if available)
  - Buttons: [Approve] [Deny] [Investigate]

Timeout:
  - Auto-deny after 1 hour if no approval
  - Escalate if action becomes critical while waiting
```

---

## 11. IMPLEMENTATION BLUEPRINT

### Final Folder Structure

```
observability-learning/
├── README.md
├── docker-compose.yml                       # UPDATED
├── ARCHITECTURE_DESIGN.md                   # THIS FILE
│
├── shared-intelligence-engine/              # NEW SERVICE
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── pytest.ini
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                          # FastAPI app
│   │   ├── telemetry_access.py              # Prom + Loki client
│   │   ├── feature_extraction.py            # signal extraction
│   │   ├── scenario_engine.py               # load + match scenarios
│   │   ├── anomaly_detection.py             # baseline + novelty
│   │   ├── forecasting.py                   # trend prediction
│   │   ├── risk_scoring.py                  # impact * likelihood
│   │   ├── recommendation_engine.py         # action suggestion
│   │   ├── evidence_generation.py           # explain why
│   │   ├── metrics_exporter.py              # Prometheus export
│   │   ├── background_jobs.py               # proactive analysis
│   │   ├── config.py                        # env-based config
│   │   └── models.py                        # Pydantic schemas
│   ├── scenarios/                           # Scenario YAML files
│   │   ├── schema.yaml                      # JSON Schema for scenarios
│   │   ├── compute/
│   │   │   ├── memory_leak.yaml
│   │   │   ├── noisy_neighbor.yaml
│   │   │   ├── connection_pool_exhaustion.yaml
│   │   │   └── ...
│   │   └── storage/
│   │       ├── osd_down.yaml
│   │       ├── pool_full_forecast.yaml
│   │       ├── noisy_pvc.yaml
│   │       └── ...
│   └── tests/
│       ├── test_scenario_engine.py
│       ├── test_anomaly_detection.py
│       ├── test_forecasting.py
│       └── test_risk_scoring.py
│
├── compute-agent/                           # RENAMED from aiops-bridge
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── pytest.ini
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                          # FastAPI + webhook
│   │   ├── webhook_handler.py               # alert parsing
│   │   ├── pipeline.py                      # 6-step workflow (mostly unchanged)
│   │   ├── approval_workflow.py             # unchanged
│   │   ├── xyops_client.py                  # unchanged
│   │   ├── oie_client.py                    # NEW: calls shared engine
│   │   ├── metrics_exporter.py              # NEW: expose /metrics
│   │   ├── config.py
│   │   └── models.py
│   └── tests/
│       └── (existing tests still pass)
│
├── storage-agent/                           # NEW SERVICE
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── pytest.ini
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── webhook_handler.py
│   │   ├── pipeline.py
│   │   ├── oie_client.py
│   │   ├── xyops_client.py
│   │   ├── storage_actions.py               # execute OSD reweight, etc.
│   │   ├── metrics_exporter.py
│   │   ├── config.py
│   │   └── models.py
│   └── tests/
│
├── sre-reasoning-agent/                     # OPTIONAL (Phase 5)
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py
│   │   ├── llm_client.py                    # OpenAI or local LLM
│   │   ├── reasoning_engine.py              # hypothesis generation
│   │   └── models.py
│   └── tests/
│
├── storage-simulator/                       # NEW (Demo)
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── metrics.py                       # Prometheus metrics
│   └── scenarios/
│       ├── healthy.py
│       ├── osd_down.py
│       ├── pool_full.py
│       └── ...
│
├── prometheus/
│   ├── prometheus.yml                       # UPDATED: new scrape targets
│   └── alert-rules.yml                      # UPDATED: storage + OIE alerts
│
├── alertmanager/
│   └── alertmanager.yml                     # UPDATED: route to both agents
│
├── grafana/
│   └── provisioning/
│       ├── dashboards/
│       │   ├── obs-overview.json            # existing
│       │   ├── agentic-ai-operations.json   # NEW
│       │   └── scenario-intelligence.json   # NEW
│       └── dashboards.yaml                  # provisioning manifest
│
├── otel-collector/, tempo/, loki/, etc.    # UNCHANGED
└── docs/
    ├── ARCHITECTURE.md
    ├── DEPLOYMENT.md
    ├── SCENARIOS.md
    └── TROUBLESHOOTING.md
```

---

## 12. SAFE DEPLOYMENT & MIGRATION PLAN

### Phase 1: Minimum Refactor (Days 1-2)
**Goal**: Extract OIE without breaking compute-agent

**Tasks**:
1. Create `shared-intelligence-engine/` directory
2. Move scenario correlation logic from aiops-bridge → OIE
3. Create OIE `/analyze` endpoint
4. Test OIE independently
5. Update compute-agent to call OIE
6. **Validation**: Compute agent still works end-to-end

**Risk**: Low (compute-agent logic unchanged, just delegating)

### Phase 2: Shared Intelligence Evolution (Days 3-4)
**Goal**: Build out OIE modules (anomaly, forecast, risk scoring)

**Tasks**:
1. Implement anomaly detection module
2. Implement forecasting engine
3. Implement risk scoring
4. Add background job for continuous analysis
5. Expose `/metrics` from OIE
6. Update Prometheus scrape config

**Risk**: Low (compute-agent not affected, OIE runs independently)

### Phase 3: Storage Agent Integration (Days 5-6)
**Goal**: Build storage-agent + storage-simulator

**Tasks**:
1. Create storage-simulator (generates Ceph metrics)
2. Create storage-agent (uses same patterns as compute-agent)
3. Add storage scenarios to OIE
4. Update Alertmanager routing
5. Create storage workflow in xyOps
6. Test storage incident flow end-to-end

**Risk**: Low (storage is new, doesn't affect compute)

### Phase 4: Proactive Analysis & Forecasting (Days 7-8)
**Goal**: Enable continuous degradation prediction

**Tasks**:
1. Enable background jobs in OIE (every 5/15 min)
2. Create tickets proactively for threshold breaches
3. Update Grafana dashboards
4. Add alerts for forecast thresholds

**Risk**: Medium (proactive tickets might overwhelm; start with low thresholds)

### Phase 5: SRE Reasoning Layer (Days 9-10, optional)
**Goal**: LLM-powered interpretation for complex unknowns

**Tasks**:
1. Create sre-reasoning-agent service
2. Integrate with OIE for unknown anomalies
3. Add hypothesis generation
4. Test with complex failure modes

**Risk**: Medium (LLM dependencies, cost, latency)

---

## 13. BUILD ORDER FOR NEXT 10 DAYS

### Day 1-2: Shared Intelligence Engine (Foundation)

#### Day 1 Morning: Setup + Telemetry Access
- [ ] Create `shared-intelligence-engine/` folder structure
- [ ] Create `Dockerfile` (Python 3.11, fastapi, prometheus-client)
- [ ] Create `requirements.txt`
- [ ] Implement `telemetry_access.py`:
  - Prometheus query client (httpx)
  - Loki log client
  - Metric caching (30s TTL)
  - Circuit breaker pattern

#### Day 1 Afternoon: Feature Extraction
- [ ] Implement `feature_extraction.py`:
  - `ComputeFeatures` dataclass
  - `StorageFeatures` dataclass
  - Feature computation functions

#### Day 2 Morning: Scenario Engine
- [ ] Create `scenarios/` directory + YAML files
- [ ] Implement `scenario_engine.py`:
  - Load scenarios from YAML
  - Match logic
  - Confidence scoring
- [ ] Write scenario files for:
  - `compute/memory_leak.yaml`
  - `compute/noisy_neighbor.yaml`
  - `storage/osd_down.yaml`
  - `storage/pool_full.yaml`

#### Day 2 Afternoon: FastAPI + Metrics
- [ ] Implement `main.py` (FastAPI app):
  - `POST /analyze`
  - `POST /forecast`
  - `GET /scenarios`
  - `GET /metrics` (Prometheus)
  - Errors handling + logging
- [ ] Implement `metrics_exporter.py`
- [ ] Write unit tests

---

### Day 3-4: Anomaly & Forecasting Modules

#### Day 3 Morning: Anomaly Detection
- [ ] Implement `anomaly_detection.py`:
  - Baseline computation
  - Z-score calculation
  - Novelty detection (Isolation Forest)
- [ ] Test against real Prometheus data

#### Day 3 Afternoon: Forecasting
- [ ] Implement `forecasting.py`:
  - Linear regression
  - Time-to-critical calculation
  - Confidence estimation
- [ ] Test with storage fill scenarios

#### Day 4 Morning: Risk & Recommendation
- [ ] Implement `risk_scoring.py`
- [ ] Implement `recommendation_engine.py`
- [ ] Implement `evidence_generation.py`

#### Day 4 Afternoon: OIE Testing + Integration
- [ ] Full integration test (alert → analysis)
- [ ] Update docker-compose to include OIE
- [ ] Update Prometheus config (scrape OIE)
- [ ] Deploy to local stack

---

### Day 5-6: Compute Agent Retrofit + Storage Agent

#### Day 5 Morning: Compute Agent Retrofit
- [ ] Rename `aiops-bridge/` → `compute-agent/`
- [ ] Implement `oie_client.py`
- [ ] Update `main.py` to call OIE instead of inline logic
- [ ] Add `metrics_exporter.py` + `/metrics` endpoint
- [ ] Test end-to-end with existing flow

#### Day 5 Afternoon: Storage Simulator
- [ ] Create `storage-simulator/` folder
- [ ] Implement Prometheus metrics endpoint
- [ ] Implement scenarios (osd_down, pool_full, noisy_pvc)
- [ ] Add `/scenario/{name}` API

#### Day 6 Morning: Storage Agent
- [ ] Create `storage-agent/` folder (mirror compute-agent pattern)
- [ ] Implement webhook handler + pipeline
- [ ] Implement `oie_client.py`
- [ ] Implement `metrics_exporter.py`

#### Day 6 Afternoon: Integration & Testing
- [ ] Update Alertmanager routing (route storage alerts)
- [ ] Create storage workflow in xyOps
- [ ] Test end-to-end: storage-simulator → storage-agent → xyOps
- [ ] Verify both agents work independently

---

### Day 7-8: Monitoring, Dashboards, Proactive Analysis

#### Day 7 Morning: Prometheus & Alerts
- [ ] Update `prometheus.yml` (scrape targets + alert rules)
- [ ] Add storage alerts (CephOSDDown, PoolFull, etc.)
- [ ] Add OIE alerts (AnomalyScoreHigh, ForecastCritical)
- [ ] Test alert firing

#### Day 7 Afternoon: Grafana Dashboards
- [ ] Create `agentic-ai-operations.json`:
  - Row 1: Agent Activity
  - Row 2: Noisy Neighbour Reductions
  - Row 3: Autonomous Actions
  - Row 4: Escalations
- [ ] Create `scenario-intelligence.json`:
  - Scenario match heatmap
  - Anomaly scores
  - Forecasts

#### Day 8 Morning: Background Jobs
- [ ] Implement `background_jobs.py` in OIE:
  - Every 5 min: detect anomalies proactively
  - Every 15 min: update forecasts
  - Every hour: refresh scenario metrics
- [ ] Write background job tests

#### Day 8 Afternoon: Testing & Documentation
- [ ] Integration tests (all agents together)
- [ ] Load test (Prometheus query performance)
- [ ] Update README + docs
- [ ] Deploy full stack and validate

---

### Day 9-10: Optional SRE Reasoning Agent + Polish

#### Day 9 Morning: SRE Reasoning Agent (optional)
- [ ] Create `sre-reasoning-agent/` folder
- [ ] Implement LLM client (OpenAI or local model)
- [ ] Implement hypothesis generator

**OR Skip to Day 10 if not doing SRE agent**

#### Day 9 Afternoon / Day 10 Morning: Final Testing
- [ ] Chaos test: trigger memory leak, watch compute-agent
- [ ] Chaos test: trigger OSD down, watch storage-agent
- [ ] Chaos test: slow degradation, watch proactive analysis
- [ ] Performance test: OIE query latency under load
- [ ] Negative test: LLM unavailable, systems still work

#### Day 10 Afternoon: Cleanup & Documentation
- [ ] Write runbooks for common alerts
- [ ] Document agent decision explainability
- [ ] Record demo video (alert → xyOps → Tempo trace)
- [ ] Clean up code + linting
- [ ] Final git commit + push

---

## 14. ANTI-PATTERNS TO AVOID

### ❌ Don't: Copy-Paste Agent Logic
**Problem**: Compute agent and storage agent share 60% code (webhook, xyOps calling, metrics)

**Solution**: Extract to shared libraries:
```python
# shared_lib/
├── webhook_handler.py
├── xyops_client.py
├── metrics_base.py
├── oie_client.py
```

Both agents import and extend.

---

### ❌ Don't: LLM as Foundation
**Problem**: If LLM fails, entire agent stalls

**Solution**: Deterministic rules + patterns first, LLM for edge cases.
```
Decision tree:
  1. Check scenario thresholds (deterministic) → 90% coverage
  2. If no match, compute anomaly score (statistical) → 9% coverage
  3. If unknown, call LLM (optional) → 1% coverage
```

---

### ❌ Don't: Expose Every Metric
**Problem**: Metrics cardinality explosion (label combinations)

**Solution**: Export only high-value metrics with bounded labels:
```prometheus
✅ compute_agent_actions_total{action, result}
❌ compute_agent_action_duration_ms{action, service, pod, node, replica}
```

---

### ❌ Don't: Block on LLM Calls
**Problem**: Agent latency = LLM API latency (network timeouts)

**Solution**:
```python
# Async, non-blocking
analysis = await oie.analyze(...)  # deterministic, fast
enrichment = await llm_enrich(analysis)  # async, optional
# Return analysis immediately, enrich in background for next request
```

---

### ❌ Don't: Approve Every Risky Action
**Problem**: Approval ticket fatigue, humans stop reading

**Solution**: Tiered approval:
```
Autonomous (no approval):
  - restart pod (replica_count ≥ 2)
  - scale +1 replica
  
Fast Approval (1-click):
  - 75% confidence scenario match
  - low-risk action
  - evidence presented clearly
  
Full Review:
  - unknown anomaly
  - high-risk action
  - evidence conflicting
```

---

### ❌ Don't: Forget Rollback
**Problem**: Executed action breaks things, no way back

**Solution**: For each action, define:
```yaml
action: restart_service
rollback:
  - command: revert_config
  - command: restore_snapshot
  - estimated_time: 5m
  - runbook_url: https://...
```

---

### ❌ Don't: Mix Domains in Single Scenario
**Problem**: Scenario tries to match compute + storage metrics, low confidence

**Solution**: Separate by domain:
```yaml
# compute/memory_leak.yaml — only uses compute features
# storage/pool_fill.yaml — only uses storage features
```

Cross-domain scenarios (rare) go in `shared/` with explicit dual-domain label.

---

### ❌ Don't: Ignore Baseline Drift
**Problem**: Normal ranges change after deployment, old thresholds stale

**Solution**:
```python
# Every 7 days, recompute baselines
# Alert if baseline shifts >10% (suggests deployment side-effect)
# Confidence score adjusts based on baseline age
```

---

### ❌ Don't: Execute Without Evidence Trail
**Problem**: Action executed, no one knows why

**Solution**: Every action creation includes:
```json
{
  "action": "restart_service",
  "evidence": {...},
  "scenario_match": {name, confidence},
  "risk_score": 0.82,
  "approved_by": "alice@company.com",
  "approval_timestamp": "2026-03-20T10:23:00Z",
  "trace_id": "abc123def"  # Link to Tempo
}
```

---

### ❌ Don't: Hardcode LLM Provider
**Problem**: Locked into OpenAI, cost or API changes block you

**Solution**:
```python
# config.py
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # openai | anthropic | local

if LLM_PROVIDER == "local":
    llm_client = LocalOllamaClient()
elif LLM_PROVIDER == "anthropic":
    llm_client = AnthropicClient()
else:
    llm_client = OpenAIClient()
```

Support pluggable LLM.

---

## 15. FINAL RECOMMENDATIONS

### What to Build First RIGHT NOW

**Day 1**: Start with `shared-intelligence-engine` — it's the foundation.

**Immediately after**: Retrofit `compute-agent` to use OIE (2 days, low risk).

**Then**: Build `storage-agent` + simulator (2 days, new capability).

**Finally**: Dashboards + proactive analysis (2-3 days, polish).

### Sample LLM Strategy

**Option 1: OpenAI + Cost Control** (my recommendation for demo)
```python
# Use gpt-4o-mini (cheap, fast)
# Call only when confidence too low to decide
# Cache results for 1 hour (reduce calls)
# Fallback to deterministic if LLM unavailable
```

**Option 2: Local LLM** (for cost-conscious enterprise)
```python
# Deploy Ollama sidecar with Llama 2 or Mistral 7B
# Runs locally, no API calls, <1s latency
# Tradeoff: lower quality interpretation
# Good for: hypothesis generation, not critical decisions
```

**My choice**: Start with OpenAI (it's better), switch to local if cost becomes issue.

### Explainability Strategy

Every decision must be answerable:
- **What happened?** Alert name + metric spikes
- **Why did we match this scenario?** Confidence + evidence
- **What are we doing?** Action + reasoning
- **Why approve/deny?** Risk score + prerequisite check
- **What if it fails?** Rollback plan + alternative actions

**Implement**:
```python
class Decision:
    what: str       # "memory_leak_emergence"
    why: dict       # {confidence, evidence, risk_score}
    action: str     # "restart_service"
    reasoning: str  # link to scenario YAML + confidence breakdown
    confidence: float
    trace_id: str   # link to Tempo for investigation
```

When human reviews decision in xyOps, they see all this.

---

## SUMMARY TABLE

| Aspect | Current State | After Phase 1-4 | After Phase 5 |
|--------|---------------|-----------------|---------------|
| **Agents** | 1 (compute) | 2 (compute + storage) | 2 + optional SRE |
| **Shared logic** | None (no storage) | 100% via OIE | 100% via OIE |
| **Scenarios known** | ~5 (ad-hoc) | 20+ (YAML files) | 20+ + LLM interpretation |
| **Proactive analysis** | No | Yes (background jobs) | Yes + forecasting |
| **Autonomous actions** | 3-5 | 10-15 | 10-15 |
| **Metrics clarity** | Low | High | Very high |
| **Explainability** | Manual | Automatic (evidence) | Automatic + narrative |
| **Enterprise-ready** | No | Mostly | Yes |
| **Extensibility** | Hard | Easy (add scenarios) | Easy (add domains) |

---

## FINAL MESSAGE

You've built a solid observability + AIOps foundation. The challenge now is intelligent **scale** without **duplication**.

This architecture:
✅ Preserves your working system  
✅ Avoids duplicating intelligence between agents  
✅ Makes every decision observable + explainable  
✅ Supports 20+ degradation patterns  
✅ Enables continuous proactive analysis  
✅ Sets foundation for network/database/security agents later  

**Start tomorrow** with Phase 1 (OIE). By day 3, you'll have compute-agent + OIE working together. By day 7, storage-agent joins. By day 10, you have a unified multi-agent platform.

Ready? 🚀

---

**Document Version**: 1.0  
**Date**: March 2026  
**Status**: Design Complete, Ready for Implementation
