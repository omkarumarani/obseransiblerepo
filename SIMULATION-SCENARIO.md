# AIOps Platform — Simulation Scenario & Agent Workflow

## Simulation Scenario Overview

The AIOps platform simulates a **banking production environment** with intelligent agents that autonomously detect, analyze, and remediate infrastructure incidents.

### Simulated Environment

**Core Services (Bank Infrastructure):**
- `frontend-api` (port 8080) — Customer-facing web API
- `backend-api` (port 8081) — Core banking logic and database access
- `storage-simulator` (port 9200) — Simulated Ceph storage cluster

**Fault Injection:**
- `troublemaker` service randomly injects realistic production failures:
  - CPU spikes (mining processes)
  - Memory leaks (gradual OOM scenarios)
  - Disk I/O storms (fsync floods)
  - Network latency injection
  - Connection pool exhaustion
  - Storage OSD failures

**Observability Stack:**
- Prometheus (metrics collection + alerting)
- Loki (log aggregation)
- Tempo (distributed tracing)
- Grafana (unified visualization)
- Alertmanager (alert routing)

**Remediation Infrastructure:**
- Gitea (Git repository for Ansible playbooks)
- Ansible Runner (executes/simulates remediation playbooks)
- xyOps (ticketing + approval canvas)

---

## How Agents Work Together

### 6-Agent Pipeline Architecture

When Prometheus detects an anomaly (e.g., error rate > 40%), it fires an alert to **Alertmanager**, which routes it to the appropriate agent service based on domain:

```
Prometheus Alert → Alertmanager → compute-agent (compute domain)
                                ├→ storage-agent (storage domain)
```

### Agent 1: Alert Intake & Session Creation

**Responsibility:** Receive Alertmanager webhook and create incident tracking

**Actions:**
1. Parse alert payload (service name, severity, dashboard URL)
2. Create pipeline session (keyed by service name)
3. Create skeleton ticket in xyOps
4. Assign OTel trace ID for end-to-end observability
5. Post initial comment: "🤖 AI pipeline starting..."

**Output:**
```json
{
  "session_id": "frontend-api",
  "ticket_num": 1234,
  "status": "pipeline_started"
}
```

---

### Agent 2: Log Collection

**Responsibility:** Query Loki for relevant logs around the incident timeframe

**Actions:**
1. Build LogQL query: `{service_name="frontend-api"} |= "error" | json`
2. Fetch logs from 15 minutes before alert start
3. Filter for error-level messages
4. Extract stack traces and error patterns
5. Store in session for Agent 4

**Example LogQL:**
```
{service_name="frontend-api", level="error"} 
  | json 
  | line_format "{{.timestamp}} {{.message}}"
```

**Output:**
```text
2026-03-20T10:15:23Z ERROR: DatabaseConnectionError: connection pool exhausted
2026-03-20T10:15:24Z ERROR: TimeoutError: query execution exceeded 30s
2026-03-20T10:15:25Z ERROR: HTTP 503 returned to client
```

---

### Agent 3: Metric Collection

**Responsibility:** Query Prometheus for contextual metrics

**Actions:**
1. Build PromQL queries for:
   - Error rate trend: `rate(http_requests_total{status="5xx"}[5m])`
   - Latency percentiles: `histogram_quantile(0.99, ...)`
   - Resource utilization: CPU, memory, disk I/O
   - Dependency health: database connections, cache hit rate
2. Fetch 1-hour historical context
3. Identify anomaly patterns (sudden spikes, gradual degradation)

**Example PromQL:**
```
rate(http_requests_total{
  service_name="frontend-api",
  status=~"5.."
}[5m])
```

**Output:**
```json
{
  "error_rate_5m": 0.42,
  "p99_latency_ms": 3400,
  "cpu_usage_percent": 85,
  "db_connections_active": 95,
  "db_connections_max": 100
}
```

---

### Agent 4: AI Analysis & Root Cause Detection

**Responsibility:** Synthesize logs + metrics → root cause + remediation plan

**Actions:**
1. **Scenario Matching** (deterministic):
   - Load scenario catalog (`obs-intelligence/scenarios/*.yaml`)
   - Match symptoms against known patterns
   - Return pre-defined RCA + action if confidence > 90%

2. **AI Analysis** (if no scenario match):
   - Build prompt with logs + metrics + context
   - Call GPT-4o with system prompt:
     ```
     You are a senior SRE analyzing a production incident.
     Service: {service_name}
     Alert: {alert_name}
     Logs: {logs}
     Metrics: {metrics}
     
     Provide:
     1. Root cause (1-2 sentences)
     2. Recommended action (restart_service | scale_resources | ...)
     3. Remediation command (Ansible playbook or API call)
     4. Confidence score (0.0-1.0)
     ```
   - Parse GPT-4 response → structured RCA

3. **Risk Scoring**:
   - Calculate risk score (0.0-1.0) based on:
     - Severity (critical=1.0, warning=0.5, info=0.1)
     - Impact scope (production=1.0, staging=0.3)
     - Error rate magnitude
   - Map to risk level: LOW | MEDIUM | HIGH | CRITICAL

**Output:**
```json
{
  "root_cause": "Database connection pool exhausted due to slow queries",
  "recommended_action": "restart_service",
  "remediation": "ansible-playbook restart-db-pool.yml",
  "confidence": 0.89,
  "risk_score": 0.72,
  "risk_level": "high",
  "scenario_matched": "db-connection-pool-exhaustion",
  "provider": "scenario-catalog"  // or "gpt-4o"
}
```

---

### Agent 5: Incident Ticket Enrichment

**Responsibility:** Replace skeleton ticket with full AI-enriched incident report

**Actions:**
1. Build rich Markdown ticket body:
   - Executive summary
   - Metrics table (before/after)
   - Root cause analysis
   - Recommended remediation (Ansible playbook)
   - Rollback plan
   - GitHub PR suggestion (if code change needed)
2. Update xyOps ticket via REST API
3. Post comment: "📋 Full incident report ready"

**Ticket Body Structure:**
```markdown
## Incident Summary
**Alert:** HighErrorRate  
**Service:** frontend-api  
**Risk:** HIGH (0.72)  
**Root Cause:** Database connection pool exhausted

## Metrics
| Metric | Before | During | Threshold |
|--------|--------|--------|-----------|
| Error Rate | 0.02 | 0.42 | 0.10 |
| P99 Latency | 120ms | 3400ms | 500ms |

## Root Cause Analysis
Database connection pool (max 100) reached capacity due to slow queries 
(avg 8.2s) holding connections open. Query plan shows missing index on 
`transactions.user_id` column causing full table scans.

## Recommended Remediation
1. Immediate: Restart DB connection pool (clears stale connections)
2. Short-term: Add index on `transactions.user_id`
3. Long-term: Implement connection timeout (30s max)

**Ansible Playbook:** `compute/restart-db-pool.yml`
```

---

### Agent 6: Approval Gateway & Autonomy Decision

**Responsibility:** Decide autonomous execution vs. human approval

**Actions:**
1. **Load service tier** from registry:
   - `production` tier: 10 successful approvals needed for autonomy
   - `staging` tier: 5 approvals
   - `development` tier: 2 approvals
   - `sandbox` tier: 1 approval

2. **Check autonomy rules** (6-level hierarchy):
   ```
   Level 1: HUMAN_ONLY flag → block (e.g., "delete production database")
   Level 2: APPROVAL_REQUIRED env var → gate
   Level 3: Risk ceiling (risk > 0.8) → gate
   Level 4: Trust score check → if trust < threshold, gate
   Level 5: Tier override → production always gated first time
   Level 6: AUTONOMOUS → execute immediately
   ```

3. **Path A: AUTONOMOUS** (trust score met):
   - Post comment: "✅ Autonomous execution authorized (trust: 12/10)"
   - Execute remediation immediately
   - Record outcome in approval history (JSONL)
   - Create Git PR with audit trail
   - Auto-merge PR if tier allows (staging/dev/sandbox)

4. **Path B: APPROVAL_GATED** (trust score not met):
   - Create gated approval ticket in xyOps
   - Add interactive buttons: [Approve] [Reject] [Escalate]
   - Post comment: "⏸️ Awaiting human approval (trust: 3/10)"
   - Wait for human decision via webhook callback
   - On approval: execute remediation + record success
   - On reject: escalate to SRE team

**Autonomy Decision Logic:**
```python
if action_type == "HUMAN_ONLY":
    return "HUMAN_ONLY"
elif risk_score > 0.8:
    return "APPROVAL_GATED"
elif trust_score < tier_threshold:
    return "APPROVAL_GATED"
elif tier == "production" and first_time_action:
    return "APPROVAL_GATED"
else:
    return "AUTONOMOUS"
```

**Output (Autonomous):**
```json
{
  "decision": "AUTONOMOUS",
  "trust_score": 12,
  "trust_threshold": 10,
  "execution_started": true,
  "pr_url": "http://gitea:3002/aiops-org/ansible-playbooks/pulls/42",
  "pr_merged": true
}
```

**Output (Gated):**
```json
{
  "decision": "APPROVAL_GATED",
  "trust_score": 3,
  "trust_threshold": 10,
  "approval_ticket_num": 1235,
  "approval_url": "http://xyops:5522/tickets/1235",
  "reason": "Insufficient trust history (3/10 approvals)"
}
```

---

## Trust-Based Learning Loop

### How Services Graduate to Autonomy

```
Incident 1:  Trust=0/10  →  APPROVAL_GATED  →  Human approves  →  Success  →  Trust=1/10
Incident 2:  Trust=1/10  →  APPROVAL_GATED  →  Human approves  →  Success  →  Trust=2/10
...
Incident 10: Trust=9/10  →  APPROVAL_GATED  →  Human approves  →  Success  →  Trust=10/10
Incident 11: Trust=10/10 →  AUTONOMOUS      →  Auto-executes   →  Success  →  Trust=11/10
Incident 12: Trust=11/10 →  AUTONOMOUS      →  Auto-executes   →  Success  →  Trust=12/10
```

**Trust Score Calculation:**
```python
trust_score = (successful_approvals * success_rate) / total_decisions
success_rate = successful_executions / (successful + failed + partial)
```

**Trust Decay:**
- Each failure reduces trust by 2 points
- Trust never drops below 0
- After 3 consecutive failures, service demoted to APPROVAL_GATED

---

## Multi-Agent Collaboration Flow

### End-to-End Timeline Example

**Incident:** Frontend API error rate 42%, database connection pool exhausted

```
T+0s    Agent 1:  Alert received → Session created → Ticket #1234
T+2s    Agent 2:  Loki query → 47 error logs collected
T+5s    Agent 3:  Prometheus query → 12 metrics collected
T+8s    Agent 4:  Pattern match → Scenario "db-connection-pool-exhaustion" (confidence 0.94)
T+10s   Agent 5:  Ticket enriched → GitHub PR suggested
T+12s   Agent 6:  Trust check (7/10) → APPROVAL_GATED → Ticket #1235 created
T+3m    Human:    Reviewed → [Approve] clicked
T+3m5s  Agent 6:  Ansible playbook executed → DB pool restarted
T+3m20s System:   Error rate dropped to 0.02% → Alert resolved
T+3m25s Agent 6:  PR merged → Trust updated (8/10) → xyOps comment posted
```

### Agent Communication

Agents communicate via:
1. **Shared session state** (in-memory dictionary keyed by session_id)
2. **xyOps ticket comments** (visible audit trail)
3. **REST API endpoints** (each agent is an HTTP endpoint)
4. **OTel tracing** (all inter-agent calls traced with `trace_id`)

---

## Key Differentiators

### 1. Graduated Autonomy (Not Binary)
- Traditional: human-in-loop OR fully autonomous (risky)
- AIOps: gradual trust-building per service + action type

### 2. Multi-Signal Analysis
- Logs + Metrics + Traces + Historical patterns
- Not just "metric crossed threshold"

### 3. Git-Backed Audit Trail
- Every remediation = Git PR with diff
- Immutable history of what changed and why

### 4. Observability-Native
- Every agent action instrumented with OTel
- trace_id links Prometheus → Tempo → Loki → Gitea → xyOps

### 5. Scenario Catalog (Deterministic Fallback)
- 20 known scenarios (10 compute, 10 storage)
- Fast path: pattern match → execute (no AI needed)
- Slow path: novel incident → GPT-4 analysis

---

## Failure Scenarios Tested

### Compute Domain (Handled by `compute-agent`)
1. `high-error-rate` — HTTP 5xx > 10%
2. `high-cpu-usage` — CPU > 80% sustained
3. `memory-leak` — Gradual memory growth
4. `disk-full` — Filesystem > 90%
5. `connection-pool-exhaustion` — DB connections maxed
6. `latency-spike` — P99 > 5s
7. `noisy-neighbour` — One process starving others
8. `thread-leak` — Thread count growing unbounded
9. `dns-resolution-failure` — Service discovery broken
10. `tls-cert-expiry` — Certificate expiring in <7 days

### Storage Domain (Handled by `storage-agent`)
1. `ceph-osd-down` — One OSD offline
2. `ceph-multi-osd-failure` — Two+ OSDs offline
3. `ceph-pool-full` — Pool > 85% capacity
4. `ceph-pool-near-full` — Pool > 70% capacity
5. `pvc-high-latency` — PVC write latency > 100ms
6. `pvc-noisy-neighbour` — One PVC saturating OSD
7. `ceph-pg-degraded` — Placement groups not optimal
8. `ceph-slow-ops` — Operations queued > 30s
9. `pvc-iops-limit` — IOPS quota exceeded
10. `ceph-network-partition` — Mon quorum lost

---

## Observability & Monitoring

### Grafana Dashboards
1. **Agentic AI Operations** — Pipeline execution metrics
2. **Trust Score Heatmap** — Per-service autonomy progress
3. **Approval History Timeline** — Human decisions over time
4. **Remediation Success Rate** — Per action type

### Prometheus Metrics Exposed
```
compute_agent_actions_total{action_type}
compute_agent_autonomous_actions_total
compute_agent_approval_required_total
compute_agent_ai_analysis_total{status="ai"|"deterministic"}
compute_agent_alert_processing_seconds (histogram)
```

---

## Summary

The AIOps platform is a **self-healing production environment simulator** where:

✅ **Realistic failures** are injected by `troublemaker`  
✅ **6 AI agents** collaborate to detect → analyze → remediate incidents  
✅ **Trust-based autonomy** enables gradual graduation from human-in-loop to fully autonomous  
✅ **Full observability** with OTel traces linking every decision  
✅ **Git-backed audit trail** ensures every change is traceable  
✅ **Scenario catalog + GPT-4** provide fast deterministic + slow AI analysis paths  

The goal: **Demonstrate autonomous incident response** that is:
- **Safe** (trust-based graduation, not blind automation)
- **Transparent** (full audit trail, OTel tracing, ticket comments)
- **Explainable** (RCA + confidence scores + scenario matching)
- **Production-ready** (handles 20+ realistic failure scenarios)

---

**Next Steps:**
1. Run `docker compose up --build --profile troublemaker`
2. Open Grafana → "Agentic AI Operations" dashboard
3. Watch alerts fire → agents execute → tickets update → remediation complete
4. Monitor trust scores increase as services graduate to autonomy
5. Use new **AIOps Command Center** (http://localhost:3000) to visualize agent workflow in real-time
