# AIOps Observability Platform — Knowledge Base

> **What is this?** A production-grade, self-healing AIOps platform built on a real observability
> stack. When something breaks, the system automatically detects it, diagnoses the root cause,
> generates a fix, tests it, asks for human approval, and runs it — fully autonomous for low-risk
> actions, human-gated for high-risk ones.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture — The Four Layers](#2-architecture--the-four-layers)
3. [Service Map](#3-service-map)
4. [The Full Agentic Loop (End-to-End)](#4-the-full-agentic-loop-end-to-end)
5. [Chaos Engineering — Troublemaker Scenarios](#5-chaos-engineering--troublemaker-scenarios)
6. [Prometheus Alert Rules](#6-prometheus-alert-rules)
7. [The 6-Step Agent Pipeline](#7-the-6-step-agent-pipeline)
8. [The Intelligence Engine (obs-intelligence)](#8-the-intelligence-engine-obs-intelligence)
9. [Scenario Catalog — All 20 Scenarios](#9-scenario-catalog--all-20-scenarios)
10. [Risk Scoring Model](#10-risk-scoring-model)
11. [The Validate-First Ansible Workflow](#11-the-validate-first-ansible-workflow)
12. [Autonomy Rules Engine](#12-autonomy-rules-engine)
13. [Agentic AI Properties — Evidence](#13-agentic-ai-properties--evidence)
14. [What Is Fully Working](#14-what-is-fully-working)
15. [Technology Stack](#15-technology-stack)
16. [Configuration Reference](#16-configuration-reference)
17. [Learning Path This Demonstrates](#17-learning-path-this-demonstrates)

---

## 1. System Overview

```
Traffic / Chaos
      │
      ▼
Demo Apps (frontend-api, backend-api)
      │  OTel spans + metrics + logs
      ▼
Observability Core (OTel Collector → Prometheus, Tempo, Loki, Grafana)
      │  alert webhooks
      ▼
AIOps Intelligence Layer (compute-agent, storage-agent, obs-intelligence)
      │  tickets + approvals + playbooks
      ▼
Action / Execution Layer (xyOps, Ansible Runner, Gitea)
```

The platform demonstrates the complete journey from raw telemetry to automated remediation with
a human-in-the-loop approval gate backed by a Git PR audit trail.

---

## 2. Architecture — The Four Layers

### Layer 1 — Demo Applications
Services that emit real OpenTelemetry signals (traces, metrics, logs) to demonstrate what a
production microservices system looks like. Deliberately built with fault injection endpoints
(`/error`, `/slow`, `/backend-error`) so chaos can be induced on demand.

### Layer 2 — Observability Core
The full "three pillars + one" observability stack:

| Signal | Collector | Storage | Query |
|---|---|---|---|
| Metrics | OTel Collector → Prometheus | Prometheus TSDB | PromQL |
| Traces | OTel Collector → Tempo | Tempo (TraceQL) | TraceQL / Grafana |
| Logs | OTel Collector → Loki | Loki (log chunks) | LogQL |
| Dashboards | — | — | Grafana (cross-linked exemplars) |

Alertmanager evaluates Prometheus rules every 15 seconds, groups by `domain` label, and routes
webhooks to the appropriate AIOps agent.

### Layer 3 — AIOps Intelligence Layer
Three specialised services:

- **compute-agent** (:9000) — handles compute-domain alerts (latency, errors, traffic, CPU)
- **storage-agent** (:9001) — handles storage-domain alerts (OSD failures, pool capacity, PVC IO)
- **obs-intelligence** (:9100) — shared intelligence engine used by both agents; runs background
  anomaly detection and forecasting loops 24/7, dispatches predictive pre-alerts before Prometheus fires

### Layer 4 — Action / Execution Layer

- **xyOps** (:5522) — AIOps platform; incident tickets, visual workflow canvas, approval events
- **ansible-runner** (:8090) — executes Ansible playbooks; supports `POST /validate` (dry-run) and
  `POST /run` (real or simulated execution)
- **gitea** (:3002) — self-hosted Git; every approved remediation creates a PR (audit trail)

---

## 3. Service Map

| Service | Port | Profile | Purpose |
|---|---|---|---|
| `otel-collector` | 4317/4318/8889/13133 | always | Central telemetry hub — OTLP in, Prom/Tempo/Loki out |
| `prometheus` | 9090 | always | Metrics TSDB + alert evaluation |
| `tempo` | 3200 | always | Distributed trace storage |
| `loki` | 3100 | always | Log aggregation |
| `grafana` | 3001 | always | Dashboards (cross-linked: metric → trace → log) |
| `alertmanager` | 9093 | always | Alert dedup, grouping, webhook routing |
| `frontend-api` | 8080 | always | Demo user-facing FastAPI app; OTel instrumented |
| `backend-api` | 8081 | always | Demo backend FastAPI app; OTel instrumented |
| `loadgen` | — | loadgen | Steady background traffic generator |
| `troublemaker` | 8088 | troublemaker | Chaos traffic generator (8 fault scenarios) |
| `compute-agent` | 9000 | always | AIOps agent — compute domain |
| `storage-agent` | 9001 | always | AIOps agent — storage domain |
| `obs-intelligence` | 9100 | always | Shared intelligence engine |
| `storage-simulator` | 9200 | always | Ceph scenario emulator (Prometheus metrics) |
| `xyops` | 5522/5523 | always | AIOps platform (tickets, workflows, approval) |
| `ansible-runner` | 8090 | always | Playbook executor |
| `gitea` | 3002 | always | Self-hosted Git (PR audit trail) |

---

## 4. The Full Agentic Loop (End-to-End)

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │  1.  Traffic anomaly appears in demo app (or chaos injected)        │
 │  2.  OTel Collector scrapes metrics; Prometheus evaluates rules      │
 │  3.  Alert fires → Alertmanager groups by domain                    │
 │  4.  Alertmanager POSTs webhook to domain agent                     │
 │  5.  Agent runs 6-step pipeline (logs → metrics → analyze → ticket) │
 │  6.  obs-intelligence: feature extract → scenario match → risk score│
 │       → SRE reasoning → LLM narrative (GPT-4o / Claude fallback)    │
 │  7.  xyOps incident ticket enriched with RCA + evidence + playbook  │
 │  8.  Approval gate: pre-validate playbook → Gitea branch + PR       │
 │  9.  xyOps approval ticket: "Merge PR on Gitea, then Approve here"  │
 │ 10.  Human merges PR (authorisation in Git history)                  │
 │ 11.  Human clicks Approve in xyOps                                  │
 │ 12.  Agent checks PR is merged → POST /run to ansible-runner        │
 │ 13.  Execution result posted back to xyOps incident ticket          │
 │ 14.  Outcome recorded → obs-intelligence scenario outcome counter   │
 │ 15.  Background loops: record feeds future confidence scores        │
 └─────────────────────────────────────────────────────────────────────┘
```

For **low-risk autonomous actions** (e.g. reduce OTel sampling, throttle noisy PVC), steps 8–11
are skipped and Ansible runs immediately without human involvement.

---

## 5. Chaos Engineering — Troublemaker Scenarios

The `troublemaker` service injects 8 realistic failure modes with configurable weights:

| Scenario | Target Endpoints | What It Creates |
|---|---|---|
| `steady_normal` | `/ok`, `/backend-ok` | Baseline — flat rate, low latency |
| `burst_traffic` | All endpoints | Vertical spike in `http_requests_total`; triggers `TrafficSpike` alert |
| `latency_spike` | `/slow`, `/backend-slow` | p95/p99 diverge from p50; triggers `HighP99Latency` alert |
| `error_spike` | `/error` | Error counter spikes; triggers `HighErrorRate` / `CriticalErrorRate` |
| `backend_failure` | `/backend-error` | 3-span ERROR waterfall visible in Tempo |
| `slow_backend` | `/backend-slow` | Downstream latency blame visible in Tempo span attribution |
| `mixed_chaos` | Random | Realistic noisy real-world baseline |
| `slow_burn` | Alternating | Gradual latency drift over multiple Prometheus scrape windows |

**Feature flags** (env vars): `ENABLE_BURSTS`, `ENABLE_ERRORS`, `ENABLE_SLOW_CALLS`

**Audit log**: every run appended to `/data/scenario_schedule.csv` with columns:
`timestamp_start`, `timestamp_end`, `scenario`, `endpoints`, `request_count`, `burst_size`,
`ok`, `err`, `avg_ms`, `max_ms`, `notes`

Query via `GET http://localhost:8088/scenarios?json` to correlate anomalies with exact chaos events.

**Storage simulator** (complement for storage domain):

| Scenario | What it emits |
|---|---|
| `healthy` | Normal `storage_osd_up=3`, `storage_pool_usage_percent=40` |
| `osd_down` | `storage_osd_up=2` → triggers `CephOSDDown` alert |
| `multi_osd_failure` | `storage_osd_up=1` → triggers `CephMultipleOSDsDown` alert |
| `pool_full` | `storage_pool_usage_percent=99` → triggers `CephPoolFullCritical` |
| `latency_spike` | `storage_latency_ms=850` → triggers `CephHighLatency` |
| `noisy_pvc` | `storage_pvc_iops=9500` → triggers `CephNoisyPVC` |

Inject via `POST http://localhost:9200/scenario/{name}`

---

## 6. Prometheus Alert Rules

15 alert rules across 2 domains (evaluated every 15s):

### Compute Domain

| Alert | Severity | Condition | For |
|---|---|---|---|
| `HighErrorRate` | warning | error rate > 5% | 1 min |
| `CriticalErrorRate` | critical | error rate > 20% | 30s |
| `HighP99Latency` | warning | p99 > 3s | 2 min |
| `HighMedianLatency` | warning | p50 > 1s | 3 min |
| `TrafficSpike` | warning | current rate > 5× 10-min baseline | 30s |
| `CollectorDroppingData` | warning | OTel collector drop rate > 0 | 2 min |

### Storage Domain (`domain: storage` label)

| Alert | Severity | Condition | For |
|---|---|---|---|
| `CephOSDDown` | warning | any OSD `up=0` | 30s |
| `CephMultipleOSDsDown` | critical | ≥2 OSDs down | 30s |
| `CephPoolNearFull` | warning | pool usage > 75% | 2 min |
| `CephPoolFullCritical` | critical | pool usage > 90% | 30s |
| `CephHighLatency` | warning | storage latency > 500ms | 2 min |
| `CephNoisyPVC` | warning | PVC IOPS > 8000 | 1 min |
| `CephClusterDegraded` | critical | cluster health != HEALTH_OK | 30s |
| `StorageIOBrownout` | warning | multi-metric combined | 3 min |
| `CephPoolFillBreachForecast` | warning | fill rate projection | 5 min |

---

## 7. The 6-Step Agent Pipeline

Both `compute-agent` and `storage-agent` implement identical 6-step pipelines exposed as
`POST /pipeline/{step}` endpoints. Each step is a visual node on the xyOps workflow canvas.
State is shared via an in-memory `PipelineSession` keyed by `session_id` (= service_name).

```
POST /pipeline/start
  → Creates PipelineSession
  → Creates skeleton xyOps incident ticket immediately
  → Returns: ticket_id, ticket_num, session_id

POST /pipeline/agent/logs
  → LogQL query to Loki (last 50 log lines)
  → Stores in session.logs
  → Posts "[>>] Fetching..." / "[OK] Fetched 50 lines" to ticket

POST /pipeline/agent/metrics
  → 4 PromQL instant queries: error_rate_pct, p99_latency_ms, p50_latency_ms, rps
  → Calls _extract_features() → ObsFeatures dataclass (20+ typed fields)
  → Stores in session.metrics

POST /pipeline/agent/analyze
  → Runs full intelligence pipeline (see §8)
  → Stores risk_score, risk_level, evidence on session

POST /pipeline/agent/ticket
  → Calls build_enriched_ticket_body()
  → Updates skeleton ticket with: RCA, metrics table, log snippet,
    Ansible playbook YAML, test plan, PR description

POST /pipeline/agent/approval
  → Checks autonomy rules (see §12)
  → Pre-validates playbook via POST /validate to ansible-runner
  → If passes: creates Gitea branch → commit → PR
  → Creates xyOps approval ticket with Approve/Decline event buttons
```

### Session Lifecycle
- Sessions stored in `_sessions: dict[str, PipelineSession]`
- 1-hour TTL; background GC via `_gc_sessions()` runs every 10 minutes
- **Not persisted** — service restart loses in-flight sessions

---

## 8. The Intelligence Engine (obs-intelligence)

The shared brain of the platform. Runs as an independent FastAPI service at `:9100`.

### 8.1 Background Loops (APScheduler)

**Analysis loop** — every 60 seconds:
- Queries Prometheus for Z-score anomaly detection across all golden signals
- Publishes `obs_intelligence_anomaly_z_score{metric_name, domain}` gauge
- Calls `_dispatch_predictive_alerts()` → fires HTTP pre-alerts to agents when thresholds crossed
  (risk ≥ 0.75, confidence ≥ 0.70, z-score ≥ 2.5) and no existing Prometheus alert is already firing

**Forecasting loop** — every 5 minutes:
- NumPy linear regression on Prometheus range-query time series
- Detects exponential growth pattern via coefficient of determination (R²)
- Publishes `obs_intelligence_forecast_breach_minutes{metric_name}` gauge
- Updates `current_intelligence["forecasts"]`

### 8.2 Analysis Pipeline (5 Stages)

```
ObsFeatures (20+ typed fields from metrics + logs)
       │
       ▼
[1] Scenario Correlator
    - Alert-name fnmatch pre-filter
    - Per-condition weighted scoring: scored_weight / total_weight = confidence
    - Filters by per-scenario confidence_threshold
    - Returns: ScenarioMatch + ScenarioDef (best match)
       │
       ▼
[2] Risk Scorer
    - 30% severity score (info=0.1, warning=0.4, critical=0.85, page=1.0)
    - 40% scenario confidence (0.0–1.0)
    - 15% log anomaly score (error count + log_anomaly_detected flag)
    - 15% forecast urgency (proximity to critical thresholds)
    - Returns: RiskAssessment (risk_score 0.0–1.0, risk_level, blast_radius, time_to_impact)
       │
       ▼
[3] Recommender
    - Maps (ScenarioMatch + RiskAssessment) → action_type + ansible_playbook
    - Applies autonomy clamping against autonomy rules
    - Returns: Recommendation (action_type, playbook_hint, autonomy_level)
       │
       ▼
[4] SRE Reasoning Agent (DETERMINISTIC — no LLM)
    - Produces: causal_chain[], predicted_impact, recommended_actions[]
    - to_prompt_block() renders verified structured facts for LLM injection
    - Prevents LLM from re-deriving reasoning (prompt injection mitigation)
       │
       ▼
[5] LLM Enricher (optional — graceful degradation when no API key)
    - Primary: OpenAI GPT-4o
    - Auto-fallback: Anthropic Claude
    - Returns: LLMEnrichment (rca_summary, ansible_playbook, test_cases[],
                              pr_title, pr_description, rollback_steps[])
    - If both unavailable: returns None; deterministic result used instead
```

### 8.3 Key API Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Loop counters, active anomalies count, ai_enabled flag |
| `GET /metrics` | Prometheus scrape (`obs_intelligence_*` gauges + counters) |
| `GET /intelligence/current` | Real-time state: anomalies + forecasts from background loops |
| `POST /analyze` | On-demand analysis for service/domain |
| `POST /intelligence/record-outcome` | Records `{scenario_id, outcome}` to Prometheus counter |

---

## 9. Scenario Catalog — All 20 Scenarios

### Compute Domain (10 scenarios)

| Scenario ID | Display Name | Typical Action |
|---|---|---|
| `error_spike` | Error Spike | investigate_errors |
| `latency_regression` | Latency Regression | investigate_latency |
| `memory_leak_emergence` | Memory Leak Emergence | restart_service |
| `cpu_saturation` | CPU Saturation | cpu_scale_out |
| `noisy_neighbor_effect` | Noisy Neighbour Effect | throttle_noisy_neighbour |
| `queue_backlog` | Queue Backlog | scale_workers |
| `cascading_timeout_chain` | Cascading Timeout Chain | circuit_break_dependency |
| `baseline_shift_after_deploy` | Post-Deploy Baseline Shift | rollback_deploy |
| `collector_overload` | Collector Overload | reduce_otel_sampling |
| `recurring_failure_signature` | Recurring Failure Signature | deep_dive_investigation |

### Storage Domain (10 scenarios)

| Scenario ID | Display Name | Typical Action |
|---|---|---|
| `single_osd_down` | Single OSD Down | osd_reweight |
| `multi_osd_failure` | Multiple OSD Failure | multi_osd_escalate (HUMAN ONLY) |
| `cluster_degraded_health` | Cluster Degraded Health | cluster_assessment |
| `pool_near_full` | Pool Near Full | pool_expand_advisory |
| `pool_full_critical` | Pool Full Critical | pool_critical_action |
| `pool_fill_forecast_breach` | Pool Fill Forecast Breach | pool_expand_advisory |
| `pvc_latency_degradation` | PVC Latency Degradation | investigate_io |
| `noisy_pvc_iops` | Noisy PVC IOPS | pvc_throttle |
| `ceph_rebalance_storm` | Ceph Rebalance Storm | investigate_io |
| `storage_io_brownout` | Storage IO Brownout | escalate |

Each scenario YAML defines:
- `alert_name_patterns`: fnmatch patterns that pre-filter eligible alerts
- `conditions[]`: list of `{field, op, value, weight}` checked against `ObsFeatures`
- `confidence_threshold`: minimum confidence to use this scenario
- `rca`: human-readable root cause template
- `playbook_hint`: maps to specific Ansible playbook file

---

## 10. Risk Scoring Model

```
risk_score = (0.30 × severity_score)
           + (0.40 × scenario_confidence)
           + (0.15 × log_anomaly_score)
           + (0.15 × forecast_urgency)
```

| Component | How calculated |
|---|---|
| severity_score | info=0.10, warning=0.40, critical=0.85, page=1.0 |
| scenario_confidence | Weighted match score from scenario correlator (0.0–1.0) |
| log_anomaly_score | Normalised error log count + `log_anomaly_detected` flag |
| forecast_urgency | How close current metric is to critical threshold per domain |

| risk_score | risk_level | Typical outcome |
|---|---|---|
| 0.0 – 0.30 | low | Autonomous action |
| 0.30 – 0.55 | medium | Approval recommended |
| 0.55 – 0.75 | high | Approval required |
| 0.75 – 1.0 | critical | Approval required; human-only if action in HUMAN_ONLY set |

---

## 11. The Validate-First Ansible Workflow

This is the PR-gated remediation workflow. Implemented in `compute-agent/app/approval_workflow.py`.

```
AI generates Ansible playbook YAML + test_cases[]
          │
          ▼
  POST /validate → ansible-runner (dry-run with structured test cases)
          │
    ┌─────┴─────┐
  FAIL         PASS
    │             │
    ▼             ▼
Post failure   validation_passed = True
table to       validation_result stored on ApprovalRequest
xyOps ticket         │
STOP                 ▼
             git commit playbook YAML to new branch
             aiops/{alert_name_slug}-{short_uuid}
                     │
                     ▼
             create PR: branch → main
             (repo: aiops-org/ansible-playbooks on Gitea)
                     │
                     ▼
          Create xyOps APPROVAL TICKET with:
          - Validated playbook YAML
          - Test case results (✅/❌ table)
          - PR link (Gitea:3002)
          - TWO-STEP instructions:
            Step 1: Merge PR on Gitea (review YAML diff)
            Step 2: Click Approve here in xyOps
                     │
          ┌──────────┴──────────┐
     User merges PR         User clicks Decline
          │                       │
     User clicks Approve    close_pull_request()
          │                 status = "declined"
          ▼
     check_pr_merged() called
          │
    ┌─────┴──────┐
  NOT MERGED    MERGED
    │               │
    ▼               ▼
  Reset to       POST /run → ansible-runner
  pending        Execution result (stdout, rc, duration,
  Post reminder  test_results) posted to incident ticket
  comment             │
                      ▼
              POST /intelligence/record-outcome
              {scenario_id, outcome: "resolved"/"failed"}
```

### ApprovalRequest State Machine

```
pending → validation_failed  (pre-commit validation failed)
pending → pending             (PR not merged yet, reminder sent)
pending → approved            (PR merged + xyOps Approve clicked)
pending → declined            (xyOps Decline clicked)
approved → executed           (Ansible run completed)
```

---

## 12. Autonomy Rules Engine

Three-tier decision model. **Most restrictive rule wins.**

### Compute Domain

| Tier | Actions |
|---|---|
| `HUMAN_ONLY` (never autonomous) | _(none currently)_ |
| `APPROVAL_REQUIRED` | rollback_deploy, restart_service, circuit_break_dependency, scale_workers, deep_dive_investigation, investigate_errors, investigate_latency, cpu_scale_out |
| `AUTONOMOUS_ALLOWED` | reduce_otel_sampling, throttle_noisy_neighbour |
| Risk gate | All actions (including autonomous) require approval if risk_score ≥ 0.70 |

### Storage Domain

| Tier | Actions |
|---|---|
| `HUMAN_ONLY` (never autonomous) | multi_osd_escalate ← data loss risk |
| `APPROVAL_REQUIRED` | osd_reweight, pool_expand_advisory, pool_critical_action, investigate_io, cluster_assessment, escalate |
| `AUTONOMOUS_ALLOWED` | pvc_throttle ← reversible, single-PVC QoS |
| Risk gate | All actions require approval if risk_score ≥ 0.65 (tighter than compute) |

---

## 13. Agentic AI Properties — Evidence

| Property | Implementation |
|---|---|
| **Autonomous decision making** | Autonomy rules engine + risk gating; two action tiers can execute without human per domain |
| **Multi-agent coordination** | compute-agent + storage-agent share obs-intelligence via HTTP; obs-intelligence dispatches predictive pre-alerts back to both agents |
| **Tool use** | Each pipeline step is a discrete tool: log fetcher (Loki), metric analyst (Prometheus), intelligence engine, LLM enricher, approval gateway, git client, playbook executor |
| **Memory / state** | In-session: `PipelineSession` dict (1h TTL); cross-restart: `obs_intelligence_scenario_outcome_total{scenario_id, outcome}` Prometheus counter |
| **Self-healing closed loop** | Anomaly → analysis → playbook generation → validation → human gate → execution → outcome recorded → feeds future confidence |
| **Predictive / proactive** | Z-score + forecasting loops; `[PREDICTIVE]` xyOps tickets created before Prometheus alert fires |
| **LLM multi-provider failover** | OpenAI GPT-4o primary; Anthropic Claude automatic fallback; deterministic path if both unavailable |
| **Prompt injection mitigation** | SRE Reasoning Agent runs first (deterministic); LLM is given pre-computed verified facts and told to write narrative from them, not re-derive |
| **GitOps audit trail** | Every remediation = Gitea PR on `main`; merged only with human review |
| **Graceful degradation** | LLM down → full deterministic RCA + playbook; ansible-runner unreachable → playbook in ticket for manual run |

---

## 14. What Is Fully Working

| Feature | Status | Notes |
|---|---|---|
| OTel collection → Prometheus + Tempo + Loki → Grafana | ✅ | Cross-linked exemplars enabled |
| All 8 troublemaker chaos scenarios + CSV audit log | ✅ | |
| 6 storage-simulator injectable Ceph scenarios | ✅ | |
| 15 Prometheus alert rules + Alertmanager routing | ✅ | |
| Dual-agent webhook receivers | ✅ | |
| 6-step pipeline (all steps implemented) | ✅ | |
| 20-scenario intelligence catalog (deterministic) | ✅ | LLM-independent |
| Background Z-score anomaly + linear forecasting | ✅ | APScheduler, 60s + 5m |
| Predictive pre-alerting to agents | ✅ | Suppressed if real alert already firing |
| LLM enrichment (GPT-4o + Claude failover) | ✅ | Requires `OPENAI_API_KEY` in `.env` |
| Pre-commit playbook validation | ✅ | Fails hard — no git push if tests fail |
| Gitea: branch commit + PR create/merge/close | ✅ | |
| xyOps: tickets, approve/decline events, approval canvas | ✅ | Requires `XYOPS_API_KEY` in `.env` |
| PR-merge gate before Ansible execution | ✅ | `check_pr_merged()` |
| Outcome feedback loop to obs-intelligence | ✅ | `POST /intelligence/record-outcome` |
| Prometheus metrics from all agents | ✅ | `compute_agent_*`, `storage_agent_*`, `obs_intelligence_*` |
| Ansible runner — simulated execution | ✅ | No `ansible-playbook` binary needed |
| Ansible runner — real execution | ⚙️ | Set `ANSIBLE_LIVE_MODE=true` |
| Approval state persistence across restarts | ⚙️ | In-memory; needs Redis/DB for production |
| Real Ceph cluster operations | ℹ️ | Storage-simulator emulates metrics only |

---

## 15. Technology Stack

| Category | Technology | Version |
|---|---|---|
| Language | Python | 3.12 |
| Web framework | FastAPI + Uvicorn | latest |
| Telemetry SDK | opentelemetry-sdk | latest |
| HTTP client | httpx (async) | latest |
| Scheduler | APScheduler | 3.x |
| Metrics | Prometheus Python client | latest |
| ML/Math | NumPy | latest |
| Containerisation | Docker Compose | v2 |
| Metrics store | Prometheus | latest |
| Trace store | Grafana Tempo | latest |
| Log store | Grafana Loki | latest |
| Dashboards | Grafana | latest |
| Alerting | Prometheus Alertmanager | latest |
| OTel Collector | OpenTelemetry Collector Contrib | latest |
| AIOps platform | xyOps (self-hosted) | latest |
| Playbook executor | Custom ansible-runner FastAPI | — |
| Git server | Gitea | latest |
| LLM (primary) | OpenAI GPT-4o | gpt-4o |
| LLM (fallback) | Anthropic Claude | claude-3-5-haiku |

---

## 16. Configuration Reference

All secrets live in `.env` (gitignored). Copy `.env.example` to `.env` and fill in:

```env
# Required for AI enrichment (one of these two)
OPENAI_API_KEY=sk-proj-...
# CLAUDE_API_KEY=sk-ant-...

# Required for xyOps integration
# Get from: xyOps UI → Admin → API Keys → Create Key
XYOPS_API_KEY=your-xyops-api-key
```

Key environment variables in `docker-compose.yml` (non-secret):

| Variable | Service | Default | Purpose |
|---|---|---|---|
| `REQUIRE_APPROVAL` | compute-agent, storage-agent | `true` | Enable approval gate |
| `AI_MODEL` | compute-agent, storage-agent | `gpt-4o` | OpenAI model to use |
| `ANSIBLE_LIVE_MODE` | ansible-runner | `false` | Execute real playbooks |
| `ANOMALY_Z_THRESHOLD` | obs-intelligence | `2.5` | Z-score alert threshold |
| `PREDICTIVE_RISK_THRESHOLD` | obs-intelligence | `0.75` | Min risk for pre-alert |
| `PREDICTIVE_CONFIDENCE_THRESHOLD` | obs-intelligence | `0.70` | Min confidence for pre-alert |
| `WORKFLOW_STEP_DELAY` | compute-agent, storage-agent | `5` | Seconds between pipeline steps |

---

## 17. Learning Path This Demonstrates

This project teaches the following disciplines in a working, integrated system:

### Observability Engineering
- Instrumenting microservices with OpenTelemetry (traces, metrics, logs)
- Designing golden-signal metrics (error rate, latency p99/p50, throughput)
- Writing PromQL alert rules: rate vs irate, histogram quantiles, baseline comparison
- Alertmanager routing trees, grouping, inhibition
- Grafana dashboard design with trace-metric-log cross-linking via exemplars

### Platform Engineering
- Multi-service Docker Compose orchestration
- Service mesh via docker network aliases
- Secret management: `.env` + variable substitution in Compose (never commit real keys)
- Health check patterns and startup ordering

### AIOps / AI Engineering
- Structured agentic pipeline design (discrete tools, shared state)
- Deterministic-first reasoning: compute before LLM calls, not inside them
- Multi-provider LLM failover patterns
- Prompt engineering: injecting pre-computed facts to constrain LLM narrative
- Autonomy rules engines: tiered action risk classification
- Feedback loops: outcome recording → future scenario confidence

### GitOps / Change Management
- Every automated change goes through a PR (Git as authorisation record)
- Validate before commit — never push untested code
- Human-in-the-loop gates for high-risk actions
- Merge = authorisation; execution only after merge

### Chaos Engineering
- Systematic fault injection (error, latency, traffic, backend, slow-burn)
- Correlating telemetry signals with known chaos events via audit log
- Weighted scenario scheduling for realistic traffic patterns
