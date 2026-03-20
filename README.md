# Observability Learning — Multi-Agent AIOps Platform

A full-stack, containerised **AIOps observability platform** that closes the loop from Prometheus alert → AI analysis → enriched incident ticket → human-approved Ansible remediation — with a shared intelligence engine used by two specialised domain agents.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Service Map & Ports](#service-map--ports)
- [What Has Been Built](#what-has-been-built)
- [What Is Still To Add](#what-is-still-to-add)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Key Workflows](#key-workflows)
- [Environment Variables Reference](#environment-variables-reference)

---

## Architecture Overview

```
                         ┌─────────────────────────────────────────────┐
                         │         Observability Stack                  │
                         │  Prometheus · Loki · Tempo · Grafana · OTEL  │
                         └───────────────┬─────────────────────────────┘
                                         │  alerts
                                         ▼
                                   Alertmanager
                                  /             \
                    domain=compute               domain=storage
                         │                            │
                         ▼                            ▼
                  Compute Agent                 Storage Agent
                  (port 9000)                   (port 9001)
                         │                            │
                         └──────────┬─────────────────┘
                                    │  enrichment
                                    ▼
                        Obs-Intelligence Engine
                              (port 9100)
                    ┌──────────────────────────────────┐
                    │  Scenario Correlator              │
                    │  Feature Extractor                │
                    │  Risk Scorer                      │
                    │  Anomaly Detector (Z-score)       │
                    │  Forecaster (numpy linreg)        │
                    │  Recommender                      │
                    │  Evidence Builder                 │
                    │  SRE Reasoning Agent (det.)       │
                    │  LLM Enricher (OpenAI → Claude)   │
                    │  Background: predictive dispatch  │
                    │  POST /intelligence/record-outcome│
                    └──────────────────────────────────┘
                                    │
                                    ▼
                               xyOps (5522)
                    ┌──────────────────────────────┐
                    │  Incident Ticket              │
                    │  Approval Workflow Canvas     │
                    │  Job Executor                 │
                    └──────────────────────────────┘
                                    │ approved
                                    ▼
                           ansible-runner (8090)
                                    │
                                    ▼
                             Gitea PR / Audit (3002)
```

**Alert routing** (alertmanager):
- Labels `domain: compute` → Compute Agent
- Labels `domain: storage` → Storage Agent
- Both agents call the shared Obs-Intelligence Engine for scenario matching, risk scoring, and LLM-enriched recommendations before creating a xyOps ticket.

---

## Service Map & Ports

| Service | Port(s) | Purpose |
|---|---|---|
| **otel-collector** | 4317 (gRPC), 4318 (HTTP), 8889 (Prom scrape), 13133 (health) | Receives all OTLP telemetry, exports to Prometheus / Tempo / Loki |
| **prometheus** | 9090 | Metrics TSDB + alert rule evaluation |
| **loki** | 3100 | Log aggregation (LogQL) |
| **tempo** | 3200 | Distributed trace storage (TraceQL) |
| **grafana** | 3000 | Dashboards — metrics, traces, logs, agent decisions |
| **alertmanager** | 9093 | Alert grouping/dedup + webhook routing |
| **compute-agent** | 9000 | Compute AIOps agent — `POST /webhook`, `GET /health`, `GET /metrics` |
| **storage-agent** | 9001 | Storage AIOps agent — `POST /webhook`, `GET /health`, `GET /metrics` |
| **obs-intelligence** | 9100 | Shared intelligence engine — `GET /intelligence/current`, `POST /analyze`, `POST /intelligence/record-outcome` |
| **storage-simulator** | 9200 | Ceph scenario emulator — `POST /scenario/{name}`, `GET /metrics` |
| **xyops** | 5522, 5523 | AIOps platform — tickets, workflows, job scheduler |
| **ansible-runner** | 8090 | Playbook executor — `POST /run` |
| **gitea** | 3002 | Self-hosted Git — Ansible PR audit trail |
| **frontend-api** | 8080 | Demo app — emits OTel spans/metrics/logs |
| **backend-api** | 8081 | Demo app — emits OTel spans/metrics/logs |
| **loadgen** | — | Steady traffic generator (opt-in profile) |
| **troublemaker** | 8088 | Compute chaos engine — CPUSpike, MemoryLeak, etc. (opt-in profile) |

---

## What Has Been Built

### Phase 1 — Core Observability Stack
- **OTel Collector** with Prometheus, Tempo, and Loki exporters
- **Grafana** with pre-provisioned datasources (Prometheus, Tempo, Loki) and cross-linking (exemplar → trace → log)
- **frontend-api / backend-api** demo services instrumented with OTel SDK (spans + metrics + logs)
- **loadgen** and **troublemaker** for realistic traffic and chaos scenarios

### Phase 2 — Alert Pipeline & AIOps Integration
- **Prometheus alert rules** (`prometheus/alert-rules.yml`) for CPU, memory, latency, error rate, storage metrics
- **Alertmanager** routing: compute vs storage domain labels, inhibition rules
- **aiops-bridge** (now `compute-agent`) — webhook receiver, AI analysis, xyOps ticket creation, Ansible execution
- **xyOps** — AIOps platform provisioned with incident workflows and approval canvas
- **ansible-runner** — dry-run playbook executor; set `ANSIBLE_LIVE_MODE=true` for real hosts
- **Gitea** — auto-provisioned with org `aiops-org` / repo `ansible-playbooks`; every approval creates a PR branch

### Phase 3 — Domain Split & Storage Agent
- **compute-agent** — evolved from aiops-bridge; now emits `compute_agent_*` Prometheus metrics
- **storage-agent** — new storage-domain agent mirroring compute-agent architecture
  - Autonomy rules: `HUMAN_ONLY` actions (multi-OSD), `APPROVAL_REQUIRED` (OSD reweight, pool expand), `AUTONOMOUS` (PVC throttle)
- **storage-simulator** — Ceph scenario emulator (healthy / osd_down / multi_osd_failure / pool_full / latency_spike / noisy_pvc)

### Phase 4 — Shared Obs-Intelligence Engine (OIE)
- **obs_intelligence Python package** (`obs-intelligence/app/obs_intelligence/`):
  - `scenario_loader.py` — loads YAML scenario definitions from `obs-intelligence/scenarios/`
  - `scenario_correlator.py` — matches features against scenario thresholds with confidence weighting
  - `feature_extractor.py` — maps Prometheus metrics + Loki logs → typed `ObsFeatures`
  - `models.py` — shared Pydantic models (`ObsFeatures`, `ScenarioMatch`, `RiskAssessment`, `Recommendation`, `EvidenceReport`)
  - `telemetry_client.py` — cached Prometheus + Loki query client
- **20 scenario YAML files** — 10 compute (`cpu_saturation`, `memory_leak_emergence`, `error_spike`, `latency_regression`, `noisy_neighbor_effect`, `queue_backlog`, `cascading_timeout_chain`, `baseline_shift_after_deploy`, `collector_overload`, `recurring_failure_signature`) + 10 storage (`ceph_rebalance_storm`, `cluster_degraded_health`, `multi_osd_failure`, `pool_full_critical`, `pool_near_full`, `pool_fill_forecast_breach`, `single_osd_down`, `pvc_latency_degradation`, `noisy_pvc_iops`, `storage_io_brownout`)
- **obs-intelligence FastAPI service** (`main.py`) — exposes `GET /health`, `GET /metrics`, `GET /intelligence/current`, `POST /analyze`
- **Background analysis loops** (`background.py`) — APScheduler: anomaly scan every 60 s, forecasting every 5 min
- Two Grafana dashboards: `agentic-ai-overview.json` and `obs-intelligence-detail.json`

### Phase 5 — Risk Scoring, Evidence Builder & LLM Enrichment
- `risk_scorer.py` — weighted scoring: 30% severity + 40% scenario confidence + 15% log anomaly + 15% forecast urgency → 0.0–1.0 `RiskAssessment`
- `recommender.py` — maps best scenario → `Recommendation` with playbook, MTTR, autonomy clamping
- `evidence_builder.py` — assembles `EvidenceReport` (matched signals, metric snapshots, log counts, confidence factors)
- `llm_enricher.py` — optional GPT-4o / Claude enrichment; returns `LLMEnrichment` with root cause hypothesis, investigation steps, rollback hint
- `anomaly_detector.py` — Z-score anomaly detection via PromQL `avg_over_time` / `stddev_over_time`
- `forecaster.py` — numpy linear regression + exponential growth detection; returns `ForecastResult` with trend and predicted time-to-critical
- `metrics_publisher.py` — publishes `obs_intelligence_*` Prometheus gauges and counters
- Both agents' `pipeline.py` updated with full 6-step intelligence pipeline:
  1. Feature extraction
  2. Scenario correlation
  3. Risk scoring
  4. Recommendation
  5. Evidence building
  6. LLM enrichment
- Enriched xyOps tickets now include: risk level badge, risk score, evidence observations, LLM root cause narrative

### Phase 6 — Continuous Intelligence & Predictive Alerts
- **Predictive alert dispatch** — obs-intelligence background loop now fires `POST /predictive-alert` on compute-agent and storage-agent when `risk_score > 0.75 AND confidence > 0.7` and **no active Prometheus alert exists** — catching degradation before it pages
- **`POST /predictive-alert`** endpoint on both agents — creates `[PREDICTIVE]`-tagged xyOps tickets; always `approval_gated` (human must approve proactive remediations)
- **`background.py`** fully realised — APScheduler drives two loops:
  - Anomaly scan every 60 s (Z-score per metric/service)
  - Forecast + predictive dispatch every 5 min
- **New Prometheus alert rules** — `NoAlertFiringButHighRisk` and `HighAnomalyZScore` for surfacing intelligence-layer signals in the alerting pipeline
- **Predictive Alert Workflow** provisioned in xyOps for both compute and storage domains — canvas nodes distinguish predictive from reactive alerts
- **`agentic-ai-overview` Grafana dashboard** updated with "Active Intelligence Signals" row showing predictive alert rates and anomaly Z-scores
- **Predictive alert counters** added to `telemetry.py` for all three services

### Phase 7 — SRE Reasoning Layer
- **`sre_reasoning_agent.py`** — fully deterministic (no LLM calls, unit-testable) `SREReasoningAgent` that produces a structured `SREAssessment`:
  - `degradation_summary` — one-liner per domain derived from live signal values
  - `causal_chain` — root cause chain built from top scenario match + `contributing_factors` + domain inferences
  - `predicted_impact` — maps risk level → blast radius + time-to-impact
  - `recommended_actions` — scenario playbook actions + domain-specific steps
  - `autonomy_recommendation` — `human_only` / `approval_gated` / `autonomous`
  - `urgency` — `critical` / `high` / `medium` / `low` from risk + time-to-impact
  - `evidence_strength` — scored from scenario confidence + log anomaly + error metrics
  - `to_prompt_block()` — serialises the assessment as a structured block injected into the LLM prompt
- **LLM enricher redesigned** — LLM receives SREAssessment as structured context and **writes the narrative** around it; it no longer generates its own reasoning (narrative writer, not reasoner)
- **OpenAI → Claude automatic failover** — `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` are independent; if OpenAI call raises an exception, Claude is tried automatically; returns `None` and falls back to deterministic analysis only if both fail
- **Outcome tracking**:
  - `obs_intelligence_scenario_outcome_total{scenario_id, outcome}` Prometheus counter added to `metrics_publisher.py`
  - `POST /intelligence/record-outcome` endpoint on obs-intelligence — agents call it on alert resolution/escalation
  - Alert resolution hooks in both compute-agent and storage-agent automatically POST the outcome when a firing alert resolves
- **`recurring_failure_signature` scenario** — upgraded to `human_only` autonomy; fires when `recurrence_count >= 3` within 6 h; `recurrence_count` field added to `ObsFeatures`
- **Grafana "SRE Incident Timeline" dashboard** (`sre-incident-timeline.json`) — 4 rows, 12 panels: risk gauges, firing alerts, scenario outcomes barchart, HTTP error rate / latency timeseries, storage pool fill, risk score timeline, outcome rate, scenario confidence heatmap
- **`CONTRIBUTING-SCENARIOS.md`** — full scenario YAML authoring guide covering all fields, available `ObsFeatures` fields, confidence scoring formula, autonomy selection guidance, and two worked examples
- **`DEMO-RUNBOOK.md`** — step-by-step end-to-end rehearsal guide from `docker compose up` through chaos injection, predictive alert observation, approval workflows, Ansible execution, and outcome recording

---

## What Is Still To Add

| Item | Description | Effort |
|---|---|---|
| **End-to-end Docker test** | `docker compose up --build` + curl all agent health endpoints + trigger a test alert + verify xyOps ticket with risk score and SRE reasoning block | Low |
| **oie_client.py in agents** | Replace direct obs_intelligence package import with HTTP client to `http://obs-intelligence:9100/analyze` for true service decoupling | Medium |
| **Multi-agent correlation** | When compute and storage alerts fire simultaneously, detect cross-domain cascading failures and produce a unified SREAssessment | High |
| **Recurrence counter persistence** | `recurrence_count` is currently in-memory; persist it in Redis or SQLite so obs-intelligence survives restarts without losing recurrence history | Medium |
| **Notification integrations** | Slack / PagerDuty / email receivers beyond xyOps webhook; notification routing by risk level | Medium |
| **Ansible live mode** | Real Ansible playbook execution against actual infrastructure targets (set `ANSIBLE_LIVE_MODE=true`) | Medium |
| **Scenario schema validation** | JSON Schema / Pydantic validation of `obs-intelligence/scenarios/*.yaml` files at startup with clear schema errors | Low |
| **Auth / API key middleware** | Bearer token validation on agent webhook endpoints and obs-intelligence `/analyze`; currently open | Medium |
| **Persistent intelligence state** | Replace in-memory current state with Redis or SQLite so obs-intelligence survives restarts | Medium |
| **Unit test suite (agents)** | Extend existing compute-agent test suite to cover risk_scorer, recommender, evidence_builder, and sre_reasoning_agent | Medium |
| **SREAssessment feedback loop** | Feed recorded outcomes back into scenario confidence weights over time (reinforcement learning lite) | High |
| **Outcome dashboard drill-down** | Grafana panel linking scenario outcome bars to the originating xyOps ticket and Ansible run log | Low |

---

## Prerequisites

### Required Software

| Tool | Minimum Version | Purpose |
|---|---|---|
| **Docker Desktop** | 4.x (Docker Engine 24+) | Container runtime — all services run in Docker |
| **Docker Compose** | v2.x (bundled with Docker Desktop) | Multi-container orchestration |
| **Git** | 2.x | Clone the repository |

> Docker Desktop for Windows requires WSL 2 or Hyper-V. Ensure at least **8 GB RAM** allocated to Docker.

### Optional (for AI enrichment)

| Credential | Where to set | Purpose |
|---|---|---|
| **OpenAI API key** | `OPENAI_API_KEY` env var in `docker-compose.yml` or `.env` file | GPT-4o LLM enrichment in agent pipelines |
| **Anthropic API key** | `ANTHROPIC_API_KEY` env var | Claude-3 alternative to OpenAI |

> Without an API key the agents fall back to deterministic analysis — full scenario matching, risk scoring, and evidence building still work; only the LLM narrative is skipped.

### Network & Ports

Ensure the following host ports are free before starting:

```
3000  Grafana
3002  Gitea
3100  Loki
3200  Tempo
4317  OTel Collector gRPC
5522  xyOps
8080  frontend-api
8081  backend-api
8090  ansible-runner
9000  compute-agent
9001  storage-agent
9090  Prometheus
9093  Alertmanager
9100  obs-intelligence
9200  storage-simulator
```

### First-Time xyOps Setup (after first `docker compose up`)

1. Open [http://localhost:5522](http://localhost:5522) — login: `admin / admin`
2. **Admin → API Keys → Create Key** → copy the key
3. Paste the key as `XYOPS_API_KEY` in `docker-compose.yml` for both `compute-agent` and `storage-agent`
4. `docker compose restart compute-agent storage-agent`

---

## Quick Start

```bash
# Clone
git clone https://github.com/omkarumarani/obseransiblerepo.git
cd obseransiblerepo

# Start core stack
docker compose up --build

# With load generator + chaos (troublemaker)
docker compose --profile loadgen --profile troublemaker up --build

# With storage simulator always on (already in default compose)
docker compose up --build

# Health checks
curl http://localhost:9000/health      # compute-agent
curl http://localhost:9001/health      # storage-agent
curl http://localhost:9100/health      # obs-intelligence

# Manually trigger a compute test alert
curl -X POST http://localhost:9000/webhook \
  -H "Content-Type: application/json" \
  -d '{"status":"firing","alerts":[{"status":"firing","labels":{"alertname":"HighMemoryUsage","service_name":"frontend-api","severity":"warning","domain":"compute"},"annotations":{"summary":"Memory above 80%","description":"Test","dashboard_url":"http://localhost:3000"},"startsAt":"2026-03-20T10:00:00Z"}]}'

# Trigger a storage scenario
curl -X POST http://localhost:9200/scenario/osd_down

# Check pre-computed intelligence state
curl http://localhost:9100/intelligence/current

# View obs-intelligence metrics (Prometheus format)
curl http://localhost:9100/metrics

# Stop everything (preserves data volumes)
docker compose down

# Stop and wipe all data
docker compose down -v
```

---

## Key Workflows

### Compute Alert → Ticket

```
Prometheus fires HighMemoryUsage (domain=compute)
  → Alertmanager → POST compute-agent:9000/webhook
    → pipeline.py step 1: extract_features (memory_growth_rate, error_rate, p99_latency)
    → step 2: match_scenarios (top match: memory_leak_emergence, confidence 0.82)
    → step 3: score_risk (risk_score=0.74, level=high)
    → step 4: recommend (action: restart_service, approval_required=true)
    → step 5: build_evidence (3 matched signals, 12 log errors)
    → step 6: LLM enrich (root cause narrative, investigation steps)
  → POST xyops:5522/api/app/create_ticket → ticket with risk badge + evidence
  → Approval workflow canvas (human approves)
  → POST ansible-runner:8080/run → Ansible playbook
  → Gitea PR created for audit trail
```

### Storage Alert → Ticket

Same as above via `storage-agent:9001/webhook` with storage-domain scenarios (CephOSDDown, pool_full_critical, etc.).

### Background Intelligence

```
obs-intelligence (every 60 s):
  → query Prometheus for known service metrics
  → Z-score anomaly detection per metric
  → publish obs_intelligence_anomaly_zscore{metric, service}

obs-intelligence (every 5 min):
  → query Prometheus query_range for 1h history
  → numpy linear regression per metric
  → publish obs_intelligence_forecast_value{metric, service, horizon}

Domain agents (on each alert):
  → GET obs-intelligence:9100/intelligence/current
  → merge pre-computed anomalies + forecasts into LLM context
```

---

## Environment Variables Reference

| Variable | Service | Default | Description |
|---|---|---|---|
| `XYOPS_URL` | compute-agent, storage-agent | `http://xyops:5522` | xyOps base URL |
| `XYOPS_API_KEY` | compute-agent, storage-agent | _(set after first login)_ | xyOps REST API key |
| `OPENAI_API_KEY` | obs-intelligence | _(optional)_ | OpenAI enrichment (tried first) |
| `ANTHROPIC_API_KEY` | obs-intelligence | _(optional)_ | Anthropic Claude fallback (used if OpenAI fails) |
| `OPENAI_MODEL` | obs-intelligence | `gpt-4o-mini` | OpenAI model override |
| `CLAUDE_MODEL` | obs-intelligence | `claude-3-5-haiku-20241022` | Claude model override |
| `PROMETHEUS_URL` | all agents | `http://prometheus:9090` | Prometheus query URL |
| `LOKI_URL` | all agents | `http://loki:3100` | Loki query URL |
| `OBS_INTELLIGENCE_URL` | compute-agent, storage-agent | `http://obs-intelligence:9100` | Intelligence engine URL |
| `REQUIRE_APPROVAL` | compute-agent | `true` | Set `false` to auto-execute without human sign-off |
| `STORAGE_REQUIRE_APPROVAL` | storage-agent | `true` | Set `false` for auto-execute |
| `ANSIBLE_LIVE_MODE` | ansible-runner | `false` | Set `true` for real playbook execution |
| `ANOMALY_Z_THRESHOLD` | obs-intelligence | `2.5` | Z-score threshold to flag anomalies |
| `FORECAST_HORIZON_MINUTES` | obs-intelligence | `60` | Forecast window in minutes |
| `GITHUB_REPO` | compute-agent | _(your repo)_ | For Ansible playbook PR audit trail |
| `GITEA_ENABLED` | compute-agent | `true` | Enable Gitea PR creation on approval |
| `WORKFLOW_STEP_DELAY_SECONDS` | compute-agent, storage-agent | `5` | Seconds each xyOps canvas node stays highlighted |
