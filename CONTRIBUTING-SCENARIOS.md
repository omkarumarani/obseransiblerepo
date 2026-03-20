# CONTRIBUTING — Scenario YAML Authoring Guide

> This guide explains how to add new failure-pattern scenarios to the
> `obs-intelligence` engine for any agent domain — compute, storage, network,
> database, or security.

---

## Table of Contents

1. [Overview](#overview)
2. [Directory structure](#directory-structure)
3. [Scenario YAML schema reference](#scenario-yaml-schema-reference)
4. [Step-by-step: add a new scenario](#step-by-step-add-a-new-scenario)
5. [Autonomy levels explained](#autonomy-levels-explained)
6. [Condition operators reference](#condition-operators-reference)
7. [Available ObsFeatures fields](#available-obsfeatures-fields)
8. [Adding scenarios for new domains](#adding-scenarios-for-new-domains)
9. [Testing your scenario locally](#testing-your-scenario-locally)
10. [Worked examples](#worked-examples)

---

## Overview

The obs-intelligence engine matches incoming Prometheus/Loki alert signals
against a **scenario catalog** — a directory of YAML files that each define
one known failure pattern.  When a pattern matches with sufficient confidence:

```
ObsFeatures (metrics + logs)
        │
        ▼
ScenarioCorrelator  ──scores──►  ScenarioMatch (confidence 0–1)
        │
        ▼
Recommender (+ autonomy rules)  ──►  Recommendation
        │
        ▼
SREReasoningAgent.assess()  ──►  SREAssessment (causal chain, impact, actions)
        │
        ▼
LLM enricher (OpenAI / Claude)  ──►  rich narrative written from structured facts
        │
        ▼
xyOps ticket + approval gate
```

Scenarios are **declarative** — you describe the conditions that define a
pattern; the engine handles all the scoring, reasoning, and routing logic.

---

## Directory structure

```
obs-intelligence/
  scenarios/
    scenario.schema.json          ← JSON Schema for CI validation
    compute/
      cpu_saturation.yaml
      error_spike.yaml
      recurring_failure_signature.yaml
      ...
    storage/
      osd_down.yaml
      pool_fill_critical.yaml
      ...
    network/                      ← create this for network agent scenarios
      .gitkeep                    ← empty dir marker until first scenario added
    database/                     ← for database agent scenarios
    security/                     ← for security agent scenarios
```

Each YAML file = one scenario.  The filename should match `scenario_id`.

---

## Scenario YAML schema reference

```yaml
# ── Identity ──────────────────────────────────────────────────────────────────
scenario_id:    my_scenario_id      # snake_case, unique within the domain
display_name:   "Human-Readable Name"
domain:         compute             # compute | storage | network | database | security
version:        "1.0"

# ── Alert-name pre-filter (optional but recommended) ─────────────────────────
# The scenario is only scored if the incoming alert_name matches at least one
# of these fnmatch patterns.  Leave empty to score against every alert.
alert_name_patterns:
  - "HighErrorRate*"
  - "ServiceDegraded*"

alert_match_weight: 0.20   # base confidence added when alert name matches (0.0–1.0)

# ── Threshold ─────────────────────────────────────────────────────────────────
confidence_threshold: 0.40  # min confidence to include in match results (0.0–1.0)

# ── Remediation metadata ──────────────────────────────────────────────────────
action:       restart_service   # playbook action key (see recommender._PLAYBOOK_MAP)
autonomy:     approval_gated    # autonomous | approval_gated | human_only
irreversible: false             # true = extra confirmation before execution

# ── Narrative (used in LLM prompt + ticket body) ─────────────────────────────
rca: >
  Describe what is likely happening and why.  This text is injected into the
  SRE Reasoning Layer and into the LLM prompt as pre-computed SRE analysis.

playbook_hint: >
  Step-by-step troubleshooting guide for the on-call engineer.

# ── Recurrence policy (optional) ─────────────────────────────────────────────
# When present, the scenario description engine will track alert recurrences
# and populate ObsFeatures.recurrence_count.
recurrence_policy:
  min_occurrences: 3   # fires human_only autonomy upgrade when >= this count
  window_hours:    6   # look-back window for recurrence counting

# ── Conditions ────────────────────────────────────────────────────────────────
# Each condition maps to a field of ObsFeatures.  The scenario's confidence
# score is: sum of matched weights / total weights (capped at 1.0).
conditions:
  - field:     error_rate          # ObsFeatures field name
    operator:  gte                 # gt | lt | gte | lte | eq | ne | true | false
    threshold: 0.10                # numeric threshold (ignored for true/false ops)
    weight:    0.50                # contribution to confidence score (must be > 0)

  - field:     log_anomaly_detected
    operator:  "true"
    weight:    0.30

  - field:     latency_p99
    operator:  gte
    threshold: 1.5
    weight:    0.20
```

---

## Step-by-step: add a new scenario

### 1. Choose or create the domain directory

```
obs-intelligence/scenarios/<domain>/
```

If the directory does not exist, create it and add a `.gitkeep` file.

### 2. Create the YAML file

Name the file `<scenario_id>.yaml`, e.g. `connection_pool_exhaustion.yaml`.

### 3. Fill in the required fields

All of these are required:
- `scenario_id` — unique, snake_case
- `display_name` — human-readable
- `domain` — one of `compute | storage | network | database | security`
- `conditions` — at least one condition with `field`, `operator`, `weight`

### 4. Choose an `action`

The `action` field must match a key in `recommender._PLAYBOOK_MAP` (see
[recommender.py](obs-intelligence/app/obs_intelligence/recommender.py)) or be
`escalate` (generic escalation, no playbook).

To add a new action type, add it to `_PLAYBOOK_MAP` and `_DURATION_MAP` in
`recommender.py`.

### 5. Choose `autonomy`

| Value | Meaning |
|---|---|
| `autonomous` | Engine executes playbook without human approval (safe/low-blast-radius only) |
| `approval_gated` | Human must click Approve in xyOps before playbook runs |
| `human_only` | SRE must investigate manually — no automation allowed |

See [Autonomy levels explained](#autonomy-levels-explained) for guidelines.

### 6. Validate with schema

```bash
cd obs-intelligence
pip install jsonschema pyyaml
python -c "
import yaml, json, jsonschema, pathlib
schema = json.loads(pathlib.Path('scenarios/scenario.schema.json').read_text())
data = yaml.safe_load(pathlib.Path('scenarios/<domain>/<your_file>.yaml').read_text())
jsonschema.validate(data, schema)
print('Schema OK')
"
```

### 7. Run the unit tests

```bash
cd obs-intelligence/app
pip install -e '.[test]'
pytest tests/ -k scenario
```

---

## Autonomy levels explained

### `autonomous`
Use when:
- The action is fully reversible (e.g. cache flush, metric subscription change)
- Blast radius is limited to a single service
- Risk score is consistently below 0.5
- The playbook has been tested in staging for at least 2 weeks

Examples: `reduce_otel_sampling`, `pvc_throttle`, `cache_eviction`

### `approval_gated`
Use when:
- The action restarts, reconfigures, or modifies a production service
- Blast radius could affect multiple consumers
- A human should confirm the RCA before automation runs
- Risk score may occasionally exceed 0.5

Examples: `restart_service`, `cpu_scale_out`, `rollback_deploy`, `osd_reweight`

**Default for any new scenario.** When in doubt, use `approval_gated`.

### `human_only`
Use when:
- The failure pattern is recurring and the root cause is unknown
- The action is irreversible (set `irreversible: true` too)
- The scenario involves security, data loss, or compliance risk
- The signal is ambiguous and the SRE must investigate before any action

Examples: `recurring_failure_signature`, security breach indicators,
data-corruption scenarios

---

## Condition operators reference

| Operator | Applies to | Meaning |
|---|---|---|
| `gt`  | numeric | `field > threshold` |
| `lt`  | numeric | `field < threshold` |
| `gte` | numeric | `field >= threshold` |
| `lte` | numeric | `field <= threshold` |
| `eq`  | numeric | `field == threshold` |
| `ne`  | numeric | `field != threshold` |
| `true`  | boolean | `bool(field) is True`  |
| `false` | boolean | `bool(field) is False` |

For boolean fields like `log_anomaly_detected`, use `"true"` (quoted string)
in the YAML to avoid YAML parsing ambiguity.

---

## Available ObsFeatures fields

These are the fields you can reference in `conditions[].field`.

### Compute metrics
| Field | Type | Description |
|---|---|---|
| `error_rate` | float 0.0–1.0 | Proportion of 5xx responses |
| `latency_p95` | float (seconds) | 95th-percentile request latency |
| `latency_p99` | float (seconds) | 99th-percentile request latency |
| `cpu_usage` | float 0.0–1.0 | CPU utilisation fraction |
| `memory_usage` | float 0.0–1.0 | Memory utilisation fraction |
| `request_rate` | float | Requests per second |
| `active_connections` | int | Current open connections |

### Storage metrics
| Field | Type | Description |
|---|---|---|
| `osd_up_count` | int | Number of OSDs with status=up |
| `osd_total_count` | int | Total number of OSDs |
| `pool_usage_pct` | float 0.0–1.0 | Storage pool fill level |
| `cluster_health_score` | int 0–2 | 0=ERROR 1=WARN 2=OK |
| `degraded_pgs` | int | Degraded Placement Groups |
| `io_latency` | float (seconds) | PVC IO latency |
| `pvc_iops` | float | PVC IOPS |

### Log-derived signals
| Field | Type | Description |
|---|---|---|
| `recent_error_count` | int | ERROR lines in last 5 min |
| `recent_warning_count` | int | WARN lines in last 5 min |
| `log_anomaly_detected` | bool | Heuristic spike in error log rate |

### Recurrence tracking
| Field | Type | Description |
|---|---|---|
| `recurrence_count` | int | Times same alert fired in last 6h |

---

## Adding scenarios for new domains

To add scenarios for `network`, `database`, or `security` agents:

### 1. Create the scenarios directory

```bash
mkdir obs-intelligence/scenarios/network
touch obs-intelligence/scenarios/network/.gitkeep
```

### 2. Add domain-specific features to ObsFeatures

Edit [`obs-intelligence/app/obs_intelligence/models.py`](obs-intelligence/app/obs_intelligence/models.py)
and add fields to the `ObsFeatures` dataclass:

```python
# ── Network metrics ──────────────────────────────────────────────────────────
packet_loss_pct: float = 0.0       # packet loss percentage (0.0–1.0)
tcp_retransmit_rate: float = 0.0   # TCP retransmit ratio
dns_resolution_errors: int = 0     # DNS lookup failures in last 5 min
```

### 3. Update the feature extractor

Edit [`obs-intelligence/app/obs_intelligence/feature_extractor.py`](obs-intelligence/app/obs_intelligence/feature_extractor.py)
to populate the new fields from Prometheus metrics or Loki log patterns.

### 4. Update the `scenario_loader` domain validation

Edit `_VALID_DOMAINS` in
[`obs-intelligence/app/obs_intelligence/scenario_loader.py`](obs-intelligence/app/obs_intelligence/scenario_loader.py):

```python
_VALID_DOMAINS = frozenset({"compute", "storage", "network", "database", "security"})
```

### 5. Write your scenario YAMLs

Create `obs-intelligence/scenarios/network/packet_loss_cascade.yaml` etc.

### 6. Update the SRE Reasoning Agent

Add domain-specific signal interpretation to
`SREReasoningAgent._build_degradation_summary()`,
`_build_causal_chain()`, and `_build_recommended_actions()` in
[`obs-intelligence/app/obs_intelligence/sre_reasoning_agent.py`](obs-intelligence/app/obs_intelligence/sre_reasoning_agent.py).

---

## Testing your scenario locally

```bash
cd "obs-intelligence/app"

# Syntax check
py -3 -c "
import yaml
from obs_intelligence.scenario_loader import load_scenarios
scenarios = load_scenarios('../scenarios')
print(f'Loaded {len(scenarios)} scenarios')
for s in scenarios:
    print(f'  {s.scenario_id:45s}  autonomy={s.autonomy:15s}  conditions={len(s.conditions)}')
"

# Correlation test with synthetic features
py -3 - <<'EOF'
from datetime import datetime, timezone
from obs_intelligence.models import ObsFeatures
from obs_intelligence.scenario_correlator import load_catalog, match_scenarios

catalog = load_catalog(domain="compute", scenarios_dir="../scenarios")
features = ObsFeatures(
    alert_name="HighErrorRate",
    service_name="frontend-api",
    severity="warning",
    domain="compute",
    timestamp=datetime.now(timezone.utc),
    error_rate=0.18,
    latency_p99=1.8,
    log_anomaly_detected=True,
    recent_error_count=12,
)
matches = match_scenarios(features, catalog)
for m in matches:
    print(f"  {m.scenario_id:45s}  confidence={m.confidence:.0%}  matched={m.matched_features}")
EOF
```

---

## Worked examples

### Example 1 — Network: TCP Retransmit Storm

```yaml
scenario_id:    tcp_retransmit_storm
display_name:   "TCP Retransmit Storm"
domain:         network
version:        "1.0"

alert_name_patterns:
  - "HighRetransmitRate*"
  - "NetworkDegradation*"

alert_match_weight: 0.25
confidence_threshold: 0.40

action:       circuit_break_dependency
autonomy:     approval_gated
irreversible: false

rca: >
  TCP retransmit rate has exceeded the baseline threshold, indicating network
  congestion or a partial link failure between services. This typically
  manifests as elevated application latency and intermittent connection errors.

playbook_hint: >
  1. Check physical/virtual link health: ifconfig, ethtool, cloud VPC metrics
  2. Review traffic shaping policies and QoS settings
  3. Enable circuit breaker on affected service pair to shed load
  4. Coordinate with network team for packet capture analysis

conditions:
  - field:     tcp_retransmit_rate
    operator:  gte
    threshold: 0.05
    weight:    0.60

  - field:     latency_p99
    operator:  gte
    threshold: 0.5
    weight:    0.40
```

### Example 2 — Database: Connection Pool Exhaustion

```yaml
scenario_id:    db_connection_pool_exhaustion
display_name:   "Database Connection Pool Exhaustion"
domain:         database
version:        "1.0"

alert_name_patterns:
  - "DBConnectionPoolFull*"
  - "DBTooManyConnections*"

alert_match_weight: 0.30
confidence_threshold: 0.40

action:       investigate_errors
autonomy:     approval_gated
irreversible: false

rca: >
  The database connection pool has reached or exceeded its configured maximum.
  New queries are queuing or timing out. Common causes: connection leak in
  application code, missing connection.close() calls, or sudden traffic spike.

playbook_hint: >
  1. Check current pool utilisation: SHOW PROCESSLIST / pg_stat_activity
  2. Identify long-running or abandoned connections
  3. Temporarily increase max_connections if emergency headroom needed
  4. Review application ORM connection management and pool settings

conditions:
  - field:     active_connections
    operator:  gte
    threshold: 90
    weight:    0.60

  - field:     error_rate
    operator:  gte
    threshold: 0.05
    weight:    0.40
```

### Example 3 — Security: Anomalous Login Spike (human_only)

```yaml
scenario_id:    anomalous_login_spike
display_name:   "Anomalous Authentication Spike"
domain:         security
version:        "1.0"

alert_name_patterns:
  - "AbnormalLoginRate*"
  - "AuthFailureStorm*"

alert_match_weight: 0.40
confidence_threshold: 0.35

action:       escalate
autonomy:     human_only   # Security events must always involve a human
irreversible: false

rca: >
  Authentication attempt rate has spiked significantly above baseline,
  which may indicate a credential stuffing attack, misconfigured automation,
  or a compromised service account. Manual security investigation required
  before any automated action is taken.

playbook_hint: >
  1. Identify source IPs and GeoIP distribution of failed auth attempts
  2. Check for known bad IPs against threat intelligence feeds
  3. Alert security team immediately — do NOT block IPs without approval
  4. Review WAF / rate-limiter logs for correlated traffic

conditions:
  - field:     recent_error_count
    operator:  gte
    threshold: 50
    weight:    0.50

  - field:     log_anomaly_detected
    operator:  "true"
    weight:    0.50
```

---

For questions, open a GitHub issue with the `scenario-catalog` label.
