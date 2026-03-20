# End-to-End Demo Runbook — AIOps Observability Platform

> **Target audience**: SRE/DevOps engineers demonstrating the Phase 5 SRE
> Reasoning Layer capabilities to stakeholders.
>
> **Estimated demo time**: 12–15 minutes live, 5 minutes recorded.

---

## Prerequisites

| Requirement | Check |
|---|---|
| Docker Desktop running | `docker info` |
| `docker-compose` v2+ | `docker compose version` |
| Gitea reachable at http://localhost:3000 | Browser check |
| xyOps reachable at http://localhost:5000 | Browser check |
| Grafana reachable at http://localhost:3001 | Browser check |

---

## Demo setup (do this before presenting)

```powershell
cd "c:\t\Observability Learning"

# Bring the full stack up
docker compose up -d --build

# Wait ~30 seconds for all services to initialize
Start-Sleep -Seconds 30

# Verify all containers are healthy
docker compose ps
```

Expected: all containers in `Up` or `Up (healthy)` state with no restarts.

Open tabs in the browser before starting:
1. **Grafana** http://localhost:3001 → Dashboards → **SRE Incident Timeline**
2. **xyOps** http://localhost:5000 → Tickets
3. **Prometheus** http://localhost:9090 → Alerts
4. **obs-intelligence logs**: `docker compose logs -f obs-intelligence`

---

## Demo narrative and step-by-step

### Step 1 — Baseline: "Everything is healthy"

> "Here is our stack — three Python microservices (frontend-api, backend-api),
> a simulated storage cluster, an AIOps bridge, and our obs-intelligence engine
> which continuously analyses all Prometheus and Loki signals.
> The Grafana SRE Incident Timeline shows green across all services."

Point to:
- Grafana row 1 (Active Incidents & Risk): risk scores near 0
- Prometheus Alerts page: zero firing alerts

---

### Step 2 — Inject chaos: Pool fill approaching critical

```powershell
# Tell the troublemaker to push pool_usage above 85%
docker compose exec troublemaker \
  python troublemaker.py storage_fill --level 0.87
```

> "I'm now simulating a storage pool filling up — 87% of capacity.
> In production this could be excessive log retention, a backup that
> didn't clean up, or unexpectedly rapid data growth."

Wait 15–20 seconds for `loadgen` to start emitting elevated metrics.

---

### Step 3 — obs-intelligence detects the anomaly (predictive alert)

Watch `obs-intelligence` logs:

```
INFO obs_intelligence.background  - Analysis loop started
INFO obs_intelligence.scenario_correlator - Match: pool_fill_critical  confidence=0.71
INFO obs_intelligence.sre_reasoning_agent - SREAssessment built: urgency=high  autonomy=approval_gated
INFO obs_intelligence.llm_enricher - Calling OpenAI (gpt-4o-mini) for enrichment
INFO obs_intelligence.background  - Predictive alert fired: StoragePoolFillCritical
```

> "Before Prometheus has even fired an alert, obs-intelligence has already
> matched the pool_fill_critical scenario with 71% confidence.
> The SRE Reasoning Agent built a deterministic causal chain — no LLM
> involved yet — and then asked Claude or OpenAI to write a human-readable
> narrative from those structured facts."

Point to:
- xyOps Tickets list: new `[PREDICTIVE] Storage Pool Fill Critical` ticket
- Ticket body: RCA summary, causal chain, recommended actions, confidence score

---

### Step 4 — Prometheus alert fires (alert manager webhook)

After ~60 seconds, Prometheus threshold is breached:

```
StoragePoolFillCritical  FIRING  service=storage-simulator  pool_usage_pct=0.87
```

> "Now the Prometheus alert fires. AlertManager sends a webhook to
> storage-agent, which picks up the scenario analysis already completed
> by obs-intelligence and adds live step commentary to the xyOps ticket."

Point to:
- Prometheus Alerts page: `StoragePoolFillCritical` now FIRING (red)
- xyOps ticket activity feed: step comments being added in real time

---

### Step 5 — Approval gate

In xyOps:
1. Open the `[PREDICTIVE] Storage Pool Fill Critical` ticket
2. Click the **Approve** button on the Remediation Plan

> "The autonomy level for this scenario is `approval_gated` — automation
> will not run until a human explicitly approves.
> In production this is where your on-call SRE verifies the RCA before
> any infrastructure changes take place.
> I'll click Approve now."

---

### Step 6 — Ansible playbook executes

Watch `ansible-runner` logs:

```powershell
docker compose logs -f ansible-runner
```

Expected output:
```
PLAY [storage-simulator] ****
TASK [Gather facts] ****
TASK [Reduce pool fill to safe level] ****  changed
PLAY RECAP: storage-simulator ok=2 changed=1
```

> "Ansible is now executing the approved remediation playbook —
> in this case, flushing old snapshots and compressing cold data to
> bring pool usage back below 75%.
> All steps are reflected back in the xyOps ticket."

---

### Step 7 — Alert resolves, outcome recorded

After the playbook runs, `troublemaker` resets to normal levels.
Within 30–60 seconds Prometheus alert resolves.

Watch storage-agent logs:

```
INFO storage_agent - Storage alert resolved: StoragePoolFillCritical / storage-simulator
INFO storage_agent - Outcome recorded in obs-intelligence: resolved
```

> "When the alert resolves, storage-agent automatically calls
> `POST /intelligence/record-outcome` on obs-intelligence.
> This increments the `obs_intelligence_scenario_outcome_total` Prometheus
> counter for scenario `pool_fill_critical` with outcome `resolved`."

---

### Step 8 — SRE Incident Timeline updates

Switch to Grafana → SRE Incident Timeline:

> "The Grafana dashboard has now updated.
> Row 1 shows the risk score for storage-simulator returning to 0.
> Row 4 — the SRE Reasoning Layer section — shows:
> - the risk score spike and recovery on the timeline
> - a new 'resolved' bar in the Outcomes by Scenario ID chart
> - scenario match confidence history over the past hour."

Point to each panel in turn.

---

### Step 9 (bonus) — Recurring failure demo

> "Let me show one more scenario: the recurring_failure_signature.
> If the same alert fires 3 or more times within a 6-hour window,
> obs-intelligence upgrades the autonomy from `approval_gated` to
> `human_only`. No automation — the SRE must investigate."

```powershell
# Fire the same alert 3 times quickly to simulate recurrence
docker compose exec troublemaker python troublemaker.py error_spike --count 3 --interval 10
```

Point to:
- xyOps new ticket: autonomy shown as `HUMAN ONLY — manual investigation required`
- No Approve button visible — the ticket cannot be auto-remediated

---

## Demo teardown

```powershell
# Reset troublemaker to baseline
docker compose exec troublemaker python troublemaker.py reset

# Or tear down everything
docker compose down
```

---

## Troubleshooting the demo

| Symptom | Fix |
|---|---|
| No predictive alert ticket in xyOps | Check obs-intelligence logs for LLM errors; ensure at least one of `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` is set in `.env` |
| Ansible step not running after Approve | Check aiops-bridge logs; verify `XYOPS_URL` env var points to `http://xyops-main:5000` |
| Grafana shows "No data" in SRE panels | Verify obs-intelligence is exposing metrics on port 9100; check Prometheus scrape config |
| Alert never fires | Confirm `troublemaker` ran successfully and `pool_usage_pct` metric is above threshold (check Prometheus query: `ceph_pool_percent_used`) |
| Claude failover log shown | OpenAI API key may be missing or exhausted; Claude is the automatic fallback — demo still works |

---

## Key talking points

1. **Deterministic reasoning first** — `SREReasoningAgent` builds the causal
   chain, impact prediction, and recommended actions *without* an LLM.
   The LLM only writes the narrative paragraph. No hallucinated RCAs.

2. **Two LLM providers** — OpenAI is tried first; if it fails (rate limit,
   outage, missing key), Claude is the automatic fallback.
   If both fail, the deterministic analysis still reaches the engineer.

3. **Transparency** — `confidence` scores and matched feature conditions are
   visible in every xyOps ticket so the SRE knows exactly why automation
   was triggered.

4. **Autonomy ladder** — `autonomous → approval_gated → human_only`.
   Recurring failures automatically step up to `human_only`,
   preventing automation from masking deeper systemic problems.

5. **Closed feedback loop** — every resolution is recorded back into
   obs-intelligence as a `scenario_outcome` metric, enabling Grafana
   dashboards and future ML training on outcome data.
