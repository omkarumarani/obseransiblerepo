# AIOps Integration: xyOps + Observability Stack
## Demo Startup & Usage Guide

---

## What Was Added

```
Observability Learning Project        xyOps (AIOps Platform)
──────────────────────────────        ─────────────────────────────────
frontend-api  ──OTel──┐               lib/telemetry.js  (NEW — OTel SDK)
backend-api   ──OTel──┤               lib/main.js       (modified)
aiops-bridge  ──OTel──┤               package.json      (OTel packages added)
xyops         ──OTel──┘
                      ▼
              otel-collector:4317
              ┌───────┬────────┬────────┐
           traces   metrics   logs
              │        │        │
           Tempo   Prometheus  Loki
                              │
                          Alertmanager ← Prometheus alert rules
                              │
                          aiops-bridge (new)
                              │
                           xyOps API
                              │
                         Incident Tickets
```

---

## Step 1 — First Start (Full Stack)

```powershell
cd "c:\t\Observability Learning"

# Start full AIOps stack (includes xyOps, Alertmanager, AIOps bridge)
docker compose --profile loadgen up --build
```

This starts:
- All original services (otel-collector, prometheus, tempo, loki, grafana, frontend-api, backend-api, loadgen)
- **NEW**: alertmanager, aiops-bridge, xyops

Build time: ~5-10 minutes on first run (xyOps builds Node.js + satellite agent from source).

---

## Step 2 — Verify Services Are Up

| Service | URL | What to check |
|---|---|---|
| Grafana | http://localhost:3001 | Login admin/admin, see dashboards |
| Prometheus | http://localhost:9090 | Targets tab → all green |
| **Alertmanager** | http://localhost:9093 | Alerts tab (empty when all clear) |
| **xyOps** | http://localhost:5522 | Login admin/admin → Tickets tab |
| **AIOps Bridge** | http://localhost:9000/health | Should return `{"status":"ok"}` |
| Tempo | http://localhost:3200/ready | Returns `ready` |
| Loki | http://localhost:3100/ready | Returns `ready` |

---

## Step 3 — Configure xyOps API Key (one-time setup)

1. Open **http://localhost:5522** → login `admin` / `admin`
2. Click **Admin** (top-right) → **API Keys** → **Create New Key**
3. Name it: `aiops-bridge`
4. Copy the generated key
5. Edit `docker-compose.yml` — find the `aiops-bridge` service:
   ```yaml
   XYOPS_API_KEY: "PASTE_YOUR_KEY_HERE"
   ```
6. Restart the bridge:
   ```powershell
   docker compose restart aiops-bridge
   ```

---

## Step 4 — Trigger the AIOps Loop (Demo)

### Option A: Manual test (immediate, no waiting)

Send a fake alert directly to the bridge to create a xyOps ticket:

```powershell
# Fire a test "HighErrorRate" alert for frontend-api
$body = @{
  status = "firing"
  alerts = @(
    @{
      status = "firing"
      labels = @{
        alertname   = "HighErrorRate"
        service_name = "frontend-api"
        severity    = "warning"
      }
      annotations = @{
        summary      = "High error rate on frontend-api"
        description  = "frontend-api error rate is 15% over last 2 minutes."
        dashboard_url = "http://localhost:3001/d/obs-overview"
      }
      startsAt = (Get-Date -Format "o")
    }
  )
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Method POST -Uri "http://localhost:9000/webhook" `
  -ContentType "application/json" -Body $body
```

Then go to **http://localhost:5522** → **Tickets** to see the incident ticket with:
- Alert name, service, severity
- Link to Grafana dashboard
- The OTel trace_id of the bridge span

Click the trace_id → open Grafana → Tempo → paste the trace_id to see the full incident detection span waterfall.

### Option B: Real trigger via Troublemaker (automatic)

```powershell
# Start the troublemaker — it will cause enough errors to fire real Prometheus alerts
docker compose --profile loadgen --profile troublemaker up --build
```

Wait ~2-5 minutes. When an `error_spike` or `backend_failure` scenario fires:
1. **Prometheus** evaluates the `HighErrorRate` rule
2. **Alertmanager** routes it to the bridge
3. **AIOps Bridge** creates a xyOps ticket with trace_id
4. Go to xyOps → Tickets to see the incident

### Option C: Resolve the alert

```powershell
# Send a "resolved" payload to close open tickets
$body = @{
  status = "resolved"
  alerts = @(
    @{
      status = "resolved"
      labels = @{
        alertname    = "HighErrorRate"
        service_name = "frontend-api"
        severity     = "warning"
      }
      annotations = @{
        summary = "High error rate on frontend-api"
      }
    }
  )
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Method POST -Uri "http://localhost:9000/webhook" `
  -ContentType "application/json" -Body $body
```

The bridge will find and close the open ticket automatically.

---

## Step 5 — See xyOps Spans in Tempo

1. Open **http://localhost:5522** and perform any action (view tickets, check jobs)
2. Open **Grafana** → **Explore** → select **Tempo** datasource
3. Query: `{service.name="xyops"}` (TraceQL)
4. You'll see HTTP request spans FROM xyOps in the same trace store as your frontend/backend spans

---

## Step 6 — Check Prometheus Alert Rules

```powershell
# See all alert rules and their current state
curl http://localhost:9090/api/v1/rules | python -m json.tool

# See active (firing) alerts
curl http://localhost:9090/api/v1/alerts | python -m json.tool
```

In the Prometheus UI: http://localhost:9090/alerts — shows each rule as green (inactive) or red (firing).

---

## The Complete AIOps Loop

```
1. Traffic flows through frontend-api → backend-api
2. OTel SDK records every request as spans (traces), counters (metrics), log lines
3. Prometheus collects metrics every 15s via otel-collector:8889
4. Prometheus evaluates alert-rules.yml every 15s:
     - HighErrorRate: error rate > 5% for 1 min  → fires
     - HighP99Latency: p99 > 3s for 2 min        → fires
     - TrafficSpike: current > 5x baseline        → fires
5. Alertmanager receives the firing alert, waits group_wait (30s), routes to aiops-bridge
6. aiops-bridge.POST /webhook:
     a. Creates OTel span (visible in Tempo)
     b. Calls xyOps /api/app/create_ticket/v1
     c. Embeds trace_id in the ticket body
7. xyOps ticket created: severity, service, dashboard link, trace_id
8. Engineer opens xyOps → clicks trace_id → sees Tempo waterfall showing:
     [frontend-api SERVER span]
       [backend-api CLIENT span]
         [aiops-bridge process_alert span] ← the moment the incident was detected
9. Alert clears → aiops-bridge auto-closes the ticket with resolution timestamp
10. Complete audit trail: detected ← trace_id → resolved
```

---

## Useful Commands

```powershell
# Watch bridge logs (see all webhook processing in real time)
docker compose logs -f aiops-bridge

# Watch Alertmanager routing decisions
docker compose logs -f alertmanager

# Watch xyOps job/alert activity
docker compose logs -f xyops

# Check bridge health
curl http://localhost:9000/health

# View all Prometheus alert rules
curl http://localhost:9090/api/v1/rules

# Reload Prometheus config/rules without restart
curl -X POST http://localhost:9090/-/reload

# Stop everything (keep data)
docker compose down

# Stop + delete all data (fresh start)
docker compose down -v
```

---

## How Long to Keep Running for Good Traces

| Duration | What you see |
|---|---|
| 2 minutes | First traces appear in Tempo, first metrics in Prometheus |
| 5 minutes | Prometheus has enough data to evaluate rate() functions accurately |
| 15 minutes | You'll have a meaningful metrics graph in Grafana |
| 1 hour | Enough data to see patterns, latency histograms filled in |
| 24 hours | Full day view in Grafana, can see time-of-day patterns |

**The loadgen sends a request every 2 seconds** → at 2 min you have ~60 traces.  
**The troublemaker scenarios run every 5-25 seconds** → you'll see at least 10+ distinct scenarios per hour.

To see a Prometheus alert actually fire, you need the troublemaker running AND to wait for an `error_spike` scenario (weighted at 2/15 = 13% probability) — expect to see one within 5-10 minutes.

---

## Enterprise Next Steps

When ready to take this to production:

1. Replace `XYOPS_API_KEY: ""` with a real API key from a secrets manager
2. Add PagerDuty/OpsGenie to `alertmanager/alertmanager.yml` receivers
3. Add Prometheus remote_write to a long-term metrics store (Thanos/Grafana Mimir)
4. Replace Loki/Tempo local storage with S3 buckets
5. Deploy on Kubernetes using Helm charts (kube-prometheus-stack + grafana-helm)
6. Wire xyOps SSO to Azure AD / Okta
7. Add xyOps satellite agents to your production servers for infrastructure monitoring
