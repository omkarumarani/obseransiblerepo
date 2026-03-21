# AIOps Command Center React UI

A real-time visualization dashboard for monitoring the 6-agent AIOps pipeline workflow.

## Features

- **Live Pipeline View**: Visual node graph showing real-time agent execution flow
- **Agent Details**: Click any agent to see detailed output and logs
- **Incident Dashboard**: Risk assessment, scenario matching, and autonomy decisions
- **Real-time Updates**: Auto-refresh every 3 seconds with LIVE indicator
- **Trust Score Progress**: Track service graduation to autonomous execution

## Architecture

- **React 18** + TypeScript
- **React Flow** for pipeline visualization
- **Material-UI (MUI)** for components
- **Axios** for API communication
- **Nginx** for production serving + API proxy

## Local Development

```bash
# Install dependencies
npm install

# Start development server (requires compute-agent running on :9000)
npm start

# Build for production
npm run build
```

## Docker Deployment

```bash
# Build image
docker build -t command-center:latest .

# Run container
docker run -p 3000:3000 --network aiops-network command-center:latest
```

## Environment Variables

- `REACT_APP_API_BASE_URL`: Compute agent URL (default: `http://localhost:9000`)

## API Endpoints Used

- `GET /pipeline/session/{session_id}` — Retrieve pipeline session state
- `GET /autonomy/status/{service}` — Get service autonomy status
- `GET /autonomy/tiers` — List all tier configurations
- `GET /autonomy/history` — Approval history
- `GET /health` — Health check

## Components

### PipelineFlowView
Horizontal React Flow graph with 6 agent nodes showing status, duration, and live updates.

### AgentNode
Custom React Flow node with:
- Color-coded status (idle/running/success/failed/skipped)
- Duration display
- Click → opens details drawer

### IncidentDashboard
Below-the-pipeline dashboard showing:
- Active incident metadata
- Risk score gauge
- Matched scenario with confidence
- Autonomy decision (AUTONOMOUS / APPROVAL_GATED / HUMAN_ONLY)
- Trust score progress bar

### AgentDetailsDrawer
Right-side drawer with:
- Full agent output (JSON pretty-print)
- Execution timestamps
- Error messages (if failed)

## Usage

1. Open `http://localhost:3000` in browser
2. Enter pipeline session ID (default: service name)
3. Watch agents execute in real-time
4. Click any agent node to see detailed output
5. Monitor risk score and autonomy decisions below pipeline

## Integration with Existing Stack

The Command Center connects to:
- **Compute Agent** (:9000) — Primary pipeline API
- **xyOps** (:5522) — Via compute-agent ticket links
- **Grafana** (:3001) — Via dashboard links in incidents

All API requests are proxied through Nginx to avoid CORS issues in production.
