// ═══════════════════════════════════════════════════════════════════════════════
// AIOps Command Center — Main Application Component
// ═══════════════════════════════════════════════════════════════════════════════

import React, { useState, useEffect, useCallback } from 'react';
import { formatDistanceToNow } from 'date-fns';
import {
  ThemeProvider,
  createTheme,
  CssBaseline,
  Container,
  Box,
  Typography,
  AppBar,
  Toolbar,
  Chip,
  TextField,
  Button,
  Alert,
  CircularProgress,
  Tabs,
  Tab,
} from '@mui/material';
import { Refresh, Circle } from '@mui/icons-material';
import PipelineFlowView from './components/PipelineFlowView';
import IncidentDashboard from './components/IncidentDashboard';
import AgentDetailsDrawer from './components/AgentDetailsDrawer';
import HistoryTab from './components/HistoryTab';
import PlaybackMode from './components/PlaybackMode';
import ScenariosTab from './components/ScenariosTab';
import LearningTab from './components/LearningTab';
import SessionRiskSparkline from './components/charts/SessionRiskSparkline';
import ErrorBoundary from './components/ErrorBoundary';
import { useUiStore } from './store/uiStore';
import { getPipelineSession, healthCheck } from './api';
import { PipelineSession, AgentStep, IncidentInfo, HistoricalPipelineRun } from './types';

// Dark theme for AIOps aesthetic
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#2196f3',
    },
    secondary: {
      main: '#ff9800',
    },
    background: {
      default: '#0a0e27',
      paper: '#1a1f3a',
    },
    text: {
      primary: '#e1e4e8',
      secondary: '#8b949e',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
  },
});

const POLL_INTERVAL_MS = 3000; // 3 seconds

// Agent metadata
const AGENT_NAMES = [
  'Start',
  'Logs',
  'Metrics',
  'Analyze',
  'Ticket',
  'Approval',
];

const App: React.FC = () => {
  // Global UI state (Zustand)
  const currentTab = useUiStore((state) => state.currentTab);
  const setCurrentTab = useUiStore((state) => state.setCurrentTab);
  const lastRefreshAt = useUiStore((state) => state.lastRefreshAt);
  const setLastRefreshAt = useUiStore((state) => state.setLastRefreshAt);
  const [playbackRun, setPlaybackRun] = useState<HistoricalPipelineRun | null>(null);
  
  const [sessionId, setSessionId] = useState<string>('default');
  const [inputSessionId, setInputSessionId] = useState<string>('default');
  const [session, setSession] = useState<PipelineSession | null>(null);
  const [agents, setAgents] = useState<AgentStep[]>([]);
  const [selectedAgent, setSelectedAgent] = useState<AgentStep | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [isLive, setIsLive] = useState(false);
  const [isHealthy, setIsHealthy] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // Build real agent steps from session data (no mocks, no Math.random)
  const sessionToAgents = useCallback((session: PipelineSession): AgentStep[] => {
    // Matches actual stage values set by compute-agent/app/pipeline.py
    const stageOrder = [
      'created', 'started', 'logs', 'metrics', 'analyzed',
      'ticket_enriched', 'awaiting_approval', 'autonomous_executing', 'complete',
    ];
    const currentIdx = stageOrder.indexOf(session.stage);

    // Stage that marks each agent as "done" (currentIdx must exceed this to show success)
    const agentStageMap: Record<number, string> = {
      1: 'started',
      2: 'logs',
      3: 'metrics',
      4: 'analyzed',
      5: 'ticket_enriched',
      6: 'awaiting_approval',
    };

    return AGENT_NAMES.map((name, index) => {
      const agentId = index + 1;
      const doneStage = agentStageMap[agentId];
      const doneIdx   = stageOrder.indexOf(doneStage);
      const status: AgentStep['status'] =
        currentIdx > doneIdx  ? 'success' :
        currentIdx === doneIdx ? 'running' : 'idle';

      let output: Record<string, unknown> | undefined;
      if (status === 'success') {
        if (agentId === 1) output = { ticket_id: session.ticket_id, ticket_num: session.ticket_num };
        if (agentId === 2) output = { log_lines: session.logs?.split('\n').length ?? 0, preview: session.logs?.slice(0, 200) };
        if (agentId === 3) output = session.metrics ? { ...session.metrics } : undefined;
        if (agentId === 4) output = session.analysis ? { ...session.analysis } : undefined;
        if (agentId === 5) output = { ticket_id: session.ticket_id, ticket_num: session.ticket_num };
        if (agentId === 6) output = { approval_id: session.approval_id, decision: session.autonomy_decision };
      }

      // Use real timing; agent 6 may complete via autonomous_executing instead of awaiting_approval
      const duration = status === 'success' && session.stage_durations
        ? (agentId === 6
            ? (session.stage_durations['awaiting_approval']
                ?? session.stage_durations['autonomous_executing']
                ?? session.stage_durations['complete']
                ?? 0)
            : (session.stage_durations[doneStage] ?? 0))
        : 0;

      return { id: agentId, name, status, duration, output };
    });
  }, []);

  // Convert session to incident info
  const sessionToIncident = useCallback((session: PipelineSession): IncidentInfo => {
    return {
      service_name: session.service_name,
      alert_name: session.alert_name,
      severity: session.severity,
      started_at: new Date(session.created_at * 1000).toISOString(),
      risk_score: session.risk_score || 0,
      risk_level: session.risk_level || 'unknown',
      scenario_match: session.analysis?.scenario_id,
      scenario_confidence: session.analysis?.scenario_confidence,
      autonomy_decision: session.autonomy_decision || 'APPROVAL_GATED',
      trust_progress: session.trust_progress || '0 / 0 approvals',
      trust_metrics: session.trust_metrics,
    };
  }, []);

  // Fetch pipeline session
  const fetchSession = useCallback(async () => {
    try {
      setError(null);
      const data = await getPipelineSession(sessionId);
      setSession(data);
      setLastRefreshAt(new Date().toISOString());
      
      // Build agents from real session data
      const generatedAgents = sessionToAgents(data);
      setAgents(generatedAgents);
      
      // Check if pipeline is active (any agent running)
      const hasRunningAgent = generatedAgents.some(a => a.status === 'running');
      setIsLive(hasRunningAgent);
    } catch (err: any) {
      console.error('Failed to fetch session:', err);
      setError(err.message || 'Failed to fetch pipeline session');
      setIsLive(false);
    }
  }, [sessionId, sessionToAgents, setLastRefreshAt]);

  // Health check
  const checkHealth = useCallback(async () => {
    const healthy = await healthCheck();
    setIsHealthy(healthy);
  }, []);

  // Initial load
  useEffect(() => {
    fetchSession();
    checkHealth();
  }, [fetchSession, checkHealth]);

  // Auto-refresh polling
  useEffect(() => {
    const interval = setInterval(() => {
      fetchSession();
    }, POLL_INTERVAL_MS);

    return () => clearInterval(interval);
  }, [fetchSession]);

  // Handle session ID change
  const handleChangeSession = () => {
    setLoading(true);
    setSessionId(inputSessionId);
    setTimeout(() => setLoading(false), 500);
  };

  // Handle agent node click
  const handleNodeClick = (agentId: number) => {
    const agent = agents.find(a => a.id === agentId);
    if (agent) {
      setSelectedAgent(agent);
      setDrawerOpen(true);
    }
  };

  // Handle tab change
  const handleTabChange = (_: React.SyntheticEvent, newValue: 'live' | 'history' | 'scenarios' | 'learning') => {
    setCurrentTab(newValue);
  };

  // Handle history run selection (enter playback mode)
  const handleSelectHistoricalRun = (run: HistoricalPipelineRun) => {
    setPlaybackRun(run);
  };

  // Handle close playback mode
  const handleClosePlayback = () => {
    setPlaybackRun(null);
  };

  // If in playback mode, show only playback component
  if (playbackRun) {
    return (
      <ThemeProvider theme={darkTheme}>
        <CssBaseline />
        <PlaybackMode run={playbackRun} onClose={handleClosePlayback} />
      </ThemeProvider>
    );
  }

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      
      {/* App Bar */}
      <AppBar position="static" sx={{ background: '#1a1f3a', boxShadow: 'none', borderBottom: '1px solid rgba(255, 255, 255, 0.1)' }}>
        <Toolbar>
          <Typography variant="h5" sx={{ flexGrow: 1, fontWeight: 700 }}>
            🤖 AIOps Command Center
          </Typography>
          
          {/* Live indicator */}
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mr: 3 }}>
            <Circle 
              sx={{ 
                fontSize: 12, 
                color: isLive ? '#4caf50' : '#757575',
                animation: isLive ? 'pulse 2s infinite' : 'none',
              }} 
            />
            <Typography variant="body2" sx={{ fontWeight: 600 }}>
              {isLive ? 'LIVE' : 'IDLE'}
            </Typography>
          </Box>
          
          {/* Health indicator */}
          <Chip
            label={isHealthy ? 'HEALTHY' : 'OFFLINE'}
            size="small"
            sx={{
              background: isHealthy ? '#4caf50' : '#f44336',
              color: '#fff',
              fontWeight: 600,
            }}
          />
        </Toolbar>
      </AppBar>

      {/* Tabs */}
      <Box sx={{ borderBottom: 1, borderColor: 'divider', background: '#1a1f3a' }}>
        <Tabs 
          value={currentTab} 
          onChange={handleTabChange}
          sx={{
            '& .MuiTab-root': {
              color: 'text.secondary',
              fontWeight: 600,
              fontSize: '0.95rem',
            },
            '& .Mui-selected': {
              color: '#2196f3 !important',
            },
          }}
        >
          <Tab label="🔴 Live Pipeline" value="live" />
          <Tab label="📜 History" value="history" />
          <Tab label="📚 Scenarios" value="scenarios" />
          <Tab label="🧠 Learning" value="learning" />
        </Tabs>
      </Box>

      <Container maxWidth="xl" sx={{ py: 4 }}>
        {/* LIVE TAB */}
        {currentTab === 'live' && (
          <>
            {/* Session selector */}
            <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
              <TextField
                label="Pipeline Session ID"
                value={inputSessionId}
                onChange={(e) => setInputSessionId(e.target.value)}
                size="small"
                sx={{ flexGrow: 1, maxWidth: 400 }}
                variant="outlined"
              />
              <Button
                variant="contained"
                onClick={handleChangeSession}
                disabled={loading || inputSessionId === sessionId}
                startIcon={loading ? <CircularProgress size={16} /> : <Refresh />}
              >
                Load Session
              </Button>
            </Box>

            {/* Error display */}
            {error && (
              <Alert severity="error" sx={{ mb: 3 }}>
                {error}
              </Alert>
            )}

            {/* Pipeline Flow */}
            {session && agents.length > 0 && (
              <>
                <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
                  Agent Pipeline — {session.session_id}
                </Typography>
                
                <PipelineFlowView agents={agents} onNodeClick={handleNodeClick} />
                
                {/* Incident Dashboard */}
                <IncidentDashboard incident={sessionToIncident(session)} />
                
                {/* Session metadata */}
                <Box sx={{ mt: 3 }} className="grid grid-cols-1 gap-3 lg:grid-cols-2">
                  <SessionRiskSparkline
                    riskScore={session.risk_score || 0}
                    startedAtIso={new Date(session.created_at * 1000).toISOString()}
                    sessionId={session.session_id}
                  />
                <Box sx={{ p: 2, background: '#1a1f3a', borderRadius: 2 }}>
                  <Typography variant="caption" color="text.secondary">
                    Session ID: {session.session_id} | 
                    Stage: {session.stage} | 
                    Ticket: #{session.ticket_num} | 
                    Age: {session.age_seconds}s | 
                    Last updated: {formatDistanceToNow(new Date(lastRefreshAt), { addSuffix: true })}
                  </Typography>
                </Box>
                </Box>
              </>
            )}

            {/* Loading state */}
            {!session && !error && (
              <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 400 }}>
                <CircularProgress />
              </Box>
            )}
          </>
        )}

        {/* HISTORY TAB */}
        {currentTab === 'history' && (
          <ErrorBoundary fallbackLabel="History failed to load">
            <HistoryTab onSelectRun={handleSelectHistoricalRun} />
          </ErrorBoundary>
        )}

        {/* SCENARIOS TAB */}
        {currentTab === 'scenarios' && (
          <ErrorBoundary fallbackLabel="Scenarios failed to load">
            <ScenariosTab onPlaybackRun={handleSelectHistoricalRun} />
          </ErrorBoundary>
        )}

        {currentTab === 'learning' && (
          <ErrorBoundary fallbackLabel="Learning failed to load">
            <LearningTab />
          </ErrorBoundary>
        )}
      </Container>

      {/* Agent Details Drawer */}
      <AgentDetailsDrawer
        open={drawerOpen}
        agent={selectedAgent}
        onClose={() => setDrawerOpen(false)}
      />

      <style>{`
        @keyframes pulse {
          0%, 100% {
            opacity: 1;
          }
          50% {
            opacity: 0.3;
          }
        }
      `}</style>
    </ThemeProvider>
  );
};

export default App;
