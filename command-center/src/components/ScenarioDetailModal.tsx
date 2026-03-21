// ═══════════════════════════════════════════════════════════════════════════════
// Scenario Detail Modal — Full Detail View for Scenario
// ═══════════════════════════════════════════════════════════════════════════════

import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  IconButton,
  Box,
  Typography,
  Chip,
  Divider,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Grid,
  CircularProgress,
  Alert,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
} from '@mui/material';
import {
  Close,
  Code,
  Description,
  Build,
  PlayArrow,
} from '@mui/icons-material';
import { Scenario, HistoricalPipelineRun, AutonomyDecision } from '../types';
import { getScenarioRuns } from '../api';

interface ScenarioDetailModalProps {
  scenario: Scenario | null;
  onClose: () => void;
  onPlaybackRun: (run: HistoricalPipelineRun) => void;
}

const ScenarioDetailModal: React.FC<ScenarioDetailModalProps> = ({ 
  scenario, 
  onClose,
  onPlaybackRun,
}) => {
  const [historicalRuns, setHistoricalRuns] = useState<HistoricalPipelineRun[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (scenario) {
      fetchHistoricalRuns();
    }
  }, [scenario]);

  const fetchHistoricalRuns = async () => {
    if (!scenario) return;
    
    setLoading(true);
    setError(null);
    try {
      const runs = await getScenarioRuns(scenario.scenario_id);
      setHistoricalRuns(runs);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch historical runs');
    } finally {
      setLoading(false);
    }
  };

  const getAutonomyColor = (decision: AutonomyDecision) => {
    switch (decision) {
      case 'AUTONOMOUS': return '#4caf50';
      case 'APPROVAL_GATED': return '#ff9800';
      case 'HUMAN_ONLY': return '#f44336';
      default: return '#757575';
    }
  };

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  if (!scenario) return null;

  return (
    <Dialog 
      open={!!scenario} 
      onClose={onClose}
      maxWidth="lg"
      fullWidth
      PaperProps={{
        sx: {
          background: '#0a0e27',
          border: '1px solid rgba(255, 255, 255, 0.1)',
        },
      }}
    >
      <DialogTitle sx={{ 
        borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
      }}>
        <Box>
          <Typography variant="h5" sx={{ fontWeight: 700, mb: 0.5 }}>
            {scenario.display_name}
          </Typography>
          <Typography variant="caption" color="text.secondary" sx={{ fontFamily: 'monospace' }}>
            {scenario.scenario_id}
          </Typography>
        </Box>
        <IconButton onClick={onClose}>
          <Close />
        </IconButton>
      </DialogTitle>

      <DialogContent sx={{ p: 3 }}>
        {/* Header Stats */}
        <Grid container spacing={2} sx={{ mb: 3 }}>
          <Grid item xs={12} md={3}>
            <Paper sx={{ p: 2, background: '#1a1f3a' }}>
              <Typography variant="caption" color="text.secondary">
                Autonomy Badge
              </Typography>
              <Chip
                label={scenario.autonomy_badge}
                sx={{
                  mt: 1,
                  background: getAutonomyColor(scenario.autonomy_badge),
                  color: '#fff',
                  fontWeight: 600,
                }}
              />
            </Paper>
          </Grid>
          <Grid item xs={12} md={3}>
            <Paper sx={{ p: 2, background: '#1a1f3a' }}>
              <Typography variant="caption" color="text.secondary">
                Times Seen
              </Typography>
              <Typography variant="h5" sx={{ mt: 1, fontWeight: 700 }}>
                {scenario.times_seen}
              </Typography>
            </Paper>
          </Grid>
          <Grid item xs={12} md={3}>
            <Paper sx={{ p: 2, background: '#1a1f3a' }}>
              <Typography variant="caption" color="text.secondary">
                Avg Resolution
              </Typography>
              <Typography variant="h5" sx={{ mt: 1, fontWeight: 700 }}>
                {scenario.average_resolution_time 
                  ? `${scenario.average_resolution_time.toFixed(1)}s`
                  : 'N/A'}
              </Typography>
            </Paper>
          </Grid>
          <Grid item xs={12} md={3}>
            <Paper sx={{ p: 2, background: '#1a1f3a' }}>
              <Typography variant="caption" color="text.secondary">
                Success Rate
              </Typography>
              <Typography 
                variant="h5" 
                sx={{ 
                  mt: 1, 
                  fontWeight: 700,
                  color: scenario.success_rate && scenario.success_rate >= 90 ? '#4caf50' : '#ff9800',
                }}
              >
                {scenario.success_rate !== undefined ? `${scenario.success_rate.toFixed(1)}%` : 'N/A'}
              </Typography>
            </Paper>
          </Grid>
        </Grid>

        {/* Action */}
        <Paper sx={{ p: 2, mb: 3, background: '#1a1f3a' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
            <Build sx={{ fontSize: 18, color: '#ff9800' }} />
            <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
              Action
            </Typography>
          </Box>
          <Typography variant="body2">{scenario.action}</Typography>
        </Paper>

        {/* YAML Conditions Table */}
        <Paper sx={{ p: 2, mb: 3, background: '#1a1f3a' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
            <Code sx={{ fontSize: 18, color: '#2196f3' }} />
            <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
              Detection Conditions
            </Typography>
          </Box>
          
          {scenario.conditions && scenario.conditions.length > 0 ? (
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell sx={{ fontWeight: 700 }}>Type</TableCell>
                    <TableCell sx={{ fontWeight: 700 }}>Metric/Pattern</TableCell>
                    <TableCell sx={{ fontWeight: 700 }}>Operator</TableCell>
                    <TableCell sx={{ fontWeight: 700 }}>Threshold</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {scenario.conditions.map((condition, idx) => (
                    <TableRow key={idx}>
                      <TableCell>
                        <Chip 
                          label={condition.metric ? 'Metric' : 'Log'}
                          size="small"
                          sx={{ 
                            background: condition.metric ? '#2196f3' : '#ff9800',
                            color: '#fff',
                            fontSize: '0.7rem',
                          }}
                        />
                      </TableCell>
                      <TableCell sx={{ fontFamily: 'monospace', fontSize: '0.85rem' }}>
                        {condition.metric || condition.log_pattern || 'N/A'}
                      </TableCell>
                      <TableCell sx={{ fontFamily: 'monospace' }}>
                        {condition.operator || 'contains'}
                      </TableCell>
                      <TableCell sx={{ fontWeight: 600 }}>
                        {condition.threshold !== undefined ? condition.threshold : 'N/A'}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          ) : (
            <Typography variant="body2" color="text.secondary">
              No conditions defined
            </Typography>
          )}
        </Paper>

        {/* RCA Template */}
        <Paper sx={{ p: 2, mb: 3, background: '#1a1f3a' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
            <Description sx={{ fontSize: 18, color: '#4caf50' }} />
            <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
              Root Cause Analysis Template
            </Typography>
          </Box>
          <Typography 
            variant="body2" 
            sx={{ 
              lineHeight: 1.8,
              background: '#0a0e27',
              p: 2,
              borderRadius: 1,
              fontFamily: 'monospace',
              fontSize: '0.85rem',
            }}
          >
            {scenario.rca_template}
          </Typography>
        </Paper>

        {/* Playbook Hint */}
        <Paper sx={{ p: 2, mb: 3, background: '#1a1f3a' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
            <Build sx={{ fontSize: 18, color: '#ff9800' }} />
            <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
              Ansible Playbook
            </Typography>
          </Box>
          <Typography 
            variant="body2" 
            sx={{ 
              fontFamily: 'monospace',
              background: '#0a0e27',
              p: 2,
              borderRadius: 1,
            }}
          >
            {scenario.playbook_hint}
          </Typography>
        </Paper>

        {/* Historical Runs */}
        <Paper sx={{ p: 2, background: '#1a1f3a' }}>
          <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2 }}>
            📜 Historical Runs ({historicalRuns.length})
          </Typography>

          {loading && (
            <Box sx={{ display: 'flex', justifyContent: 'center', py: 3 }}>
              <CircularProgress />
            </Box>
          )}

          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          {!loading && !error && historicalRuns.length > 0 && (
            <List sx={{ maxHeight: 300, overflowY: 'auto' }}>
              {historicalRuns.map((run) => (
                <ListItem key={run.session_id} disablePadding>
                  <ListItemButton 
                    onClick={() => onPlaybackRun(run)}
                    sx={{
                      border: '1px solid rgba(255, 255, 255, 0.1)',
                      borderRadius: 1,
                      mb: 1,
                      '&:hover': {
                        background: '#0a0e27',
                        borderColor: '#2196f3',
                      },
                    }}
                  >
                    <PlayArrow sx={{ mr: 1, color: '#2196f3' }} />
                    <ListItemText
                      primary={
                        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
                          <Typography variant="body2" sx={{ fontWeight: 600 }}>
                            {run.service_name}
                          </Typography>
                          <Typography variant="caption" sx={{ fontFamily: 'monospace' }}>
                            {run.alert_name}
                          </Typography>
                          <Chip
                            label={run.outcome}
                            size="small"
                            sx={{
                              background: 
                                run.outcome === 'success' ? '#4caf50' :
                                run.outcome === 'failure' ? '#f44336' : '#ff9800',
                              color: '#fff',
                              fontSize: '0.65rem',
                            }}
                          />
                        </Box>
                      }
                      secondary={
                        <Typography variant="caption" color="text.secondary">
                          {formatTimestamp(run.timestamp)} • Risk: {run.risk_score.toFixed(2)} • Duration: {run.duration.toFixed(1)}s
                        </Typography>
                      }
                    />
                  </ListItemButton>
                </ListItem>
              ))}
            </List>
          )}

          {!loading && !error && historicalRuns.length === 0 && (
            <Box sx={{ textAlign: 'center', py: 3 }}>
              <Typography variant="body2" color="text.secondary">
                No historical runs found for this scenario
              </Typography>
            </Box>
          )}
        </Paper>
      </DialogContent>
    </Dialog>
  );
};

export default ScenarioDetailModal;
