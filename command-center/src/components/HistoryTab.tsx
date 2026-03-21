// ═══════════════════════════════════════════════════════════════════════════════
// History Tab — List of Past Pipeline Runs
// ═══════════════════════════════════════════════════════════════════════════════

import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Grid,
  IconButton,
  CircularProgress,
  Alert,
} from '@mui/material';
import { Refresh, PlayArrow } from '@mui/icons-material';
import { HistoricalPipelineRun, HistoryFilters, AutonomyDecision, PipelineOutcome } from '../types';
import { getHistoricalRuns } from '../api';

interface HistoryTabProps {
  onSelectRun: (run: HistoricalPipelineRun) => void;
}

const HistoryTab: React.FC<HistoryTabProps> = ({ onSelectRun }) => {
  const [runs, setRuns] = useState<HistoricalPipelineRun[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Filters
  const [filters, setFilters] = useState<HistoryFilters>({
    domain: 'all',
    autonomyDecision: 'all',
    outcome: 'all',
    searchText: '',
  });

  // Fetch historical runs
  const fetchRuns = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getHistoricalRuns(filters);
      setRuns(data);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch historical runs');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRuns();
  }, [filters]);

  // Helper functions for status colors
  const getOutcomeColor = (outcome: PipelineOutcome) => {
    switch (outcome) {
      case 'success': return '#4caf50';
      case 'failure': return '#f44336';
      case 'pending': return '#ff9800';
      default: return '#757575';
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

  const getRiskColor = (score: number) => {
    if (score < 0.3) return '#4caf50';
    if (score < 0.7) return '#ff9800';
    return '#f44336';
  };

  // Format timestamp
  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  };

  // Format duration
  const formatDuration = (seconds: number) => {
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    const minutes = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${minutes}m ${secs.toFixed(0)}s`;
  };

  return (
    <Box>
      {/* Filter Bar */}
      <Paper sx={{ p: 2, mb: 3, background: '#1a1f3a' }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={3}>
            <TextField
              fullWidth
              size="small"
              label="Search"
              placeholder="Service, alert, scenario..."
              value={filters.searchText}
              onChange={(e) => setFilters({ ...filters, searchText: e.target.value })}
            />
          </Grid>
          
          <Grid item xs={12} md={2}>
            <FormControl fullWidth size="small">
              <InputLabel>Domain</InputLabel>
              <Select
                value={filters.domain || 'all'}
                label="Domain"
                onChange={(e) => setFilters({ ...filters, domain: e.target.value as any })}
              >
                <MenuItem value="all">All Domains</MenuItem>
                <MenuItem value="compute">Compute</MenuItem>
                <MenuItem value="storage">Storage</MenuItem>
                <MenuItem value="network">Network</MenuItem>
                <MenuItem value="database">Database</MenuItem>
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12} md={2}>
            <FormControl fullWidth size="small">
              <InputLabel>Autonomy</InputLabel>
              <Select
                value={filters.autonomyDecision || 'all'}
                label="Autonomy"
                onChange={(e) => setFilters({ ...filters, autonomyDecision: e.target.value as any })}
              >
                <MenuItem value="all">All Decisions</MenuItem>
                <MenuItem value="AUTONOMOUS">Autonomous</MenuItem>
                <MenuItem value="APPROVAL_GATED">Approval Gated</MenuItem>
                <MenuItem value="HUMAN_ONLY">Human Only</MenuItem>
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12} md={2}>
            <FormControl fullWidth size="small">
              <InputLabel>Outcome</InputLabel>
              <Select
                value={filters.outcome || 'all'}
                label="Outcome"
                onChange={(e) => setFilters({ ...filters, outcome: e.target.value as any })}
              >
                <MenuItem value="all">All Outcomes</MenuItem>
                <MenuItem value="success">Success</MenuItem>
                <MenuItem value="failure">Failure</MenuItem>
                <MenuItem value="pending">Pending</MenuItem>
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12} md={3}>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <IconButton
                onClick={fetchRuns}
                sx={{ background: 'rgba(255, 255, 255, 0.05)' }}
                disabled={loading}
              >
                <Refresh />
              </IconButton>
              <Typography variant="caption" sx={{ display: 'flex', alignItems: 'center', color: 'text.secondary' }}>
                {runs.length} runs found
              </Typography>
            </Box>
          </Grid>
        </Grid>
      </Paper>

      {/* Error display */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {/* Loading spinner */}
      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', py: 5 }}>
          <CircularProgress />
        </Box>
      )}

      {/* History Table */}
      {!loading && runs.length > 0 && (
        <TableContainer component={Paper} sx={{ background: '#1a1f3a' }}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell sx={{ fontWeight: 700 }}>Timestamp</TableCell>
                <TableCell sx={{ fontWeight: 700 }}>Service</TableCell>
                <TableCell sx={{ fontWeight: 700 }}>Alert</TableCell>
                <TableCell sx={{ fontWeight: 700 }}>Domain</TableCell>
                <TableCell sx={{ fontWeight: 700 }}>Scenario</TableCell>
                <TableCell sx={{ fontWeight: 700 }}>Risk Score</TableCell>
                <TableCell sx={{ fontWeight: 700 }}>Autonomy</TableCell>
                <TableCell sx={{ fontWeight: 700 }}>Outcome</TableCell>
                <TableCell sx={{ fontWeight: 700 }}>Duration</TableCell>
                <TableCell sx={{ fontWeight: 700 }}>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {runs.map((run) => (
                <TableRow 
                  key={run.session_id}
                  hover
                  sx={{ cursor: 'pointer' }}
                  onClick={() => onSelectRun(run)}
                >
                  <TableCell>{formatTimestamp(run.timestamp)}</TableCell>
                  
                  <TableCell>
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>
                      {run.service_name}
                    </Typography>
                  </TableCell>
                  
                  <TableCell>
                    <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: '0.85rem' }}>
                      {run.alert_name}
                    </Typography>
                  </TableCell>
                  
                  <TableCell>
                    <Chip
                      label={run.domain}
                      size="small"
                      sx={{
                        textTransform: 'capitalize',
                        fontWeight: 600,
                        fontSize: '0.75rem',
                      }}
                    />
                  </TableCell>
                  
                  <TableCell>
                    <Typography variant="body2" sx={{ fontSize: '0.85rem' }}>
                      {run.scenario_matched || 'N/A'}
                    </Typography>
                  </TableCell>
                  
                  <TableCell>
                    <Chip
                      label={run.risk_score.toFixed(2)}
                      size="small"
                      sx={{
                        background: getRiskColor(run.risk_score),
                        color: '#fff',
                        fontWeight: 700,
                      }}
                    />
                  </TableCell>
                  
                  <TableCell>
                    <Chip
                      label={run.autonomy_decision}
                      size="small"
                      sx={{
                        background: getAutonomyColor(run.autonomy_decision),
                        color: '#fff',
                        fontWeight: 600,
                        fontSize: '0.7rem',
                      }}
                    />
                  </TableCell>
                  
                  <TableCell>
                    <Chip
                      label={run.outcome}
                      size="small"
                      sx={{
                        background: getOutcomeColor(run.outcome),
                        color: '#fff',
                        fontWeight: 600,
                        textTransform: 'capitalize',
                      }}
                    />
                  </TableCell>
                  
                  <TableCell>
                    <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                      {formatDuration(run.duration)}
                    </Typography>
                  </TableCell>
                  
                  <TableCell>
                    <IconButton
                      size="small"
                      onClick={(e) => {
                        e.stopPropagation();
                        onSelectRun(run);
                      }}
                      sx={{ color: '#2196f3' }}
                    >
                      <PlayArrow />
                    </IconButton>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}

      {/* Empty state */}
      {!loading && runs.length === 0 && (
        <Box sx={{ textAlign: 'center', py: 8 }}>
          <Typography variant="h6" color="text.secondary">
            No pipeline runs found
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
            Try adjusting your filters or wait for new incidents to be processed
          </Typography>
        </Box>
      )}
    </Box>
  );
};

export default HistoryTab;
