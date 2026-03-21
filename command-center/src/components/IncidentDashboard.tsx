// ═══════════════════════════════════════════════════════════════════════════════
// Incident Dashboard — Shows incident details and metrics below pipeline
// ═══════════════════════════════════════════════════════════════════════════════

import React from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Chip,
  LinearProgress,
  Grid,
  Divider,
} from '@mui/material';
import {
  Warning,
  CheckCircle,
  Error as ErrorIcon,
  Lock,
  LockOpen,
  Person,
} from '@mui/icons-material';
import { IncidentInfo, AutonomyDecision, RiskLevel } from '../types';

interface IncidentDashboardProps {
  incident: IncidentInfo;
}

const getSeverityIcon = (severity: string) => {
  switch (severity) {
    case 'critical':
      return <ErrorIcon sx={{ color: '#f44336' }} />;
    case 'warning':
      return <Warning sx={{ color: '#ff9800' }} />;
    default:
      return <CheckCircle sx={{ color: '#4caf50' }} />;
  }
};

const getSeverityColor = (severity: string): string => {
  switch (severity) {
    case 'critical':
      return '#f44336';
    case 'warning':
      return '#ff9800';
    default:
      return '#4caf50';
  }
};

const getRiskColor = (riskLevel: RiskLevel): string => {
  switch (riskLevel) {
    case 'critical':
      return '#d32f2f';
    case 'high':
      return '#f44336';
    case 'medium':
      return '#ff9800';
    case 'low':
      return '#4caf50';
    default:
      return '#757575';
  }
};

const getAutonomyIcon = (decision: AutonomyDecision) => {
  switch (decision) {
    case 'AUTONOMOUS':
      return <LockOpen sx={{ color: '#4caf50' }} />;
    case 'APPROVAL_GATED':
      return <Lock sx={{ color: '#ff9800' }} />;
    case 'HUMAN_ONLY':
      return <Person sx={{ color: '#f44336' }} />;
  }
};

const getAutonomyColor = (decision: AutonomyDecision): string => {
  switch (decision) {
    case 'AUTONOMOUS':
      return '#4caf50';
    case 'APPROVAL_GATED':
      return '#ff9800';
    case 'HUMAN_ONLY':
      return '#f44336';
  }
};

const IncidentDashboard: React.FC<IncidentDashboardProps> = ({ incident }) => {
  const {
    service_name,
    alert_name,
    severity,
    started_at,
    risk_score,
    risk_level,
    scenario_match,
    scenario_confidence,
    autonomy_decision,
    trust_progress,
    trust_metrics,
  } = incident;

  return (
    <Card sx={{ background: '#1a1f3a', borderRadius: 2, mt: 3 }}>
      <CardContent>
        <Typography variant="h5" sx={{ mb: 3, fontWeight: 600 }}>
          Active Incident
        </Typography>
        
        <Grid container spacing={3}>
          {/* Left Column — Incident Details */}
          <Grid item xs={12} md={4}>
            <Box>
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                SERVICE NAME
              </Typography>
              <Typography variant="h6" sx={{ mb: 2 }}>
                <code style={{ 
                  background: '#0a0e27', 
                  padding: '4px 8px', 
                  borderRadius: 4,
                  fontSize: '0.9rem',
                }}>
                  {service_name}
                </code>
              </Typography>
              
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                ALERT NAME
              </Typography>
              <Typography variant="h6" sx={{ mb: 2 }}>
                {alert_name}
              </Typography>
              
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                {getSeverityIcon(severity)}
                <Chip
                  label={severity.toUpperCase()}
                  size="small"
                  sx={{
                    background: getSeverityColor(severity),
                    color: '#fff',
                    fontWeight: 600,
                  }}
                />
              </Box>
              
              <Typography variant="caption" color="text.secondary">
                Started: {new Date(started_at).toLocaleString()}
              </Typography>
            </Box>
          </Grid>
          
          <Divider orientation="vertical" flexItem />
          
          {/* Middle Column — Risk Assessment */}
          <Grid item xs={12} md={3}>
            <Box>
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                RISK SCORE
              </Typography>
              
              <Box sx={{ mb: 3 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="h4" sx={{ fontWeight: 700, color: getRiskColor(risk_level) }}>
                    {(risk_score * 100).toFixed(0)}
                  </Typography>
                  <Chip
                    label={risk_level.toUpperCase()}
                    size="small"
                    sx={{
                      background: getRiskColor(risk_level),
                      color: '#fff',
                      fontWeight: 600,
                    }}
                  />
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={risk_score * 100}
                  sx={{
                    height: 8,
                    borderRadius: 4,
                    background: '#0a0e27',
                    '& .MuiLinearProgress-bar': {
                      background: getRiskColor(risk_level),
                    },
                  }}
                />
              </Box>
              
              {scenario_match && (
                <>
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                    TOP MATCHED SCENARIO
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    <code style={{ 
                      background: '#0a0e27', 
                      padding: '4px 8px', 
                      borderRadius: 4,
                      fontSize: '0.85rem',
                    }}>
                      {scenario_match}
                    </code>
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Confidence: {((scenario_confidence || 0) * 100).toFixed(1)}%
                  </Typography>
                </>
              )}
            </Box>
          </Grid>
          
          <Divider orientation="vertical" flexItem />
          
          {/* Right Column — Autonomy Decision */}
          <Grid item xs={12} md={4}>
            <Box>
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                AUTONOMY DECISION
              </Typography>
              
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 3 }}>
                {getAutonomyIcon(autonomy_decision)}
                <Chip
                  label={autonomy_decision.replace('_', ' ')}
                  size="medium"
                  sx={{
                    background: getAutonomyColor(autonomy_decision),
                    color: '#fff',
                    fontWeight: 700,
                    fontSize: '0.85rem',
                    px: 1,
                  }}
                />
              </Box>
              
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                TRUST SCORE PROGRESS
              </Typography>
              
              <Box sx={{ 
                background: '#0a0e27', 
                p: 2, 
                borderRadius: 2,
                border: '1px solid rgba(255, 255, 255, 0.1)',
              }}>
                <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                  {trust_progress}
                </Typography>

                {trust_metrics && (
                  <Box sx={{ mt: 2, display: 'grid', gap: 1.25 }}>
                    <Typography variant="body2">
                      {trust_metrics.approvals_recorded} approvals on record
                    </Typography>
                    <Typography variant="body2">
                      {(trust_metrics.success_rate * 100).toFixed(0)}% resolved successfully
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {trust_metrics.path_to_next_tier}
                    </Typography>
                  </Box>
                )}
                
                {autonomy_decision !== 'AUTONOMOUS' && (
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 2 }}>
                    More successful approvals are required before the service can graduate to autonomous execution
                  </Typography>
                )}
              </Box>
            </Box>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );
};

export default IncidentDashboard;
