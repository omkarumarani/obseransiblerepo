// ═══════════════════════════════════════════════════════════════════════════════
// Scenario Card — Interactive Card for Scenario Knowledge Base
// ═══════════════════════════════════════════════════════════════════════════════

import React from 'react';
import {
  Card,
  CardContent,
  CardActionArea,
  Box,
  Typography,
  Chip,
  Divider,
} from '@mui/material';
import {
  TrendingUp,
  Schedule,
  Visibility,
  CheckCircle,
} from '@mui/icons-material';
import { Scenario, AutonomyDecision } from '../types';

interface ScenarioCardProps {
  scenario: Scenario;
  onClick: (scenario: Scenario) => void;
}

const ScenarioCard: React.FC<ScenarioCardProps> = ({ scenario, onClick }) => {
  const getAutonomyColor = (decision: AutonomyDecision) => {
    switch (decision) {
      case 'AUTONOMOUS': return '#4caf50';
      case 'APPROVAL_GATED': return '#ff9800';
      case 'HUMAN_ONLY': return '#f44336';
      default: return '#757575';
    }
  };

  const formatTimeAgo = (timestamp?: string) => {
    if (!timestamp) return 'Never';
    
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);
    
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    return `${diffDays}d ago`;
  };

  const formatDuration = (seconds?: number) => {
    if (!seconds) return 'N/A';
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    const minutes = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${minutes}m ${secs.toFixed(0)}s`;
  };

  const getSuccessRateColor = (rate?: number) => {
    if (!rate) return '#757575';
    if (rate >= 90) return '#4caf50';
    if (rate >= 70) return '#ff9800';
    return '#f44336';
  };

  return (
    <Card sx={{ 
      height: '100%',
      background: '#1a1f3a',
      border: '1px solid rgba(255, 255, 255, 0.1)',
      transition: 'all 0.3s ease',
      '&:hover': {
        transform: 'translateY(-4px)',
        boxShadow: '0 8px 24px rgba(33, 150, 243, 0.3)',
        borderColor: '#2196f3',
      },
    }}>
      <CardActionArea onClick={() => onClick(scenario)} sx={{ height: '100%' }}>
        <CardContent>
          {/* Header */}
          <Box sx={{ mb: 2 }}>
            <Typography 
              variant="h6" 
              sx={{ 
                fontWeight: 700, 
                mb: 1,
                fontSize: '1rem',
                lineHeight: 1.3,
              }}
            >
              {scenario.display_name}
            </Typography>
            <Typography 
              variant="caption" 
              sx={{ 
                color: 'text.secondary',
                fontFamily: 'monospace',
                display: 'block',
              }}
            >
              {scenario.scenario_id}
            </Typography>
          </Box>

          {/* Autonomy Badge and Action */}
          <Box sx={{ mb: 2, display: 'flex', flexDirection: 'column', gap: 1 }}>
            <Chip
              label={scenario.autonomy_badge}
              size="small"
              sx={{
                background: getAutonomyColor(scenario.autonomy_badge),
                color: '#fff',
                fontWeight: 600,
                fontSize: '0.7rem',
                width: 'fit-content',
              }}
            />
            <Typography variant="body2" sx={{ fontSize: '0.85rem', color: 'text.secondary' }}>
              <strong>Action:</strong> {scenario.action}
            </Typography>
          </Box>

          <Divider sx={{ my: 2, opacity: 0.2 }} />

          {/* Statistics Grid */}
          <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 1.5 }}>
            {/* Times Seen */}
            <Box>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 0.5 }}>
                <Visibility sx={{ fontSize: 14, color: '#2196f3' }} />
                <Typography variant="caption" color="text.secondary">
                  Seen
                </Typography>
              </Box>
              <Typography variant="body2" sx={{ fontWeight: 700 }}>
                {scenario.times_seen}×
              </Typography>
            </Box>

            {/* Last Seen */}
            <Box>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 0.5 }}>
                <Schedule sx={{ fontSize: 14, color: '#ff9800' }} />
                <Typography variant="caption" color="text.secondary">
                  Last Seen
                </Typography>
              </Box>
              <Typography variant="body2" sx={{ fontWeight: 700 }}>
                {formatTimeAgo(scenario.last_seen)}
              </Typography>
            </Box>

            {/* Avg Resolution Time */}
            <Box>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 0.5 }}>
                <TrendingUp sx={{ fontSize: 14, color: '#4caf50' }} />
                <Typography variant="caption" color="text.secondary">
                  Avg Time
                </Typography>
              </Box>
              <Typography variant="body2" sx={{ fontWeight: 700 }}>
                {formatDuration(scenario.average_resolution_time)}
              </Typography>
            </Box>

            {/* Success Rate */}
            <Box>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 0.5 }}>
                <CheckCircle sx={{ fontSize: 14, color: getSuccessRateColor(scenario.success_rate) }} />
                <Typography variant="caption" color="text.secondary">
                  Success
                </Typography>
              </Box>
              <Typography 
                variant="body2" 
                sx={{ 
                  fontWeight: 700,
                  color: getSuccessRateColor(scenario.success_rate),
                }}
              >
                {scenario.success_rate !== undefined ? `${scenario.success_rate.toFixed(1)}%` : 'N/A'}
              </Typography>
            </Box>
          </Box>

          {/* Confidence Threshold */}
          <Box sx={{ mt: 2, pt: 2, borderTop: '1px solid rgba(255, 255, 255, 0.1)' }}>
            <Typography variant="caption" color="text.secondary">
              Confidence Threshold
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 0.5 }}>
              <Box 
                sx={{ 
                  flex: 1, 
                  height: 6, 
                  background: '#0a0e27', 
                  borderRadius: 1,
                  overflow: 'hidden',
                }}
              >
                <Box 
                  sx={{ 
                    width: `${scenario.confidence_threshold}%`, 
                    height: '100%',
                    background: 'linear-gradient(90deg, #2196f3, #4caf50)',
                  }} 
                />
              </Box>
              <Typography variant="caption" sx={{ fontWeight: 700, minWidth: 35 }}>
                {scenario.confidence_threshold}%
              </Typography>
            </Box>
          </Box>
        </CardContent>
      </CardActionArea>
    </Card>
  );
};

export default ScenarioCard;
