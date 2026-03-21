// ═══════════════════════════════════════════════════════════════════════════════
// Agent Details Drawer — Side panel showing full agent output
// ═══════════════════════════════════════════════════════════════════════════════

import React from 'react';
import {
  Drawer,
  Box,
  Typography,
  IconButton,
  Divider,
  Chip,
  LinearProgress,
} from '@mui/material';
import { Close, Psychology, VerifiedUser } from '@mui/icons-material';
import { AgentStep } from '../types';

interface AgentDetailsDrawerProps {
  open: boolean;
  agent: AgentStep | null;
  onClose: () => void;
}

const AgentDetailsDrawer: React.FC<AgentDetailsDrawerProps> = ({ open, agent, onClose }) => {
  if (!agent) return null;

  const getStatusColor = (status: string): string => {
    switch (status) {
      case 'success':
        return '#4caf50';
      case 'failed':
        return '#f44336';
      case 'running':
        return '#2196f3';
      case 'skipped':
        return '#ff9800';
      default:
        return '#757575';
    }
  };

  const getValidationColor = (status: string): string => {
    switch (status) {
      case 'corroborated':    return '#4caf50';
      case 'weak_support':    return '#ff9800';
      case 'divergent':       return '#f44336';
      case 'insufficient_context': return '#9c27b0';
      default:                return '#757575';
    }
  };

  // Extract Block F local validation info from agent 4 (Analyze step) output
  const analysisOutput = (agent.id === 4 && agent.output && typeof agent.output === 'object')
    ? (agent.output as Record<string, any>)
    : null;
  const localValidationStatus: string | undefined = analysisOutput?.local_validation_status;
  const localValidationCompleted: boolean = analysisOutput?.local_validation_completed ?? false;
  const showLocalValidation = localValidationCompleted && !!localValidationStatus;

  return (
    <Drawer
      anchor="right"
      open={open}
      onClose={onClose}
      PaperProps={{
        sx: {
          width: { xs: '100%', sm: 600 },
          background: '#1a1f3a',
          color: '#e1e4e8',
        },
      }}
    >
      <Box sx={{ p: 3 }}>
        {/* Header */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Box>
            <Typography variant="overline" color="text.secondary">
              Agent {agent.id}
            </Typography>
            <Typography variant="h5" sx={{ fontWeight: 600 }}>
              {agent.name}
            </Typography>
          </Box>
          <IconButton onClick={onClose} sx={{ color: '#e1e4e8' }}>
            <Close />
          </IconButton>
        </Box>

        <Divider sx={{ mb: 3, borderColor: 'rgba(255, 255, 255, 0.1)' }} />

        {/* Status & Duration */}
        <Box sx={{ mb: 3 }}>
          <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
            <Box>
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                STATUS
              </Typography>
              <Chip
                label={agent.status.toUpperCase()}
                sx={{
                  background: getStatusColor(agent.status),
                  color: '#fff',
                  fontWeight: 600,
                }}
              />
            </Box>
            
            {agent.duration > 0 && (
              <Box>
                <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                  DURATION
                </Typography>
                <Typography variant="h6">
                  {agent.duration.toFixed(2)}s
                </Typography>
              </Box>
            )}
          </Box>

          {agent.startedAt && (
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
              Started: {new Date(agent.startedAt).toLocaleString()}
            </Typography>
          )}
          
          {agent.completedAt && (
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
              Completed: {new Date(agent.completedAt).toLocaleString()}
            </Typography>
          )}
        </Box>

        <Divider sx={{ mb: 3, borderColor: 'rgba(255, 255, 255, 0.1)' }} />

        {/* Local LLM Validation — Block F (only shown for Analyze step) */}
        {showLocalValidation && analysisOutput && (
          <>
            <Box sx={{ mb: 3 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                <Psychology sx={{ color: '#7c4dff', fontSize: 18 }} />
                <Typography variant="caption" color="text.secondary" sx={{ textTransform: 'uppercase', letterSpacing: 1 }}>
                  Local LLM Validation
                </Typography>
              </Box>

              {/* Validation verdict badge */}
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2, flexWrap: 'wrap' }}>
                <Chip
                  icon={<VerifiedUser sx={{ fontSize: 16 }} />}
                  label={localValidationStatus.replace(/_/g, ' ').toUpperCase()}
                  sx={{
                    background: getValidationColor(localValidationStatus),
                    color: '#fff',
                    fontWeight: 700,
                    fontSize: '0.78rem',
                  }}
                />
                {analysisOutput.local_model && (
                  <Typography variant="caption" color="text.secondary">
                    via {analysisOutput.local_model}
                  </Typography>
                )}
              </Box>

              {/* Confidence */}
              {typeof analysisOutput.local_validation_confidence === 'number' && (
                <Box sx={{ mb: 2 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                    <Typography variant="caption" color="text.secondary">Local confidence</Typography>
                    <Typography variant="caption" sx={{ fontWeight: 600 }}>
                      {(analysisOutput.local_validation_confidence * 100).toFixed(0)}%
                    </Typography>
                  </Box>
                  <LinearProgress
                    variant="determinate"
                    value={analysisOutput.local_validation_confidence * 100}
                    sx={{
                      height: 6, borderRadius: 3,
                      background: 'rgba(255,255,255,0.1)',
                      '& .MuiLinearProgress-bar': {
                        background: getValidationColor(localValidationStatus),
                        borderRadius: 3,
                      },
                    }}
                  />
                </Box>
              )}

              {/* Knowledge similarity + supporting incidents row */}
              <Box sx={{ display: 'flex', gap: 3, mb: 2 }}>
                {typeof analysisOutput.knowledge_top_similarity === 'number' && (
                  <Box>
                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>Top similarity</Typography>
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>
                      {(analysisOutput.knowledge_top_similarity * 100).toFixed(1)}%
                    </Typography>
                  </Box>
                )}
                {typeof analysisOutput.local_similar_count === 'number' && (
                  <Box>
                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>Similar incidents</Typography>
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>
                      {analysisOutput.local_similar_count}
                    </Typography>
                  </Box>
                )}
                {analysisOutput.validation_mode && (
                  <Box>
                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>Mode</Typography>
                    <Typography variant="body2" sx={{ fontWeight: 600, fontSize: '0.8rem' }}>
                      {analysisOutput.validation_mode.replace(/_/g, ' ')}
                    </Typography>
                  </Box>
                )}
              </Box>

              {/* Reasoning summary */}
              {analysisOutput.local_validation_reason && (
                <Box
                  sx={{
                    background: 'rgba(124, 77, 255, 0.08)',
                    border: '1px solid rgba(124, 77, 255, 0.3)',
                    borderRadius: 2,
                    p: 1.5,
                  }}
                >
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>Reasoning</Typography>
                  <Typography variant="body2" sx={{ fontSize: '0.82rem', lineHeight: 1.5, color: '#c5b8ff' }}>
                    {analysisOutput.local_validation_reason}
                  </Typography>
                </Box>
              )}
            </Box>
            <Divider sx={{ mb: 3, borderColor: 'rgba(255, 255, 255, 0.1)' }} />
          </>
        )}

        {/* Output */}
        <Box>
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 2 }}>
            OUTPUT
          </Typography>
          
          <Box
            sx={{
              background: '#0a0e27',
              p: 2,
              borderRadius: 2,
              border: '1px solid rgba(255, 255, 255, 0.1)',
              maxHeight: 'calc(100vh - 400px)',
              overflow: 'auto',
            }}
          >
            {agent.output ? (
              <pre
                style={{
                  margin: 0,
                  fontFamily: 'monospace',
                  fontSize: '0.85rem',
                  lineHeight: 1.6,
                  color: '#e1e4e8',
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word',
                }}
              >
                {typeof agent.output === 'string'
                  ? agent.output
                  : JSON.stringify(agent.output, null, 2)}
              </pre>
            ) : (
              <Typography variant="body2" color="text.secondary" sx={{ fontStyle: 'italic' }}>
                No output available
              </Typography>
            )}
          </Box>
        </Box>

        {/* Error (if any) */}
        {agent.error && (
          <>
            <Divider sx={{ my: 3, borderColor: 'rgba(255, 255, 255, 0.1)' }} />
            <Box>
              <Typography variant="caption" sx={{ display: 'block', mb: 2, color: '#f44336' }}>
                ERROR
              </Typography>
              <Box
                sx={{
                  background: 'rgba(244, 67, 54, 0.1)',
                  border: '1px solid #f44336',
                  p: 2,
                  borderRadius: 2,
                }}
              >
                <Typography variant="body2" sx={{ fontFamily: 'monospace', color: '#f44336' }}>
                  {agent.error}
                </Typography>
              </Box>
            </Box>
          </>
        )}
      </Box>
    </Drawer>
  );
};

export default AgentDetailsDrawer;
