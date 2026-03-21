// ═══════════════════════════════════════════════════════════════════════════════
// Narration Panel — Agent Findings Display During Playback
// ═══════════════════════════════════════════════════════════════════════════════

import React from 'react';
import {
  Box,
  Typography,
  Paper,
  Divider,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from '@mui/material';
import { CheckCircle, Cancel, Timeline, Assessment, Description, Build } from '@mui/icons-material';
import { AgentSnapshot } from '../types';

interface NarrationPanelProps {
  agent: AgentSnapshot | null;
  currentTime: number;
}

const NarrationPanel: React.FC<NarrationPanelProps> = ({ agent, currentTime }) => {
  if (!agent) {
    return (
      <Box sx={{ p: 3, height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Typography variant="body2" color="text.secondary">
          Select an agent to view details
        </Typography>
      </Box>
    );
  }

  const findings = agent.findings;
  const agentEndTime = agent.timestamp + (agent.duration * 1000);
  const isActive = currentTime >= agent.timestamp && currentTime < agentEndTime;
  const isComplete = currentTime >= agentEndTime;

  return (
    <Box sx={{ height: '100%', overflowY: 'auto' }}>
      {/* Agent Header */}
      <Box sx={{ 
        p: 2, 
        borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
        position: 'sticky',
        top: 0,
        background: '#141829',
        zIndex: 10,
      }}>
        <Typography variant="h6" sx={{ fontWeight: 700, mb: 1 }}>
          {agent.name}
        </Typography>
        <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
          <Chip
            label={agent.status}
            size="small"
            sx={{
              background: 
                agent.status === 'success' ? '#4caf50' :
                agent.status === 'failed' ? '#f44336' :
                agent.status === 'running' ? '#2196f3' :
                '#757575',
              color: '#fff',
              fontWeight: 600,
              textTransform: 'uppercase',
            }}
          />
          <Typography variant="caption" color="text.secondary">
            {agent.duration.toFixed(1)}s
          </Typography>
        </Box>
      </Box>

      {/* Agent Findings */}
      <Box sx={{ p: 2 }}>
        {/* Log Excerpts */}
        {findings?.log_excerpts && findings.log_excerpts.length > 0 && (
          <Paper sx={{ p: 2, mb: 2, background: '#1a1f3a' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
              <Description fontSize="small" sx={{ color: '#ff9800' }} />
              <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                Log Excerpts
              </Typography>
            </Box>
            <Divider sx={{ mb: 1, opacity: 0.2 }} />
            {findings.log_excerpts.map((log, idx) => (
              <Box 
                key={idx} 
                sx={{ 
                  p: 1, 
                  mb: 1, 
                  background: '#0a0e27', 
                  borderRadius: 1,
                  fontFamily: 'monospace',
                  fontSize: '0.75rem',
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-all',
                }}
              >
                {log}
              </Box>
            ))}
          </Paper>
        )}

        {/* Metric Values */}
        {findings?.metric_values && Object.keys(findings.metric_values).length > 0 && (
          <Paper sx={{ p: 2, mb: 2, background: '#1a1f3a' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
              <Assessment fontSize="small" sx={{ color: '#2196f3' }} />
              <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                Metric Values
              </Typography>
            </Box>
            <Divider sx={{ mb: 1, opacity: 0.2 }} />
            {Object.entries(findings.metric_values).map(([metric, value]) => (
              <Box key={metric} sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                <Typography variant="body2" color="text.secondary" sx={{ fontSize: '0.85rem' }}>
                  {metric}
                </Typography>
                <Typography variant="body2" sx={{ fontWeight: 600, fontFamily: 'monospace', fontSize: '0.85rem' }}>
                  {typeof value === 'number' ? value.toFixed(2) : value}
                </Typography>
              </Box>
            ))}
          </Paper>
        )}

        {/* Scenario Matches */}
        {findings?.scenario_matches && findings.scenario_matches.length > 0 && (
          <Paper sx={{ p: 2, mb: 2, background: '#1a1f3a' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
              <Timeline fontSize="small" sx={{ color: '#4caf50' }} />
              <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                Scenario Matches
              </Typography>
            </Box>
            <Divider sx={{ mb: 1, opacity: 0.2 }} />
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell sx={{ color: 'text.secondary', fontWeight: 600 }}>Scenario</TableCell>
                    <TableCell align="right" sx={{ color: 'text.secondary', fontWeight: 600 }}>Confidence</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {findings.scenario_matches.map((match, idx) => (
                    <TableRow key={idx}>
                      <TableCell sx={{ fontSize: '0.85rem' }}>{match.scenario}</TableCell>
                      <TableCell align="right">
                        <Chip
                          label={`${match.confidence}%`}
                          size="small"
                          sx={{
                            background: match.confidence > 80 ? '#4caf50' : match.confidence > 60 ? '#ff9800' : '#757575',
                            color: '#fff',
                            fontWeight: 600,
                            fontSize: '0.7rem',
                          }}
                        />
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        )}

        {/* Root Cause Analysis */}
        {findings?.rca_text && (
          <Paper sx={{ p: 2, mb: 2, background: '#1a1f3a' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
              <CheckCircle fontSize="small" sx={{ color: '#4caf50' }} />
              <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                Root Cause Analysis
              </Typography>
            </Box>
            <Divider sx={{ mb: 1, opacity: 0.2 }} />
            <Typography variant="body2" sx={{ lineHeight: 1.6, fontSize: '0.85rem' }}>
              {findings.rca_text}
            </Typography>
          </Paper>
        )}

        {/* Ansible Playbook Hint */}
        {findings?.ansible_playbook_hint && (
          <Paper sx={{ p: 2, mb: 2, background: '#1a1f3a' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
              <Build fontSize="small" sx={{ color: '#ff9800' }} />
              <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                Ansible Playbook
              </Typography>
            </Box>
            <Divider sx={{ mb: 1, opacity: 0.2 }} />
            <Typography 
              variant="body2" 
              sx={{ 
                fontFamily: 'monospace', 
                fontSize: '0.85rem',
                background: '#0a0e27',
                p: 1,
                borderRadius: 1,
              }}
            >
              {findings.ansible_playbook_hint}
            </Typography>
          </Paper>
        )}

        {/* Agent Output (fallback if no structured findings) */}
        {!findings && agent.output && (
          <Paper sx={{ p: 2, background: '#1a1f3a' }}>
            <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 600 }}>
              Output
            </Typography>
            <Divider sx={{ mb: 1, opacity: 0.2 }} />
            <Box sx={{ 
              p: 1, 
              background: '#0a0e27', 
              borderRadius: 1,
              fontFamily: 'monospace',
              fontSize: '0.75rem',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
            }}>
              {JSON.stringify(agent.output, null, 2)}
            </Box>
          </Paper>
        )}

        {/* Empty state */}
        {!findings && !agent.output && (
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <Typography variant="body2" color="text.secondary">
              No findings available for this agent
            </Typography>
          </Box>
        )}
      </Box>
    </Box>
  );
};

export default NarrationPanel;
