// ═══════════════════════════════════════════════════════════════════════════════
// Agent Node Component for React Flow
// ═══════════════════════════════════════════════════════════════════════════════

import React from 'react';
import { Handle, Position } from 'reactflow';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Chip,
  IconButton,
} from '@mui/material';
import {
  CheckCircle,
  Error,
  Pending,
  RadioButtonUnchecked,
  SkipNext,
  KeyboardArrowRight,
} from '@mui/icons-material';
import { AgentStatus } from '../types';

interface AgentNodeData {
  agentId: number;
  name: string;
  status: AgentStatus;
  duration: number;
  output?: string;
  onOpenDetails: () => void;
}

interface AgentNodeProps {
  data: AgentNodeData;
}

const getStatusIcon = (status: AgentStatus) => {
  switch (status) {
    case 'success':
      return <CheckCircle sx={{ color: '#4caf50' }} />;
    case 'failed':
      return <Error sx={{ color: '#f44336' }} />;
    case 'running':
      return <Pending sx={{ color: '#2196f3', animation: 'pulse 1.5s infinite' }} />;
    case 'skipped':
      return <SkipNext sx={{ color: '#ff9800' }} />;
    default:
      return <RadioButtonUnchecked sx={{ color: '#757575' }} />;
  }
};

const getStatusColor = (status: AgentStatus): string => {
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

const AgentNode: React.FC<AgentNodeProps> = ({ data }) => {
  const { agentId, name, status, duration, output, onOpenDetails } = data;

  const sourceLabel = React.useMemo(() => {
    if (agentId !== 4 || !output || typeof output !== 'object') {
      return '';
    }
    const provider = String((output as any).provider || (output as any).external_source || '').trim();
    const localStatus = String((output as any).local_validation_status || '').trim();
    if (provider && localStatus && localStatus !== 'unavailable') {
      return `EXT + ${localStatus.replace('_', ' ').toUpperCase()}`;
    }
    if (provider) {
      return provider.toUpperCase();
    }
    return '';
  }, [agentId, output]);

  return (
    <Box sx={{ position: 'relative' }}>
      {agentId > 1 && (
        <Handle
          type="target"
          position={Position.Left}
          style={{
            background: '#555',
            width: 12,
            height: 12,
            border: '2px solid #0a0e27',
          }}
        />
      )}
      
      <Card
        sx={{
          minWidth: 200,
          background: status === 'running' ? 'rgba(33, 150, 243, 0.08)' : '#1a1f3a',
          border: `2px solid ${getStatusColor(status)}`,
          borderRadius: 2,
          cursor: 'pointer',
          transition: 'all 0.3s ease',
          '&:hover': {
            transform: 'translateY(-2px)',
            boxShadow: `0 4px 12px ${getStatusColor(status)}40`,
          },
        }}
        onClick={onOpenDetails}
      >
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
            {getStatusIcon(status)}
            <Typography variant="caption" color="text.secondary">
              Agent {agentId}
            </Typography>
          </Box>
          
          <Typography variant="h6" sx={{ mb: 1, fontSize: '0.95rem', fontWeight: 600 }}>
            {name}
          </Typography>

          {sourceLabel && (
            <Chip
              label={sourceLabel}
              size="small"
              sx={{
                mb: 1,
                background: '#0a0e27',
                color: '#9ad1ff',
                border: '1px solid rgba(154, 209, 255, 0.35)',
                fontSize: '0.68rem',
              }}
            />
          )}
          
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <Chip
              label={status.toUpperCase()}
              size="small"
              sx={{
                background: getStatusColor(status),
                color: '#fff',
                fontWeight: 600,
                fontSize: '0.7rem',
              }}
            />
            
            {duration > 0 && (
              <Typography variant="caption" color="text.secondary">
                {duration.toFixed(1)}s
              </Typography>
            )}
          </Box>
          
          <IconButton
            size="small"
            sx={{
              position: 'absolute',
              right: 4,
              top: 4,
              opacity: 0.6,
              '&:hover': { opacity: 1 },
            }}
          >
            <KeyboardArrowRight fontSize="small" />
          </IconButton>
        </CardContent>
      </Card>
      
      {agentId < 6 && (
        <Handle
          type="source"
          position={Position.Right}
          style={{
            background: '#555',
            width: 12,
            height: 12,
            border: '2px solid #0a0e27',
          }}
        />
      )}
      
      <style>{`
        @keyframes pulse {
          0%, 100% {
            opacity: 1;
          }
          50% {
            opacity: 0.5;
          }
        }
      `}</style>
    </Box>
  );
};

export default AgentNode;
