// ═══════════════════════════════════════════════════════════════════════════════
// Decision Replay — Step-by-Step Autonomy Decision Visualization
// ═══════════════════════════════════════════════════════════════════════════════

import React from 'react';
import {
  Box,
  Typography,
  Paper,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Chip,
} from '@mui/material';
import { CheckCircle, Cancel, HelpOutline } from '@mui/icons-material';
import { DecisionStep, AutonomyDecision } from '../types';

interface DecisionReplayProps {
  decisionSteps: DecisionStep[];
  currentTime: number;
  autonomyDecision: AutonomyDecision;
}

const DecisionReplay: React.FC<DecisionReplayProps> = ({ 
  decisionSteps, 
  currentTime,
  autonomyDecision,
}) => {
  if (!decisionSteps || decisionSteps.length === 0) {
    return null;
  }

  // Determine which steps should be visible based on playback time
  // Assume each decision step takes 500ms to display
  const visibleStepsCount = Math.min(
    decisionSteps.length,
    Math.floor(currentTime / 500)
  );

  const getStepIcon = (result: 'PASS' | 'FAIL') => {
    if (result === 'PASS') {
      return <CheckCircle sx={{ color: '#4caf50', fontSize: 20 }} />;
    } else if (result === 'FAIL') {
      return <Cancel sx={{ color: '#f44336', fontSize: 20 }} />;
    }
    return <HelpOutline sx={{ color: '#757575', fontSize: 20 }} />;
  };

  const getStepColor = (result: 'PASS' | 'FAIL') => {
    return result === 'PASS' ? '#4caf50' : '#f44336';
  };

  const getDecisionColor = (decision: AutonomyDecision) => {
    switch (decision) {
      case 'AUTONOMOUS': return '#4caf50';
      case 'APPROVAL_GATED': return '#ff9800';
      case 'HUMAN_ONLY': return '#f44336';
      default: return '#757575';
    }
  };

  return (
    <Paper sx={{ p: 3, background: '#1a1f3a', border: '2px solid #2196f3' }}>
      <Box sx={{ mb: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h6" sx={{ fontWeight: 700 }}>
          🧠 Decision Replay — Agent 6 Reasoning
        </Typography>
        <Chip
          label={autonomyDecision}
          sx={{
            background: getDecisionColor(autonomyDecision),
            color: '#fff',
            fontWeight: 700,
            fontSize: '0.85rem',
          }}
        />
      </Box>

      <Stepper activeStep={visibleStepsCount - 1} orientation="vertical">
        {decisionSteps.slice(0, visibleStepsCount).map((step, index) => (
          <Step key={step.step} completed={true}>
            <StepLabel
              StepIconComponent={() => getStepIcon(step.result)}
              sx={{
                '& .MuiStepLabel-label': {
                  color: '#e1e4e8',
                  fontWeight: 600,
                },
              }}
            >
              Step {step.step}: {step.description}
            </StepLabel>
            <StepContent>
              <Box sx={{ 
                p: 2, 
                background: '#0a0e27', 
                borderRadius: 1,
                borderLeft: `4px solid ${getStepColor(step.result)}`,
              }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                  <Chip
                    label={step.result}
                    size="small"
                    sx={{
                      background: getStepColor(step.result),
                      color: '#fff',
                      fontWeight: 600,
                      fontSize: '0.7rem',
                    }}
                  />
                </Box>
                {step.detail && (
                  <Typography variant="body2" sx={{ fontSize: '0.85rem', lineHeight: 1.6 }}>
                    {step.detail}
                  </Typography>
                )}
              </Box>
            </StepContent>
          </Step>
        ))}
      </Stepper>

      {/* Final Decision Summary (shown when all steps are visible) */}
      {visibleStepsCount === decisionSteps.length && (
        <Box sx={{ 
          mt: 3, 
          p: 2, 
          background: getDecisionColor(autonomyDecision),
          borderRadius: 1,
          border: '2px solid rgba(255, 255, 255, 0.3)',
        }}>
          <Typography variant="subtitle2" sx={{ fontWeight: 700, color: '#fff', mb: 0.5 }}>
            Final Decision
          </Typography>
          <Typography variant="body1" sx={{ fontWeight: 700, fontSize: '1.1rem', color: '#fff' }}>
            {autonomyDecision.replace('_', ' ')}
          </Typography>
          <Typography variant="caption" sx={{ color: 'rgba(255, 255, 255, 0.8)', display: 'block', mt: 1 }}>
            {autonomyDecision === 'AUTONOMOUS' && 
              'This incident will be resolved automatically without human approval.'}
            {autonomyDecision === 'APPROVAL_GATED' && 
              'This incident requires approval before executing remediation.'}
            {autonomyDecision === 'HUMAN_ONLY' && 
              'This incident requires full human intervention.'}
          </Typography>
        </Box>
      )}
    </Paper>
  );
};

export default DecisionReplay;
