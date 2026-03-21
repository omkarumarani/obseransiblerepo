// ═══════════════════════════════════════════════════════════════════════════════
// Playback Mode — Timeline Replay of Historical Pipeline Runs
// ═══════════════════════════════════════════════════════════════════════════════

import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Typography,
  Paper,
  IconButton,
  Slider,
  Select,
  MenuItem,
  FormControl,
  Chip,
  Button,
} from '@mui/material';
import {
  PlayArrow,
  Pause,
  SkipNext,
  SkipPrevious,
  Close,
} from '@mui/icons-material';
import { HistoricalPipelineRun, PlaybackState, AgentSnapshot, AgentStatus } from '../types';
import PipelineFlowView from './PipelineFlowView';
import NarrationPanel from './NarrationPanel';
import DecisionReplay from './DecisionReplay';

interface PlaybackModeProps {
  run: HistoricalPipelineRun;
  onClose: () => void;
}

const PlaybackMode: React.FC<PlaybackModeProps> = ({ run, onClose }) => {
  const [playbackState, setPlaybackState] = useState<PlaybackState>({
    isPlaying: false,
    currentTime: 0,
    totalDuration: run.duration * 1000, // Convert to milliseconds
    playbackSpeed: 1,
  });

  const [currentAgents, setCurrentAgents] = useState(run.agents);
  const [selectedAgentIndex, setSelectedAgentIndex] = useState<number | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const lastUpdateRef = useRef<number>(Date.now());

  // Calculate which agents should be active at current time
  const updateAgentsForTime = (timeMs: number) => {
    const updatedAgents = run.agents.map((agent) => {
      const agentEndTime = agent.timestamp + (agent.duration * 1000);
      
      let status: AgentStatus = 'idle';
      
      if (timeMs < agent.timestamp) {
        status = 'idle';
      } else if (timeMs >= agent.timestamp && timeMs < agentEndTime) {
        status = 'running';
      } else if (timeMs >= agentEndTime) {
        status = agent.status; // Use the final status from the historical run
      }
      
      return {
        ...agent,
        status,
      };
    });
    
    setCurrentAgents(updatedAgents);
    
    // Auto-select agent that just started
    const activeAgentIndex = updatedAgents.findIndex(a => a.status === 'running');
    if (activeAgentIndex !== -1 && activeAgentIndex !== selectedAgentIndex) {
      setSelectedAgentIndex(activeAgentIndex);
    }
  };

  // Playback loop
  useEffect(() => {
    if (!playbackState.isPlaying) {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      return;
    }

    const animate = () => {
      const now = Date.now();
      const deltaTime = now - lastUpdateRef.current;
      lastUpdateRef.current = now;

      setPlaybackState((prev) => {
        const newTime = Math.min(
          prev.currentTime + deltaTime * prev.playbackSpeed,
          prev.totalDuration
        );

        // Auto-pause at end
        if (newTime >= prev.totalDuration) {
          return { ...prev, currentTime: prev.totalDuration, isPlaying: false };
        }

        updateAgentsForTime(newTime);
        return { ...prev, currentTime: newTime };
      });

      animationFrameRef.current = requestAnimationFrame(animate);
    };

    animationFrameRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [playbackState.isPlaying, playbackState.playbackSpeed]);

  // Update agents when scrubber moves
  useEffect(() => {
    updateAgentsForTime(playbackState.currentTime);
  }, [playbackState.currentTime]);

  // Playback controls
  const handlePlayPause = () => {
    lastUpdateRef.current = Date.now();
    setPlaybackState((prev) => ({ ...prev, isPlaying: !prev.isPlaying }));
  };

  const handleStepForward = () => {
    const nextAgentIndex = currentAgents.findIndex((a, idx) => 
      idx > (selectedAgentIndex || -1) && a.timestamp > playbackState.currentTime
    );
    
    if (nextAgentIndex !== -1) {
      const nextTime = currentAgents[nextAgentIndex].timestamp;
      setPlaybackState((prev) => ({ ...prev, currentTime: nextTime, isPlaying: false }));
      setSelectedAgentIndex(nextAgentIndex);
    }
  };

  const handleStepBackward = () => {
    const prevAgentIndex = [...currentAgents].reverse().findIndex((a, idx) => {
      const actualIndex = currentAgents.length - 1 - idx;
      return actualIndex < (selectedAgentIndex || currentAgents.length) && a.timestamp < playbackState.currentTime - 100;
    });
    
    if (prevAgentIndex !== -1) {
      const actualIndex = currentAgents.length - 1 - prevAgentIndex;
      const prevTime = currentAgents[actualIndex].timestamp;
      setPlaybackState((prev) => ({ ...prev, currentTime: prevTime, isPlaying: false }));
      setSelectedAgentIndex(actualIndex);
    }
  };

  const handleScrubberChange = (_: Event, value: number | number[]) => {
    const newTime = Array.isArray(value) ? value[0] : value;
    setPlaybackState((prev) => ({ ...prev, currentTime: newTime, isPlaying: false }));
  };

  const handleSpeedChange = (speed: PlaybackState['playbackSpeed']) => {
    setPlaybackState((prev) => ({ ...prev, playbackSpeed: speed }));
  };

  // Format time display
  const formatTime = (ms: number) => {
    const totalSeconds = Math.floor(ms / 1000);
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  const getProgressPercent = () => {
    return (playbackState.currentTime / playbackState.totalDuration) * 100;
  };

  return (
    <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column', background: '#0a0e27' }}>
      {/* Header */}
      <Box sx={{ 
        p: 2, 
        background: '#1a1f3a', 
        borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
      }}>
        <Box>
          <Typography variant="h6" sx={{ fontWeight: 700 }}>
            🎬 Playback Mode
          </Typography>
          <Typography variant="caption" color="text.secondary">
            {run.service_name} • {run.alert_name} • {new Date(run.timestamp).toLocaleString()}
          </Typography>
        </Box>
        <IconButton onClick={onClose}>
          <Close />
        </IconButton>
      </Box>

      {/* Main Content */}
      <Box sx={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        {/* Pipeline View (Left) */}
        <Box sx={{ flex: 1, p: 3, overflowY: 'auto' }}>
          <Paper sx={{ p: 3, background: '#1a1f3a', mb: 3 }}>
            <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 600 }}>
              Agent Pipeline
            </Typography>
            <PipelineFlowView 
              agents={currentAgents.map((a, idx) => ({
                id: idx + 1,
                name: a.name,
                status: a.status,
                duration: a.duration,
                output: a.output,
              }))}
              onNodeClick={(agentId) => setSelectedAgentIndex(agentId - 1)}
            />
          </Paper>

          {/* Decision Replay Overlay */}
          {run.agents.some(a => a.findings?.decision_tree_steps) && (
            <DecisionReplay
              decisionSteps={run.agents.find(a => a.findings?.decision_tree_steps)?.findings?.decision_tree_steps || []}
              currentTime={playbackState.currentTime}
              autonomyDecision={run.autonomy_decision}
            />
          )}

          {/* Final Results */}
          <Paper sx={{ p: 2, background: '#1a1f3a', mt: 3 }}>
            <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 600 }}>
              Final Results
            </Typography>
            <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
              <Chip
                label={`Outcome: ${run.outcome}`}
                sx={{
                  background: run.outcome === 'success' ? '#4caf50' : run.outcome === 'failure' ? '#f44336' : '#ff9800',
                  color: '#fff',
                  fontWeight: 600,
                }}
              />
              {run.gitea_pr_url && (
                <Chip
                  label="Gitea PR Created"
                  component="a"
                  href={run.gitea_pr_url}
                  target="_blank"
                  clickable
                  sx={{ background: '#4caf50', color: '#fff' }}
                />
              )}
              {run.ansible_run_result && (
                <Chip
                  label={`Ansible: ${run.ansible_run_result}`}
                  sx={{ background: '#ff9800', color: '#fff' }}
                />
              )}
              {run.ticket_url && (
                <Chip
                  label="View Ticket"
                  component="a"
                  href={run.ticket_url}
                  target="_blank"
                  clickable
                  sx={{ background: '#2196f3', color: '#fff' }}
                />
              )}
            </Box>
          </Paper>
        </Box>

        {/* Narration Panel (Right) */}
        <Box sx={{ width: 400, borderLeft: '1px solid rgba(255, 255, 255, 0.1)', background: '#141829' }}>
          <NarrationPanel
            agent={selectedAgentIndex !== null ? run.agents[selectedAgentIndex] : null}
            currentTime={playbackState.currentTime}
          />
        </Box>
      </Box>

      {/* Playback Controls (Bottom) */}
      <Paper sx={{ 
        p: 2, 
        background: '#1a1f3a', 
        borderTop: '1px solid rgba(255, 255, 255, 0.1)',
      }}>
        {/* Timeline Scrubber */}
        <Box sx={{ mb: 2, px: 2 }}>
          <Slider
            value={playbackState.currentTime}
            min={0}
            max={playbackState.totalDuration}
            onChange={handleScrubberChange}
            sx={{
              color: '#2196f3',
              '& .MuiSlider-thumb': {
                width: 16,
                height: 16,
              },
            }}
          />
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
            <Typography variant="caption" color="text.secondary">
              {formatTime(playbackState.currentTime)}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              {getProgressPercent().toFixed(0)}%
            </Typography>
            <Typography variant="caption" color="text.secondary">
              {formatTime(playbackState.totalDuration)}
            </Typography>
          </Box>
        </Box>

        {/* Control Buttons */}
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: 2 }}>
          <IconButton onClick={handleStepBackward} size="large">
            <SkipPrevious />
          </IconButton>
          
          <IconButton 
            onClick={handlePlayPause} 
            size="large"
            sx={{ 
              background: '#2196f3',
              '&:hover': { background: '#1976d2' },
            }}
          >
            {playbackState.isPlaying ? <Pause /> : <PlayArrow />}
          </IconButton>
          
          <IconButton onClick={handleStepForward} size="large">
            <SkipNext />
          </IconButton>

          {/* Speed Control */}
          <FormControl size="small" sx={{ ml: 4, minWidth: 80 }}>
            <Select
              value={playbackState.playbackSpeed}
              onChange={(e) => handleSpeedChange(e.target.value as PlaybackState['playbackSpeed'])}
              sx={{ color: '#fff' }}
            >
              <MenuItem value={0.5}>0.5x</MenuItem>
              <MenuItem value={1}>1x</MenuItem>
              <MenuItem value={2}>2x</MenuItem>
              <MenuItem value={4}>4x</MenuItem>
            </Select>
          </FormControl>
        </Box>
      </Paper>
    </Box>
  );
};

export default PlaybackMode;
