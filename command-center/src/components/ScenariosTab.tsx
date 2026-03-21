// ═══════════════════════════════════════════════════════════════════════════════
// Scenarios Tab — Scenario Knowledge Map
// ═══════════════════════════════════════════════════════════════════════════════

import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Tabs,
  Tab,
  Grid,
  CircularProgress,
  Alert,
  Paper,
  TextField,
  InputAdornment,
} from '@mui/material';
import { Search } from '@mui/icons-material';
import { Scenario, HistoricalPipelineRun } from '../types';
import { getAllScenarios } from '../api';
import ScenarioCard from './ScenarioCard';
import ScenarioDetailModal from './ScenarioDetailModal';

interface ScenariosTabProps {
  onPlaybackRun: (run: HistoricalPipelineRun) => void;
}

const ScenariosTab: React.FC<ScenariosTabProps> = ({ onPlaybackRun }) => {
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [domainTab, setDomainTab] = useState<'all' | 'compute' | 'storage'>('all');
  const [selectedScenario, setSelectedScenario] = useState<Scenario | null>(null);
  const [searchText, setSearchText] = useState('');

  useEffect(() => {
    fetchScenarios();
  }, []);

  const fetchScenarios = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getAllScenarios();
      setScenarios(data);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch scenarios');
    } finally {
      setLoading(false);
    }
  };

  const handleDomainTabChange = (_: React.SyntheticEvent, newValue: 'all' | 'compute' | 'storage') => {
    setDomainTab(newValue);
  };

  const handleScenarioClick = (scenario: Scenario) => {
    setSelectedScenario(scenario);
  };

  const handleCloseDetail = () => {
    setSelectedScenario(null);
  };

  // Filter scenarios by domain and search text
  const filteredScenarios = scenarios.filter((scenario) => {
    const matchesDomain = domainTab === 'all' || scenario.domain === domainTab;
    const matchesSearch = searchText === '' || 
      scenario.display_name.toLowerCase().includes(searchText.toLowerCase()) ||
      scenario.scenario_id.toLowerCase().includes(searchText.toLowerCase()) ||
      scenario.action.toLowerCase().includes(searchText.toLowerCase());
    
    return matchesDomain && matchesSearch;
  });

  // Count scenarios by domain
  const computeCount = scenarios.filter(s => s.domain === 'compute').length;
  const storageCount = scenarios.filter(s => s.domain === 'storage').length;

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h5" sx={{ fontWeight: 700, mb: 1 }}>
          📚 Scenario Knowledge Map
        </Typography>
        <Typography variant="body2" color="text.secondary">
          All known incident scenarios with statistics and historical data
        </Typography>
      </Box>

      {/* Search Bar */}
      <Paper sx={{ p: 2, mb: 3, background: '#1a1f3a' }}>
        <TextField
          fullWidth
          placeholder="Search scenarios by name, ID, or action..."
          value={searchText}
          onChange={(e) => setSearchText(e.target.value)}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <Search />
              </InputAdornment>
            ),
          }}
          size="small"
        />
      </Paper>

      {/* Domain Tabs */}
      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs 
          value={domainTab} 
          onChange={handleDomainTabChange}
          sx={{
            '& .MuiTab-root': {
              color: 'text.secondary',
              fontWeight: 600,
            },
            '& .Mui-selected': {
              color: '#2196f3 !important',
            },
          }}
        >
          <Tab 
            label={`All (${scenarios.length})`} 
            value="all" 
          />
          <Tab 
            label={`🖥️ Compute (${computeCount})`} 
            value="compute" 
          />
          <Tab 
            label={`💾 Storage (${storageCount})`} 
            value="storage" 
          />
        </Tabs>
      </Box>

      {/* Error display */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {/* Loading spinner */}
      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', py: 8 }}>
          <CircularProgress />
        </Box>
      )}

      {/* Scenarios Grid */}
      {!loading && filteredScenarios.length > 0 && (
        <Grid container spacing={3}>
          {filteredScenarios.map((scenario) => (
            <Grid item xs={12} sm={6} md={4} lg={3} key={scenario.scenario_id}>
              <ScenarioCard 
                scenario={scenario} 
                onClick={handleScenarioClick}
              />
            </Grid>
          ))}
        </Grid>
      )}

      {/* Empty state */}
      {!loading && filteredScenarios.length === 0 && (
        <Box sx={{ textAlign: 'center', py: 8 }}>
          <Typography variant="h6" color="text.secondary">
            No scenarios found
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
            {searchText ? 'Try adjusting your search query' : 'No scenarios available in this domain'}
          </Typography>
        </Box>
      )}

      {/* Scenario Detail Modal */}
      <ScenarioDetailModal
        scenario={selectedScenario}
        onClose={handleCloseDetail}
        onPlaybackRun={(run) => {
          handleCloseDetail();
          onPlaybackRun(run);
        }}
      />
    </Box>
  );
};

export default ScenariosTab;
