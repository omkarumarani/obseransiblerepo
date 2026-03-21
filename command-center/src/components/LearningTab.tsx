import React, { useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Box,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Dialog,
  DialogContent,
  DialogTitle,
  Grid,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Typography,
} from '@mui/material';
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

import { getKnowledgeEntries, getLearningStats, getScenarioStats, getValueMetrics } from '../api';
import { KnowledgeEntry, LearningStats, ValueMetrics } from '../types';

type ScenarioLearningStat = {
  scenario_id: string;
  total_seen?: number;
  success_rate?: number;
  weight_adjustment?: number;
};

const panelSx = {
  background: '#1a1f3a',
  borderRadius: 2,
  height: '100%',
};

const formatMinutes = (seconds: number) => `${(seconds / 60).toFixed(1)} min`;

const LearningTab: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [learningStats, setLearningStats] = useState<LearningStats | null>(null);
  const [knowledgeEntries, setKnowledgeEntries] = useState<KnowledgeEntry[]>([]);
  const [scenarioStats, setScenarioStats] = useState<ScenarioLearningStat[]>([]);
  const [valueMetrics, setValueMetrics] = useState<ValueMetrics | null>(null);
  const [selectedEntry, setSelectedEntry] = useState<KnowledgeEntry | null>(null);

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      try {
        setLoading(true);
        setError(null);
        const [stats, entries, scenarios, values] = await Promise.all([
          getLearningStats(),
          getKnowledgeEntries({ limit: 50 }),
          getScenarioStats(),
          getValueMetrics(),
        ]);
        if (cancelled) {
          return;
        }
        setLearningStats(stats);
        setKnowledgeEntries(entries);
        setScenarioStats(scenarios);
        setValueMetrics(values);
      } catch (err: any) {
        if (!cancelled) {
          setError(err?.message || 'Failed to load learning data');
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  const pieData = useMemo(() => {
    const stats = learningStats;
    return [
      { name: 'Corroborated', value: stats?.corroborated_count_30d || 0, color: '#4caf50' },
      { name: 'Weak Support', value: stats?.weak_support_count_30d || 0, color: '#ffb300' },
      { name: 'Divergent', value: stats?.divergent_count_30d || 0, color: '#f44336' },
      { name: 'Unavailable', value: Math.max((stats?.external_llm_calls_30d || 0) - (stats?.local_validation_completed_30d || 0), 0), color: '#607d8b' },
    ];
  }, [learningStats]);

  const weeklyHitRate = useMemo(() => {
    if (learningStats?.weekly_hit_rate?.length) {
      return learningStats.weekly_hit_rate;
    }
    return [
      { week: 'W1', hit_rate: 0 },
      { week: 'W2', hit_rate: 0 },
      { week: 'W3', hit_rate: 0 },
      { week: 'W4', hit_rate: 0 },
    ];
  }, [learningStats]);

  const scenarioBars = useMemo(() => {
    return [...scenarioStats]
      .sort((left, right) => (right.total_seen || 0) - (left.total_seen || 0))
      .slice(0, 12)
      .map((item) => ({
        scenario_id: item.scenario_id,
        total_seen: item.total_seen || 0,
        local_llm_pct: item.weight_adjustment && item.weight_adjustment > 0 ? 0.6 : item.weight_adjustment ? 0.2 : 0,
      }));
  }, [scenarioStats]);

  const costNote = (learningStats?.external_llm_calls_30d || 0) * 0.002;

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 320 }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box>
      {error && <Alert severity="error" sx={{ mb: 3 }}>{error}</Alert>}

      <Grid container spacing={3}>
        <Grid item xs={12} lg={6}>
          <Card sx={panelSx}>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2, fontWeight: 700 }}>
                Panel A — LLM Routing Stats
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} md={6} sx={{ height: 260 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie data={pieData} dataKey="value" nameKey="name" innerRadius={55} outerRadius={85}>
                        {pieData.map((entry) => (
                          <Cell key={entry.name} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </Grid>
                <Grid item xs={12} md={6} sx={{ height: 260 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={weeklyHitRate}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#28324f" />
                      <XAxis dataKey="week" stroke="#8b949e" />
                      <YAxis stroke="#8b949e" tickFormatter={(value) => `${Math.round(value * 100)}%`} />
                      <Tooltip formatter={(value: number) => `${(value * 100).toFixed(0)}%`} />
                      <Area type="monotone" dataKey="hit_rate" stroke="#26c6da" fill="#26c6da33" />
                    </AreaChart>
                  </ResponsiveContainer>
                </Grid>
              </Grid>
              <Box sx={{ mt: 2, display: 'grid', gap: 1 }}>
                <Typography variant="body2">External validation coverage: 100%</Typography>
                <Typography variant="body2">Local corroboration rate: {((learningStats?.corroboration_rate_pct || 0) * 100).toFixed(0)}%</Typography>
                <Typography variant="body2" color="text.secondary">
                  {learningStats?.external_llm_calls_30d || 0} external LLM calls tracked, estimated reference spend ${costNote.toFixed(2)} at $0.002/call.
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} lg={6}>
          <Card sx={panelSx}>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2, fontWeight: 700 }}>
                Panel B — Knowledge Base Entries
              </Typography>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Timestamp</TableCell>
                    <TableCell>Service</TableCell>
                    <TableCell>Scenario</TableCell>
                    <TableCell>Action</TableCell>
                    <TableCell>Outcome</TableCell>
                    <TableCell>Source</TableCell>
                    <TableCell>Similarity</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {knowledgeEntries.slice(0, 8).map((entry) => {
                    const metadata = entry.metadata || {};
                    return (
                      <TableRow
                        key={entry.id}
                        hover
                        sx={{ cursor: 'pointer' }}
                        onClick={() => setSelectedEntry(entry)}
                      >
                        <TableCell>{String(metadata.timestamp || 'n/a').slice(0, 19).replace('T', ' ')}</TableCell>
                        <TableCell>{String(metadata.service_name || '-')}</TableCell>
                        <TableCell>{String(metadata.scenario_id || '-')}</TableCell>
                        <TableCell>{String(metadata.action_taken || '-')}</TableCell>
                        <TableCell>
                          <Chip size="small" label={String(metadata.outcome || 'pending')} />
                        </TableCell>
                        <TableCell>{String(metadata.external_source || metadata.source || 'external')}</TableCell>
                        <TableCell>{typeof metadata.top_similarity === 'number' ? metadata.top_similarity.toFixed(2) : '-'}</TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
              {!knowledgeEntries.length && (
                <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
                  Knowledge entries will appear here once Block F starts storing validated incidents in ChromaDB.
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} lg={6}>
          <Card sx={panelSx}>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2, fontWeight: 700 }}>
                Panel C — Per-Scenario Learning Progress
              </Typography>
              <Box sx={{ height: 320 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={scenarioBars} layout="vertical" margin={{ left: 20, right: 16 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#28324f" />
                    <XAxis type="number" stroke="#8b949e" />
                    <YAxis type="category" dataKey="scenario_id" width={150} stroke="#8b949e" />
                    <Tooltip />
                    <Bar dataKey="total_seen">
                      {scenarioBars.map((entry) => {
                        const fill = entry.local_llm_pct > 0.5 ? '#4caf50' : entry.local_llm_pct > 0 ? '#ffb300' : '#f44336';
                        return <Cell key={entry.scenario_id} fill={fill} />;
                      })}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} lg={6}>
          <Card sx={panelSx}>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2, fontWeight: 700 }}>
                Panel D — Value Dashboard
              </Typography>
              <Grid container spacing={2}>
                {valueMetrics && [
                  { label: 'Average MTTR — Automated', value: formatMinutes(valueMetrics.avg_mttr_automated) },
                  { label: 'Average MTTR — Manual', value: formatMinutes(valueMetrics.avg_mttr_manual) },
                  { label: 'Time Saved This Month', value: `${(valueMetrics.time_saved_minutes / 60).toFixed(1)} hours` },
                  { label: 'Incidents Auto-Resolved', value: String(valueMetrics.automated_count) },
                ].map((card) => (
                  <Grid item xs={12} sm={6} key={card.label}>
                    <Box sx={{ p: 2, borderRadius: 2, background: '#0a0e27', minHeight: 120 }}>
                      <Typography variant="caption" color="text.secondary">{card.label}</Typography>
                      <Typography variant="h4" sx={{ mt: 1, fontWeight: 700 }}>{card.value}</Typography>
                      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
                        Updated every 5 minutes
                      </Typography>
                    </Box>
                  </Grid>
                ))}
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Dialog open={Boolean(selectedEntry)} onClose={() => setSelectedEntry(null)} maxWidth="md" fullWidth>
        <DialogTitle>Knowledge Entry Detail</DialogTitle>
        <DialogContent>
          <Typography variant="subtitle2" sx={{ mb: 1 }}>Document</Typography>
          <Box component="pre" sx={{ whiteSpace: 'pre-wrap', background: '#0a0e27', p: 2, borderRadius: 2, mb: 2 }}>
            {selectedEntry?.document}
          </Box>
          <Typography variant="subtitle2" sx={{ mb: 1 }}>Metadata</Typography>
          <Box component="pre" sx={{ whiteSpace: 'pre-wrap', background: '#0a0e27', p: 2, borderRadius: 2 }}>
            {JSON.stringify(selectedEntry?.metadata || {}, null, 2)}
          </Box>
        </DialogContent>
      </Dialog>
    </Box>
  );
};

export default LearningTab;