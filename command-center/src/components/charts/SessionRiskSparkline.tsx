import React, { useMemo, useState, useEffect } from 'react';
import { formatDistanceToNow } from 'date-fns';
import { Box, CircularProgress, Typography } from '@mui/material';
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { getRiskHistory, RiskPoint } from '../../api';

interface SessionRiskSparklineProps {
  riskScore: number;
  startedAtIso?: string;
  /** When provided, fetches real time-series from Prometheus via ui-backend */
  sessionId?: string;
}

const SessionRiskSparkline: React.FC<SessionRiskSparklineProps> = ({
  riskScore,
  startedAtIso,
  sessionId,
}) => {
  const [liveData, setLiveData] = useState<RiskPoint[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [dataSource, setDataSource] = useState<'prometheus' | 'synthetic'>('synthetic');

  useEffect(() => {
    if (!sessionId) return;
    let cancelled = false;
    setLoading(true);
    getRiskHistory(sessionId)
      .then((series) => {
        if (!cancelled && series.length > 0) {
          setLiveData(series);
          setDataSource('prometheus');
        }
      })
      .catch(() => { /* falls through to synthetic */ })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [sessionId]);

  const syntheticData = useMemo<RiskPoint[]>(() => {
    const points = 15;
    const base = Math.max(0, Math.min(1, riskScore || 0));
    return Array.from({ length: points }, (_, i) => {
      const drift = (i - points / 2) * 0.012;
      const jitter = (i % 2 === 0 ? 1 : -1) * 0.02;
      const value = Math.max(0, Math.min(1, base + drift + jitter));
      return { t: i + 1, risk: Number((value * 100).toFixed(1)) };
    });
  }, [riskScore]);

  const data = liveData ?? syntheticData;

  const startedLabel = startedAtIso
    ? formatDistanceToNow(new Date(startedAtIso), { addSuffix: true })
    : 'unknown';

  return (
    <Box className="rounded-lg border border-cyan-500/20 bg-[#101736] p-3">
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
        <Typography variant="subtitle2" sx={{ color: '#90caf9', fontWeight: 700 }}>
          Risk Trend
        </Typography>
        {loading ? (
          <CircularProgress size={12} sx={{ color: '#29b6f6' }} />
        ) : (
          <Typography variant="caption" sx={{ color: '#8b949e' }}>
            {dataSource === 'prometheus' ? '🟢 live' : '⚪ synthetic'}
          </Typography>
        )}
      </Box>
      <Box sx={{ width: '100%', height: 130 }}>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <defs>
              <linearGradient id="riskFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#29b6f6" stopOpacity={0.8} />
                <stop offset="95%" stopColor="#29b6f6" stopOpacity={0.05} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#2a335f" />
            <XAxis dataKey="t" tick={{ fill: '#8b949e', fontSize: 11 }} />
            <YAxis domain={[0, 100]} tick={{ fill: '#8b949e', fontSize: 11 }} />
            <Tooltip
              contentStyle={{ backgroundColor: '#1a1f3a', border: '1px solid #3f4d86' }}
              labelStyle={{ color: '#c7d2fe' }}
            />
            <Area type="monotone" dataKey="risk" stroke="#29b6f6" fill="url(#riskFill)" />
          </AreaChart>
        </ResponsiveContainer>
      </Box>
      <Typography variant="caption" color="text.secondary">
        Incident started {startedLabel}
      </Typography>
    </Box>
  );
};

export default SessionRiskSparkline;

