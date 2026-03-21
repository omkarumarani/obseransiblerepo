import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import React from 'react';
import SessionRiskSparkline from './SessionRiskSparkline';

// Recharts uses SVG + ResizeObserver which are not in jsdom — mock the whole lib.
vi.mock('recharts', () => ({
  AreaChart: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="area-chart">{children}</div>
  ),
  Area: () => null,
  XAxis: () => null,
  YAxis: () => null,
  CartesianGrid: () => null,
  Tooltip: () => null,
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="responsive-container">{children}</div>
  ),
}));

// Mock the API so tests don't make real HTTP calls.
vi.mock('../../api', () => ({
  getRiskHistory: vi.fn().mockResolvedValue([]),
}));

describe('SessionRiskSparkline', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders without crashing', () => {
    render(<SessionRiskSparkline riskScore={0.75} />);
    expect(screen.getByText('Risk Trend')).toBeInTheDocument();
  });

  it('renders the recharts container', () => {
    render(<SessionRiskSparkline riskScore={0.5} />);
    expect(screen.getByTestId('responsive-container')).toBeInTheDocument();
  });

  it('shows the incident started label', () => {
    const oneMinuteAgo = new Date(Date.now() - 60_000).toISOString();
    render(<SessionRiskSparkline riskScore={0.5} startedAtIso={oneMinuteAgo} />);
    expect(screen.getByText(/Incident started/i)).toBeInTheDocument();
  });

  it('shows time-ago text in the incident label', () => {
    const twoMinutesAgo = new Date(Date.now() - 2 * 60_000).toISOString();
    render(<SessionRiskSparkline riskScore={0.5} startedAtIso={twoMinutesAgo} />);
    // date-fns formatDistanceToNow produces something like "2 minutes ago"
    expect(screen.getByText(/minute/i)).toBeInTheDocument();
  });

  it('shows synthetic indicator when no sessionId provided', () => {
    render(<SessionRiskSparkline riskScore={0.3} />);
    expect(screen.getByText(/synthetic/i)).toBeInTheDocument();
  });

  it('calls getRiskHistory when sessionId is provided', async () => {
    const { getRiskHistory } = await import('../../api');
    render(<SessionRiskSparkline riskScore={0.8} sessionId="test-session-123" />);
    // The useEffect fires asynchronously; just verify the call was made.
    expect(getRiskHistory).toHaveBeenCalledWith('test-session-123');
  });
});
