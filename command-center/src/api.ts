// ═══════════════════════════════════════════════════════════════════════════════
// API Service — Compute Agent REST Client
// ═══════════════════════════════════════════════════════════════════════════════

import axios from 'axios';
import {
  PipelineSession,
  AutonomyStatus,
  HistoricalPipelineRun,
  HistoryFilters,
  Scenario,
  LearningStats,
  KnowledgeEntry,
  ValueMetrics,
} from './types';

// Empty string → relative URLs; Vite dev proxy and nginx both handle routing.
// Override with VITE_API_BASE_URL env var if needed (e.g. cross-origin staging).
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? '';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// ── Pipeline Session Endpoints ─────────────────────────────────────────────────

export const getPipelineSession = async (sessionId: string): Promise<PipelineSession> => {
  const response = await api.get(`/pipeline/session/${sessionId}`);
  return response.data;
};

export const listPipelineSessions = async (): Promise<string[]> => {
  try {
    const response = await api.get('/pipeline/sessions');
    return response.data.sessions || [];
  } catch (error) {
    console.warn('List sessions endpoint not available, using default session');
    return ['default'];
  }
};

// ── Autonomy Endpoints ─────────────────────────────────────────────────────────

export const getAutonomyStatus = async (serviceName: string): Promise<AutonomyStatus> => {
  const response = await api.get(`/autonomy/status/${serviceName}`);
  return response.data;
};

export const getAutonomyTiers = async (): Promise<any> => {
  const response = await api.get('/autonomy/tiers');
  return response.data;
};

export const getAutonomyHistory = async (): Promise<any> => {
  const response = await api.get('/autonomy/history');
  return response.data;
};

// ── Health Check ───────────────────────────────────────────────────────────────

export const healthCheck = async (): Promise<boolean> => {
  try {
    const response = await api.get('/health');
    return response.status === 200;
  } catch (error) {
    return false;
  }
};

// ── Historical Pipeline Data ───────────────────────────────────────────────────

export const getHistoricalRuns = async (filters?: HistoryFilters): Promise<HistoricalPipelineRun[]> => {
  try {
    const params = new URLSearchParams();
    
    if (filters?.domain && filters.domain !== 'all') {
      params.append('domain', filters.domain);
    }
    if (filters?.scenario) {
      params.append('scenario', filters.scenario);
    }
    if (filters?.dateRange) {
      params.append('start_date', filters.dateRange.start);
      params.append('end_date', filters.dateRange.end);
    }
    if (filters?.autonomyDecision && filters.autonomyDecision !== 'all') {
      params.append('autonomy_decision', filters.autonomyDecision);
    }
    if (filters?.outcome && filters.outcome !== 'all') {
      params.append('outcome', filters.outcome);
    }
    if (filters?.searchText) {
      params.append('search', filters.searchText);
    }
    
    const url = `/pipeline/history${params.toString() ? '?' + params.toString() : ''}`;
    const response = await api.get(url);
    return response.data.runs || [];
  } catch (error) {
    console.error('Failed to fetch historical runs:', error);
    return [];
  }
};

export const getHistoricalSession = async (sessionId: string): Promise<HistoricalPipelineRun | null> => {
  try {
    const response = await api.get(`/pipeline/history/${sessionId}`);
    return response.data;
  } catch (error) {
    console.error('Failed to fetch historical session:', error);
    return null;
  }
};

// ── Scenario Knowledge Base ───────────────────────────────────────────────────

export const getAllScenarios = async (): Promise<Scenario[]> => {
  try {
    const response = await api.get('/scenarios');
    return response.data.scenarios || [];
  } catch (error) {
    console.error('Failed to fetch scenarios:', error);
    return [];
  }
};

export const getScenariosByDomain = async (domain: 'compute' | 'storage'): Promise<Scenario[]> => {
  try {
    const response = await api.get(`/scenarios?domain=${domain}`);
    return response.data.scenarios || [];
  } catch (error) {
    console.error('Failed to fetch scenarios:', error);
    return [];
  }
};

export const getScenarioDetail = async (scenarioId: string): Promise<Scenario | null> => {
  try {
    const response = await api.get(`/scenarios/${scenarioId}`);
    return response.data;
  } catch (error) {
    console.error('Failed to fetch scenario detail:', error);
    return null;
  }
};

// ── Risk History (Prometheus-backed sparkline data) ───────────────────────────

export interface RiskPoint {
  t: number;
  risk: number;
}

export const getRiskHistory = async (
  sessionId: string,
  domain?: string,
): Promise<RiskPoint[]> => {
  try {
    const params = domain ? `?domain=${domain}` : '';
    const response = await api.get(`/pipeline/session/${sessionId}/risk-history${params}`);
    return response.data.series || [];
  } catch (error) {
    console.warn('Risk history unavailable, sparkline will use local synthesis');
    return [];
  }
};

export const getScenarioRuns = async (scenarioId: string): Promise<HistoricalPipelineRun[]> => {
  try {
    const response = await api.get(`/scenarios/${scenarioId}/runs`);
    return response.data.runs || [];
  } catch (error) {
    console.error('Failed to fetch scenario runs:', error);
    return [];
  }
};

export const getLearningStats = async (): Promise<LearningStats> => {
  const response = await api.get('/intelligence/learning-stats');
  return response.data;
};

export const getKnowledgeEntries = async (params?: {
  serviceName?: string;
  scenarioId?: string;
  limit?: number;
}): Promise<KnowledgeEntry[]> => {
  const search = new URLSearchParams();
  if (params?.serviceName) {
    search.append('service_name', params.serviceName);
  }
  if (params?.scenarioId) {
    search.append('scenario_id', params.scenarioId);
  }
  search.append('limit', String(params?.limit ?? 50));
  const response = await api.get(`/intelligence/knowledge-entries?${search.toString()}`);
  return response.data.entries || [];
};

export const getScenarioStats = async (): Promise<any[]> => {
  const response = await api.get('/intelligence/scenario-stats');
  return response.data.scenarios || [];
};

export const getValueMetrics = async (): Promise<ValueMetrics> => {
  const response = await api.get('/metrics/value');
  return response.data;
};

export default api;
