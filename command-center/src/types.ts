// ═══════════════════════════════════════════════════════════════════════════════
// Type Definitions for AIOps Command Center
// ═══════════════════════════════════════════════════════════════════════════════

export type AgentStatus = 'idle' | 'running' | 'success' | 'failed' | 'skipped';

export type AutonomyDecision = 'AUTONOMOUS' | 'APPROVAL_GATED' | 'HUMAN_ONLY';

export type RiskLevel = 'low' | 'medium' | 'high' | 'critical' | 'unknown';

export type SeverityLevel = 'info' | 'warning' | 'critical';

export interface AgentStep {
  id: number;
  name: string;
  status: AgentStatus;
  duration: number; // seconds
  startedAt?: string;
  completedAt?: string;
  output?: any;
  error?: string;
}

export interface PipelineSession {
  session_id: string;
  service_name: string;
  alert_name: string;
  severity: SeverityLevel;
  summary: string;
  stage: string;
  ticket_id: string;
  ticket_num: number;
  approval_id: string;
  approval_ticket_num: number;
  risk_score: number;
  risk_level: RiskLevel;
  created_at: number;
  age_seconds: number;
  completed_at?: number | null;
  mttr_seconds?: number;
  outcome?: PipelineOutcome | 'running';
  
  // Detailed agent info (extended from basic session)
  agents?: AgentStep[];
  
  // Raw pipeline data for client-side agent step construction
  logs?: string;
  metrics?: Record<string, unknown>;
  stage_durations?: Record<string, number>;
  
  // Analysis results
  analysis?: {
    root_cause?: string;
    recommended_action?: string;
    confidence?: number;
    scenario_id?: string;
    scenario_confidence?: number;
    provider?: string;
    model?: string;
    knowledge_entry_id?: string;
    local_validation_status?: string;
    local_validation_confidence?: number;
    local_validation_reason?: string;
    local_validation_completed?: boolean;
    knowledge_top_similarity?: number;
    local_model?: string;
    source?: string;
    validation_mode?: string;
    validated_by?: string[];
    local_similar_count?: number;
  };
  
  // Autonomy decision
  autonomy_decision?: AutonomyDecision;
  trust_score?: number;
  trust_progress?: string;
  trust_metrics?: TrustMetrics;
}

export interface NextTierTarget {
  name: string;
  approvals_needed: number;
  success_rate_needed: number;
}

export interface TrustMetrics {
  approvals_recorded: number;
  success_rate: number;
  successful_runs: number;
  executed_runs: number;
  next_tier: NextTierTarget;
  path_to_next_tier: string;
}

export interface IncidentInfo {
  service_name: string;
  alert_name: string;
  severity: SeverityLevel;
  started_at: string;
  risk_score: number;
  risk_level: RiskLevel;
  scenario_match?: string;
  scenario_confidence?: number;
  autonomy_decision: AutonomyDecision;
  trust_progress: string;
  trust_metrics?: TrustMetrics;
}

export interface AutonomyStatus {
  service_name: string;
  tier: string;
  autonomous: boolean;
  trust_score: number;
  total_approvals: number;
  total_successes: number;
  success_rate: number;
  threshold: number;
  needs_approvals: number;
  approvals_recorded?: number;
  next_tier?: NextTierTarget;
}

export interface ValueMetrics {
  avg_mttr_automated: number;
  avg_mttr_manual: number;
  automated_count: number;
  manual_count: number;
  incidents_last_30d: number;
  time_saved_minutes: number;
}

export interface LearningStats {
  external_llm_calls_30d: number;
  local_validation_attempts_30d: number;
  local_validation_completed_30d: number;
  corroborated_count_30d: number;
  weak_support_count_30d: number;
  divergent_count_30d: number;
  insufficient_context_count_30d: number;
  avg_top_similarity_30d: number;
  knowledge_entries_total: number;
  knowledge_entries_with_success_outcome: number;
  local_validation_coverage_pct: number;
  corroboration_rate_pct: number;
  weekly_hit_rate: Array<{ week: string; hit_rate: number }>;
  scenario_count?: number;
}

export interface KnowledgeEntry {
  id: string;
  document: string;
  metadata: Record<string, unknown>;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Historical Pipeline Data for Playback
// ═══════════════════════════════════════════════════════════════════════════════

export type PipelineOutcome = 'success' | 'failure' | 'pending';

export interface HistoricalPipelineRun {
  session_id: string;
  timestamp: string; // ISO timestamp
  service_name: string;
  alert_name: string;
  domain: 'compute' | 'storage' | 'network' | 'database';
  scenario_matched?: string;
  risk_score: number;
  autonomy_decision: AutonomyDecision;
  outcome: PipelineOutcome;
  duration: number; // Total run duration in seconds
  agents: AgentSnapshot[];
  
  // Additional metadata
  gitea_pr_url?: string;
  ansible_run_result?: string;
  ticket_url?: string;
}

export interface AgentSnapshot extends AgentStep {
  // Timestamp when agent started
  timestamp: number; // Relative to run start (0-based, in milliseconds)
  
  // Agent findings/output
  findings?: {
    log_excerpts?: string[];
    metric_values?: Record<string, number>;
    scenario_matches?: ScenarioMatch[];
    rca_text?: string;
    ansible_playbook_hint?: string;
    decision_tree_steps?: DecisionStep[];
  };
}

export interface ScenarioMatch {
  scenario: string;
  confidence: number; // 0-100
}

export interface DecisionStep {
  step: number;
  description: string;
  result: 'PASS' | 'FAIL';
  detail?: string;
}

export interface HistoryFilters {
  domain?: 'compute' | 'storage' | 'network' | 'database' | 'all';
  scenario?: string;
  dateRange?: {
    start: string; // ISO date
    end: string;   // ISO date
  };
  autonomyDecision?: AutonomyDecision | 'all';
  outcome?: PipelineOutcome | 'all';
  searchText?: string;
}

export interface PlaybackState {
  isPlaying: boolean;
  currentTime: number; // milliseconds from start
  totalDuration: number; // milliseconds
  playbackSpeed: 0.5 | 1 | 2 | 4;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Scenario Knowledge Base
// ═══════════════════════════════════════════════════════════════════════════════

export interface ScenarioCondition {
  metric?: string;
  operator?: string;
  threshold?: number;
  log_pattern?: string;
  service_type?: string;
}

export interface Scenario {
  scenario_id: string;
  display_name: string;
  domain: 'compute' | 'storage';
  description: string;
  autonomy_badge: AutonomyDecision;
  action: string;
  confidence_threshold: number; // 0-100
  
  // YAML condition definition
  conditions: ScenarioCondition[];
  
  // RCA and remediation
  rca_template: string;
  playbook_hint: string;
  
  // Statistics from history
  times_seen: number;
  last_seen?: string; // ISO timestamp
  average_resolution_time?: number; // seconds
  success_rate?: number; // 0-100
  
  // Historical runs that matched this scenario
  historical_runs?: string[]; // Array of session_ids
}
