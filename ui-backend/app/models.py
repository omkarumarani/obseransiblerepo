"""
Data models for UI Backend API
"""
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime


# ─────────────────────────────────────────────────────────────
# Pipeline History Models (Section 1.2)
# ─────────────────────────────────────────────────────────────

class PipelineRunSummary(BaseModel):
    """Summary of a single pipeline execution"""
    session_id: str
    timestamp: str
    service_name: str
    alert_name: str
    scenario_matched: Optional[str] = None
    risk_score: float
    autonomy_decision: Literal["autonomous", "approval_gated", "human_only"]
    outcome: Literal["success", "failure", "pending", "running"]
    duration: Optional[float] = None  # seconds
    domain: Literal["compute", "storage"]


class AgentTransition(BaseModel):
    """Single agent state transition with timing and output"""
    agent_name: str
    status: Literal["idle", "running", "completed", "failed", "skipped"]
    timestamp: str
    duration: float  # seconds
    output: Optional[str] = None
    log_excerpts: List[str] = []
    metric_values: Dict[str, float] = {}


class DecisionStep(BaseModel):
    """One step in the autonomy decision tree"""
    step_number: int
    description: str
    condition: str
    result: bool
    color: Literal["pass", "fail"]


class ScenarioMatch(BaseModel):
    """Scenario matching result with confidence"""
    scenario_id: str
    confidence: float
    match_table: List[Dict[str, Any]] = []


class FinalOutcome(BaseModel):
    """Final outcome of pipeline execution"""
    gitea_pr_url: Optional[str] = None
    ansible_run_result: Optional[str] = None
    approval_status: Optional[str] = None


class PipelineRunFull(BaseModel):
    """Complete pipeline execution snapshot for playback"""
    session_id: str
    summary: PipelineRunSummary
    agent_transitions: List[AgentTransition]
    scenario_match: Optional[ScenarioMatch] = None
    rca_text: Optional[str] = None
    playbook_hint: Optional[str] = None
    decision_tree_steps: List[DecisionStep] = []
    final_outcome: FinalOutcome


class HistoryFilters(BaseModel):
    """Filters for pipeline history query"""
    domain: Optional[Literal["compute", "storage"]] = None
    scenario: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    autonomy_decision: Optional[str] = None
    outcome: Optional[str] = None


# ─────────────────────────────────────────────────────────────
# Scenario Knowledge Models (Section 1.3)
# ─────────────────────────────────────────────────────────────

class ScenarioCard(BaseModel):
    """Scenario card for knowledge map grid"""
    scenario_id: str
    display_name: str
    domain: Literal["compute", "storage"]
    autonomy_badge: Literal["autonomous", "approval_gated", "human_only"]
    action: str
    confidence_threshold: float
    times_seen: int = 0
    last_seen: Optional[str] = None
    avg_resolution_time: Optional[float] = None  # seconds
    success_rate: Optional[float] = None  # 0.0 to 1.0


class YAMLCondition(BaseModel):
    """Single YAML condition from scenario definition"""
    field: str
    operator: str
    value: Any
    description: str


class HistoricalRun(BaseModel):
    """Brief historical run reference for scenario detail"""
    session_id: str
    timestamp: str
    outcome: str


class ScenarioDetail(BaseModel):
    """Full scenario details with historical runs"""
    scenario_id: str
    display_name: str
    domain: str
    autonomy_badge: str
    yaml_conditions: List[YAMLCondition]
    rca_template: str
    playbook_hint: str
    historical_runs: List[HistoricalRun] = []
    statistics: ScenarioCard


# ─────────────────────────────────────────────────────────────
# Live Pipeline Models (Section 1.1 - existing)
# ─────────────────────────────────────────────────────────────

class AgentState(BaseModel):
    """Current state of a single agent"""
    name: str
    status: Literal["idle", "running", "completed", "failed", "skipped"]
    output: Optional[Dict[str, Any]] = None


class IncidentData(BaseModel):
    """Incident metadata for dashboard"""
    service_name: str
    alert_name: str
    severity: str
    risk_score: float
    scenario_matched: Optional[str] = None
    grafana_url: Optional[str] = None


class PipelineState(BaseModel):
    """Current state of a live pipeline session"""
    session_id: str
    status: Literal["running", "completed", "failed"]
    agents: List[AgentState]
    incident: Optional[IncidentData] = None
    autonomy_decision: Optional[str] = None
    approval_required: bool = False


# ─────────────────────────────────────────────────────────────
# Autonomy Status Models
# ─────────────────────────────────────────────────────────────

class ServiceTierStatus(BaseModel):
    """Autonomy tier status for a single service"""
    service_name: str
    tier: str
    risk_ceiling: float
    approvals_required: int
    approvals_current: int
    last_decision: Optional[str] = None


class AutonomyStatusResponse(BaseModel):
    """Overall autonomy status across all services"""
    services: List[ServiceTierStatus]


# ─────────────────────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str = "1.0.0"
    services: Dict[str, str] = {}
