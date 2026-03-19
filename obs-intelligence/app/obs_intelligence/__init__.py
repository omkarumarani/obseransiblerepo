"""
obs_intelligence — Shared intelligence core for the multi-agent AIOps platform.

Importable as a Python package by compute-agent and storage-agent.
Also runs as a standalone FastAPI service on port 9100.

Usage (from another agent after package is installed):
    from obs_intelligence.models import ObsFeatures, EvidenceReport
"""

from obs_intelligence.models import (
    AnomalySignal,
    EvidenceReport,
    ForecastResult,
    ObsFeatures,
    Recommendation,
    RiskAssessment,
    ScenarioMatch,
)

__version__ = "1.0.0"

__all__ = [
    "AnomalySignal",
    "EvidenceReport",
    "ForecastResult",
    "ObsFeatures",
    "Recommendation",
    "RiskAssessment",
    "ScenarioMatch",
]
