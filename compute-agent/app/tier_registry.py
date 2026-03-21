"""
compute-agent/app/tier_registry.py
────────────────────────────────────────────────────────────────────────────────
Service tier registry — maps service names to environment tiers and defines
the autonomy policy for each tier.

Tiers (highest trust requirement → lowest):
  PRODUCTION   (Tier 1) — live customer-facing services
  STAGING      (Tier 2) — pre-production / QA environments
  DEVELOPMENT  (Tier 3) — developer integration environments
  SANDBOX      (Tier 4) — playground / demo environments

Tier Policy fields:
  min_approvals_for_autonomy  — how many historical human approvals +
                                autonomous successes are needed before
                                a (service, action) pair is eligible
                                for fully autonomous execution.
  min_success_rate            — fraction of past executions that must
                                have succeeded (0.0–1.0).
  risk_ceiling                — max risk_score allowed for autonomous
                                execution (above this → always gate human).
  auto_merge_pr               — when True the agent may merge the Gitea PR
                                automatically on approval; when False the
                                human must merge first.
  history_window_days         — how far back to look when computing trust.

Service → Tier mapping
──────────────────────
Override via environment variable SERVICE_TIER_MAP_JSON (JSON dict):
  {"frontend-api": "production", "loadgen": "sandbox"}

Or drop a file at SERVICE_TIER_MAP_PATH (default /data/service_tiers.json):
  {"frontend-api": "production", "backend-api": "production", ...}

Unknown services default to PRODUCTION (safest assumption).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("aiops-bridge.tier_registry")


# ═══════════════════════════════════════════════════════════════════════════════
# Tier enum + Policy
# ═══════════════════════════════════════════════════════════════════════════════

class ServiceTier(str, Enum):
    PRODUCTION  = "production"
    STAGING     = "staging"
    DEVELOPMENT = "development"
    SANDBOX     = "sandbox"


@dataclass(frozen=True)
class TierPolicy:
    tier: ServiceTier
    min_approvals_for_autonomy: int    # ≥ this many approved runs before unlocking
    min_success_rate: float            # 0.0–1.0 required exec success fraction
    risk_ceiling: float                # autonomous only below this risk score
    auto_merge_pr: bool                # agent may auto-merge Gitea PR
    history_window_days: int           # days of history to consider
    description: str


# Default policies per tier
_TIER_POLICIES: dict[ServiceTier, TierPolicy] = {
    ServiceTier.PRODUCTION: TierPolicy(
        tier=ServiceTier.PRODUCTION,
        min_approvals_for_autonomy=10,
        min_success_rate=0.95,
        risk_ceiling=0.50,
        auto_merge_pr=False,     # humans must merge PRs in production
        history_window_days=90,
        description=(
            "Production: requires 10+ approved executions with ≥95% success rate "
            "and risk < 0.50 before autonomous mode unlocks. "
            "Human must always merge the Gitea PR."
        ),
    ),
    ServiceTier.STAGING: TierPolicy(
        tier=ServiceTier.STAGING,
        min_approvals_for_autonomy=5,
        min_success_rate=0.90,
        risk_ceiling=0.65,
        auto_merge_pr=True,
        history_window_days=60,
        description=(
            "Staging: requires 5+ approved executions with ≥90% success rate "
            "and risk < 0.65 before autonomous mode unlocks. "
            "Agent may auto-merge Gitea PRs."
        ),
    ),
    ServiceTier.DEVELOPMENT: TierPolicy(
        tier=ServiceTier.DEVELOPMENT,
        min_approvals_for_autonomy=2,
        min_success_rate=0.80,
        risk_ceiling=0.80,
        auto_merge_pr=True,
        history_window_days=30,
        description=(
            "Development: requires 2+ approved executions with ≥80% success rate "
            "and risk < 0.80 before autonomous mode unlocks."
        ),
    ),
    ServiceTier.SANDBOX: TierPolicy(
        tier=ServiceTier.SANDBOX,
        min_approvals_for_autonomy=1,
        min_success_rate=0.70,
        risk_ceiling=1.0,
        auto_merge_pr=True,
        history_window_days=14,
        description=(
            "Sandbox: unlocks autonomous mode after just 1 successful approval. "
            "All risk levels permitted. Suitable for demos and learning environments."
        ),
    ),
}

# ── Built-in defaults ──────────────────────────────────────────────────────────
# Services known from the docker-compose in this workspace.
# Override via SERVICE_TIER_MAP_JSON or /data/service_tiers.json.
_BUILTIN_SERVICE_TIERS: dict[str, ServiceTier] = {
    "frontend-api":       ServiceTier.PRODUCTION,
    "backend-api":        ServiceTier.PRODUCTION,
    "storage-simulator":  ServiceTier.STAGING,
    "compute-agent":      ServiceTier.STAGING,
    "obs-intelligence":   ServiceTier.STAGING,
    "ansible-runner":     ServiceTier.DEVELOPMENT,
    "loadgen":            ServiceTier.SANDBOX,
    "troublemaker":       ServiceTier.SANDBOX,
}

# Runtime-loaded overrides (populated once on first access)
_loaded_overrides: dict[str, ServiceTier] | None = None


def _load_overrides() -> dict[str, ServiceTier]:
    """
    Load service-to-tier overrides from:
      1. JSON env var SERVICE_TIER_MAP_JSON
      2. JSON file at SERVICE_TIER_MAP_PATH (/data/service_tiers.json)

    Returns a dict of service_name → ServiceTier.
    """
    global _loaded_overrides
    if _loaded_overrides is not None:
        return _loaded_overrides

    overrides: dict[str, ServiceTier] = {}

    # 1. From environment variable
    env_json = os.getenv("SERVICE_TIER_MAP_JSON", "")
    if env_json:
        try:
            raw = json.loads(env_json)
            for svc, tier_str in raw.items():
                try:
                    overrides[svc] = ServiceTier(tier_str.lower())
                except ValueError:
                    logger.warning("Unknown tier '%s' for service '%s' in SERVICE_TIER_MAP_JSON", tier_str, svc)
        except json.JSONDecodeError as exc:
            logger.warning("Could not parse SERVICE_TIER_MAP_JSON: %s", exc)

    # 2. From file (file overrides env if both present for the same service)
    map_path = os.getenv("SERVICE_TIER_MAP_PATH", "/data/service_tiers.json")
    try:
        with open(map_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for svc, tier_str in raw.items():
            try:
                overrides[svc] = ServiceTier(tier_str.lower())
            except ValueError:
                logger.warning("Unknown tier '%s' for service '%s' in %s", tier_str, svc, map_path)
        logger.info("Loaded %d service tier overrides from %s", len(raw), map_path)
    except FileNotFoundError:
        pass  # no file — use env-only or built-ins
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not load service tier map from %s: %s", map_path, exc)

    _loaded_overrides = overrides
    return _loaded_overrides


def get_service_tier(service_name: str) -> ServiceTier:
    """
    Return the ServiceTier for a given service name.

    Resolution order:
      1. Loaded overrides (env var or /data/service_tiers.json)
      2. Built-in defaults
      3. PRODUCTION (safest fallback for unknown services)
    """
    # Normalise: strip suffixes like "-svc", "-service"
    key = service_name.lower().strip()

    overrides = _load_overrides()
    if key in overrides:
        return overrides[key]

    if key in _BUILTIN_SERVICE_TIERS:
        return _BUILTIN_SERVICE_TIERS[key]

    logger.debug(
        "Service '%s' not in tier registry — defaulting to PRODUCTION (safest)", service_name
    )
    return ServiceTier.PRODUCTION


def get_tier_policy(tier: ServiceTier) -> TierPolicy:
    """Return the TierPolicy for the given ServiceTier."""
    return _TIER_POLICIES[tier]


def reload_overrides() -> dict[str, ServiceTier]:
    """Force a reload of tier overrides (useful after config file update)."""
    global _loaded_overrides
    _loaded_overrides = None
    return _load_overrides()


def list_all_tiers() -> dict[str, dict]:
    """Return a full view of service → tier mapping (for status endpoints)."""
    overrides = _load_overrides()
    combined: dict[str, ServiceTier] = {**_BUILTIN_SERVICE_TIERS, **overrides}
    return {
        svc: {
            "tier": tier.value,
            "policy": {
                "min_approvals_for_autonomy": _TIER_POLICIES[tier].min_approvals_for_autonomy,
                "min_success_rate": _TIER_POLICIES[tier].min_success_rate,
                "risk_ceiling": _TIER_POLICIES[tier].risk_ceiling,
                "auto_merge_pr": _TIER_POLICIES[tier].auto_merge_pr,
                "history_window_days": _TIER_POLICIES[tier].history_window_days,
            },
        }
        for svc, tier in sorted(combined.items())
    }
