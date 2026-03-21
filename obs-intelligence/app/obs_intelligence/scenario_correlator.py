"""
obs_intelligence/scenario_correlator.py
────────────────────────────────────────────────────────────────────────────────
Scenario correlator — matches an ObsFeatures snapshot against the loaded
scenario catalog and returns ranked ScenarioMatch results.

Algorithm
─────────
For each ScenarioDef:
  1. Alert-name pre-filter: if alert_name_patterns is non-empty and the
     incoming alert_name matches none of them → skip (hard filter).
  2. Scoring:
       alert_base   = alert_match_weight  (if patterns present AND matched)
       condition_score = sum of weights for each condition that holds
       total_weight = sum(all condition weights) + alert_match_weight (if any)
       confidence   = (alert_base + condition_score) / total_weight
  3. Threshold filter: include only scenarios with
     confidence >= confidence_threshold.

Results are sorted by confidence descending.

Usage
─────
    from obs_intelligence.scenario_correlator import load_catalog, match_scenarios, match_best

    # Once at agent startup — cache the result:
    catalog = load_catalog(domain="storage")

    # Per-alert, inside the pipeline:
    matches = match_scenarios(features, catalog)    # → list[ScenarioMatch]
    best_match, best_def = match_best(features, catalog)  # → best pair or (None, None)
"""

from __future__ import annotations

import fnmatch
import logging
import os
from typing import Sequence

from obs_intelligence.models import ObsFeatures, ScenarioMatch
from obs_intelligence.outcome_store import OutcomeStore
from obs_intelligence.scenario_loader import ConditionDef, ScenarioDef, load_scenarios

# Module-level singleton — shared across all requests in this process
_outcome_store = OutcomeStore()

logger = logging.getLogger("obs_intelligence.scenario_correlator")

_DEFAULT_SCENARIOS_DIR = os.getenv("SCENARIOS_DIR", "/app/scenarios")


# ═══════════════════════════════════════════════════════════════════════════════
# Public: catalog loading helper
# ═══════════════════════════════════════════════════════════════════════════════

def load_catalog(
    domain: str | None = None,
    scenarios_dir: str | None = None,
) -> list[ScenarioDef]:
    """
    Convenience wrapper around load_scenarios().

    Intended to be called *once* at agent startup and the result stored
    at module level (or in a class attribute) to avoid repeated disk I/O.

    Parameters
    ----------
    domain :
        Optional domain filter ("compute" | "storage").
    scenarios_dir :
        Override the default scenarios directory.  Defaults to the
        ``SCENARIOS_DIR`` env var or ``/app/scenarios``.
    """
    return load_scenarios(scenarios_dir or _DEFAULT_SCENARIOS_DIR, domain=domain)


# ═══════════════════════════════════════════════════════════════════════════════
# Public: matching
# ═══════════════════════════════════════════════════════════════════════════════

def match_scenarios(
    features: ObsFeatures,
    scenarios: Sequence[ScenarioDef],
) -> list[ScenarioMatch]:
    """
    Score every scenario against *features* and return matches above each
    scenario's confidence_threshold, sorted by confidence descending.

    Returns an empty list if *scenarios* is empty or nothing matches.
    """
    return [m for _, m, _ in _score_all(features, scenarios)]


def match_best(
    features: ObsFeatures,
    scenarios: Sequence[ScenarioDef],
) -> tuple[ScenarioMatch | None, ScenarioDef | None]:
    """
    Return the highest-confidence (ScenarioMatch, ScenarioDef) pair,
    or (None, None) if nothing in *scenarios* matches *features*.

    Prefer this over match_scenarios() when you need the full ScenarioDef
    (e.g. to access action, autonomy, rca, and playbook_hint).
    """
    scored = _score_all(features, scenarios)
    if scored:
        _, best_match, best_def = scored[0]
        return best_match, best_def
    return None, None


# ═══════════════════════════════════════════════════════════════════════════════
# Internal scoring engine
# ═══════════════════════════════════════════════════════════════════════════════

def _score_all(
    features: ObsFeatures,
    scenarios: Sequence[ScenarioDef],
) -> list[tuple[float, ScenarioMatch, ScenarioDef]]:
    """
    Return a list of (confidence, ScenarioMatch, ScenarioDef) tuples for
    every scenario that passes the confidence threshold, sorted descending.
    """
    alert_name = features.alert_name
    results: list[tuple[float, ScenarioMatch, ScenarioDef]] = []

    for s in scenarios:

        # ── 1. Alert-name pre-filter ──────────────────────────────────────────
        name_matches = True
        if s.alert_name_patterns:
            name_matches = any(
                fnmatch.fnmatch(alert_name, pat) for pat in s.alert_name_patterns
            )
            if not name_matches:
                continue   # none of the declared patterns match → skip

        # ── 2. Scoring ────────────────────────────────────────────────────────
        # alert_match_weight contributes only when patterns are declared AND matched.
        alert_weight = s.alert_match_weight if s.alert_name_patterns else 0.0
        total_weight = sum(c.weight for c in s.conditions) + alert_weight

        if total_weight <= 0:
            continue

        scored_weight = alert_weight if (s.alert_name_patterns and name_matches) else 0.0
        matched_features: list[str] = []

        for cond in s.conditions:
            value = getattr(features, cond.field, None)
            if value is None:
                continue   # field not present on ObsFeatures — skip silently
            if _eval_condition(cond, value):
                scored_weight += cond.weight
                matched_features.append(cond.field)

        confidence = scored_weight / total_weight
        # ── 2b. Outcome-feedback weight adjustment ────────────────────────
        adj = _outcome_store.get_weight_adjustment(s.scenario_id)
        confidence = max(0.0, min(1.0, confidence + adj))
        # ── 3. Threshold filter ───────────────────────────────────────────────
        if confidence < s.confidence_threshold:
            continue

        results.append((
            confidence,
            ScenarioMatch(
                scenario_id      = s.scenario_id,
                display_name     = s.display_name,
                confidence       = round(confidence, 4),
                domain           = s.domain,
                matched_features = matched_features,
                scenario_file    = s.scenario_file,
            ),
            s,
        ))

    results.sort(key=lambda t: t[0], reverse=True)

    logger.debug(
        "match_scenarios: alert=%-30s  catalog=%d  hits=%d",
        alert_name, len(scenarios), len(results),
    )
    return results


def _eval_condition(cond: ConditionDef, value: object) -> bool:
    """Evaluate one condition against the feature value.  Returns True if it holds."""
    op = cond.operator

    # ── Boolean operators ─────────────────────────────────────────────────────
    if op == "true":
        return bool(value)
    if op == "false":
        return not bool(value)

    # ── Numeric operators ─────────────────────────────────────────────────────
    try:
        v = float(value)     # type: ignore[arg-type]
        t = cond.threshold
        if op == "gt":  return v > t
        if op == "lt":  return v < t
        if op == "gte": return v >= t
        if op == "lte": return v <= t
        if op == "eq":  return v == t
        if op == "ne":  return v != t
    except (TypeError, ValueError):
        pass

    return False
