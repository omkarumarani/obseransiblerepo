"""
obs_intelligence/scenario_loader.py
────────────────────────────────────────────────────────────────────────────────
Scenario YAML loader and schema validator.

Each scenario YAML (one file per pattern) defines:
  - alert_name_patterns   — fnmatch patterns that pre-filter the scenario
  - alert_match_weight    — base confidence score when the alert name matches
  - conditions            — ObsFeatures field checks; each adds weighted score
  - action / autonomy     — what to do and who may authorise it
  - rca / playbook_hint   — text used by agents in ticket body + Ansible stub

Called at agent startup via load_scenarios().  Any schema error raises
ScenarioSchemaError so the service fails fast before handling real alerts.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("obs_intelligence.scenario_loader")

# Default scenarios directory — overrideable via SCENARIOS_DIR env var.
_DEFAULT_SCENARIOS_DIR = os.getenv("SCENARIOS_DIR", "/app/scenarios")

# ── Valid field value sets ────────────────────────────────────────────────────
_VALID_OPERATORS = frozenset({"gt", "lt", "gte", "lte", "eq", "ne", "true", "false"})
_VALID_DOMAINS   = frozenset({"compute", "storage"})
_VALID_AUTONOMY  = frozenset({"autonomous", "approval_gated", "human_only"})


# ═══════════════════════════════════════════════════════════════════════════════
# Public exception
# ═══════════════════════════════════════════════════════════════════════════════

class ScenarioSchemaError(ValueError):
    """Raised when a YAML file fails schema validation (fail-fast on startup)."""


# ═══════════════════════════════════════════════════════════════════════════════
# Data models
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConditionDef:
    """A single matching rule applied to an ObsFeatures field."""

    field: str
    operator: str     # gt | lt | gte | lte | eq | ne  (numeric)
                      # true | false                    (boolean field)
    threshold: float = 0.0   # ignored for 'true'/'false' operators
    weight: float    = 0.1   # contribution to confidence score when matched


@dataclass
class ScenarioDef:
    """
    Fully-parsed scenario definition loaded from a YAML file.

    Instances are produced by load_scenarios() and consumed by the
    scenario correlator to score ObsFeatures snapshots.
    """

    scenario_id:           str
    display_name:          str
    domain:                str                    # "compute" | "storage"
    conditions:            list[ConditionDef]

    # ── Remediation metadata ──────────────────────────────────────────────────
    action:                str  = "escalate"      # e.g. "osd_reweight", "cpu_scale_out"
    autonomy:              str  = "approval_gated"  # autonomous | approval_gated | human_only
    irreversible:          bool = False
    rca:                   str  = ""
    playbook_hint:         str  = ""

    # ── Matching parameters ───────────────────────────────────────────────────
    alert_name_patterns:   list = field(default_factory=list)
    # fnmatch patterns — if non-empty, the alert name must match at least one
    # pattern or the scenario is skipped entirely.

    alert_match_weight:    float = 0.0
    # Base confidence score added when alert_name_patterns matches.
    # Useful for alert-name-driven scenarios where metric data may not yet
    # be populated (e.g. the very first alert before Prometheus has data).

    confidence_threshold:  float = 0.3
    # Minimum confidence required to include this scenario in match results.

    # ── Metadata ─────────────────────────────────────────────────────────────
    version:               str  = "1.0"
    scenario_file:         str  = ""   # set by load_scenarios() after parsing


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def load_scenarios(
    scenarios_dir: str | Path | None = None,
    domain: str | None = None,
) -> list[ScenarioDef]:
    """
    Load and validate all scenario YAML files in *scenarios_dir*.

    Parameters
    ----------
    scenarios_dir :
        Root directory containing ``compute/`` and/or ``storage/``
        subdirectories.  Defaults to the ``SCENARIOS_DIR`` env var or
        ``/app/scenarios``.
    domain :
        If supplied, only load YAMLs from that subdirectory
        (e.g. ``domain="compute"`` → reads from ``scenarios_dir/compute/``).

    Returns
    -------
    list[ScenarioDef]
        All valid scenarios sorted ascending by scenario_id.

    Raises
    ------
    ScenarioSchemaError
        Any schema violation causes an immediate raise — agents are expected
        to call load_scenarios() at startup and abort if this fires.
    """
    root = Path(scenarios_dir or _DEFAULT_SCENARIOS_DIR)

    if not root.exists():
        logger.warning(
            "Scenarios directory '%s' not found — running without scenario catalog", root
        )
        return []

    if domain:
        subdirs = [root / domain]
    else:
        subdirs = sorted(p for p in root.iterdir() if p.is_dir())

    scenarios: list[ScenarioDef] = []
    for subdir in subdirs:
        if not subdir.is_dir():
            continue
        for yaml_path in sorted(subdir.glob("*.yaml")):
            try:
                s = _load_one(yaml_path)
                scenarios.append(s)
                logger.debug("Loaded scenario: %-40s  (%s)", s.scenario_id, yaml_path.name)
            except ScenarioSchemaError:
                raise   # re-raise immediately — fail fast on startup
            except Exception as exc:
                raise ScenarioSchemaError(
                    f"Failed to parse '{yaml_path}': {exc}"
                ) from exc

    logger.info(
        "Scenario catalog loaded: %d scenario(s) from '%s'%s",
        len(scenarios),
        root,
        f"  [domain={domain}]" if domain else "",
    )
    return sorted(scenarios, key=lambda s: s.scenario_id)


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _load_one(yaml_path: Path) -> ScenarioDef:
    """Parse and validate a single YAML file into a ScenarioDef."""
    with yaml_path.open(encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    # ── Required top-level fields ─────────────────────────────────────────────
    for key in ("scenario_id", "display_name", "domain"):
        if key not in raw:
            raise ScenarioSchemaError(
                f"'{yaml_path.name}': missing required field '{key}'"
            )

    raw_conditions = raw.get("conditions")
    if not raw_conditions:
        raise ScenarioSchemaError(
            f"'{yaml_path.name}': 'conditions' must be a non-empty list"
        )

    # ── Parse conditions ──────────────────────────────────────────────────────
    conditions: list[ConditionDef] = []
    for i, c in enumerate(raw_conditions):
        if not isinstance(c, dict):
            raise ScenarioSchemaError(
                f"'{yaml_path.name}': condition[{i}] must be a mapping, got {type(c).__name__}"
            )
        for req in ("field", "operator"):
            if req not in c:
                raise ScenarioSchemaError(
                    f"'{yaml_path.name}': condition[{i}] missing required key '{req}'"
                )
        op = str(c["operator"])
        if op not in _VALID_OPERATORS:
            raise ScenarioSchemaError(
                f"'{yaml_path.name}': condition[{i}] operator '{op}' is not valid. "
                f"Choose from: {sorted(_VALID_OPERATORS)}"
            )
        weight = float(c.get("weight", 0.1))
        if weight <= 0:
            raise ScenarioSchemaError(
                f"'{yaml_path.name}': condition[{i}] weight must be > 0, got {weight}"
            )
        conditions.append(ConditionDef(
            field=str(c["field"]),
            operator=op,
            threshold=float(c.get("threshold", 0.0)),
            weight=weight,
        ))

    # ── Validate domain / autonomy ────────────────────────────────────────────
    domain = str(raw["domain"])
    if domain not in _VALID_DOMAINS:
        raise ScenarioSchemaError(
            f"'{yaml_path.name}': domain '{domain}' must be one of {sorted(_VALID_DOMAINS)}"
        )

    autonomy = str(raw.get("autonomy", "approval_gated"))
    if autonomy not in _VALID_AUTONOMY:
        raise ScenarioSchemaError(
            f"'{yaml_path.name}': autonomy '{autonomy}' must be one of {sorted(_VALID_AUTONOMY)}"
        )

    ct = float(raw.get("confidence_threshold", 0.3))
    if not (0.0 < ct <= 1.0):
        raise ScenarioSchemaError(
            f"'{yaml_path.name}': confidence_threshold must be in (0, 1], got {ct}"
        )

    amw = float(raw.get("alert_match_weight", 0.0))
    if amw < 0 or amw > 1:
        raise ScenarioSchemaError(
            f"'{yaml_path.name}': alert_match_weight must be in [0, 1], got {amw}"
        )

    return ScenarioDef(
        scenario_id          = str(raw["scenario_id"]),
        display_name         = str(raw["display_name"]),
        domain               = domain,
        conditions           = conditions,
        action               = str(raw.get("action", "escalate")),
        autonomy             = autonomy,
        irreversible         = bool(raw.get("irreversible", False)),
        rca                  = str(raw.get("rca", "")),
        playbook_hint        = str(raw.get("playbook_hint", "")),
        alert_name_patterns  = [str(p) for p in raw.get("alert_name_patterns", [])],
        alert_match_weight   = amw,
        confidence_threshold = ct,
        version              = str(raw.get("version", "1.0")),
        scenario_file        = str(yaml_path),
    )
