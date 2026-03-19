"""
obs-intelligence/app/obs_intelligence/telemetry_client.py
────────────────────────────────────────────────────────────────────────────
Shared Prometheus + Loki fetchers used by both compute-agent and storage-agent.

Extracted from:
  - compute-agent/app/ai_analyst.py  (fetch_loki_logs, fetch_prometheus_context)
  - storage-agent/app/storage_analyst.py (fetch_loki_logs, fetch_storage_metrics)

Public API
──────────
  fetch_loki_context(label_query, http, limit, loki_url)
      → str
      Fetches the last `limit` log lines matching a LogQL label selector.
      Returns newline-joined lines in chronological order, empty str on failure.

  fetch_instant_metric(promql, http, prometheus_url)
      → list[dict]
      Executes a Prometheus instant-vector query (/api/v1/query).
      Returns the result list (each item: {metric: {...}, value: [ts, "val"]}),
      or an empty list on failure.

  fetch_metric_range(promql, http, prometheus_url, start, end, step)
      → list[[float, str]]
      Executes a Prometheus range query (/api/v1/query_range).
      Returns the [timestamp, value] pairs for the first series, or [] on failure.
      Intended for forecasting and anomaly detection workloads (Phase 4+).
"""

from __future__ import annotations

import logging
import os
import time as _time
from typing import Any

import httpx

logger = logging.getLogger("obs_intelligence.telemetry_client")

_DEFAULT_LOKI_URL = os.getenv("LOKI_URL", "http://loki:3100")
_DEFAULT_PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")


# ─────────────────────────────────────────────────────────────────────────────
# Loki
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_loki_context(
    label_query: str,
    http: httpx.AsyncClient,
    limit: int = 50,
    loki_url: str | None = None,
) -> str:
    """
    Query Loki for the last `limit` log lines matching `label_query`.

    Args:
        label_query: LogQL label selector, e.g. '{service_name="frontend-api"}'
        http:        Shared async HTTP client from the calling pipeline.
        limit:       Maximum number of log lines to return.
        loki_url:    Override; defaults to the LOKI_URL environment variable.

    Returns:
        Newline-joined log lines in chronological order (oldest first),
        or an empty string on any failure so callers can always use the result
        in string contexts without additional None checks.
    """
    url = loki_url or _DEFAULT_LOKI_URL
    try:
        resp = await http.get(
            f"{url}/loki/api/v1/query_range",
            params={
                "query": label_query,
                "limit": str(limit),
                "direction": "backward",
            },
            timeout=8.0,
        )
        if resp.status_code != 200:
            logger.warning(
                "Loki returned HTTP %d for query %r", resp.status_code, label_query
            )
            return ""
        data = resp.json()
        lines: list[str] = []
        for stream in data.get("data", {}).get("result", []):
            for _ts, msg in stream.get("values", []):
                lines.append(msg)
        lines.reverse()  # backward → forward chronological order
        result = "\n".join(lines[-limit:])
        logger.debug("Loki: returned %d lines for %r", len(lines), label_query)
        return result
    except Exception as exc:
        logger.warning("Loki fetch failed (query=%r): %s", label_query, exc)
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Prometheus
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_instant_metric(
    promql: str,
    http: httpx.AsyncClient,
    prometheus_url: str | None = None,
    timeout: float = 5.0,
) -> list[dict[str, Any]]:
    """
    Execute a Prometheus instant-vector query (GET /api/v1/query).

    Args:
        promql:          PromQL expression evaluated at current time.
        http:            Shared async HTTP client from the calling pipeline.
        prometheus_url:  Override; defaults to PROMETHEUS_URL env var.
        timeout:         HTTP request timeout in seconds.

    Returns:
        The Prometheus ``result`` list — a list of dicts, each with shape:
            {"metric": {"__name__": ..., <extra labels>},
             "value":  [unix_timestamp_float, "numeric_string"]}
        Returns an empty list on failure so callers can safely check ``if series``.
    """
    url = prometheus_url or _DEFAULT_PROMETHEUS_URL
    try:
        resp = await http.get(
            f"{url}/api/v1/query",
            params={"query": promql},
            timeout=timeout,
        )
        if resp.status_code != 200:
            logger.warning(
                "Prometheus returned HTTP %d for expr %r",
                resp.status_code,
                promql[:120],
            )
            return []
        data = resp.json()
        results: list[dict[str, Any]] = data.get("data", {}).get("result", [])
        logger.debug(
            "Prometheus instant query: %d series for %r", len(results), promql[:60]
        )
        return results
    except Exception as exc:
        logger.warning(
            "Prometheus instant query failed (expr=%r): %s", promql[:120], exc
        )
        return []


async def fetch_metric_range(
    promql: str,
    http: httpx.AsyncClient,
    prometheus_url: str | None = None,
    start: str | None = None,
    end: str | None = None,
    step: str = "30s",
    timeout: float = 10.0,
) -> list[list]:
    """
    Execute a Prometheus range query (GET /api/v1/query_range).

    Intended for anomaly detection (rolling window stats) and forecasting
    workloads introduced in Phase 4+.  Returns data for the *first* series
    only; for multi-series results call fetch_instant_metric and process
    labels in the caller.

    Args:
        promql:          PromQL expression.
        http:            Shared async HTTP client.
        prometheus_url:  Override; defaults to PROMETHEUS_URL env var.
        start:           RFC3339 or Unix timestamp string (default: 30 min ago).
        end:             RFC3339 or Unix timestamp string (default: now).
        step:            Resolution step, e.g. "30s", "1m", "5m".
        timeout:         HTTP request timeout in seconds.

    Returns:
        List of [unix_timestamp_float, "value_string"] pairs for the first
        series, or an empty list on failure / no data.
    """
    url = prometheus_url or _DEFAULT_PROMETHEUS_URL
    now = int(_time.time())
    params: dict[str, str] = {
        "query": promql,
        "step":  step,
        "start": start or str(now - 1800),  # 30-minute default lookback
        "end":   end   or str(now),
    }
    try:
        resp = await http.get(
            f"{url}/api/v1/query_range",
            params=params,
            timeout=timeout,
        )
        if resp.status_code != 200:
            logger.warning(
                "Prometheus range query HTTP %d for %r",
                resp.status_code,
                promql[:60],
            )
            return []
        data = resp.json()
        results = data.get("data", {}).get("result", [])
        if not results:
            return []
        return results[0].get("values", [])
    except Exception as exc:
        logger.warning("Prometheus range query failed (%r): %s", promql[:60], exc)
        return []
