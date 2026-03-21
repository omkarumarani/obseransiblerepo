"""
compute-agent/app/telemetry.py
──────────────────────────────
OpenTelemetry bootstrap for the Compute Agent service.

Follows the same pattern as the other services in this learning stack
so that bridge spans appear seamlessly alongside frontend-api and
backend-api spans in Tempo — sharing the same trace_id when a Prometheus
alert fires while a user request is in flight.

Key additions vs the other services:
  - HTTPXClientInstrumentor:  auto-injects W3C traceparent header into
    every outgoing httpx call to xyOps, so xyOps API calls are child spans.
  - incident_counter:        Prometheus metric counting created/resolved incidents.
  - alert_processing_histogram: latency of processing each incoming alert.
"""

import logging
import os
from typing import Optional

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
    OTLPMetricExporter as GRPCMetricExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter as GRPCSpanExporter,
)
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
    OTLPLogExporter as GRPCLogExporter,
)
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry._logs import set_logger_provider
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor

logger = logging.getLogger("compute-agent.telemetry")

# ── Module-level OTel instrument handles ─────────────────────────────────────
incident_counter: Optional[metrics.Counter] = None
alert_processing_histogram: Optional[metrics.Histogram] = None
webhook_counter: Optional[metrics.Counter] = None

# ── Prometheus-client compute agent action counters ───────────────────────────
# Served via GET /metrics for direct Prometheus scraping.
# Low-cardinality labels only — do NOT add pod/replica/trace_id labels.
from prometheus_client import Counter as PromCounter, Histogram as PromHistogram

compute_agent_actions_total = PromCounter(
    "compute_agent_actions_total",
    "Total actions taken by the compute agent",
    ["action_type"],
)
compute_agent_restarts_total = PromCounter(
    "compute_agent_restarts_total",
    "Total service restart actions initiated",
)
compute_agent_noisy_neighbour_reductions_total = PromCounter(
    "compute_agent_noisy_neighbour_reductions_total",
    "Total noisy-neighbour throttle actions taken",
)
compute_agent_escalations_total = PromCounter(
    "compute_agent_escalations_total",
    "Total escalations to human SRE (too risky for automation)",
)
compute_agent_autonomous_actions_total = PromCounter(
    "compute_agent_autonomous_actions_total",
    "Total actions taken without human approval",
)
compute_agent_approval_required_total = PromCounter(
    "compute_agent_approval_required_total",
    "Total remediations routed to approval gate",
)
compute_agent_ai_analysis_total = PromCounter(
    "compute_agent_ai_analysis_total",
    "Total AI analyses attempted by the compute agent",
    ["status"],  # success | failed | skipped
)
compute_agent_predictive_incidents_total = PromCounter(
    "compute_agent_predictive_incidents_total",
    "Total predictive incident tickets created by the compute agent (pre-alert, approval-gated)",
)
compute_agent_webhook_received_total = PromCounter(
    "compute_agent_webhook_received_total",
    "Total Alertmanager webhooks received by compute agent",
    ["group_status"],
)
compute_agent_alert_processing_seconds = PromHistogram(
    "compute_agent_alert_processing_seconds",
    "Time to process a single compute alert end-to-end",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)


class OtelContextFilter(logging.Filter):
    """Injects active OTel trace_id/span_id into every stdlib log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx.is_valid:
            record.otel_trace_id = format(ctx.trace_id, "032x")
            record.otel_span_id = format(ctx.span_id, "016x")
        else:
            record.otel_trace_id = "0" * 32
            record.otel_span_id = "0" * 16
        return True


def get_tracer() -> trace.Tracer:
    """Return a named Tracer for manual span creation in main.py."""
    return trace.get_tracer("compute-agent")


def setup_telemetry(fastapi_app=None, service_name: str = "compute-agent") -> None:
    """
    Bootstrap all OTel providers for the Compute Agent.

    Call once at module import time BEFORE creating routes.

    Parameters
    ----------
    fastapi_app:
        The FastAPI() instance.  Pass None to skip FastAPIInstrumentor
        (useful if called before the app is created).
    service_name:
        Overrideable service name (defaults to env OTEL_SERVICE_NAME).
    """
    global incident_counter, alert_processing_histogram, webhook_counter

    if os.getenv("DISABLE_OTEL_EXPORTERS", "false").lower() == "true":
        logger.info("OpenTelemetry exporters disabled for this process")
        incident_counter = None
        alert_processing_histogram = None
        webhook_counter = None
        return

    effective_name = os.getenv("OTEL_SERVICE_NAME", service_name)
    resource = Resource.create(
        {
            "service.name": effective_name,
            "service.version": "1.0.0",
            "service.namespace": "aiops",
        }
    )

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")

    # ── Tracer provider ───────────────────────────────────────────────────────
    span_exporter = GRPCSpanExporter() if endpoint else ConsoleSpanExporter()
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    trace.set_tracer_provider(tracer_provider)

    # ── Meter provider ────────────────────────────────────────────────────────
    metric_exporter = GRPCMetricExporter() if endpoint else ConsoleMetricExporter()
    metric_reader = PeriodicExportingMetricReader(
        metric_exporter,
        export_interval_millis=int(os.getenv("OTEL_METRIC_EXPORT_INTERVAL_MS", "15000")),
    )
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)
    meter = metrics.get_meter("compute-agent")

    # ── Logging enrichment ────────────────────────────────────────────────────
    # CRITICAL ORDER:
    # 1. Set formatter + add OtelContextFilter to EXISTING handlers FIRST.
    # 2. Add the OTel LoggingHandler AFTER, so it never gets our custom format.
    #    The OTel handler calls self.format(record) internally to build the log
    #    body; if it has %(otel_trace_id)s in its format it will crash because
    #    it runs format() outside the filter chain.
    # 3. Filters must be on HANDLERS (not root Logger) so they fire for records
    #    that propagate up from child loggers via callHandlers().
    fmt = (
        "%(asctime)s [%(levelname)s] %(name)s "
        "trace_id=%(otel_trace_id)s span_id=%(otel_span_id)s "
        "- %(message)s"
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    otel_ctx_filter = OtelContextFilter()
    for handler in root.handlers:
        handler.setFormatter(logging.Formatter(fmt))
        handler.addFilter(otel_ctx_filter)

    # ── Logger provider (ships logs to Loki via collector) ────────────────────
    if endpoint:
        log_provider = LoggerProvider(resource=resource)
        log_provider.add_log_record_processor(
            BatchLogRecordProcessor(GRPCLogExporter())
        )
        set_logger_provider(log_provider)
        # Do NOT give the OTel handler our custom format — it uses self.format()
        # for its internal log body translation, not for terminal output.
        otel_handler = LoggingHandler(logger_provider=log_provider)
        root.addHandler(otel_handler)

    # ── Auto-instrumentation ──────────────────────────────────────────────────
    # HTTPXClientInstrumentor injects W3C traceparent into all outgoing
    # httpx calls so that xyOps API calls appear as child spans in Tempo.
    HTTPXClientInstrumentor().instrument()

    if fastapi_app is not None:
        FastAPIInstrumentor.instrument_app(
            fastapi_app,
            excluded_urls="health",
        )

    # ── Custom metric instruments ─────────────────────────────────────────────
    incident_counter = meter.create_counter(
        name="aiops_incidents_total",
        description="Total incidents created or resolved in xyOps",
        unit="1",
    )
    alert_processing_histogram = meter.create_histogram(
        name="aiops_alert_processing_seconds",
        description="Time taken to process one Alertmanager alert and call xyOps",
        unit="s",
    )
    webhook_counter = meter.create_counter(
        name="aiops_webhooks_received_total",
        description="Total Alertmanager webhook payloads received",
        unit="1",
    )

    logger.info(
        "[AIOps Bridge] OTel initialised  service=%s  endpoint=%s",
        effective_name,
        endpoint or "console",
    )
