"""
storage-agent/app/telemetry.py
───────────────────────────────────────────────────────────────
OTel bootstrap + Prometheus-client agent action counters.

Two metric systems coexist:
  1. OTel OTLP metrics  — pushed to otel-collector → Prometheus scrapes collector
  2. prometheus-client  — serves /metrics endpoint that Prometheus scrapes directly

The prometheus-client counters track storage agent actions with low cardinality
so they can feed Grafana panels and compare with compute agent actions.
"""

import logging
import os
from typing import Optional

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter as GRPCMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as GRPCSpanExporter
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter as GRPCLogExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry._logs import set_logger_provider
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor

# prometheus-client counters — served via GET /metrics
from prometheus_client import Counter, Histogram

logger = logging.getLogger("storage-agent.telemetry")

# ── OTel module-level handles ──────────────────────────────────────────────────
webhook_counter: Optional[metrics.Counter] = None
alert_processing_histogram: Optional[metrics.Histogram] = None

# ── Prometheus-client storage agent action counters ───────────────────────────
# Low-cardinality labels only: action_type, status, severity, alert_name
# These feed directly into the Grafana "Agentic AI Operations" dashboard.

storage_agent_actions_total = Counter(
    "storage_agent_actions_total",
    "Total actions taken by the storage agent (label: action_type)",
    ["action_type"],
)
storage_agent_osd_reweights_total = Counter(
    "storage_agent_osd_reweights_total",
    "Total Ceph OSD reweight actions initiated",
)
storage_agent_noisy_pvc_throttles_total = Counter(
    "storage_agent_noisy_pvc_throttles_total",
    "Total noisy-PVC throttle actions initiated",
)
storage_agent_escalations_total = Counter(
    "storage_agent_escalations_total",
    "Total escalations to human SRE (scenario too risky for automation)",
)
storage_agent_autonomous_remediations_total = Counter(
    "storage_agent_autonomous_remediations_total",
    "Total remediations executed without human approval",
)
storage_agent_approval_required_total = Counter(
    "storage_agent_approval_required_total",
    "Total remediations that required human approval before execution",
)
storage_agent_ai_analysis_total = Counter(
    "storage_agent_ai_analysis_total",
    "Total AI analyses attempted",
    ["status"],  # success | failed | skipped
)
storage_agent_webhook_received_total = Counter(
    "storage_agent_webhook_received_total",
    "Total Alertmanager webhooks received",
    ["group_status"],  # firing | resolved
)
storage_agent_predictive_incidents_total = Counter(
    "storage_agent_predictive_incidents_total",
    "Total predictive incident tickets created by the storage agent (pre-alert, approval-gated)",
)
storage_agent_alert_processing_seconds = Histogram(
    "storage_agent_alert_processing_seconds",
    "Time to process a single storage alert end-to-end",
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
    return trace.get_tracer("storage-agent")


def setup_telemetry(fastapi_app=None, service_name: str = "storage-agent") -> None:
    """Bootstrap OTel providers for the Storage Agent."""
    global webhook_counter, alert_processing_histogram

    effective_name = os.getenv("OTEL_SERVICE_NAME", service_name)
    resource = Resource.create({
        "service.name": effective_name,
        "service.version": "1.0.0",
        "service.namespace": "aiops",
    })

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")

    # ── Trace provider ──────────────────────────────────────────────────────────
    tp = TracerProvider(resource=resource)
    if endpoint:
        tp.add_span_processor(BatchSpanProcessor(GRPCSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(tp)

    # ── Metrics provider ────────────────────────────────────────────────────────
    readers = []
    if endpoint:
        readers.append(PeriodicExportingMetricReader(GRPCMetricExporter(endpoint=endpoint)))
    mp = MeterProvider(resource=resource, metric_readers=readers)
    metrics.set_meter_provider(mp)
    meter = mp.get_meter("storage-agent")

    webhook_counter = meter.create_counter(
        "storage_agent_webhooks_total",
        description="Alertmanager webhooks received by storage agent",
    )
    alert_processing_histogram = meter.create_histogram(
        "storage_agent_processing_duration_ms",
        description="Alert processing duration in milliseconds",
        unit="ms",
    )

    # ── Log provider ────────────────────────────────────────────────────────────
    if endpoint:
        lp = LoggerProvider(resource=resource)
        lp.add_log_record_processor(BatchLogRecordProcessor(GRPCLogExporter(endpoint=endpoint)))
        set_logger_provider(lp)
        otel_handler = LoggingHandler(logger_provider=lp)
        logging.getLogger().addHandler(otel_handler)

    # ── Auto-instrumentation ────────────────────────────────────────────────────
    HTTPXClientInstrumentor().instrument()
    if fastapi_app is not None:
        FastAPIInstrumentor.instrument_app(fastapi_app)

    # ── Inject trace_id into stdlib log records ─────────────────────────────────
    ctx_filter = OtelContextFilter()
    for handler in logging.getLogger().handlers:
        handler.addFilter(ctx_filter)

    logger.info("OTel telemetry initialised  service=%s  endpoint=%s", effective_name, endpoint or "console")
