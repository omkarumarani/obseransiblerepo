"""
obs-intelligence/app/telemetry.py
──────────────────────────────────
OpenTelemetry bootstrap for the Obs-Intelligence Engine service.

Follows the same pattern as compute-agent/app/telemetry.py so that
obs-intelligence spans appear alongside agent spans in Grafana Tempo,
sharing the same trace context when agents call this service.
"""

import logging
import os

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
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry._logs import set_logger_provider
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor

logger = logging.getLogger("obs-intelligence.telemetry")

_OTEL_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4317")
_SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "obs-intelligence")


def bootstrap(app=None) -> None:
    """Bootstrap OTel traces, metrics, logs and auto-instrument FastAPI + httpx."""
    resource = Resource.create({
        "service.name": _SERVICE_NAME,
        "deployment.environment": os.getenv("DEPLOYMENT_ENVIRONMENT", "local"),
    })

    # ── Traces ────────────────────────────────────────────────────────────────
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(
        BatchSpanProcessor(GRPCSpanExporter(endpoint=_OTEL_ENDPOINT, insecure=True))
    )
    trace.set_tracer_provider(tracer_provider)

    # ── Metrics ───────────────────────────────────────────────────────────────
    reader = PeriodicExportingMetricReader(
        GRPCMetricExporter(endpoint=_OTEL_ENDPOINT, insecure=True),
        export_interval_millis=int(os.getenv("OTEL_METRIC_EXPORT_INTERVAL_MS", "15000")),
    )
    meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(meter_provider)

    # ── Logs ──────────────────────────────────────────────────────────────────
    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(
        BatchLogRecordProcessor(GRPCLogExporter(endpoint=_OTEL_ENDPOINT, insecure=True))
    )
    set_logger_provider(logger_provider)
    logging.getLogger().addHandler(
        LoggingHandler(level=logging.NOTSET, logger_provider=logger_provider)
    )

    # ── Auto-instrumentation ──────────────────────────────────────────────────
    HTTPXClientInstrumentor().instrument()
    if app is not None:
        FastAPIInstrumentor.instrument_app(app)

    logger.info(
        "OTel bootstrap complete: endpoint=%s service=%s", _OTEL_ENDPOINT, _SERVICE_NAME
    )
